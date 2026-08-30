"""Run Luke rescue candidates on complementary time and depth pilot panels.

The time panel preserves all 384 channels in short, prespecified epochs.  The
depth panel preserves the complete 10,473.6 s recording in a 96-channel strip.
Together they expose local waveform failures and cumulative sorting failures
without committing to a full-session, full-depth sort.

The time panel is stratified by prior curated neural-event presence: good
(``good_pre_shared``, 73%), neutral (``neutral_template``, 49%), and
pathological (43%, existing sort).  A sparse 3,951 s window is retained only as
a negative control because it cannot initialize Kilosort's waveform PCA.  The
default new work is the good and neutral controls plus the full-duration core
depth strip::

    python testing/luke_two_axis_pilot.py --plan-only
    python testing/luke_two_axis_pilot.py --prepare --pilot good_pre_shared
    python testing/luke_two_axis_pilot.py --run --score --pilot good_pre_shared
    python testing/luke_two_axis_pilot.py --prepare --pilot core_depth_strip
    python testing/luke_two_axis_pilot.py --run --score --pilot core_depth_strip

``--run`` requires the patched Kilosort environment and CUDA.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from testing.luke_claimmask_window_sweep import (
    ClaimSetting,
    cross_unit_near_coincident_fraction,
    load_reference_settings,
    local_match_mask,
)
from testing.luke_upstream_sorter_ablation import parse_extraction_counts
from testing.luke_upstream_stage_ablation import (
    DEFAULT_REVIEW,
    PIPELINE_ROOT,
    RAW_ROOT,
    STREAM_ID,
)


# The interrupted full-session cache is deliberately not used: SpikeInterface
# preallocated its full byte length before all chunks were populated.  Exact
# size therefore looked valid while later session regions were zero-filled.
INCOMPLETE_SOURCE_RECORDING = (
    PIPELINE_ROOT / "full_session_rescue/single_ks_preprocessing_claim_off/recording"
)
SOURCE_PROVENANCE = INCOMPLETE_SOURCE_RECORDING / "provenance.json"
SOURCE_PROBE = INCOMPLETE_SOURCE_RECORDING / "probe.json"
RAW_BINARY = (
    RAW_ROOT
    / "Luke0730_V2V1_g0_imec1/Luke0730_V2V1_g0_t0.imec1.ap.bin"
)
SOURCE_FRAMES = 314_204_094
N_SAVED_CHANNELS = 385
OUTPUT_ROOT = Path(
    "/media/huklab/Data/NPX/Ryansorting/Luke/Luke0804_two_axis_pilot_imec1"
)
CONDITIONING_V2_OUTPUT_ROOT = Path(
    "/media/huklab/Data/NPX/Ryansorting/Luke/Luke0804_conditioning_v2_pilot_imec1"
)
CLAIM_OFF = ClaimSetting("claim_off", 0.0, 0.0)
N_CHANNELS = 384
DTYPE = np.dtype("int16")


@dataclass(frozen=True)
class Pilot:
    name: str
    axis: str
    role: str
    start_s: float | None = None
    duration_s: float | None = None
    first_channel: int | None = None
    n_channels: int | None = None
    run_by_default: bool = True
    existing_sort: str | None = None


PILOTS = {
    pilot.name: pilot
    for pilot in (
        Pilot(
            "quiet_negative_control",
            "time",
            "quiet negative control; may be too sparse for sorter initialization",
            start_s=3951.0,
            duration_s=120.0,
            run_by_default=False,
        ),
        Pilot(
            "good_pre_shared",
            "time",
            "good full-depth epoch; 73% curated neural-event presence",
            start_s=7095.0,
            duration_s=120.0,
        ),
        Pilot(
            "neutral_template",
            "time",
            "neutral full-depth template epoch; 49% curated neural-event presence",
            start_s=7215.0,
            duration_s=120.0,
        ),
        Pilot(
            "shared_combined_reference",
            "time",
            "neutral/shared full-depth window",
            start_s=7095.0,
            duration_s=240.0,
            run_by_default=False,
            existing_sort=str(
                PIPELINE_ROOT
                / "motion_candidate_replication/shared_template/sorts/single_ks_preprocessing"
            ),
        ),
        Pilot(
            "pathological",
            "time",
            "registration-outlier full-depth window",
            start_s=8160.0,
            duration_s=120.0,
            run_by_default=False,
            existing_sort=str(
                PIPELINE_ROOT
                / "upstream_sorter_ablation/sorts/single_ks_preprocessing"
            ),
        ),
        Pilot(
            "core_depth_strip",
            "depth",
            "full-duration core strip containing channels 191 and 216",
            first_channel=176,
            n_channels=96,
        ),
        Pilot(
            "upper_depth_strip",
            "depth",
            "optional full-duration upper strip covering the 3180 um event mode",
            first_channel=272,
            n_channels=96,
            run_by_default=False,
        ),
    )
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--score", action="store_true")
    parser.add_argument("--benchmark-io", action="store_true")
    parser.add_argument("--benchmark-duration-s", type=float, default=120.0)
    parser.add_argument("--pilot", action="append", choices=tuple(PILOTS))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--conditioning-policy",
        choices=("legacy", "conditioning_v2"),
        default="legacy",
        help="conditioning_v2 uses phase correction only and excludes channel 191 in Kilosort",
    )
    parser.add_argument("--review-events", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--time-bin-s", type=float, default=300.0)
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = (
            CONDITIONING_V2_OUTPUT_ROOT
            if args.conditioning_policy == "conditioning_v2"
            else OUTPUT_ROOT
        )
    return args


def selected_pilots(names: list[str] | None) -> list[Pilot]:
    if names:
        return [PILOTS[name] for name in dict.fromkeys(names)]
    return [pilot for pilot in PILOTS.values() if pilot.run_by_default]


def source_shape() -> tuple[int, int]:
    return SOURCE_FRAMES, N_CHANNELS


def pilot_channel_ids(pilot: Pilot) -> np.ndarray:
    if pilot.axis == "time":
        return np.arange(N_CHANNELS, dtype=int)
    if pilot.first_channel is None or pilot.n_channels is None:
        raise ValueError(f"Depth pilot lacks a channel selection: {pilot}")
    stop = pilot.first_channel + pilot.n_channels
    if pilot.first_channel < 0 or stop > N_CHANNELS:
        raise ValueError(f"Invalid channel range [{pilot.first_channel}, {stop})")
    return np.arange(pilot.first_channel, stop, dtype=int)


def pilot_frame_range(pilot: Pilot, fs: float, n_frames: int) -> tuple[int, int]:
    if pilot.axis == "depth":
        return 0, n_frames
    if pilot.start_s is None or pilot.duration_s is None:
        raise ValueError(f"Time pilot lacks a frame selection: {pilot}")
    start = int(round(pilot.start_s * fs))
    stop = start + int(round(pilot.duration_s * fs))
    if start < 0 or stop > n_frames:
        raise ValueError(f"Pilot {pilot.name} exceeds the source recording")
    return start, stop


def bad_channel_rows(
    recording_channel_ids: np.ndarray, physical_bad_channels: tuple[int, ...] = (191,)
) -> list[int]:
    """Map physical channel ids to Kilosort binary-row indices after slicing."""
    channel_ids = np.asarray(recording_channel_ids, dtype=int)
    return np.flatnonzero(np.isin(channel_ids, physical_bad_channels)).astype(int).tolist()


def recording_path(output_dir: Path, pilot: Pilot) -> Path:
    return output_dir / "recordings" / pilot.name


def sort_path(
    output_dir: Path, pilot: Pilot, conditioning_policy: str = "legacy"
) -> Path:
    if pilot.existing_sort is not None and conditioning_policy == "legacy":
        return Path(pilot.existing_sort)
    return output_dir / "sorts" / pilot.name / "single_ks_preprocessing_claim_off"


def build_plan(
    pilots: list[Pilot],
    output_dir: Path,
    review_path: Path,
    conditioning_policy: str = "legacy",
) -> dict:
    _, fs = load_reference_settings()
    n_frames, _ = source_shape()
    events = pd.read_csv(review_path)
    rows = []
    for pilot in pilots:
        start, stop = pilot_frame_range(pilot, fs, n_frames)
        channels = pilot_channel_ids(pilot)
        locations = events["peak_depth_um"].to_numpy(float)
        if pilot.axis == "time":
            selected_events = (events["sample_index"] >= start) & (
                events["sample_index"] < stop
            )
        else:
            # NP1 channel ids in this recording advance by approximately 10 um.
            lo_um = float(channels[0] * 10)
            hi_um = float(channels[-1] * 10)
            selected_events = (locations >= lo_um) & (locations <= hi_um)
        rows.append(
            {
                **asdict(pilot),
                "start_frame": start,
                "stop_frame": stop,
                "selected_channel_first": int(channels[0]),
                "selected_channel_last": int(channels[-1]),
                "selected_channel_count": len(channels),
                "duration_s_effective": (stop - start) / fs,
                "estimated_int16_gib": (stop - start)
                * len(channels)
                * DTYPE.itemsize
                / 1024**3,
                "n_reviewed_events_in_scope": int(selected_events.sum()),
                "n_neural_unmatched_in_scope": int(
                    (
                        selected_events
                        & events["review_label"].eq("neural")
                        & events["status"].eq("unmatched")
                    ).sum()
                ),
                "sort_path": str(sort_path(output_dir, pilot, conditioning_policy)),
                "new_sort_required": (
                    conditioning_policy == "conditioning_v2"
                    or pilot.existing_sort is None
                ),
            }
        )
    return {
        "condition": "single_ks_preprocessing_claim_off",
        "source_raw_root": str(RAW_ROOT),
        "source_stream_id": STREAM_ID,
        "conditioning_policy": conditioning_policy,
        "source_stage": (
            "phase_shift_only_bad_channel_191_excluded_in_sorter"
            if conditioning_policy == "conditioning_v2"
            else "phase_shift_saturation_blank_bad_channel_interpolation"
        ),
        "saturation_policy": (
            "voltage_unchanged; preserve a separate artifact mask for downstream exclusion"
            if conditioning_policy == "conditioning_v2"
            else "samplewise prefilter blanking"
        ),
        "bad_channel_191_policy": (
            "pass local binary row to Kilosort bad_channels; do not synthesize voltage"
            if conditioning_policy == "conditioning_v2"
            else "spatial interpolation"
        ),
        "rejected_incomplete_cache": str(INCOMPLETE_SOURCE_RECORDING),
        "source_frames": n_frames,
        "source_channels": N_CHANNELS,
        "sampling_frequency_hz": fs,
        "pilots": rows,
        "advancement_rule": (
            "Advance a preprocessing or motion candidate only if it preserves reviewed "
            "neural recovery in the time panel and does not create detection expansion, "
            "near-coincident spikes, transient units, or edge accumulation in the depth panel."
        ),
    }


def load_source_recording(
    channel_ids: np.ndarray | None = None, conditioning_policy: str = "legacy"
):
    from probeinterface import read_probeinterface
    from spikeinterface.core import BinaryRecordingExtractor
    from spikeinterface.extractors.neuropixels_utils import (
        get_neuropixels_sample_shifts,
    )
    from spikeinterface.preprocessing import (
        blank_staturation,
        interpolate_bad_channels,
        phase_shift,
    )

    graph = json.loads(SOURCE_PROVENANCE.read_text())
    interpolation_args = dict(graph["kwargs"])
    blank_graph = interpolation_args.pop("recording")
    interpolation_args["bad_channel_ids"] = [
        int(str(channel_id).rsplit("AP", 1)[-1])
        for channel_id in interpolation_args["bad_channel_ids"]
    ]
    blank_args = dict(blank_graph["kwargs"])
    phase_graph = blank_args.pop("recording")
    phase_args = dict(phase_graph["kwargs"])
    raw_graph = phase_args.pop("recording")
    _, fs = load_reference_settings()
    raw = BinaryRecordingExtractor(
        file_paths=[RAW_BINARY],
        sampling_frequency=fs,
        dtype=DTYPE,
        num_channels=N_SAVED_CHANNELS,
        channel_ids=np.arange(N_SAVED_CHANNELS),
        is_filtered=False,
    ).channel_slice(channel_ids=np.arange(N_CHANNELS))
    raw = raw.set_probegroup(read_probeinterface(SOURCE_PROBE))
    for name, values in raw_graph.get("properties", {}).items():
        if name not in {"location", "group"}:
            raw.set_property(name, np.asarray(values))
    # NP1 AP acquisition: 12 channels per ADC and 13 acquisition cycles.
    # SpikeGLXRecordingExtractor normally adds this property, but the direct
    # binary reconstruction must restore it explicitly.
    raw.set_property(
        "inter_sample_shift",
        get_neuropixels_sample_shifts(N_CHANNELS, 12, 13),
    )
    if channel_ids is not None:
        channel_ids = np.asarray(channel_ids, dtype=int)
        raw = raw.channel_slice(channel_ids=channel_ids)
        # Saved weights index all 383 good channels. Recompute after slicing;
        # the core strip contains every nonzero neighbor of bad channel 191.
        interpolation_args.pop("weights", None)
    shifted = phase_shift(raw, **phase_args)
    if conditioning_policy == "conditioning_v2":
        recording = shifted
        if recording.get_num_samples() != SOURCE_FRAMES:
            raise RuntimeError(
                f"Raw-stage length changed: {recording.get_num_samples()} != {SOURCE_FRAMES}"
            )
        return recording
    if conditioning_policy not in {"legacy", "legacy_no_bad_interpolation"}:
        raise ValueError(f"Unknown conditioning policy: {conditioning_policy}")
    blanked = blank_staturation(shifted, **blank_args)
    if conditioning_policy == "legacy_no_bad_interpolation":
        selected_ids = set(int(channel_id) for channel_id in blanked.get_channel_ids())
        selected_bad = selected_ids.intersection(interpolation_args["bad_channel_ids"])
        if selected_bad:
            raise ValueError(
                "legacy_no_bad_interpolation is valid only when known bad channels "
                f"are absent; found {sorted(selected_bad)}"
            )
        return blanked
    recording = interpolate_bad_channels(blanked, **interpolation_args)
    if recording.get_num_samples() != SOURCE_FRAMES:
        raise RuntimeError(
            f"Raw-stage length changed: {recording.get_num_samples()} != {SOURCE_FRAMES}"
        )
    return recording


def prepare_pilots(
    pilots: list[Pilot], output_dir: Path, conditioning_policy: str = "legacy"
) -> None:
    _, fs = load_reference_settings()
    n_frames, _ = source_shape()
    for pilot in pilots:
        if pilot.existing_sort is not None and conditioning_policy == "legacy":
            print(f"Reusing existing {pilot.name} sort: {pilot.existing_sort}")
            continue
        target = recording_path(output_dir, pilot)
        manifest = target / "pilot_manifest.json"
        start, stop = pilot_frame_range(pilot, fs, n_frames)
        channels = pilot_channel_ids(pilot)
        if target.exists():
            if not manifest.exists():
                raise RuntimeError(f"Ambiguous pilot recording: {target}")
            print(f"Reusing {target}")
            continue
        source = load_source_recording(
            channels if pilot.axis == "depth" else None, conditioning_policy
        )
        selected = source.frame_slice(start_frame=start, end_frame=stop)
        if pilot.axis == "time":
            selected = selected.channel_slice(channel_ids=channels)
        target.parent.mkdir(parents=True, exist_ok=True)
        # Measured network-to-network benchmark on this recording (120 s):
        # 1 worker 11.9 MiB/s, 4 workers 17.5 MiB/s, 8 workers 29.0 MiB/s.
        # Ten-second chunks with 8 workers use about 440 MiB for this strip.
        selected.save(folder=target, n_jobs=8, chunk_duration="10s", progress_bar=True)
        manifest.write_text(
            json.dumps(
                {
                    **asdict(pilot),
                    "condition": "single_ks_preprocessing_claim_off",
                    "source_raw_root": str(RAW_ROOT),
                    "source_raw_binary": str(RAW_BINARY),
                    "source_stream_id": STREAM_ID,
                    "conditioning_policy": conditioning_policy,
                    "source_stage": (
                        "phase_shift_only_bad_channel_191_excluded_in_sorter"
                        if conditioning_policy == "conditioning_v2"
                        else "phase_shift_saturation_blank_bad_channel_interpolation"
                    ),
                    "saturation_policy": (
                        "voltage_unchanged; downstream artifact exclusion required"
                        if conditioning_policy == "conditioning_v2"
                        else "samplewise prefilter blanking"
                    ),
                    "start_frame": start,
                    "stop_frame": stop,
                    "channel_ids": channels.tolist(),
                    "sampling_frequency_hz": fs,
                    "expected_binary_bytes": (stop - start)
                    * len(channels)
                    * DTYPE.itemsize,
                },
                indent=2,
            )
            + "\n"
        )


def benchmark_io(output_dir: Path, duration_s: float) -> pd.DataFrame:
    """Benchmark identical network-to-network strip writes before a long copy."""
    if duration_s <= 0:
        raise ValueError("--benchmark-duration-s must be positive")
    channels = pilot_channel_ids(PILOTS["core_depth_strip"])
    source = load_source_recording(channels)
    fs = float(source.get_sampling_frequency())
    stop = min(source.get_num_samples(), int(round(duration_s * fs)))
    selected = source.frame_slice(start_frame=0, end_frame=stop)
    rows = []
    for n_jobs in (1, 4, 8):
        target = output_dir / "io_benchmark" / f"jobs_{n_jobs}"
        if target.exists():
            raise RuntimeError(f"Benchmark output already exists: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        selected.save(
            folder=target,
            n_jobs=n_jobs,
            chunk_duration="10s",
            progress_bar=True,
        )
        elapsed = time.perf_counter() - started
        n_bytes = (stop * len(channels) * DTYPE.itemsize)
        rows.append(
            {
                "n_jobs": n_jobs,
                "duration_s": stop / fs,
                "bytes": n_bytes,
                "elapsed_s": elapsed,
                "throughput_mib_s": n_bytes / elapsed / 1024**2,
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "io_benchmark/results.csv", index=False)
    return frame


def assert_gpu_and_patch() -> None:
    import torch
    from kilosort.parameters import DEFAULT_SETTINGS

    missing = {"cross_peel_claim_ms", "cross_peel_claim_um"} - set(DEFAULT_SETTINGS)
    if missing:
        raise RuntimeError(f"Patched Kilosort settings are missing: {sorted(missing)}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run this phase outside the sandbox")


def run_pilots(
    pilots: list[Pilot], output_dir: Path, conditioning_policy: str = "legacy"
) -> None:
    import spikeinterface.core as sc
    from spikeinterface.sorters import run_sorter

    from testing.luke_claimmask_window_sweep import build_sorter_params

    assert_gpu_and_patch()
    for pilot in pilots:
        result = (
            sort_path(output_dir, pilot, conditioning_policy)
            / "sorter_output/spike_times.npy"
        )
        if result.exists():
            print(f"Reusing completed {pilot.name} sort: {result.parent.parent}")
            continue
        if pilot.existing_sort is not None and conditioning_policy == "legacy":
            raise FileNotFoundError(result)
        target = sort_path(output_dir, pilot, conditioning_policy)
        if target.exists():
            raise RuntimeError(f"Ambiguous pilot sort: {target}")
        recording_dir = recording_path(output_dir, pilot)
        if not (recording_dir / "pilot_manifest.json").exists():
            raise FileNotFoundError(f"Prepare {pilot.name} first: {recording_dir}")
        recording = sc.load(recording_dir)
        bad_rows: list[int] | None = None
        if conditioning_policy == "conditioning_v2":
            bad_rows = bad_channel_rows(recording.get_channel_ids())
        params = build_sorter_params(CLAIM_OFF, bad_channels=bad_rows)
        target.parent.mkdir(parents=True, exist_ok=True)
        log_path = output_dir / "logs" / f"{pilot.name}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(log_path)
        root_logger = logging.getLogger()
        root_logger.addHandler(handler)
        try:
            with log_path.open("a") as log_file, contextlib.redirect_stdout(
                log_file
            ), contextlib.redirect_stderr(log_file):
                run_sorter(
                    "kilosort4",
                    recording,
                    folder=str(target),
                    verbose=True,
                    remove_existing_folder=False,
                    **params,
                )
        finally:
            root_logger.removeHandler(handler)
            handler.close()
        print(f"Completed {pilot.name}; log: {log_path}")


def temporal_unit_metrics(
    times: np.ndarray,
    clusters: np.ndarray,
    fs: float,
    duration_s: float,
    time_bin_s: float,
) -> pd.DataFrame:
    n_bins = max(1, int(np.ceil(duration_s / time_bin_s)))
    rows = []
    for unit in np.unique(clusters):
        unit_times = np.sort(times[clusters == unit])
        seconds = unit_times / fs
        occupied = np.unique(np.minimum((seconds / time_bin_s).astype(int), n_bins - 1))
        isi = np.diff(unit_times)
        rows.append(
            {
                "unit_id": int(unit),
                "n_spikes": len(unit_times),
                "first_spike_s": float(seconds[0]),
                "last_spike_s": float(seconds[-1]),
                "lifetime_s": float(seconds[-1] - seconds[0]),
                "active_time_bin_fraction": float(len(occupied) / n_bins),
                "refractory_violation_fraction": float(
                    np.mean(isi < int(round(1.5e-3 * fs)))
                )
                if len(isi)
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def temporal_bin_metrics(
    times: np.ndarray, fs: float, duration_s: float, time_bin_s: float
) -> pd.DataFrame:
    n_bins = max(1, int(np.ceil(duration_s / time_bin_s)))
    edges = np.linspace(0.0, duration_s, n_bins + 1)
    counts, _ = np.histogram(times / fs, bins=edges)
    widths = np.diff(edges)
    return pd.DataFrame(
        {
            "bin_start_s": edges[:-1],
            "bin_stop_s": edges[1:],
            "spike_count": counts,
            "spikes_per_s": counts / widths,
        }
    )


def _read_labels(result: Path) -> tuple[pd.DataFrame, str]:
    labels = pd.read_csv(result / "cluster_KSLabel.tsv", sep="\t")
    return labels, next(column for column in labels if column != "cluster_id")


def parse_last_sort_runtime_s(log_text: str) -> float | None:
    """Return the last completed Kilosort runtime from an append-only log."""
    matches = re.findall(r"kilosort4 run time\s+([0-9.]+)s", log_text)
    return float(matches[-1]) if matches else None


def circular_shift_coincidence_null(
    times: np.ndarray,
    clusters: np.ndarray,
    depths: np.ndarray,
    duration_frames: int,
    tolerance_frames: int,
    seed: int,
    n_repeats: int = 3,
) -> float:
    """Rate/unit-preserving null for cross-unit temporal coincidence."""
    rng = np.random.default_rng(seed)
    units = np.unique(clusters)
    values = []
    for _ in range(n_repeats):
        shifted = times.copy()
        for unit in units:
            mask = clusters == unit
            offset = rng.integers(tolerance_frames + 1, duration_frames)
            shifted[mask] = (shifted[mask] + offset) % duration_frames
        values.append(
            cross_unit_near_coincident_fraction(
                shifted, clusters, depths, tolerance_frames
            )
        )
    return float(np.median(values))


def score_pilot(
    pilot: Pilot,
    output_dir: Path,
    review_path: Path,
    time_bin_s: float,
    conditioning_policy: str = "legacy",
    result_override: Path | None = None,
    score_name: str | None = None,
    log_override: Path | None = None,
) -> dict:
    result = (
        Path(result_override)
        if result_override is not None
        else sort_path(output_dir, pilot, conditioning_policy) / "sorter_output"
    )
    if not (result / "spike_times.npy").exists():
        raise FileNotFoundError(result)
    _, fs = load_reference_settings()
    n_frames, _ = source_shape()
    start, stop = pilot_frame_range(pilot, fs, n_frames)
    duration_s = (stop - start) / fs
    effective_time_bin_s = min(time_bin_s, max(1.0, duration_s / 12.0))
    times = np.load(result / "spike_times.npy").reshape(-1).astype(np.int64)
    clusters = np.load(result / "spike_clusters.npy").reshape(-1)
    depths = np.load(result / "spike_positions.npy")[:, 1].astype(float)
    valid = (times >= 0) & (times < stop - start)
    times, clusters, depths = times[valid], clusters[valid], depths[valid]
    labels, label_column = _read_labels(result)
    contamination = pd.read_csv(result / "cluster_ContamPct.tsv", sep="\t")
    contam_column = next(column for column in contamination if column != "cluster_id")
    unit_metrics = temporal_unit_metrics(
        times, clusters, fs, duration_s, effective_time_bin_s
    )
    bins = temporal_bin_metrics(times, fs, duration_s, effective_time_bin_s)
    summary_name = score_name or pilot.name
    target = output_dir / "scores" / summary_name
    target.mkdir(parents=True, exist_ok=True)
    unit_metrics.to_csv(target / "unit_temporal_metrics.csv", index=False)
    bins.to_csv(target / "time_bin_metrics.csv", index=False)

    events = pd.read_csv(review_path)
    if pilot.axis == "time":
        in_scope = (events["sample_index"] >= start) & (events["sample_index"] < stop)
        event_samples = events.loc[in_scope, "sample_index"].to_numpy(np.int64) - start
    else:
        channels = pilot_channel_ids(pilot)
        lo_um, hi_um = channels[0] * 10.0, channels[-1] * 10.0
        in_scope = events["peak_depth_um"].between(lo_um, hi_um)
        event_samples = events.loc[in_scope, "sample_index"].to_numpy(np.int64)
    scoped = events.loc[in_scope].copy()
    tolerance = int(round(0.5e-3 * fs))
    scoped["recovered"] = local_match_mask(
        event_samples,
        scoped["peak_depth_um"].to_numpy(float),
        times,
        depths,
        tolerance,
        100.0,
    )
    scoped.to_csv(target / "reviewed_event_recovery.csv", index=False)
    neural_unmatched = (scoped["review_label"] == "neural") & (
        scoped["status"] == "unmatched"
    )
    observed_recovery = (
        float(scoped.loc[neural_unmatched, "recovered"].mean())
        if neural_unmatched.any()
        else np.nan
    )
    seed = sum(pilot.name.encode("utf-8"))
    rng = np.random.default_rng(seed)
    jitter_recoveries = []
    if neural_unmatched.any():
        target_samples = event_samples[neural_unmatched.to_numpy()]
        target_depths = scoped.loc[neural_unmatched, "peak_depth_um"].to_numpy(float)
        for _ in range(100):
            offsets = rng.uniform(0.020 * fs, 0.500 * fs, len(target_samples))
            offsets *= rng.choice((-1.0, 1.0), len(target_samples))
            jittered = np.mod(
                target_samples + np.rint(offsets).astype(np.int64), stop - start
            )
            jitter_recoveries.append(
                float(
                    local_match_mask(
                        jittered, target_depths, times, depths, tolerance, 100.0
                    ).mean()
                )
            )
    jitter_recovery = float(np.mean(jitter_recoveries)) if jitter_recoveries else np.nan

    full_count = len(np.load(result / "full_st.npy", mmap_mode="r"))
    log_path = log_override or output_dir / "logs" / f"{pilot.name}.log"
    universal_count = learned_log_count = sort_runtime_s = None
    if log_path.exists():
        log_text = log_path.read_text()
        universal_count, learned_log_count = parse_extraction_counts(log_text)
        sort_runtime_s = parse_last_sort_runtime_s(log_text)
    if learned_log_count is not None and learned_log_count != full_count:
        raise RuntimeError(
            f"Learned count mismatch: log={learned_log_count}, full_st={full_count}"
        )
    channel_depths = np.asarray(
        np.load(result / "channel_positions.npy"), dtype=float
    )[:, 1]
    edge_distance = np.minimum(depths - channel_depths.min(), channel_depths.max() - depths)
    rate_mean = float(bins["spikes_per_s"].mean())
    observed_coincidence = cross_unit_near_coincident_fraction(
        times, clusters, depths, tolerance
    )
    coincidence_null = circular_shift_coincidence_null(
        times,
        clusters,
        depths,
        stop - start,
        tolerance,
        seed,
    )
    summary = {
        "pilot": summary_name,
        "axis": pilot.axis,
        "role": pilot.role,
        "duration_s": duration_s,
        "time_bin_s": effective_time_bin_s,
        "sort_runtime_s": sort_runtime_s,
        "n_channels": len(pilot_channel_ids(pilot)),
        "universal_detection_count": universal_count,
        "learned_detection_count": full_count,
        "n_final_spikes": len(times),
        "n_units": int(len(np.unique(clusters))),
        "n_ks_good": int(labels[label_column].astype(str).str.lower().eq("good").sum()),
        "median_contamination_pct": float(
            np.median(contamination[contam_column].to_numpy(float))
        ),
        "cross_unit_near_coincident_fraction": observed_coincidence,
        "cross_unit_coincidence_shift_null": coincidence_null,
        "cross_unit_coincidence_excess": observed_coincidence - coincidence_null,
        "median_unit_refractory_violation_fraction": float(
            unit_metrics["refractory_violation_fraction"].median()
        ),
        "transient_unit_fraction_active_lt_5pct_bins": float(
            (unit_metrics["active_time_bin_fraction"] < 0.05).mean()
        ),
        "spike_rate_cv_across_time_bins": float(
            bins["spikes_per_s"].std(ddof=0) / rate_mean
        )
        if rate_mean
        else np.nan,
        "edge_spike_fraction_within_40um": float(np.mean(edge_distance <= 40.0)),
        "n_reviewed_events": len(scoped),
        "n_neural_unmatched_events": int(neural_unmatched.sum()),
        "neural_unmatched_recovery": observed_recovery,
        "neural_unmatched_jitter_recovery": jitter_recovery,
        "neural_unmatched_recovery_excess": observed_recovery - jitter_recovery,
    }
    (target / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    args = parse_args()
    os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/luke-two-axis-pilot-numba")
    pilots = selected_pilots(args.pilot)
    plan = build_plan(
        pilots, args.output_dir, args.review_events, args.conditioning_policy
    )
    if args.plan_only:
        print(json.dumps(plan, indent=2))
        return
    if not (args.prepare or args.run or args.score or args.benchmark_io):
        raise SystemExit(
            "Choose --plan-only, --prepare, --run, --score, or --benchmark-io"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "pilot_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    if args.benchmark_io:
        print(benchmark_io(args.output_dir, args.benchmark_duration_s).to_string(index=False))
    if args.prepare:
        prepare_pilots(pilots, args.output_dir, args.conditioning_policy)
    if args.run:
        run_pilots(pilots, args.output_dir, args.conditioning_policy)
    if args.score:
        summaries = [
            score_pilot(
                pilot,
                args.output_dir,
                args.review_events,
                args.time_bin_s,
                args.conditioning_policy,
            )
            for pilot in pilots
        ]
        frame = pd.DataFrame(summaries)
        score_path = args.output_dir / "pilot_scores.csv"
        if score_path.exists():
            prior = pd.read_csv(score_path)
            frame = pd.concat(
                [prior.loc[~prior["pilot"].isin(frame["pilot"])], frame],
                ignore_index=True,
            )
        frame = frame.sort_values("pilot").reset_index(drop=True)
        frame.to_csv(score_path, index=False)
        print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
