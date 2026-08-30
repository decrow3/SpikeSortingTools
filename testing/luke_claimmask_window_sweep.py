"""Run a short-window Kilosort claim-mask sweep for Luke 2025-08-04 imec1.

The original motion-corrected binary has been deleted, so this script rebuilds
the same lazy preprocessing graph from the surviving raw SpikeGLX stream,
cached channel metrics, and saved full-session DREDGE motion.  It then saves two
diagnostic windows once and reuses them for every claim-mask setting.

Examples
--------
Inspect inputs and the exact jobs without writing anything::

    python testing/luke_claimmask_window_sweep.py --plan-only

Prepare the two cached windows, run every sort, and score reviewed events::

    python testing/luke_claimmask_window_sweep.py --prepare --run --score

This requires the patched Kilosort environment and an NVIDIA GPU for ``--run``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


LUKE_ROOT = Path("/mnt/NPX/Luke/20250804")
RAW_ROOT = LUKE_ROOT / "Luke0804_V2V1_g0"
PIPELINE_ROOT = LUKE_ROOT / "dredge_pipeline_results_Luke0804_V2V1_g0_imec1"
REFERENCE_OPS = (
    LUKE_ROOT
    / "patched_pipeline_results_Luke0804_V2V1_g0_imec1"
    / "kilosort4/sorter_output/ops.npy"
)
DEFAULT_REVIEW = Path(
    "testing/outputs/luke_multichannel_event_validation/imec1/event_stage_trace.csv"
)
DEFAULT_OUTPUT = PIPELINE_ROOT / "claimmask_window_sweep"
STREAM_ID = "imec1.ap"


@dataclass(frozen=True)
class Window:
    name: str
    start_s: float
    duration_s: float


@dataclass(frozen=True)
class ClaimSetting:
    name: str
    claim_ms: float
    claim_um: float


# The first window joins two adjacent review epochs and contains all 12 events
# passing the conservative automatic gate.  The second probes the period where
# the patched full-session sort missed 21/27 visually neural unmatched events.
WINDOWS = (
    Window("shared_template", 7095.0, 240.0),
    Window("registration_outlier", 8160.0, 120.0),
)

CLAIM_GRID = (
    ClaimSetting("claim_off", 0.0, 0.0),
    ClaimSetting("claim_ms0p1_um25", 0.10, 25.0),
    ClaimSetting("claim_ms0p1_um50", 0.10, 50.0),
    ClaimSetting("claim_ms0p25_um25", 0.25, 25.0),
    ClaimSetting("claim_ms0p25_um50", 0.25, 50.0),
    ClaimSetting("claim_ms0p25_um75", 0.25, 75.0),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--review-events", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--score", action="store_true")
    parser.add_argument("--n-jitters", type=int, default=250)
    parser.add_argument("--seed", type=int, default=20250804)
    return parser.parse_args()


def window_frames(window: Window, sampling_frequency: float) -> tuple[int, int]:
    start = int(round(window.start_s * sampling_frequency))
    stop = start + int(round(window.duration_s * sampling_frequency))
    return start, stop


def event_local_samples(
    sample_indices: np.ndarray, window: Window, sampling_frequency: float
) -> np.ndarray:
    start, _ = window_frames(window, sampling_frequency)
    return np.asarray(sample_indices, dtype=np.int64) - start


def events_in_window(events: pd.DataFrame, window: Window) -> pd.DataFrame:
    stop_s = window.start_s + window.duration_s
    return events[
        (events["time_seconds"] >= window.start_s)
        & (events["time_seconds"] < stop_s)
    ].copy()


def required_paths(review_path: Path) -> tuple[Path, ...]:
    return (
        RAW_ROOT,
        PIPELINE_ROOT / "conditioning/channel_metrics.npy",
        PIPELINE_ROOT / "motion/dredge-motion/motion.npy",
        PIPELINE_ROOT / "motion/dredge-motion/time_bins.npy",
        PIPELINE_ROOT / "motion/dredge-motion/depth_bins.npy",
        REFERENCE_OPS,
        review_path,
    )


def validate_inputs(review_path: Path) -> None:
    missing = [str(path) for path in required_paths(review_path) if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n  " + "\n  ".join(missing))


def load_reference_settings() -> tuple[dict, float]:
    ops = np.load(REFERENCE_OPS, allow_pickle=True).item()
    return dict(ops["settings"]), float(ops["fs"])


def build_sorter_params(
    setting: ClaimSetting, bad_channels: list[int] | None = None
) -> dict:
    # Import here so --plan-only remains useful on machines lacking the sorting
    # environment.  NUMBA_CACHE_DIR also avoids trying to cache in site-packages.
    os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/luke-claimmask-numba-cache")
    from spikeinterface.sorters import get_default_sorter_params

    defaults = get_default_sorter_params("kilosort4")
    reference, _ = load_reference_settings()
    params = dict(defaults)
    for key in defaults:
        if key in reference:
            params[key] = reference[key]
    params.update(
        do_correction=False,
        save_extra_vars=True,
        cross_peel_claim_ms=setting.claim_ms,
        cross_peel_claim_um=setting.claim_um,
    )
    if bad_channels is not None:
        params["bad_channels"] = [int(channel) for channel in bad_channels]
    return params


def plan(review_path: Path, output_dir: Path) -> dict:
    validate_inputs(review_path)
    settings, fs = load_reference_settings()
    events = pd.read_csv(review_path)
    windows = []
    for window in WINDOWS:
        selected = events_in_window(events, window)
        start, stop = window_frames(window, fs)
        windows.append(
            {
                **asdict(window),
                "start_sample": start,
                "stop_sample": stop,
                "n_reviewed_events": len(selected),
                "n_visual_neural_unmatched": int(
                    ((selected["review_label"] == "neural") & (selected["status"] == "unmatched")).sum()
                ),
                "n_automatic_neural_like_unmatched": int(
                    (selected["automatic_neural_like"] & (selected["status"] == "unmatched")).sum()
                ),
                "estimated_int16_gib": (stop - start) * 384 * 2 / 1024**3,
            }
        )
    return {
        "raw_root": str(RAW_ROOT),
        "stream_id": STREAM_ID,
        "pipeline_root": str(PIPELINE_ROOT),
        "reference_ops": str(REFERENCE_OPS),
        "output_dir": str(output_dir),
        "sampling_frequency_hz": fs,
        "reference_claim_ms": settings.get("cross_peel_claim_ms"),
        "reference_claim_um": settings.get("cross_peel_claim_um"),
        "windows": windows,
        "claim_grid": [asdict(item) for item in CLAIM_GRID],
        "n_sort_jobs": len(WINDOWS) * len(CLAIM_GRID),
        "interpretation_guardrail": (
            "Choose a setting only if recovery improves without a large increase "
            "in cross-unit near-coincident spikes, contamination, or refractory violations."
        ),
    }


def make_motion_corrected_recording():
    os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/luke-claimmask-numba-cache")
    import spikeinterface.full as si
    from spikeinterface.core.motion import Motion
    from spikeinterface.preprocessing import astype
    from spikeinterface.sortingcomponents.motion import interpolate_motion

    # Importing the project package may load optional motion dependencies, so do
    # it only for actual preparation rather than for planning/scoring.
    from pipelineold.preprocess import condition_signal

    raw = si.read_spikeglx(
        folder_path=RAW_ROOT, load_sync_channel=False, stream_id=STREAM_ID
    )
    _, for_sorting = condition_signal(
        raw,
        cache_dir=PIPELINE_ROOT / "conditioning",
        noise_thresh=0.3,
        uV_thresh=0.5e3,
        recalc=False,
    )
    motion_dir = PIPELINE_ROOT / "motion/dredge-motion"
    motion = Motion(
        displacement=np.load(motion_dir / "motion.npy"),
        temporal_bins_s=np.load(motion_dir / "time_bins.npy"),
        spatial_bins_um=np.load(motion_dir / "depth_bins.npy"),
    )
    return astype(
        interpolate_motion(astype(for_sorting, "float"), motion, border_mode="force_zeros"),
        "int16",
    )


def prepare_windows(output_dir: Path, fs: float) -> None:
    import spikeinterface.full as si

    corrected = make_motion_corrected_recording()
    observed_fs = float(corrected.get_sampling_frequency())
    if not np.isclose(observed_fs, fs, rtol=0, atol=1e-6):
        raise RuntimeError(f"Sampling-rate mismatch: reconstructed={observed_fs}, ops={fs}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for window in WINDOWS:
        target = output_dir / "recordings" / window.name
        start, stop = window_frames(window, fs)
        metadata = target / "window_manifest.json"
        if target.exists():
            if not metadata.exists():
                raise RuntimeError(f"Refusing ambiguous existing recording folder: {target}")
            saved = si.load_extractor(target)
            expected = stop - start
            if saved.get_num_samples() != expected:
                raise RuntimeError(f"Cached window length mismatch in {target}")
            print(f"Reusing {target}")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        sliced = corrected.frame_slice(start_frame=start, end_frame=stop)
        sliced.save(folder=target, n_jobs=-1, chunk_duration="1s", progress_bar=True)
        metadata.write_text(
            json.dumps(
                {
                    **asdict(window),
                    "start_sample": start,
                    "stop_sample": stop,
                    "sampling_frequency_hz": fs,
                    "source": str(RAW_ROOT),
                    "conditioning_cache": str(PIPELINE_ROOT / "conditioning"),
                    "motion_cache": str(PIPELINE_ROOT / "motion/dredge-motion"),
                },
                indent=2,
            )
            + "\n"
        )


def assert_gpu_and_patch() -> None:
    import torch
    from kilosort.parameters import DEFAULT_SETTINGS

    missing = {
        "cross_peel_claim_ms",
        "cross_peel_claim_um",
    } - set(DEFAULT_SETTINGS)
    if missing:
        raise RuntimeError(f"Kilosort is missing claim-mask parameters: {sorted(missing)}")
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU is available; refusing to start Kilosort jobs")


def run_sorts(output_dir: Path) -> None:
    os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/luke-claimmask-numba-cache")
    import spikeinterface.full as si
    from spikeinterface.sorters import run_sorter

    assert_gpu_and_patch()
    for window in WINDOWS:
        recording_dir = output_dir / "recordings" / window.name
        if not recording_dir.exists():
            raise FileNotFoundError(f"Prepare the recording first: {recording_dir}")
        recording = si.load_extractor(recording_dir)
        for setting in CLAIM_GRID:
            job_dir = output_dir / "sorts" / window.name / setting.name
            result = job_dir / "sorter_output/spike_times.npy"
            if result.exists():
                print(f"Reusing completed sort {job_dir}")
                continue
            if job_dir.exists():
                raise RuntimeError(
                    f"Partial/ambiguous sort exists at {job_dir}; move it aside before retrying"
                )
            job_dir.parent.mkdir(parents=True, exist_ok=True)
            params = build_sorter_params(setting)
            print(f"Running {window.name} / {setting.name}")
            run_sorter(
                "kilosort4",
                recording,
                folder=str(job_dir),
                verbose=True,
                remove_existing_folder=False,
                **params,
            )


def local_match_mask(
    event_samples: np.ndarray,
    event_depths: np.ndarray,
    spike_samples: np.ndarray,
    spike_depths: np.ndarray,
    time_tolerance: int,
    depth_tolerance_um: float,
) -> np.ndarray:
    order = np.argsort(spike_samples)
    times = np.asarray(spike_samples, dtype=np.int64)[order]
    depths = np.asarray(spike_depths, dtype=float)[order]
    present = np.zeros(len(event_samples), dtype=bool)
    for index, (sample, depth) in enumerate(zip(event_samples, event_depths)):
        left = np.searchsorted(times, sample - time_tolerance, side="left")
        right = np.searchsorted(times, sample + time_tolerance, side="right")
        present[index] = np.any(np.abs(depths[left:right] - depth) <= depth_tolerance_um)
    return present


def cross_unit_near_coincident_fraction(
    times: np.ndarray,
    clusters: np.ndarray,
    depths: np.ndarray,
    time_tolerance: int,
    depth_tolerance_um: float = 75.0,
) -> float:
    """Fraction of spikes in a near-synchronous, nearby, cross-unit pair."""
    order = np.argsort(times)
    times = np.asarray(times, dtype=np.int64)[order]
    clusters = np.asarray(clusters)[order]
    depths = np.asarray(depths, dtype=float)[order]
    marked = np.zeros(len(times), dtype=bool)
    for left in range(len(times)):
        right = left + 1
        while right < len(times) and times[right] - times[left] <= time_tolerance:
            if clusters[right] != clusters[left] and abs(depths[right] - depths[left]) <= depth_tolerance_um:
                marked[left] = True
                marked[right] = True
            right += 1
    return float(marked.mean()) if len(marked) else 0.0


def _population_masks(events: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "visual_neural_unmatched": (events["review_label"] == "neural")
        & (events["status"] == "unmatched"),
        "automatic_neural_like_unmatched": events["automatic_neural_like"]
        & (events["status"] == "unmatched"),
        "all_reviewed": pd.Series(True, index=events.index),
    }


def score_sorts(
    output_dir: Path, review_path: Path, fs: float, n_jitters: int, seed: int
) -> pd.DataFrame:
    events = pd.read_csv(review_path)
    tolerance = int(round(0.5e-3 * fs))
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for window in WINDOWS:
        selected = events_in_window(events, window)
        for setting in CLAIM_GRID:
            result_dir = output_dir / "sorts" / window.name / setting.name / "sorter_output"
            if not (result_dir / "spike_times.npy").exists():
                continue
            times = np.load(result_dir / "spike_times.npy").reshape(-1).astype(np.int64)
            clusters = np.load(result_dir / "spike_clusters.npy").reshape(-1)
            positions = np.load(result_dir / "spike_positions.npy")
            # Kilosort can export a handful of batch-padding spikes just beyond
            # a short recording boundary.  They are not real samples in this
            # window and must not enter recovery or quality metrics.
            _, stop_sample = window_frames(Window(window.name, 0.0, window.duration_s), fs)
            valid = (times >= 0) & (times < stop_sample)
            n_excess_spikes = int((~valid).sum())
            times = times[valid]
            clusters = clusters[valid]
            depths = positions[valid, 1]
            labels_path = result_dir / "cluster_KSLabel.tsv"
            contamination_path = result_dir / "cluster_ContamPct.tsv"
            n_ks_good = np.nan
            median_contamination = np.nan
            if labels_path.exists():
                labels = pd.read_csv(labels_path, sep="\t")
                label_column = next(column for column in labels if column != "cluster_id")
                n_ks_good = int(labels[label_column].astype(str).str.lower().eq("good").sum())
            if contamination_path.exists():
                contamination = pd.read_csv(contamination_path, sep="\t")
                contam_column = next(
                    column for column in contamination if column != "cluster_id"
                )
                median_contamination = float(
                    np.median(contamination[contam_column].to_numpy(float))
                )
            duplicate_fraction = cross_unit_near_coincident_fraction(
                times, clusters, depths, tolerance
            )
            isi_limit = int(round(1.5e-3 * fs))
            refractory = []
            active_bin_fractions = []
            n_time_bins = max(1, int(np.ceil(window.duration_s / 10.0)))
            for unit in np.unique(clusters):
                unit_times = np.sort(times[clusters == unit])
                if len(unit_times) > 1:
                    refractory.append(float(np.mean(np.diff(unit_times) < isi_limit)))
                occupied = np.unique(
                    np.minimum(
                        (unit_times / (10.0 * fs)).astype(int), n_time_bins - 1
                    )
                )
                active_bin_fractions.append(len(occupied) / n_time_bins)
            for population, mask in _population_masks(selected).items():
                subset = selected[mask]
                samples = event_local_samples(subset["sample_index"].to_numpy(), window, fs)
                event_depths = subset["peak_depth_um"].to_numpy(float)
                observed = float(
                    local_match_mask(samples, event_depths, times, depths, tolerance, 100.0).mean()
                ) if len(subset) else np.nan
                null = []
                if len(subset):
                    for _ in range(n_jitters):
                        offsets_ms = rng.uniform(20.0, 500.0, len(subset))
                        offsets_ms *= rng.choice((-1.0, 1.0), len(subset))
                        shifted = samples + np.rint(offsets_ms * fs / 1000.0).astype(np.int64)
                        null.append(
                            float(local_match_mask(shifted, event_depths, times, depths, tolerance, 100.0).mean())
                        )
                rows.append(
                    {
                        "window": window.name,
                        "setting": setting.name,
                        "claim_ms": setting.claim_ms,
                        "claim_um": setting.claim_um,
                        "population": population,
                        "n_events": len(subset),
                        "observed_recovery": observed,
                        "jitter_null_mean": float(np.mean(null)) if null else np.nan,
                        "recovery_above_null": observed - float(np.mean(null)) if null else np.nan,
                        "n_spikes": len(times),
                        "n_excess_spikes_removed": n_excess_spikes,
                        "n_units": len(np.unique(clusters)),
                        "n_ks_good": n_ks_good,
                        "median_contamination_pct": median_contamination,
                        "cross_unit_near_coincident_fraction": duplicate_fraction,
                        "median_unit_refractory_violation_fraction": float(np.median(refractory)) if refractory else np.nan,
                        "median_unit_active_10s_bin_fraction": float(np.median(active_bin_fractions)) if active_bin_fractions else np.nan,
                    }
                )
    result = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_dir / "claimmask_window_sweep_scores.csv", index=False)
    return result


def main() -> None:
    args = parse_args()
    validate_inputs(args.review_events)
    manifest = plan(args.review_events, args.output_dir)
    print(json.dumps(manifest, indent=2))
    if args.plan_only:
        return
    if not (args.prepare or args.run or args.score):
        raise SystemExit("Choose at least one of --prepare, --run, --score, or --plan-only")
    _, fs = load_reference_settings()
    if args.prepare:
        prepare_windows(args.output_dir, fs)
    if args.run:
        run_sorts(args.output_dir)
    if args.score:
        scores = score_sorts(
            args.output_dir, args.review_events, fs, args.n_jitters, args.seed
        )
        print(scores.to_string(index=False) if len(scores) else "No completed sorts to score")


if __name__ == "__main__":
    main()
