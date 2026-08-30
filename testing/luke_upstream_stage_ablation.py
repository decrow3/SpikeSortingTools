"""Trace fixed Luke imec1 events through conditioning and motion correction.

This diagnostic is deliberately upstream of Kilosort.  It reconstructs the
lazy SpikeInterface preprocessing graph used for Luke 2025-08-04, adds a few
one-factor counterfactuals, and measures the same reviewed events after every
stage.  It does not change production defaults or materialize a full binary.

Run with the repository's ``spikeinterface`` environment::

    python testing/luke_upstream_stage_ablation.py

Use ``--motion-only`` for the inexpensive displacement/support summary, or
``--max-events 10`` for a smoke test of waveform extraction.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

LUKE_ROOT = Path("/mnt/NPX/Luke/20250804")
RAW_ROOT = LUKE_ROOT / "Luke0804_V2V1_g0"
PIPELINE_ROOT = LUKE_ROOT / "dredge_pipeline_results_Luke0804_V2V1_g0_imec1"
MOTION_ROOT = PIPELINE_ROOT / "motion"
DEFAULT_REVIEW = REPO_ROOT / "testing/outputs/luke_multichannel_event_validation/imec1/event_stage_trace.csv"
DEFAULT_OUTPUT = REPO_ROOT / "testing/outputs/luke_upstream_stage_ablation/imec1"
STREAM_ID = "imec1.ap"


@dataclass(frozen=True)
class Window:
    name: str
    start_s: float
    duration_s: float


WINDOWS = (
    Window("shared_template", 7095.0, 240.0),
    Window("registration_outlier", 8160.0, 120.0),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-events", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--motion-only", action="store_true")
    parser.add_argument("--max-events", type=int)
    parser.add_argument(
        "--stage",
        dest="stages",
        action="append",
        help="Extract only this named stage (repeatable; minimal_bandpass is required as baseline)",
    )
    return parser.parse_args()


def robust_sigma(values: np.ndarray, axis=None) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    median = np.median(values, axis=axis, keepdims=True)
    mad = np.median(np.abs(values - median), axis=axis)
    return np.maximum(mad / 0.6744897501960817, np.finfo(float).eps)


def normalized_correlation(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64).ravel()
    second = np.asarray(second, dtype=np.float64).ravel()
    first -= np.mean(first)
    second -= np.mean(second)
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    return float(np.dot(first, second) / denominator) if denominator else float("nan")


def max_channel_shift_correlation(
    reference: np.ndarray, candidate: np.ndarray, max_shift_channels: int = 8
) -> float:
    """Return the best flattened correlation after a small channel-axis shift."""
    if reference.shape != candidate.shape or reference.ndim != 2:
        raise ValueError("Waveform arrays must have the same (time, channel) shape")
    correlations = []
    for shift in range(-max_shift_channels, max_shift_channels + 1):
        if shift < 0:
            correlations.append(normalized_correlation(reference[:, -shift:], candidate[:, :shift]))
        elif shift > 0:
            correlations.append(normalized_correlation(reference[:, :-shift], candidate[:, shift:]))
        else:
            correlations.append(normalized_correlation(reference, candidate))
    finite = np.asarray(correlations)[np.isfinite(correlations)]
    return float(np.max(finite)) if finite.size else float("nan")


def _depth_profile(values: np.ndarray, depths: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unique_depths = np.unique(depths)
    profile = np.array([np.max(values[depths == depth]) for depth in unique_depths])
    return unique_depths, profile


def event_metrics(
    traces: np.ndarray,
    channel_depths_um: np.ndarray,
    fs: float,
    search_half_ms: float = 0.6,
    core_half_ms: float = 1.5,
) -> tuple[dict[str, float], np.ndarray]:
    """Measure waveform distortion and fragmentation in one centered snippet."""
    traces = np.asarray(traces, dtype=np.float32)
    if traces.ndim != 2 or traces.shape[1] != len(channel_depths_um):
        raise ValueError("traces must have shape (samples, channels)")
    center = traces.shape[0] // 2
    baseline_exclusion = int(round(2.0e-3 * fs))
    baseline_mask = np.ones(traces.shape[0], dtype=bool)
    baseline_mask[max(0, center - baseline_exclusion) : center + baseline_exclusion + 1] = False
    noise = robust_sigma(traces[baseline_mask], axis=0)
    search_half = int(round(search_half_ms * 1e-3 * fs))
    search = traces[center - search_half : center + search_half + 1] / noise[None, :]
    local_time, peak_channel = np.unravel_index(int(np.argmin(search)), search.shape)
    aligned = center - search_half + int(local_time)

    event_half = 1
    negative_amplitude = np.maximum(
        0.0, -np.min(traces[aligned - event_half : aligned + event_half + 1], axis=0)
    )
    snr = negative_amplitude / noise
    active = snr >= 4.0
    weights = np.where(active, np.maximum(snr**2 - 16.0, 0.0), 0.0)
    peak_depth = float(channel_depths_um[peak_channel])
    local = np.abs(channel_depths_um - peak_depth) <= 100.0
    total_weight = float(np.sum(weights))
    local_fraction = float(np.sum(weights[local]) / total_weight) if total_weight else 0.0
    if total_weight:
        depth_center = float(np.sum(weights * channel_depths_um) / total_weight)
        depth_sd = float(
            np.sqrt(np.sum(weights * (channel_depths_um - depth_center) ** 2) / total_weight)
        )
    else:
        depth_sd = float("nan")

    waveform = traces[:, peak_channel]
    peak_noise = float(noise[peak_channel])
    extrema, _ = find_peaks(
        np.abs(waveform),
        height=3.0 * peak_noise,
        prominence=1.5 * peak_noise,
        distance=max(1, int(round(0.15e-3 * fs))),
    )
    core_half = int(round(0.5e-3 * fs))
    broad_half = int(round(3.0e-3 * fs))
    broad = waveform[max(0, aligned - broad_half) : aligned + broad_half + 1]
    core = waveform[max(0, aligned - core_half) : aligned + core_half + 1]
    broad_energy = float(np.sum(np.square(broad, dtype=np.float64)))
    core_energy = float(np.sum(np.square(core, dtype=np.float64)))
    sidelobe_ratio = max(0.0, broad_energy - core_energy) / max(core_energy, np.finfo(float).eps)

    _, profile = _depth_profile(snr, np.asarray(channel_depths_um))
    spatial_peaks, _ = find_peaks(profile, height=4.0, prominence=1.0, distance=2)
    if profile.size and profile[0] >= 4.0:
        spatial_peaks = np.append(spatial_peaks, 0)
    if profile.size > 1 and profile[-1] >= 4.0:
        spatial_peaks = np.append(spatial_peaks, profile.size - 1)

    core_wave_half = int(round(core_half_ms * 1e-3 * fs))
    core_wave = traces[
        max(0, aligned - core_wave_half) : aligned + core_wave_half + 1
    ].copy()
    expected = 2 * core_wave_half + 1
    if core_wave.shape[0] != expected:
        raise ValueError("Snippet is too short around the aligned event")

    metrics = {
        "aligned_offset_ms": float((aligned - center) * 1e3 / fs),
        "peak_channel": int(peak_channel),
        "peak_depth_um": peak_depth,
        "peak_amplitude_counts": float(max(0.0, -np.min(waveform))),
        "peak_snr": float(snr[peak_channel]),
        "active_channels": int(np.sum(active)),
        "local_energy_fraction": local_fraction,
        "footprint_depth_sd_um": depth_sd,
        "temporal_extrema_3sigma": int(len(extrema)),
        "extra_temporal_extrema": int(max(0, len(extrema) - 1)),
        "sidelobe_to_core_energy": float(sidelobe_ratio),
        "spatial_peak_count_4sigma": int(len(np.unique(spatial_peaks))),
        "zero_fraction": float(np.mean(traces == 0)),
    }
    return metrics, core_wave


def motion_window_metrics(
    displacement: np.ndarray,
    temporal_bins_s: np.ndarray,
    spatial_bins_um: np.ndarray,
    window: Window,
    recording_t_start_s: float,
) -> dict[str, float]:
    absolute_start = recording_t_start_s + window.start_s
    mask = (temporal_bins_s >= absolute_start) & (
        temporal_bins_s < absolute_start + window.duration_s
    )
    values = np.asarray(displacement[mask], dtype=float)
    if values.shape[0] < 2:
        raise ValueError(f"Too few motion bins for {window.name}")
    rigid = np.median(values, axis=1)
    spread = np.percentile(values, 95, axis=1) - np.percentile(values, 5, axis=1)
    bin_steps = np.abs(np.diff(values, axis=0))
    gradients = np.abs(np.diff(values, axis=1) / np.diff(spatial_bins_um)[None, :])
    maximum_index = int(np.argmax(spread))
    return {
        "window": window.name,
        "recording_t_start_s": recording_t_start_s,
        "absolute_start_s": absolute_start,
        "n_motion_bins": int(values.shape[0]),
        "rigid_excursion_p95_p5_um": float(np.percentile(rigid, 95) - np.percentile(rigid, 5)),
        "median_nonrigid_spread_um": float(np.median(spread)),
        "p95_nonrigid_spread_um": float(np.percentile(spread, 95)),
        "max_nonrigid_spread_um": float(np.max(spread)),
        "max_spread_relative_time_s": float(temporal_bins_s[mask][maximum_index] - recording_t_start_s),
        "p99_abs_rigid_step_um": float(np.percentile(np.abs(np.diff(rigid)), 99)),
        "p99_abs_depth_bin_step_um": float(np.percentile(bin_steps, 99)),
        "p99_abs_spatial_gradient_um_per_um": float(np.percentile(gradients, 99)),
    }


def add_peak_support(
    summaries: list[dict[str, float]],
    peaks: np.ndarray,
    peak_locations: np.ndarray,
    fs: float,
    depth_edges_um: np.ndarray,
) -> None:
    sample_indices = peaks["sample_index"]
    for summary, window in zip(summaries, WINDOWS):
        start = int(round(window.start_s * fs))
        stop = int(round((window.start_s + window.duration_s) * fs))
        left = int(np.searchsorted(sample_indices, start, side="left"))
        right = int(np.searchsorted(sample_indices, stop, side="left"))
        times = np.asarray(sample_indices[left:right], dtype=float) / fs - window.start_s
        depths = np.asarray(peak_locations["y"][left:right], dtype=float)
        time_edges = np.arange(0.0, window.duration_s + 1.0, 1.0)
        counts, _, _ = np.histogram2d(times, depths, bins=(time_edges, depth_edges_um))
        summary.update(
            n_motion_detection_peaks=int(right - left),
            motion_detection_peaks_per_s=float((right - left) / window.duration_s),
            median_peaks_per_time_depth_bin=float(np.median(counts)),
            p10_peaks_per_time_depth_bin=float(np.percentile(counts, 10)),
            empty_time_depth_bin_fraction=float(np.mean(counts == 0)),
            low_support_bin_fraction_lt20=float(np.mean(counts < 20)),
        )


def build_recording_stages(raw):
    from spikeinterface.preprocessing import (
        astype,
        blank_staturation,
        common_reference,
        filter,
        interpolate_bad_channels,
        phase_shift,
    )
    from spikeinterface.sortingcomponents.motion import interpolate_motion
    from spikeinterface.core.motion import Motion

    gain_values = np.unique(raw.get_property("gain_to_uV"))
    if len(gain_values) != 1:
        raise ValueError(f"Expected one gain value, got {gain_values}")
    gain_uv_per_bit = float(gain_values[0])
    shifted = phase_shift(raw) if np.any(raw.get_property("inter_sample_shift")) else raw
    blanked = blank_staturation(shifted, 500.0 / gain_uv_per_bit, direction="both")
    similarity, noise = np.load(PIPELINE_ROOT / "conditioning/channel_metrics.npy")
    bad = (similarity < -0.5) | (noise > 0.3)
    bad_ids = raw.get_channel_ids()[bad]
    interpolated = interpolate_bad_channels(blanked, bad_ids)

    def wideband(recording):
        return filter(
            recording,
            band=[300.0, 6000.0],
            btype="bandpass",
            filter_order=12,
            ftype="butter",
            direction="forward-backward",
        )

    minimal = wideband(shifted)
    blanked_bandpass = wideband(blanked)
    interpolated_bandpass = wideband(interpolated)
    blanked_local = common_reference(
        blanked_bandpass, reference="local", operator="median", local_radius=(40, 140)
    )
    current = common_reference(
        interpolated_bandpass, reference="local", operator="median", local_radius=(40, 140)
    )
    minimal_local = common_reference(
        minimal, reference="local", operator="median", local_radius=(40, 140)
    )
    global_control = common_reference(
        interpolated_bandpass, reference="global", operator="median"
    )
    # Counterfactual for an otherwise identical pipeline that does not quantize
    # the output of the high-order filter and reference operations to int16.
    interpolated_float = astype(interpolated, "float32")
    float_bandpass = filter(
        interpolated_float,
        band=[300.0, 6000.0],
        btype="bandpass",
        filter_order=12,
        ftype="butter",
        direction="forward-backward",
        dtype="float32",
    )
    float_conditioned = common_reference(
        float_bandpass,
        reference="local",
        operator="median",
        local_radius=(40, 140),
        dtype="float32",
    )
    motion_dir = MOTION_ROOT / "dredge-motion"
    motion = Motion(
        displacement=np.load(motion_dir / "motion.npy"),
        temporal_bins_s=np.load(motion_dir / "time_bins.npy"),
        spatial_bins_um=np.load(motion_dir / "depth_bins.npy"),
    )
    corrected = astype(
        interpolate_motion(astype(current, "float"), motion, border_mode="force_zeros"),
        "int16",
    )
    float_corrected = astype(
        interpolate_motion(float_conditioned, motion, border_mode="force_zeros"),
        "int16",
    )
    return {
        "interpolated_unfiltered": interpolated,
        "minimal_bandpass": minimal,
        "blanked_bandpass": blanked_bandpass,
        "interpolated_bandpass": interpolated_bandpass,
        "blanked_local_reference_control": blanked_local,
        "minimal_local_reference_control": minimal_local,
        "global_reference_control": global_control,
        "current_conditioned": current,
        "float_conditioned_control": float_conditioned,
        "motion_corrected": corrected,
        "float_motion_corrected_control": float_corrected,
    }, bad_ids, gain_uv_per_bit


def selected_events(review_path: Path, max_events: int | None) -> pd.DataFrame:
    events = pd.read_csv(review_path)
    mask = np.zeros(len(events), dtype=bool)
    for window in WINDOWS:
        mask |= (events["time_seconds"] >= window.start_s) & (
            events["time_seconds"] < window.start_s + window.duration_s
        )
    events = events.loc[mask].sort_values("sample_index").reset_index(drop=True)
    if max_events is not None:
        if max_events < 1:
            raise ValueError("--max-events must be positive")
        events = events.iloc[:max_events].copy()
    return events


def extract_stage_metrics(stages: dict, events: pd.DataFrame, fs: float) -> pd.DataFrame:
    half_samples = int(round(5.0e-3 * fs))
    channel_depths = np.asarray(next(iter(stages.values())).get_channel_locations())[:, 1]
    rows: list[dict] = []
    baseline_waves: dict[str, np.ndarray] = {}
    for stage_name, recording in stages.items():
        print(f"Extracting {len(events)} events from {stage_name}", flush=True)
        for event in events.itertuples(index=False):
            sample = int(event.sample_index)
            traces = recording.get_traces(
                start_frame=sample - half_samples,
                end_frame=sample + half_samples + 1,
                return_scaled=False,
            )
            metrics, core_wave = event_metrics(traces, channel_depths, fs)
            if stage_name == "minimal_bandpass":
                baseline_waves[event.review_id] = core_wave
                correlation = 1.0
            else:
                correlation = max_channel_shift_correlation(
                    baseline_waves[event.review_id], core_wave
                )
            rows.append(
                {
                    "review_id": event.review_id,
                    "sample_index": sample,
                    "time_seconds": float(event.time_seconds),
                    "window": event.window,
                    "status": event.status,
                    "review_label": event.review_label,
                    "automatic_neural_like": bool(event.automatic_neural_like),
                    "stage": stage_name,
                    "max_shift_waveform_correlation_to_minimal": correlation,
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def summarize_stage_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    value_columns = [
        "peak_amplitude_counts",
        "peak_snr",
        "active_channels",
        "local_energy_fraction",
        "footprint_depth_sd_um",
        "extra_temporal_extrema",
        "sidelobe_to_core_energy",
        "spatial_peak_count_4sigma",
        "zero_fraction",
        "max_shift_waveform_correlation_to_minimal",
    ]
    groups = ["window", "review_label", "status", "stage"]
    summary = metrics.groupby(groups, dropna=False)[value_columns].median().reset_index()
    counts = metrics.groupby(groups, dropna=False).size().rename("n_events").reset_index()
    return counts.merge(summary, on=groups, validate="one_to_one")


def main() -> None:
    args = parse_args()
    os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/luke-upstream-numba-cache")
    import spikeinterface.extractors as se

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw = se.read_spikeglx(
        folder_path=RAW_ROOT, load_sync_channel=False, stream_id=STREAM_ID
    )
    fs = float(raw.get_sampling_frequency())
    t_start = float(raw.get_time_info()["t_start"] or 0.0)
    motion_dir = MOTION_ROOT / "dredge-motion"
    displacement = np.load(motion_dir / "motion.npy")
    temporal_bins = np.load(motion_dir / "time_bins.npy")
    spatial_bins = np.load(motion_dir / "depth_bins.npy")
    motion_summaries = [
        motion_window_metrics(displacement, temporal_bins, spatial_bins, window, t_start)
        for window in WINDOWS
    ]
    peaks = np.load(MOTION_ROOT / "peaks.npy", mmap_mode="r")
    locations = np.load(MOTION_ROOT / "peak_locations.npy", mmap_mode="r")
    if len(spatial_bins) > 1:
        midpoints = (spatial_bins[:-1] + spatial_bins[1:]) / 2
        depth_edges = np.r_[
            spatial_bins[0] - (midpoints[0] - spatial_bins[0]),
            midpoints,
            spatial_bins[-1] + (spatial_bins[-1] - midpoints[-1]),
        ]
    else:
        depth_edges = np.array([-np.inf, np.inf])
    add_peak_support(motion_summaries, peaks, locations, fs, depth_edges)
    pd.DataFrame(motion_summaries).to_csv(args.output_dir / "motion_window_summary.csv", index=False)

    manifest = {
        "raw_root": str(RAW_ROOT),
        "stream_id": STREAM_ID,
        "review_events": str(args.review_events),
        "sampling_frequency_hz": fs,
        "recording_t_start_s": t_start,
        "windows": [asdict(window) for window in WINDOWS],
        "motion_bins_are_absolute_time": True,
        "motion_only": args.motion_only,
        "max_events": args.max_events,
    }
    if not args.motion_only:
        events = selected_events(args.review_events, args.max_events)
        stages, bad_ids, gain = build_recording_stages(raw)
        if args.stages:
            unknown = set(args.stages) - set(stages)
            if unknown:
                raise ValueError(f"Unknown stages: {sorted(unknown)}")
            if "minimal_bandpass" not in args.stages:
                raise ValueError("--stage minimal_bandpass is required as the waveform baseline")
            stages = {name: stage for name, stage in stages.items() if name in args.stages}
        manifest.update(
            stages=list(stages),
            n_events=len(events),
            bad_channel_ids=[str(value) for value in bad_ids],
            gain_uv_per_bit=gain,
        )
        metrics = extract_stage_metrics(stages, events, fs)
        metrics.to_csv(args.output_dir / "event_stage_metrics.csv", index=False)
        summarize_stage_metrics(metrics).to_csv(
            args.output_dir / "event_stage_summary.csv", index=False
        )
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote upstream diagnostic outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
