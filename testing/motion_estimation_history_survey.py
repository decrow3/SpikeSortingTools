"""Survey historical motion-estimation caches without applying motion correction.

The survey intentionally separates estimator diagnostics from voltage resampling
and sorting outcomes.  It profiles cached peak support, saved displacement fields,
and cross-method agreement.  Historical pipeline caches do not consistently save
peak-detection or preprocessing parameters, so those are reported only when they
are explicitly recoverable from a manifest or MEDiCINe parameter file.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/motion-estimation-history-mpl")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


NPX_ROOT = Path("/mnt/NPX")
PATCH_ROOT = Path("/media/huklab/Data/NPX/Spikesorting")
DEFAULT_OUTPUT = Path("testing/outputs/motion_estimation_history_survey")
SAMPLING_FREQUENCY_HZ = 30_000.0
MAX_PROFILE_PEAKS = 100_000
METHOD_DIRS = {
    "kilosort_style": "ks-motion",
    "decentralized": "decentralized-motion",
    "dredge": "dredge-motion",
    "medicine": "medicine",
}


@dataclass(frozen=True)
class SurveyConfig:
    roots: tuple[str, ...] = (str(NPX_ROOT), str(PATCH_ROOT))
    maximum_depth: int = 7
    maximum_profile_peaks: int = MAX_PROFILE_PEAKS
    sampling_frequency_hz: float = SAMPLING_FREQUENCY_HZ


def discover_motion_dirs(config: SurveyConfig) -> list[Path]:
    """Find pipeline result motion directories while bounding mount traversal."""
    found: set[Path] = set()
    for root_text in config.roots:
        root = Path(root_text)
        if not root.exists():
            continue
        result = subprocess.run(
            [
                "find",
                str(root),
                "-mindepth",
                "1",
                "-maxdepth",
                str(config.maximum_depth),
                "-type",
                "d",
                "-path",
                "*pipeline_results*/motion",
                "-print",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        found.update(Path(line) for line in result.stdout.splitlines() if line)
    return sorted(found)


def _safe_percentile(values: np.ndarray, percentiles: Iterable[float]) -> np.ndarray:
    finite = np.asarray(values)[np.isfinite(values)]
    if finite.size == 0:
        return np.full(len(tuple(percentiles)), np.nan)
    return np.nanpercentile(finite, tuple(percentiles))


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    mask = np.isfinite(left) & np.isfinite(right)
    if mask.sum() < 4:
        return np.nan
    left_valid = left[mask]
    right_valid = right[mask]
    if np.nanstd(left_valid) == 0 or np.nanstd(right_valid) == 0:
        return np.nan
    return float(np.corrcoef(left_valid, right_valid)[0, 1])


def _normalize_motion_shape(
    displacement: np.ndarray, time_bins: np.ndarray, depth_bins: np.ndarray
) -> np.ndarray:
    displacement = np.asarray(displacement, dtype=float)
    displacement = np.squeeze(displacement)
    if displacement.ndim == 1:
        displacement = displacement[:, None]
    if displacement.shape[0] == time_bins.size:
        return displacement
    if displacement.shape[1] == time_bins.size:
        return displacement.T
    raise ValueError(
        f"Motion shape {displacement.shape} does not match "
        f"{time_bins.size} time bins and {depth_bins.size} depth bins"
    )


def _path_identity(motion_dir: Path) -> dict[str, object]:
    parts = motion_dir.parts
    subject = "unknown"
    session = "unknown"
    storage = "other"
    if str(motion_dir).startswith(str(NPX_ROOT)) and len(parts) > 4:
        storage = "mnt_npx"
        subject = parts[3]
        session = parts[4]
    elif str(motion_dir).startswith(str(PATCH_ROOT)):
        storage = "patching_drive"
        subject = "patched"
        session = motion_dir.parent.name
    parent_name = motion_dir.parent.name
    probe = "imec1" if "imec1" in parent_name.lower() else (
        "imec0" if "imec0" in parent_name.lower() else "unknown"
    )
    return {
        "motion_dir": str(motion_dir),
        "storage": storage,
        "subject": subject,
        "session": session,
        "pipeline_name": parent_name,
        "probe": probe,
    }


def profile_peaks(
    motion_dir: Path,
    *,
    maximum_profile_peaks: int,
    sampling_frequency_hz: float,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Profile peak support with a bounded deterministic sample.

    Counts and recording duration are exact when the peak array is sorted by
    sample index. Distribution metrics use a deterministic set of contiguous
    blocks. Rate variability is computed only inside observed blocks so
    unsampled intervals are never mistaken for zero-rate intervals.
    """
    peak_path = motion_dir / "peaks.npy"
    location_path = motion_dir / "peak_locations.npy"
    empty = {
        "has_peaks": False,
        "has_peak_locations": location_path.exists(),
        "peak_count": np.nan,
        "peak_duration_s": np.nan,
        "peak_rate_hz": np.nan,
        "profiled_peak_count": 0,
        "peak_rate_cv_10s": np.nan,
        "peak_rate_p99_over_median_10s": np.nan,
        "peak_rate_low_decile_fraction": np.nan,
        "positive_peak_fraction": np.nan,
        "abs_amplitude_p50": np.nan,
        "abs_amplitude_p90": np.nan,
        "abs_amplitude_p99": np.nan,
        "depth_occupancy_entropy": np.nan,
        "time_depth_low_support_fraction": np.nan,
    }
    if not peak_path.exists():
        return empty, pd.DataFrame()

    peaks = np.load(peak_path, mmap_mode="r")
    count = len(peaks)
    if count == 0:
        empty.update({"has_peaks": True, "peak_count": 0, "profiled_peak_count": 0})
        return empty, pd.DataFrame()
    names = peaks.dtype.names or ()
    if "sample_index" not in names:
        empty.update({"has_peaks": True, "peak_count": count})
        return empty, pd.DataFrame()

    first_sample = float(peaks[0]["sample_index"])
    last_sample = float(peaks[-1]["sample_index"])
    duration_s = max((last_sample - first_sample) / sampling_frequency_hz, 1e-9)
    if count <= maximum_profile_peaks:
        sample_slices = [(0, count)]
    else:
        block_count = min(3, maximum_profile_peaks)
        block_size = max(1, maximum_profile_peaks // block_count)
        starts = np.linspace(0, count - block_size, block_count, dtype=np.int64)
        sample_slices = [(int(start), int(start + block_size)) for start in starts]
    selected_peaks = np.concatenate([np.asarray(peaks[start:stop]) for start, stop in sample_slices])
    block_relative_times: list[np.ndarray] = []
    block_edges: list[np.ndarray] = []
    block_centers: list[np.ndarray] = []
    block_rates: list[np.ndarray] = []
    for start, stop in sample_slices:
        block = np.asarray(peaks[start:stop])
        block_time = (block["sample_index"].astype(float) - first_sample) / sampling_frequency_hz
        block_relative_times.append(block_time)
        if block_time.size < 2 or block_time[-1] <= block_time[0]:
            continue
        local_edges = np.arange(block_time[0], block_time[-1], 10.0)
        if local_edges.size == 0 or local_edges[-1] < block_time[-1]:
            local_edges = np.append(local_edges, block_time[-1])
        if local_edges.size < 3:
            local_edges = np.linspace(block_time[0], block_time[-1], 3)
        local_counts, _ = np.histogram(block_time, bins=local_edges)
        local_widths = np.diff(local_edges)
        valid = local_widths >= min(5.0, float(np.nanmax(local_widths)))
        block_edges.append(local_edges)
        block_centers.append(((local_edges[:-1] + local_edges[1:]) / 2)[valid])
        block_rates.append((local_counts.astype(float) / local_widths)[valid])
    relative_time_s = np.concatenate(block_relative_times)
    rates_10s = np.concatenate(block_rates) if block_rates else np.array([], dtype=float)
    median_rate = float(np.nanmedian(rates_10s)) if rates_10s.size else np.nan
    rate_cv = (
        float(np.nanstd(rates_10s) / np.nanmean(rates_10s))
        if rates_10s.size and np.nanmean(rates_10s) > 0
        else np.nan
    )
    burst_ratio = (
        float(np.nanpercentile(rates_10s, 99) / median_rate)
        if np.isfinite(median_rate) and median_rate > 0
        else np.nan
    )
    low_decile = (
        float(np.mean(rates_10s <= np.nanpercentile(rates_10s, 10)))
        if rates_10s.size
        else np.nan
    )

    amplitudes = (
        selected_peaks["amplitude"].astype(float)
        if "amplitude" in names
        else np.full(len(selected_peaks), np.nan)
    )
    amplitude_percentiles = _safe_percentile(np.abs(amplitudes), (50, 90, 99))

    locations = None
    depth_entropy = np.nan
    low_support_fraction = np.nan
    time_depth = pd.DataFrame()
    if location_path.exists():
        peak_locations = np.load(location_path, mmap_mode="r")
        if len(peak_locations) == count and peak_locations.dtype.names and "y" in peak_locations.dtype.names:
            locations = np.concatenate(
                [
                    np.asarray(peak_locations[start:stop]["y"], dtype=float)
                    for start, stop in sample_slices
                ]
            )
            finite_depth = locations[np.isfinite(locations)]
            if finite_depth.size:
                depth_edges = np.linspace(
                    np.nanpercentile(finite_depth, 0.5),
                    np.nanpercentile(finite_depth, 99.5),
                    21,
                )
                if np.unique(depth_edges).size > 2:
                    depth_counts, _ = np.histogram(finite_depth, bins=depth_edges)
                    depth_prob = depth_counts[depth_counts > 0] / depth_counts.sum()
                    depth_entropy = float(
                        -np.sum(depth_prob * np.log(depth_prob)) / np.log(len(depth_counts))
                    )
                    support_cells: list[np.ndarray] = []
                    location_offset = 0
                    for block_time, local_edges, (start, stop) in zip(
                        block_relative_times, block_edges, sample_slices
                    ):
                        block_length = stop - start
                        block_locations = locations[location_offset : location_offset + block_length]
                        location_offset += block_length
                        histogram, _, _ = np.histogram2d(
                            block_time,
                            block_locations,
                            bins=(local_edges, depth_edges),
                        )
                        support_cells.append(histogram.reshape(-1))
                    sampled_support = np.concatenate(support_cells) if support_cells else np.array([])
                    positive_support = sampled_support[sampled_support > 0]
                    if positive_support.size:
                        low_threshold = max(10.0, float(np.nanpercentile(positive_support, 10)))
                        low_support_fraction = float(np.mean(sampled_support < low_threshold))

    time_centers = np.concatenate(block_centers) if block_centers else np.array([], dtype=float)
    time_depth = pd.DataFrame(
        {
            "relative_time_s": time_centers,
            "estimated_peak_rate_hz": rates_10s,
        }
    )

    result = {
        "has_peaks": True,
        "has_peak_locations": locations is not None,
        "peak_count": int(count),
        "peak_duration_s": duration_s,
        "peak_rate_hz": count / duration_s,
        "profiled_peak_count": int(len(selected_peaks)),
        "peak_rate_cv_10s": rate_cv,
        "peak_rate_p99_over_median_10s": burst_ratio,
        "peak_rate_low_decile_fraction": low_decile,
        "positive_peak_fraction": float(np.nanmean(amplitudes > 0)),
        "abs_amplitude_p50": float(amplitude_percentiles[0]),
        "abs_amplitude_p90": float(amplitude_percentiles[1]),
        "abs_amplitude_p99": float(amplitude_percentiles[2]),
        "depth_occupancy_entropy": depth_entropy,
        "time_depth_low_support_fraction": low_support_fraction,
    }
    return result, time_depth


def profile_motion_method(
    method: str,
    method_dir: Path,
    peak_time_profile: pd.DataFrame,
) -> tuple[dict[str, object], pd.DataFrame]:
    motion_path = method_dir / "motion.npy"
    time_path = method_dir / "time_bins.npy"
    depth_path = method_dir / "depth_bins.npy"
    if not (motion_path.exists() and time_path.exists() and depth_path.exists()):
        return {}, pd.DataFrame()

    displacement = np.load(motion_path)
    time_bins = np.asarray(np.load(time_path), dtype=float).reshape(-1)
    depth_bins = np.asarray(np.load(depth_path), dtype=float).reshape(-1)
    field = _normalize_motion_shape(displacement, time_bins, depth_bins)
    finite = np.isfinite(field)
    if not finite.any() or time_bins.size < 2:
        return {}, pd.DataFrame()

    relative_time = time_bins - np.nanmin(time_bins)
    dt = float(np.nanmedian(np.diff(time_bins)))
    rigid = np.nanmedian(field, axis=1)
    rigid_centered = rigid - np.nanmedian(rigid)
    residual = field - rigid[:, None]
    step = np.diff(field, axis=0)
    rigid_step = np.diff(rigid)
    speed = np.abs(rigid_step) / max(dt, 1e-9)
    field_p01, field_p99 = _safe_percentile(field, (1, 99))
    rigid_p05, rigid_p95 = _safe_percentile(rigid, (5, 95))
    step_p50, step_p99 = _safe_percentile(np.abs(step), (50, 99))
    max_step = float(np.nanmax(np.abs(step))) if step.size else np.nan
    spatial_spread = np.nanpercentile(field, 90, axis=1) - np.nanpercentile(field, 10, axis=1)
    spatial_gradient = np.diff(field, axis=1)
    repeated_extreme_fraction = float(
        max(
            np.mean(np.isclose(field, np.nanmin(field), atol=1e-8)),
            np.mean(np.isclose(field, np.nanmax(field), atol=1e-8)),
        )
    )

    peak_rate_at_top_steps = np.nan
    peak_rate_percentile_at_top_steps = np.nan
    speed_peak_rate_correlation = np.nan
    if not peak_time_profile.empty and speed.size:
        peak_t = peak_time_profile["relative_time_s"].to_numpy(dtype=float)
        peak_rate = peak_time_profile["estimated_peak_rate_hz"].to_numpy(dtype=float)
        interpolated_rate = np.interp(relative_time[1:], peak_t, peak_rate)
        speed_peak_rate_correlation = _pearson(speed, interpolated_rate)
        cutoff = np.nanpercentile(speed, 95)
        selected_rate = interpolated_rate[speed >= cutoff]
        if selected_rate.size:
            peak_rate_at_top_steps = float(np.nanmedian(selected_rate))
            sorted_rate = np.sort(peak_rate[np.isfinite(peak_rate)])
            if sorted_rate.size:
                peak_rate_percentile_at_top_steps = float(
                    np.mean(sorted_rate <= peak_rate_at_top_steps)
                )

    abrupt_flag = bool(
        np.isfinite(max_step)
        and (
            max_step > 50.0
            or (
                np.isfinite(step_p99)
                and step_p99 > 0
                and max_step / step_p99 > 4.0
                and max_step > 20.0
            )
        )
    )
    boundary_flag = repeated_extreme_fraction > 0.03
    low_support_jump_flag = bool(
        np.isfinite(peak_rate_percentile_at_top_steps)
        and peak_rate_percentile_at_top_steps < 0.15
        and np.isfinite(max_step)
        and max_step > 20.0
    )

    profile = {
        "method": method,
        "method_dir": str(method_dir),
        "time_bins": int(time_bins.size),
        "depth_bins": int(depth_bins.size),
        "duration_s": float(np.nanmax(time_bins) - np.nanmin(time_bins) + dt),
        "time_bin_s": dt,
        "depth_min_um": float(np.nanmin(depth_bins)),
        "depth_max_um": float(np.nanmax(depth_bins)),
        "field_p01_um": float(field_p01),
        "field_p99_um": float(field_p99),
        "field_span_p01_p99_um": float(field_p99 - field_p01),
        "rigid_excursion_p95_p5_um": float(rigid_p95 - rigid_p05),
        "median_nonrigid_spread_um": float(np.nanmedian(spatial_spread)),
        "p95_nonrigid_spread_um": float(np.nanpercentile(spatial_spread, 95)),
        "median_abs_spatial_gradient_um": (
            float(np.nanmedian(np.abs(spatial_gradient))) if spatial_gradient.size else 0.0
        ),
        "median_abs_step_um": float(step_p50),
        "p99_abs_step_um": float(step_p99),
        "max_abs_step_um": max_step,
        "max_to_p99_step_ratio": (
            max_step / step_p99 if np.isfinite(step_p99) and step_p99 > 0 else np.nan
        ),
        "p99_rigid_speed_um_s": float(np.nanpercentile(speed, 99)) if speed.size else np.nan,
        "repeated_extreme_fraction": repeated_extreme_fraction,
        "speed_peak_rate_correlation": speed_peak_rate_correlation,
        "peak_rate_at_top_motion_steps": peak_rate_at_top_steps,
        "peak_rate_percentile_at_top_motion_steps": peak_rate_percentile_at_top_steps,
        "abrupt_estimate_flag": abrupt_flag,
        "boundary_or_clipping_flag": boundary_flag,
        "low_support_jump_flag": low_support_jump_flag,
    }
    trace = pd.DataFrame(
        {
            "method": method,
            "relative_time_s": relative_time,
            "rigid_displacement_um": rigid_centered,
            "nonrigid_spread_um": spatial_spread,
        }
    )
    trace.attrs["field"] = field
    trace.attrs["depth_bins"] = depth_bins
    return profile, trace


def cross_method_agreement(
    identity: dict[str, object], traces: dict[str, pd.DataFrame]
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    methods = sorted(traces)
    for left_index, left in enumerate(methods):
        for right in methods[left_index + 1 :]:
            left_trace = traces[left]
            right_trace = traces[right]
            start = max(
                float(left_trace.relative_time_s.min()),
                float(right_trace.relative_time_s.min()),
            )
            stop = min(
                float(left_trace.relative_time_s.max()),
                float(right_trace.relative_time_s.max()),
            )
            if stop <= start:
                continue
            points = max(30, min(len(left_trace), len(right_trace), 2000))
            common_time = np.linspace(start, stop, points)
            left_rigid = np.interp(
                common_time,
                left_trace.relative_time_s,
                left_trace.rigid_displacement_um,
            )
            right_rigid = np.interp(
                common_time,
                right_trace.relative_time_s,
                right_trace.rigid_displacement_um,
            )
            correlation = _pearson(left_rigid, right_rigid)
            slope = np.nan
            if np.nanstd(left_rigid) > 0:
                slope = float(np.polyfit(left_rigid, right_rigid, 1)[0])
            rows.append(
                {
                    **identity,
                    "left_method": left,
                    "right_method": right,
                    "common_points": points,
                    "rigid_correlation": correlation,
                    "absolute_rigid_correlation": abs(correlation) if np.isfinite(correlation) else np.nan,
                    "right_on_left_slope": slope,
                    "median_absolute_difference_um": float(np.nanmedian(np.abs(left_rigid - right_rigid))),
                }
            )
    return rows


def recover_explicit_parameters(motion_dir: Path) -> dict[str, object]:
    result: dict[str, object] = {
        "detect_threshold": np.nan,
        "detect_radius_um": np.nan,
        "filter_min_hz": np.nan,
        "filter_max_hz": np.nan,
        "parameter_provenance": "not_saved",
    }
    medicine_path = motion_dir / "medicine" / "medicine_parameters.json"
    if medicine_path.exists():
        try:
            parameters = json.loads(medicine_path.read_text())
            result.update(
                {
                    "medicine_time_bin_s": parameters.get("time_bin_size", np.nan),
                    "medicine_time_kernel_s": parameters.get("time_kernel_width", np.nan),
                    "medicine_depth_bins": parameters.get("num_depth_bins", np.nan),
                }
            )
        except (OSError, json.JSONDecodeError):
            pass
    return result


def classify_multimethod_run(method_rows: pd.DataFrame, agreement_rows: pd.DataFrame) -> str:
    if len(method_rows) < 2:
        return "single_method_unvalidated"
    dredge_dc = agreement_rows[
        agreement_rows.left_method.eq("decentralized")
        & agreement_rows.right_method.eq("dredge")
    ]
    if dredge_dc.empty:
        return "multimethod_without_dredge_dc_anchor"
    correlation = float(dredge_dc.absolute_rigid_correlation.iloc[0])
    kilosort = method_rows[method_rows.method.eq("kilosort_style")]
    kilosort_outlier = bool(
        not kilosort.empty
        and (
            bool(kilosort.abrupt_estimate_flag.iloc[0])
            or float(kilosort.max_to_p99_step_ratio.iloc[0]) > 4.0
        )
    )
    if correlation >= 0.65:
        return (
            "dredge_dc_consensus_with_kilosort_outliers"
            if kilosort_outlier
            else "dredge_dc_consensus"
        )
    if correlation < 0.35:
        return "dredge_dc_disagreement"
    return "partial_dredge_dc_agreement"


def build_survey(
    output_dir: Path = DEFAULT_OUTPUT,
    config: SurveyConfig = SurveyConfig(),
) -> dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    motion_dirs = discover_motion_dirs(config)

    run_rows: list[dict[str, object]] = []
    method_rows: list[dict[str, object]] = []
    agreement_rows: list[dict[str, object]] = []

    for motion_dir in motion_dirs:
        identity = _path_identity(motion_dir)
        peak_profile, peak_time_profile = profile_peaks(
            motion_dir,
            maximum_profile_peaks=config.maximum_profile_peaks,
            sampling_frequency_hz=config.sampling_frequency_hz,
        )
        parameters = recover_explicit_parameters(motion_dir)
        traces: dict[str, pd.DataFrame] = {}
        local_method_rows: list[dict[str, object]] = []
        for method, directory_name in METHOD_DIRS.items():
            profile, trace = profile_motion_method(
                method,
                motion_dir / directory_name,
                peak_time_profile,
            )
            if profile:
                row = {**identity, **peak_profile, **profile}
                method_rows.append(row)
                local_method_rows.append(row)
                traces[method] = trace
        local_agreement = cross_method_agreement(identity, traces)
        agreement_rows.extend(local_agreement)
        local_methods = pd.DataFrame(local_method_rows)
        local_agreements = pd.DataFrame(local_agreement)
        classification = classify_multimethod_run(local_methods, local_agreements)
        run_rows.append(
            {
                **identity,
                **peak_profile,
                **parameters,
                "method_count": len(traces),
                "methods": "+".join(sorted(traces)),
                "estimate_support_class": classification,
                "has_standard_plot_suite": (motion_dir / "motion_comparison.png").exists(),
            }
        )

    runs = pd.DataFrame(run_rows)
    methods = pd.DataFrame(method_rows)
    agreements = pd.DataFrame(agreement_rows)
    runs.to_csv(output_dir / "run_inventory.csv", index=False)
    methods.to_csv(output_dir / "method_metrics.csv", index=False)
    agreements.to_csv(output_dir / "cross_method_agreement.csv", index=False)

    return finalize_survey(runs, methods, agreements, output_dir, config)


def finalize_survey(
    runs: pd.DataFrame,
    methods: pd.DataFrame,
    agreements: pd.DataFrame,
    output_dir: Path = DEFAULT_OUTPUT,
    config: SurveyConfig = SurveyConfig(),
) -> dict[str, object]:
    """Render and summarize already-profiled survey tables."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    jump_contexts = audit_motion_jump_contexts(methods)
    jump_contexts.to_csv(output_dir / "motion_jump_peak_context.csv", index=False)

    multimethod = runs[runs.method_count >= 2].copy()
    figure_paths = make_figures(runs, methods, agreements, output_dir)

    manifest = {
        "config": asdict(config),
        "motion_directories": int(len(runs)),
        "motion_fields": int(len(methods)),
        "multimethod_directories": int(len(multimethod)),
        "directories_with_peaks": int(runs.has_peaks.sum()),
        "directories_with_locations": int(runs.has_peak_locations.sum()),
        "classification_counts": {
            str(key): int(value)
            for key, value in runs.estimate_support_class.value_counts().items()
        },
        "limitations": [
            "Historical detect thresholds, noise exclusions, and preprocessing settings were not consistently saved.",
            "Peak-support distribution metrics use a deterministic bounded sample; peak counts and durations are exact.",
            "Cross-method agreement is evidence of reproducibility, not ground-truth accuracy.",
            "This survey does not apply motion or use sorter yield to score estimator quality.",
        ],
        "outputs": [
            "run_inventory.csv",
            "method_metrics.csv",
            "cross_method_agreement.csv",
            "motion_jump_peak_context.csv",
            *[path.name for path in figure_paths],
        ],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def _window_slice_indices(
    sample_indices: np.ndarray,
    start_sample: int,
    stop_sample: int,
) -> tuple[int, int]:
    left = int(np.searchsorted(sample_indices, start_sample, side="left"))
    right = int(np.searchsorted(sample_indices, stop_sample, side="right"))
    return left, right


def _window_peak_metrics(
    peaks: np.ndarray,
    locations: np.ndarray | None,
    sample_indices: np.ndarray,
    start_sample: int,
    stop_sample: int,
    sampling_frequency_hz: float,
) -> dict[str, float]:
    left, right = _window_slice_indices(sample_indices, start_sample, stop_sample)
    duration_s = max((stop_sample - start_sample) / sampling_frequency_hz, 1e-9)
    window = np.asarray(peaks[left:right])
    result = {
        "rate_hz": float(len(window) / duration_s),
        "abs_amplitude_median": np.nan,
        "abs_amplitude_p99": np.nan,
        "simultaneous_channels_p99": np.nan,
        "simultaneous_channels_max": np.nan,
        "active_channel_fraction": np.nan,
        "depth_median_um": np.nan,
        "depth_iqr_um": np.nan,
    }
    if len(window) == 0:
        return result
    names = window.dtype.names or ()
    if "amplitude" in names:
        amplitude = np.abs(window["amplitude"].astype(float))
        result["abs_amplitude_median"] = float(np.nanmedian(amplitude))
        result["abs_amplitude_p99"] = float(np.nanpercentile(amplitude, 99))
    unique_samples, multiplicity = np.unique(window["sample_index"], return_counts=True)
    if unique_samples.size:
        result["simultaneous_channels_p99"] = float(np.nanpercentile(multiplicity, 99))
        result["simultaneous_channels_max"] = float(np.nanmax(multiplicity))
    if "channel_index" in names:
        active_channels = np.unique(window["channel_index"])
        result["active_channel_fraction"] = float(len(active_channels) / 384.0)
    if locations is not None:
        depth = np.asarray(locations[left:right]["y"], dtype=float)
        finite = depth[np.isfinite(depth)]
        if finite.size:
            result["depth_median_um"] = float(np.nanmedian(finite))
            result["depth_iqr_um"] = float(np.diff(np.nanpercentile(finite, [25, 75]))[0])
    return result


def audit_motion_jump_contexts(
    methods: pd.DataFrame,
    *,
    sampling_frequency_hz: float = SAMPLING_FREQUENCY_HZ,
    events_per_field: int = 3,
    separation_s: float = 30.0,
) -> pd.DataFrame:
    """Profile cached peaks around the largest saved displacement steps."""
    rows: list[dict[str, object]] = []
    for method_row in methods.to_dict(orient="records"):
        motion_dir = Path(str(method_row["motion_dir"]))
        method_dir = Path(str(method_row["method_dir"]))
        peak_path = motion_dir / "peaks.npy"
        if not peak_path.exists():
            continue
        motion = np.load(method_dir / "motion.npy")
        times = np.asarray(np.load(method_dir / "time_bins.npy"), dtype=float).reshape(-1)
        depths = np.asarray(np.load(method_dir / "depth_bins.npy"), dtype=float).reshape(-1)
        field = _normalize_motion_shape(motion, times, depths)
        if len(times) < 2:
            continue
        field_steps = np.diff(field, axis=0)
        step_magnitude = np.nanmax(np.abs(field_steps), axis=1)
        order = np.argsort(step_magnitude)[::-1]
        selected: list[int] = []
        for index in order:
            event_time = float(times[index + 1] - times[0])
            if all(abs(event_time - float(times[other + 1] - times[0])) >= separation_s for other in selected):
                selected.append(int(index))
            if len(selected) >= events_per_field:
                break

        peaks = np.load(peak_path, mmap_mode="r")
        if len(peaks) == 0 or not peaks.dtype.names or "sample_index" not in peaks.dtype.names:
            continue
        sample_indices = peaks["sample_index"]
        locations_path = motion_dir / "peak_locations.npy"
        locations = np.load(locations_path, mmap_mode="r") if locations_path.exists() else None
        first_sample = int(sample_indices[0])

        for rank, index in enumerate(selected, start=1):
            event_relative_s = float(times[index + 1] - times[0])
            center_sample = first_sample + int(round(event_relative_s * sampling_frequency_hz))
            windows = {
                "pre": (-15.0, -5.0),
                "event": (-2.0, 2.0),
                "post": (5.0, 15.0),
            }
            metrics: dict[str, dict[str, float]] = {}
            for label, (start_s, stop_s) in windows.items():
                metrics[label] = _window_peak_metrics(
                    peaks,
                    locations,
                    sample_indices,
                    center_sample + int(round(start_s * sampling_frequency_hz)),
                    center_sample + int(round(stop_s * sampling_frequency_hz)),
                    sampling_frequency_hz,
                )
            context_rate = np.nanmean([metrics["pre"]["rate_hz"], metrics["post"]["rate_hz"]])
            context_amp = np.nanmean(
                [metrics["pre"]["abs_amplitude_p99"], metrics["post"]["abs_amplitude_p99"]]
            )
            signed_step = field_steps[index]
            max_depth_index = int(np.nanargmax(np.abs(signed_step)))
            maximum_step = float(signed_step[max_depth_index])
            coherent_fraction = float(
                np.mean(
                    (np.sign(signed_step) == np.sign(maximum_step))
                    & (np.abs(signed_step) >= 0.5 * abs(maximum_step))
                )
            )
            rows.append(
                {
                    "motion_dir": str(motion_dir),
                    "pipeline_name": method_row["pipeline_name"],
                    "subject": method_row["subject"],
                    "session": method_row["session"],
                    "probe": method_row["probe"],
                    "method": method_row["method"],
                    "jump_rank": rank,
                    "event_relative_s": event_relative_s,
                    "event_fraction_of_duration": event_relative_s / max(float(times[-1] - times[0]), 1e-9),
                    "max_step_um": maximum_step,
                    "max_step_depth_um": float(depths[min(max_depth_index, len(depths) - 1)]),
                    "coherent_depth_fraction": coherent_fraction,
                    "pre_peak_rate_hz": metrics["pre"]["rate_hz"],
                    "event_peak_rate_hz": metrics["event"]["rate_hz"],
                    "post_peak_rate_hz": metrics["post"]["rate_hz"],
                    "event_to_context_peak_rate_ratio": (
                        metrics["event"]["rate_hz"] / context_rate if context_rate > 0 else np.nan
                    ),
                    "event_to_context_amp_p99_ratio": (
                        metrics["event"]["abs_amplitude_p99"] / context_amp if context_amp > 0 else np.nan
                    ),
                    "event_simultaneous_channels_p99": metrics["event"]["simultaneous_channels_p99"],
                    "event_simultaneous_channels_max": metrics["event"]["simultaneous_channels_max"],
                    "event_active_channel_fraction": metrics["event"]["active_channel_fraction"],
                    "event_depth_iqr_um": metrics["event"]["depth_iqr_um"],
                    "context_depth_iqr_um": np.nanmean(
                        [metrics["pre"]["depth_iqr_um"], metrics["post"]["depth_iqr_um"]]
                    ),
                }
            )
    return pd.DataFrame(rows)


def finalize_saved_survey(
    output_dir: Path = DEFAULT_OUTPUT,
    config: SurveyConfig = SurveyConfig(),
) -> dict[str, object]:
    """Finalize figures and manifest from existing profile tables."""
    output_dir = Path(output_dir)
    runs = pd.read_csv(output_dir / "run_inventory.csv")
    methods = pd.read_csv(output_dir / "method_metrics.csv")
    agreements = pd.read_csv(output_dir / "cross_method_agreement.csv")
    return finalize_survey(runs, methods, agreements, output_dir, config)


def make_figures(
    runs: pd.DataFrame,
    methods: pd.DataFrame,
    agreements: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    paths: list[Path] = []
    if methods.empty:
        return paths

    colors = {
        "kilosort_style": "#3569a8",
        "decentralized": "#d08b27",
        "dredge": "#506b3f",
        "medicine": "#b85c7a",
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for method, group in methods.groupby("method"):
        axes[0].scatter(
            group.peak_rate_hz,
            group.p99_abs_step_um,
            label=method,
            color=colors.get(method, "#555555"),
            alpha=0.75,
            s=35,
        )
        axes[1].scatter(
            group.peak_rate_p99_over_median_10s,
            group.max_to_p99_step_ratio,
            label=method,
            color=colors.get(method, "#555555"),
            alpha=0.75,
            s=35,
        )
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Detected peaks per second")
    axes[0].set_ylabel("99th percentile field step (µm)")
    axes[0].set_title("Peak volume and estimator step size")
    axes[1].set_xlabel("Peak-rate burst ratio (99th / median, 10 s)")
    axes[1].set_ylabel("Largest step / 99th percentile step")
    axes[1].set_title("Input bursts and isolated estimator jumps")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].axhline(4.0, color="#333333", linestyle="--", linewidth=1)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    path = output_dir / "peak_support_vs_estimator_stability.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    if not agreements.empty:
        selected = agreements[
            agreements.left_method.ne("kilosort_style")
            & agreements.right_method.ne("kilosort_style")
        ].copy()
        selected["pair"] = selected.left_method + " vs " + selected.right_method
        order = sorted(selected.pair.unique())
        fig, ax = plt.subplots(figsize=(10, 5))
        data = [
            selected.loc[selected.pair.eq(pair), "absolute_rigid_correlation"].dropna()
            for pair in order
        ]
        ax.boxplot(data, labels=order, showfliers=True)
        ax.axhline(0.65, color="#333333", linestyle="--", linewidth=1)
        ax.set_ylim(-0.03, 1.03)
        ax.set_ylabel("Absolute rigid-trace correlation")
        ax.set_title("Cross-method agreement in matched historical runs")
        ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        path = output_dir / "cross_method_agreement_distribution.png"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)

    fig, ax = plt.subplots(figsize=(10, 5))
    counts = (
        methods.groupby(["method", "abrupt_estimate_flag"])
        .size()
        .unstack(fill_value=0)
        .rename(columns={False: "not flagged", True: "abruptness flagged"})
    )
    counts.plot(
        kind="bar",
        stacked=True,
        color=["#b9c4cc", "#d08b27"],
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Saved motion fields")
    ax.set_title("Abrupt-estimate flags by algorithm")
    ax.legend(frameon=False)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    path = output_dir / "abrupt_estimate_flags_by_method.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)
    return paths


if __name__ == "__main__":
    print(json.dumps(build_survey(), indent=2))
