"""Directly calibrate Luke DREDGE displacement against peak-raster shifts.

This discovery-only diagnostic does not apply a motion field and does not use
sorted unit labels.  It constructs amplitude-by-depth fingerprints from the
pre-registration localized peaks, independently matches non-overlapping time
blocks by explicit spatial cross-correlation, and compares the observed shift
in physical micrometers with the saved DREDGE rigid displacement difference.

Because DREDGE used the same peak source, this is an implementation-independent
remeasurement rather than a fully independent biological motion sensor.  The
deterministic peak split halves and raster sensitivity grid expose unstable
matches instead of treating the fitted field as truth.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from testing.luke_rigid025_depth_strip import relative_motion_bins


MOTION_ROOT = Path(
    "/mnt/NPX/Luke/20250804/"
    "dredge_pipeline_results_Luke0804_V2V1_g0_imec1/motion"
)
OUTPUT = Path("testing/outputs/luke_direct_motion_scale_audit")
FS = 30_000.0


@dataclass(frozen=True)
class RasterSpec:
    name: str
    depth_bin_um: float
    smooth_um: float
    amplitude_resolved: bool


SPECS = (
    RasterSpec("amp_depth_dz2_smooth6", 2.0, 6.0, True),
    RasterSpec("amp_depth_dz2_smooth10", 2.0, 10.0, True),
    RasterSpec("amp_depth_dz4_smooth10", 4.0, 10.0, True),
    RasterSpec("amp_depth_dz4_smooth20", 4.0, 20.0, True),
    RasterSpec("depth_only_dz2_smooth10", 2.0, 10.0, False),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion-root", type=Path, default=MOTION_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--time-bin-s", type=float, default=10.0)
    parser.add_argument("--pair-separation-s", type=float, default=60.0)
    parser.add_argument("--pair-stride-s", type=float, default=120.0)
    parser.add_argument("--minimum-predicted-shift-um", type=float, default=6.0)
    parser.add_argument("--maximum-shift-um", type=float, default=60.0)
    parser.add_argument("--plan-only", action="store_true")
    return parser.parse_args()


def deterministic_half(sample_index: np.ndarray, channel_index: np.ndarray) -> np.ndarray:
    """Stable pseudo-random half assignment without loading a RNG state."""
    sample = np.asarray(sample_index, dtype=np.uint64)
    channel = np.asarray(channel_index, dtype=np.uint64)
    mixed = sample * np.uint64(11400714819323198485) + channel * np.uint64(0x9E3779B1)
    return (mixed >> np.uint64(63)).astype(np.int8)


def amplitude_edges(amplitudes: np.ndarray) -> np.ndarray:
    values = np.abs(np.asarray(amplitudes, dtype=float))
    edges = np.unique(np.quantile(values[np.isfinite(values)], [0, 0.25, 0.5, 0.7, 0.85, 0.95, 1]))
    if len(edges) < 3:
        raise ValueError("Amplitude distribution has too few distinct values")
    edges[-1] = np.nextafter(edges[-1], np.inf)
    return edges


def build_base_rasters(
    peaks: np.ndarray,
    locations: np.ndarray,
    *,
    time_bin_s: float,
    duration_s: float,
    depth_bin_um: float,
    depth_stop_um: float,
    amplitude_bin_edges: np.ndarray,
    chunk_size: int = 2_000_000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Accumulate full and deterministic split-half amplitude-depth rasters."""
    n_time = int(np.ceil(duration_s / time_bin_s))
    n_depth = int(np.ceil(depth_stop_um / depth_bin_um))
    n_amp = len(amplitude_bin_edges) - 1
    size = n_time * n_amp * n_depth
    rasters = [np.zeros(size, dtype=np.float32) for _ in range(3)]
    for start in range(0, len(peaks), chunk_size):
        stop = min(start + chunk_size, len(peaks))
        peak = peaks[start:stop]
        depth = np.asarray(locations["y"][start:stop], dtype=float)
        time_index = np.asarray(
            peak["sample_index"] // int(round(FS * time_bin_s)), dtype=np.int64
        )
        depth_index = np.floor(depth / depth_bin_um).astype(np.int64)
        amplitude_index = np.searchsorted(
            amplitude_bin_edges, np.abs(peak["amplitude"]), side="right"
        ) - 1
        valid = (
            (time_index >= 0)
            & (time_index < n_time)
            & (depth_index >= 0)
            & (depth_index < n_depth)
            & (amplitude_index >= 0)
            & (amplitude_index < n_amp)
        )
        flat = (
            (time_index[valid] * n_amp + amplitude_index[valid]) * n_depth
            + depth_index[valid]
        )
        rasters[0] += np.bincount(flat, minlength=size).astype(np.float32)
        halves = deterministic_half(
            peak["sample_index"][valid], peak["channel_index"][valid]
        )
        for half in (0, 1):
            rasters[half + 1] += np.bincount(
                flat[halves == half], minlength=size
            ).astype(np.float32)
    return tuple(value.reshape(n_time, n_amp, n_depth) for value in rasters)


def prepare_raster(base: np.ndarray, spec: RasterSpec, base_depth_bin_um: float) -> np.ndarray:
    factor = int(round(spec.depth_bin_um / base_depth_bin_um))
    if factor < 1 or not np.isclose(factor * base_depth_bin_um, spec.depth_bin_um):
        raise ValueError("Raster depth-bin ratio must be a positive integer")
    values = np.asarray(base, dtype=np.float32)
    if factor > 1:
        usable = values.shape[2] // factor * factor
        values = values[:, :, :usable].reshape(
            values.shape[0], values.shape[1], usable // factor, factor
        ).sum(axis=3)
    if not spec.amplitude_resolved:
        values = values.sum(axis=1, keepdims=True)
    values = np.log1p(values)
    values = gaussian_filter1d(
        values, spec.smooth_um / spec.depth_bin_um, axis=2, mode="nearest"
    )
    values -= values.mean(axis=2, keepdims=True)
    values /= values.std(axis=2, keepdims=True) + np.float32(1e-6)
    return values


def shift_score(
    first: np.ndarray,
    second: np.ndarray,
    shift_um: float,
    depth_bin_um: float,
    depth_margin_um: float = 200.0,
) -> float:
    depth = np.arange(first.shape[1], dtype=float) * depth_bin_um
    shifted = np.vstack(
        [np.interp(depth, depth + shift_um, row, left=np.nan, right=np.nan) for row in second]
    )
    valid_depth = (depth >= depth_margin_um) & (depth <= depth[-1] - depth_margin_um)
    valid = np.isfinite(shifted) & valid_depth[None, :]
    return float(np.mean(first[valid] * shifted[valid]))


def estimate_pair_shift(
    first: np.ndarray,
    second: np.ndarray,
    *,
    depth_bin_um: float,
    maximum_shift_um: float,
    step_um: float = 0.5,
) -> dict[str, float]:
    """Return the depth translation from first to second in micrometers."""
    shifts = np.arange(-maximum_shift_um, maximum_shift_um + step_um / 2, step_um)
    scores = np.asarray(
        [shift_score(first, second, value, depth_bin_um) for value in shifts]
    )
    best = int(np.nanargmax(scores))
    observed = -float(shifts[best])
    if 0 < best < len(scores) - 1:
        left, center, right = scores[best - 1 : best + 2]
        denominator = left - 2 * center + right
        if denominator != 0:
            observed -= float(0.5 * (left - right) / denominator * step_um)
    zero_score = float(scores[np.argmin(np.abs(shifts))])
    competitors = scores[np.abs(shifts - shifts[best]) >= 5.0]
    runner_up = float(np.nanmax(competitors)) if len(competitors) else np.nan
    return {
        "observed_shift_um": observed,
        "peak_score": float(scores[best]),
        "score_gain_vs_zero": float(scores[best] - zero_score),
        "score_margin_vs_distant_peak": float(scores[best] - runner_up),
        "hit_search_boundary": bool(best in (0, len(scores) - 1)),
    }


def robust_line(predicted: np.ndarray, observed: np.ndarray) -> dict[str, float]:
    x = np.asarray(predicted, dtype=float)
    y = np.asarray(observed, dtype=float)
    if len(x) < 3:
        return {"slope": np.nan, "intercept_um": np.nan, "correlation": np.nan}
    scale = max(float(np.median(np.abs(y - np.median(y)))), 1.0)
    fit = least_squares(
        lambda beta: (y - (beta[0] + beta[1] * x)) / scale,
        x0=np.asarray([0.0, 1.0]),
        loss="soft_l1",
    )
    return {
        "slope": float(fit.x[1]),
        "intercept_um": float(fit.x[0]),
        "correlation": float(np.corrcoef(x, y)[0, 1]),
    }


def bootstrap_slope(
    predicted: np.ndarray, observed: np.ndarray, *, seed: int = 20250804, repeats: int = 1000
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    slopes = []
    for _ in range(repeats):
        take = rng.integers(0, len(predicted), len(predicted))
        slopes.append(robust_line(predicted[take], observed[take])["slope"])
    return float(np.quantile(slopes, 0.025)), float(np.quantile(slopes, 0.975))


def run(args: argparse.Namespace) -> dict:
    peaks_path = args.motion_root / "peaks.npy"
    locations_path = args.motion_root / "peak_locations.npy"
    field_dir = args.motion_root / "dredge-motion"
    peaks = np.load(peaks_path, mmap_mode="r")
    locations = np.load(locations_path, mmap_mode="r")
    displacement = np.load(field_dir / "motion.npy")
    absolute_times = np.load(field_dir / "time_bins.npy")
    relative_times, acquisition_start_s = relative_motion_bins(absolute_times)
    duration_s = max(
        float(peaks["sample_index"][-1]) / FS,
        float(relative_times[-1] + np.median(np.diff(relative_times)) / 2),
    )
    base_depth_bin_um = min(value.depth_bin_um for value in SPECS)
    sample_amplitude = peaks["amplitude"][:: max(1, len(peaks) // 1_000_000)]
    amp_edges = amplitude_edges(sample_amplitude)
    plan = {
        "peaks": int(len(peaks)),
        "duration_s": duration_s,
        "time_bin_s": args.time_bin_s,
        "pair_separation_s": args.pair_separation_s,
        "pair_stride_s": args.pair_stride_s,
        "minimum_predicted_shift_um": args.minimum_predicted_shift_um,
        "raster_specs": [asdict(value) for value in SPECS],
        "amplitude_edges_uv": amp_edges.tolist(),
        "prospective_holdout_labels_accessed": False,
        "sorter_labels_accessed": False,
    }
    if args.plan_only:
        print(json.dumps(plan, indent=2))
        return plan
    base = build_base_rasters(
        peaks,
        locations,
        time_bin_s=args.time_bin_s,
        duration_s=duration_s,
        depth_bin_um=base_depth_bin_um,
        depth_stop_um=3840.0,
        amplitude_bin_edges=amp_edges,
    )
    rigid = np.nanmedian(displacement, axis=1)
    centers = np.arange(base[0].shape[0]) * args.time_bin_s + args.time_bin_s / 2
    predicted_at_centers = np.interp(centers, relative_times, rigid)
    lag = int(round(args.pair_separation_s / args.time_bin_s))
    stride = int(round(args.pair_stride_s / args.time_bin_s))
    pair_starts = np.arange(0, len(centers) - lag, stride, dtype=int)
    predicted_delta = predicted_at_centers[pair_starts + lag] - predicted_at_centers[pair_starts]
    pair_starts = pair_starts[np.abs(predicted_delta) >= args.minimum_predicted_shift_um]
    rows = []
    for spec in SPECS:
        prepared = [prepare_raster(value, spec, base_depth_bin_um) for value in base]
        for first_index in pair_starts:
            second_index = first_index + lag
            measurements = [
                estimate_pair_shift(
                    value[first_index],
                    value[second_index],
                    depth_bin_um=spec.depth_bin_um,
                    maximum_shift_um=args.maximum_shift_um,
                )
                for value in prepared
            ]
            rows.append(
                {
                    "raster_spec": spec.name,
                    "first_time_s": centers[first_index],
                    "second_time_s": centers[second_index],
                    "predicted_dredge_delta_um": float(
                        predicted_at_centers[second_index] - predicted_at_centers[first_index]
                    ),
                    **measurements[0],
                    "half_a_shift_um": measurements[1]["observed_shift_um"],
                    "half_b_shift_um": measurements[2]["observed_shift_um"],
                    "split_half_difference_um": abs(
                        measurements[1]["observed_shift_um"]
                        - measurements[2]["observed_shift_um"]
                    ),
                    "split_half_same_direction": bool(
                        np.sign(measurements[1]["observed_shift_um"])
                        == np.sign(measurements[2]["observed_shift_um"])
                    ),
                }
            )
    pairs = pd.DataFrame(rows)
    pairs["qualified"] = (
        (~pairs.hit_search_boundary)
        & (pairs.peak_score >= 0.50)
        & (pairs.score_margin_vs_distant_peak >= 0.002)
        & (pairs.split_half_difference_um <= 10.0)
        & pairs.split_half_same_direction
    )
    summary_rows = []
    for spec, values in pairs.groupby("raster_spec", sort=False):
        selected = values[values.qualified]
        fit = robust_line(
            selected.predicted_dredge_delta_um.to_numpy(),
            selected.observed_shift_um.to_numpy(),
        )
        low, high = (
            bootstrap_slope(
                selected.predicted_dredge_delta_um.to_numpy(),
                selected.observed_shift_um.to_numpy(),
            )
            if len(selected) >= 3
            else (np.nan, np.nan)
        )
        summary_rows.append(
            {
                "raster_spec": spec,
                "candidate_pairs": int(len(values)),
                "qualified_pairs": int(len(selected)),
                **fit,
                "slope_ci95_low": low,
                "slope_ci95_high": high,
                "median_absolute_error_um": float(
                    np.median(
                        np.abs(
                            selected.observed_shift_um
                            - selected.predicted_dredge_delta_um
                        )
                    )
                )
                if len(selected)
                else np.nan,
                "direction_agreement_fraction": float(
                    np.mean(
                        np.sign(selected.observed_shift_um)
                        == np.sign(selected.predicted_dredge_delta_um)
                    )
                )
                if len(selected)
                else np.nan,
                "median_observed_to_dredge_ratio": float(
                    np.median(
                        selected.observed_shift_um
                        / selected.predicted_dredge_delta_um
                    )
                )
                if len(selected)
                else np.nan,
                "fraction_observed_below_2um": float(
                    np.mean(np.abs(selected.observed_shift_um) < 2.0)
                )
                if len(selected)
                else np.nan,
            }
        )
    summary = pd.DataFrame(summary_rows)
    gain_rows = []
    for spec, values in pairs[pairs.qualified].groupby("raster_spec", sort=False):
        for gain in (0.0, 0.25, 0.5, 0.75, 1.0):
            errors = np.abs(
                values.observed_shift_um
                - gain * values.predicted_dredge_delta_um
            )
            gain_rows.append(
                {
                    "raster_spec": spec,
                    "gain": gain,
                    "qualified_pairs": int(len(values)),
                    "median_absolute_error_um": float(np.median(errors)),
                    "mean_absolute_error_um": float(np.mean(errors)),
                }
            )
    gain_scores = pd.DataFrame(gain_rows)
    primary = summary[summary.raster_spec == SPECS[0].name].iloc[0]
    primary_gains = gain_scores[gain_scores.raster_spec == SPECS[0].name]
    best_primary_gain = float(
        primary_gains.sort_values(
            ["median_absolute_error_um", "mean_absolute_error_um"]
        ).iloc[0].gain
    )
    result = {
        **plan,
        "inferred_acquisition_start_s": acquisition_start_s,
        "candidate_pairs_per_spec": int(len(pair_starts)),
        "primary_spec": SPECS[0].name,
        "primary_qualified_pairs": int(primary.qualified_pairs),
        "primary_observed_per_dredge_slope": float(primary.slope),
        "primary_slope_ci95": [float(primary.slope_ci95_low), float(primary.slope_ci95_high)],
        "primary_correlation": float(primary.correlation),
        "primary_best_discrete_gain": best_primary_gain,
        "interpretation_guardrail": (
            "Shared pre-registration peaks make this a direct scale remeasurement, "
            "not an independent biological ground truth."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(args.output_dir / "pair_measurements.csv", index=False)
    summary.to_csv(args.output_dir / "scale_summary.csv", index=False)
    gain_scores.to_csv(args.output_dir / "gain_scores.csv", index=False)
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
