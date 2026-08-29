"""Characterize reproducible motion scales in Luke's pathological window.

This first-pass study uses every surviving full-session motion estimate on the
two simultaneously recorded probes.  It puts all methods on a common time and
depth grid, separates rigid, linear-depth, and residual spatial components,
and measures temporal bandwidth plus cross-method/cross-probe agreement.

It is descriptive rather than a production estimator sweep: the saved methods
used different native bins and parameters.  Its purpose is to bound a focused,
cache-safe re-estimation grid without treating any one fitted field as truth.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import detrend, welch


LUKE_ROOT = Path("/mnt/NPX/Luke/20250804")
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "outputs/luke_motion_scale_characterization"


@dataclass(frozen=True)
class Window:
    name: str
    start_s: float
    duration_s: float


WINDOW = Window("registration_outlier", 8160.0, 120.0)
METHOD_DIRS = {
    "iterative_template": "ks-motion",
    "decentralized": "decentralized-motion",
    "dredge": "dredge-motion",
    "medicine": "medicine",
}
PROBES = ("imec0", "imec1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--common-dt-s", type=float, default=2.0)
    parser.add_argument("--depth-start-um", type=float, default=300.0)
    parser.add_argument("--depth-stop-um", type=float, default=3500.0)
    parser.add_argument("--depth-step-um", type=float, default=200.0)
    return parser.parse_args()


def pipeline_root(probe: str) -> Path:
    return LUKE_ROOT / f"dredge_pipeline_results_Luke0804_V2V1_g0_{probe}"


def load_motion_arrays(probe: str, method: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    root = pipeline_root(probe) / "motion" / METHOD_DIRS[method]
    return (
        np.load(root / "motion.npy"),
        np.load(root / "time_bins.npy"),
        np.load(root / "depth_bins.npy"),
    )


def recording_t_start(probe: str) -> float:
    # All motion bins are absolute time.  Their first center is half a native
    # bin after the extractor start for the saved SI estimators.
    _, times, _ = load_motion_arrays(probe, "dredge")
    return float(times[0] - 0.5 * np.median(np.diff(times)))


def interpolate_field(
    displacement: np.ndarray,
    times_s: np.ndarray,
    depths_um: np.ndarray,
    target_times_s: np.ndarray,
    target_depths_um: np.ndarray,
) -> np.ndarray:
    """Linearly sample a displacement field without extrapolating its domain."""
    displacement = np.asarray(displacement, dtype=float)
    times_s = np.asarray(times_s, dtype=float)
    depths_um = np.asarray(depths_um, dtype=float)
    if displacement.shape != (len(times_s), len(depths_um)):
        raise ValueError("Displacement shape does not match time/depth axes")
    if target_times_s.min() < times_s.min() or target_times_s.max() > times_s.max():
        raise ValueError("Target times exceed motion support")
    if target_depths_um.min() < depths_um.min() or target_depths_um.max() > depths_um.max():
        raise ValueError("Target depths exceed motion support")
    depth_sampled = np.vstack(
        [np.interp(target_depths_um, depths_um, row) for row in displacement]
    )
    return np.column_stack(
        [np.interp(target_times_s, times_s, depth_sampled[:, index]) for index in range(len(target_depths_um))]
    )


def decompose_spatial_field(field: np.ndarray, depths_um: np.ndarray) -> dict[str, np.ndarray | float]:
    """Orthogonally split dynamic displacement into rigid, linear, and residual parts."""
    field = np.asarray(field, dtype=float)
    dynamic = field - np.mean(field, axis=0, keepdims=True)
    z = np.asarray(depths_um, dtype=float)
    z = (z - np.mean(z)) / max(np.std(z), np.finfo(float).eps)
    design = np.column_stack((np.ones(len(z)), z))
    coefficients = np.linalg.lstsq(design, dynamic.T, rcond=None)[0].T
    rigid = coefficients[:, 0]
    linear = coefficients[:, 1, None] * z[None, :]
    fitted = rigid[:, None] + linear
    residual = dynamic - fitted
    total_energy = float(np.sum(dynamic**2))
    if total_energy == 0:
        fractions = (0.0, 0.0, 0.0)
    else:
        fractions = (
            float(np.sum(rigid[:, None] ** 2) * field.shape[1] / total_energy),
            float(np.sum(linear**2) / total_energy),
            float(np.sum(residual**2) / total_energy),
        )
    return {
        "dynamic": dynamic,
        "rigid": rigid,
        "linear": linear,
        "residual": residual,
        "rigid_energy_fraction": fractions[0],
        "linear_depth_energy_fraction": fractions[1],
        "residual_nonrigid_energy_fraction": fractions[2],
    }


def temporal_bandwidth_metrics(trace: np.ndarray, dt_s: float) -> dict[str, float]:
    trace = detrend(np.asarray(trace, dtype=float), type="linear")
    if not np.any(trace):
        return {"f50_hz": 0.0, "f90_hz": 0.0, "half_power_period_s": np.inf, "p90_power_period_s": np.inf}
    frequencies, power = welch(trace, fs=1.0 / dt_s, nperseg=min(64, len(trace)))
    positive = frequencies > 0
    frequencies, power = frequencies[positive], power[positive]
    cumulative = np.cumsum(power)
    cumulative /= cumulative[-1]
    f50 = float(frequencies[np.searchsorted(cumulative, 0.5)])
    f90 = float(frequencies[np.searchsorted(cumulative, 0.9)])
    return {
        "f50_hz": f50,
        "f90_hz": f90,
        "half_power_period_s": 1.0 / f50,
        "p90_power_period_s": 1.0 / f90,
    }


def spatial_correlation_length_um(residual: np.ndarray, depths_um: np.ndarray, threshold: float = 0.5) -> float:
    residual = np.asarray(residual, dtype=float)
    if residual.shape[1] < 2:
        return float("nan")
    correlations = np.corrcoef(residual.T)
    separations = np.abs(np.subtract.outer(depths_um, depths_um))
    distances = np.unique(separations[separations > 0])
    for distance in distances:
        values = correlations[np.isclose(separations, distance)]
        values = values[np.isfinite(values)]
        if values.size and np.median(values) < threshold:
            return float(distance)
    return float(np.max(distances))


def summarize_field(field: np.ndarray, depths_um: np.ndarray, dt_s: float) -> dict[str, float]:
    decomposition = decompose_spatial_field(field, depths_um)
    rigid = np.asarray(decomposition["rigid"])
    spread = np.percentile(field, 95, axis=1) - np.percentile(field, 5, axis=1)
    steps = np.abs(np.diff(rigid))
    metrics = {
        "rigid_excursion_p95_p5_um": float(np.percentile(rigid, 95) - np.percentile(rigid, 5)),
        "p99_rigid_speed_um_per_s": float(np.percentile(steps / dt_s, 99)),
        "median_nonrigid_spread_um": float(np.median(spread)),
        "p95_nonrigid_spread_um": float(np.percentile(spread, 95)),
        "max_nonrigid_spread_um": float(np.max(spread)),
        "rigid_energy_fraction": float(decomposition["rigid_energy_fraction"]),
        "linear_depth_energy_fraction": float(decomposition["linear_depth_energy_fraction"]),
        "residual_nonrigid_energy_fraction": float(decomposition["residual_nonrigid_energy_fraction"]),
        "residual_spatial_corr_length_um": spatial_correlation_length_um(
            np.asarray(decomposition["residual"]), depths_um
        ),
    }
    metrics.update(temporal_bandwidth_metrics(rigid, dt_s))
    return metrics


def correlation(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=float).ravel()
    second = np.asarray(second, dtype=float).ravel()
    valid = np.isfinite(first) & np.isfinite(second)
    if valid.sum() < 3 or np.std(first[valid]) == 0 or np.std(second[valid]) == 0:
        return float("nan")
    return float(np.corrcoef(first[valid], second[valid])[0, 1])


def best_lag_correlation(first: np.ndarray, second: np.ndarray, max_lag_bins: int) -> tuple[int, float]:
    candidates = []
    for lag in range(-max_lag_bins, max_lag_bins + 1):
        if lag < 0:
            value = correlation(first[-lag:], second[:lag])
        elif lag > 0:
            value = correlation(first[:-lag], second[lag:])
        else:
            value = correlation(first, second)
        candidates.append((lag, value))
    finite = [(lag, value) for lag, value in candidates if np.isfinite(value)]
    return max(finite, key=lambda item: item[1]) if finite else (0, float("nan"))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    relative_times = np.arange(
        WINDOW.start_s + args.common_dt_s / 2,
        WINDOW.start_s + WINDOW.duration_s,
        args.common_dt_s,
    )
    depths = np.arange(args.depth_start_um, args.depth_stop_um + 0.1, args.depth_step_um)
    fields: dict[tuple[str, str], np.ndarray] = {}
    decompositions = {}
    rows = []
    for probe in PROBES:
        t_start = recording_t_start(probe)
        absolute_times = relative_times + t_start
        for method in METHOD_DIRS:
            displacement, times, native_depths = load_motion_arrays(probe, method)
            field = interpolate_field(displacement, times, native_depths, absolute_times, depths)
            fields[(probe, method)] = field
            decompositions[(probe, method)] = decompose_spatial_field(field, depths)
            rows.append(
                {
                    "probe": probe,
                    "method": method,
                    "native_dt_s": float(np.median(np.diff(times))),
                    "native_depth_step_um": float(np.median(np.diff(native_depths))),
                    **summarize_field(field, depths, args.common_dt_s),
                }
            )
    pd.DataFrame(rows).to_csv(args.output_dir / "field_scale_summary.csv", index=False)

    agreement_rows = []
    methods = list(METHOD_DIRS)
    for probe in PROBES:
        for left_index, left in enumerate(methods):
            for right in methods[left_index + 1 :]:
                first = decompositions[(probe, left)]
                second = decompositions[(probe, right)]
                agreement_rows.append(
                    {
                        "scope": "within_probe_cross_method",
                        "probe": probe,
                        "left_method": left,
                        "right_method": right,
                        "rigid_correlation": correlation(first["rigid"], second["rigid"]),
                        "nonrigid_field_correlation": correlation(first["residual"], second["residual"]),
                    }
                )
    for method in methods:
        first = decompositions[("imec0", method)]
        second = decompositions[("imec1", method)]
        lag, value = best_lag_correlation(first["rigid"], second["rigid"], max_lag_bins=5)
        agreement_rows.append(
            {
                "scope": "cross_probe_same_method",
                "probe": "imec0_vs_imec1",
                "left_method": method,
                "right_method": method,
                "rigid_correlation": correlation(first["rigid"], second["rigid"]),
                "nonrigid_field_correlation": correlation(first["residual"], second["residual"]),
                "best_lag_s": lag * args.common_dt_s,
                "best_lag_rigid_correlation": value,
            }
        )
    pd.DataFrame(agreement_rows).to_csv(args.output_dir / "estimator_agreement.csv", index=False)

    manifest = {
        "window": asdict(WINDOW),
        "common_dt_s": args.common_dt_s,
        "common_depths_um": depths.tolist(),
        "methods": METHOD_DIRS,
        "probes": list(PROBES),
        "time_basis": "frame-relative window converted to each extractor's absolute t_start",
        "interpretation": (
            "Saved estimates used different native parameters. Agreement bounds supported scales; "
            "it does not select a production estimator."
        ),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote motion-scale characterization to {args.output_dir}")


if __name__ == "__main__":
    main()
