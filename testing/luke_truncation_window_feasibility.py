"""Assess smaller amplitude-fit windows without candidate data.

This is calibration and baseline feasibility only. It does not alter production
QC, inspect the Option A candidate arm, or rerun a sorter.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.truncation import fit_amp_cdf, is_saturated
from testing.luke_c2_v4_truncation_diagnostic import TRUE_MISSING_PCTS, _truncated_sample


SCHEMA = "luke-truncation-window-feasibility-v1"
WINDOW_SIZES = (250, 500, 1000)
SYNTHETIC_REPLICATES = 200
SYNTHETIC_SEED = 20260906
MIN_FINITE_FITS = 2
MAX_ISI_S = 10.0
EFFECT_PP = 5.0
FS_HZ = 29999.835983263598
INTERVALS = {
    "nominated": (7200.0, 7320.0),
    "H1": (3120.0, 3240.0),
    "H2": (5640.0, 5760.0),
    "H3": (9600.0, 9720.0),
}
DEPTH_BAND_UM = (1810.0, 3710.0)


def exact_windows(n_spikes: int, size: int, origin: int = 0) -> list[tuple[int, int]]:
    """Return non-overlapping exact-count half-open windows."""
    if size < 1 or origin < 0:
        raise ValueError("size and origin must be nonnegative/positive")
    available = max(0, int(n_spikes) - int(origin))
    count = available // int(size)
    return [
        (int(origin) + index * int(size), int(origin) + (index + 1) * int(size))
        for index in range(count)
    ]


def fit_exact_window(amplitudes: np.ndarray) -> dict[str, float | bool]:
    values = np.asarray(amplitudes, dtype=np.float64)
    if values.size == 0:
        raise ValueError("cannot fit an empty window")
    popt, estimate = fit_amp_cdf(values)
    fallback = bool(np.allclose(popt, [float(np.mean(values)), 1.0, 1.0]))
    return {
        "estimate_pct": float(estimate),
        "saturated": bool(is_saturated([estimate])[0]),
        "fallback": fallback,
    }


def _synthetic_rows(window_size: int, rng: np.random.Generator) -> list[dict]:
    rows = []
    for truth in TRUE_MISSING_PCTS:
        estimates = []
        statuses = []
        for _ in range(SYNTHETIC_REPLICATES):
            try:
                result = fit_exact_window(_truncated_sample(rng, truth, window_size))
                estimates.append(result["estimate_pct"])
                statuses.append(result)
            except Exception:
                statuses.append({"fallback": False, "saturated": False, "failed": True})
        values = np.asarray(estimates, dtype=float)
        finite = values[np.isfinite(values)]
        rows.append({
            "window_size": window_size,
            "true_missing_pct": truth,
            "replicates": SYNTHETIC_REPLICATES,
            "bias_pp": float(np.mean(finite) - truth) if finite.size else np.nan,
            "absolute_error_pp": float(np.mean(np.abs(finite - truth))) if finite.size else np.nan,
            "spread_sd_pp": float(np.std(finite, ddof=1)) if finite.size > 1 else np.nan,
            "p05_estimate_pct": float(np.percentile(finite, 5)) if finite.size else np.nan,
            "p95_estimate_pct": float(np.percentile(finite, 95)) if finite.size else np.nan,
            "fit_failure_fraction": float(sum(bool(row.get("failed")) for row in statuses) / SYNTHETIC_REPLICATES),
            "fallback_fraction": float(sum(bool(row.get("fallback")) for row in statuses) / SYNTHETIC_REPLICATES),
            "boundary_pinned_fraction": float(sum(bool(row.get("saturated")) for row in statuses) / SYNTHETIC_REPLICATES),
        })
    # Independent paired samples answer false-positive and true-effect questions.
    for scenario, baseline_truth, candidate_truth in (
        ("unchanged", 10.0, 10.0),
        ("genuine_5pp", 10.0, 5.0),
    ):
        baseline, candidate = [], []
        for _ in range(SYNTHETIC_REPLICATES):
            baseline.append(fit_exact_window(_truncated_sample(rng, baseline_truth, window_size))["estimate_pct"])
            candidate.append(fit_exact_window(_truncated_sample(rng, candidate_truth, window_size))["estimate_pct"])
        change = np.asarray(baseline) - np.asarray(candidate)
        rows.append({
            "window_size": window_size,
            "comparison": scenario,
            "false_or_true_positive_rate_at_5pp": float(np.mean(change >= EFFECT_PP)),
            "median_change_pp": float(np.median(change)),
            "change_sd_pp": float(np.std(change, ddof=1)),
        })
    # Fixed model-mismatch check: two retained amplitude populations with a
    # common truncation threshold, which is outside the single-population model.
    estimates = []
    for _ in range(SYNTHETIC_REPLICATES):
        a = _truncated_sample(rng, 10.0, window_size // 2)
        b = _truncated_sample(rng, 10.0, window_size - window_size // 2) * 2.0
        estimates.append(fit_exact_window(np.concatenate([a, b]))["estimate_pct"])
    rows.append({
        "window_size": window_size,
        "comparison": "mixed_scale_model_mismatch",
        "truth_missing_pct": 10.0,
        "bias_pp": float(np.mean(estimates) - 10.0),
        "absolute_error_pp": float(np.mean(np.abs(np.asarray(estimates) - 10.0))),
        "spread_sd_pp": float(np.std(estimates, ddof=1)),
    })
    return rows


def _load_baseline() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    curated = Path("/mnt/NPX/Luke/20250804/rescue_pipeline_results_Luke0804_V2V1_g0_imec0/cur/cur_output")
    full = np.load(curated / "full_st.npy", mmap_mode="r")
    kept = np.load(curated / "kept_spikes.npy", mmap_mode="r")
    times = np.load(curated / "spike_times.npy", mmap_mode="r").reshape(-1).astype(np.int64)
    clusters = np.load(curated / "spike_clusters.npy", mmap_mode="r").reshape(-1).astype(np.int64)
    positions = np.load(curated / "spike_positions.npy", mmap_mode="r")
    labels = np.genfromtxt(curated / "cluster_KSLabel.tsv", delimiter="\t", names=True, dtype=None, encoding="utf-8")
    label_name = next(name for name in labels.dtype.names if name != "cluster_id")
    good = [int(cid) for cid, label in zip(labels["cluster_id"], labels[label_name]) if str(label).lower() == "good"]
    seconds = times / FS_HZ
    in_nominated = (
        (seconds >= INTERVALS["nominated"][0]) & (seconds < INTERVALS["nominated"][1])
        & (positions[:, 1] >= DEPTH_BAND_UM[0]) & (positions[:, 1] <= DEPTH_BAND_UM[1])
    )
    cohort = [cid for cid in good if np.any((clusters == cid) & in_nominated)]
    amplitudes = full[kept][:, 2].astype(np.float64)
    return times, clusters, amplitudes, cohort


def baseline_rows() -> list[dict]:
    times, clusters, amplitudes, cohort = _load_baseline()
    rows = []
    for interval_name, interval in INTERVALS.items():
        interval_seconds = interval[1] - interval[0]
        for size in WINDOW_SIZES:
            measured = 0
            statuses: dict[str, int] = {}
            unit_medians: dict[int, float] = {}
            shifted_medians: dict[int, float] = {}
            physical_coverage = []
            for cid in cohort:
                mask = (clusters == cid) & (times / FS_HZ >= interval[0]) & (times / FS_HZ < interval[1])
                unit_times = times[mask]
                unit_amplitudes = amplitudes[mask]
                order = np.argsort(unit_times, kind="stable")
                unit_times, unit_amplitudes = unit_times[order], unit_amplitudes[order]
                estimates = []
                window_coverage = 0.0
                for start, stop in exact_windows(unit_times.size, size, 0):
                    result = fit_exact_window(unit_amplitudes[start:stop])
                    window_coverage += float(unit_times[stop - 1] - unit_times[start]) / FS_HZ
                    if not result["saturated"] and not result["fallback"] and np.isfinite(result["estimate_pct"]):
                        estimates.append(result["estimate_pct"])
                physical_coverage.append(min(1.0, window_coverage / interval_seconds))
                status = "measured" if len(estimates) >= MIN_FINITE_FITS else (
                    "too_few_spikes" if len(exact_windows(unit_times.size, size, 0)) == 0 else "insufficient_finite_fits"
                )
                statuses[status] = statuses.get(status, 0) + 1
                if status == "measured":
                    measured += 1
                    unit_medians[cid] = float(np.median(estimates))
                shifted = []
                for start, stop in exact_windows(unit_times.size, size, size // 2):
                    result = fit_exact_window(unit_amplitudes[start:stop])
                    if not result["saturated"] and not result["fallback"] and np.isfinite(result["estimate_pct"]):
                        shifted.append(result["estimate_pct"])
                if len(shifted) >= MIN_FINITE_FITS:
                    shifted_medians[cid] = float(np.median(shifted))
            paired = sorted(set(unit_medians) & set(shifted_medians))
            differences = [abs(unit_medians[cid] - shifted_medians[cid]) for cid in paired]
            rows.append({
                "interval": interval_name,
                "duration_s": interval_seconds,
                "window_size": size,
                "cohort_denominator": len(cohort),
                "measurable_units": measured,
                "unit_coverage_fraction": measured / len(cohort) if cohort else np.nan,
                "physical_time_coverage_fraction": float(np.mean(physical_coverage)) if physical_coverage else np.nan,
                "status_counts": statuses,
                "origin_shift_paired_units": len(paired),
                "origin_shift_median_abs_difference_pp": float(np.median(differences)) if differences else np.nan,
                "origin_shift_p90_abs_difference_pp": float(np.percentile(differences, 90)) if differences else np.nan,
            })
    return rows


def run(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SYNTHETIC_SEED)
    synthetic = []
    for size in WINDOW_SIZES:
        synthetic.extend(_synthetic_rows(size, rng))
    baseline = baseline_rows()
    pd.DataFrame(synthetic).to_csv(output / "synthetic_window_calibration.csv", index=False)
    pd.DataFrame(baseline).to_csv(output / "baseline_window_feasibility.csv", index=False)
    summary = {
        "schema": SCHEMA,
        "window_sizes": WINDOW_SIZES,
        "synthetic_replicates_per_truth_and_comparison": SYNTHETIC_REPLICATES,
        "synthetic_seed": SYNTHETIC_SEED,
        "baseline_intervals": INTERVALS,
        "baseline_cohort_source": "rescue baseline only; 124 units present in nominated interval/depth band",
        "production_qc_unchanged": True,
        "historical_1000_result_preserved": True,
        "calibration_is_not_real_neuron_validation": True,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("docs/outputs/luke_truncation_window_feasibility_v1"))
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2))


if __name__ == "__main__":
    main()