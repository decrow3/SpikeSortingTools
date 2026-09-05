"""Post-hoc amplitude-truncation diagnostic for the C2 v4 staircase.

This is deliberately *not* production truncation QC.  Production uses
1,000-spike windows, while the staircase contains only 687 admitted events.
Here we use frozen 250-spike, phase-specific windows and calibrate that smaller
sample size against synthetic logistic ground truth before inspecting C2.

The diagnostic reads retained L1 curation outputs; it never reruns sorting and
never modifies an L1 cache.  Its primary unit of analysis is one capturing
output cluster in one staircase phase.  Capturing clusters are not pooled:
their Kilosort amplitudes need not share a scale, and pooling fragments could
manufacture a multimodal amplitude distribution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.truncation import (
    fit_amp_cdf,
    is_saturated,
    missing_pct_from_normalisation,
)
from testing.ladder_score import _exclusive_pairs, truth_digest
from testing.luke_c2_staircase_control import STAIRCASE, staircase_admitted_truth
from testing.luke_rescue_c2_drift_challenge_v4 import PRESPEC

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_C2_ROOT = REPO_ROOT / "testing/outputs/luke_rescue_c2_drift_challenge_v4"
DEFAULT_OUTPUT = DEFAULT_C2_ROOT / "truncation_diagnostic"

SCHEMA = "luke-c2-v4-truncation-diagnostic-v1"
SPIKES_PER_WINDOW = 250
TRUE_MISSING_PCTS = (0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 40.0)
CALIBRATION_REPLICATES = 100
CALIBRATION_SEED = 20260904


def exact_count_windows(n_spikes: int, size: int = SPIKES_PER_WINDOW) -> list[tuple[int, int]]:
    """Return centred, non-overlapping [start, stop) windows of exactly size.

    Production's historical helper treats the stop as inclusive when recording
    it but exclusive when slicing, fitting 999 samples in a nominal 1,000-spike
    window.  That discrepancy matters at n=250, so this diagnostic fixes the
    interval semantics and records that departure explicitly.
    """
    n_windows = int(n_spikes) // int(size)
    if n_windows == 0:
        return []
    used = n_windows * int(size)
    start = (int(n_spikes) - used) // 2
    return [(i, i + int(size)) for i in range(start, start + used, int(size))]


def staircase_phase(samples, fs: float) -> np.ndarray:
    """Assign samples to the 0/40-um hard plateaus using right-edge semantics."""
    samples = np.asarray(samples, dtype=np.int64)
    plateau_frames = float(STAIRCASE["plateau_s"]) * float(fs)
    index = np.floor(samples / plateau_frames).astype(np.int64)
    index = np.clip(index, 0, int(STAIRCASE["n_plateaus"]) - 1)
    levels = np.asarray(STAIRCASE["levels_um"], dtype=float)
    return levels[index % levels.size]


def _truncated_sample(rng, true_missing_pct: float, n: int) -> np.ndarray:
    x0, k = 20.0, 0.5
    p = float(true_missing_pct) / 100.0
    threshold = x0 + np.log(p / (1.0 - p)) / k
    draw = rng.logistic(x0, 1.0 / k, size=int(n / (1.0 - p)) + 4000)
    kept = draw[draw > threshold]
    if kept.size < n:
        raise RuntimeError("synthetic calibration did not draw enough retained events")
    return kept[:n]


def calibrate_small_window(
    *,
    window_size: int = SPIKES_PER_WINDOW,
    replicates: int = CALIBRATION_REPLICATES,
    seed: int = CALIBRATION_SEED,
) -> pd.DataFrame:
    """Measure small-window sampling error under the fitter's assumed model."""
    rows = []
    rng = np.random.default_rng(seed)
    for truth in TRUE_MISSING_PCTS:
        estimates = []
        for _ in range(int(replicates)):
            _, estimate = fit_amp_cdf(_truncated_sample(rng, truth, window_size))
            estimates.append(float(estimate))
        values = np.asarray(estimates)
        rows.append({
            "true_missing_pct": truth,
            "n_replicates": int(replicates),
            "median_estimated_pct": float(np.median(values)),
            "median_bias_pp": float(np.median(values) - truth),
            "median_abs_error_pp": float(np.median(np.abs(values - truth))),
            "p05_estimated_pct": float(np.percentile(values, 5)),
            "p95_estimated_pct": float(np.percentile(values, 95)),
            "saturated_fraction": float(np.mean(is_saturated(values))),
        })
    return pd.DataFrame(rows)


def load_curated_arrays(curated: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load the exact time/cluster/amplitude triplet used by production QC."""
    times = np.load(curated / "spike_times.npy").reshape(-1).astype(np.int64)
    clusters = np.load(curated / "spike_clusters.npy").reshape(-1)
    full_st = np.load(curated / "full_st.npy")
    kept = np.load(curated / "kept_spikes.npy")
    st = full_st[kept]
    if not (times.size == clusters.size == st.shape[0]):
        raise ValueError(f"unaligned curated arrays in {curated}")
    if not np.array_equal(times, st[:, 0].astype(np.int64)):
        raise ValueError(f"full_st times do not match spike_times in {curated}")
    # pipeline.qc.run_qc uses KilosortResults.st[:, 2], not amplitudes.npy.
    amplitudes = st[:, 2].astype(np.float64)
    return times, clusters, amplitudes


V4_ARM_PREFIX = "staircase_40um_"
V4_ARMS = ("static", "moved", "moved_corrected")
V4_EXPECTED_CELLS = 14 * 7


def _result_identity(result: dict, arm_prefix: str = V4_ARM_PREFIX,
                     arm_names: tuple = V4_ARMS) -> tuple[str, str, str]:
    """Donor, arm and sorter config from a retained L1 result's snippet name.

    Defaults describe the C2 v4 layout. The threshold-staircase comparison names
    its recordings `<donor>_static` / `<donor>_staircase` with no condition
    segment, so it passes `arm_prefix=""` and its own arm names.
    """
    name = Path(result["snippet_dir"]).name
    donor, rest = name.split("_", 1)
    if not rest.startswith(arm_prefix):
        raise ValueError(f"not a staircase result: {name}")
    arm = rest[len(arm_prefix):]
    if arm not in arm_names:
        raise ValueError(f"unknown staircase arm in {name}")
    return donor, arm, str(result["sorter_config"])


def find_staircase_results(c2_root: Path, arm_prefix: str = V4_ARM_PREFIX,
                           arm_names: tuple = V4_ARMS,
                           expected: int | None = V4_EXPECTED_CELLS
                           ) -> list[tuple[Path, dict]]:
    """Find exactly one retained L1 result for every frozen staircase cell."""
    found = {}
    for path in (c2_root / "runs/_l1").rglob("l1_result.json"):
        result = json.loads(path.read_text())
        contract = result.get("score", {}).get("truth_contract") or {}
        admission = contract.get("admission") or {}
        if admission.get("schema") != STAIRCASE["schema"]:
            continue
        key = _result_identity(result, arm_prefix, arm_names)
        if key in found:
            raise ValueError(f"duplicate staircase L1 result for {key}")
        found[key] = (path, result)
    if expected is not None and len(found) != expected:
        raise ValueError(f"expected {expected} staircase cells, found {len(found)}")
    return [found[key] for key in sorted(found)]


def reconstruct_truth(fs: float, contract: dict) -> np.ndarray:
    guard = int(PRESPEC["train"]["guard_s"] * fs)
    duration = float(STAIRCASE["duration_s"])
    regular = np.arange(
        guard,
        int(duration * fs) - guard,
        int(round(fs / PRESPEC["train"]["rate_hz"])),
        dtype=np.int64,
    )
    truth, _ = staircase_admitted_truth(regular, fs)
    train = np.asarray(truth["inj0"], dtype=np.int64)
    if truth_digest({"inj0": train}) != contract["truth_sha256"]:
        raise ValueError("reconstructed staircase truth does not match its contract")
    return train


def _fit_window(amps: np.ndarray) -> dict:
    popt, estimate = fit_amp_cdf(amps)
    from_a = float(missing_pct_from_normalisation([popt])[0])
    fallback = bool(np.allclose(popt, [float(np.mean(amps)), 1.0, 1.0]))
    return {
        "estimated_missing_pct": float(estimate),
        "normalisation_missing_pct": from_a,
        "estimate_disagreement_pp": float(abs(estimate - from_a)),
        "saturated": bool(is_saturated([estimate])[0]),
        "fallback_signature": fallback,
        "fit_x0": float(popt[0]),
        "fit_k": float(popt[1]),
        "fit_A": float(popt[2]),
    }


def analyze_cell(result_path: Path, result: dict, arm_prefix: str = V4_ARM_PREFIX,
                 arm_names: tuple = V4_ARMS) -> tuple[list[dict], list[dict]]:
    donor, arm, sorter = _result_identity(result, arm_prefix, arm_names)
    score = result["score"]
    contract = score["truth_contract"]
    fs = float(score["fs"])
    truth = reconstruct_truth(fs, contract)
    curated = result_path.parent / "cur_output"
    times, clusters, amplitudes = load_curated_arrays(curated)
    truth_levels = staircase_phase(truth, fs)
    output_levels = staircase_phase(times, fs)
    unit = score["primary"]["units"][0]
    capturing = [int(v) for v in unit["capturing_output_units"]]
    tol = int(round(float(score["primary"]["tolerance_ms"]) / 1000.0 * fs))

    phase_rows, window_rows = [], []
    for cluster in capturing:
        for level in map(float, STAIRCASE["levels_um"]):
            ttruth = np.sort(truth[truth_levels == level])
            mask = (clusters == cluster) & (output_levels == level)
            order = np.argsort(times[mask], kind="stable")
            tout = times[mask][order]
            amps = amplitudes[mask][order]
            ti, oi = _exclusive_pairs(ttruth, tout, tol)
            tp = int(ti.size)
            fp = int(tout.size - oi.size)
            fn = int(ttruth.size - ti.size)
            windows = exact_count_windows(tout.size)
            base = {
                "template": donor,
                "arm": arm,
                "sorter": sorter,
                "phase_um": level,
                "output_cluster": cluster,
                "is_best_output_cluster": cluster == unit["best_output_unit"],
                "n_truth_phase": int(ttruth.size),
                "n_output_spikes_phase": int(tout.size),
                "tp_phase": tp,
                "fp_phase": fp,
                "fn_phase": fn,
                "true_missing_pct": 100.0 * fn / ttruth.size,
                "whole_train_best_cluster_missing_pct": (
                    100.0 * float(unit["fn"]) / float(unit["n_truth"])
                ),
                "whole_train_n_capturing": int(unit["n_output_units_capturing"]),
                "whole_train_split": bool(unit["split"]),
                "precision_pct": 100.0 * tp / tout.size if tout.size else np.nan,
                "n_windows": len(windows),
                "eligible": bool(windows),
                "result_path": str(result_path),
            }
            estimates = []
            for wi, (start, stop) in enumerate(windows):
                fitted = _fit_window(amps[start:stop])
                row = {
                    **base,
                    "window_index": wi,
                    "window_start_output_index": start,
                    "window_stop_output_index_exclusive": stop,
                    "window_n_spikes": stop - start,
                    "window_start_s": float(tout[start] / fs),
                    "window_stop_s": float(tout[stop - 1] / fs),
                    **fitted,
                }
                row["error_vs_truth_pp"] = (
                    row["estimated_missing_pct"] - row["true_missing_pct"]
                )
                estimates.append(row)
                window_rows.append(row)
            usable = [r for r in estimates if not r["saturated"] and not r["fallback_signature"]]
            base.update({
                "n_usable_windows": len(usable),
                "median_estimated_missing_pct": (
                    float(np.median([r["estimated_missing_pct"] for r in usable]))
                    if usable else np.nan
                ),
                "median_error_vs_truth_pp": (
                    float(np.median([r["error_vs_truth_pp"] for r in usable]))
                    if usable else np.nan
                ),
                "any_saturated": any(r["saturated"] for r in estimates),
                "any_fallback_signature": any(r["fallback_signature"] for r in estimates),
            })
            phase_rows.append(base)
    return phase_rows, window_rows


def select_primary_phase_clusters(phase: pd.DataFrame) -> pd.DataFrame:
    """Choose the strongest capturing cluster separately for each phase.

    This produces one donor-level observation per cell and phase.  It prevents
    a donor with extra fragments from receiving extra weight, without pretending
    that amplitudes from separately normalised Kilosort clusters can be pooled.
    """
    keys = ["template", "arm", "sorter", "phase_um"]
    ordered = phase.sort_values(
        keys + ["tp_phase", "precision_pct", "n_output_spikes_phase", "output_cluster"],
        ascending=[True, True, True, True, False, False, True, True],
        kind="stable",
    )
    # Do not use groupby.first(): it selects the first *non-null value in each
    # column* and can splice an estimate from a secondary fragment into the row
    # of an ineligible primary fragment.
    return ordered.drop_duplicates(keys, keep="first").reset_index(drop=True)


def paired_arm_comparisons(primary: pd.DataFrame) -> list[dict]:
    """Matched-donor phase contrasts; missing estimates remain unfilled."""
    out = []
    comparisons = (
        ("rescue", "moved", "static"),
        ("rescue_rigid", "moved", "static"),
        ("legacy_style", "moved", "static"),
        ("rescue", "moved_corrected", "static"),
    )
    for sorter, arm, baseline in comparisons:
        for level in map(float, STAIRCASE["levels_um"]):
            cols = [
                "template", "true_missing_pct", "median_estimated_missing_pct",
                "whole_train_best_cluster_missing_pct",
            ]
            moved = primary[
                (primary.sorter == sorter) & (primary.arm == arm) &
                (primary.phase_um == level)
            ][cols]
            static = primary[
                (primary.sorter == sorter) & (primary.arm == baseline) &
                (primary.phase_um == level)
            ][cols]
            pair = moved.merge(static, on="template", suffixes=("_arm", "_baseline"))
            estimable = pair.dropna(subset=[
                "median_estimated_missing_pct_arm",
                "median_estimated_missing_pct_baseline",
            ]).copy()
            record = {
                "sorter": sorter,
                "arm": arm,
                "baseline": baseline,
                "phase_um": level,
                "n_donors": int(len(pair)),
                "n_donors_estimable": int(len(estimable)),
                "median_delta_true_missing_pp": float(np.median(
                    pair.true_missing_pct_arm - pair.true_missing_pct_baseline
                )),
                "median_delta_whole_train_missing_pp": float(np.median(
                    pair.whole_train_best_cluster_missing_pct_arm -
                    pair.whole_train_best_cluster_missing_pct_baseline
                )),
                "median_delta_estimated_missing_pp": None,
            }
            if len(estimable):
                record["median_delta_estimated_missing_pp"] = float(np.median(
                    estimable.median_estimated_missing_pct_arm -
                    estimable.median_estimated_missing_pct_baseline
                ))
            out.append(record)
    return out


def mechanistic_headline(primary: pd.DataFrame) -> dict:
    """Compact statement of what truncation adds to the staircase result."""
    moved = primary[(primary.arm == "moved") & (primary.sorter == "rescue")]
    per_donor = moved.groupby("template", sort=True).agg(
        split=("whole_train_split", "first"),
        whole_missing_pct=("whole_train_best_cluster_missing_pct", "first"),
        median_phase_true_missing_pct=("true_missing_pct", "median"),
        median_phase_estimated_missing_pct=("median_estimated_missing_pct", "median"),
        max_phase_estimated_missing_pct=("median_estimated_missing_pct", "max"),
        n_phase_estimates=("median_estimated_missing_pct", "count"),
    )
    split = per_donor[per_donor.split]

    def whole_median(arm: str, sorter: str) -> float:
        subset = primary[(primary.arm == arm) & (primary.sorter == sorter)]
        return float(subset.groupby("template").whole_train_best_cluster_missing_pct.first().median())

    static = primary[(primary.arm == "static") & (primary.sorter == "rescue")]
    corrected = primary[
        (primary.arm == "moved_corrected") & (primary.sorter == "rescue")
    ]
    paired = static.merge(
        corrected,
        on=["template", "phase_um"],
        suffixes=("_static", "_corrected"),
    ).dropna(subset=[
        "median_estimated_missing_pct_static",
        "median_estimated_missing_pct_corrected",
    ])
    return {
        "rescue_unadjusted": {
            "n_split_donors": int(split.shape[0]),
            "n_donors": int(per_donor.shape[0]),
            "median_whole_train_best_cluster_missing_pct_split_donors": float(
                split.whole_missing_pct.median()
            ),
            "median_within_phase_true_missing_pct_split_donors": float(
                split.median_phase_true_missing_pct.median()
            ),
            "median_within_phase_estimated_missing_pct_split_donors": float(
                split.median_phase_estimated_missing_pct.median()
            ),
            "split_donors_with_both_phase_estimates_below_5pct": int((
                (split.n_phase_estimates == 2) &
                (split.max_phase_estimated_missing_pct < 5.0)
            ).sum()),
        },
        "median_whole_train_best_cluster_missing_pct": {
            "static_rescue": whole_median("static", "rescue"),
            "moved_rescue": whole_median("moved", "rescue"),
            "moved_rescue_rigid": whole_median("moved", "rescue_rigid"),
            "moved_legacy_style": whole_median("moved", "legacy_style"),
            "moved_corrected_rescue": whole_median("moved_corrected", "rescue"),
        },
        "static_vs_exact_corrected": {
            "n_phase_pairs": int(len(paired)),
            "max_abs_estimated_missing_difference_pp": float(np.max(np.abs(
                paired.median_estimated_missing_pct_corrected -
                paired.median_estimated_missing_pct_static
            ))) if len(paired) else None,
        },
    }


def summarize(
    phase: pd.DataFrame,
    windows: pd.DataFrame,
    calibration: pd.DataFrame,
    primary: pd.DataFrame,
) -> dict:
    eligible = phase[phase["eligible"]].copy()
    usable = windows[(~windows["saturated"]) & (~windows["fallback_signature"])].copy()
    grouped = []
    for keys, part in eligible.groupby(["arm", "sorter", "phase_um"], sort=True):
        arm, sorter, level = keys
        upart = usable[
            (usable.arm == arm) & (usable.sorter == sorter) & (usable.phase_um == level)
        ]
        grouped.append({
            "arm": arm,
            "sorter": sorter,
            "phase_um": float(level),
            "n_cluster_phases_eligible": int(len(part)),
            "n_windows_usable": int(len(upart)),
            "median_true_missing_pct": float(part.true_missing_pct.median()),
            "median_estimated_missing_pct": (
                float(upart.estimated_missing_pct.median()) if len(upart) else None
            ),
            "median_error_vs_truth_pp": (
                float(upart.error_vs_truth_pp.median()) if len(upart) else None
            ),
        })
    low = calibration[calibration.true_missing_pct <= 20.0]
    return {
        "schema": SCHEMA,
        "status": "post-hoc diagnostic; not production QC or a promotion endpoint",
        "window_policy": {
            "spikes_per_window": SPIKES_PER_WINDOW,
            "phase_specific": True,
            "phases_um": list(map(float, STAIRCASE["levels_um"])),
            "max_isi": None,
            "clusters_pooled": False,
            "interval_semantics": "[start, stop), exactly 250 amplitudes",
        },
        "calibration": {
            "model": "logistic amplitudes truncated at a known quantile",
            "replicates_per_truth_level": CALIBRATION_REPLICATES,
            "seed": CALIBRATION_SEED,
            "max_abs_median_bias_pp_0p5_to_20": float(low.median_bias_pp.abs().max()),
            "warning": (
                "sampling intervals widen with missingness; saturation remains censoring, "
                "and real multimodal/non-logistic amplitudes can violate this calibration"
            ),
        },
        "coverage": {
            "n_staircase_cells": 98,
            "n_capturing_clusters": int(
                phase[["template", "arm", "sorter", "output_cluster"]].drop_duplicates().shape[0]
            ),
            "n_cluster_phases": int(len(phase)),
            "n_cluster_phases_eligible": int(phase.eligible.sum()),
            "n_windows": int(len(windows)),
            "n_usable_windows": int(len(usable)),
            "n_saturated_windows": int(windows.saturated.sum()),
            "n_fallback_signature_windows": int(windows.fallback_signature.sum()),
        },
        "group_summary": grouped,
        "mechanistic_headline": mechanistic_headline(primary),
        "paired_primary_phase_comparisons": paired_arm_comparisons(primary),
        "interpretation_guard": (
            "Compare the estimate with injected-truth missingness. Per-fragment flat "
            "truncation alongside high whole-train FN is evidence of temporal identity "
            "fragmentation, not evidence that no spikes were lost."
        ),
    }


def run(c2_root: Path = DEFAULT_C2_ROOT, output: Path = DEFAULT_OUTPUT,
        arm_prefix: str = V4_ARM_PREFIX, arm_names: tuple = V4_ARMS,
        expected: int | None = V4_EXPECTED_CELLS) -> dict:
    c2_root, output = Path(c2_root), Path(output)
    phase_rows, window_rows = [], []
    for path, result in find_staircase_results(c2_root, arm_prefix, arm_names, expected):
        p, w = analyze_cell(path, result, arm_prefix, arm_names)
        phase_rows.extend(p)
        window_rows.extend(w)
    phase = pd.DataFrame(phase_rows)
    windows = pd.DataFrame(window_rows)
    calibration = calibrate_small_window()
    primary = select_primary_phase_clusters(phase)
    summary = summarize(phase, windows, calibration, primary)

    output.mkdir(parents=True, exist_ok=True)
    phase.to_csv(output / "cluster_phase.csv", index=False)
    primary.to_csv(output / "primary_phase.csv", index=False)
    windows.to_csv(output / "windows.csv", index=False)
    calibration.to_csv(output / "calibration.csv", index=False)
    input_results = find_staircase_results(c2_root, arm_prefix, arm_names, expected)
    inputs = sorted(str(p) for p, _ in input_results)
    content_digest = hashlib.sha256()
    for path, _ in sorted(input_results, key=lambda item: str(item[0])):
        content_digest.update(path.read_bytes())
    summary["provenance"] = {
        "c2_root": str(c2_root.resolve()),
        "l1_result_count": len(inputs),
        "l1_result_paths_sha256": hashlib.sha256("\n".join(inputs).encode()).hexdigest(),
        "l1_result_contents_sha256": content_digest.hexdigest(),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--c2-root", type=Path, default=DEFAULT_C2_ROOT)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = ap.parse_args()
    print(json.dumps(run(args.c2_root, args.output), indent=2))


if __name__ == "__main__":
    main()
