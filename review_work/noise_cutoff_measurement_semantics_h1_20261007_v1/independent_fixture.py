#!/usr/bin/env python3
"""Independent known-answer fixtures for the R1 noise-cutoff formula.

This file deliberately does not import the production evaluator or any project
helper. Expected values are fixed from explicit histogram-count arithmetic.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


OUT = Path(__file__).with_name("FIXTURE_RECEIPT.json")


def independent_measure(amplitudes: np.ndarray, n_edges: int = 6) -> dict:
    """Independent transcription used only for fixtures, with diagnostics."""
    a = np.asarray(amplitudes, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"value": None, "passed": False, "undefined_reason": "empty_after_finite_filter"}
    if np.any(a < 0):
        return {"value": None, "passed": False, "undefined_reason": "negative_retained_value"}
    maximum = float(a.max())
    if not np.isfinite(maximum) or maximum <= 0:
        return {"value": None, "passed": False, "undefined_reason": "nonpositive_or_nonfinite_max"}

    hist, edges = np.histogram(a, bins=np.linspace(0.0, maximum, n_edges))
    peak_index = int(np.argmax(hist))
    occupied_top = int(np.count_nonzero(hist[peak_index:-1] > 0))
    high_start = int(math.ceil(0.5 * occupied_top + peak_index))
    high = hist[high_start:]
    high = high[high >= 1]
    nonzero = hist[hist != 0]
    base = {
        "hist": hist.tolist(),
        "edges": edges.tolist(),
        "peak_index": peak_index,
        "occupied_top": occupied_top,
        "high_start": high_start,
        "high": high.tolist(),
        "retained_count": int(a.size),
        "maximum": maximum,
    }
    if high.size == 0:
        return {**base, "value": None, "passed": False, "undefined_reason": "empty_positive_tail"}
    if nonzero.size < 2:
        return {**base, "value": None, "passed": False, "undefined_reason": "fewer_than_two_occupied_bins"}
    tail_std = float(np.std(high))
    if not np.isfinite(tail_std) or tail_std <= 0:
        return {**base, "value": None, "passed": False, "undefined_reason": "zero_or_nonfinite_tail_std"}
    first_low = float(nonzero[1])
    tail_mean = float(np.mean(high))
    value = float((first_low - tail_mean) / tail_std)
    return {
        **base,
        "first_low": first_low,
        "tail_mean": tail_mean,
        "tail_std": tail_std,
        "value": value,
        "passed": bool(np.isfinite(value) and value < 5.0),
        "undefined_reason": None,
    }


def amplitudes_for_counts(counts: list[int]) -> np.ndarray:
    """Create values in five fixed bins [0,.2),...,[.8,1]."""
    centers = [0.1, 0.3, 0.5, 0.7, 0.9]
    values: list[float] = []
    for count, center in zip(counts, centers):
        values.extend([center] * count)
    # Preserve the requested last-bin count while fixing the dynamic maximum.
    values[-1] = 1.0
    return np.asarray(values, dtype=np.float64)


def close(a: float, b: float, tol: float = 1e-12) -> bool:
    return bool(abs(a - b) <= tol)


def run() -> dict:
    cases = []

    pass_counts = [10, 8, 6, 4, 2]
    pass_expected = 4.0 / math.sqrt(8.0 / 3.0)
    pass_actual = independent_measure(amplitudes_for_counts(pass_counts))
    cases.append({
        "name": "manual_finite_pass",
        "expected_hist": pass_counts,
        "expected_value": pass_expected,
        "expected_pass": True,
        "actual": pass_actual,
        "ok": pass_actual["hist"] == pass_counts and close(pass_actual["value"], pass_expected) and pass_actual["passed"],
    })

    fail_counts = [20, 18, 2, 1, 2]
    fail_expected = (18.0 - (5.0 / 3.0)) / math.sqrt(2.0 / 9.0)
    fail_actual = independent_measure(amplitudes_for_counts(fail_counts))
    cases.append({
        "name": "manual_finite_fail",
        "expected_hist": fail_counts,
        "expected_value": fail_expected,
        "expected_pass": False,
        "actual": fail_actual,
        "ok": fail_actual["hist"] == fail_counts and close(fail_actual["value"], fail_expected) and not fail_actual["passed"],
    })

    for name, values, reason in [
        ("empty", np.asarray([], dtype=float), "empty_after_finite_filter"),
        ("negative", np.asarray([0.1, -0.1, 1.0]), "negative_retained_value"),
        ("zero_tail_variance", amplitudes_for_counts([10, 8, 2, 2, 2]), "zero_or_nonfinite_tail_std"),
    ]:
        actual = independent_measure(values)
        cases.append({"name": name, "expected_undefined_reason": reason, "actual": actual,
                      "ok": actual["value"] is None and actual["undefined_reason"] == reason})

    # The downstream decision is strict; this isolates the boundary without
    # pretending a histogram fixture happened to produce exactly 5.
    cases.append({
        "name": "strict_threshold_exactly_five",
        "input_scalar": 5.0,
        "expected_pass": False,
        "actual_pass": bool(np.isfinite(5.0) and 5.0 < 5.0),
        "ok": not bool(np.isfinite(5.0) and 5.0 < 5.0),
    })

    core = amplitudes_for_counts(pass_counts)
    core_result = independent_measure(core)
    outlier_result = independent_measure(np.r_[core, 10.0])
    cases.append({
        "name": "dynamic_max_outlier_changes_construction",
        "core": core_result,
        "with_outlier": outlier_result,
        "expected": "histogram or definedness changes because every edge is rescaled to the new maximum",
        "ok": core_result.get("hist") != outlier_result.get("hist") and core_result.get("value") != outlier_result.get("value"),
    })

    receipt = {
        "schema": "noise-cutoff-independent-fixture-receipt-v1",
        "production_helper_imported": False,
        "n_edges": 6,
        "n_histogram_bins": 5,
        "numpy_std_ddof": 0,
        "cases": cases,
        "all_passed": all(case["ok"] for case in cases),
    }
    OUT.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    if not receipt["all_passed"]:
        raise SystemExit(1)
    return receipt


if __name__ == "__main__":
    result = run()
    print(json.dumps({"all_passed": result["all_passed"], "cases": len(result["cases"])}))
