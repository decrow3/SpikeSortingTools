#!/usr/bin/env python3
"""Discrete boundary/rate-step fixture for CW common-target null support."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def has_partner(query: np.ndarray, reference: np.ndarray, lo: int = 8, hi: int = 29):
    return np.array([
        np.any((np.abs(reference - q) >= lo) & (np.abs(reference - q) <= hi))
        for q in query
    ], dtype=bool)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("output", type=Path)
    args = ap.parse_args()

    left, right, delta, max_lag = 0, 1000, 100, 29
    # C = I intersect (I + delta), then trim the maximum coincidence lag.
    c_left, c_right = max(left, left + delta), min(right, right + delta)
    support_left, support_right = c_left + max_lag, c_right - max_lag

    # Uniform query roots; reference events have a sharp target-time rate step.
    roots = np.arange(left, right, 10)
    reference = np.arange(908, right, 10)

    observed = roots[(roots >= support_left) & (roots < support_right)]
    null_roots = roots[(roots + delta >= support_left) & (roots + delta < support_right)]
    shifted = null_roots + delta
    shared_reference = reference[(reference >= support_left) & (reference < support_right)]
    corrected_observed = has_partner(observed, shared_reference)
    corrected_null = has_partner(shifted, shared_reference)

    # Superseded support: observed target positions and shifted target positions
    # span different domains, allowing the reference-rate step to bias the null.
    old_roots = roots[(roots + delta >= left) & (roots + delta < right)]
    old_observed = has_partner(old_roots, reference)
    old_null = has_partner(old_roots + delta, reference)

    assert np.array_equal(observed, shifted)
    assert np.array_equal(corrected_observed, corrected_null)
    assert old_observed.mean() != old_null.mean()
    assert support_left == 129 and support_right == 971

    result = {
        "status": "pass",
        "interval_I": [left, right],
        "signed_offset": delta,
        "common_target_C": [c_left, c_right],
        "trimmed_common_target_support": [support_left, support_right],
        "corrected": {
            "observed_queries": int(observed.size),
            "null_queries": int(shifted.size),
            "target_positions_identical": True,
            "reference_events": int(shared_reference.size),
            "observed_fraction": float(corrected_observed.mean()),
            "null_fraction": float(corrected_null.mean()),
            "contrast": float(corrected_observed.mean() - corrected_null.mean()),
        },
        "superseded_different_target_support": {
            "observed_fraction": float(old_observed.mean()),
            "null_fraction": float(old_null.mean()),
            "contrast": float(old_observed.mean() - old_null.mean()),
        },
        "boundary_checks": {
            "exact_8_in_target_band": bool(has_partner(np.array([100]), np.array([108]), 8, 29)[0]),
            "exact_29_in_target_band": bool(has_partner(np.array([100]), np.array([129]), 8, 29)[0]),
            "exact_30_outside_target_band": bool(not has_partner(np.array([100]), np.array([130]), 8, 29)[0]),
        },
        "interpretation": (
            "A sharp reference-rate step biases the superseded comparison because observed and shifted "
            "targets occupy different time domains. On one common target domain, the same target positions "
            "and reference events give exactly zero contrast."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
