"""Focused independent clock-boundary cases for frozen K10 evaluator v2."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


PACKET = Path(__file__).resolve().parents[1] / "joint_model_k10_evaluator_h1_20261012_v2"
SPEC = importlib.util.spec_from_file_location("frozen_k10_evaluator_v2_bounds", PACKET / "EVALUATOR.py")
assert SPEC is not None and SPEC.loader is not None
evaluator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluator)


def truth(samples):
    n = len(samples)
    return {
        "event_id": np.asarray([f"e{i}" for i in range(n)]),
        "unit_id": np.ones(n, dtype=np.int64),
        "sample_index": np.asarray(samples, dtype=np.int64),
        "placement_um": np.zeros(n),
        "frame": np.asarray(["tissue"] * n),
        "amplitude": np.ones(n),
        "is_overlap": np.zeros(n, dtype=bool),
        "decoy_group": np.asarray([""] * n),
    }


def output(samples):
    n = len(samples)
    return {
        "sample_index": np.asarray(samples, dtype=np.int64),
        "cluster_id": np.full(n, 7, dtype=np.int64),
        "placement_um": np.zeros(n),
        "assigned_frame": np.asarray(["tissue"] * n),
        "amplitude": np.ones(n),
    }


def config(bounds):
    return {
        "time_tolerance_samples": 15,
        "spatial_tolerance_um": 40.0,
        "session_bounds_samples": bounds,
        "null_shifts_samples": [20],
        "clock_frame": "session_acquisition_samples",
        "clock_identity": "v2-bounds-rereview-clock",
    }


@pytest.mark.parametrize(
    "bounds",
    [
        [1000.25, 2000],
        [1000, 2000.25],
        [float("nan"), 2000],
        [1000, float("inf")],
    ],
)
def test_fractional_and_nonfinite_bounds_fail_closed(bounds):
    with pytest.raises(ValueError, match="integer"):
        evaluator.evaluate_saved_outputs(truth([1500]), output([1500]), config(bounds))


def test_exact_integral_float_bounds_are_accepted_and_normalized_to_ints():
    result = evaluator.evaluate_saved_outputs(
        truth([1000, 1999]), output([1000, 1999]), config([1000.0, 2000.0])
    )
    assert result["config"]["session_bounds_samples"] == [1000, 2000]
    assert all(isinstance(value, int) for value in result["config"]["session_bounds_samples"])


def test_half_open_lower_inclusive_upper_exclusive_semantics_are_retained():
    evaluator.evaluate_saved_outputs(
        truth([1000, 1999]), output([1000, 1999]), config([1000, 2000])
    )
    with pytest.raises(ValueError, match="outside"):
        evaluator.evaluate_saved_outputs(truth([2000]), output([]), config([1000, 2000]))


def test_nonzero_origin_circular_null_uses_repaired_exact_bounds():
    result = evaluator.evaluate_saved_outputs(
        truth([1990]), output([1990, 1010]), config([1000, 2000])
    )
    score = result["units"][0]["best_score"]
    assert score["tp"] == 1
    assert score["null_match_counts"] == [1]
    assert score["above_chance"] is False

