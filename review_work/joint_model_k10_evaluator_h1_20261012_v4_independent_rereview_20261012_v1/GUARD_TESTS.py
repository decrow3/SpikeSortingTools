"""Focused independent exact-key-set cases for K10 evaluator v4."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


PACKET = Path(__file__).resolve().parents[1] / "joint_model_k10_evaluator_h1_20261012_v4"
SPEC = importlib.util.spec_from_file_location("frozen_k10_evaluator_v4_guard", PACKET / "EVALUATOR.py")
assert SPEC is not None and SPEC.loader is not None
evaluator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluator)


DEVELOPMENT = {"N1": {"recovery": 1.0}, "N2": {"recovery": 0.0}}
CONFIG = {
    "baseline_arm": "N0",
    "candidate_arms": ["N1", "N2"],
    "selection_primary_endpoint": "recovery",
    "selection_direction": "higher",
    "endpoint_rules": {
        "recovery": {"direction": "higher", "practical_loss_margin": 0.1}
    },
}


def exact_holdout():
    return {
        "candidate_arm": "N1",
        "baseline_arm": "N0",
        "endpoints": {
            "recovery": {"estimate": 0.0, "ci_low": -0.1, "ci_high": 0.1, "n_units": 1}
        },
    }


def test_exact_three_key_summary_remains_accepted():
    report = evaluator.selection_aware_arm_report(DEVELOPMENT, exact_holdout(), CONFIG)
    assert report["selected_arm"] == "N1"
    assert report["overall_decision"] == "PASS_NONINFERIOR"


@pytest.mark.parametrize(
    "payload",
    [
        None,
        [],
        {"candidate_arm": "N1", "baseline_arm": "N0"},
        {**exact_holdout(), "metadata": {}},
        {**exact_holdout(), "non_selected_candidate_holdout": {"candidate_arm": "N2"}},
    ],
)
def test_nonmapping_missing_or_extra_top_level_keys_fail_same_guard(payload):
    with pytest.raises(ValueError, match="exactly one selected-arm-versus-baseline summary"):
        evaluator.selection_aware_arm_report(DEVELOPMENT, payload, CONFIG)

