"""Independent synthetic-only adversarial review of frozen K7 score oracle."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


PACKET = Path(__file__).resolve().parents[1] / "joint_model_k7_frame_score_h1_20261012_v1"
REPOSITORY = Path(__file__).resolve().parents[2]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))
SPEC = importlib.util.spec_from_file_location("frozen_joint_model_k7_v1", PACKET / "IMPLEMENTATION.py")
assert SPEC is not None and SPEC.loader is not None
k7 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = k7
SPEC.loader.exec_module(k7)


def candidate(event_id, sample, state, block, **extra):
    return {
        "event_id": event_id,
        "sample_index": sample,
        "state_um": state,
        "block_id": block,
        **extra,
    }


def residuals(state, count, score):
    return [
        {"state_um": state, "residual_probe": float(np.exp(score)), "residual_tissue": 1.0}
        for _ in range(count)
    ]


def score(training, rows, **changes):
    options = dict(probe_fixed_max_score=-0.5, tissue_fixed_min_score=0.5)
    options.update(changes)
    return k7.score_frame_residuals(np.asarray(training), rows, **options)


def test_candidate_union_is_stable_complete_and_residual_independent():
    unit = [candidate("u0", 10, -80, 0, residual_probe=999), candidate("both", 20, 0, 1)]
    matched = [candidate("both", 20, 0, 1, label="ignored"), candidate("mf", 30, 80, 2)]
    got = k7.build_candidate_union(unit, matched)
    assert [row["event_id"] for row in got] == ["u0", "both", "mf"]
    assert got[1]["sources"] == ["unit", "matched_filter"]
    assert got[2]["sources"] == ["matched_filter"]
    assert all("residual_probe" not in row and "label" not in row for row in got)


@pytest.mark.parametrize("field,value", [("sample_index", 11), ("state_um", 40), ("block_id", 2)])
def test_duplicate_event_conflict_fails_for_every_key(field, value):
    first = candidate("same", 10, 0, 1)
    second = dict(first); second[field] = value
    with pytest.raises(ValueError, match="conflicting"):
        k7.build_candidate_union([first], [second])


def test_residual_direction_and_exact_label_boundaries():
    probe = score([0, 80], residuals(80, 50, -0.5))
    tissue = score([0, 80], residuals(80, 50, 0.5))
    assert probe["reported_label"] == "probe_fixed"
    assert tissue["reported_label"] == "tissue_fixed"
    assert probe["score"] == pytest.approx(-0.5)
    assert tissue["score"] == pytest.approx(0.5)


def test_median_of_state_medians_is_not_event_weighted():
    rows = residuals(-80, 50, -3.0) + residuals(80, 500, 1.0) + residuals(160, 51, 2.0)
    got = score([-80, 0], rows, probe_fixed_max_score=-1.0, tissue_fixed_min_score=1.0)
    assert got["score"] == pytest.approx(1.0)
    assert got["contributing_state_medians"] == pytest.approx({-80: -3.0, 80: 1.0, 160: 2.0})


def test_default_eligibility_count_and_abstention_boundaries():
    assert score([0, 40], residuals(80, 50, -4.0))["reason"] == "training_state_support"
    assert score([0, 80], residuals(40, 100, -4.0))["reason"] == "held_out_state_support"
    assert score([0, 80], residuals(80, 49, -4.0))["reason"] == "held_out_state_support"
    got = score([0, 80], residuals(80, 50, 0.0))
    assert got["reported_label"] == "undetermined"
    assert got["operational_frame"] == "tissue"


def test_contract_80um_support_cannot_be_weakened_by_argument():
    with pytest.raises(ValueError, match="80|informative"):
        score([0, 40], residuals(40, 50, -4.0), informative_abs_state_um=40)


def test_contract_50_event_support_cannot_be_weakened_by_argument():
    with pytest.raises(ValueError, match="50|minimum"):
        score([0, 80], residuals(80, 1, -4.0), minimum_events_per_state=1)


def test_training_80um_rule_is_not_tightened_by_heldout_cutoff():
    got = score([0, 80], residuals(160, 50, -2.0), informative_abs_state_um=160)
    assert got["reason"] == "scored"
    assert got["reported_label"] == "probe_fixed"


def test_extreme_int64_lattice_training_separation_does_not_overflow():
    low = ((np.iinfo(np.int64).min + 39) // 40) * 40
    high = (np.iinfo(np.int64).max // 40) * 40
    got = score(np.asarray([low, high], dtype=np.int64), residuals(80, 50, -2.0))
    assert got["reason"] == "scored"
    assert got["training_max_separation_um"] == high - low


def test_positive_finite_residual_extremes_produce_finite_log_score():
    tiny = np.nextafter(0.0, 1.0)
    huge = np.finfo(np.float64).max
    rows = [{"state_um": 80, "residual_probe": huge, "residual_tissue": tiny}] * 50
    got = score([0, 80], rows)
    assert np.isfinite(got["score"])
    assert got["score"] == pytest.approx(np.log(huge) - np.log(tiny))


def test_all_residual_rows_fail_closed_even_when_state_is_noncontributing():
    rows = residuals(80, 50, 0.0) + [{"state_um": 0, "residual_probe": np.nan, "residual_tissue": 1.0}]
    with pytest.raises(ValueError, match="finite"):
        score([0, 80], rows)


def test_member_local_group_evidence_never_propagates_or_merges():
    got = k7.annotate_candidate_group(
        [
            {"member_id": "classified", "reported_label": "probe_fixed"},
            {"member_id": "pulse", "reported_label": "undetermined"},
            {"member_id": "other", "reported_label": "tissue_fixed"},
        ],
        pulse_coincidence_pass_ids=["pulse"],
    )
    by_id = {row["member_id"]: row for row in got["members"]}
    assert by_id["classified"]["probe_fixed_evidence"] is True
    assert by_id["pulse"]["probe_fixed_evidence"] is True
    assert by_id["other"]["probe_fixed_evidence"] is False
    assert by_id["other"]["probe_fixed_candidate_group"] is True
    assert got["automatic_group_merge"] is False


@pytest.mark.parametrize("bad", [True, 1.5, 1 + 0j, "80", object(), 2**63])
def test_integer_and_range_inputs_fail_closed(bad):
    with pytest.raises(ValueError):
        k7.build_candidate_union([candidate("x", bad, 80, 0)], [])
    with pytest.raises(ValueError):
        k7.build_candidate_union([candidate("x", 1, bad, 0)], [])
    with pytest.raises(ValueError):
        k7.build_candidate_union([candidate("x", 1, 80, bad)], [])


@pytest.mark.parametrize("bad", [True, 0.0, -1.0, np.nan, np.inf, 1 + 0j, "1"])
def test_residual_dtypes_ranges_fail_closed(bad):
    with pytest.raises(ValueError):
        score([0, 80], [{"state_um": 80, "residual_probe": bad, "residual_tissue": 1.0}])


@pytest.mark.parametrize("bad", [True, np.nan, np.inf, 1 + 0j, "1"])
def test_threshold_dtypes_ranges_fail_closed(bad):
    with pytest.raises(ValueError):
        score([0, 80], residuals(80, 50, 0.0), probe_fixed_max_score=bad)


def test_lattice_and_group_identity_boundaries_fail_closed():
    with pytest.raises(ValueError, match="lattice"):
        k7.build_candidate_union([candidate("x", 1, 79, 0)], [])
    with pytest.raises(ValueError, match="lattice"):
        score([0, 80], [{"state_um": 79, "residual_probe": 1.0, "residual_tissue": 1.0}])
    with pytest.raises(ValueError, match="unique"):
        k7.annotate_candidate_group([
            {"member_id": "x", "reported_label": "undetermined"},
            {"member_id": "x", "reported_label": "probe_fixed"},
        ])
    with pytest.raises(ValueError, match="unknown"):
        k7.annotate_candidate_group(
            [{"member_id": "x", "reported_label": "undetermined"}],
            pulse_coincidence_pass_ids=["missing"],
        )
