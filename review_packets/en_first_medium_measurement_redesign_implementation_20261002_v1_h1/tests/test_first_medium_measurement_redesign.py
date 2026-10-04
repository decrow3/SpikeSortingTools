import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from testing.first_medium_measurement_redesign import (
    MatchRule,
    assess_margin_rules,
    conditional_amplitude_diagnostic,
    half_open_cell_indices,
    reference_event_retention,
    retention_difference,
    validate_half_open_cells,
)


def sort(times, clusters):
    order = np.argsort(times, kind="stable")
    return {
        "st": np.asarray(times, dtype=np.int64)[order],
        "cl": np.asarray(clusters, dtype=np.int64)[order],
    }


RULE = MatchRule(tolerance_frames=0, minimum_overlap=0.1, primary_retention=0.5)


def test_reference_self_is_one_and_self_difference_is_zero():
    ref = sort([10, 20, 30, 110, 120, 130], [1, 1, 1, 2, 2, 2])
    result = reference_event_retention(ref, ref, [1, 2], RULE)
    assert result["R"] == 1.0
    assert result["unit_retention"].retention.tolist() == [1.0, 1.0]
    delta = retention_difference(
        ref, ref, ref, [1, 2], RULE,
        bootstrap_replicates=200, bootstrap_seed=7,
    )
    assert delta["DeltaR"] == 0.0
    assert delta["bootstrap"]["one_sided_lower_95"] == 0.0
    assert delta["proposal_rule"]["passes"] is False
    assert delta["frozen_replacement_rule"]["passes"] is False


def test_consistent_bijective_candidate_relabel_is_invariant():
    ref = sort([10, 20, 30, 110, 120, 130], [1, 1, 1, 2, 2, 2])
    original = sort([10, 20, 30, 110, 120, 130], [11, 11, 11, 12, 12, 12])
    permuted = sort([10, 20, 30, 110, 120, 130], [912, 912, 912, 701, 701, 701])
    a = reference_event_retention(ref, original, [1, 2], RULE)
    b = reference_event_retention(ref, permuted, [1, 2], RULE)
    assert a["R"] == b["R"] == 1.0
    assert a["unit_retention"].retention.tolist() == b["unit_retention"].retention.tolist()


def test_duplicates_receive_no_double_credit_and_deletions_reduce_retention():
    ref = sort([10, 20, 30, 40], [1, 1, 1, 1])
    duplicated = sort([10, 10, 20, 20, 30, 30, 40, 40], [8] * 8)
    result = reference_event_retention(ref, duplicated, [1], RULE)
    assert result["R"] == 1.0
    assert result["credited_candidate_events"] == 4
    assert result["unit_retention"].matched_reference_events.item() == 4
    deleted = reference_event_retention(ref, sort([10, 20], [8, 8]), [1], RULE)
    assert deleted["R"] == 0.5


def test_tied_split_and_merge_are_unmatched_zero():
    one = sort([10, 20, 30, 40], [1, 1, 1, 1])
    split = sort([10, 20, 30, 40], [8, 8, 9, 9])
    split_result = reference_event_retention(one, split, [1], RULE)
    assert split_result["R"] == 0.0
    assert split_result["unit_retention"].status.item() == "unmatched_zero"

    ref = sort([10, 20, 30, 40], [1, 1, 2, 2])
    merged = sort([10, 20, 30, 40], [8, 8, 8, 8])
    merge_result = reference_event_retention(ref, merged, [1, 2], RULE)
    assert merge_result["R"] == 0.0
    assert merge_result["unit_retention"].retention.tolist() == [0.0, 0.0]


def test_absent_partner_contributes_exact_zero():
    ref = sort([10, 20, 110, 120], [1, 1, 2, 2])
    candidate = sort([10, 20], [11, 11])
    result = reference_event_retention(ref, candidate, [1, 2], RULE)
    assert result["R"] == 0.5
    rows = result["unit_retention"].set_index("reference_unit")
    assert rows.loc[1, "retention"] == 1.0
    assert rows.loc[2, "retention"] == 0.0
    assert rows.loc[2, "status"] == "unmatched_zero"


def test_half_open_cell_boundaries_and_diagnostics():
    edges = validate_half_open_cells([0, 60, 120], recording_stop_frame=120)
    assert half_open_cell_indices([0, 59, 60, 119], edges).tolist() == [0, 0, 1, 1]
    with pytest.raises(ValueError, match="outside"):
        half_open_cell_indices([120], edges)
    ref = sort([0, 59, 60, 119], [1, 1, 1, 1])
    result = reference_event_retention(ref, ref, [1], RULE, cell_edges_frames=edges)
    cells = result["cell_diagnostics"]
    assert cells.reference_events.tolist() == [2, 2]
    assert cells.matched_reference_events.tolist() == [2, 2]
    assert set(cells.scope) == {"state_or_longitudinal_diagnostic_only"}


def test_conditional_amplitude_exact_support_union_and_unchanged_fit_gate():
    ref = pd.DataFrame({
        "cluster_id": [1, 1, 1, 2],
        "status": ["finite_interior", "finite_interior", "boundary_pinned", "no_fit"],
        "start_s": [0.0, 10.0, 20.0, np.nan],
        "end_s": [10.0, 20.0, 30.0, np.nan],
        "missing_pct": [20.0, 10.0, 50.0, np.nan],
    })
    candidate = pd.DataFrame({
        "cluster_id": [11, 11, 11],
        "status": ["finite_interior", "finite_interior", "boundary_pinned"],
        "start_s": [5.0, 15.0, 0.0],
        "end_s": [15.0, 25.0, 30.0],
        "missing_pct": [5.0, 5.0, 50.0],
    })
    primary = pd.DataFrame({"baseline_cluster": [1], "candidate_cluster": [11]})
    result = conditional_amplitude_diagnostic(
        ref, candidate, primary, [1, 2], duration_s=30.0,
        minimum_valid_windows=2, minimum_common_time_fraction=0.5,
    )
    rows = result["unit_support"].set_index("reference_unit")
    assert rows.loc[1, "reference_support_union_s"] == 20.0
    assert rows.loc[1, "candidate_support_union_s"] == 20.0
    assert rows.loc[1, "common_time_s"] == 15.0
    assert rows.loc[1, "supported"]
    assert rows.loc[1, "reference_minus_candidate_missingness_pp"] == pytest.approx(125 / 15)
    assert rows.loc[2, "status"] == "unmatched"
    assert result["supported_unit_fraction_full_cohort"] == 0.5
    assert result["reference_support_union_fraction_full_cohort_time"] == pytest.approx(1 / 3)
    assert result["common_time_fraction_full_cohort_time"] == 0.25
    assert "no refit" in result["fit_quality_rule"]


def test_margin_assessment_is_visible_and_replacement_targets_the_margin():
    ref = sort([10, 20, 30, 40], [1, 1, 1, 1])
    existing = sort([10, 20], [11, 11])
    repaired = ref
    result = retention_difference(
        ref, existing, repaired, [1], RULE,
        bootstrap_replicates=100, bootstrap_seed=9, minimum_effect=0.05,
    )
    assert result["DeltaR"] == 0.5
    assert result["proposal_rule"]["suitability"] == "rejected_as_minimum_effect_evidence"
    assert result["frozen_replacement_rule"]["definition"].endswith("> minimum_effect")
    assert result["frozen_replacement_rule"]["passes"] is True
    near_margin = assess_margin_rules(0.051, 0.001, minimum_effect=0.05)
    assert near_margin["proposal_passes"] is True
    assert near_margin["replacement_passes"] is False


def test_frozen_contract_is_execution_disabled_and_outcome_blind():
    path = Path("configs/en_first_medium_measurement_redesign_v1.json")
    contract = json.loads(path.read_text())
    assert contract["execution_enabled"] is False
    assert contract["outcome_blind"] is True
    assert contract["conditional_amplitude"]["refit_allowed"] is False
    assert contract["conditional_amplitude"]["threshold_relaxation_allowed"] is False
    assert contract["margin_review"]["frozen_replacement_rule"].endswith("> 0.05")
