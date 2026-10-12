import numpy as np
import pytest

from testing.joint_model_k7_frame_score import (
    annotate_candidate_group,
    build_candidate_union,
    score_frame_residuals,
)


def candidate(event_id, sample, state, block):
    return {"event_id": event_id, "sample_index": sample, "state_um": state, "block_id": block}


def residual_rows(state, count, score):
    ratio = float(np.exp(score))
    return [
        {"state_um": state, "residual_probe": ratio, "residual_tissue": 1.0}
        for _ in range(count)
    ]


def test_candidate_union_keeps_matched_filter_only_events_and_stable_sources():
    got = build_candidate_union(
        [candidate("both", 10, -80, 0), candidate("unit", 20, -80, 0)],
        [candidate("both", 10, -80, 0), candidate("mf", 30, -80, 0)],
    )
    assert [row["event_id"] for row in got] == ["both", "unit", "mf"]
    assert got[0]["sources"] == ["unit", "matched_filter"]
    assert got[2]["sources"] == ["matched_filter"]


def test_candidate_union_rejects_conflicting_duplicate_metadata():
    with pytest.raises(ValueError, match="conflicting"):
        build_candidate_union(
            [candidate("same", 10, -80, 0)],
            [candidate("same", 11, -80, 0)],
        )


def test_state_balancing_uses_median_of_state_medians_not_event_weighting():
    rows = residual_rows(-80, 50, -2.0) + residual_rows(-160, 500, 1.0)
    result = score_frame_residuals(
        np.asarray([0, -80], np.int64), rows,
        probe_fixed_max_score=-0.6, tissue_fixed_min_score=0.6,
    )
    assert result["score"] == pytest.approx(-0.5)
    assert result["reported_label"] == "undetermined"
    assert result["state_event_counts"] == {-160: 500, -80: 50}


def test_single_state_training_abstains_despite_extreme_probe_preference():
    result = score_frame_residuals(
        np.asarray([-80] * 100, np.int64), residual_rows(-80, 100, -5.0),
        probe_fixed_max_score=-0.5, tissue_fixed_min_score=0.5,
    )
    assert result["reported_label"] == "undetermined"
    assert result["operational_frame"] == "tissue"
    assert result["score"] is None
    assert result["reason"] == "training_state_support"


def test_informative_state_and_minimum_count_gates():
    rows = residual_rows(-40, 100, -3.0) + residual_rows(-80, 49, -3.0)
    result = score_frame_residuals(
        np.asarray([0, -80], np.int64), rows,
        probe_fixed_max_score=-0.5, tissue_fixed_min_score=0.5,
    )
    assert result["reported_label"] == "undetermined"
    assert result["reason"] == "held_out_state_support"
    assert result["state_event_counts"] == {-80: 49}


@pytest.mark.parametrize(
    "score,expected",
    [(-1.0, "probe_fixed"), (0.0, "undetermined"), (1.0, "tissue_fixed")],
)
def test_explicit_external_thresholds_classify_expected_direction(score, expected):
    result = score_frame_residuals(
        np.asarray([0, -80], np.int64), residual_rows(-80, 50, score),
        probe_fixed_max_score=-0.5, tissue_fixed_min_score=0.5,
    )
    assert result["reported_label"] == expected


def test_mixed_group_keeps_member_level_labels_and_never_auto_merges():
    result = annotate_candidate_group([
        {"member_id": "probe", "reported_label": "probe_fixed"},
        {"member_id": "unknown", "reported_label": "undetermined"},
        {"member_id": "tissue", "reported_label": "tissue_fixed"},
    ])
    assert result["any_probe_fixed_evidence"] is True
    assert result["automatic_group_merge"] is False
    by_id = {row["member_id"]: row for row in result["members"]}
    assert by_id["probe"]["probe_fixed_evidence"] is True
    assert by_id["unknown"]["reported_label"] == "undetermined"
    assert by_id["unknown"]["probe_fixed_candidate_group"] is True
    assert by_id["tissue"]["reported_label"] == "tissue_fixed"


def test_independent_pulse_evidence_is_member_local():
    result = annotate_candidate_group(
        [
            {"member_id": "pulse", "reported_label": "undetermined"},
            {"member_id": "other", "reported_label": "undetermined"},
        ],
        pulse_coincidence_pass_ids=["pulse"],
    )
    assert result["members"][0]["probe_fixed_evidence"] is True
    assert result["members"][1]["probe_fixed_evidence"] is False
    assert result["automatic_group_merge"] is False


@pytest.mark.parametrize("bad", [0.0, -1.0, np.inf, np.nan])
def test_residuals_must_be_finite_and_positive(bad):
    with pytest.raises(ValueError, match="strictly positive|finite real"):
        score_frame_residuals(
            np.asarray([0, -80], np.int64),
            [{"state_um": -80, "residual_probe": bad, "residual_tissue": 1.0}],
            probe_fixed_max_score=-0.5, tissue_fixed_min_score=0.5,
        )


@pytest.mark.parametrize("bad", [True, 1.5, 1 + 0j, "10", object()])
def test_candidate_clocks_fail_closed(bad):
    with pytest.raises(ValueError, match="sample_index"):
        build_candidate_union([candidate("x", bad, -80, 0)], [])


def test_invalid_threshold_order_and_unknown_pulse_member_fail_closed():
    with pytest.raises(ValueError, match="below"):
        score_frame_residuals(
            np.asarray([0, -80], np.int64), residual_rows(-80, 50, 0),
            probe_fixed_max_score=0.5, tissue_fixed_min_score=-0.5,
        )
    with pytest.raises(ValueError, match="unknown"):
        annotate_candidate_group(
            [{"member_id": "x", "reported_label": "undetermined"}],
            pulse_coincidence_pass_ids=["missing"],
        )


@pytest.mark.parametrize("states", [[0, -60], [20, -80]])
def test_non_lattice_training_states_fail_closed(states):
    with pytest.raises(ValueError, match="40-um lattice"):
        score_frame_residuals(
            np.asarray(states, np.int64), residual_rows(-80, 50, 0),
            probe_fixed_max_score=-0.5, tissue_fixed_min_score=0.5,
        )


@pytest.mark.parametrize("bad", [True, 80.5, 1 + 0j, "80"])
def test_support_parameters_require_exact_integers(bad):
    with pytest.raises(ValueError, match="exact integer"):
        score_frame_residuals(
            np.asarray([0, -80], np.int64), residual_rows(-80, 50, 0),
            probe_fixed_max_score=-0.5, tissue_fixed_min_score=0.5,
            informative_abs_state_um=bad,
        )
