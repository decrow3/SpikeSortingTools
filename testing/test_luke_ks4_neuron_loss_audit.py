import numpy as np
import pandas as pd

from testing.luke_ks4_neuron_loss_audit import (
    cohort_summary,
    nearest_local_matches,
    reviewed_mua_candidates,
    stage_outcome,
)


def test_nearest_local_matches_obeys_time_depth_and_label_filters():
    present, labels = nearest_local_matches(
        np.array([100, 200, 300]),
        np.array([20.0, 100.0, 40.0]),
        np.array([98, 102, 201, 299]),
        np.array([1, 2, 3, 4]),
        target_depths=np.array([22.0, 60.0, 102.0, 42.0]),
        time_radius=3,
        depth_radius_um=10.0,
        allowed_labels={1, 3},
    )
    assert present.tolist() == [True, True, False]
    assert labels.tolist() == [1, 3, -1]


def test_nearest_local_matches_supports_label_depth_lookup():
    present, labels = nearest_local_matches(
        np.array([10]),
        np.array([50.0]),
        np.array([9, 11]),
        np.array([0, 1]),
        label_depths=np.array([100.0, 51.0]),
        time_radius=2,
        depth_radius_um=5.0,
    )
    assert present.tolist() == [True]
    assert labels.tolist() == [1]


def test_stage_outcome_prefers_final_table_over_learned_proxy():
    assert stage_outcome(False, True, True) == "retained_in_good_unit"
    assert stage_outcome(False, True, False) == "retained_in_mua_only"
    assert (
        stage_outcome(True, False, False)
        == "lost_in_duplicate_removal_or_finalization"
    )
    assert stage_outcome(False, False, False) == "absent_from_learned_and_final_tables"


def test_cohort_summary_decomposes_measured_exclusions():
    traces = pd.DataFrame(
        {
            "cohort": ["reviewed"] * 4,
            "learned_present": [True, True, True, False],
            "final_present": [True, True, False, False],
            "good_present": [True, False, False, False],
        }
    )
    row = cohort_summary(traces).iloc[0]
    assert row.n_events == 4
    assert row.final_present == 2
    assert row.learned_but_not_final == 1
    assert row.final_mua_only == 1


def test_reviewed_mua_candidates_requires_supported_clean_unit():
    traces = pd.DataFrame(
        {
            "cohort": ["reviewed_neural_discovery"] * 4,
            "final_present": [True] * 4,
            "good_present": [False] * 4,
            "final_unit": [7, 7, 8, 8],
        }
    )
    metrics = pd.DataFrame(
        {
            "unit_id": [7, 8],
            "ks_good": [False, False],
            "contamination_pct": [2.0, 25.0],
            "refractory_violation_fraction_1p5ms": [0.005, 0.02],
            "spike_count": [200, 300],
            "presence_fraction_300s": [0.8, 1.0],
        }
    ).set_index("unit_id")
    result = reviewed_mua_candidates(traces, metrics).set_index("unit_id")
    assert bool(result.loc[7, "bounded_review_candidate"])
    assert not bool(result.loc[8, "bounded_review_candidate"])
