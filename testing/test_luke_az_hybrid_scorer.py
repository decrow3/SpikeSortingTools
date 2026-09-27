import json

import numpy as np

from testing.luke_az_hybrid_scorer import (
    associate_full_train,
    circular_shift_chance_controls,
    score_with_frozen_association,
    tolerance_samples,
)


def test_actual_frequency_tolerance_is_twelve_samples():
    assert tolerance_samples(29999.759166666667) == 12


def test_full_train_association_precedes_region_scoring_and_ties_are_fixed():
    truth = np.array([100, 200, 300, 400])
    donors = np.array([7, 7, 7, 7])
    output = np.array([101, 201, 301, 401, 900])
    labels = np.array([4, 4, 5, 5, 4])
    association = associate_full_train(truth, donors, output, labels, 12)
    row = association["associations"][0]
    assert row["primary_label"] == 5  # same TP, higher precision than label 4
    assert row["tp_margin"] == 0

    result = score_with_frozen_association(
        truth,
        donors,
        np.array([False, False, True, True]),
        output,
        labels,
        np.array([False, False, True, True, False]),
        association["associations"],
        12,
    )
    by_region = {r["region"]: r for r in result["scores"]}
    assert by_region["episode"]["recall"] == 1.0
    assert by_region["rest"]["recall"] == 0.0


def test_global_exclusivity_and_preexclusive_duplicates_are_separate():
    truth = np.array([100, 110])
    donors = np.array([1, 1])
    output = np.array([105, 106])
    labels = np.array([9, 9])
    association = associate_full_train(truth, donors, output, labels, 12)
    result = score_with_frozen_association(
        truth,
        donors,
        np.array([False, False]),
        output,
        labels,
        np.array([False, False]),
        association["associations"],
        12,
    )
    assert len(result["matches"]) == 2
    assert result["scores"][1]["preexclusive_duplicate_candidates"] == 2
    assert len(result["preexclusive_candidates"]) == 2


def test_false_merge_candidate_is_reported_not_resolved():
    truth = np.array([100, 200, 300, 400])
    donors = np.array([1, 1, 2, 2])
    output = np.array([100, 200, 300, 400])
    labels = np.array([8, 8, 8, 8])
    association = associate_full_train(truth, donors, output, labels, 0)
    assert association["false_merge_candidates"] == [
        {"label": 8, "donor_ids": [1, 2], "donor_count": 2}
    ]


def test_chance_controls_are_seeded_and_preserve_donor_counts():
    truth = np.array([100, 200, 300, 400])
    donors = np.array([1, 1, 2, 2])
    output = truth.copy()
    labels = np.array([10, 10, 20, 20])
    a = circular_shift_chance_controls(
        truth,
        donors,
        output,
        labels,
        tolerance=12,
        start_sample=0,
        end_sample=1000,
        shifts=4,
    )
    b = circular_shift_chance_controls(
        truth,
        donors,
        output,
        labels,
        tolerance=12,
        start_sample=0,
        end_sample=1000,
        shifts=4,
    )
    assert a == b
    assert len(a) == 8


def test_empty_output_retains_every_truth_donor_as_unmatched():
    truth = np.array([100, 200, 300, 400])
    donors = np.array([1, 1, 2, 2])
    output = np.array([], dtype=np.int64)
    labels = np.array([], dtype=np.int64)
    association = associate_full_train(truth, donors, output, labels, 12)
    assert association["pair_table"] == []
    assert association["false_merge_candidates"] == []
    assert [row["donor_id"] for row in association["associations"]] == [1, 2]
    assert all(row["association_status"] == "unmatched" for row in association["associations"])
    assert all(row["primary_label"] is None for row in association["associations"])
    assert all(row["full_train_recall"] == 0.0 for row in association["associations"])
    assert all(row["full_train_precision"] is None for row in association["associations"])

    scored = score_with_frozen_association(
        truth,
        donors,
        np.array([False, True, False, True]),
        output,
        labels,
        np.array([], dtype=bool),
        association["associations"],
        12,
    )
    assert scored["matches"] == []
    assert all(row["tp"] == 0 and row["fp"] == 0 for row in scored["scores"])
    assert all(row["fn"] == row["truth_count"] for row in scored["scores"])
    assert all(row["precision"] is None for row in scored["scores"])
    json.dumps({"association": association, "scored": scored}, allow_nan=False)


def test_existing_labels_with_all_zero_tp_do_not_create_primary_label():
    association = associate_full_train(
        np.array([100, 200]),
        np.array([7, 7]),
        np.array([500, 600]),
        np.array([3, 4]),
        12,
    )
    row = association["associations"][0]
    assert row["primary_label"] is None
    assert row["runner_up_label"] is None
    assert row["tp_margin"] is None
    assert row["exact_rank_tie"] is False
    assert association["false_merge_candidates"] == []
    assert len(association["pair_table"]) == 2


def test_mixed_matched_and_unmatched_donors_score_without_fabricated_link():
    truth = np.array([100, 200, 300, 400])
    donors = np.array([1, 1, 2, 2])
    output = np.array([101, 201, 900])
    labels = np.array([8, 8, 9])
    association = associate_full_train(truth, donors, output, labels, 12)
    by_donor = {row["donor_id"]: row for row in association["associations"]}
    assert by_donor[1]["primary_label"] == 8
    assert by_donor[2]["primary_label"] is None
    assert association["false_merge_candidates"] == []

    scored = score_with_frozen_association(
        truth,
        donors,
        np.array([False, True, False, True]),
        output,
        labels,
        np.array([False, True, False]),
        association["associations"],
        12,
    )
    rows = {(row["donor_id"], row["region"]): row for row in scored["scores"]}
    assert rows[(1, "rest")]["recall"] == 1.0
    assert rows[(1, "episode")]["recall"] == 1.0
    assert rows[(2, "rest")]["recall"] == 0.0
    assert rows[(2, "episode")]["recall"] == 0.0
    assert rows[(2, "rest")]["primary_label"] is None
    assert rows[(2, "rest")]["precision"] is None


def test_multiple_unmatched_donors_remain_separate_in_chance_control():
    truth = np.array([100, 200, 300])
    donors = np.array([4, 5, 6])
    controls = circular_shift_chance_controls(
        truth,
        donors,
        np.array([], dtype=np.int64),
        np.array([], dtype=np.int64),
        tolerance=12,
        start_sample=0,
        end_sample=1000,
        shifts=3,
    )
    assert len(controls) == 9
    assert {row["donor_id"] for row in controls} == {4, 5, 6}
    assert all(row["best_label"] is None for row in controls)
    assert all(row["exclusive_tp"] == 0 and row["recall"] == 0.0 for row in controls)
