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
