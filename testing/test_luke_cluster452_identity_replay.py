import numpy as np

from testing.luke_cluster452_identity_replay import (
    exclusive_pairs,
    match_summary,
    refractory_fraction,
    select_anchor,
    template_cosine,
)


def test_exclusive_pairs_never_reuses_events_and_is_symmetric():
    first = np.array([10, 11, 30])
    second = np.array([10, 29])
    ai, bi = exclusive_pairs(first, second, 1)
    bj, aj = exclusive_pairs(second, first, 1)
    assert list(zip(ai, bi)) == [(0, 0), (2, 1)]
    assert list(zip(aj, bj)) == [(0, 0), (2, 1)]


def test_match_summary_keeps_unmatched_denominator_and_label_identity():
    result = match_summary(
        np.array([10, 20, 30, 40]), np.array([9, 21, 80]), np.array([4, 5, 6]), 1
    )
    assert result["matched_events"] == 2
    assert result["matched_fraction"] == 0.5
    assert result["label_fractions"] == {"4": 0.25, "5": 0.25}


def test_template_cosine_recovers_small_lag():
    reference = np.zeros((9, 2)); reference[3, 0] = -2; reference[4, 1] = 1
    shifted = np.zeros_like(reference); shifted[5, 0] = -2; shifted[6, 1] = 1
    cosine, lag = template_cosine(reference, shifted)
    assert np.isclose(cosine, 1)
    assert lag == 2


def test_refractory_fraction_preserves_duplicate_timestamps():
    assert refractory_fraction(np.array([0, 0, 100]), 10) == 0.5


def test_anchor_selection_excludes_tiny_or_wrong_target_fraction_winners():
    rows = [
        {"legacy_cluster": 1, "reference_anchor_events": 3,
         "reference_matched_fraction": 1.0, "reference_target_fraction": 0.0,
         "excess_over_null": 1.0, "target_is_dominant": False},
        {"legacy_cluster": 2, "reference_anchor_events": 1000,
         "reference_matched_fraction": 0.9, "reference_target_fraction": 0.7,
         "excess_over_null": 0.6, "target_is_dominant": True},
    ]
    gates = {"minimum_reference_events": 20, "minimum_observed_fraction": 0.25,
             "minimum_excess_over_shift_null": 0.10,
             "target_must_be_dominant_rescue_label": True}
    selected, passed = select_anchor(rows, gates)
    assert passed
    assert selected["legacy_cluster"] == 2
