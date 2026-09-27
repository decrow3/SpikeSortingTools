import numpy as np
import pytest

from testing.experimental.merge_map_identifiability import (
    infer_map_by_row_intersection,
    source_equivalent_aggregate,
)


def test_unique_ambiguous_absent_and_contradictory_intersections():
    original = np.array([
        [0, 1, -1],
        [0, 2, -1],
        [1, -1, -1],
        [2, -1, -1],
        [3, -1, -1],
        [4, -1, -1],
        [4, -1, -1],
    ])
    merged = np.array([
        [0, 1, -1],
        [0, 2, -1],
        [1, -1, -1],
        [2, -1, -1],
        [0, -1, -1],
        [0, -1, -1],
        [1, -1, -1],
    ])
    result = infer_map_by_row_intersection(
        original, merged, source_universe=np.arange(6)
    )
    assert result.resolved == {0: 0, 1: 1, 2: 2, 3: 0}
    assert result.ambiguous == {}
    assert result.absent == (5,)
    assert result.contradictory == (4,)

    symmetric = infer_map_by_row_intersection(
        np.array([[0, 1], [1, 0]]),
        np.array([[3, 4], [4, 3]]),
    )
    assert symmetric.resolved == {}
    assert symmetric.ambiguous == {0: (3, 4), 1: (3, 4)}


def test_logaddexp_duplicate_padding_noise_and_winner_change():
    candidates = np.array([
        [0, 1, 2, -1],
        [0, 1, 2, -1],
    ])
    log_liks = np.array([
        [2.0, 1.5, 1.4, -np.inf, 0.0],
        [1.2, 1.0, 0.5, -np.inf, 1.2],
    ], dtype=np.float64)
    responsibilities = np.array([
        [0.40, 0.25, 0.20, 0.00, 0.15],
        [0.30, 0.25, 0.20, 0.00, 0.25],
    ], dtype=np.float64)
    result = source_equivalent_aggregate(
        candidates, log_liks, responsibilities.copy(), np.array([0, 1, 1])
    )

    expected_group1 = np.logaddexp(1.5, 1.4)
    assert result.candidates[0].tolist() == [1, 0, -1, -1]
    assert result.log_liks[0, 0] == pytest.approx(expected_group1)
    assert result.responsibilities[0, 0] == pytest.approx(0.45)
    assert result.responsibilities[0, -1] == pytest.approx(0.15)
    assert result.labels[0] == 1  # aggregation, not map uncertainty, changes winner

    # Distinct source alternatives become exact duplicates after mapping and
    # are aggregated. The fixed map can therefore change the winner.
    assert result.candidates[1, 0] == 1
    assert result.log_liks[1, 0] == pytest.approx(np.logaddexp(1.0, 0.5))
    assert result.responsibilities[1, 0] == pytest.approx(0.45)
    assert result.responsibilities[1, -1] == pytest.approx(0.25)
    assert result.labels[1] == 1


def test_stable_score_ties_keep_original_candidate_order():
    result = source_equivalent_aggregate(
        np.array([[0, 1, -1]]),
        np.array([[2.0, 2.0, -np.inf, 2.0]]),
        np.array([[0.4, 0.3, 0.0, 0.3]]),
        np.array([0, 1, 1]),
    )
    assert result.candidates[0, :2].tolist() == [0, 1]
    assert result.labels.tolist() == [0]


def test_final_winner_majority_vote_does_not_identify_source_map():
    original = np.array([
        [0, 1, 2],
        [0, 1, 2],
        [0, 1, 2],
        [0, -1, -1],
    ])
    merged = np.array([
        [1, 0, -1],
        [1, 0, -1],
        [1, 0, -1],
        [0, -1, -1],
    ])
    inferred = infer_map_by_row_intersection(original, merged)
    assert inferred.resolved[0] == 0

    final_winners = merged[:, 0]
    rows_with_zero = (original == 0).any(axis=1)
    voted = np.bincount(final_winners[rows_with_zero]).argmax()
    assert voted == 1
    assert voted != inferred.resolved[0]


def test_noninteger_ids_and_uncovered_sources_fail_closed():
    with pytest.raises(ValueError, match="integer"):
        infer_map_by_row_intersection(np.array([[0.0]]), np.array([[0]]))
    with pytest.raises(ValueError, match="does not cover"):
        source_equivalent_aggregate(
            np.array([[2]]), np.array([[1.0, 0.0]]),
            np.array([[0.8, 0.2]]), np.array([0, 1]),
        )
    with pytest.raises(ValueError, match="source score order"):
        source_equivalent_aggregate(
            np.array([[0, 1, -1]]),
            np.array([[2.0, 1.0, -np.inf, 0.0]]),
            np.array([[0.2, 0.4, 0.0, 0.4]]),
            np.array([0, 1, 1]),
        )
