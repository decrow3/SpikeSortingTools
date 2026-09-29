import numpy as np

from testing.em2d_cross_arm_event_overlap import count_one_to_one, score_pairs, summarize


def arm(times, labels, channels=None):
    times = np.asarray(times, dtype=np.int64)
    return {
        "times": times,
        "labels": np.asarray(labels, dtype=np.int64),
        "channels": np.zeros_like(times) if channels is None else np.asarray(channels),
    }


def test_count_one_to_one_does_not_reuse_events():
    assert count_one_to_one(np.array([10, 11, 20]), np.array([10, 20]), 1) == 2
    assert count_one_to_one(np.array([10, 20]), np.array([12, 22]), 1) == 0
    assert count_one_to_one(np.array([10, 20]), np.array([12, 22]), 2) == 2


def test_reciprocal_best_preserves_ambiguous_and_unmatched_units():
    reference = arm([10, 20, 30, 100, 110, 120, 1000], [0, 0, 0, 1, 1, 1, 2])
    comparator = arm([10, 20, 30, 100, 110, 120, 2000], [4, 4, 4, 5, 5, 5, 6])
    scores = score_pairs(reference, comparator, tolerance=0, minimum=2)
    summary, best, reciprocal = summarize(scores, reference, comparator)
    assert set(zip(reciprocal.reference_unit, reciprocal.comparator_unit)) == {(0, 4), (1, 5)}
    assert summary["reference_units_without_candidate"] == 1
    assert summary["comparator_units_without_candidate"] == 1
    assert summary["reciprocal_pairs_f1_ge_0p8"] == 2
    assert not best.ambiguous.any()
