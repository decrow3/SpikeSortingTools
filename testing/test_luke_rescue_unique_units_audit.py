import numpy as np

from testing.luke_rescue_unique_units_audit import (
    exclusive_event_pairs,
    mutual_best_matches,
    spatial_null_distribution,
)


def _sort(st, cl, good):
    return {
        "st": np.asarray(st, dtype=np.int64),
        "cl": np.asarray(cl, dtype=np.int64),
        "label": {int(c): "good" for c in good},
        "good": set(good),
    }


def test_exclusive_event_pairs_cannot_reuse_one_target_event():
    ai, bi = exclusive_event_pairs([100, 105], [103], tolerance=5)
    assert len(ai) == len(bi) == 1
    assert len(np.unique(bi)) == 1


def test_exclusive_event_pairs_are_direction_stable():
    a = np.array([100, 105, 300, 500])
    b = np.array([103, 299, 504])
    ai, bi = exclusive_event_pairs(a, b, tolerance=5)
    bj, aj = exclusive_event_pairs(b, a, tolerance=5)
    assert {(int(a[i]), int(b[j])) for i, j in zip(ai, bi)} == {
        (int(a[j]), int(b[i])) for i, j in zip(bj, aj)
    }


def test_mutual_best_matching_is_symmetric_with_duplicate_nearby_events():
    a = _sort([100, 105, 300], [0, 0, 1], {0, 1})
    b = _sort([103, 300], [4, 5], {4, 5})
    ab = mutual_best_matches(a, b)
    ba = mutual_best_matches(b, a)
    forward = set(zip(ab.rescue_cluster, ab.legacy_cluster))
    reverse = {(b, a) for a, b in zip(ba.rescue_cluster, ba.legacy_cluster)}
    assert forward == reverse


def test_spatial_null_evidence_requires_depth_and_excess_over_shifted_null():
    anchor = np.array([1_000, 5_000, 11_000, 19_000], dtype=np.int64)
    target = {
        "st": np.array([1_001, 5_001, 11_001, 19_001], dtype=np.int64),
        "cl": np.array([4, 4, 4, 4], dtype=np.int64),
        "label": {4: "good"},
    }
    observed, ranked, evidence = spatial_null_distribution(
        anchor, 100.0, target, {4: 110.0}, fs=1_000.0, duration_s=2_000.0
    )
    assert observed == 1.0
    assert ranked[0][:2] == (4, 1.0)
    assert evidence["null_median_fraction"] == 0.0
    assert evidence["shared_detection_supported"]

    _, ranked_far, evidence_far = spatial_null_distribution(
        anchor, 100.0, target, {4: 400.0}, fs=1_000.0, duration_s=2_000.0
    )
    assert ranked_far == []
    assert not evidence_far["shared_detection_supported"]
