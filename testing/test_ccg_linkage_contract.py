import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from testing.experimental.ccg_linkage_contract import (
    all_pair_lags,
    deduplicate_within_labels,
    histogram_half_open_last_closed,
    shifted_segment_null,
    single_link_components,
    union_route_mask,
)


def test_all_pairs_are_oriented_inclusive_and_segment_local():
    segments = np.array([[0, 25], [100, 125]])
    a = np.array([10, 20, 24])
    b = np.array([12, 22, 100])
    lags = np.sort(all_pair_lags(a, b, segments=segments, max_abs_lag=10))
    assert lags.tolist() == [-8, -2, 2, 2]
    reverse = np.sort(all_pair_lags(b, a, segments=segments, max_abs_lag=10))
    assert np.array_equal(reverse, np.sort(-lags))

    # The event at 100 is in the second segment and cannot pair with 24.
    assert 76 not in lags
    assert all_pair_lags([], b, segments=segments, max_abs_lag=10).size == 0


def test_histogram_boundaries_are_explicit():
    lags = np.array([-2, -1, 0, 1, 2])
    counts = histogram_half_open_last_closed(lags, np.array([-2, 0, 2]))
    assert counts.tolist() == [2, 3]  # -2,-1 | 0,1,2(final right included)


def test_shift_null_uses_common_support_without_wrap_and_equal_exposure():
    result = shifted_segment_null(
        np.array([15, 25, 115, 125]),
        np.array([15, 25, 115, 125]),
        segments=np.array([[0, 40], [100, 140]]),
        offsets=np.array([-10, 0, 10]),
        lags=np.array([-5, 0, 5]),
    )
    assert result.common_support_samples == 40
    assert np.all(result.exposure_samples == np.array([30, 40, 30]))
    assert result.counts.shape == (3, 3)
    assert result.counts[1, 1] == 4
    assert result.counts[0, 1] == 2
    assert result.counts[2, 1] == 2


def test_censoring_changes_post_partition_pairs_and_keeps_high_score():
    times = np.array([100, 106, 200, 206])
    scores = np.array([1.0, 3.0, 4.0, 2.0])
    one_label = np.zeros(4, dtype=int)
    split_labels = np.array([0, 1, 0, 1])

    keep_merged = deduplicate_within_labels(
        times, one_label, scores, radius_samples=7
    )
    keep_split = deduplicate_within_labels(
        times, split_labels, scores, radius_samples=7
    )
    assert times[keep_merged].tolist() == [106, 200]
    assert keep_split.all()

    pre = all_pair_lags(
        times[split_labels == 0], times[split_labels == 1],
        segments=np.array([[0, 300]]), max_abs_lag=10,
    )
    post = all_pair_lags(
        times[keep_merged & (split_labels == 0)],
        times[keep_merged & (split_labels == 1)],
        segments=np.array([[0, 300]]), max_abs_lag=10,
    )
    assert pre.tolist() == [6, 6]
    assert post.size == 0


def test_ccg_shape_is_not_an_identity_classifier():
    # Distinct synthetic sources share a driver with a fixed delay: no zero-lag
    # coincidences and a narrow +10 peak do not make them the same neuron.
    a = np.arange(100, 601, 100)
    b = a + 10
    distinct = all_pair_lags(a, b, segments=np.array([[0, 700]]), max_abs_lag=20)
    assert np.count_nonzero(distinct == 0) == 0
    assert np.count_nonzero(distinct == 10) == a.size

    # One refractory train split into alternating labels has cross-label peaks.
    one_train = np.arange(100, 901, 100)
    left, right = one_train[::2], one_train[1::2]
    partition = all_pair_lags(
        left, right, segments=np.array([[0, 1000]]), max_abs_lag=150
    )
    assert np.count_nonzero(partition == 100) == right.size
    assert np.count_nonzero(partition == -100) == right.size


def test_single_link_transitivity_and_force_union_bypass_pair_guards():
    qda = np.eye(3, dtype=bool)
    qda[1, 2] = qda[2, 1] = True
    force = np.eye(3, dtype=bool)
    force[0, 1] = force[1, 0] = True
    final = union_route_mask(qda, force)
    assert not final[0, 2]
    assert single_link_components(final).tolist() == [0, 0, 0]

    distances = np.logical_not(final).astype(float)
    single = fcluster(linkage(squareform(distances), method="single"), 0.5, criterion="distance")
    complete = fcluster(linkage(squareform(distances), method="complete"), 0.5, criterion="distance")
    assert np.unique(single).size == 1
    assert np.unique(complete).size == 2

    # The 0-1 edge can enter through force even if it was absent from QDA's
    # distance/correlation/coverage/score-qualified route.
    assert force[0, 1] and not qda[0, 1]


def test_optional_route_is_an_independent_union_path():
    qda = np.eye(2, dtype=bool)
    force = np.eye(2, dtype=bool)
    optional = np.array([[1, 1], [1, 1]], dtype=bool)
    final = union_route_mask(qda, force, optional)
    assert single_link_components(final).tolist() == [0, 0]
