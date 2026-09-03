import numpy as np

from testing.luke_rescue_stitch_fullsession_eval import _apply_remap


def _sort(clusters, times, labels):
    return {
        "st": np.asarray(times, dtype=np.int64),
        "cl": np.asarray(clusters, dtype=np.int64),
        "label": dict(labels),
        "good": {c for c, v in labels.items() if v == "good"},
    }


def test_apply_remap_folds_family_into_largest_good_member():
    # cluster 5 (good, 3 spikes) + cluster 9 (mua, 1 spike) are a family;
    # cluster 7 (good) stands alone.
    sort = _sort(
        clusters=[5, 5, 5, 9, 7, 7],
        times=[10, 40, 70, 20, 30, 60],
        labels={5: "good", 7: "good", 9: "mua"},
    )
    new, remap = _apply_remap(sort, [[5, 9]])

    assert remap == {9: 5}
    assert 9 not in new["label"]
    assert new["good"] == {5, 7}
    # spike vector is time-sorted and 9 -> 5
    assert list(new["st"]) == [10, 20, 30, 40, 60, 70]
    assert list(new["cl"]) == [5, 5, 7, 5, 7, 5]


def test_apply_remap_keeps_higher_count_good_member_as_survivor():
    sort = _sort(
        clusters=[1, 1, 2, 2, 2],
        times=[5, 6, 1, 2, 3],
        labels={1: "good", 2: "good"},
    )
    new, remap = _apply_remap(sort, [[1, 2]])
    assert remap == {1: 2}  # 2 has more spikes
    assert new["good"] == {2}
