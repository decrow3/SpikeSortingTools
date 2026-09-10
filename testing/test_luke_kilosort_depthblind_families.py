import numpy as np

from testing.luke_kilosort_depthblind_families_v2 import chronological_segments, threshold_components


def test_threshold_components_allow_multi_member_chain():
    similarity = np.eye(5, dtype=float)
    similarity[0, 1] = similarity[1, 0] = 0.97
    similarity[1, 2] = similarity[2, 1] = 0.96
    similarity[0, 2] = similarity[2, 0] = 0.80
    similarity[3, 4] = similarity[4, 3] = 0.949

    labels, sizes = threshold_components(similarity, 0.95)

    assert labels[0] == labels[1] == labels[2]
    assert labels[3] != labels[4]
    assert sorted(sizes.tolist()) == [1, 1, 3]


def test_chronological_segments_break_only_at_long_gaps():
    times = np.array([0.0, 0.1, 2.1, 4.2, 4.3])
    segments = chronological_segments(times, max_gap_s=2.0)
    assert [segment.tolist() for segment in segments] == [[0, 1, 2], [3, 4]]
