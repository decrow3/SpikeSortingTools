import numpy as np

from testing.luke_imec0_similar_pair_audit import nearest_distance_frames


def test_nearest_distance_frames_handles_edges_and_ties():
    values = np.array([0, 5, 10, 20])
    reference = np.array([3, 8, 30])
    assert nearest_distance_frames(values, reference).tolist() == [3, 2, 2, 10]


def test_nearest_distance_frames_handles_empty_reference():
    distances = nearest_distance_frames(np.array([1, 2]), np.array([], dtype=int))
    assert np.all(distances == np.iinfo(np.int64).max)
