import numpy as np

from testing.luke_yates_detection_stage_audit import (
    choose_batches,
    collapse_candidates,
    extrema_candidates,
    probe_depth_exposure_mm,
    spatial_neighbors,
)


def test_choose_batches_is_bounded_and_evenly_spaced():
    batches = choose_batches(100, 6, False)
    assert batches.tolist() == [0, 19, 39, 59, 79, 99]
    assert choose_batches(4, 6, False).tolist() == [0, 1, 2, 3]


def test_extrema_candidates_detects_sign_and_time():
    x = np.zeros((2, 10), dtype=float)
    x[0, 4] = -7
    x[1, 7] = 9
    t, c, s = extrema_candidates(x, 6, True)
    assert (t.tolist(), c.tolist(), s.tolist()) == ([4], [0], [7.0])
    t, c, s = extrema_candidates(x, 8, False)
    assert (t.tolist(), c.tolist(), s.tolist()) == ([7], [1], [9.0])


def test_collapse_candidates_uses_physical_neighbors_and_time():
    positions = np.array([[0, 0], [0, 20], [0, 300]], dtype=float)
    neighbors = spatial_neighbors(positions, 100)
    times = np.array([10, 11, 11])
    channels = np.array([0, 1, 2])
    scores = np.array([8, 7, 6], dtype=float)
    keep = collapse_candidates(times, channels, scores, neighbors, 30, 2)
    assert keep.tolist() == [0, 2]


def test_depth_exposure_counts_each_shank_once():
    positions = np.array([[0, 0], [0, 1000], [200, 0], [200, 500]], dtype=float)
    shanks = np.array([0, 0, 1, 1])
    assert probe_depth_exposure_mm(positions, shanks) == 1.5


def test_depth_exposure_infers_widely_separated_shanks():
    positions = np.array([[0, 0], [0, 1000], [200, 0], [200, 500]], dtype=float)
    assert probe_depth_exposure_mm(positions, np.zeros(4)) == 1.5
