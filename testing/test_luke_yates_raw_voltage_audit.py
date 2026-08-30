import numpy as np

from testing.luke_yates_raw_voltage_audit import (
    collapse_candidates,
    depth_exposure_mm,
    extrema_candidates,
    local_median_reference,
    nearest_neighbors,
    shank_median_reference,
    spatial_neighbors,
)


def test_depth_exposure_sums_shanks_once():
    locations = np.array([[0, 0], [0, 100], [200, 0], [200, 100]], float)
    shanks = np.array([0, 0, 1, 1])
    assert depth_exposure_mm(locations, shanks) == 0.2


def test_local_reference_removes_shared_signal():
    values = np.tile(np.arange(10, dtype=np.float32)[:, None], (1, 3))
    locations = np.array([[0, 0], [0, 20], [0, 40]], float)
    neighbors = spatial_neighbors(locations, np.zeros(3), 100)
    assert np.allclose(local_median_reference(values, neighbors), 0)


def test_nearest_neighbors_has_fixed_count_and_stays_on_shank():
    locations = np.array(
        [[0, 0], [0, 20], [0, 40], [200, 0], [200, 20], [200, 40]], float
    )
    shanks = np.array([0, 0, 0, 1, 1, 1])
    neighbors = nearest_neighbors(locations, shanks, 2)
    assert all(len(group) == 2 for group in neighbors)
    assert all(np.all(shanks[group] == shanks[channel]) for channel, group in enumerate(neighbors))
    assert all(channel in group for channel, group in enumerate(neighbors))


def test_shank_reference_is_independent_between_shanks():
    values = np.array([[1, 3, 10, 14], [2, 4, 20, 24]], np.float32)
    referenced = shank_median_reference(values, np.array([0, 0, 1, 1]))
    assert np.allclose(referenced, [[-1, 1, -2, 2], [-1, 1, -2, 2]])


def test_spatiotemporal_collapse_keeps_largest_neighbor():
    locations = np.array([[0, 0], [0, 20], [0, 200]], float)
    neighbors = spatial_neighbors(locations, np.zeros(3), 100)
    times = np.array([10, 11, 10])
    channels = np.array([0, 1, 2])
    scores = np.array([5.0, 8.0, 6.0])
    keep = collapse_candidates(times, channels, scores, neighbors, 30, 2)
    assert set(keep) == {1, 2}


def test_extrema_polarity():
    values = np.array([[0, 0], [-1, 1], [-5, 5], [-1, 1], [0, 0]], np.float32)
    nt, nc, na = extrema_candidates(values, negative=True)
    pt, pc, pa = extrema_candidates(values, negative=False)
    assert nt.tolist() == [2] and nc.tolist() == [0] and na.tolist() == [5]
    assert pt.tolist() == [2] and pc.tolist() == [1] and pa.tolist() == [5]
