import numpy as np

from testing.luke_cluster553_depth_trace import (
    depth_profile,
    movement_supported,
    shifted_profile_match,
)


def test_depth_profile_collapses_channels_at_same_depth_by_max_ptp():
    wave = np.array([[0, 0, 0], [-2, -4, -3], [0, 0, 0]], dtype=float)
    depths, profile = depth_profile(wave, np.array([100.0, 100.0, 120.0]))
    assert depths.tolist() == [100.0, 120.0]
    assert profile.tolist() == [4.0, 3.0]


def test_shifted_profile_match_and_movement_rule():
    reference = np.array([0.0, 1.0, 3.0, 1.0, 0.0])
    candidate = np.array([0.0, 0.0, 1.0, 3.0, 1.0])
    cosine, lag = shifted_profile_match(reference, candidate, maximum_lag=2)
    assert cosine > 0.99
    assert lag == 1
    rows = [{"role": "failing", "profile_cosine": cosine, "profile_shift_um": 40.0,
             "local_full_st_coverage": 0.7}]
    assert movement_supported(rows, 0.9, {
        "minimum_absolute_shift_um": 40.0, "minimum_coverage_loss_from_reference": 0.1
    }, 0.8) == (True, [0])
