import numpy as np

from testing.luke_motion_scale_characterization import (
    best_lag_correlation,
    decompose_spatial_field,
    interpolate_field,
    temporal_bandwidth_metrics,
)


def test_interpolate_field_respects_time_and_depth_axes():
    times = np.array([0.0, 2.0])
    depths = np.array([0.0, 100.0])
    field = times[:, None] + depths[None, :] / 10.0
    sampled = interpolate_field(field, times, depths, np.array([1.0]), np.array([50.0]))
    assert sampled.shape == (1, 1)
    assert sampled[0, 0] == 6.0


def test_spatial_decomposition_recovers_rigid_linear_and_residual_energy():
    times = np.arange(20.0)
    depths = np.linspace(0.0, 300.0, 4)
    z = (depths - depths.mean()) / depths.std()
    field = np.sin(times / 3)[:, None] + 2 * np.cos(times / 5)[:, None] * z[None, :]
    result = decompose_spatial_field(field, depths)
    assert result["residual_nonrigid_energy_fraction"] < 1e-20
    assert np.isclose(
        result["rigid_energy_fraction"] + result["linear_depth_energy_fraction"], 1.0
    )


def test_temporal_bandwidth_places_slow_sinusoid_below_fast_sinusoid():
    times = np.arange(256.0)
    slow = np.sin(2 * np.pi * times / 64)
    fast = np.sin(2 * np.pi * times / 8)
    assert temporal_bandwidth_metrics(slow, 1.0)["f50_hz"] < temporal_bandwidth_metrics(fast, 1.0)["f50_hz"]


def test_best_lag_correlation_recovers_shift_direction():
    first = np.sin(np.arange(100) / 7.0)
    second = np.r_[np.zeros(3), first[:-3]]
    lag, value = best_lag_correlation(first, second, max_lag_bins=5)
    assert lag == 3
    assert value > 0.99
