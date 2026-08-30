import numpy as np

from testing.luke_direct_motion_scale_audit import (
    RasterSpec,
    deterministic_half,
    estimate_pair_shift,
    prepare_raster,
    robust_line,
)


def test_deterministic_half_is_repeatable_and_mixed():
    samples = np.arange(1000, dtype=np.int64)
    channels = np.arange(1000, dtype=np.int64) % 384
    first = deterministic_half(samples, channels)
    second = deterministic_half(samples, channels)
    np.testing.assert_array_equal(first, second)
    assert set(first) == {0, 1}
    assert 0.4 < first.mean() < 0.6


def test_pair_shift_recovers_physical_translation():
    depth_bin_um = 2.0
    depth = np.arange(1920) * depth_bin_um
    first = np.vstack(
        [
            np.exp(-0.5 * ((depth - center) / width) ** 2)
            for center, width in [(700, 35), (1700, 60), (2900, 45)]
        ]
    )
    true_shift_um = 13.5
    second = np.vstack(
        [np.interp(depth, depth + true_shift_um, row, left=0, right=0) for row in first]
    )
    result = estimate_pair_shift(
        first,
        second,
        depth_bin_um=depth_bin_um,
        maximum_shift_um=30,
        step_um=0.5,
    )
    assert abs(result["observed_shift_um"] - true_shift_um) < 0.6
    assert not result["hit_search_boundary"]


def test_prepare_raster_can_collapse_amplitude_and_depth_bins():
    base = np.arange(2 * 3 * 8, dtype=np.float32).reshape(2, 3, 8)
    result = prepare_raster(
        base,
        RasterSpec("test", 4.0, 4.0, False),
        base_depth_bin_um=2.0,
    )
    assert result.shape == (2, 1, 4)
    np.testing.assert_allclose(result.mean(axis=2), 0, atol=1e-5)


def test_robust_line_downweights_one_large_outlier():
    x = np.arange(-10, 11, dtype=float)
    y = 1.25 * x + 2.0
    y[-1] = 100.0
    result = robust_line(x, y)
    assert abs(result["slope"] - 1.25) < 0.15
    assert abs(result["intercept_um"] - 2.0) < 0.5
