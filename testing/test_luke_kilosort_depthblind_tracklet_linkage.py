import numpy as np

from testing.luke_kilosort_depthblind_tracklet_linkage_v1 import (
    best_lagged_cosine,
    centered_crop,
    nearest_cross_count,
)


def test_temporal_alignment_shifts_only_one_side():
    base = np.zeros((49, 16), dtype=np.float32)
    base[20:23, 4] = [-1.0, 0.5, 0.2]
    base[21:24, 5] = [-0.6, 0.3, 0.1]
    delayed = np.zeros_like(base)
    delayed[2:] = base[:-2]
    other = np.zeros_like(base)
    other[18:21, 10] = [0.2, -1.0, 0.1]

    score, lag = best_lagged_cosine(np.stack([base, delayed, other]))

    assert score[0, 1] > 0.999
    assert abs(int(lag[0, 1])) == 2
    assert score[0, 2] < 0.1


def test_centered_crop_and_cross_count_boundaries():
    wave = np.arange(61 * 2, dtype=np.float32).reshape(61, 2)
    crop = centered_crop(wave, peak_sample=20, samples=49)
    np.testing.assert_array_equal(crop[4:], wave[:45])
    np.testing.assert_array_equal(crop[:4], 0)

    a = np.array([1.0, 2.0, 3.0])
    b = np.array([0.9998, 2.001, 4.0])
    assert nearest_cross_count(a, b, 0.0003) == 1
