import numpy as np

from testing.luke_lighthouse_ransac_field_v1 import fit_ransac


def test_ransac_recovers_affine_depth_field_despite_one_outlier():
    depth = np.array([0, 500, 1000, 2000, 3000, 3500, 3800], float)
    expected = 12 + 8 * ((depth - 2000) / 1000)
    observed = expected.copy(); observed[3] += 200
    intercept, slope, inliers = fit_ransac(depth, observed, residual_threshold_um=10)
    assert abs(intercept - 12) < 1e-8
    assert abs(slope - 8) < 1e-8
    assert inliers.sum() == 6
