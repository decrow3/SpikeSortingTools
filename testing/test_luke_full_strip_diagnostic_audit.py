import numpy as np

from testing.luke_full_strip_diagnostic_audit import cosine, linear_slope_per_hour


def test_cosine_identity_and_orthogonality():
    assert np.isclose(cosine([1, 2], [1, 2]), 1.0)
    assert np.isclose(cosine([1, 0], [0, 1]), 0.0)


def test_linear_slope_per_hour():
    assert np.isclose(
        linear_slope_per_hour(np.array([0.0, 3600.0]), np.array([2.0, 5.0])),
        3.0,
    )
