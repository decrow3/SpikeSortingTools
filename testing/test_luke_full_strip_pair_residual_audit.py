import numpy as np

from testing.luke_full_strip_pair_residual_audit import (
    one_to_one_centers,
    shift_template,
)


def test_one_to_one_centers():
    centers = one_to_one_centers(np.array([10, 30]), np.array([11, 29]), 2)
    assert centers.tolist() == [10, 30]


def test_shift_template_zero_pads():
    value = np.arange(5)[:, None]
    assert shift_template(value, 1).ravel().tolist() == [0, 0, 1, 2, 3]
    assert shift_template(value, -1).ravel().tolist() == [1, 2, 3, 4, 0]
