import numpy as np

from testing.luke_dartsort_descriptive_scorecard_v1 import event_bin, round_half_away


def test_round_half_away():
    values = np.array([-1.5, -0.5, -0.49, 0.49, 0.5, 1.5])
    assert np.array_equal(round_half_away(values), [-2, -1, 0, 0, 1, 2])


def test_event_boundary_goes_to_later_cell():
    # The implementation follows the frozen float64 formula exactly.
    samples = np.array([0, int(np.ceil(0.25 * 29999.759166666667))], dtype=np.int64)
    assert np.array_equal(event_bin(samples), [0, 1])
