import numpy as np

from testing.luke_aw_validate_handoff import mask_at, q


def test_mask_at_uses_half_open_intervals():
    times = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    assert np.array_equal(mask_at(times, [(0.5, 1.5)]), [False, True, True, False, False])


def test_quantile_summary_empty_and_populated():
    assert q(np.array([])) == {"n": 0, "median": None, "p95": None, "max": None}
    result = q(np.array([1.0, 2.0, 3.0]))
    assert result["n"] == 3
    assert result["median"] == 2.0
    assert result["max"] == 3.0
