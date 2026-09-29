import numpy as np
import pytest

from testing.em_huklaban5_voltage_equivalence_arms import (
    assert_digest,
    compare,
    render_mapping,
    stock_mapping,
)


def test_hash_preflight_refuses_changed_input():
    with pytest.raises(RuntimeError, match="differs"):
        assert_digest("fixture", "changed", "frozen")


def test_stock_nearest_substitutes_interior_hole_but_exact_zero_fills():
    from spikeinterface.preprocessing import get_spatial_interpolation_kernel

    source = np.asarray([[0.0, 0.0], [40.0, 0.0], [0.0, 40.0], [40.0, 40.0]])
    target = np.asarray([[20.0, 20.0]])
    mapping = stock_mapping(get_spatial_interpolation_kernel, source, target, 0)

    assert mapping.tolist() == [0]
    traces = np.asarray([[3.0, 5.0, 7.0, 11.0]], dtype=np.float32)
    assert render_mapping(traces, mapping).tolist() == [[3.0]]


def test_explicit_tolerance_accepts_one_to_two_float32_ulps():
    reference = np.asarray([[0.125, 0.25]], dtype=np.float32)
    actual = reference.copy()
    actual[0, 1] = np.nextafter(actual[0, 1], np.float32(np.inf), dtype=np.float32)

    result = compare(actual, reference, np.asarray([-200]))

    assert result["max_abs_error"] < 1e-6
    assert result["count_abs_error_gt_1e_6"] == 0
    assert result["zero_positions_equal"] is True
