import numpy as np

from testing.luke_conditioning_order_audit import comparison_metrics


def test_comparison_metrics_separates_bad_channel():
    channel_ids = np.array([190, 191, 192])
    left = np.arange(12, dtype=np.int16).reshape(4, 3)
    right = left.copy()
    right[:, 1] += 1
    metrics = {
        row["group"]: row
        for row in comparison_metrics(left, right, channel_ids, "test")
    }
    assert metrics["excluding_191"]["fraction_equal"] == 1.0
    assert metrics["channel_191"]["fraction_equal"] == 0.0
    assert metrics["channel_191"]["max_abs_difference_counts"] == 1.0


def test_comparison_metrics_rejects_shape_mismatch():
    try:
        comparison_metrics(
            np.zeros((2, 3)), np.zeros((3, 3)), np.arange(3), "test"
        )
    except ValueError as error:
        assert "equal-shaped" in str(error)
    else:
        raise AssertionError("Expected unequal trace shapes to fail")
