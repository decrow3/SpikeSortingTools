import numpy as np

from testing.luke_cluster553_template_calibration_audit import quantiles


def test_quantiles_are_monotonic_and_include_median():
    result = quantiles(np.arange(100, dtype=float))
    assert result["0.5"] == 49.5
    assert list(result.values()) == sorted(result.values())
