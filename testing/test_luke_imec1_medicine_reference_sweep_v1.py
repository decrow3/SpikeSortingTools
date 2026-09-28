import numpy as np
import pandas as pd

from testing.luke_imec1_medicine_reference_sweep_v1 import (
    CONFIGS, EXTRACTION_TIMEOUT_S, FIT_TIMEOUT_S, contiguous_intervals,
    erode_boolean, pareto_front, sample_field,
)


def test_boolean_erosion_and_intervals():
    mask = np.array([0, 1, 1, 1, 1, 1, 0], bool)
    eroded = erode_boolean(mask, 2)
    assert eroded.tolist() == [False, False, False, True, False, False, False]
    assert contiguous_intervals(np.arange(8) * 0.25, mask) == [(0.25, 1.5)]


def test_sample_field_bilinear_and_support():
    field = {
        "session_time_s": np.array([0.0, 1.0]),
        "depth_um": np.array([0.0, 10.0]),
        "displacement_um": np.array([[0.0, 10.0], [2.0, 12.0]]),
    }
    got = sample_field(field, np.array([0.5, -1.0]), np.array([5.0, 5.0]))
    assert np.allclose(got[0], 6.0)
    assert np.isnan(got[1])


def test_pareto_front():
    table = pd.DataFrame({"name": ["a", "b", "c"], "x": [1, 2, 1], "y": [2, 1, 3]})
    assert set(pareto_front(table, ["x", "y"]).name) == {"a", "b"}


def test_amendment_k_arms_and_timeouts():
    assert len(CONFIGS) == 16
    assert CONFIGS["amp50"] == {"amplitude_threshold_quantile": 0.5}
    assert CONFIGS["amp25"] == {"amplitude_threshold_quantile": 0.75}
    assert [CONFIGS[f"kern{x}"]["time_kernel_width"] for x in ("5", "10", "30")] == [5, 10, 30]
    assert FIT_TIMEOUT_S == 900
    assert EXTRACTION_TIMEOUT_S > FIT_TIMEOUT_S
