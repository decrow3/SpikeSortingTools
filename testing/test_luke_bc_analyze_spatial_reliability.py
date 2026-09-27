import importlib.util
from pathlib import Path

import numpy as np


_SPEC = importlib.util.spec_from_file_location(
    "luke_bc_analyze_spatial_reliability",
    Path(__file__).with_name("luke_bc_analyze_spatial_reliability.py"),
)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
waveform_metrics = _MODULE.waveform_metrics


def test_waveform_metrics_centers_each_channel_over_time():
    # Deliberately asymmetric: transposing this [time, channel] view changes the
    # centered cosine, so it catches advanced-indexing axis reversal.
    left = np.array([[1.0, 10.0, -2.0], [2.0, 8.0, 4.0], [5.0, 7.0, 9.0], [9.0, 3.0, 8.0]])
    right = np.array([[2.0, 7.0, -1.0], [3.0, 6.0, 5.0], [7.0, 4.0, 7.0], [8.0, 1.0, 10.0]])
    got = waveform_metrics(left, right)
    lc = left - left.mean(axis=0, keepdims=True)
    rc = right - right.mean(axis=0, keepdims=True)
    expected = np.vdot(lc, rc) / np.sqrt(np.vdot(lc, lc) * np.vdot(rc, rc))
    wrong = waveform_metrics(left.T, right.T)["centered_cosine"]
    assert np.isclose(got["centered_cosine"], expected)
    assert not np.isclose(got["centered_cosine"], wrong)


def test_waveform_metrics_rejects_non_matrix_or_mismatched_views():
    with np.testing.assert_raises(ValueError):
        waveform_metrics(np.ones(4), np.ones(4))
    with np.testing.assert_raises(ValueError):
        waveform_metrics(np.ones((4, 2)), np.ones((2, 4)))
