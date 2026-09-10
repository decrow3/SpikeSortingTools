import numpy as np
import pytest

from spikeinterface.core import NumpyRecording

from testing.checked_int16_recording import CheckedInt16Recording


def test_checked_int16_rounds_nearest_without_clipping():
    source = NumpyRecording(
        [np.array([[-1.6, -0.4, 0.6, 1.5]], dtype="float32")],
        sampling_frequency=30_000,
    )
    result = CheckedInt16Recording(source).get_traces()
    assert np.array_equal(result, np.array([[-2, 0, 1, 2]], dtype="int16"))


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_checked_int16_refuses_nonfinite(value):
    source = NumpyRecording([np.array([[value]], dtype="float32")], 30_000)
    with pytest.raises(ValueError, match="nonfinite"):
        CheckedInt16Recording(source).get_traces()


@pytest.mark.parametrize("value", [-32769.0, 32768.0])
def test_checked_int16_refuses_overflow(value):
    source = NumpyRecording([np.array([[value]], dtype="float32")], 30_000)
    with pytest.raises(OverflowError, match="out-of-range"):
        CheckedInt16Recording(source).get_traces()
