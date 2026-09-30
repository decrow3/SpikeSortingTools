import numpy as np
import spikeinterface as si

from testing.en_rounded_field import build_adapter, round_half_away, rounded_rigid_field


def _recording():
    traces = np.tile(np.arange(10, 90, 10, dtype=np.int16), (8, 1))
    recording = si.NumpyRecording(traces, sampling_frequency=4.0)
    recording.set_channel_locations(
        np.c_[[16.0, 48.0, 0.0, 32.0, 16.0, 48.0, 0.0, 32.0],
              [0.0, 0.0, 20.0, 20.0, 40.0, 40.0, 60.0, 60.0]]
    )
    return recording


def test_rounding_is_half_away_and_median_referenced():
    assert np.array_equal(round_half_away(np.asarray([-1.5, -.5, .5, 1.5])), [-2, -1, 1, 2])
    rounded, reference = rounded_rigid_field(np.asarray([-20.0, 0.0, 20.0, 60.0]))
    assert reference == 10.0
    assert np.array_equal(rounded, [-40, 0, 0, 40])


def test_constant_positive_40_sign_and_outer_zero():
    corrected = build_adapter(_recording(), np.asarray([0.0, 1.0, 2.0]), np.asarray([40.0, 40.0, 40.0]))
    expected = np.asarray([50, 60, 70, 80, 0, 0, 0, 0], dtype=np.int16)
    assert np.array_equal(corrected.get_traces(start_frame=0, end_frame=8), np.tile(expected, (8, 1)))


def test_q0_is_byte_identical_and_temporal_cells_are_stepwise():
    source = _recording()
    q0 = build_adapter(source, np.asarray([0.0, 1.0, 2.0]), np.asarray([0.0, 0.0, 0.0]))
    assert q0.get_traces(start_frame=0, end_frame=8).tobytes() == source.get_traces(start_frame=0, end_frame=8).tobytes()
    stepped = build_adapter(source, np.asarray([0.0, 1.0, 2.0]), np.asarray([0.0, 40.0, 40.0]))
    out = stepped.get_traces(start_frame=0, end_frame=8)
    assert np.array_equal(out[:2], source.get_traces(start_frame=0, end_frame=2))
    expected = np.asarray([50, 60, 70, 80, 0, 0, 0, 0], dtype=np.int16)
    assert np.array_equal(out[2:], np.tile(expected, (6, 1)))
