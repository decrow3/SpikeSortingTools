from pathlib import Path

import numpy as np
import pytest
import torch

from testing.joint_model_k1_k2 import (
    CARSignalSideChannel,
    detect_frozen_pulses,
    freeze_pulse_sorter_lag,
    load_exact_reader,
)


READER_PATH = Path(__file__).with_name("prospective_a_prewhitening_si_e36d3066.py")


def make_recording(tmp_path, *, highpass=True, invert=False, artifact_threshold=np.inf):
    reader = load_exact_reader(READER_PATH)
    frames = 30
    channels = 5
    values = (
        np.arange(frames * channels, dtype=np.int16).reshape(frames, channels) % 37
    ) - 18
    values[:, 2] += np.asarray([0, 3, -7, 5, -2] * 6, dtype=np.int16)
    path = tmp_path / "fixture.raw"
    path.write_bytes(values.tobytes())
    kwargs = {
        "binary_path": path,
        "num_samples": frames,
        "num_channels_raw": channels,
        "dtype": "int16",
        "sampling_frequency": 30_000.0,
        "batch_size": 8,
        "padding": 2,
        "channel_map": np.array([4, 1, 3, 0, 2]),
        "channel_ids": np.arange(channels),
        "channel_locations": np.column_stack((np.zeros(channels), np.arange(channels) * 20.0)),
        "highpass_filter": np.array([0.25, -0.5, 0.25], dtype=np.float32) if highpass else None,
        "temporal_centering": True,
        "do_car": True,
        "invert_sign": invert,
        "artifact_threshold": artifact_threshold,
        "scale": 0.5,
        "shift": 1.0,
    }
    return reader, values, kwargs


@pytest.mark.parametrize(
    "start,end,selector,highpass,invert",
    [
        (0, 7, None, False, False),
        (6, 19, [0, 2, 4], True, False),
        (7, 9, np.array([True, False, True, False, True]), True, True),
        (23, 30, slice(1, 4), False, True),
    ],
)
def test_side_channel_traces_are_bitwise_exact(
    tmp_path, start, end, selector, highpass, invert
):
    reader, _, kwargs = make_recording(tmp_path, highpass=highpass, invert=invert)
    reference = reader.ProspectiveAPrewhiteningRecording(**kwargs)
    candidate_recording = reader.ProspectiveAPrewhiteningRecording(**kwargs)
    candidate = CARSignalSideChannel(candidate_recording, reader)
    expected = reference._recording_segments[0].get_traces(start, end, selector)
    actual, hp_median = candidate.get_traces_and_hp_median(start, end, selector)
    assert expected.dtype == actual.dtype == np.float32
    assert expected.shape == actual.shape
    assert np.array_equal(expected, actual)
    assert hp_median.shape == (end - start,)
    assert hp_median.dtype == np.float32


def test_removed_median_known_answer_without_highpass(tmp_path):
    reader, _, kwargs = make_recording(tmp_path, highpass=False, invert=False)
    recording = reader.ProspectiveAPrewhiteningRecording(**kwargs)
    candidate = CARSignalSideChannel(recording, reader)
    _, actual = candidate.get_traces_and_hp_median(8, 16, None)

    segment = recording._recording_segments[0]
    raw = segment._read_frames(6, 18).T * 0.5 + 1.0
    padded = torch.from_numpy(raw.astype(np.float32, copy=False))[segment.channel_map]
    centered = padded - padded.mean(1).unsqueeze(1)
    expected = torch.median(centered, 0)[0][2:10].numpy(force=True)
    assert np.array_equal(actual, expected)


def test_whole_batch_artifact_zeroes_traces_and_side_channel(tmp_path):
    reader, _, kwargs = make_recording(
        tmp_path, highpass=True, artifact_threshold=0.01
    )
    reference = reader.ProspectiveAPrewhiteningRecording(**kwargs)
    candidate = CARSignalSideChannel(
        reader.ProspectiveAPrewhiteningRecording(**kwargs), reader
    )
    expected = reference._recording_segments[0].get_traces(8, 16, None)
    actual, median = candidate.get_traces_and_hp_median(8, 16, None)
    assert np.array_equal(expected, np.zeros_like(expected))
    assert np.array_equal(actual, expected)
    assert np.array_equal(median, np.zeros_like(median))


def test_side_channel_rejects_non_car_reader(tmp_path):
    reader, _, kwargs = make_recording(tmp_path)
    kwargs["do_car"] = False
    recording = reader.ProspectiveAPrewhiteningRecording(**kwargs)
    with pytest.raises(ValueError, match="do_car"):
        CARSignalSideChannel(recording, reader)


def test_reader_hash_fails_closed(tmp_path):
    changed = tmp_path / "reader.py"
    changed.write_bytes(READER_PATH.read_bytes() + b"\n# changed\n")
    with pytest.raises(ValueError, match="SHA-256"):
        load_exact_reader(changed)


def test_frozen_detector_threshold_alignment_refractory_and_origin():
    signal = np.zeros(260, dtype=np.float32)
    signal[10] = -130.0
    signal[50] = -200.0
    signal[140:142] = -150.0
    signal[230] = -120.0
    detected = detect_frozen_pulses(
        signal, sampling_frequency=30_000.0, sample_origin=1000
    )
    assert detected.tolist() == [1050, 1140]


def test_frozen_detector_prefers_earlier_equal_depth_within_refractory():
    signal = np.zeros(200, dtype=np.float32)
    signal[20] = signal[60] = -150.0
    assert detect_frozen_pulses(signal, sampling_frequency=30_000.0).tolist() == [20]


def test_k2_ccg_uses_all_focus_detections_and_reports_peak_width():
    pulses = np.array([100, 300, 500])
    detections = np.array(
        [105, 106, 107, 305, 306, 307, 506, 900, 106, 306]
    )
    channels = np.array([251, 252, 251, 251, 252, 252, 251, 251, 99, 99])
    result = freeze_pulse_sorter_lag(
        pulses,
        detections,
        channels,
        sampling_frequency=30_000.0,
        clock_identity="fixture-session-clock",
    )
    assert result["lag_convention"] == "detection_sample_minus_pulse_sample"
    assert result["n_focus_detections"] == 8
    assert result["peak_lag_samples"] == 6
    assert result["peak_count"] == 3
    assert result["half_max_min_count"] == 2
    assert result["half_max_left_lag_samples"] == 5
    assert result["half_max_right_lag_samples"] == 7
    assert result["half_max_width_sample_bins"] == 3


@pytest.mark.parametrize(
    "pulses,detections,channels,clock,match",
    [
        ([100.5], [101], [251], "clock", "integer"),
        ([100], [101.5], [251], "clock", "integer"),
        ([100], [101], [251.5], "clock", "integer"),
        ([100], [101], [251], "", "clock_identity"),
        ([], [101], [251], "clock", "pulse"),
        ([100], [101], [99], "clock", "251/252"),
    ],
)
def test_k2_guards(pulses, detections, channels, clock, match):
    with pytest.raises(ValueError, match=match):
        freeze_pulse_sorter_lag(
            pulses,
            detections,
            channels,
            sampling_frequency=30_000.0,
            clock_identity=clock,
        )
