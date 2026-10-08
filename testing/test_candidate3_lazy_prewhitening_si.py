from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest


pytest.importorskip("spikeinterface")
torch = pytest.importorskip("torch")
from kilosort.io import BinaryFiltered
from kilosort.preprocessing import get_highpass_filter
from spikeinterface.core import BaseRecording, BaseRecordingSegment, NumpyRecording

from npx_preprocessing.motion import KilosortPrewhiteningRecording


FS = 29999.835983263598


def geometry(n_channels: int) -> np.ndarray:
    return np.c_[32.0 * (np.arange(n_channels) % 2), 20.0 * np.arange(n_channels)]


def recording(values, *, t_start=17.25):
    rec = NumpyRecording(values, sampling_frequency=FS, t_starts=[t_start])
    rec.set_channel_locations(geometry(values.shape[1]))
    return rec


def eager(values, parent_indices, *, scale=None, shift=None):
    work = values
    if values.dtype == np.uint16:
        work = values.astype(np.float32) - np.float32(2**15)
    else:
        work = values.astype(np.float32, copy=False)
    if scale is not None:
        work = work * scale
    if shift is not None:
        work = work + shift
    producer = object.__new__(BinaryFiltered)
    producer.chan_map = torch.as_tensor(parent_indices, dtype=torch.int64)
    producer.invert_sign = False
    producer.do_CAR = True
    producer.hp_filter = get_highpass_filter(fs=FS, cutoff=300, device=torch.device("cpu"))
    producer.artifact_threshold = np.inf
    producer.whiten_mat = None
    producer.dshift = None
    producer.device = torch.device("cpu")
    return BinaryFiltered.filter(
        producer, torch.from_numpy(np.ascontiguousarray(work.T)).float()
    ).numpy(force=True).T


@pytest.mark.parametrize("length", [63, 64, 127, 256])
def test_exact_request_lengths_match_eager_installed_filter(length):
    rng = np.random.default_rng(6100 + length)
    values = rng.integers(-1200, 1201, size=(500, 8), dtype=np.int16)
    rec = recording(values)
    wrapped = KilosortPrewhiteningRecording(rec)
    start = 37
    actual = wrapped.get_traces(start_frame=start, end_frame=start + length)
    expected = eager(values[start : start + length], np.arange(8))
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.float32


def test_padded_context_and_core_match_eager_padded_request_with_positive_control():
    rng = np.random.default_rng(6201)
    values = rng.integers(-2000, 2001, size=(420, 8), dtype=np.int16)
    wrapped = KilosortPrewhiteningRecording(recording(values))
    padded = wrapped.get_traces(start_frame=80, end_frame=336)
    eager_padded = eager(values[80:336], np.arange(8))
    np.testing.assert_array_equal(padded, eager_padded)
    np.testing.assert_array_equal(padded[40:-40], eager_padded[40:-40])
    eager_unpadded_core = eager(values[120:296], np.arange(8))
    assert not np.array_equal(padded[40:-40], eager_unpadded_core)


def test_channel_order_geometry_nonzero_origin_and_scale_shift():
    rng = np.random.default_rng(6301)
    values = rng.integers(-1000, 1001, size=(256, 8), dtype=np.int16)
    ids = np.array(["a", "b", "c", "d", "e", "f", "g", "h"])
    rec = NumpyRecording(values, sampling_frequency=FS, t_starts=[17.25], channel_ids=ids)
    rec.set_channel_locations(geometry(8))
    selected = np.array(["h", "c", "f", "a"])
    scale = np.float32(0.25)
    shift = np.float32(-1.5)
    wrapped = KilosortPrewhiteningRecording(
        rec, channel_ids=selected, scale=scale, shift=shift
    )
    actual = wrapped.get_traces(start_frame=11, end_frame=202)
    expected = eager(values[11:202], np.array([7, 2, 5, 0]), scale=scale, shift=shift)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(wrapped.channel_ids, selected)
    np.testing.assert_array_equal(wrapped.get_channel_locations(), geometry(8)[[7, 2, 5, 0]])
    assert wrapped.sample_index_to_time(0) == pytest.approx(17.25)


def test_multiple_segment_clocks_and_uint16_signed_conversion():
    rng = np.random.default_rng(6401)
    first = rng.integers(0, 65536, size=(127, 6), dtype=np.uint16)
    second = rng.integers(0, 65536, size=(191, 6), dtype=np.uint16)
    rec = NumpyRecording(
        [first, second], sampling_frequency=FS, t_starts=[11.5, 43.75]
    )
    rec.set_channel_locations(geometry(6))
    wrapped = KilosortPrewhiteningRecording(rec)
    for segment, (source, origin) in enumerate(((first, 11.5), (second, 43.75))):
        actual = wrapped.get_traces(segment_index=segment)
        np.testing.assert_array_equal(actual, eager(source, np.arange(6)))
        assert wrapped.sample_index_to_time(0, segment_index=segment) == pytest.approx(origin)


class TrackingSegment(BaseRecordingSegment):
    def __init__(self, values, calls):
        super().__init__(sampling_frequency=FS, t_start=9.0)
        self.values = values
        self.calls = calls

    def get_num_samples(self):
        return len(self.values)

    def get_traces(self, start_frame, end_frame, channel_indices):
        self.calls.append((start_frame, end_frame, channel_indices))
        return self.values[start_frame:end_frame, channel_indices]


class TrackingRecording(BaseRecording):
    def __init__(self, values, calls):
        super().__init__(FS, np.arange(values.shape[1]), values.dtype)
        self.set_channel_locations(geometry(values.shape[1]))
        self.add_recording_segment(TrackingSegment(values, calls))


def test_is_lazy_and_reads_only_exact_requested_interval():
    values = np.arange(300 * 6, dtype=np.int16).reshape(300, 6)
    calls = []
    wrapped = KilosortPrewhiteningRecording(TrackingRecording(values, calls))
    assert calls == []
    wrapped.get_traces(start_frame=41, end_frame=168)
    assert len(calls) == 1
    assert calls[0][0:2] == (41, 168)


def test_dartsort_none_cross_environment_consumer(tmp_path):
    rng = np.random.default_rng(6501)
    values = rng.integers(-800, 801, size=(127, 6), dtype=np.int16)
    boundary = KilosortPrewhiteningRecording(recording(values)).get_traces()
    input_path = tmp_path / "boundary.npy"
    output_path = tmp_path / "consumer.json"
    np.save(input_path, boundary, allow_pickle=False)
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["NUMBA_CACHE_DIR"] = str(tmp_path / "numba")
    subprocess.run(
        [
            "/home/huklab/Documents/DARTsort/.venv/bin/python",
            str(root / "testing/candidate3_lazy_prewhitening_dartsort_consumer.py"),
            str(input_path),
            str(output_path),
        ],
        cwd=root,
        env=env,
        check=True,
    )
    result = json.loads(output_path.read_text())
    assert all(result.values()) or result["dtype"] == "float32"
    assert result == {
        "array_equal": True,
        "channel_ids_equal": True,
        "copy_flag_false": True,
        "dtype": "float32",
        "geometry_equal": True,
        "same_object": True,
        "time_origin_equal": True,
        "workdir_is_none": True,
        "workdir_same_object": True,
    }


def test_invalid_configuration_fails_closed():
    values = np.zeros((64, 4), dtype=np.int16)
    rec = recording(values)
    with pytest.raises(ValueError, match="unique"):
        KilosortPrewhiteningRecording(rec, channel_ids=[0, 0])
    with pytest.raises(ValueError, match="exist"):
        KilosortPrewhiteningRecording(rec, channel_ids=[99])
    with pytest.raises(ValueError, match="finite"):
        KilosortPrewhiteningRecording(rec, scale=np.nan)
    with pytest.raises(ValueError, match="one value per parent channel"):
        KilosortPrewhiteningRecording(rec, shift=[1, 2])
    with pytest.raises(ValueError, match="cpu"):
        KilosortPrewhiteningRecording(rec, device="cuda")
    wrapped = KilosortPrewhiteningRecording(rec)
    with pytest.raises(ValueError, match="contain samples"):
        wrapped.get_traces(start_frame=4, end_frame=4)
