from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from testing.em_huklaban5_voltage_equivalence import (
    dd_source_time_origin,
    get_traces_with_aligned_parent_chunks,
    load_relative_recording,
)


def test_relative_recording_load_uses_json_directory_as_base_folder(tmp_path):
    recording_json = tmp_path / "S_L" / "recording.json"
    recording_json.parent.mkdir()
    recording_json.write_text('{"relative_paths": true}\n')
    calls = []
    sentinel = object()
    si = SimpleNamespace(
        load=lambda path, *, base_folder: (
            calls.append((path, base_folder)),
            sentinel,
        )[1]
    )

    loaded = load_relative_recording(si, recording_json)

    assert loaded is sentinel
    assert calls == [(recording_json, recording_json.parent)]


def test_dd_source_clock_uses_materializer_frame_axis_not_parent_clock():
    parent = SimpleNamespace(
        get_num_samples=lambda: 300,
        get_sampling_frequency=lambda: 30.0,
        sample_index_to_time=lambda frame: 3957.0 + frame / 30.0,
    )
    receipt = {
        "source_window": {
            "start_frame": 27_000,
            "end_frame_exclusive": 27_300,
            "fs": 30.0,
            "start_s": 900.0,
            "end_s": 910.0,
        }
    }

    assert dd_source_time_origin(parent, receipt) == 900.0


def test_dd_source_clock_rejects_parent_frame_mismatch():
    parent = SimpleNamespace(
        get_num_samples=lambda: 299,
        get_sampling_frequency=lambda: 30.0,
    )
    receipt = {
        "source_window": {
            "start_frame": 27_000,
            "end_frame_exclusive": 27_300,
            "fs": 30.0,
            "start_s": 900.0,
            "end_s": 910.0,
        }
    }

    with pytest.raises(ValueError, match="frame count"):
        dd_source_time_origin(parent, receipt)


def test_aligned_trace_read_uses_complete_materialization_chunks_then_slices():
    values = np.arange(46, dtype=np.float32).reshape(23, 2)
    calls = []

    class Recording:
        def get_num_samples(self):
            return len(values)

        def get_traces(self, *, start_frame, end_frame):
            calls.append((start_frame, end_frame))
            return values[start_frame:end_frame]

    actual = get_traces_with_aligned_parent_chunks(
        Recording(), 7, 19, chunk_frames=5
    )

    np.testing.assert_array_equal(actual, values[7:19])
    assert calls == [(5, 10), (10, 15), (15, 20)]
