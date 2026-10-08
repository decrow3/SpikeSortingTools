#!/usr/bin/env python3
"""Small cross-environment consumer for the frozen DARTsort none boundary."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from spikeinterface.core import NumpyRecording

import dartsort
from dartsort.util.main_util import ds_all_to_workdir, ds_will_copy_recording
from dartsort.util.preprocess_util import preprocess


def main(input_path: str, output_path: str) -> None:
    values = np.load(input_path, allow_pickle=False)
    recording = NumpyRecording(
        values,
        sampling_frequency=29999.835983263598,
        t_starts=[17.25],
    )
    recording.set_channel_locations(
        np.c_[np.arange(values.shape[1], dtype=float), 20.0 * np.arange(values.shape[1])]
    )
    cfg = dartsort.DARTsortInternalConfig(
        preprocessing="none",
        copy_recording_to_tmpdir="no",
        work_in_tmpdir=False,
    )
    copy_flag = ds_will_copy_recording(cfg)
    consumed = preprocess(recording, strategy=cfg.preprocessing, dtype=cfg.preprocessing_dtype)
    prepared, work_dir = ds_all_to_workdir(
        internal_cfg=cfg,
        output_dir=Path(output_path).parent,
        work_dir=None,
        recording=consumed,
        overwrite=False,
    )
    observed = prepared.get_traces()
    result = {
        "copy_flag_false": copy_flag is False,
        "same_object": consumed is recording,
        "workdir_same_object": prepared is recording,
        "workdir_is_none": work_dir is None,
        "array_equal": bool(np.array_equal(observed, values)),
        "dtype": str(observed.dtype),
        "time_origin_equal": bool(consumed.sample_index_to_time(0) == 17.25),
        "channel_ids_equal": bool(np.array_equal(consumed.channel_ids, recording.channel_ids)),
        "geometry_equal": bool(
            np.array_equal(consumed.get_channel_locations(), recording.get_channel_locations())
        ),
    }
    Path(output_path).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if not all(
        (
            result["same_object"],
            result["copy_flag_false"],
            result["workdir_same_object"],
            result["workdir_is_none"],
            result["array_equal"],
            result["time_origin_equal"],
            result["channel_ids_equal"],
            result["geometry_equal"],
            result["dtype"] == "float32",
        )
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
