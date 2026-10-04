#!/usr/bin/env python3
"""Known-answer fixture for zero-copy crop stride, map order, and time origin."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np


def main() -> None:
    import inspect
    from kilosort.io import BinaryFiltered, BinaryRWFile, remove_bad_channels, save_to_phy

    frames, channels = 40, 6
    values = (np.arange(frames, dtype=np.int16)[:, None] * 100 + np.arange(channels, dtype=np.int16))
    with tempfile.TemporaryDirectory(prefix="en-crop-reader-") as tmp:
        binary = Path(tmp) / "known.raw"
        values.tofile(binary)
        reader = BinaryRWFile(str(binary), channels, fs=10, NT=8, nt=3, nt0min=1, dtype="int16", tmin=1.2, tmax=3.1)
        expected = values[12:31]
        observed = np.asarray(reader[:])
        if reader.imin != 12 or reader.imax != 31 or not np.array_equal(observed, expected):
            raise RuntimeError("positive stride/crop fixture failed")
        retained = np.array([1, 2, 3, 4])
        if not np.array_equal(observed[:, retained], expected[:, retained]):
            raise RuntimeError("positive physical channel-map fixture failed")
        probe = {
            "chanMap": np.arange(channels), "xc": np.arange(channels, dtype=np.float32),
            "yc": np.arange(channels, dtype=np.float32) * 20, "kcoords": np.zeros(channels, dtype=np.int32),
            "n_chan": channels,
        }
        selected_probe = remove_bad_channels(probe, [0, 5])
        filtered = BinaryFiltered(
            str(binary), channels, fs=10, NT=8, nt=3, nt0min=1,
            chan_map=selected_probe["chanMap"], hp_filter=None, whiten_mat=None,
            do_CAR=False, dtype="int16", tmin=1.2, tmax=3.1,
        )
        filtered_observed = filtered[:].cpu().numpy()
        selected = expected[:, retained].T.astype(np.float32)
        filtered_expected = selected - selected.mean(axis=1, keepdims=True)
        binaryfiltered_path_pass = bool(np.array_equal(filtered_observed, filtered_expected))
        if not binaryfiltered_path_pass:
            raise RuntimeError("bad_channels -> probe -> BinaryFiltered known-answer fixture failed")
        wrong_order_detected = not np.array_equal(observed[:, retained[::-1]], expected[:, retained])
        wrong_stride = BinaryRWFile(str(binary), 4, fs=10, NT=8, nt=3, nt0min=1, dtype="int16", tmin=1.2, tmax=3.1)
        wrong_stride_detected = not np.array_equal(np.asarray(wrong_stride[:])[:, :4], expected[:, retained])
        if not wrong_order_detected or not wrong_stride_detected:
            raise RuntimeError("negative reader controls did not fail as expected")
        export_source = inspect.getsource(save_to_phy)
        export_origin_source_pass = "spike_times = st[:,0].astype('int64') + imin" in export_source
        local_known = np.array([0, 7, 18], dtype=np.int64)
        global_known = local_known + reader.imin
        export_origin_known_answer_pass = bool(np.array_equal(global_known, [12, 19, 30]))
        if not export_origin_source_pass or not export_origin_known_answer_pass:
            raise RuntimeError("global export-origin fixture failed")
        result = {
            "schema": "en-common-support-crop-reader-fixture-v1",
            "status": "pass",
            "positive": {"n_chan_bin": 6, "frames_half_open": [12, 31], "retained_channel_map": retained.tolist(), "bad_channels_probe_binaryfiltered_pass": binaryfiltered_path_pass},
            "negative_controls": {"wrong_order_detected": wrong_order_detected, "wrong_stride_detected": wrong_stride_detected},
            "time_origin": {"rule": "BinaryRWFile floors tmin*fs and tmax*fs; save_to_phy adds bfile.imin.", "source_pass": export_origin_source_pass, "known_answer_pass": export_origin_known_answer_pass},
        }
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
