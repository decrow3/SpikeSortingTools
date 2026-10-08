#!/usr/bin/env python3
import json
import sys
from pathlib import Path

import numpy as np
from spikeinterface.core import NumpyRecording

from dartsort.util.preprocess_util import DSPreprocessingError, preprocess


def recording(traces_time_channel: np.ndarray) -> NumpyRecording:
    rec = NumpyRecording(traces_list=[traces_time_channel], sampling_frequency=30000.0)
    rec.set_channel_locations(np.array([[0.0, 0.0], [32.0, 0.0], [0.0, 20.0], [32.0, 20.0]]))
    return rec


def consume(kind: str, folder: Path) -> None:
    expected = np.load(folder / f"{kind}_prewhitening.npy", allow_pickle=False)
    rec = recording(expected.T)
    got_rec = preprocess(rec, strategy="none", dtype="float32")
    got = got_rec.get_traces().T
    integer_rejected = False
    try:
        preprocess(recording(np.load(folder / f"{kind}_integer_input.npy", allow_pickle=False).T), strategy="none")
    except DSPreprocessingError:
        integer_rejected = True
    result = {
        "kind": kind,
        "array_equal": bool(np.array_equal(got, expected)),
        "max_abs_error": float(np.max(np.abs(got - expected))),
        "channel_ids_equal": bool(np.array_equal(got_rec.channel_ids, rec.channel_ids)),
        "geometry_equal": bool(np.array_equal(got_rec.get_channel_locations(), rec.get_channel_locations())),
        "integer_none_rejected": integer_rejected,
    }
    (folder / f"{kind}_consumer.json").write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    if not all((result["array_equal"], result["channel_ids_equal"], result["geometry_equal"], integer_rejected)):
        raise SystemExit(1)


if __name__ == "__main__":
    consume(sys.argv[1], Path(sys.argv[2]))
