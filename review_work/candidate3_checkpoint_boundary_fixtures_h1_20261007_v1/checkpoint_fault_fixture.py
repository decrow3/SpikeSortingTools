#!/usr/bin/env python3
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np

from dartsort.peel.peel_base import BasePeeler


class FakePeeler:
    def out_datasets(self):
        return [SimpleNamespace(name="times_samples")]


class FailingDataset:
    chunks = (8,)

    def resize(self, *args, **kwargs):
        raise RuntimeError("frozen injected append failure")


def fresh(path: Path, chunks: np.ndarray):
    with h5py.File(path, "w") as h5:
        h5.create_dataset("last_chunk_start", data=-1, dtype=np.int64)
        h5.create_dataset("last_chunk_index", data=-1, dtype=np.int64)
        h5.create_dataset("chunk_starts_samples", data=chunks)
        h5.create_dataset("times_samples", shape=(0,), maxshape=(None,), chunks=(8,), dtype=np.int64)


def check(fake, path, chunks):
    done, last, _ = BasePeeler.check_resuming(fake, path, chunks)
    return {"done": bool(done), "last_chunk_index": int(last), "next_chunk_index": int(last + 1)}


def main(output: Path):
    fake = FakePeeler()
    chunks = np.array([0, 100], dtype=np.int64)
    fault = output / "fault.h5"
    fresh(fault, chunks)
    injected = False
    with h5py.File(fault, "r+") as h5:
        try:
            BasePeeler.gather_chunk_result(
                fake, cur_n_spikes=0, chunk_start_samples=0,
                chunk_result={"n_spikes": 1, "times_samples": np.array([10], dtype=np.int64)},
                h5_spike_datasets={"times_samples": FailingDataset()}, output_h5=h5,
                residual_file=None, ignore_resuming=False, skip_features=False,
            )
        except RuntimeError as exc:
            injected = str(exc) == "frozen injected append failure"
    fault_resume = check(fake, fault, chunks)
    with h5py.File(fault, "r") as h5:
        fault_rows = len(h5["times_samples"])

    positive = output / "positive.h5"
    fresh(positive, chunks)
    with h5py.File(positive, "r+") as h5:
        BasePeeler.gather_chunk_result(
            fake, cur_n_spikes=0, chunk_start_samples=0,
            chunk_result={"n_spikes": 1, "times_samples": np.array([10], dtype=np.int64)},
            h5_spike_datasets={"times_samples": h5["times_samples"]}, output_h5=h5,
            residual_file=None, ignore_resuming=False, skip_features=False,
        )
    positive_resume = check(fake, positive, chunks)
    mismatch_rejected = False
    try:
        check(fake, positive, np.array([0, 101], dtype=np.int64))
    except ValueError:
        mismatch_rejected = True

    result = {
        "injected_failure_observed": injected,
        "fault_rows_written": fault_rows,
        "fault_resume": fault_resume,
        "false_completed_chunk": bool(injected and fault_rows == 0 and fault_resume["next_chunk_index"] == 1),
        "positive_resume": positive_resume,
        "changed_chunk_starts_rejected": mismatch_rejected,
        "verdict": "CHECKPOINT_UNSAFE_MARKER_PRECEDES_APPEND",
    }
    (output / "checkpoint_result.json").write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    if not (result["false_completed_chunk"] and positive_resume["next_chunk_index"] == 1 and mismatch_rejected):
        raise SystemExit(1)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
