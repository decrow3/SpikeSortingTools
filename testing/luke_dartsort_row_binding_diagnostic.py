#!/usr/bin/env python3
"""Narrow NPZ/HDF5 event-row binding diagnostic; reads only times and labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import h5py
import numpy as np


def array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(memoryview(np.ascontiguousarray(array)).cast("B")).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--h5", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    started = time.monotonic()
    args.output.mkdir(parents=True)
    try:
        with np.load(args.npz) as z:
            nt = np.asarray(z["times_samples"], dtype=np.int64)
            nl = np.asarray(z["labels"], dtype=np.int32)
        with h5py.File(args.h5, "r") as h5:
            ht = h5["times_samples"][:]
            hl = h5["labels"][:]
        time_equal = nt == ht
        label_equal = nl == hl
        mismatch = np.flatnonzero(~time_equal | ~label_equal)
        examples = []
        for i in mismatch[:20]:
            examples.append({"row": int(i), "npz_time": int(nt[i]), "h5_time": int(ht[i]),
                             "delta_samples": int(nt[i] - ht[i]),
                             "npz_label": int(nl[i]), "h5_label": int(hl[i])})
        diff = nt[~time_equal] - ht[~time_equal]
        # Exact time-multiset equality tests whether this can be explained by row ordering alone.
        nt_sorted = np.sort(nt)
        ht_sorted = np.sort(ht)
        report = {
            "schema": "dartsort-npz-h5-row-binding-diagnostic-v1",
            "status": "complete",
            "scope": "times_samples and labels only; no saved features or outcomes",
            "rows": int(len(nt)),
            "times_equal_same_row": int(time_equal.sum()),
            "times_mismatch_same_row": int((~time_equal).sum()),
            "labels_equal_same_row": int(label_equal.sum()),
            "labels_mismatch_same_row": int((~label_equal).sum()),
            "either_mismatch_same_row": int(len(mismatch)),
            "time_multiset_equal": bool(np.array_equal(nt_sorted, ht_sorted)),
            "npz_times_sha256_array_bytes": array_sha256(nt),
            "h5_times_sha256_array_bytes": array_sha256(ht),
            "npz_labels_sha256_array_bytes": array_sha256(nl),
            "h5_labels_sha256_array_bytes": array_sha256(hl),
            "time_difference_on_mismatches": {
                "min_samples": int(diff.min()) if len(diff) else None,
                "max_samples": int(diff.max()) if len(diff) else None,
                "median_samples": float(np.median(diff)) if len(diff) else None,
                "mean_samples": float(np.mean(diff)) if len(diff) else None,
                "unique_delta_count": int(np.unique(diff).size) if len(diff) else 0,
            },
            "first_mismatches": examples,
            "elapsed_s": time.monotonic() - started,
        }
        atomic_json(args.output / "DIAGNOSTIC.json", report)
        atomic_json(args.output / "COMPLETE.json", {"status": "complete", "diagnostic_sha256": hashlib.sha256((args.output / "DIAGNOSTIC.json").read_bytes()).hexdigest(), "complete": True})
    except Exception as exc:
        atomic_json(args.output / "FAILED.json", {"status": "failed", "error_type": type(exc).__name__, "error": str(exc), "elapsed_s": time.monotonic() - started})
        raise


if __name__ == "__main__":
    main()
