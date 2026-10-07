#!/usr/bin/env python3
"""Validate the v2 DARTsort NPZ↔persistent-HDF5 row-index contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import h5py
import numpy as np


REQUIRED_PERSISTENT = {"denoised_ptp_amplitudes", "point_source_localizations"}


def atomic_json(path: Path, value: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def validate_row_index_binding(npz_path: Path, h5_path: Path, *, full_arrays: bool = True) -> dict:
    npz_path = npz_path.resolve(strict=True)
    h5_path = h5_path.resolve(strict=True)
    with np.load(npz_path) as z:
        parent = (npz_path.parent / Path(str(z["parent_h5_path"].item()))).resolve(strict=True)
        if parent != h5_path:
            raise ValueError(f"NPZ parent_h5_path resolves to {parent}, not {h5_path}")
        names = set(map(str, z["loaded_persistent_features"].tolist()))
        ephemeral = set(map(str, z["ephemeral_feature_names"].tolist()))
        if not REQUIRED_PERSISTENT <= names:
            raise ValueError("required HDF5 features are not declared loaded persistent features")
        if REQUIRED_PERSISTENT & ephemeral or any(name in z.files for name in REQUIRED_PERSISTENT):
            raise ValueError("NPZ unexpectedly overrides a required persistent feature")
        n = int(z["times_samples"].shape[0])
        if z["labels"].shape != (n,) or z["channels"].shape != (n,):
            raise ValueError("NPZ core row shapes differ")
        npz_geom = np.asarray(z["geom"])
        npz_channels = np.asarray(z["channels"]) if full_arrays else None
        npz_times = np.asarray(z["times_samples"]) if full_arrays else None
        npz_labels = np.asarray(z["labels"]) if full_arrays else None
    with h5py.File(h5_path, "r") as h5:
        for name in REQUIRED_PERSISTENT | {"times_samples", "labels", "channels", "geom"}:
            if name not in h5:
                raise ValueError(f"missing HDF5 dataset {name}")
        if h5["times_samples"].shape != (n,) or h5["labels"].shape != (n,) or h5["channels"].shape != (n,):
            raise ValueError("HDF5 core row shapes differ from NPZ")
        if h5["denoised_ptp_amplitudes"].shape != (n,) or h5["point_source_localizations"].shape != (n, 3):
            raise ValueError("persistent feature row count differs from NPZ")
        if not np.array_equal(npz_geom, h5["geom"][:]):
            raise ValueError("NPZ/HDF5 geometry mismatch")
        channels_equal = None
        if full_arrays:
            channels_equal = bool(np.array_equal(npz_channels, h5["channels"][:]))
            if not channels_equal:
                raise ValueError("NPZ/HDF5 channels mismatch; row association rejected")
            if npz_times.dtype.kind != "i" or npz_labels.dtype.kind != "i":
                raise ValueError("NPZ final times/labels are not integer arrays")
    return {
        "rows": n,
        "parent_h5_resolved": str(parent),
        "channels_equal": channels_equal,
        "geometry_equal": True,
        "required_features_persistent_not_overridden": True,
        "loaded_persistent_features": sorted(names),
        "ephemeral_feature_names": sorted(ephemeral),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--h5", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    started = time.monotonic()
    try:
        report = validate_row_index_binding(args.npz, args.h5)
        report.update(schema="dartsort-row-index-binding-validation-v2", status="complete",
                      elapsed_s=time.monotonic() - started)
        atomic_json(args.output / "VALIDATION.json", report)
        digest = hashlib.sha256((args.output / "VALIDATION.json").read_bytes()).hexdigest()
        atomic_json(args.output / "COMPLETE.json", {"status": "complete", "validation_sha256": digest, "complete": True})
    except Exception as exc:
        atomic_json(args.output / "FAILED.json", {"status": "failed", "error_type": type(exc).__name__,
                                                   "error": str(exc), "elapsed_s": time.monotonic() - started})
        raise


if __name__ == "__main__":
    main()
