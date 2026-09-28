"""Convert one accepted DARTsort DREDGE pickle to a portable frozen NPZ."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np


SCHEMA = "luke-dartsort-native-motion-export-v1"
WINDOW_START_S = 930.0
REFERENCE_S = (970.0, 975.0)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.audit.exists():
        raise RuntimeError("export target exists")
    # The pickle is loaded only in its recorded DARTsort environment.
    payload = pickle.loads(args.input.read_bytes())
    if not isinstance(payload, dict) or payload.get("si_motion") is not None:
        raise RuntimeError("unexpected DARTsort motion payload")
    motion = payload.get("dredge_motion_est")
    if motion is None:
        raise RuntimeError("DARTsort export lacks a DREDGE motion estimate")
    relative_time_s = np.asarray(motion.time_bin_centers_s, dtype=float)
    depth_um = np.asarray(motion.spatial_bin_centers_um, dtype=float)
    displacement_um = np.asarray(motion.displacement, dtype=float).T
    geom = np.asarray(payload["geom"], dtype=float)
    if displacement_um.shape != (300, 9):
        raise RuntimeError("unexpected DARTsort motion shape")
    if relative_time_s.shape != (300,) or depth_um.shape != (9,):
        raise RuntimeError("unexpected DARTsort motion axes")
    if not np.allclose(relative_time_s, np.arange(300, dtype=float) + 0.5):
        raise RuntimeError("unexpected DARTsort 1 Hz temporal grid")
    if not np.all(np.diff(depth_um) > 0) or not np.all(np.isfinite(displacement_um)):
        raise RuntimeError("invalid DARTsort motion values")
    time_s = relative_time_s + WINDOW_START_S
    reference = (time_s >= REFERENCE_S[0]) & (time_s < REFERENCE_S[1])
    if reference.sum() != 5:
        raise RuntimeError("unexpected DARTsort reference support")
    offset_um = np.median(displacement_um[reference], axis=0)
    displacement_um = displacement_um - offset_um
    np.savez_compressed(
        args.output,
        time_s=time_s,
        depth_um=depth_um,
        displacement_um=displacement_um,
        reference_offset_um=offset_um,
        geom=geom,
    )
    audit = {
        "schema": SCHEMA,
        "input": str(args.input.resolve()),
        "input_sha256": sha(args.input),
        "output": str(args.output.resolve()),
        "output_sha256": sha(args.output),
        "source_kind": "DARTsort native DREDGE nonrigid motion estimate",
        "source_time_origin_s": 0.0,
        "absolute_window_start_s": WINDOW_START_S,
        "reference_window_s": list(REFERENCE_S),
        "temporal_grid_s": 1.0,
        "time_bins": int(len(time_s)),
        "spatial_bins": int(len(depth_um)),
        "spatial_bin_centers_um": depth_um.tolist(),
        "reference_offset_um": offset_um.tolist(),
        "centered_range_um": [float(displacement_um.min()), float(displacement_um.max())],
        "sign_policy": "preserve DREDGE displacement; corrected depth = observed depth - displacement",
    }
    args.audit.write_text(json.dumps(audit, indent=2) + "\n")


if __name__ == "__main__":
    main()
