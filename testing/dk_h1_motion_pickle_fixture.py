#!/usr/bin/env python3
"""Cross-process DARTsort MotionInfo pickle fixture for DK's exact adapter."""
from __future__ import annotations

from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile

import numpy as np

from testing.luke_dh_exact_chunk_motion import ExactChunkLatticeMotion

ROOT = Path(__file__).resolve().parents[1]
DH = ROOT / "testing/outputs/dh_hybrid_truth_contract_20260928"
BC = ROOT / "testing/outputs/luke_au_cpu_preparation/full_probe_extraction_v1/bc_attempt5/full_probe_final_label_templates.npz"
DART_PY = Path("/home/huklab/anaconda3/envs/spike-sort-challengers/bin/python")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify(folder: Path, expected: Path) -> None:
    from dartsort.util.motion import MotionInfo
    info = MotionInfo.try_load(folder)
    if info is None:
        raise RuntimeError("pickle did not load")
    with np.load(expected, allow_pickle=False) as z:
        states = np.asarray(z["states_um"], float)
        centers = np.asarray(z["chunk_center_samples"], int)
        fs = float(z["sampling_frequency_hz"])
    got = info.disp_at_s(centers / fs, info.geom[:, 1], grid=True)
    want = np.broadcast_to(states[None, :], got.shape)
    if not np.array_equal(got, want):
        raise RuntimeError("cross-process loaded motion changed states")
    print(json.dumps({"status": "pass", "grid_shape": list(got.shape), "motion_class": type(info.dredge_motion_est).__name__, "motion_module": type(info.dredge_motion_est).__module__}))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify-folder", type=Path)
    ap.add_argument("--expected", type=Path)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    if args.verify_folder:
        verify(args.verify_folder, args.expected)
        return

    from dartsort.util.motion import MotionInfo
    with np.load(DH / "EXACT_QUERY_MOTION.npz", allow_pickle=False) as z:
        states = np.asarray(z["states_um"], float)
        fs = float(z["sampling_frequency_hz"])
        chunk = int(z["chunk_length_samples"])
        n_samples = int(z["n_samples"])
    with np.load(BC, allow_pickle=False) as z:
        geom = np.asarray(z["geometry_um"], float)[202:384]
    estimate = ExactChunkLatticeMotion(states, sampling_frequency_hz=fs, chunk_length_samples=chunk, n_samples=n_samples)
    motion = MotionInfo.from_motion_est(geom=geom, dredge_motion_est=estimate)
    tmp = Path(tempfile.mkdtemp(prefix="dk_motion_pickle_", dir="/tmp"))
    try:
        motion.save(tmp)
        pickle_path = tmp / "motion.pkl"
        env = dict(os.environ)
        env["NUMBA_CACHE_DIR"] = "/tmp/dk_numba_cache"
        proc = subprocess.run([
            str(DART_PY), "-m", "testing.dk_h1_motion_pickle_fixture",
            "--verify-folder", str(tmp), "--expected", str(DH / "EXACT_QUERY_MOTION.npz"),
        ], cwd=ROOT, env=env, check=True, capture_output=True, text=True)
        child = json.loads(proc.stdout.strip().splitlines()[-1])
        result = {
            "status": "pass_cross_process_pickle_resume",
            "pickle_bytes": pickle_path.stat().st_size,
            "pickle_sha256_ephemeral": sha(pickle_path),
            "child": child,
            "required_runtime_module": "testing.luke_dh_exact_chunk_motion",
            "limitation": "A resumed h5 process must retain this importable module path and independently validate its own installed DARTsort/source hashes.",
        }
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps(result, sort_keys=True))
    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    main()
