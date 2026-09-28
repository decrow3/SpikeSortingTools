"""Run only DC's native-bank control through the frozen DA W2 RF evaluator."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import resource
import subprocess
import time
from pathlib import Path


ROOT = Path("/home/huklab/Documents/RyanSorting/SpikeSortingTools")
PACKET = Path("/mnt/NPX/Luke/DARTsort_motion_experiments/dc_motion_domain_and_native_bank_review_20260928")
OUT = PACKET / "native_rf_run"
ARM = Path("/mnt/NPX/Luke/DARTsort_motion_experiments/dc_native_bank_control_20260928/NATIVE_REMATCH0/dartsort_sorting.npz")
EVALUATOR = ROOT / "testing/cp_w2_rf_evaluator.py"
PYTHON = Path("/home/huklab/Documents/DataRowleyV1V2/DataRowleyV1V2/.venv/bin/python")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def main() -> None:
    expected_arm_hash = "6bbdc7442781d56c4991bb81344451b60e9bd9da6a434ce7bc49509206a8b15a"
    if sha256(ARM) != expected_arm_hash:
        raise ValueError("Native-bank final sort hash mismatch")
    if OUT.exists() or OUT.with_name(OUT.name + ".partial").exists():
        raise FileExistsError("native RF output already exists")
    started_wall = dt.datetime.now().astimezone()
    atomic_json(PACKET / "NATIVE_RF_START_RECEIPT.json", {
        "status": "started",
        "started_local": started_wall.isoformat(),
        "expected_elapsed_minutes": [1, 2],
        "arm": "NATIVE_REMATCH0",
        "arm_sha256": expected_arm_hash,
        "evaluator": str(EVALUATOR),
        "evaluator_sha256": sha256(EVALUATOR),
        "scope": "new arm only; unchanged frozen DA development evaluator; outer holdout sealed",
        "limits": {"threads": 2, "readers": 1, "gpu": False, "raw_voltage_reads": 0},
    })
    env = dict(os.environ)
    env.update({
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
        "OPENBLAS_NUM_THREADS": "2",
        "NUMEXPR_NUM_THREADS": "2",
    })
    command = [
        str(PYTHON), str(EVALUATOR),
        "--arm", f"NATIVE_REMATCH0={ARM}",
        "--config", "/media/huklab/Data/NPX/Ryansorting/Luke/DARTsort_motion_experiments/jobs/luke0804-imec1-full-session-dots-rf-v1/source/dots_rf_config.json",
        "--stimulus-cache", "/media/huklab/Data/NPX/Ryansorting/Luke/DARTsort_motion_experiments/luke0804-imec1-dots-rf/stimulus_cache_v1",
        "--gaze-csv", "/media/huklab/Data/NPX/Ryansorting/Luke/DARTsort_motion_experiments/luke0804-imec1-dots-rf/gaze_calibration_development_v2/data_root/processed_declan/Luke_2025-08-04/dpi_calibration/right_eye/calibrated_dpi.csv",
        "--interval-metadata", str(ROOT / "testing/inputs/da_rematching_development_rf_20260928/W2_INTERVAL_METADATA.json"),
        "--output", str(OUT), "--device", "cpu",
    ]
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    t0 = time.perf_counter()
    with (PACKET / "native_rf.log").open("w") as log:
        completed = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
    elapsed = time.perf_counter() - t0
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    atomic_json(PACKET / "NATIVE_RF_RUNTIME.json", {
        "status": "complete" if completed.returncode == 0 else "failed",
        "returncode": completed.returncode,
        "elapsed_wall_s": elapsed,
        "user_cpu_s": after.ru_utime - before.ru_utime,
        "system_cpu_s": after.ru_stime - before.ru_stime,
        "finished_local": dt.datetime.now().astimezone().isoformat(),
    })
    if completed.returncode:
        raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
