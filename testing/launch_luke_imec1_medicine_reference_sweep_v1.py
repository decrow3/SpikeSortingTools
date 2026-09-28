#!/usr/bin/env python
"""Launch the estimator-only stage-1 sweep as a persistent systemd user service."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unit", required=True)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    unit = args.unit.removesuffix(".service")
    if not unit.startswith("luke-medicine-stage1-") or any(
        char not in "abcdefghijklmnopqrstuvwxyz0123456789-" for char in unit
    ):
        raise ValueError("Invalid unit name")
    job = args.job.resolve()
    job.mkdir(parents=True, exist_ok=False)
    output = args.output.resolve()
    runtime_cache = output / "runtime_cache"
    for cache_name in ("tmp", "numba", "mpl"):
        (runtime_cache / cache_name).mkdir(parents=True, exist_ok=True)
    script = (ROOT / "testing/luke_imec1_medicine_reference_sweep_v1.py").resolve()
    child = [
        "/home/huklab/anaconda3/envs/spikeinterface/bin/python", str(script),
        "all", "--output", str(output),
    ]
    command = [
        "systemd-run", "--user", "--unit=" + unit,
        "--property=Type=exec", "--property=RemainAfterExit=yes",
        "--property=CPUQuota=400%", "--property=MemoryMax=48G",
        "--property=TasksMax=128", "--property=KillMode=control-group",
        "--property=WorkingDirectory=" + str(ROOT), "--property=TimeoutStopSec=30",
        "--property=StandardOutput=append:" + str(job / "stdout.log"),
        "--property=StandardError=append:" + str(job / "stderr.log"),
        "--setenv=OPENBLAS_NUM_THREADS=4", "--setenv=OMP_NUM_THREADS=4",
        "--setenv=MKL_NUM_THREADS=4", "--setenv=PYTHONDONTWRITEBYTECODE=1",
        "--setenv=TMPDIR=" + str(runtime_cache / "tmp"),
        "--setenv=NUMBA_CACHE_DIR=" + str(runtime_cache / "numba"),
        "--setenv=MPLCONFIGDIR=" + str(runtime_cache / "mpl"),
        "--setenv=LUKE_PILOT_DARTSORT_SOURCE=/tmp/dartsort-pilot-source",
        "--setenv=PYTHONPATH=/tmp/dartsort-pilot-source/src", *child,
    ]
    (job / "launch_request.json").write_text(json.dumps({
        "command": command, "child": child, "output": str(output),
        "no_sort": True, "voltage_modified": False,
        "single_fit_timeout_s": 900, "extraction_timeout_s": 3600,
        "gpu_budget_s": 28800,
    }, indent=2) + "\n")
    result = subprocess.run(command, capture_output=True, text=True)
    (job / "launcher_result.json").write_text(json.dumps({
        "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr,
    }, indent=2) + "\n")
    print(result.stdout + result.stderr, flush=True)
    result.check_returncode()


if __name__ == "__main__":
    main()
