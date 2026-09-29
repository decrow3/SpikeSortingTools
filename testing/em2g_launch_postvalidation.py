#!/usr/bin/env python3
"""Launch completed-sort post-validation in an independent systemd service."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
SORT_RECEIPT = Path(
    "/home/huklaban5/DARTsort_experiment_scratch/"
    "em2f_full_20260929/rounded_kriging_v1/receipt.json"
)


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def absolute_preserving_symlinks(path: Path) -> Path:
    """Make a path absolute without dereferencing a virtualenv interpreter."""
    return Path(os.path.abspath(path))


def service_command(python: Path, output: Path, job: Path, unit: str) -> list[str]:
    return [
        "systemd-run",
        "--user",
        "--unit=" + unit,
        "--property=Type=exec",
        "--property=RuntimeMaxSec=3600",
        "--property=TimeoutStopSec=30",
        "--property=CPUQuota=200%",
        "--property=MemoryMax=24G",
        "--property=TasksMax=64",
        "--property=KillMode=control-group",
        "--property=WorkingDirectory=" + str(ROOT),
        "--property=StandardOutput=append:" + str(job / "stdout.log"),
        "--property=StandardError=append:" + str(job / "stderr.log"),
        "--setenv=OMP_NUM_THREADS=2",
        "--setenv=OPENBLAS_NUM_THREADS=2",
        "--setenv=MKL_NUM_THREADS=2",
        str(python),
        str(ROOT / "testing/em2g_full_postvalidation.py"),
        "--python",
        str(python),
        "--output",
        str(output),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--job-dir", required=True, type=Path)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    args = parser.parse_args()
    output = args.output.resolve()
    job = args.job_dir.resolve()
    if output.exists() or job.exists():
        raise FileExistsError("post-validation output and job directory must be new")
    if not SORT_RECEIPT.is_file():
        raise RuntimeError("full-session sort has no terminal receipt; refusing to wait")
    receipt = json.loads(SORT_RECEIPT.read_text())
    if receipt.get("status") != "complete" or receipt.get("exit_status") != 0:
        raise RuntimeError("full-session sort did not complete successfully")
    python = absolute_preserving_symlinks(args.python)
    if not python.is_file():
        raise FileNotFoundError(python)
    job.mkdir(parents=True)
    unit = "em2g-" + hashlib.sha256(str(output).encode()).hexdigest()[:16]
    state = subprocess.run(
        ["systemctl", "--user", "show", unit, "--property=ActiveState", "--value"],
        capture_output=True,
        text=True,
    )
    if state.stdout.strip() in ("active", "activating", "deactivating"):
        raise RuntimeError("post-validation service is already live")
    subprocess.run(["systemctl", "--user", "reset-failed", unit], capture_output=True)
    command = service_command(python, output, job, unit)
    atomic_json(
        job / "REQUEST.json",
        {
            "schema": "em2g-full-postvalidation-launch-v1",
            "service": unit,
            "command": command,
            "output": str(output),
            "sort_receipt": str(SORT_RECEIPT),
            "sort_receipt_status": receipt["status"],
            "created_epoch": time.time(),
        },
    )
    subprocess.run(command, check=True)
    atomic_json(
        job / "DISPATCH.json",
        {"status": "dispatched", "service": unit, "started_epoch": time.time()},
    )
    print(json.dumps({"service": unit, "output": str(output), "job": str(job)}))


if __name__ == "__main__":
    main()
