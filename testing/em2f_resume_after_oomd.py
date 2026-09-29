#!/usr/bin/env python3
"""Audit and resume the EM.2f full sort after its systemd-oomd interruption."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from testing.em2b_launch import prepare


SCHEMA = "em2f-oomd-recovery-v1"
EVENT_DATASETS = (
    "channels",
    "collisioncleaned_tpca_features",
    "collisioncleaned_voltages",
    "denoised_ptp_amplitudes",
    "labels",
    "point_source_localizations",
    "scalings",
    "scores",
    "template_inds",
    "time_shifts",
    "times_samples",
    "times_seconds",
    "up_inds",
)


def audit_matching(path: Path) -> dict[str, object]:
    import h5py

    with h5py.File(path, "r") as h5:
        lengths = {name: int(h5[name].shape[0]) for name in EVENT_DATASETS}
        chunk_starts = h5["chunk_starts_samples"]
        total_chunks = int(chunk_starts.shape[0])
        last_index = int(h5["last_chunk_index"][()])
        last_start = int(h5["last_chunk_start"][()])
        expected_last_start = int(chunk_starts[last_index])
    unique_lengths = set(lengths.values())
    complete = (
        last_index == total_chunks - 1
        and last_start == expected_last_start
        and len(unique_lengths) == 1
    )
    result = {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "total_chunks": total_chunks,
        "last_chunk_index": last_index,
        "last_chunk_start": last_start,
        "expected_last_chunk_start": expected_last_start,
        "completed_fraction": (last_index + 1) / total_chunks,
        "event_count": next(iter(unique_lengths)) if len(unique_lengths) == 1 else None,
        "event_dataset_lengths": lengths,
        "complete_and_aligned": complete,
    }
    if not complete:
        raise RuntimeError("matching1.h5 is not a complete, aligned checkpoint")
    return result


def service_command(config: dict[str, object], output: Path, job: Path, unit: str) -> list[str]:
    repo = Path(config["repo"]).resolve()
    runtime = int(config["budgets_seconds"]["sort"] + config["budgets_seconds"]["qc_phy"] + 600)
    return [
        "systemd-run",
        "--user",
        "--unit=" + unit,
        "--collect",
        "--property=Type=exec",
        "--property=RuntimeMaxSec=" + str(runtime),
        "--property=TimeoutStopSec=30",
        "--property=CPUQuota=400%",
        "--property=MemoryHigh=128G",
        "--property=MemoryMax=160G",
        "--property=ManagedOOMPreference=avoid",
        "--property=TasksMax=128",
        "--property=KillMode=control-group",
        "--property=WorkingDirectory=" + str(repo),
        "--property=StandardOutput=append:" + str(job / "stdout.log"),
        "--property=StandardError=append:" + str(job / "stderr.log"),
        "--setenv=PYTHONPATH=" + str(repo) + ":" + str(PROJECT_ROOT),
        "--setenv=OMP_NUM_THREADS=1",
        "--setenv=OPENBLAS_NUM_THREADS=1",
        str(repo / ".venv/bin/python"),
        "-u",
        str(output / "source/pipeline.py"),
        "run",
        "--config",
        str(output / "config.json"),
        "--output",
        str(output),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--job", required=True, type=Path)
    args = parser.parse_args()
    output, job = args.output.resolve(), args.job.resolve()
    if job.exists():
        raise FileExistsError("recovery job directory is immutable; choose a fresh path")

    config = prepare(args.config.resolve(), output, resume=True)
    required = (
        "preprocess-complete.json",
        "detect-complete.json",
        "motion-complete.json",
        "sort/matching1.h5",
        "sort/matching1_models/template_data.npz",
    )
    missing = [name for name in required if not (output / name).is_file()]
    if missing:
        raise FileNotFoundError(f"recovery prerequisites missing: {missing}")
    if (output / "receipt.json").exists() or (output / "sort/dartsort_sorting.npz").exists():
        raise RuntimeError("sort is already terminal; recovery refused")

    matching = audit_matching(output / "sort/matching1.h5")
    job.mkdir(parents=True)
    identity = str(output) + "\0" + str(job) + "\0" + SCHEMA
    unit = "em2f-recovery-" + hashlib.sha256(identity.encode()).hexdigest()[:16]
    command = service_command(config, output, job, unit)
    request = {
        "schema": SCHEMA,
        "created_epoch": time.time(),
        "original_service": "em2b-3d54ab4c99beb6ff.service",
        "diagnosed_termination": "systemd-oomd SIGKILL during clustering/refinement",
        "scientific_config_unchanged": True,
        "recovery_behavior": (
            "DARTsort fast-forwards completed matching1 peeling, then reruns "
            "clustering/refinement and finalization"
        ),
        "matching_checkpoint": matching,
        "service": unit,
        "command": command,
    }
    (job / "REQUEST.json").write_text(json.dumps(request, indent=2, sort_keys=True) + "\n")

    state = subprocess.run(
        ["systemctl", "--user", "show", unit, "--property=ActiveState", "--value"],
        capture_output=True,
        text=True,
    )
    if state.stdout.strip() in ("active", "activating", "deactivating"):
        raise RuntimeError("recovery service identity is unexpectedly live")
    subprocess.run(command, check=True)
    dispatch = {"schema": SCHEMA, "service": unit, "dispatched_epoch": time.time(), "command": command}
    (job / "DISPATCH.json").write_text(json.dumps(dispatch, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"service": unit, "job": str(job), "output": str(output)}))


if __name__ == "__main__":
    main()
