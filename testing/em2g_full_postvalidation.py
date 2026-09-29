#!/usr/bin/env python3
"""Run the frozen full-session validation suite after a successful sort."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]
SORT_RUN = Path(
    "/home/huklaban5/DARTsort_experiment_scratch/em2f_full_20260929/rounded_kriging_v1"
)
MASTER = ROOT / "configs/em2g_full_slice_validation.v1.json"
FULL_DESCRIPTIVE = ROOT / "configs/em2h_full_session_descriptive.v1.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def step_commands(python: Path, output: Path) -> list[tuple[str, list[str]]]:
    slices = output / "slices"
    packets = output / "packets"
    commands = [
        (
            "slice_full_sort",
            [
                str(python),
                str(ROOT / "testing/em2g_slice_full_sort.py"),
                "--contract",
                str(MASTER),
                "--output",
                str(slices),
            ],
        ),
        (
            "prepare_window_packets",
            [
                str(python),
                str(ROOT / "testing/em2g_prepare_window_analyses.py"),
                "--contract",
                str(MASTER),
                "--slices",
                str(slices / "RESULT.json"),
                "--output",
                str(packets),
            ],
        ),
    ]
    for name in ("W2", "W3"):
        commands.extend(
            [
                (
                    f"{name.lower()}_scorecard",
                    [
                        str(python),
                        str(ROOT / "testing/em2b_w2_scorecard.py"),
                        "--contract",
                        str(packets / f"{name}_scorecard_contract.json"),
                        "--arms-manifest",
                        str(packets / f"{name}_scorecard_arms.json"),
                        "--output",
                        str(output / f"{name.lower()}_scorecard"),
                    ],
                ),
                (
                    f"{name.lower()}_event_overlap",
                    [
                        str(python),
                        str(ROOT / "testing/em2d_cross_arm_event_overlap.py"),
                        "--config",
                        str(packets / f"{name}_event_overlap.json"),
                        "--output",
                        str(output / f"{name.lower()}_event_overlap"),
                    ],
                ),
            ]
        )
    commands.append(
        (
            "full_session_descriptive",
            [
                str(python),
                str(ROOT / "testing/em2h_full_session_descriptive.py"),
                "--contract",
                str(FULL_DESCRIPTIVE),
                "--output",
                str(output / "full_session_descriptive"),
            ],
        )
    )
    return commands


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)

    receipt_path = SORT_RUN / "receipt.json"
    if not receipt_path.is_file():
        raise RuntimeError("full-session sort has no terminal receipt; refusing to wait")
    sort_receipt = json.loads(receipt_path.read_text())
    if sort_receipt.get("status") != "complete" or sort_receipt.get("exit_status") != 0:
        raise RuntimeError("full-session sort did not complete successfully")
    for contract in (MASTER, FULL_DESCRIPTIVE):
        if not contract.is_file():
            raise FileNotFoundError(contract)

    output.mkdir(parents=True)
    started = time.time()
    steps = []
    try:
        for name, command in step_commands(args.python, output):
            stage_started = time.time()
            atomic_json(
                output / "RUNNING.json",
                {"status": "running", "stage": name, "command": command, "started_epoch": stage_started},
            )
            stdout_path = output / f"{name}.stdout.log"
            stderr_path = output / f"{name}.stderr.log"
            environment = os.environ.copy()
            environment.update(
                {
                    "OMP_NUM_THREADS": "2",
                    "OPENBLAS_NUM_THREADS": "2",
                    "MKL_NUM_THREADS": "2",
                }
            )
            with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
                completed = subprocess.run(
                    command,
                    cwd=ROOT,
                    env=environment,
                    stdout=stdout,
                    stderr=stderr,
                    check=False,
                )
            step = {
                "name": name,
                "command": command,
                "exit_status": completed.returncode,
                "started_epoch": stage_started,
                "finished_epoch": time.time(),
                "stdout": str(stdout_path),
                "stderr": str(stderr_path),
                "stdout_sha256": sha256(stdout_path),
                "stderr_sha256": sha256(stderr_path),
            }
            steps.append(step)
            atomic_json(output / f"{name}.receipt.json", step)
            if completed.returncode != 0:
                raise RuntimeError(f"post-validation stage failed: {name}")
        (output / "RUNNING.json").unlink(missing_ok=True)
        products = {
            "slices": sha256(output / "slices/RESULT.json"),
            "w2_scorecard": sha256(output / "w2_scorecard/RESULT.json"),
            "w2_event_overlap": sha256(output / "w2_event_overlap/RESULT.json"),
            "w3_scorecard": sha256(output / "w3_scorecard/RESULT.json"),
            "w3_event_overlap": sha256(output / "w3_event_overlap/RESULT.json"),
            "full_session_descriptive": sha256(
                output / "full_session_descriptive/RESULT.json"
            ),
        }
        atomic_json(
            output / "RECEIPT.json",
            {
                "schema": "em2g-full-postvalidation-receipt-v1",
                "status": "complete",
                "exit_status": 0,
                "sort_receipt_sha256": sha256(receipt_path),
                "master_contract_sha256": sha256(MASTER),
                "full_descriptive_contract_sha256": sha256(FULL_DESCRIPTIVE),
                "steps": steps,
                "products": products,
                "started_epoch": started,
                "finished_epoch": time.time(),
                "rf_evaluated": False,
                "voltage_read": False,
            },
        )
    except BaseException as error:
        atomic_json(
            output / "FAILURE.json",
            {
                "schema": "em2g-full-postvalidation-failure-v1",
                "status": "failed",
                "error": repr(error),
                "traceback": traceback.format_exc(),
                "steps": steps,
                "started_epoch": started,
                "finished_epoch": time.time(),
            },
        )
        raise


if __name__ == "__main__":
    main()
