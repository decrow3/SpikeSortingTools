#!/usr/bin/env python3
"""Persistently hand a successful full-session sort to EM.2 post-validation."""
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
DEFAULT_SORT_RUN = Path(
    "/home/huklaban5/DARTsort_experiment_scratch/"
    "em2f_full_20260929/rounded_kriging_v1"
)
DEFAULT_SORT_UNIT = "em2b-3d54ab4c99beb6ff.service"
BOUND_RELATIVE_INPUTS = (
    "testing/em2g_wait_and_launch_postvalidation.py",
    "testing/em2g_launch_postvalidation.py",
    "testing/em2g_full_postvalidation.py",
    "testing/em2g_slice_full_sort.py",
    "testing/em2g_prepare_window_analyses.py",
    "testing/em2b_w2_scorecard.py",
    "testing/em2d_cross_arm_event_overlap.py",
    "testing/em2h_full_session_descriptive.py",
    "configs/em2g_full_slice_validation.v1.json",
    "configs/em2h_full_session_descriptive.v1.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def absolute_preserving_symlinks(path: Path) -> Path:
    """Make a path absolute without dereferencing a virtualenv interpreter."""
    return Path(os.path.abspath(path))


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def bound_inputs() -> dict[str, str]:
    return {relative: sha256(ROOT / relative) for relative in BOUND_RELATIVE_INPUTS}


def validate_bound_inputs(request: dict[str, object]) -> None:
    expected = request.get("bound_inputs_sha256")
    if expected != bound_inputs():
        raise RuntimeError("post-validation source or contract changed after scheduling")


def service_state(unit: str) -> dict[str, str]:
    result = subprocess.run(
        [
            "systemctl",
            "--user",
            "show",
            unit,
            "--property=ActiveState",
            "--property=SubState",
            "--property=MainPID",
            "--property=Result",
            "--property=ExecMainStatus",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise RuntimeError(f"systemctl show failed: {result.stderr.strip()}")
    return dict(
        line.split("=", 1)
        for line in result.stdout.splitlines()
        if "=" in line
    )


def terminal_decision(
    sort_run: Path, state: dict[str, str]
) -> tuple[str, dict[str, object] | None]:
    receipt_path = sort_run / "receipt.json"
    failure_path = sort_run / "failure.json"
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("status") == "complete" and receipt.get("exit_status") == 0:
            return "launch", receipt
        return "fail", {"reason": "unsuccessful terminal receipt", "receipt": receipt}
    if failure_path.is_file():
        return "fail", {
            "reason": "sort failure receipt",
            "failure": json.loads(failure_path.read_text()),
        }
    if state.get("ActiveState") in {"active", "activating", "reloading"}:
        return "wait", None
    return "fail", {
        "reason": "sort service became terminal without a receipt",
        "service_state": state,
    }


def waiter_command(
    python: Path,
    state_dir: Path,
    validation_output: Path,
    validation_job_dir: Path,
    sort_run: Path,
    sort_unit: str,
    unit: str,
    timeout_seconds: int,
) -> list[str]:
    worker = [
        str(python),
        str(Path(__file__).resolve()),
        "wait",
        "--state-dir",
        str(state_dir),
        "--validation-output",
        str(validation_output),
        "--validation-job-dir",
        str(validation_job_dir),
        "--python",
        str(python),
        "--sort-run",
        str(sort_run),
        "--sort-unit",
        sort_unit,
        "--timeout-seconds",
        str(timeout_seconds),
    ]
    return [
        "systemd-run",
        "--user",
        "--unit=" + unit,
        "--property=Type=exec",
        "--property=RuntimeMaxSec=" + str(timeout_seconds + 300),
        "--property=TimeoutStopSec=15",
        "--property=CPUQuota=50%",
        "--property=MemoryMax=1G",
        "--property=TasksMax=32",
        "--property=KillMode=control-group",
        "--property=WorkingDirectory=" + str(ROOT),
        "--property=StandardOutput=append:" + str(state_dir / "stdout.log"),
        "--property=StandardError=append:" + str(state_dir / "stderr.log"),
        *worker,
    ]


def wait_and_launch(args: argparse.Namespace) -> None:
    started = time.time()
    deadline = started + args.timeout_seconds
    try:
        request = json.loads((args.state_dir / "REQUEST.json").read_text())
        validate_bound_inputs(request)
        while True:
            state = service_state(args.sort_unit)
            decision, evidence = terminal_decision(args.sort_run, state)
            atomic_json(
                args.state_dir / "STATUS.json",
                {
                    "schema": "em2g-postvalidation-handoff-status-v1",
                    "status": decision,
                    "service_state": state,
                    "updated_epoch": time.time(),
                    "deadline_epoch": deadline,
                },
            )
            if decision == "fail":
                raise RuntimeError(json.dumps(evidence, sort_keys=True))
            if decision == "launch":
                validate_bound_inputs(request)
                command = [
                    str(args.python),
                    str(ROOT / "testing/em2g_launch_postvalidation.py"),
                    "--python",
                    str(args.python),
                    "--output",
                    str(args.validation_output),
                    "--job-dir",
                    str(args.validation_job_dir),
                ]
                completed = subprocess.run(command, capture_output=True, text=True)
                if completed.returncode:
                    raise RuntimeError(
                        "post-validation launcher failed: "
                        + completed.stderr.strip()
                    )
                atomic_json(
                    args.state_dir / "RECEIPT.json",
                    {
                        "schema": "em2g-postvalidation-handoff-receipt-v1",
                        "status": "dispatched",
                        "exit_status": 0,
                        "sort_receipt_sha256": sha256(args.sort_run / "receipt.json"),
                        "launcher_command": command,
                        "launcher_stdout": completed.stdout.strip(),
                        "started_epoch": started,
                        "finished_epoch": time.time(),
                    },
                )
                return
            if time.time() >= deadline:
                raise TimeoutError("sort did not publish a terminal receipt before deadline")
            time.sleep(30)
    except BaseException as error:
        atomic_json(
            args.state_dir / "FAILURE.json",
            {
                "schema": "em2g-postvalidation-handoff-failure-v1",
                "status": "failed",
                "error": repr(error),
                "traceback": traceback.format_exc(),
                "started_epoch": started,
                "finished_epoch": time.time(),
            },
        )
        raise


def schedule(args: argparse.Namespace) -> None:
    paths = (args.state_dir, args.validation_output, args.validation_job_dir)
    if any(path.exists() for path in paths):
        raise FileExistsError("handoff state and validation destinations must be new")
    python = absolute_preserving_symlinks(args.python)
    if not python.is_file():
        raise FileNotFoundError(python)
    if not args.sort_run.is_dir():
        raise FileNotFoundError(args.sort_run)
    args.state_dir.mkdir(parents=True)
    unit = "em2g-wait-" + hashlib.sha256(
        str(args.validation_output).encode()
    ).hexdigest()[:16]
    command = waiter_command(
        python,
        args.state_dir.resolve(),
        args.validation_output.resolve(),
        args.validation_job_dir.resolve(),
        args.sort_run.resolve(),
        args.sort_unit,
        unit,
        args.timeout_seconds,
    )
    atomic_json(
        args.state_dir / "REQUEST.json",
        {
            "schema": "em2g-postvalidation-handoff-request-v1",
            "service": unit,
            "command": command,
            "sort_run": str(args.sort_run.resolve()),
            "sort_unit": args.sort_unit,
            "bound_inputs_sha256": bound_inputs(),
            "created_epoch": time.time(),
        },
    )
    subprocess.run(["systemctl", "--user", "reset-failed", unit], capture_output=True)
    subprocess.run(command, check=True)
    atomic_json(
        args.state_dir / "DISPATCH.json",
        {"status": "dispatched", "service": unit, "started_epoch": time.time()},
    )
    print(json.dumps({"service": unit, "state_dir": str(args.state_dir)}))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("action", choices=("schedule", "wait"))
    result.add_argument("--state-dir", required=True, type=Path)
    result.add_argument("--validation-output", required=True, type=Path)
    result.add_argument("--validation-job-dir", required=True, type=Path)
    result.add_argument("--python", type=Path, default=Path(sys.executable))
    result.add_argument("--sort-run", type=Path, default=DEFAULT_SORT_RUN)
    result.add_argument("--sort-unit", default=DEFAULT_SORT_UNIT)
    result.add_argument("--timeout-seconds", type=int, default=65000)
    return result


def main() -> None:
    args = parser().parse_args()
    if args.timeout_seconds <= 0:
        raise ValueError("timeout must be positive")
    if args.action == "schedule":
        schedule(args)
    else:
        wait_and_launch(args)


if __name__ == "__main__":
    main()
