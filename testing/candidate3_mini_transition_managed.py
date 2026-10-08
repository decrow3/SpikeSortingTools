#!/usr/bin/env python3
"""Managed, fail-closed launcher for the Candidate-3 synthetic transition arm."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def tree_file_bytes(root: Path) -> int:
    if not root.exists():
        return 0
    total = 0
    for directory, _, names in os.walk(root, followlinks=False):
        for name in names:
            path = Path(directory) / name
            try:
                if not path.is_symlink():
                    total += path.stat().st_size
            except FileNotFoundError:
                continue
    return total


def load_contract(path: Path, expected_sha256: str) -> dict:
    if sha256(path) != expected_sha256:
        raise RuntimeError("run contract hash mismatch")
    contract = json.loads(path.read_text())
    if contract.get("schema") != "candidate3-mini-transition-managed-v2":
        raise ValueError("unexpected contract schema")
    return contract


def verify_preflight(contract_path: Path, contract: dict) -> dict:
    packet = contract_path.parent.resolve(strict=True)
    if contract_path.is_symlink() or not contract_path.is_file():
        raise RuntimeError("contract must be a regular nonsymlink file")
    checks = {}
    for relative, expected in contract["packet_hashes"].items():
        path = packet / relative
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"packet member invalid: {relative}")
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"packet hash mismatch: {relative}")
        checks[str(path)] = actual
    for name, binding in contract["input_hashes"].items():
        path = Path(binding["path"])
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"input invalid: {name}")
        actual = sha256(path)
        if actual != binding["sha256"]:
            raise RuntimeError(f"input hash mismatch: {name}")
        checks[str(path)] = actual
    commit = subprocess.check_output(
        ["git", "-C", contract["dartsort_repo"], "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != contract["dartsort_commit"]:
        raise RuntimeError("DARTsort commit mismatch")
    root = Path(contract["attempt_root"])
    failure = Path(contract["preflight_failure_receipt"])
    manager_failure = Path(contract["preflight_manager_receipt"])
    for path in (root, failure, manager_failure):
        if path.exists() or path.is_symlink():
            raise RuntimeError(f"freshness failure: {path}")
    parent = root.parent.resolve(strict=True)
    expected_parent = Path(contract["attempt_parent"]).resolve(strict=True)
    if root.resolve(strict=False).parent != parent or parent != expected_parent:
        raise RuntimeError("attempt root containment failure")
    ordinary = Path(contract["ordinary_root"])
    if ordinary.is_symlink() or not ordinary.is_dir():
        raise RuntimeError("ordinary root invalid")
    free = os.statvfs(parent).f_bavail * os.statvfs(parent).f_frsize
    if free < contract["minimum_free_bytes"]:
        raise RuntimeError("insufficient free storage")
    return {"hashes": checks, "dartsort_commit": commit, "free_bytes": free}


def terminate_group(process: subprocess.Popen, grace_seconds: int = 30) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def run_child(command: list[str], *, cwd: Path, env: dict, stdout: Path, stderr: Path,
              root: Path, baseline_bytes: int, cap_bytes: int) -> tuple[int, str | None]:
    stop_reason = None
    signal_number = None

    def handle(signum, _frame):
        nonlocal signal_number
        signal_number = signum

    old_term = signal.signal(signal.SIGTERM, handle)
    old_int = signal.signal(signal.SIGINT, handle)
    try:
        with stdout.open("ab", buffering=0) as out, stderr.open("ab", buffering=0) as err:
            process = subprocess.Popen(
                command, cwd=cwd, env=env, stdout=out, stderr=err, start_new_session=True
            )
            while process.poll() is None:
                used = baseline_bytes + tree_file_bytes(root)
                if used > cap_bytes:
                    stop_reason = f"storage_cap_exceeded:{used}>{cap_bytes}"
                    terminate_group(process)
                    break
                if signal_number is not None:
                    stop_reason = f"manager_signal:{signal_number}"
                    terminate_group(process)
                    break
                time.sleep(2)
            return process.wait(), stop_reason
    finally:
        signal.signal(signal.SIGTERM, old_term)
        signal.signal(signal.SIGINT, old_int)


def run(contract_path: Path, contract_sha256: str) -> int:
    contract = load_contract(contract_path, contract_sha256)
    try:
        preflight = verify_preflight(contract_path, contract)
    except BaseException as error:
        failure = Path(contract["preflight_failure_receipt"])
        if not failure.exists() and not failure.is_symlink():
            atomic_json(failure, {"state": "preflight_failed", "error": repr(error)})
        raise

    packet = contract_path.parent.resolve(strict=True)
    root = Path(contract["attempt_root"])
    root.mkdir(mode=0o700, exist_ok=False)
    job = root / "job"
    cache = root / "cache"
    runs = root / "runs"
    report = root / "report"
    combined = root / "combined"
    for path in (job, cache, runs, report, combined):
        path.mkdir()
    transition = runs / "transition"
    prelaunch = {
        "state": "running",
        "wrapper_pid": os.getpid(),
        "contract": str(contract_path),
        "preflight": preflight,
        "transition_command": contract["transition_command"],
        "report_command": contract["report_command"],
        "cap_bytes": contract["cap_bytes"],
        "baseline_bytes": contract["baseline_bytes"],
    }
    atomic_json(job / "PRELAUNCH.json", prelaunch)
    env = os.environ.copy()
    env["NUMBA_CACHE_DIR"] = str(cache)
    transition_command = [
        value.format(packet=str(packet), transition=str(transition))
        for value in contract["transition_command"]
    ]
    code, reason = run_child(
        transition_command,
        cwd=Path(contract["cwd"]),
        env=env,
        stdout=job / "transition.stdout.log",
        stderr=job / "transition.stderr.log",
        root=root,
        baseline_bytes=contract["baseline_bytes"],
        cap_bytes=contract["cap_bytes"],
    )
    if code or reason:
        atomic_json(job / "FINAL_STATUS.json", {
            "state": "failed", "stage": "transition", "returncode": code, "reason": reason
        })
        return code or 70

    (combined / "ordinary").symlink_to(Path(contract["ordinary_root"]), target_is_directory=True)
    (combined / "transition").symlink_to(transition, target_is_directory=True)
    report_command = [
        value.format(packet=str(packet), combined=str(combined), report=str(report))
        for value in contract["report_command"]
    ]
    code, reason = run_child(
        report_command,
        cwd=Path(contract["cwd"]),
        env=env,
        stdout=job / "report.stdout.log",
        stderr=job / "report.stderr.log",
        root=root,
        baseline_bytes=contract["baseline_bytes"],
        cap_bytes=contract["cap_bytes"],
    )
    used = contract["baseline_bytes"] + tree_file_bytes(root)
    state = "complete" if code == 0 and reason is None and used <= contract["cap_bytes"] else "failed"
    atomic_json(job / "FINAL_STATUS.json", {
        "state": state,
        "stage": "complete" if state == "complete" else "report",
        "returncode": code,
        "reason": reason,
        "combined_bytes": used,
        "cap_bytes": contract["cap_bytes"],
        "accounting_json": str(report / "ACCOUNTING.json"),
        "accounting_npz": str(report / "ACCOUNTING.npz"),
    })
    return 0 if state == "complete" else (code or 71)


def finalize(contract_path: Path, contract_sha256: str) -> int:
    contract = load_contract(contract_path, contract_sha256)
    root = Path(contract["attempt_root"])
    target = root / "job" / "MANAGER_FINAL.json" if root.is_dir() else Path(contract["preflight_manager_receipt"])
    value = {
        "state": "manager_final",
        "service_result": os.environ.get("SERVICE_RESULT"),
        "exit_code": os.environ.get("EXIT_CODE"),
        "exit_status": os.environ.get("EXIT_STATUS"),
        "final_status_present": (root / "job" / "FINAL_STATUS.json").is_file(),
    }
    if target.exists() or target.is_symlink():
        raise RuntimeError(f"refusing manager-final overwrite: {target}")
    atomic_json(target, value)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("run", "finalize"))
    parser.add_argument("contract", type=Path)
    parser.add_argument("contract_sha256")
    args = parser.parse_args(argv)
    return (
        run(args.contract, args.contract_sha256)
        if args.mode == "run"
        else finalize(args.contract, args.contract_sha256)
    )


if __name__ == "__main__":
    sys.exit(main())
