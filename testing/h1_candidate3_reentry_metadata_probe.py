#!/usr/bin/env python3
"""Fail-closed, metadata-only re-entry probe for the H1 project mount.

The live operation is intentionally fixed by a reviewed JSON contract.  It
opens one compact directory, enumerates that directory once, and lstats the
five expected packet members without opening or reading any file content.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import tempfile
import time
from typing import Any


PASS = "PASS_METADATA_GATE"
BLOCKED_TIMEOUT = "BLOCKED_TIMEOUT"
BLOCKED_OPERATION = "BLOCKED_METADATA_OPERATION"
BLOCKED_RECEIPT = "BLOCKED_RECEIPT_WRITE"


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_bound_contract(path: Path, expected_sha256: str) -> tuple[dict[str, Any], str]:
    actual = sha256_path(path)
    if actual != expected_sha256:
        raise ValueError(f"contract hash mismatch: expected {expected_sha256}, got {actual}")
    contract = json.loads(path.read_text(encoding="utf-8"))
    runner_actual = sha256_path(Path(__file__).resolve())
    if contract["runner_sha256"] != runner_actual:
        raise ValueError(
            f"runner hash mismatch: expected {contract['runner_sha256']}, got {runner_actual}"
        )
    return contract, actual


def probe_directory_metadata(target: Path, expected_members: dict[str, int]) -> dict[str, Any]:
    """Perform the entire allowed remote operation; never open member files."""
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    directory_fd = os.open(target, flags)
    try:
        directory_stat = os.fstat(directory_fd)
        if not stat.S_ISDIR(directory_stat.st_mode):
            raise RuntimeError("target is not a directory")
        observed_names = sorted(os.listdir(directory_fd))
        expected_names = sorted(expected_members)
        if observed_names != expected_names:
            raise RuntimeError(
                f"directory membership mismatch: expected {expected_names}, got {observed_names}"
            )
        observed: dict[str, dict[str, int]] = {}
        for name in expected_names:
            member_stat = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            if not stat.S_ISREG(member_stat.st_mode):
                raise RuntimeError(f"member is not a regular file: {name}")
            expected_size = expected_members[name]
            if member_stat.st_size != expected_size:
                raise RuntimeError(
                    f"size mismatch for {name}: expected {expected_size}, got {member_stat.st_size}"
                )
            observed[name] = {
                "size": member_stat.st_size,
                "mode": stat.S_IMODE(member_stat.st_mode),
            }
        return {
            "operation": "open_directory_list_once_lstat_exact_members_no_content_reads",
            "target_mode": stat.S_IMODE(directory_stat.st_mode),
            "observed_members": observed,
        }
    finally:
        os.close(directory_fd)


def write_receipt_atomic(path: Path, receipt: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode("utf-8")
    fd, temporary_name = tempfile.mkstemp(prefix=".receipt.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def run_worker_bounded(command: list[str], timeout_seconds: float) -> dict[str, Any]:
    started = time.monotonic()
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
        return {
            "timed_out": False,
            "worker_pid": process.pid,
            "worker_exit_code": process.returncode,
            "worker_stdout": stdout,
            "worker_stderr": stderr,
            "worker_reaped": True,
            "elapsed_seconds": time.monotonic() - started,
        }
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        worker_reaped = False
        try:
            stdout, stderr = process.communicate(timeout=1.0)
            worker_reaped = True
        except subprocess.TimeoutExpired:
            stdout, stderr = "", "worker remained uninterruptible after SIGKILL"
        return {
            "timed_out": True,
            "worker_pid": process.pid,
            "worker_exit_code": process.poll(),
            "worker_stdout": stdout,
            "worker_stderr": stderr,
            "worker_reaped": worker_reaped,
            "elapsed_seconds": time.monotonic() - started,
        }


def internal_worker(contract: dict[str, Any]) -> int:
    try:
        result = probe_directory_metadata(
            Path(contract["live_target"]), contract["expected_members_bytes"]
        )
    except Exception as exc:  # fail closed across filesystem exception types
        print(json.dumps({"status": BLOCKED_OPERATION, "error": repr(exc)}, sort_keys=True))
        return 20
    print(json.dumps({"status": PASS, "result": result}, sort_keys=True))
    return 0


def live_probe(contract_path: Path, expected_sha256: str, snapshot: str) -> int:
    contract, actual_contract_sha256 = load_bound_contract(contract_path, expected_sha256)
    if snapshot not in contract["snapshots"]:
        raise ValueError(f"snapshot must be one of {contract['snapshots']}")
    receipt_path = Path(contract["receipt_paths"][snapshot])
    if receipt_path.exists():
        raise FileExistsError(f"refusing to overwrite prior receipt: {receipt_path}")
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--internal-worker",
        "--contract",
        str(contract_path.resolve()),
        "--expected-contract-sha256",
        expected_sha256,
    ]
    started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    bounded = run_worker_bounded(command, float(contract["timeout_seconds"]))
    status = BLOCKED_TIMEOUT if bounded["timed_out"] else BLOCKED_OPERATION
    worker_payload: dict[str, Any] | None = None
    if not bounded["timed_out"] and bounded["worker_exit_code"] == 0:
        try:
            worker_payload = json.loads(bounded["worker_stdout"])
            if worker_payload.get("status") == PASS:
                status = PASS
        except (json.JSONDecodeError, AttributeError):
            pass
    receipt = {
        "schema": "h1-candidate3-reentry-metadata-probe-receipt-v1",
        "status": status,
        "snapshot": snapshot,
        "started_utc": started_utc,
        "contract_path": str(contract_path.resolve()),
        "contract_sha256": actual_contract_sha256,
        "runner_sha256": sha256_path(Path(__file__).resolve()),
        "target": contract["live_target"],
        "timeout_seconds": contract["timeout_seconds"],
        "bounded_worker": bounded,
        "worker_payload": worker_payload,
        "interpretation": contract["outcome_interpretation"][status],
    }
    try:
        write_receipt_atomic(receipt_path, receipt)
    except Exception as exc:
        print(json.dumps({"status": BLOCKED_RECEIPT, "error": repr(exc)}, sort_keys=True))
        return 30
    print(json.dumps({"status": status, "receipt": str(receipt_path)}, sort_keys=True))
    return 0 if status == PASS else 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--expected-contract-sha256", required=True)
    parser.add_argument("--snapshot", choices=("snapshot_1", "snapshot_2"))
    parser.add_argument("--internal-worker", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    contract, _ = load_bound_contract(args.contract, args.expected_contract_sha256)
    if args.internal_worker:
        return internal_worker(contract)
    if args.snapshot is None:
        raise SystemExit("--snapshot is required for a live probe")
    return live_probe(args.contract, args.expected_contract_sha256, args.snapshot)


if __name__ == "__main__":
    raise SystemExit(main())
