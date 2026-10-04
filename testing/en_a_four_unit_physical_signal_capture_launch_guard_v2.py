#!/usr/bin/env python3
"""Fail-closed external-review gate for the bounded A capture.

The GO receipt is deliberately external to the immutable packet it approves.
Every receipt binding is checked for type, path identity, and recomputed digest
before exactly one child process is invoked.  This module never interprets the
capture config or opens a recording.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any


SCHEMA = "en-a-four-unit-physical-signal-capture-h1-go-v2"
BINDING_NAMES = (
    "packet_manifest",
    "config",
    "service",
    "capture_source",
    "validator",
    "governing_contract_manifest",
)
PACKET_MEMBERS = {
    "config": "source/en_a_four_unit_physical_signal_capture.approved.v2.json",
    "service": "source/en_a_four_unit_physical_signal_capture_approved_v2.service",
    "capture_source": "source/en_a_four_unit_physical_signal_capture.py",
    "validator": "source/en_a_four_unit_physical_signal_capture_launch_guard_v2.py",
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class GateError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def argv_sha256(argv: list[str]) -> str:
    payload = json.dumps(argv, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def canonical(path: Path) -> str:
    return os.path.realpath(os.fspath(path))


def require_exact_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise GateError(f"{label} keys differ: expected {sorted(expected)}, got {sorted(value)}")


def parse_manifest(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        parts = line.split("  ", 1)
        if len(parts) != 2 or not SHA256_RE.fullmatch(parts[0]) or not parts[1]:
            raise GateError(f"invalid packet manifest line {line_number}")
        digest, member = parts
        if member in entries:
            raise GateError(f"duplicate packet manifest member: {member}")
        entries[member] = digest
    return entries


def load_and_validate_receipt(
    receipt_path: Path,
    expected_paths: dict[str, Path],
    launch_argv: list[str],
) -> dict[str, Any]:
    if not receipt_path.is_file():
        raise GateError("external GO receipt is absent")
    if canonical(receipt_path) == canonical(expected_paths["packet_manifest"]):
        raise GateError("GO receipt cannot be the packet manifest")
    packet_root = Path(canonical(expected_paths["packet_manifest"])).parent
    receipt_real = Path(canonical(receipt_path))
    if packet_root == receipt_real or packet_root in receipt_real.parents:
        raise GateError("GO receipt must remain outside the packet it approves")
    try:
        receipt = json.loads(receipt_path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise GateError(f"GO receipt is unreadable or malformed: {exc}") from exc
    if not isinstance(receipt, dict):
        raise GateError("GO receipt must be a JSON object")
    require_exact_keys(receipt, {"schema", "verdict", "bindings", "launch"}, "receipt")
    if type(receipt["schema"]) is not str or receipt["schema"] != SCHEMA:
        raise GateError("GO receipt schema differs")
    if type(receipt["verdict"]) is not str or receipt["verdict"] != "GO":
        raise GateError("explicit GO verdict required")
    bindings = receipt["bindings"]
    if not isinstance(bindings, dict):
        raise GateError("bindings must be an object")
    require_exact_keys(bindings, set(BINDING_NAMES), "bindings")

    observed: dict[str, str] = {}
    for name in BINDING_NAMES:
        binding = bindings[name]
        if not isinstance(binding, dict):
            raise GateError(f"binding {name} must be an object")
        require_exact_keys(binding, {"path", "sha256"}, f"binding {name}")
        if type(binding["path"]) is not str or not os.path.isabs(binding["path"]):
            raise GateError(f"binding {name} path must be an absolute string")
        if type(binding["sha256"]) is not str or not SHA256_RE.fullmatch(binding["sha256"]):
            raise GateError(f"binding {name} sha256 must be a lowercase 64-hex string")
        expected_path = expected_paths[name]
        if canonical(Path(binding["path"])) != canonical(expected_path):
            raise GateError(f"binding {name} path differs")
        if not expected_path.is_file():
            raise GateError(f"binding {name} artifact is absent")
        observed[name] = sha256(expected_path)
        if observed[name] != binding["sha256"]:
            raise GateError(f"binding {name} digest differs")

    launch = receipt["launch"]
    if not isinstance(launch, dict):
        raise GateError("launch must be an object")
    require_exact_keys(launch, {"argv", "argv_sha256"}, "launch")
    if not isinstance(launch["argv"], list) or not launch["argv"] or any(type(item) is not str for item in launch["argv"]):
        raise GateError("launch argv must be a nonempty list of strings")
    if type(launch["argv_sha256"]) is not str or not SHA256_RE.fullmatch(launch["argv_sha256"]):
        raise GateError("launch argv_sha256 must be a lowercase 64-hex string")
    if launch["argv"] != launch_argv:
        raise GateError("launch argv differs from the service command")
    if argv_sha256(launch_argv) != launch["argv_sha256"]:
        raise GateError("launch argv digest differs")

    if canonical(expected_paths["validator"]) != canonical(Path(__file__).resolve()):
        raise GateError("validator path does not identify the executing validator")
    packet_entries = parse_manifest(expected_paths["packet_manifest"])
    for name, member in PACKET_MEMBERS.items():
        if packet_entries.get(member) != observed[name]:
            raise GateError(f"packet manifest does not bind actual {name}")
    return {"schema": SCHEMA, "status": "validated", "observed_sha256": observed,
            "launch_argv_sha256": argv_sha256(launch_argv)}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--receipt", required=True, type=Path)
    result.add_argument("--packet-manifest", required=True, type=Path)
    result.add_argument("--config", required=True, type=Path)
    result.add_argument("--service", required=True, type=Path)
    result.add_argument("--capture-source", required=True, type=Path)
    result.add_argument("--validator", required=True, type=Path)
    result.add_argument("--governing-contract-manifest", required=True, type=Path)
    result.add_argument("--validate-only", action="store_true")
    result.add_argument("--launch-command", required=True, nargs=argparse.REMAINDER)
    return result


def main() -> int:
    args = parser().parse_args()
    expected_paths = {
        "packet_manifest": args.packet_manifest,
        "config": args.config,
        "service": args.service,
        "capture_source": args.capture_source,
        "validator": args.validator,
        "governing_contract_manifest": args.governing_contract_manifest,
    }
    try:
        evidence = load_and_validate_receipt(args.receipt, expected_paths, args.launch_command)
        if args.validate_only:
            evidence.update({"recording_open_attempts": 0, "recording_reads": 0,
                             "recording_bytes_read": 0, "child_invocations": 0})
            print(json.dumps(evidence, indent=2, sort_keys=True))
            return 0
        completed = subprocess.run(args.launch_command, check=False)
        return int(completed.returncode)
    except GateError as exc:
        print(json.dumps({"schema": SCHEMA, "status": "refused", "error": str(exc)}, sort_keys=True),
              file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
