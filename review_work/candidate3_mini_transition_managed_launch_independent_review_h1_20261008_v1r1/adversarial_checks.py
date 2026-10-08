#!/usr/bin/env python3
"""Independent known-answer checks for the Candidate-3 v2 managed launcher."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from testing.candidate3_mini_transition_managed import finalize, sha256, verify_preflight


def make_contract(root: Path) -> tuple[Path, dict, Path, Path]:
    packet = root / "packet"
    packet.mkdir()
    runner = packet / "runner.py"
    runner.write_text("pass\n")
    bound = root / "bound.npy"
    bound.write_bytes(b"bound")
    unbound = root / "unbound.npy"
    unbound.write_bytes(b"original")
    ordinary = root / "ordinary"
    ordinary.mkdir()
    (ordinary / "bound.json").write_text("{}\n")
    unbound_ordinary = ordinary / "threshold.h5"
    unbound_ordinary.write_bytes(b"original-stage-data")
    repo = Path("/home/huklab/Documents/DARTsort")
    commit = subprocess.check_output(["git", "-C", repo, "rev-parse", "HEAD"], text=True).strip()
    contract = {
        "schema": "candidate3-mini-transition-managed-v2",
        "packet_hashes": {"runner.py": sha256(runner)},
        "input_hashes": {"bound": {"path": str(bound), "sha256": sha256(bound)}},
        "dartsort_repo": str(repo),
        "dartsort_commit": commit,
        "attempt_parent": str(root),
        "attempt_root": str(root / "attempt"),
        "preflight_failure_receipt": str(root / "failure.json"),
        "preflight_manager_receipt": str(root / "manager.json"),
        "ordinary_root": str(ordinary),
        "minimum_free_bytes": 1,
    }
    contract_path = packet / "RUN_CONTRACT.json"
    contract_path.write_text(json.dumps(contract, sort_keys=True))
    return contract_path, contract, unbound, unbound_ordinary


def main() -> int:
    results = {}
    with tempfile.TemporaryDirectory(prefix="candidate3-v2-review-") as name:
        root = Path(name)
        contract_path, contract, unbound, unbound_ordinary = make_contract(root)
        verify_preflight(contract_path, contract)

        unbound.write_bytes(b"changed-but-not-checked")
        verify_preflight(contract_path, contract)
        results["unbound_input_change_accepted"] = True

        unbound_ordinary.write_bytes(b"changed-stage-data-and-size")
        verify_preflight(contract_path, contract)
        results["unbound_ordinary_change_accepted"] = True

        bound = Path(contract["input_hashes"]["bound"]["path"])
        bound.write_bytes(b"tampered")
        try:
            verify_preflight(contract_path, contract)
        except RuntimeError as error:
            results["bound_hash_change_rejected"] = "input hash mismatch" in str(error)
        else:
            results["bound_hash_change_rejected"] = False

    with tempfile.TemporaryDirectory(prefix="candidate3-v2-finalizer-") as name:
        root = Path(name)
        contract_path, contract, _, _ = make_contract(root)
        digest = hashlib.sha256(contract_path.read_bytes()).hexdigest()
        old = {key: os.environ.get(key) for key in ("SERVICE_RESULT", "EXIT_CODE", "EXIT_STATUS")}
        os.environ.update(SERVICE_RESULT="timeout", EXIT_CODE="killed", EXIT_STATUS="9")
        try:
            finalize(contract_path, digest)
        finally:
            for key, value in old.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
        receipt = json.loads(Path(contract["preflight_manager_receipt"]).read_text())
        results["finalizer_records_terminal_tuple"] = (
            receipt["service_result"], receipt["exit_code"], receipt["exit_status"]
        ) == ("timeout", "killed", "9")
        results["finalizer_has_timestamp"] = any("time" in key for key in receipt)

    print(json.dumps(results, indent=2, sort_keys=True))
    return 0 if all(
        results[key]
        for key in (
            "unbound_input_change_accepted",
            "unbound_ordinary_change_accepted",
            "bound_hash_change_rejected",
            "finalizer_records_terminal_tuple",
        )
    ) else 1


if __name__ == "__main__":
    sys.exit(main())
