"""Stdlib-only verifier before executing the external monitor source."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--bootstrap-sha256", required=True)
    parser.add_argument("--monitor-source", type=Path, required=True)
    parser.add_argument("--self-sha256", required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--contract-sha256", required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--target-unit-file", type=Path, required=True)
    parser.add_argument("--monitor-unit-file", type=Path, required=True)
    args, forwarded = parser.parse_known_args()
    if sha256(Path(__file__).resolve()) != args.bootstrap_sha256:
        raise RuntimeError("monitor bootstrap self hash mismatch")
    if sha256(args.monitor_source) != args.self_sha256:
        raise RuntimeError("monitor source hash mismatch before execution")
    if sha256(args.contract) != args.contract_sha256:
        raise RuntimeError("contract hash mismatch")
    authorization = json.loads(args.authorization.read_text())
    required = {
        "status": "AUTHORIZED_ONE_FRESH_CANDIDATE2_CACHE_MANAGED_RUN",
        "contract_sha256": args.contract_sha256,
        "target_service_sha256": sha256(args.target_unit_file),
        "monitor_service_sha256": sha256(args.monitor_unit_file),
    }
    if any(authorization.get(key) != value for key, value in required.items()):
        raise RuntimeError("monitor authorization differs before source execution")
    command = [
        sys.executable, "-I", "-B", "-S", "-u", str(args.monitor_source),
        "--self-sha256", args.self_sha256,
        "--contract", str(args.contract), "--contract-sha256", args.contract_sha256,
        "--authorization", str(args.authorization),
        "--target-unit-file", str(args.target_unit_file),
        "--monitor-unit-file", str(args.monitor_unit_file),
        *forwarded,
    ]
    os.execv(sys.executable, command)


if __name__ == "__main__":
    main()
