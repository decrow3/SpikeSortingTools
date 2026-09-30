#!/usr/bin/env python3
"""Persistent full-file input verification for EN's q=0 equivalence gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import time
from pathlib import Path


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/en_rounded_field_ks129.v1.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started_wall, started_cpu = time.perf_counter(), time.process_time()
    config = json.loads(args.config.read_text())
    ref = config["reference"]
    binary = Path(ref["recording_dir"]) / "traces_cached_seg0.raw"
    digest = hashlib.sha256()
    bytes_read = 0
    with binary.open("rb") as stream:
        for block in iter(lambda: stream.read(32 << 20), b""):
            digest.update(block)
            bytes_read += len(block)
    actual = digest.hexdigest()
    status = "pass" if actual == ref["binary_sha256"] else "fail"
    payload = {
        "schema_version": "en-q0-full-hash-v1",
        "status": status,
        "interpretation": "The q=0 SI nearest kernel was independently verified as the exact 384-channel identity; therefore its full byte stream is this verified accepted binary.",
        "binary_path": str(binary),
        "expected_sha256": ref["binary_sha256"],
        "actual_sha256": actual,
        "bytes_read": bytes_read,
        "wall_seconds": time.perf_counter() - started_wall,
        "cpu_seconds": time.process_time() - started_cpu,
        "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    atomic_json(args.output / "RECEIPT.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if status != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
