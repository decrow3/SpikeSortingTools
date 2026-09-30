#!/usr/bin/env python3
"""Publish or verify the compact raw XR sorting inputs required by EN Tier 1.

The packet contains spike times, cluster assignments, saved spike positions,
and Kilosort labels.  It deliberately excludes voltage, templates, features,
waveforms, and sorter scratch.  Publication uses a sibling partial directory
and an atomic rename; ``COMPLETE.json`` is written last inside the staged tree.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any


SCHEMA = "en-xr-tier1-raw-sort-handoff-v1"
REQUIRED_FILES = (
    "sort_identity.json",
    "summary.json",
    "kilosort4/sorter_output/spike_times.npy",
    "kilosort4/sorter_output/spike_clusters.npy",
    "kilosort4/sorter_output/spike_positions.npy",
    "kilosort4/sorter_output/cluster_KSLabel.tsv",
)


def sha256(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_exclusive(path: Path, value: dict[str, Any]) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def _validate_source_receipts(root: Path, expected_identity: str) -> None:
    identity = json.loads((root / "sort_identity.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    if identity.get("identity_digest") != expected_identity:
        raise RuntimeError("sort_identity.json has the wrong sort identity")
    if summary.get("status") != "complete":
        raise RuntimeError("source summary is not complete")
    if summary.get("sort_identity_digest") != expected_identity:
        raise RuntimeError("source summary has the wrong sort identity")

    declared = identity.get("files", {})
    for filename in ("spike_times.npy", "spike_clusters.npy", "cluster_KSLabel.tsv"):
        receipt = declared.get(filename)
        path = root / "kilosort4/sorter_output" / filename
        if not isinstance(receipt, dict):
            raise RuntimeError(f"sort identity does not bind {filename}")
        if path.stat().st_size != receipt.get("size_bytes") or sha256(path) != receipt.get("sha256"):
            raise RuntimeError(f"source file differs from sort identity: {filename}")


def verify(package: Path, expected_identity: str | None = None) -> dict[str, Any]:
    package = package.resolve()
    manifest_path = package / "MANIFEST.json"
    complete_path = package / "COMPLETE.json"
    manifest = json.loads(manifest_path.read_text())
    complete = json.loads(complete_path.read_text())
    if manifest.get("schema") != SCHEMA or complete.get("schema") != SCHEMA:
        raise ValueError("unsupported XR Tier-1 handoff schema")
    identity = expected_identity or manifest.get("sort_identity_digest")
    if not identity or manifest.get("sort_identity_digest") != identity:
        raise RuntimeError("handoff sort identity is unexpected")
    rows = manifest.get("files")
    if not isinstance(rows, list) or [row.get("path") for row in rows] != list(REQUIRED_FILES):
        raise RuntimeError("handoff inventory is incomplete or out of order")
    for row in rows:
        path = package / row["path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != row.get("bytes") or sha256(path) != row.get("sha256"):
            raise RuntimeError(f"handoff artifact changed: {row['path']}")
    if complete.get("status") != "complete" or complete.get("manifest_sha256") != sha256(manifest_path):
        raise RuntimeError("COMPLETE.json does not bind the manifest")
    _validate_source_receipts(package, identity)
    return manifest


def publish(source: Path, destination: Path, expected_identity: str) -> dict[str, Any]:
    source, destination = source.resolve(), destination.resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to replace existing handoff: {destination}")
    partial = destination.with_name(destination.name + ".partial")
    if partial.exists():
        raise FileExistsError(f"preserve or inspect incomplete prior publication: {partial}")
    missing = [relative for relative in REQUIRED_FILES if not (source / relative).is_file()]
    if missing:
        raise FileNotFoundError("missing required source files: " + ", ".join(missing))
    _validate_source_receipts(source, expected_identity)

    partial.mkdir(parents=True)
    rows: list[dict[str, Any]] = []
    for relative in REQUIRED_FILES:
        target = partial / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / relative, target)
        rows.append({"path": relative, "bytes": target.stat().st_size, "sha256": sha256(target)})
    manifest = {
        "schema": SCHEMA,
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "producer_host": os.uname().nodename,
        "source_root": str(source),
        "sort_identity_digest": expected_identity,
        "total_bytes": sum(row["bytes"] for row in rows),
        "files": rows,
        "scope": "raw sorting inputs for EN Tier 1; no voltage, templates, features, or waveforms",
    }
    _write_json_exclusive(partial / "MANIFEST.json", manifest)
    _write_json_exclusive(
        partial / "COMPLETE.json",
        {
            "schema": SCHEMA,
            "status": "complete",
            "manifest_sha256": sha256(partial / "MANIFEST.json"),
            "written_last_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    verify(partial, expected_identity)
    os.replace(partial, destination)
    return verify(destination, expected_identity)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    publisher = subparsers.add_parser("publish")
    publisher.add_argument("--source", required=True, type=Path)
    publisher.add_argument("--destination", required=True, type=Path)
    publisher.add_argument("--expected-sort-identity", required=True)
    verifier = subparsers.add_parser("verify")
    verifier.add_argument("--package", required=True, type=Path)
    verifier.add_argument("--expected-sort-identity")
    args = parser.parse_args()
    if args.command == "publish":
        result = publish(args.source, args.destination, args.expected_sort_identity)
    else:
        result = verify(args.package, args.expected_sort_identity)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
