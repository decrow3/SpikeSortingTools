"""Publish or verify the compact evidence needed to compare a completed sort.

This deliberately excludes recording voltage, sorter scratch, templates, and
large waveform feature arrays. Publication is additive: files are copied into
a sibling partial directory, verified there, and exposed with one atomic rename.
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


SCHEMA = "spike-sort-comparison-handoff-v1"
REQUIRED_FILES = (
    "application_contract.json",
    "sort_identity.json",
    "summary.json",
    "cur/curation_receipt.json",
    "cur/cur_output/spike_times.npy",
    "cur/cur_output/spike_clusters.npy",
    "cur/cur_output/full_st.npy",
    "cur/cur_output/kept_spikes.npy",
    "cur/cur_output/cluster_KSLabel.tsv",
    "cur/cur_output/spike_positions.npy",
    "qc/amp_truncation/truncation_qc.npz",
    "qc/qc_receipt.json",
    "qc/standard/standard_qc_receipt.json",
    "qc/matlab_export_receipt.json",
)


def sha256(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def _validate_receipts(root: Path, expected_identity: str) -> None:
    summary = json.loads((root / "summary.json").read_text())
    identity = json.loads((root / "sort_identity.json").read_text())
    if summary.get("status") != "complete":
        raise RuntimeError("source summary is not complete")
    if summary.get("sort_identity_digest") != expected_identity:
        raise RuntimeError("source summary has the wrong sort identity")
    if identity.get("identity_digest") != expected_identity:
        raise RuntimeError("sort_identity.json has the wrong sort identity")
    for relative in (
        "cur/curation_receipt.json",
        "qc/qc_receipt.json",
        "qc/standard/standard_qc_receipt.json",
        "qc/matlab_export_receipt.json",
    ):
        receipt = json.loads((root / relative).read_text())
        if receipt.get("sort_identity_digest") != expected_identity:
            raise RuntimeError(f"{relative} has the wrong sort identity")
        if receipt.get("complete") is not True:
            raise RuntimeError(f"{relative} is not complete")


def verify(package: Path, expected_identity: str | None = None) -> dict[str, Any]:
    package = package.resolve()
    manifest_path = package / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != SCHEMA:
        raise ValueError("unsupported handoff schema")
    identity = expected_identity or manifest.get("sort_identity_digest")
    if not identity or manifest.get("sort_identity_digest") != identity:
        raise RuntimeError("handoff sort identity is unexpected")
    listed = manifest.get("files")
    if not isinstance(listed, list) or [row.get("path") for row in listed] != list(REQUIRED_FILES):
        raise RuntimeError("handoff inventory is incomplete or out of order")
    for row in listed:
        path = package / row["path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        stat = path.stat()
        if stat.st_size != row.get("bytes") or sha256(path) != row.get("sha256"):
            raise RuntimeError(f"handoff artifact changed: {row['path']}")
    _validate_receipts(package, identity)
    return manifest


def publish(source: Path, destination: Path, expected_identity: str) -> dict[str, Any]:
    source, destination = source.resolve(), destination.resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to replace existing handoff: {destination}")
    partial = destination.with_name(destination.name + ".partial")
    if partial.exists():
        raise FileExistsError(f"preserve or remove incomplete prior publication: {partial}")
    missing = [name for name in REQUIRED_FILES if not (source / name).is_file()]
    if missing:
        raise FileNotFoundError("missing required source files: " + ", ".join(missing))
    _validate_receipts(source, expected_identity)
    partial.mkdir(parents=True)
    try:
        rows = []
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
            "scope": "comparison inputs only; no voltage, sorter scratch, templates, or waveform features",
        }
        _write_json(partial / "MANIFEST.json", manifest)
        verify(partial, expected_identity)
        os.replace(partial, destination)
        return verify(destination, expected_identity)
    except BaseException:
        # Preserve partial bytes as evidence and to make interrupted publication visible.
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    publish_parser = sub.add_parser("publish")
    publish_parser.add_argument("--source", required=True, type=Path)
    publish_parser.add_argument("--destination", required=True, type=Path)
    publish_parser.add_argument("--expected-sort-identity", required=True)
    verify_parser = sub.add_parser("verify")
    verify_parser.add_argument("--package", required=True, type=Path)
    verify_parser.add_argument("--expected-sort-identity")
    args = parser.parse_args()
    if args.command == "publish":
        result = publish(args.source, args.destination, args.expected_sort_identity)
    else:
        result = verify(args.package, args.expected_sort_identity)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
