#!/usr/bin/env python3
"""Publish the compact EM.1 huklaban5 result without voltage artifacts."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


ATTEMPT_NAMES = (
    "voltage_equivalence_v1",
    "voltage_equivalence_v2",
    "voltage_equivalence_v3",
    "voltage_equivalence_v4",
)
EVIDENCE_NAMES = (
    "SELECTIONS.json",
    "FRAME_ASSIGNMENTS.json",
    "RESULT.json",
    "FAILURE.json",
)
REPOSITORY_FILES = (
    "docs/EM-huklaban5-voltage-equivalence-result-20260928.md",
    "docs/EM-huklaban5-voltage-handoff-20260928.md",
    "docs/EM-lattice-mapping-audit-result-20260928.md",
    "docs/EM-spikeinterface-lattice-equivalence-and-interpolation-comparison-20260928.md",
    "npx_preprocessing/motion/lattice_remap_si.py",
    "testing/em_huklaban5_voltage_equivalence.py",
    "testing/test_em_huklaban5_voltage_equivalence.py",
    "testing/test_lattice_remap_si.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def publish(repo: Path, attempts: Path, destination: Path) -> None:
    partial = destination.with_name(destination.name + ".partial")
    if destination.exists() or partial.exists():
        raise FileExistsError("destination or partial destination already exists")
    partial.mkdir(parents=True)

    for name in ATTEMPT_NAMES:
        source_dir = attempts / name
        if not source_dir.is_dir():
            raise FileNotFoundError(source_dir)
        target_dir = partial / "evidence" / name
        target_dir.mkdir(parents=True)
        copied = 0
        for filename in EVIDENCE_NAMES:
            source = source_dir / filename
            if source.is_file():
                shutil.copy2(source, target_dir / filename)
                copied += 1
        if copied == 0:
            raise RuntimeError(f"no compact evidence in {source_dir}")

    for relative in REPOSITORY_FILES:
        source = repo / relative
        if not source.is_file():
            raise FileNotFoundError(source)
        target = partial / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    files = {}
    for path in sorted(partial.rglob("*")):
        if path.is_file():
            relative = str(path.relative_to(partial))
            files[relative] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    git_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    manifest = {
        "schema": "em-huklaban5-result-packet-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_verdict": "em1_exact_adapter_qualification_fail",
        "stock_nearest_verdict": "fail",
        "em2a_authorized": False,
        "sort_launched": False,
        "voltage_included": False,
        "git_head": git_head,
        "working_tree_snapshot": True,
        "files": files,
    }
    atomic_json(partial / "MANIFEST.json", manifest)
    for relative, record in files.items():
        if sha256(partial / relative) != record["sha256"]:
            raise RuntimeError(f"post-copy hash mismatch: {relative}")
    manifest_sha = sha256(partial / "MANIFEST.json")
    atomic_json(
        partial / "COMPLETE.json",
        {
            "schema": "em-huklaban5-result-complete-v1",
            "status": "complete",
            "scientific_verdict": manifest["scientific_verdict"],
            "manifest_sha256": manifest_sha,
            "files_verified": len(files),
            "voltage_included": False,
            "sort_launched": False,
        },
    )
    partial.replace(destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--attempts", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    publish(args.repo.resolve(), args.attempts.resolve(), args.destination.resolve())


if __name__ == "__main__":
    main()
