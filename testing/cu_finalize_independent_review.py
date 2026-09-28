#!/usr/bin/env python3
"""Finalize the bounded H1 CU independent-review packet."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("output", type=Path)
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--h5-packet", type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.report, args.output / "README.md")
    products = []
    for path in sorted(args.output.iterdir()):
        if not path.is_file() or path.name in {"MANIFEST.json", "COMPLETE.json"}:
            continue
        products.append(
            {"path": path.name, "bytes": path.stat().st_size, "sha256": digest(path)}
        )
    manifest = {
        "status": "complete_qualified_independent_review",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "products": products,
        "h5_packet": str(args.h5_packet),
        "h5_complete_sha256": digest(args.h5_packet / "COMPLETE.json"),
    }
    manifest_path = args.output / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    complete = {
        "status": "complete_qualified_independent_review",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": "sole alignment-displaced pair is single-event-compatible but unproven; no production change",
        "manifest_sha256": digest(manifest_path),
        "readme_sha256": digest(args.output / "README.md"),
        "written_last": True,
    }
    (args.output / "COMPLETE.json").write_text(
        json.dumps(complete, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
