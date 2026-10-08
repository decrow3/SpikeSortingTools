#!/usr/bin/env python3
"""Seal this compact review packet, writing COMPLETE.json strictly last."""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parent
MEMBERS = [
    "PROVENANCE.json",
    "REVIEW_RECEIPT.json",
    "REVIEW_REPORT.md",
    "review_packet.py",
    "SEAL_PACKET.py",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_fsynced(path: Path, payload: dict) -> None:
    with path.open("w") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def main() -> None:
    for stale in (ROOT / "MANIFEST.json", ROOT / "COMPLETE.json"):
        if stale.exists():
            raise SystemExit(f"refusing to overwrite {stale}")
    members = [
        {"path": name, "bytes": (ROOT / name).stat().st_size, "sha256": sha256(ROOT / name)}
        for name in MEMBERS
    ]
    now = datetime.now(ZoneInfo("America/Los_Angeles")).isoformat()
    manifest = {
        "schema": "immutable-packet-manifest-v1",
        "created_local": now,
        "packet": ROOT.name,
        "members": members,
    }
    write_fsynced(ROOT / "MANIFEST.json", manifest)
    manifest_hash = sha256(ROOT / "MANIFEST.json")
    complete = {
        "schema": "immutable-packet-complete-v1",
        "status": "COMPLETE_REVIEW_ACCEPT_NON_DECISIVE_WITH_CAVEAT",
        "completed_local": datetime.now(ZoneInfo("America/Los_Angeles")).isoformat(),
        "manifest_path": "MANIFEST.json",
        "manifest_sha256": manifest_hash,
        "production_launch": False,
        "new_h5_output": False,
        "raw_voltage_or_waveform_read": False,
        "rf_or_holdout_access": False,
        "sort_started": False,
        "parameter_sweep": False,
    }
    write_fsynced(ROOT / "COMPLETE.json", complete)
    if (ROOT / "COMPLETE.json").stat().st_mtime_ns <= (ROOT / "MANIFEST.json").stat().st_mtime_ns:
        raise SystemExit("COMPLETE.json mtime is not after MANIFEST.json")
    print(json.dumps({
        "manifest_sha256": manifest_hash,
        "complete_sha256": sha256(ROOT / "COMPLETE.json"),
        "complete_after_manifest": True,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
