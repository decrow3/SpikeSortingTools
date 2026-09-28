#!/usr/bin/env python3
"""Publish completed DK source guidance while endpoint review remains pending."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os
import shutil

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "testing/outputs/dk_h1_source_endpoint_review_20260928.partial"
DST = Path("/mnt/NPX/Luke/DARTsort_motion_experiments/dk_hybrid_w2_20260928/host_h1_source_review_v2")
FILES = [
    SRC / "H5_INTEGRATION_GUIDANCE.md",
    SRC / "H1_INSTALLED_SOURCE.json",
    SRC / "UNKNOWN_BACKGROUND_SCORER_REVIEW.json",
    SRC / "STATE_BOUNDARY_AUDIT.json",
    SRC / "STATE_BOUNDARY_EVENTS.csv",
    SRC / "MOTION_PICKLE_RESUME_FIXTURE.json",
    SRC / "SOURCE_REVIEW_REPORT.md",
    SRC / "SOURCE_REVIEW_RESOURCE_RECEIPT.json",
    ROOT / "testing/dk_h1_boundary_audit.py",
    ROOT / "testing/dk_h1_motion_pickle_fixture.py",
]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    DST.mkdir(parents=True, exist_ok=True)
    rows = []
    for source in FILES:
        dest = DST / source.name
        if dest.exists() and sha(dest) != sha(source):
            raise FileExistsError(dest)
        if not dest.exists():
            partial = dest.with_suffix(dest.suffix + ".partial")
            shutil.copy2(source, partial)
            os.replace(partial, dest)
        rows.append({"path": dest.name, "bytes": dest.stat().st_size, "sha256": sha(dest)})
    manifest = {
        "status": "source_guidance_complete_endpoint_pending",
        "published_utc": datetime.now(timezone.utc).isoformat(),
        "files": rows,
        "dh_packet": "/mnt/NPX/Luke/DARTsort_motion_experiments/dh_hybrid_truth_contract_20260928/host_h1/",
    }
    partial = DST / "MANIFEST.json.partial"
    partial.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(partial, DST / "MANIFEST.json")
    complete = {"status": manifest["status"], "manifest_sha256": sha(DST / "MANIFEST.json"), "written_last": True}
    partial = DST / "COMPLETE.json.partial"
    partial.write_text(json.dumps(complete, indent=2, sort_keys=True) + "\n")
    os.replace(partial, DST / "COMPLETE.json")


if __name__ == "__main__":
    main()
