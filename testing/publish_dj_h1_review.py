#!/usr/bin/env python3
"""Publish the compact DJ review under standing Luke publication authority."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os
import shutil

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "testing/outputs/dj_h1_independent_cpu_review_20260928"
DST = Path("/mnt/NPX/Luke/DARTsort_motion_experiments/dj_h1_independent_cpu_review_20260928")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    complete = json.loads((SRC / "COMPLETE.json").read_text())
    if not str(complete["status"]).startswith("complete"):
        raise RuntimeError("local DJ packet incomplete")
    DST.mkdir(parents=True, exist_ok=True)
    sources = sorted(path for path in SRC.iterdir() if path.is_file() and path.name != "COMPLETE.json")
    sources.append(ROOT / "testing/dj_h1_independent_cpu_review.py")
    rows = []
    for source in sources:
        dest = DST / ("source" if source.parent == ROOT / "testing" else "") / source.name
        dest.parent.mkdir(exist_ok=True)
        if dest.exists() and sha(dest) != sha(source):
            raise FileExistsError(dest)
        if not dest.exists():
            partial = dest.with_suffix(dest.suffix + ".partial")
            shutil.copy2(source, partial)
            os.replace(partial, dest)
        rows.append({"path": str(dest.relative_to(DST)), "bytes": dest.stat().st_size, "sha256": sha(dest)})
    manifest = {"status": "complete", "published_utc": datetime.now(timezone.utc).isoformat(), "files": rows}
    partial = DST / "SHARED_MANIFEST.json.partial"
    partial.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(partial, DST / "SHARED_MANIFEST.json")
    final = {"status": "complete_source_and_one_endpoint", "shared_manifest_sha256": sha(DST / "SHARED_MANIFEST.json"), "written_last": True}
    partial = DST / "COMPLETE.json.partial"
    partial.write_text(json.dumps(final, indent=2, sort_keys=True) + "\n")
    os.replace(partial, DST / "COMPLETE.json")


if __name__ == "__main__":
    main()
