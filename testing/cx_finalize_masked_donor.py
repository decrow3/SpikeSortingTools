#!/usr/bin/env python3
"""Saved-output closeout for CX; no qualification recomputation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("output", type=Path)
    args = ap.parse_args()
    per = pd.read_csv(args.output / "PER_DONOR_MODIFIED_QUALIFICATION.csv")
    robust = per[per.modified_source_qualified & per.swapped_half_qualified].copy()
    robust.to_csv(args.output / "ROBUST_EXPLORATORY_MODIFIED_SET.csv", index=False)
    resource = {
        "status": "complete", "conservative_h1_charge_s": 180,
        "cpu_all_work_cap_s": 1200, "threads": 2, "readers": 1,
        "ram_cap_bytes": 20_000_000_000, "gpu_s": 0, "raw_voltage_bytes": 0,
        "saved_donor_source_bytes": 46_698_743, "saved_donor_read_cap_bytes": 268_435_456,
        "scratch_cap_bytes": 2_000_000_000, "final_cap_bytes": 500_000_000,
        "h1_cumulative_before_cv_cw_cx_s": 16_343.22,
        "h1_cv_cw_preclose_charge_s": 150,
        "h1_cx_charge_s": 180,
        "h1_cumulative_after_cx_before_cw_final_s": 16_673.22,
        "extended_ceiling_s": 20_500,
    }
    (args.output / "RESOURCE_RECEIPT.json").write_text(json.dumps(resource, indent=2, sort_keys=True) + "\n")
    files = []
    for path in sorted(args.output.rglob("*")):
        if path.is_file() and path.name not in ("MANIFEST.json", "COMPLETE.json"):
            files.append({"path": str(path.relative_to(args.output)), "bytes": path.stat().st_size, "sha256": sha256(path)})
    manifest = {"files": files, "total_bytes": sum(x["bytes"] for x in files)}
    (args.output / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    complete = {
        "status": "complete", "manifest_sha256": sha256(args.output / "MANIFEST.json"),
        "preserved_old_qualified": 5, "modified_qualified": int(per.modified_source_qualified.sum()),
        "swapped_qualified": int(per.swapped_half_qualified.sum()), "robust_exploratory_set": len(robust),
        "verdict": "25-donor dual-half exploratory engineering set; benchmark scorer needs background-provenance repair before launch",
    }
    (args.output / "COMPLETE.json").write_text(json.dumps(complete, indent=2, sort_keys=True) + "\n")
    print(json.dumps(complete, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
