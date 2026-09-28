#!/usr/bin/env python3
"""Publish the finite DG H1 source/adapter review and corrected DE figure."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os
import shutil

ROOT = Path(__file__).resolve().parents[1]
GROUPS = [
    (
        ROOT / "testing/outputs/dd_dg_source_and_ks_adapter_review_20260928",
        Path("/mnt/NPX/Luke/DARTsort_motion_experiments/dd_lattice_w2_20260928/host_h1/dg_finite_review"),
    ),
    (
        ROOT / "testing/outputs/de_common_scorecard_revision_v2_figure_20260928",
        Path("/mnt/NPX/Luke/DARTsort_motion_experiments/de_common_outcome_scorecard_20260928/host_h1/revision_v2/figure"),
    ),
]


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def main():
    published = []
    for source, destination in GROUPS:
        destination.mkdir(parents=True, exist_ok=True)
        rows = []
        for src in sorted(p for p in source.iterdir() if p.is_file()):
            dst = destination / src.name
            if dst.exists() and sha(dst) != sha(src):
                raise FileExistsError(f"distinct destination collision: {dst}")
            if not dst.exists():
                partial = dst.with_suffix(dst.suffix + ".partial")
                shutil.copy2(src, partial); os.replace(partial, dst)
            rows.append({"path": src.name, "bytes": dst.stat().st_size, "sha256": sha(dst)})
        atomic_json(destination / "PUBLISH_MANIFEST.json", {
            "status": "complete", "published_utc": datetime.now(timezone.utc).isoformat(),
            "source": str(source), "destination": str(destination), "files": rows,
        })
        published.append({"destination": str(destination), "files": len(rows)})
    resource = {
        "status": "complete", "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "prior_h1_cumulative_active_s": 17993.22,
        "charged_active_s_cx_export_and_dg_review": 150.0,
        "new_h1_cumulative_active_s": 18143.22,
        "ceiling_active_s": 26000.0,
        "remaining_active_s": 7856.78,
        "gpu_s": 0.0, "sorts_launched": 0,
        "dd_prep_cap_s": 2400.0,
        "dd_prep_previously_charged_s": 60.0,
        "dd_prep_dg_review_charged_s": 145.0,
        "dd_prep_remaining_s": 2195.0,
        "note": "Conservative wall/analysis charge; synthetic fixture itself reported under 1 CPU second.",
        "published": published,
    }
    atomic_json(GROUPS[0][1] / "RESOURCE_UPDATE.json", resource)


if __name__ == "__main__": main()
