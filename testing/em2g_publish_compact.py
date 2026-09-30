#!/usr/bin/env python3
"""Publish the compact, terminal EM.2g result packet with hashes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time


PRODUCTS = (
    "RECEIPT.json",
    "slices/RESULT.json",
    "w2_scorecard/RESULT.json",
    "w2_scorecard/POINT_ESTIMATES.csv",
    "w2_scorecard/COMPARISONS.csv",
    "w2_event_overlap/RESULT.json",
    "w2_event_overlap/full_session_rounded_kriging_RECIPROCAL.csv",
    "w2_event_overlap/full_session_rounded_kriging_UNIT_BEST.csv",
    "w3_scorecard/RESULT.json",
    "w3_scorecard/POINT_ESTIMATES.csv",
    "w3_scorecard/COMPARISONS.csv",
    "w3_event_overlap/RESULT.json",
    "w3_event_overlap/full_session_rounded_kriging_RECIPROCAL.csv",
    "w3_event_overlap/full_session_rounded_kriging_UNIT_BEST.csv",
    "full_session_descriptive/RESULT.json",
    "full_session_descriptive/ARM_SUMMARIES.json",
    "full_session_descriptive/SEGMENT_SAFE_ISI.csv",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def publish(source: Path, report: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(f"refusing conflicting packet: {destination}")
    receipt = json.loads((source / "RECEIPT.json").read_text())
    if receipt.get("status") != "complete" or receipt.get("exit_status") != 0:
        raise RuntimeError("source validation has no successful terminal receipt")
    temporary = destination.with_name(destination.name + ".partial")
    if temporary.exists():
        raise FileExistsError(f"preserved partial packet requires review: {temporary}")
    temporary.mkdir(parents=True)
    copied = []
    for relative in PRODUCTS:
        src, dst = source / relative, temporary / relative
        if not src.is_file():
            raise FileNotFoundError(src)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied.append({"path": relative, "bytes": dst.stat().st_size, "sha256": sha256(dst)})
    report_dst = temporary / report.name
    shutil.copy2(report, report_dst)
    copied.append({"path": report.name, "bytes": report_dst.stat().st_size, "sha256": sha256(report_dst)})
    manifest = {
        "schema": "em2g-compact-publication-v1",
        "status": "complete",
        "source": str(source.resolve()),
        "source_receipt_sha256": sha256(source / "RECEIPT.json"),
        "rf_evaluated": False,
        "outer_holdout_accessed": False,
        "voltage_included": False,
        "products": copied,
        "created_epoch": time.time(),
    }
    manifest_path = temporary / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (temporary / "COMPLETE.json").write_text(
        json.dumps(
            {"status": "complete", "manifest_sha256": sha256(manifest_path)},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    temporary.rename(destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    args = parser.parse_args()
    publish(args.source.resolve(), args.report.resolve(), args.destination.resolve())
    print(args.destination.resolve())


if __name__ == "__main__":
    main()
