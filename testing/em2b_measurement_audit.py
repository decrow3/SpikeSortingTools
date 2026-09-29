#!/usr/bin/env python3
"""Audit a completed EM.2b measurement run, including failed publication gates."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


STAGES = ("preprocess", "detect", "motion", "sort", "qc_phy")
COMPACT_FILES = (
    "config.json",
    "environment.json",
    "input-fingerprint.json",
    "input-manifest.json",
    "provenance.json",
    "preprocess-complete.json",
    "detect-complete.json",
    "motion-complete.json",
    "sort-complete.json",
    "qc_phy-complete.json",
    "failure.json",
    "qc-phy/receipt.json",
    "qc-phy/sorting_qc.csv",
    "qc-phy/sorting_qc.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def compare_artifact(entry: dict[str, object]) -> dict[str, object]:
    path = Path(str(entry["resolved_path"]))
    current = {
        "exists": path.is_file(),
        "bytes": path.stat().st_size if path.is_file() else None,
        "mtime_ns": path.stat().st_mtime_ns if path.is_file() else None,
    }
    differences = []
    if not current["exists"]:
        differences.append("missing")
    for key in ("bytes", "mtime_ns"):
        if key in entry and entry[key] != current[key]:
            differences.append(key)
    if path.is_file() and "sha256" in entry:
        current["sha256"] = sha256(path)
        if entry["sha256"] != current["sha256"]:
            differences.append("sha256")
    return {
        "path": entry["path"],
        "recorded": {key: entry[key] for key in ("bytes", "mtime_ns", "sha256") if key in entry},
        "current": current,
        "differences": differences,
    }


def audit_run(run: Path) -> dict[str, object]:
    stages = {}
    differences = []
    for stage in STAGES:
        marker = run / f"{stage}-complete.json"
        if not marker.is_file():
            raise RuntimeError(f"missing completed stage marker: {marker}")
        receipt = json.loads(marker.read_text())
        if receipt.get("status") != "complete" or receipt.get("exit_status", 0) != 0:
            raise RuntimeError(f"stage is not complete: {stage}")
        artifact_rows = [compare_artifact(entry) for entry in receipt.get("artifacts", [])]
        differences.extend(
            {"stage": stage, **row} for row in artifact_rows if row["differences"]
        )
        stages[stage] = {
            "receipt_sha256": sha256(marker),
            "elapsed_seconds": receipt.get("elapsed_seconds"),
            "artifact_count": len(artifact_rows),
        }

    failure_path = run / "failure.json"
    if not failure_path.is_file():
        raise RuntimeError("expected preserved publication failure is absent")
    failure = json.loads(failure_path.read_text())
    expected_difference = (
        len(differences) == 1
        and differences[0]["stage"] == "detect"
        and differences[0]["path"] == "detection/subtraction.h5"
        and differences[0]["differences"] == ["mtime_ns"]
    )
    expected_failure = (
        failure.get("status") == "failed"
        and failure.get("exit_status") == 1
        and str(failure.get("traceback", "")).endswith(
            "RuntimeError: detect: artifacts or inputs changed before publication\n"
        )
    )
    if not expected_difference or not expected_failure:
        raise RuntimeError("run does not match the bounded detection-mtime publication failure")

    qc = json.loads((run / "qc-phy/receipt.json").read_text())
    if qc.get("status") != "complete" or qc.get("rf_evaluated") is not False:
        raise RuntimeError("sorting-only QC receipt is incomplete or accessed RF")
    sorting = run / "sort/dartsort_sorting.npz"
    expected_sorting_sha = qc.get("input_sorting_sha256")
    actual_sorting_sha = sha256(sorting)
    if actual_sorting_sha != expected_sorting_sha:
        raise RuntimeError("saved sorting differs from the QC input")
    detection_h5 = run / "detection/subtraction.h5"
    return {
        "schema": "em2b-measurement-audit-v1",
        "scientific_stages_complete": True,
        "final_launcher_receipt_published": False,
        "publication_failure_class": "detection_hdf5_mtime_changed_during_symlink_reuse",
        "publication_failure_bounded": True,
        "publication_failure_limit": (
            "The detection-stage receipt did not hash the 1.10-GB HDF5, so this audit proves "
            "unchanged size and an mtime-only inventory difference, not byte identity to its "
            "pre-sort state. The sorting and sorting-only QC outputs are independently hashed."
        ),
        "stage_receipts": stages,
        "artifact_differences": differences,
        "current_detection_hdf5": {
            "bytes": detection_h5.stat().st_size,
            "sha256": sha256(detection_h5),
        },
        "sorting": {
            "path": str(sorting),
            "bytes": sorting.stat().st_size,
            "sha256": actual_sorting_sha,
            "units": int(qc["units"]),
            "accepted_events": int(qc["accepted_events"]),
            "excluded_negative_labels": int(qc["excluded_negative_labels"]),
        },
        "qc": {
            "receipt_sha256": sha256(run / "qc-phy/receipt.json"),
            "rf_evaluated": False,
            "capabilities": qc["capabilities"],
        },
        "failure_receipt_sha256": sha256(failure_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--wall-seconds", required=True, type=float)
    parser.add_argument("--cpu-seconds", required=True, type=float)
    parser.add_argument("--peak-scratch-gib", required=True, type=float)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = audit_run(args.run.resolve())
    result["resources"] = {
        "service_wall_seconds": args.wall_seconds,
        "service_cpu_seconds": args.cpu_seconds,
        "observed_peak_scratch_gib": args.peak_scratch_gib,
        "final_scratch_gib": sum(
            path.stat().st_size for path in args.run.rglob("*") if path.is_file()
        ) / 2**30,
        "kernel_io_accounting_available": False,
        "recording_binary_bytes": json.loads(
            (args.run / "qc-phy/receipt.json").read_text()
        )["recording_binary_bytes"],
    }
    args.output.mkdir(parents=True)
    for relative in COMPACT_FILES:
        source = args.run / relative
        if source.is_file():
            target = args.output / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    (args.output / "AUDIT.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    manifest = {
        "schema": "em2b-measurement-packet-v1",
        "status": "complete_with_bounded_publication_failure",
        "files": {
            str(path.relative_to(args.output)): {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in sorted(args.output.rglob("*"))
            if path.is_file() and path.name != "MANIFEST.json"
        },
    }
    (args.output / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
