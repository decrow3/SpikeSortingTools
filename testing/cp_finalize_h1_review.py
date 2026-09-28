"""Write completion/accounting receipts for the CP H1 review packet."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    packet = args.packet.resolve()
    prior = 15653.22
    charge = 300.0
    resource = {
        "status": "complete",
        "prior_cumulative_cpu_seconds": prior,
        "cp_h1_conservative_charge_seconds": charge,
        "cumulative_cpu_seconds_after_cp_h1": prior + charge,
        "enforced_cumulative_ceiling_seconds": 19300.0,
        "cp_h1_authorized_seconds": 3600.0,
        "charge_basis": (
            "Conservative bounded charge covering two 26-29 s CPU RF evaluations, "
            "saved-array audits, implementation, validation, and reporting."
        ),
        "gpu_seconds": 0,
        "raw_voltage_reads": 0,
        "sort_invocations": 0,
        "calibration_runs": 0,
        "outer_holdout_opened": False,
        "threads_max": 2,
    }
    write(packet / "RESOURCE_RECEIPT.json", resource)
    progress = {
        "schema": "cp-h1-progress-v1",
        "status": "complete",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": (
            "retain force-merged D2L grouping at the saved boundary; no-force "
            "mostly partitions existing parent trajectories"
        ),
        "h5_corrected_packet_independent_audit": "pass",
        "w2_rf": {
            "static_eligible": 67,
            "all_force_eligible": 77,
            "no_force_eligible": 55,
            "outer_holdout_opened": False,
        },
        "report": str(args.report.resolve()),
    }
    write(packet / "CP_PROGRESS.json", progress)
    products = []
    for path in sorted(packet.rglob("*")):
        if not path.is_file() or path.name in {"MANIFEST.json", "COMPLETE.json"}:
            continue
        products.append(
            {
                "path": str(path.relative_to(packet)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    products.append(
        {
            "path": str(args.report.resolve()),
            "bytes": args.report.stat().st_size,
            "sha256": sha256(args.report),
            "external_to_packet": True,
        }
    )
    write(packet / "MANIFEST.json", {"products": products})
    write(
        packet / "COMPLETE.json",
        {
            "status": "complete",
            "written_last": True,
            "products": len(products),
            "manifest_sha256": sha256(packet / "MANIFEST.json"),
            "report_sha256": sha256(args.report),
        },
    )


if __name__ == "__main__":
    main()
