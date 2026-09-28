"""Finalize the compact CR packet with resource and product receipts."""
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
    prior = 15953.22
    charge = 300.0
    write(
        args.packet / "RESOURCE_RECEIPT.json",
        {
            "status": "complete",
            "prior_cumulative_seconds": prior,
            "cr_conservative_charge_seconds": charge,
            "cumulative_seconds_after_cr": prior + charge,
            "enforced_cumulative_ceiling_seconds": 19300.0,
            "authorized_cr_seconds": 1800.0,
            "charge_basis": (
                "Conservative charge covering frozen support construction, 23.46 s "
                "RF execution, independent structural audit, implementation and reporting."
            ),
            "gpu_seconds": 0,
            "raw_voltage_reads": 0,
            "sort_invocations": 0,
            "calibration_runs": 0,
            "outer_holdout_opened": False,
        },
    )
    products = []
    for path in sorted(args.packet.rglob("*")):
        if path.is_file() and path.name not in {"MANIFEST.json", "COMPLETE.json"}:
            products.append(
                {
                    "path": str(path.relative_to(args.packet)),
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
    write(args.packet / "MANIFEST.json", {"products": products})
    write(
        args.packet / "COMPLETE.json",
        {
            "status": "complete",
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "written_last": True,
            "products": len(products),
            "manifest_sha256": sha256(args.packet / "MANIFEST.json"),
            "report_sha256": sha256(args.report),
        },
    )


if __name__ == "__main__":
    main()
