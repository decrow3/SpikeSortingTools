#!/usr/bin/env python3
"""Backfill the established standard-QC stage for the completed rigid arm."""

from __future__ import annotations

import json
from pathlib import Path

from pipeline.downstream import run_standard_qc_stage


SOURCE = Path("/media/huklab/Data/luke_medicine_rigid_sort_20260909_v1")
EXPECTED_IDENTITY = (
    "06ea4b23a45335ef8a8428047cc54e64356905b6cdae6715e36f061f3cfdf555"
)


def main() -> None:
    identity = json.loads((SOURCE / "sort_identity.json").read_text())
    if identity.get("identity_digest") != EXPECTED_IDENTITY:
        raise RuntimeError("completed rigid-arm sort identity changed")
    receipt = run_standard_qc_stage(
        SOURCE / "recording",
        SOURCE / "cur/cur_output",
        SOURCE / "qc",
        SOURCE / "qc/standard",
        identity,
    )
    if receipt.get("complete") is not True:
        raise RuntimeError("standard-QC stage did not complete")
    if receipt.get("sort_identity_digest") != EXPECTED_IDENTITY:
        raise RuntimeError("standard-QC receipt has the wrong sort identity")
    print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
