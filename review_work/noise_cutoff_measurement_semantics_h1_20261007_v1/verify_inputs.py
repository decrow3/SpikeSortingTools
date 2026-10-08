#!/usr/bin/env python3
"""Bounded independent verification of compact packet integrity and scalar status."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path


ROOT = Path("/mnt/NPX/Luke/DARTsort_motion_experiments")
AUDIT = ROOT / "noise_cutoff_interpretation_audit_h1_20261007_v1"
CENSUS = ROOT / "post_sort_structure_census_h5_20261007_v1"
OUT = Path(__file__).with_name("INPUT_VERIFICATION.json")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_manifest(root: Path) -> dict:
    manifest_path = root / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    entries = manifest.get("members")
    if entries is None:
        entries = [dict(path=k, **v) for k, v in manifest["files"].items()]
    failures = []
    for entry in entries:
        path = root / entry["path"]
        actual_size = path.stat().st_size if path.exists() else None
        actual_hash = sha256(path) if path.exists() else None
        expected_size = entry.get("bytes", entry.get("size_bytes"))
        if actual_size != expected_size or actual_hash != entry["sha256"]:
            failures.append({"path": entry["path"], "expected_size": expected_size,
                             "actual_size": actual_size, "expected_hash": entry["sha256"],
                             "actual_hash": actual_hash})
    complete = json.loads((root / "COMPLETE.json").read_text())
    return {
        "manifest_sha256": sha256(manifest_path),
        "member_count": len(entries),
        "member_failures": failures,
        "complete_manifest_sha256": complete["manifest_sha256"],
        "complete_binds_manifest": complete["manifest_sha256"] == sha256(manifest_path),
    }


def as_bool(value: str) -> bool:
    return value.strip().lower() == "true"


def verify_census() -> dict:
    counts = defaultdict(lambda: {"units": 0, "finite_pass": 0, "finite_fail": 0, "undefined": 0})
    seen = set()
    duplicate_keys = 0
    pass_mismatches = 0
    with (CENSUS / "CENSUS_UNITS.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            key = (row["probe"], row["arm"], row["unit_id"])
            if key in seen:
                duplicate_keys += 1
            seen.add(key)
            raw = row["noise_cutoff"].strip()
            value = float(raw) if raw else math.nan
            expected_pass = math.isfinite(value) and value < 5.0
            if as_bool(row["noise_cutoff_pass"]) != expected_pass:
                pass_mismatches += 1
            if not as_bool(row["in_scoring_domain"]):
                continue
            group = counts[(row["probe"], row["arm"])]
            group["units"] += 1
            if not math.isfinite(value):
                group["undefined"] += 1
            elif value < 5.0:
                group["finite_pass"] += 1
            else:
                group["finite_fail"] += 1
    groups = {}
    for (probe, arm), values in sorted(counts.items()):
        values["finite_fail_prevalence_all_units"] = values["finite_fail"] / values["units"]
        groups[f"{probe}/{arm}"] = values
    gaps = {}
    for probe in ("imec0", "imec1"):
        gaps[probe] = groups[f"{probe}/A"]["finite_fail_prevalence_all_units"] - groups[f"{probe}/REF"]["finite_fail_prevalence_all_units"]
    return {"rows": len(seen), "duplicate_probe_arm_unit_keys": duplicate_keys,
            "saved_full_pass_mismatches": pass_mismatches, "in_domain_groups": groups,
            "arm_A_minus_REF_failure_gap": gaps, "cross_arm_identity_assumed": False}


def verify_availability() -> dict:
    amplitude_rows = []
    with (CENSUS / "AVAILABILITY_LEDGER.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["requested_field"] == "amplitude_distribution":
                amplitude_rows.append({"probe": row["probe"], "arm": row["arm"],
                                       "column_available": row["column_available"], "note": row["note"]})
    return {"amplitude_distribution_rows": amplitude_rows,
            "all_amplitude_distributions_unavailable": bool(amplitude_rows) and all(r["column_available"] == "False" for r in amplitude_rows)}


def main() -> None:
    value = {
        "schema": "noise-cutoff-input-verification-v1",
        "audit_packet": verify_manifest(AUDIT),
        "census_packet": verify_manifest(CENSUS),
        "census_scalar_check": verify_census(),
        "saved_data_check": verify_availability(),
        "raw_voltage_bytes_read": 0,
        "new_h5_output": False,
    }
    value["all_integrity_checks_pass"] = (
        not value["audit_packet"]["member_failures"]
        and value["audit_packet"]["complete_binds_manifest"]
        and not value["census_packet"]["member_failures"]
        and value["census_packet"]["complete_binds_manifest"]
        and value["census_scalar_check"]["duplicate_probe_arm_unit_keys"] == 0
        and value["census_scalar_check"]["saved_full_pass_mismatches"] == 0
        and value["saved_data_check"]["all_amplitude_distributions_unavailable"]
    )
    OUT.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    if not value["all_integrity_checks_pass"]:
        raise SystemExit(1)
    print(json.dumps({"all_integrity_checks_pass": True, "rows": value["census_scalar_check"]["rows"]}))


if __name__ == "__main__":
    main()
