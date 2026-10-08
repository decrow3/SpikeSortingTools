#!/usr/bin/env python3
"""Independent bounded review of the sealed noise-cutoff semantics packet."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
from collections import defaultdict
from pathlib import Path


ROOT = Path("/mnt/NPX/Luke/DARTsort_motion_experiments")
PACKET = ROOT / "noise_cutoff_measurement_semantics_h1_20261007_v1"
CENSUS = ROOT / "post_sort_structure_census_h5_20261007_v1"
SOURCE = ROOT / "en_full_session_r1_imec0_rescore_review_request_20261004_v1_h5/source/full_session_r1_evaluation.py"
AUDIT = ROOT / "noise_cutoff_interpretation_audit_h1_20261007_v1"
OUT = Path(__file__).with_name("REVIEW_RECEIPT.json")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_packet() -> dict:
    manifest_path = PACKET / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    failures = []
    for member in manifest["members"]:
        path = PACKET / member["path"]
        actual_bytes = path.stat().st_size if path.exists() else None
        actual_hash = sha256(path) if path.exists() else None
        if actual_bytes != member["bytes"] or actual_hash != member["sha256"]:
            failures.append({
                "path": member["path"],
                "expected_bytes": member["bytes"],
                "actual_bytes": actual_bytes,
                "expected_sha256": member["sha256"],
                "actual_sha256": actual_hash,
            })
    complete = json.loads((PACKET / "COMPLETE.json").read_text())
    listed = {member["path"] for member in manifest["members"]}
    actual = {path.name for path in PACKET.iterdir() if path.is_file()}
    expected = listed | {"MANIFEST.json", "COMPLETE.json"}
    manifest_mtime_ns = manifest_path.stat().st_mtime_ns
    complete_mtime_ns = (PACKET / "COMPLETE.json").stat().st_mtime_ns
    return {
        "manifest_sha256": sha256(manifest_path),
        "complete_sha256": sha256(PACKET / "COMPLETE.json"),
        "complete_declared_manifest_sha256": complete["manifest_sha256"],
        "complete_binds_manifest": complete["manifest_sha256"] == sha256(manifest_path),
        "member_count": len(manifest["members"]),
        "member_failures": failures,
        "unexpected_files": sorted(actual - expected),
        "missing_files": sorted(expected - actual),
        "manifest_mtime_ns": manifest_mtime_ns,
        "complete_mtime_ns": complete_mtime_ns,
        "complete_mtime_after_manifest": complete_mtime_ns > manifest_mtime_ns,
    }


def verify_input_packet(root: Path) -> dict:
    manifest_path = root / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    members = manifest.get("members")
    if members is None:
        members = [dict(path=path, **metadata) for path, metadata in manifest["files"].items()]
    failures = []
    for member in members:
        path = root / member["path"]
        expected_bytes = member.get("bytes", member.get("size_bytes"))
        if not path.exists() or path.stat().st_size != expected_bytes or sha256(path) != member["sha256"]:
            failures.append(member["path"])
    complete = json.loads((root / "COMPLETE.json").read_text())
    return {
        "path": str(root),
        "manifest_sha256": sha256(manifest_path),
        "complete_binds_manifest": complete["manifest_sha256"] == sha256(manifest_path),
        "member_count": len(members),
        "member_failures": failures,
    }


def verify_scalar_table() -> dict:
    counts = defaultdict(lambda: {"units": 0, "finite_pass": 0, "finite_fail": 0, "undefined": 0})
    seen = set()
    duplicates = 0
    pass_mismatches = 0
    with (CENSUS / "CENSUS_UNITS.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            key = (row["probe"], row["arm"], row["unit_id"])
            duplicates += key in seen
            seen.add(key)
            raw = row["noise_cutoff"].strip()
            value = float(raw) if raw else math.nan
            saved_pass = row["noise_cutoff_pass"].strip().lower() == "true"
            expected_pass = math.isfinite(value) and value < 5.0
            pass_mismatches += saved_pass != expected_pass
            if row["in_scoring_domain"].strip().lower() != "true":
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
    for key, value in sorted(counts.items()):
        value["finite_fail_prevalence_all_units"] = value["finite_fail"] / value["units"]
        groups["/".join(key)] = value
    gaps = {
        probe: groups[f"{probe}/A"]["finite_fail_prevalence_all_units"]
        - groups[f"{probe}/REF"]["finite_fail_prevalence_all_units"]
        for probe in ("imec0", "imec1")
    }
    return {
        "row_count": len(seen),
        "duplicate_keys": duplicates,
        "pass_mismatches": pass_mismatches,
        "groups": groups,
        "gaps": gaps,
    }


def verify_fixture_receipt() -> dict:
    receipt = json.loads((PACKET / "FIXTURE_RECEIPT.json").read_text())
    cases = {case["name"]: case for case in receipt["cases"]}
    manual_pass = 4.0 / math.sqrt(8.0 / 3.0)
    manual_fail = (18.0 - 5.0 / 3.0) / math.sqrt(2.0 / 9.0)
    checks = {
        "manual_pass_value": math.isclose(cases["manual_finite_pass"]["actual"]["value"], manual_pass, rel_tol=0, abs_tol=1e-12),
        "manual_fail_value": math.isclose(cases["manual_finite_fail"]["actual"]["value"], manual_fail, rel_tol=0, abs_tol=1e-12),
        "manual_pass_hist": cases["manual_finite_pass"]["actual"]["hist"] == [10, 8, 6, 4, 2],
        "manual_fail_hist": cases["manual_finite_fail"]["actual"]["hist"] == [20, 18, 2, 1, 2],
        "undefined_branches": all(cases[name]["actual"]["value"] is None for name in ("empty", "negative", "zero_tail_variance")),
        "strict_five_fails": cases["strict_threshold_exactly_five"]["actual_pass"] is False,
        "outlier_changes_hist": cases["dynamic_max_outlier_changes_construction"]["core"]["hist"] != cases["dynamic_max_outlier_changes_construction"]["with_outlier"]["hist"],
    }
    return {"checks": checks, "all_passed": all(checks.values())}


def verify_availability() -> dict:
    rows = []
    with (CENSUS / "AVAILABILITY_LEDGER.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["requested_field"] == "amplitude_distribution":
                rows.append(row)
    return {
        "rows": len(rows),
        "all_unavailable": len(rows) == 4 and all(row["column_available"] == "False" for row in rows),
    }


def main() -> None:
    packet = verify_packet()
    scalar = verify_scalar_table()
    fixture = verify_fixture_receipt()
    availability = verify_availability()
    source_hash = sha256(SOURCE)
    audit_input = verify_input_packet(AUDIT)
    census_input = verify_input_packet(CENSUS)
    git_probe = subprocess.run(
        ["git", "cat-file", "-e", "ba21bc97ad177ba5f2a1b79fb69d37f4f3eac3ad^{commit}"],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=False,
    )
    result = {
        "schema": "noise-cutoff-semantics-independent-review-receipt-v1",
        "reviewed_packet": str(PACKET),
        "packet_integrity": packet,
        "scalar_table": scalar,
        "fixture_receipt": fixture,
        "availability": availability,
        "input_packets": {"audit": audit_input, "census": census_input},
        "corroborating_source_sha256": source_hash,
        "exact_accepted_git_object_available": git_probe.returncode == 0,
        "exact_accepted_git_probe_stderr": git_probe.stderr.strip(),
        "scientific_verdict": "ACCEPT_NON_DECISIVE",
        "seal_protocol_verdict": "CAVEAT_COMPLETE_MTIME_PRECEDES_MANIFEST",
        "all_scientific_checks_passed": (
            not packet["member_failures"]
            and packet["complete_binds_manifest"]
            and not packet["unexpected_files"]
            and not packet["missing_files"]
            and scalar["duplicate_keys"] == 0
            and scalar["pass_mismatches"] == 0
            and fixture["all_passed"]
            and availability["all_unavailable"]
            and not audit_input["member_failures"]
            and audit_input["complete_binds_manifest"]
            and not census_input["member_failures"]
            and census_input["complete_binds_manifest"]
            and source_hash == "a31f1349854eff0e4c732063993be37e8cf53de5d206628f66d68a65ff7f341e"
        ),
    }
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "all_scientific_checks_passed": result["all_scientific_checks_passed"],
        "complete_mtime_after_manifest": packet["complete_mtime_after_manifest"],
        "gaps": scalar["gaps"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
