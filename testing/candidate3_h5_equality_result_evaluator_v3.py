#!/usr/bin/env python3
"""Interpret sealed Candidate 3 equality v3 packets without reading voltage."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def is_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def expected_requests(execution: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for window in execution["windows"]:
        start = window["padded_frames"][0]
        for request in execution["requests"]:
            rows.append({
                "arm": window["name"],
                "request": request["name"],
                "global_frames": [start + request["relative_start"], start + request["relative_end"]],
                "shape": [request["length"], execution["input"]["channels"]],
            })
        if window["name"] == "transition":
            request = execution["transition_extra_request"]
            rows.append({
                "arm": "transition",
                "request": request["name"],
                "global_frames": [request["global_start"], request["global_end"]],
                "shape": [request["length"], execution["input"]["channels"]],
            })
    return rows


def validate_seal(packet: Path, accepted_contract_hash: str) -> tuple[dict[str, Any], dict[str, Any], str]:
    complete_path = packet / "COMPLETE.json"
    manifest_path = packet / "MANIFEST.json"
    complete = json.loads(complete_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    if complete.get("schema") != "immutable-evidence-complete-v1":
        raise ValueError("COMPLETE schema mismatch")
    if complete.get("manifest_sha256") != sha256(manifest_path):
        raise ValueError("MANIFEST hash mismatch")
    if manifest.get("schema") != "immutable-evidence-manifest-v1":
        raise ValueError("MANIFEST schema mismatch")
    if manifest.get("status") != complete.get("status"):
        raise ValueError("MANIFEST/COMPLETE status mismatch")
    members = manifest.get("members")
    if not isinstance(members, dict):
        raise ValueError("MANIFEST members missing")
    actual = {p.name for p in packet.iterdir() if p.is_file()}
    expected = set(members) | {"MANIFEST.json", "COMPLETE.json"}
    if actual != expected:
        raise ValueError(f"packet membership mismatch: {sorted(actual ^ expected)}")
    for name, spec in members.items():
        path = packet / name
        if path.stat().st_size != spec["size"] or sha256(path) != spec["sha256"]:
            raise ValueError(f"member mismatch: {name}")
    terminals = [name for name in ("RESULT.json", "FAILURE.json") if name in members]
    if len(terminals) != 1:
        raise ValueError("packet must contain exactly one terminal member")
    if sha256(packet / "CONTRACT_COPY.json") != accepted_contract_hash:
        raise ValueError("execution contract hash mismatch")
    return manifest, complete, terminals[0]


def outcome(category: str, contract: dict[str, Any], reasons: list[str]) -> dict[str, Any]:
    return {
        "category": category,
        "coordinator_decision": contract["decisions"][category],
        "reasons": reasons,
        "scope": "candidate3 equality v3 packet interpretation only",
    }


def evaluate(packet: Path, contract: dict[str, Any]) -> dict[str, Any]:
    try:
        manifest, complete, terminal = validate_seal(
            packet, contract["accepted_execution_contract_sha256"]
        )
    except Exception as exc:
        return outcome("REJECT_SEAL_OR_SCHEMA_STOP", contract, [str(exc)])
    if terminal == "FAILURE.json":
        try:
            failure = json.loads((packet / terminal).read_text())
        except Exception as exc:
            return outcome("REJECT_SEAL_OR_SCHEMA_STOP", contract, [f"failure JSON invalid: {exc}"])
        if manifest.get("status") != "FAILED_PRESERVED_NO_RETRY":
            return outcome("REJECT_SEAL_OR_SCHEMA_STOP", contract, ["failure seal status mismatch"])
        if failure.get("status") != "FAILED_PRESERVED_NO_RETRY":
            return outcome("REJECT_SEAL_OR_SCHEMA_STOP", contract, ["failure terminal status mismatch"])
        if not isinstance(failure.get("type"), str) or not isinstance(failure.get("message"), str):
            return outcome("REJECT_SEAL_OR_SCHEMA_STOP", contract, ["failure terminal schema mismatch"])
        wall = failure.get("wall_seconds")
        if not isinstance(wall, (int, float)) or not math.isfinite(wall) or wall < 0:
            return outcome("REJECT_SEAL_OR_SCHEMA_STOP", contract, ["failure wall time invalid"])
        return outcome(
            "UPSTREAM_HANDLED_FAILURE_STOP",
            contract,
            [f"{failure.get('type', 'unknown')}: {failure.get('message', 'no message')}"]
        )
    try:
        result = json.loads((packet / "RESULT.json").read_text())
        provenance = json.loads((packet / "PROVENANCE.json").read_text())
        execution = json.loads((packet / "CONTRACT_COPY.json").read_text())
    except Exception as exc:
        return outcome("REJECT_SEAL_OR_SCHEMA_STOP", contract, [str(exc)])

    provenance_errors = []
    if provenance.get("runner_sha256") != contract["accepted_runner_sha256"]:
        provenance_errors.append("runner hash mismatch")
    if provenance.get("sha256") != contract["accepted_source_hashes"]:
        provenance_errors.append("import source hashes mismatch")
    mode = provenance.get("mode")
    if mode not in {"self_test", "real"}:
        provenance_errors.append("unknown provenance mode")
    if manifest.get("status") != result.get("verdict") or complete.get("status") != result.get("verdict"):
        provenance_errors.append("terminal status disagreement")
    if mode == "real":
        source = execution["source"]
        if provenance.get("python") != source["expected_h5_interpreter"]:
            provenance_errors.append("real interpreter mismatch")
        release = provenance.get("release")
        release_keys = (
            "review_complete", "review_complete_sha256", "dispatch_receipt",
            "dispatch_receipt_sha256", "resource_receipt", "resource_receipt_sha256",
        )
        if not isinstance(release, dict) or not all(release.get(k) for k in release_keys):
            provenance_errors.append("real release receipts missing")
        else:
            for key in ("review_complete_sha256", "dispatch_receipt_sha256", "resource_receipt_sha256"):
                if not is_sha256(release[key]):
                    provenance_errors.append(f"invalid release digest: {key}")
            path_keys = ("review_complete", "dispatch_receipt", "resource_receipt")
            if any(not isinstance(release[key], str) for key in path_keys):
                provenance_errors.append("release receipt path is not a string")
                receipt_paths = {}
            else:
                receipt_paths = {key: Path(release[key]).resolve() for key in path_keys}
                if any(not Path(release[key]).is_absolute() for key in path_keys):
                    provenance_errors.append("release receipt path is not absolute")
            if not provenance_errors:
                for path_key, hash_key in (
                    ("review_complete", "review_complete_sha256"),
                    ("dispatch_receipt", "dispatch_receipt_sha256"),
                    ("resource_receipt", "resource_receipt_sha256"),
                ):
                    if not receipt_paths[path_key].is_file() or sha256(receipt_paths[path_key]) != release[hash_key]:
                        provenance_errors.append(f"release receipt file mismatch: {path_key}")
            if not provenance_errors:
                try:
                    review = json.loads(receipt_paths["review_complete"].read_text())
                    dispatch = json.loads(receipt_paths["dispatch_receipt"].read_text())
                    resource = json.loads(receipt_paths["resource_receipt"].read_text())
                except Exception as exc:
                    provenance_errors.append(f"release receipt JSON invalid: {exc}")
                if not provenance_errors and not all(
                    isinstance(value, dict) for value in (review, dispatch, resource)
                ):
                    provenance_errors.append("release receipt top-level schema is not an object")
            if not provenance_errors:
                contract_hash = contract["accepted_execution_contract_sha256"]
                runner_hash = contract["accepted_runner_sha256"]
                if review.get("schema") != execution["release"]["review_schema"] or review.get("status") != execution["release"]["review_status"]:
                    provenance_errors.append("review receipt schema/status mismatch")
                if review.get("reviewed_contract_sha256") != contract_hash or review.get("reviewed_runner_sha256") != runner_hash:
                    provenance_errors.append("review receipt binding mismatch")
                expected_dispatch = {
                    "task": execution["release"]["dispatch_task"],
                    "status": execution["release"]["dispatch_status"],
                    "contract_sha256": contract_hash,
                    "runner_sha256": runner_hash,
                    "review_complete_sha256": release["review_complete_sha256"],
                    "resource_receipt_sha256": release["resource_receipt_sha256"],
                    "interpreter_path": source["expected_h5_interpreter"],
                    "runner_path": source["expected_h5_runner_path"],
                    "contract_path": source["expected_h5_contract_path"],
                    "output_path": source["expected_h5_output_path"],
                    "extra_kilosort_site": source["expected_h5_kilosort_site"],
                    "review_complete_path": str(receipt_paths["review_complete"]),
                    "resource_receipt_path": str(receipt_paths["resource_receipt"]),
                    "execution_mode": "execute-real",
                }
                if any(dispatch.get(k) != v for k, v in expected_dispatch.items()):
                    provenance_errors.append("dispatch receipt binding mismatch")
                resources = execution["resources"]
                expected_resource = {
                    "schema": execution["release"]["resource_receipt_schema"],
                    "status": execution["release"]["resource_receipt_status"],
                    "contract_sha256": contract_hash,
                    "runner_sha256": runner_hash,
                    "memory_max_bytes": resources["memory_max_bytes"],
                    "wall_seconds_max": resources["wall_seconds_max"],
                    "cpu_threads_max": resources["cpu_threads_max"],
                    "temporary_bytes_max": resources["temporary_bytes_max"],
                    "persistent_output_bytes_max": resources["persistent_output_bytes_max"],
                    "gpu": resources["gpu"],
                    "interpreter_path": source["expected_h5_interpreter"],
                    "runner_path": source["expected_h5_runner_path"],
                    "contract_path": source["expected_h5_contract_path"],
                    "output_path": source["expected_h5_output_path"],
                    "extra_kilosort_site": source["expected_h5_kilosort_site"],
                    "outer_receipt_path": source["expected_h5_outer_receipt_path"],
                }
                if any(resource.get(k) != v for k, v in expected_resource.items()) or not resource.get("manager_job_id"):
                    provenance_errors.append("resource receipt binding mismatch")
                expected_argv = [
                    source["expected_h5_runner_path"],
                    "--contract", source["expected_h5_contract_path"],
                    "--expected-contract-sha256", contract_hash,
                    "--output", source["expected_h5_output_path"],
                    "--extra-kilosort-site", source["expected_h5_kilosort_site"],
                    "--review-complete", str(receipt_paths["review_complete"]),
                    "--expected-review-complete-sha256", release["review_complete_sha256"],
                    "--dispatch-receipt", str(receipt_paths["dispatch_receipt"]),
                    "--expected-dispatch-receipt-sha256", release["dispatch_receipt_sha256"],
                    "--resource-receipt", str(receipt_paths["resource_receipt"]),
                    "--expected-resource-receipt-sha256", release["resource_receipt_sha256"],
                    "--execute-real",
                ]
                if provenance.get("argv") != expected_argv:
                    provenance_errors.append("real argv exact binding mismatch")
    if provenance_errors:
        return outcome("REJECT_PROVENANCE_STOP", contract, provenance_errors)

    resource_errors = []
    resources = execution["resources"]
    if result.get("cached_input_bytes") != resources["binary_read_bytes_max"]:
        resource_errors.append("cached input byte count mismatch")
    expected_read = 0 if mode == "self_test" else resources["binary_read_bytes_max"]
    if result.get("binary_read_bytes") != expected_read:
        resource_errors.append("binary read byte count mismatch")
    wall = result.get("wall_seconds")
    temp = result.get("temporary_bytes_final")
    if not isinstance(wall, (int, float)) or not math.isfinite(wall) or wall < 0 or wall > resources["wall_seconds_max"]:
        resource_errors.append("wall bound violation")
    if not isinstance(temp, int) or temp < 0 or temp > resources["temporary_bytes_max"]:
        resource_errors.append("temporary byte bound violation")
    if resource_errors:
        return outcome("REJECT_RESOURCE_OR_BOUNDS_STOP", contract, resource_errors)

    rows = result.get("requests")
    expected = expected_requests(execution)
    primary_errors = []
    if not isinstance(rows, list) or len(rows) != len(expected):
        primary_errors.append("request row count mismatch")
    else:
        for index, (row, exp) in enumerate(zip(rows, expected, strict=True)):
            for key, value in exp.items():
                if row.get(key) != value:
                    primary_errors.append(f"row {index} {key} mismatch")
            if row.get("dtype") != "float32" or row.get("finite") is not True:
                primary_errors.append(f"row {index} dtype/finite mismatch")
            if row.get("mismatch_count") != 0 or row.get("max_abs_error") != 0.0:
                primary_errors.append(f"row {index} exact equality failed")
            if row.get("first_mismatch_index") is not None:
                primary_errors.append(f"row {index} unexpected mismatch index")
            if row.get("lazy_sha256") != row.get("eager_sha256"):
                primary_errors.append(f"row {index} output hashes differ")
            if not is_sha256(row.get("lazy_sha256")) or not is_sha256(row.get("eager_sha256")):
                primary_errors.append(f"row {index} output hash syntax invalid")
    if primary_errors:
        return outcome("REJECT_PRIMARY_MISMATCH_STOP", contract, primary_errors)

    positives = result.get("positive_controls")
    positive_errors = []
    if not isinstance(positives, list) or [p.get("arm") for p in positives] != ["ordinary", "transition"]:
        positive_errors.append("positive-control arms mismatch")
    else:
        for row in positives:
            count, error = row.get("mismatch_count"), row.get("max_abs_error")
            if not isinstance(count, int) or count <= 0:
                positive_errors.append(f"{row.get('arm')} positive mismatch count nonpositive")
            if not isinstance(error, (int, float)) or not math.isfinite(error) or error <= 0:
                positive_errors.append(f"{row.get('arm')} positive max error nonpositive")
    if positive_errors:
        return outcome("REJECT_POSITIVE_CONTROL_NONINFORMATIVE_STOP", contract, positive_errors)

    boundary_expected = {
        "copy_flag": False, "preprocess_same_object": True,
        "workdir_same_object": True, "work_dir": None,
    }
    if result.get("dartsort_boundary") != [boundary_expected, boundary_expected]:
        return outcome("REJECT_DARTSORT_BOUNDARY_STOP", contract, ["DARTsort boundary mismatch"])
    if mode == "self_test" and result.get("verdict") == "PASS_SYNTHETIC_SELF_TEST":
        return outcome("PASS_SYNTHETIC_FIXTURE_ONLY_NO_H5_DECISION", contract, ["all frozen checks passed"])
    if mode == "real" and result.get("verdict") == "PASS_REAL_EQUALITY":
        return outcome("PASS_REAL_EQUALITY_ADVANCE_TO_FROZEN_NEXT_GATE", contract, ["all frozen checks passed"])
    return outcome("REJECT_PROVENANCE_STOP", contract, ["mode/result verdict mismatch"])


def seal(output: Path, verdict: dict[str, Any], provenance: dict[str, Any]) -> None:
    write_json(output / "VERDICT.json", verdict)
    write_json(output / "PROVENANCE.json", provenance)
    members = {
        p.name: {"size": p.stat().st_size, "sha256": sha256(p)}
        for p in sorted(output.iterdir()) if p.is_file()
    }
    write_json(output / "MANIFEST.json", {
        "schema": "immutable-evidence-manifest-v1", "status": verdict["category"], "members": members,
    })
    write_json(output / "COMPLETE.json", {
        "schema": "immutable-evidence-complete-v1", "status": verdict["category"],
        "manifest_sha256": sha256(output / "MANIFEST.json"),
    })


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--contract", required=True)
    p.add_argument("--expected-contract-sha256", required=True)
    p.add_argument("--packet", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    contract_path = Path(args.contract).resolve()
    if sha256(contract_path) != args.expected_contract_sha256:
        raise SystemExit("evaluator contract hash mismatch")
    contract = json.loads(contract_path.read_text())
    packet = Path(args.packet).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    try:
        verdict = evaluate(packet, contract)
    except Exception as exc:
        verdict = outcome(
            "REJECT_SEAL_OR_SCHEMA_STOP",
            contract,
            [f"unexpected evaluator schema error: {type(exc).__name__}: {exc}"],
        )
    seal(output, verdict, {
        "evaluator_sha256": sha256(Path(__file__).resolve()),
        "evaluator_contract_sha256": args.expected_contract_sha256,
        "input_packet": str(packet),
        "input_complete_sha256": sha256(packet / "COMPLETE.json") if (packet / "COMPLETE.json").is_file() else None,
    })
    print(verdict["category"])


if __name__ == "__main__":
    main()
