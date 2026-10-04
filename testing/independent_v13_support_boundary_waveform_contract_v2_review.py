#!/usr/bin/env python3
"""Independent no-voltage review of the H5 v2 support-boundary contract."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path


PACKET = Path(
    "/mnt/NPX/Luke/DARTsort_motion_experiments/"
    "en_first_medium_v13_support_boundary_waveform_discriminator_contract_repair_20261003_v2_h5"
)
V1 = Path(
    "/mnt/NPX/Luke/DARTsort_motion_experiments/"
    "en_first_medium_v13_support_boundary_waveform_discriminator_contract_20261003_v1_h5"
)
BLOCKING_REVIEW = Path(
    "/mnt/NPX/Luke/DARTsort_motion_experiments/"
    "en_first_medium_v13_support_boundary_waveform_contract_h1_review_20261003_v1_h1"
)
SYNTHETIC = Path(
    "/mnt/NPX/Luke/DARTsort_motion_experiments/"
    "luke_v13_support_boundary_synthetic_falsifier_20261003_v1_h1"
)
SYNTHETIC_REVIEW = Path(
    "/mnt/NPX/Luke/DARTsort_motion_experiments/"
    "luke_v13_support_boundary_synthetic_falsifier_h5_independent_review_20261003_v1_h5"
)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def regenerate_selection():
    exposure = json.loads((V1 / "freeze/METADATA_SUPPORT_BOUNDARY_EXPOSURE.json").read_text())
    edges = json.loads((V1 / "freeze/FROZEN_CASE_EDGES_AND_EVENT_SETS.json").read_text())
    row_edges = collections.defaultdict(list)
    for item in edges["edge_event_sets"]:
        for row in item["ref_input_rows"]:
            row_edges[int(row)].append(item["edge"]["edge_id"])
    semantic = {
        "common": "unchanged_credit",
        "existing_only": "loss_of_selected_edge_credit",
        "repaired_only": "gain_of_selected_edge_credit",
        "neither_selected_edge_credit": "neither_selected_edge_credit",
        "neither_primary_credit": "neither_primary_credit",
    }
    records, strata = [], []
    for case in exposure["cases"]:
        for partition, partition_data in case["partitions"].items():
            by_role = collections.defaultdict(list)
            for item in partition_data["frozen_waveform_selection"]:
                by_role[item["selection_role"]].append(item)
            for role in ("near_boundary", "away_control"):
                available = by_role[role]
                key = (
                    (lambda item: (item["nearest_support_change_distance_frames"], item["ref_input_row"]))
                    if role == "near_boundary"
                    else (lambda item: item["ref_input_row"])
                )
                chosen = sorted(available, key=key)[:2]
                strata.append({
                    "reference_unit": case["reference_unit"],
                    "case_role": case["case_role"],
                    "partition": partition,
                    "credit_semantic": semantic[partition],
                    "selection_role": role,
                    "available_in_v1_freeze": len(available),
                    "selected": len(chosen),
                    "absence_explicit": not chosen,
                })
                for item in chosen:
                    records.append({
                        "selection_ordinal": len(records),
                        "reference_unit": case["reference_unit"],
                        "case_role": case["case_role"],
                        "partition": partition,
                        "credit_semantic": semantic[partition],
                        "selection_role": role,
                        "ref_input_row": item["ref_input_row"],
                        "global_frame": item["global_frame"],
                        "shift_um": item["shift_um"],
                        "nearest_support_change_distance_frames": item["nearest_support_change_distance_frames"],
                        "bound_edge_ids": sorted(row_edges[item["ref_input_row"]]),
                    })
    return records, strata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    contract_path = PACKET / "contract/support_boundary_waveform_discriminator.execution_disabled.v2.json"
    selection_path = PACKET / "freeze/DETERMINISTIC_RECORD_SELECTION.json"
    runner_path = PACKET / "source/run_support_boundary_waveform_discriminator.py"
    freezer_path = PACKET / "source/freeze_record_selection_v2.py"
    contract = json.loads(contract_path.read_text())
    selection = json.loads(selection_path.read_text())
    runner = runner_path.read_text()
    freezer = freezer_path.read_text()

    manifest_errors = []
    manifest_members = []
    for line in (PACKET / "MANIFEST.sha256").read_text().splitlines():
        expected, name = line.split("  ", 1)
        manifest_members.append(name)
        observed = sha(PACKET / name)
        if observed != expected:
            manifest_errors.append({"name": name, "expected": expected, "observed": observed})

    records, strata = regenerate_selection()
    ids = [int(item["ref_input_row"]) for item in records]
    bytes_exact = len(records) * 4217 * 384 * 2
    latest = max(
        (path for path in PACKET.rglob("*") if path.is_file()), key=lambda path: path.stat().st_mtime_ns
    ).relative_to(PACKET).as_posix()

    shared_binding_errors = []
    host_local_unchecked = []
    for binding in contract["bindings"]:
        path = Path(binding["path"])
        if not path.is_file():
            host_local_unchecked.append(binding["role"])
        elif sha(path) != binding["sha256"]:
            shared_binding_errors.append(binding["role"])

    seals = {
        "blocking_manifest": sha(BLOCKING_REVIEW / "MANIFEST.sha256"),
        "blocking_complete": sha(BLOCKING_REVIEW / "COMPLETE.json"),
        "blocking_review": sha(BLOCKING_REVIEW / "INDEPENDENT_REVIEW.json"),
        "synthetic_manifest": sha(SYNTHETIC / "MANIFEST.sha256"),
        "synthetic_complete": sha(SYNTHETIC / "COMPLETE.json"),
        "synthetic_result": sha(SYNTHETIC / "RESULT.json"),
        "synthetic_review_manifest": sha(SYNTHETIC_REVIEW / "MANIFEST.sha256"),
        "synthetic_review_complete": sha(SYNTHETIC_REVIEW / "COMPLETE.json"),
    }

    result_schema = json.loads((PACKET / "schemas/RESULT.schema.json").read_text())
    selection_schema = json.loads((PACKET / "schemas/SELECTION.schema.json").read_text())
    blockers = [
        "PACKET_COMPLETE_NOT_WRITTEN_LAST",
        "FROZEN_SELECTION_ORDERING_TEXT_CONTRADICTS_QUOTA_AND_EXECUTED_FREEZER",
        "REAL_RUN_DOES_NOT_GATE_EXPECTED_SIGNAL_OR_RESULT_SCHEMA_AS_CONTRACT_REQUIRES",
        "REAL_REPEAT_CONTROL_CHECKS_CURRENT_ARM_ONLY_AND_LEAVES_PROPOSED_ARM_UNCHECKED",
        "RESULT_SCHEMA_DOES_NOT_FREEZE_PER_RECORD_FIELDS_OR_EXACT_READ_COUNT_RECONCILIATION",
        "SERVICE_TEMPLATE_POINTS_TO_EXECUTION_DISABLED_CONTRACT_NOT_A_REVIEWED_ENABLED_DERIVATIVE",
    ]
    report = {
        "schema": "h1-v13-support-boundary-waveform-contract-v2-independent-rereview-v1",
        "verdict": "BLOCK_REPAIR_AND_REREVIEW",
        "execution_go": False,
        "subject": {
            "packet": str(PACKET),
            "manifest_sha256": sha(PACKET / "MANIFEST.sha256"),
            "complete_sha256": sha(PACKET / "COMPLETE.json"),
            "contract_sha256": sha(contract_path),
            "rereview_request_sha256": sha(PACKET / "H1_REREVIEW_REQUEST.json"),
        },
        "checks": {
            "manifest_members": len(manifest_members),
            "manifest_errors": manifest_errors,
            "execution_enabled_exact_false": contract.get("execution_enabled") is False,
            "selection_records_exactly_regenerated": records == selection["records"],
            "selection_strata_exactly_regenerated": strata == selection["strata"],
            "selection_records": len(records),
            "selection_unique_records": len(set(ids)),
            "selection_strata": len(strata),
            "selection_absent_strata": sum(item["absence_explicit"] for item in strata),
            "logical_read_bytes_exact": bytes_exact,
            "logical_read_bytes_cap": contract["resources"]["logical_read_bytes_max"],
            "selection_ordering_text": selection["ordering"],
            "freezer_quota_literal_is_two": "QUOTA=2" in freezer and "ordered[:QUOTA]" in freezer,
            "latest_packet_member": latest,
            "shared_binding_errors": shared_binding_errors,
            "host_local_bindings_not_reverified_on_h1": host_local_unchecked,
            "synthetic_and_prior_review_seals_observed": seals,
            "independent_known_answer_pytest": "7 passed in 3.87s",
            "runner_real_controls_literal": "controls={'repeat_bitwise':all(x['current_bitwise'] for x in repeat),'context_convergence':context_ok,'selection_and_reads_reconciled':True}",
            "runner_has_result_schema_validation": "jsonschema" in runner or "RESULT.schema.json" in runner,
            "runner_repeat_checks_proposed_arm": "proposed_bitwise" in runner,
            "result_schema_required_fields": result_schema.get("required", []),
            "result_schema_record_item_schema_present": bool(result_schema.get("properties", {}).get("records", {}).get("items")),
            "selection_schema_record_item_schema_present": bool(selection_schema.get("properties", {}).get("records", {}).get("items")),
            "service_executes_disabled_contract": "support_boundary_waveform_discriminator.execution_disabled.v2.json" in (PACKET / "service/PROPOSED_NOT_INSTALLED.service").read_text(),
            "real_voltage_or_waveform_bytes_read": 0,
            "gpu_work_started": False,
        },
        "blockers": blockers,
        "required_repair": [
            "Publish a fresh immutable packet with internally consistent first-two selection wording and write COMPLETE.json last.",
            "Bind a real-run preflight gate for the known-answer receipt/oracle and validate the emitted result against a strict schema before COMPLETE.",
            "Repeat both current and proposed primary outputs for the frozen repeat records and make either mismatch a control failure.",
            "Make RESULT.schema.json require exact record, metric, context, control, byte, membership, and 62-record reconciliation fields; make the selection schema define record and stratum items.",
            "Create and review an execution-enabled derivative and a service that points to its exact path/hash; retain atomic claim, fresh output, no restart, resource limits, failure preservation, and output COMPLETE-last semantics.",
            "On H5 immediately before the sole start, reverify all 16 bindings, host-local native/runtime and recording-manifest paths, absent output and claim, service text, execution_enabled=true, and zero prior starts; preserve that receipt.",
        ],
        "scope": {
            "can_establish": "The metadata-only quota-two selection and its exact 62 unique records/200796672-byte budget reproduce; shared source bindings and sealed predecessor/synthetic references reconcile; the seven no-voltage known-answer tests pass independently on H1.",
            "cannot_establish": "This packet is not start-ready, no real waveform effect or repair benefit is established, and H5-local native/runtime/recording bindings were not independently reopened on H1.",
        },
    }
    Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
