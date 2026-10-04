#!/usr/bin/env python3
"""Independent metadata-only review of the masked-zero real-snippet method."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import jsonschema
import numpy as np


PACKET = Path(
    "/mnt/NPX/Luke/DARTsort_motion_experiments/"
    "en_first_medium_v13_masked_zero_real_snippet_contract_20261003_v1_h5"
)
COORDINATION = Path(
    "/mnt/NPX/Luke/DARTsort_motion_experiments/current_coordination_20261002_v1/"
    "H5_STATUS_V13_MASKED_ZERO_REAL_SNIPPET_CONTRACT_V1.json"
)
EXPECTED_QUESTION = (
    "On the unchanged frozen 62-record population, at the pre-whitening 121-sample core and against the exact current production operator, "
    "does masked-zero-domain-extension yield median supported-delta RMS >=0.5 ADC counts in at least one near-boundary stratum for at least two "
    "of REF176/REF192/REF220, while every REF208/REF190 stratum remains <0.5 counts and repeatability plus p2048/p1024 context gates pass?"
)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def membership_hash(values) -> str:
    return hashlib.sha256("".join(f"{int(value)}\n" for value in sorted(values)).encode()).hexdigest()


def support_mask(geometry, centers, shifts, frames, fs):
    dt = float(np.median(np.diff(centers)))
    positions = np.searchsorted(centers + dt / 2, frames.astype(np.float64) / fs, side="right")
    states = shifts[np.clip(positions, 0, centers.size - 1)]
    lower, upper = geometry.min(0), geometry.max(0)
    result = np.empty((384, len(frames)), dtype=bool)
    for state in np.unique(states):
        desired = geometry.copy()
        desired[:, 1] += float(state)
        result[:, states == state] = np.all((desired >= lower) & (desired <= upper), axis=1)[:, None]
    return result


def decision(gates: bool, control_values, nominated_units):
    if not gates:
        return "INCONCLUSIVE_CONTROL_OR_PROVENANCE_FAILURE_STOP"
    if any(value >= 0.5 for value in control_values):
        return "CONTROL_EFFECT_STOP"
    if len(set(nominated_units)) >= 2:
        return "POSITIVE_SELECTIVE_EFFECT_REVIEW_ONLY"
    return "NEGATIVE_INSUFFICIENT_SELECTIVITY_STOP"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    contract_path = PACKET / "contract/MASKED_ZERO_REAL_SNIPPET.execution_disabled.v1.json"
    contract = json.loads(contract_path.read_text())
    rule_path = PACKET / "DECISION_RULE.json"
    rule = json.loads(rule_path.read_text())

    manifest_errors = []
    members = []
    for line in (PACKET / "MANIFEST.sha256").read_text().splitlines():
        expected, name = line.split("  ", 1)
        members.append(name)
        observed = sha(PACKET / name)
        if observed != expected:
            manifest_errors.append({"name": name, "expected": expected, "observed": observed})

    selection_path = Path(contract["population"]["selection_path"])
    selection = json.loads(selection_path.read_text())
    records = selection["records"]
    ref_rows = [record["ref_input_row"] for record in records]
    counts = {}
    for record in records:
        key = str(record["reference_unit"])
        counts[key] = counts.get(key, 0) + 1

    shared_binding_errors = []
    host_local_unchecked = []
    for binding in contract["source_bindings"]:
        path = Path(binding["path"])
        if not path.is_file():
            host_local_unchecked.append(binding["role"])
        elif sha(path) != binding["sha256"]:
            shared_binding_errors.append(binding["role"])

    geometry_path = Path(contract["support_and_metrics"]["geometry_receipt_path"])
    field_path = Path(contract["support_and_metrics"]["field_path"])
    geometry = np.asarray(json.loads(geometry_path.read_text())["recording"]["channel_locations_um"], dtype=np.float64)
    with np.load(field_path, allow_pickle=False) as saved:
        centers = np.asarray(saved[contract["support_and_metrics"]["field_time_key"]], dtype=np.float64)
        displacement = np.asarray(saved[contract["support_and_metrics"]["field_displacement_key"]], dtype=np.float64)
    rounding = contract["support_and_metrics"]["field_rounding_um"]
    scaled = (displacement - np.median(displacement)) / rounding
    shifts = rounding * np.copysign(np.floor(np.abs(scaled) + 0.5), scaled)
    core_support_mismatches = []
    core_global_frame_errors = []
    for record in records:
        frame = int(record["global_frame"])
        primary_start = frame - contract["reader"]["primary_radius_samples"]
        nested_start = frame - contract["reader"]["nested_radius_samples"]
        primary = support_mask(
            geometry, centers, shifts,
            np.arange(primary_start, primary_start + contract["reader"]["samples_per_unique_read"], dtype=np.int64),
            contract["reader"]["sampling_frequency_hz"],
        )
        nested = support_mask(
            geometry, centers, shifts,
            np.arange(nested_start, nested_start + 2169, dtype=np.int64),
            contract["reader"]["sampling_frequency_hz"],
        )
        primary_core = primary[:, 2048:2169]
        nested_core = nested[:, 1024:1145]
        if not np.array_equal(primary_core, nested_core):
            core_support_mismatches.append(record["selection_ordinal"])
        if primary_start + 2048 != frame - 60 or nested_start + 1024 != frame - 60:
            core_global_frame_errors.append(record["selection_ordinal"])

    known_answer_errors = []
    for case in rule["known_answer_cases"]:
        controls = [case["control_max"]]
        observed = decision(case["gates"], controls, case["nominated_units_at_or_above"])
        if observed != case["expected"]:
            known_answer_errors.append({"case": case, "observed": observed})

    schema_errors = []
    schemas = {}
    for name in ("RESULT", "RESOURCE_RECEIPT", "FAILURE"):
        path = PACKET / f"schemas/{name}.schema.json"
        document = json.loads(path.read_text())
        schemas[name] = document
        try:
            jsonschema.Draft202012Validator.check_schema(document)
        except Exception as error:
            schema_errors.append({"schema": name, "error": str(error)})

    result_schema = schemas["RESULT"]
    result_records = result_schema["properties"]["records"]
    result_strata = result_schema["properties"]["strata"]
    latest = max(
        (path for path in PACKET.rglob("*") if path.is_file()), key=lambda path: path.stat().st_mtime_ns
    ).relative_to(PACKET).as_posix()
    bytes_expected = 62 * 4217 * 384 * 2
    blockers = []
    if manifest_errors or shared_binding_errors or schema_errors:
        blockers.append("PACKET_BINDING_OR_SCHEMA_INVALID")
    if contract["scientific_question"] != EXPECTED_QUESTION:
        blockers.append("DECISION_VALUE_DRIFT")
    if len(records) != 62 or len(set(ref_rows)) != 62 or membership_hash(ref_rows) != contract["population"]["unique_ref_rows_sha256"]:
        blockers.append("POPULATION_MISMATCH")
    if [record["selection_ordinal"] for record in records] != list(range(62)) or counts != contract["population"]["reference_unit_counts"]:
        blockers.append("ORDER_OR_UNIT_COUNT_MISMATCH")
    if len(selection["strata"]) != 40 or sum(item["selected"] for item in selection["strata"]) != 62:
        blockers.append("STRATUM_RECONCILIATION_MISMATCH")
    if core_support_mismatches or core_global_frame_errors:
        blockers.append("CONTEXT_CORE_OR_CLOCK_MISMATCH")
    if known_answer_errors:
        blockers.append("DECISION_PRECEDENCE_FAILURE")
    if bytes_expected != contract["resources"]["physical_api_requested_bytes_exact"]:
        blockers.append("READ_ACCOUNTING_MISMATCH")
    if latest != "COMPLETE.json":
        blockers.append("COMPLETE_NOT_LAST")

    report = {
        "schema": "h1-v13-masked-zero-real-snippet-method-independent-review-v1",
        "verdict": "GO_METHOD_ONLY_IMPLEMENTATION_CLOSURE_REQUIRED" if not blockers else "BLOCK_REPAIR_AND_REREVIEW",
        "method_go": not blockers,
        "implementation_go": False,
        "execution_go": False,
        "data_access_authorized_by_review": False,
        "subject": {
            "packet": str(PACKET),
            "manifest_sha256": sha(PACKET / "MANIFEST.sha256"),
            "complete_sha256": sha(PACKET / "COMPLETE.json"),
            "contract_sha256": sha(contract_path),
            "method_sha256": sha(PACKET / "METHOD.md"),
            "decision_rule_sha256": sha(rule_path),
            "preflight_sha256": sha(PACKET / "PREFLIGHT_RECEIPT.json"),
            "request_sha256": sha(PACKET / "H1_METHOD_REVIEW_REQUEST.json"),
            "coordination_sha256": sha(COORDINATION),
        },
        "checks": {
            "manifest_members": len(members),
            "manifest_errors": manifest_errors,
            "complete_last": latest == "COMPLETE.json",
            "execution_enabled_exact_false": contract["execution_enabled"] is False,
            "runner_or_service_present": contract["lifecycle"]["runner_or_service_present"],
            "scientific_question_exact": contract["scientific_question"] == EXPECTED_QUESTION,
            "repair_relevance": "selective nominated near-boundary effect plus strict unchanged-control guardrail; mechanism screen only",
            "selection_sha256": sha(selection_path),
            "selection_records": len(records),
            "selection_unique_records": len(set(ref_rows)),
            "selection_order_exact": [record["selection_ordinal"] for record in records] == list(range(62)),
            "selection_membership_hash": membership_hash(ref_rows),
            "reference_unit_counts": counts,
            "strata": len(selection["strata"]),
            "selected_count_reconciled": sum(item["selected"] for item in selection["strata"]),
            "shared_binding_errors": shared_binding_errors,
            "host_local_bindings_not_reverified_on_h1": host_local_unchecked,
            "core_support_mismatches": core_support_mismatches,
            "core_global_frame_errors": core_global_frame_errors,
            "one_read_bytes_exact": bytes_expected,
            "physical_api_calls_exact": contract["resources"]["physical_api_calls_exact"],
            "repeat_extra_read_bytes": contract["resources"]["repeat_extra_read_bytes"],
            "decision_known_answer_errors": known_answer_errors,
            "strict_control_guardrail": rule["threshold_counts"] == 0.5 and rule["precedence"][1]["else_if"] == "any control_stratum_median >= 0.5",
            "schema_errors": schema_errors,
            "result_schema_records_exact_count": result_records.get("minItems") == result_records.get("maxItems") == 62,
            "result_schema_encodes_exact_record_order": False,
            "result_schema_strata_item_schema_present": "items" in result_strata,
            "result_schema_encodes_exact_40_strata": result_strata.get("minItems") == result_strata.get("maxItems") == 40,
            "resource_schema_strict": schemas["RESOURCE_RECEIPT"].get("additionalProperties") is False,
            "real_voltage_or_waveform_bytes_read": 0,
        },
        "blockers": blockers,
        "single_required_implementation_closure": [
            "Publish one fresh execution-disabled integrated packet binding the exact runner, managed-service unit, enabled-derivative template, this method/decision/schemas, operator/helper/native/runtime bytes, and all external inputs by hash.",
            "The runner must enforce—not merely serialize booleans for—exact 62-record order/membership, all 40 strata including absences, physical/global clocks, one-read paired clones, identical core support, both-arm repeats, all-record both-arm context limits, strict decision precedence, exact 62-call/200796672-byte reconciliation, and strict result/resource/failure schema validation. Because the current result schema does not encode exact record order or stratum item construction, add explicit runner known-answer/adversarial reconciliation tests or strengthen the schema in that packet.",
            "Bind a disconnect-surviving one-shot managed service with Restart=no, 4-CPU/16-GiB/3600-s/zero-GPU limits, fresh output and atomic one-start claim, known-answer before binary open, preserved staged FAILURE, no retry/checkpoint claim, compact output, OUTPUT_MANIFEST then COMPLETE last.",
            "Before any execution, obtain one independent no-voltage review of the exact integrated runner/service packet and preserve an H5 prestart receipt verifying every binding, runtime and recording identity metadata, absent output/claim, zero prior starts, installed service bytes, execution_enabled=true only in the reviewed derivative, and successful validate-only/synthetic/decision/schema/failure rehearsals."
        ],
        "scope": {
            "can_establish": "The method is a bounded repair-relevant selective-effect screen with frozen population, clocks, preprocessing, metrics, aggregation, controls, decision mapping, resources and lifecycle requirements.",
            "cannot_establish": "There is no implementation or execution GO, no data-access permission from this review, and no evidence yet of real effect, repair benefit, improvement, production safety, integration readiness, biological identity or purity.",
        },
    }
    Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
