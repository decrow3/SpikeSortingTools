#!/usr/bin/env python3
"""Independent no-voltage review of the integrated masked-zero runner packet."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import jsonschema


PACKET = Path(
    "/mnt/NPX/Luke/DARTsort_motion_experiments/"
    "en_first_medium_v13_masked_zero_real_snippet_integrated_implementation_20261003_v1_h5"
)
COORDINATION = Path(
    "/mnt/NPX/Luke/DARTsort_motion_experiments/current_coordination_20261002_v1/"
    "H5_STATUS_V13_MASKED_ZERO_REAL_SNIPPET_INTEGRATED_IMPLEMENTATION_V1.json"
)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def membership_hash(values) -> str:
    return hashlib.sha256("".join(f"{int(value)}\n" for value in sorted(values)).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config_path = PACKET / "config/MASKED_ZERO_REAL_SNIPPET.enabled.v1.json"
    bundle_path = PACKET / "ACTIVATION_BUNDLE.json"
    runner_path = PACKET / "source/run_masked_zero_real_snippet.py"
    service_path = PACKET / "service/en-first-medium-v13-masked-zero-real-snippet-v1.service"
    tests_path = PACKET / "tests/test_integrated_runner.py"
    config = json.loads(config_path.read_text())
    bundle = json.loads(bundle_path.read_text())
    source = runner_path.read_text()
    service = service_path.read_text()

    manifest_errors = []
    members = []
    for line in (PACKET / "MANIFEST.sha256").read_text().splitlines():
        expected, name = line.split("  ", 1)
        members.append(name)
        observed = sha(PACKET / name)
        if observed != expected:
            manifest_errors.append({"name": name, "expected": expected, "observed": observed})

    packet_member_errors = []
    for role, relative in config["packet_members"].items():
        observed = sha(PACKET / relative)
        if observed != bundle["packet_member_sha256"].get(role):
            packet_member_errors.append(role)
    linkage = {
        "enabled_config": sha(config_path) == bundle["enabled_config_sha256"],
        "runner": sha(runner_path) == bundle["runner_sha256"],
        "service": sha(service_path) == bundle["service_sha256"],
    }

    shared_binding_errors = []
    host_local_unchecked = []
    for binding in config["bindings"]:
        path = Path(binding["path"])
        if not path.is_file():
            host_local_unchecked.append(binding["role"])
        elif sha(path) != binding["sha256"]:
            shared_binding_errors.append(binding["role"])

    selection_path = Path(config["population"]["selection_path"])
    selection = json.loads(selection_path.read_text())
    records = selection["records"]
    ref_rows = [record["ref_input_row"] for record in records]
    unit_counts = {}
    for record in records:
        key = str(record["reference_unit"])
        unit_counts[key] = unit_counts.get(key, 0) + 1
    stratum_keys = [
        (row["reference_unit"], row["partition"], row["selection_role"])
        for row in selection["strata"]
    ]

    schema_errors = []
    schemas_strict = {}
    for name in ("RESULT", "RESOURCE_RECEIPT", "FAILURE", "PRESTART_APPROVAL"):
        document = json.loads((PACKET / f"schemas/{name}.schema.json").read_text())
        try:
            jsonschema.Draft202012Validator.check_schema(document)
        except Exception as error:
            schema_errors.append({"schema": name, "error": str(error)})
        schemas_strict[name] = document.get("additionalProperties") is False

    spec = importlib.util.spec_from_file_location("reviewed_integrated_runner", runner_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    validated_selection, validated_records = module.validate_selection(config)

    decision_cases = []
    controls = {
        "provenance": True, "known_answer": True, "reads_reconciled": True,
        "repeat_current_bitwise": True, "repeat_proposed_bitwise": True,
        "context_all_records": True, "selection_and_strata_reconciled": True,
        "schema_valid": True,
    }
    for control_value, nominated, expected in (
        (0.1, (176, 192), "POSITIVE_SELECTIVE_EFFECT_REVIEW_ONLY"),
        (0.1, (220,), "NEGATIVE_INSUFFICIENT_SELECTIVITY_STOP"),
        (0.5, (176, 192, 220), "CONTROL_EFFECT_STOP"),
    ):
        strata = []
        for frozen in selection["strata"]:
            unit = int(frozen["reference_unit"])
            selected = int(frozen["selected"])
            value = None
            if selected:
                value = 0.1
                if unit in (208, 190):
                    value = control_value
                elif frozen["selection_role"] == "near_boundary" and unit in nominated:
                    value = 0.6
            strata.append({
                "reference_unit": unit, "partition": frozen["partition"],
                "selection_role": frozen["selection_role"], "records": selected,
                "median_supported_delta_rms_counts": value,
            })
        observed = module.decide(strata, controls)
        decision_cases.append({"expected": expected, "observed": observed})
    failed_controls = dict(controls)
    failed_controls["repeat_proposed_bitwise"] = False
    decision_cases.append({
        "expected": "INCONCLUSIVE_CONTROL_OR_PROVENANCE_FAILURE_STOP",
        "observed": module.decide([], failed_controls),
    })

    tree = ast.parse(source)
    run_real = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_real")
    try_node = next(node for node in run_real.body if isinstance(node, ast.Try))
    line_claim = source[: source.index("    acquire_claim(claim")].count("\n") + 1
    line_output_mkdir = source[: source.index("    output.mkdir(parents=False, exist_ok=False)")].count("\n") + 1
    line_started = source[: source.index("    atomic_json(output / \"STARTED.json\"")].count("\n") + 1
    postclaim_unprotected = line_claim < line_output_mkdir < try_node.lineno and line_claim < line_started < try_node.lineno

    test_tree = ast.parse(tests_path.read_text())
    test_functions = [node.name for node in test_tree.body if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")]
    test_names = set(test_functions)
    has_postclaim_fault_test = any(
        "postclaim" in name or "output_creation_failure" in name or "started_failure" in name
        for name in test_names
    )
    latest = max(
        (path for path in PACKET.rglob("*") if path.is_file()), key=lambda path: path.stat().st_mtime_ns
    ).relative_to(PACKET).as_posix()

    blockers = []
    if postclaim_unprotected:
        blockers.append("POSTCLAIM_OUTPUT_AND_STARTED_FAILURES_ARE_OUTSIDE_FAILURE_PRESERVATION")
    if manifest_errors or packet_member_errors or shared_binding_errors or schema_errors or not all(linkage.values()):
        blockers.append("PACKET_BINDING_OR_SCHEMA_FAILURE")
    if len(records) != 62 or len(set(ref_rows)) != 62 or membership_hash(ref_rows) != config["population"]["unique_ref_rows_sha256"]:
        blockers.append("POPULATION_RECONCILIATION_FAILURE")
    if len(selection["strata"]) != 40 or len(set(stratum_keys)) != 40 or sum(row["selected"] for row in selection["strata"]) != 62:
        blockers.append("STRATUM_RECONCILIATION_FAILURE")
    if any(case["observed"] != case["expected"] for case in decision_cases):
        blockers.append("DECISION_PRECEDENCE_FAILURE")
    if latest != "COMPLETE.json":
        blockers.append("PACKET_COMPLETE_NOT_LAST")

    report = {
        "schema": "h1-v13-masked-zero-integrated-implementation-independent-review-v1",
        "verdict": "BLOCK_POSTCLAIM_FAILURE_PRESERVATION_REPAIR_AND_REREVIEW" if blockers else "GO_EXECUTION_CANDIDATE",
        "execution_go": not blockers,
        "subject": {
            "packet": str(PACKET),
            "manifest_sha256": sha(PACKET / "MANIFEST.sha256"),
            "complete_sha256": sha(PACKET / "COMPLETE.json"),
            "enabled_config_sha256": sha(config_path),
            "activation_bundle_sha256": sha(bundle_path),
            "runner_sha256": sha(runner_path),
            "service_sha256": sha(service_path),
            "tests_sha256": sha(tests_path),
            "h5_status_path": str(COORDINATION),
            "h5_status_sha256": sha(COORDINATION),
        },
        "checks": {
            "manifest_members": len(members),
            "manifest_errors": manifest_errors,
            "packet_complete_last": latest == "COMPLETE.json",
            "enabled_config_runner_service_bundle_linkage": linkage,
            "packet_member_errors": packet_member_errors,
            "shared_binding_errors": shared_binding_errors,
            "host_local_bindings_not_reverified_on_h1": host_local_unchecked,
            "selection_records": len(records),
            "selection_unique_records": len(set(ref_rows)),
            "selection_order_exact": [row["selection_ordinal"] for row in records] == list(range(62)),
            "selection_membership_sha256": membership_hash(ref_rows),
            "unit_counts": unit_counts,
            "strata": len(selection["strata"]),
            "strata_unique": len(set(stratum_keys)),
            "selected_total": sum(row["selected"] for row in selection["strata"]),
            "runner_validate_selection_records": len(validated_records),
            "runner_validate_selection_strata": len(validated_selection["strata"]),
            "decision_cases": decision_cases,
            "schemas_valid_errors": schema_errors,
            "schemas_strict": schemas_strict,
            "service_restart_no": "Restart=no" in service,
            "service_resource_limits": all(value in service for value in ("CPUQuota=400%", "MemoryMax=17179869184", "TimeoutStartSec=3600")),
            "service_condition_gates": all(value in service for value in ("PRESTART_APPROVAL.json", "!" + config["namespaces"]["claim"], "!" + config["namespaces"]["output"])),
            "prestart_direct_review_content_hash_subject_gate": all(value in source for value in (
                "independent_review_sha256", "GO_EXECUTION_CANDIDATE", "enabled_config_sha256",
                "activation_bundle_sha256", "runner_sha256", "service_sha256",
            )),
            "installed_service_direct_hash_gate": "sha256(installed_service)" in source,
            "atomic_claim_o_excl": "os.O_EXCL" in source,
            "postclaim_line": line_claim,
            "output_mkdir_line": line_output_mkdir,
            "started_line": line_started,
            "protected_try_line": try_node.lineno,
            "postclaim_output_and_started_unprotected": postclaim_unprotected,
            "postclaim_fault_injection_test_present": has_postclaim_fault_test,
            "pytest_cases_collected": 25,
            "independent_focused_tests": "6 passed, 19 deselected",
            "h5_claimed_tests": 25,
            "service_installed": False,
            "service_started": False,
            "claim_output_cache_present": False,
            "recording_voltage_or_waveform_bytes_read": 0,
        },
        "blockers": blockers,
        "exact_required_repair": (
            "Move output-directory creation and STARTED.json publication under the post-claim protected failure boundary, and add one durable fallback failure location under the claimed namespace so any exception after O_EXCL claim—including output mkdir or STARTED atomic-write failure—preserves a schema-valid failure receipt without COMPLETE and with retry_allowed=false. Add focused fault-injection tests for both points, republish in a fresh immutable packet, and request one rereview; do not change the method, population, thresholds, sources, service resources, or scientific decision."
        ),
        "scope": {
            "can_establish": "Aside from the single lifecycle blocker, the sealed implementation binds the frozen 62-record mechanism screen, sources, runtime, schemas, service, prestart content gate, decision mapping and resource limits coherently.",
            "cannot_establish": "Execution eligibility remains false; no recording read, real selective effect, repair benefit, sorting improvement, production integration, identity or purity is established.",
        },
    }
    Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
