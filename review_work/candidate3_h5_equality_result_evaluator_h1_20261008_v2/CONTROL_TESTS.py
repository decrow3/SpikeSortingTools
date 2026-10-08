import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "candidate3_h5_equality_result_evaluator",
    ROOT / "testing/candidate3_h5_equality_result_evaluator_v2.py",
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
evaluate = MODULE.evaluate
BASE = ROOT / "review_work/candidate3_h5_real_wrapper_equality_contract_h1_20261008_v3/self_test_success"
CONTRACT = json.loads(
    (ROOT / "testing/contracts/candidate3_h5_equality_result_evaluator_20261008_v2.json").read_text()
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def reseal(packet: Path, status: str) -> None:
    (packet / "MANIFEST.json").unlink(missing_ok=True)
    (packet / "COMPLETE.json").unlink(missing_ok=True)
    members = {
        path.name: {"size": path.stat().st_size, "sha256": digest(path)}
        for path in sorted(packet.iterdir()) if path.is_file()
    }
    write_json(packet / "MANIFEST.json", {
        "schema": "immutable-evidence-manifest-v1", "status": status, "members": members,
    })
    write_json(packet / "COMPLETE.json", {
        "schema": "immutable-evidence-complete-v1", "status": status,
        "manifest_sha256": digest(packet / "MANIFEST.json"),
    })


def fixture(tmp_path: Path, mutation: str) -> Path:
    packet = tmp_path / mutation
    shutil.copytree(BASE, packet)
    if mutation == "success":
        return packet
    if mutation == "tampered_complete":
        complete = json.loads((packet / "COMPLETE.json").read_text())
        complete["manifest_sha256"] = "0" * 64
        write_json(packet / "COMPLETE.json", complete)
        return packet
    if mutation == "handled_failure":
        (packet / "RESULT.json").unlink()
        write_json(packet / "FAILURE.json", {
            "status": "FAILED_PRESERVED_NO_RETRY", "type": "RuntimeError",
            "message": "known fixture failure", "wall_seconds": 1.0,
        })
        reseal(packet, "FAILED_PRESERVED_NO_RETRY")
        return packet
    if mutation == "inconsistent_failure":
        (packet / "RESULT.json").unlink()
        write_json(packet / "FAILURE.json", {
            "status": "NOT_FROZEN", "type": "RuntimeError",
            "message": "inconsistent fixture", "wall_seconds": 1.0,
        })
        reseal(packet, "PASS_REAL_EQUALITY")
        return packet
    result_path = packet / "RESULT.json"
    result = json.loads(result_path.read_text())
    if mutation == "primary_mismatch":
        result["requests"][0]["mismatch_count"] = 1
        result["requests"][0]["max_abs_error"] = 1.0
        result["requests"][0]["first_mismatch_index"] = [0, 0]
        result["requests"][0]["lazy_sha256"] = "1" * 64
    elif mutation == "positive_control_zero":
        result["positive_controls"][0]["mismatch_count"] = 0
        result["positive_controls"][0]["max_abs_error"] = 0.0
    elif mutation == "resource_excess":
        result["wall_seconds"] = 1800.0001
    elif mutation == "missing_hashes":
        for row in result["requests"]:
            row.pop("lazy_sha256")
            row.pop("eager_sha256")
    elif mutation == "weak_real_relabel":
        result["verdict"] = "PASS_REAL_EQUALITY"
        result["binary_read_bytes"] = result["cached_input_bytes"]
        provenance_path = packet / "PROVENANCE.json"
        provenance = json.loads(provenance_path.read_text())
        provenance["mode"] = "real"
        provenance["argv"] = [
            "candidate3_h5_real_wrapper_equality_v3.py", "--contract", "contract.json", "--execute-real"
        ]
        provenance["release"] = {
            "review_complete": "/tmp/r", "review_complete_sha256": "x",
            "dispatch_receipt": "/tmp/d", "dispatch_receipt_sha256": "y",
            "resource_receipt": "/tmp/s", "resource_receipt_sha256": "z",
        }
        write_json(provenance_path, provenance)
    elif mutation == "valid_synthetic_real_packet":
        execution = json.loads((packet / "CONTRACT_COPY.json").read_text())
        source = execution["source"]
        release_contract = execution["release"]
        resources = execution["resources"]
        receipts = packet / "fixture_receipts"
        receipts.mkdir()
        review_path = (receipts / "REVIEW_COMPLETE.json").resolve()
        resource_path = (receipts / "RESOURCE.json").resolve()
        dispatch_path = (receipts / "DISPATCH.json").resolve()
        write_json(review_path, {
            "schema": release_contract["review_schema"],
            "status": release_contract["review_status"],
            "manifest_sha256": "a" * 64,
            "reviewed_contract_sha256": CONTRACT["accepted_execution_contract_sha256"],
            "reviewed_runner_sha256": CONTRACT["accepted_runner_sha256"],
        })
        resource = {
            "schema": release_contract["resource_receipt_schema"],
            "status": release_contract["resource_receipt_status"],
            "contract_sha256": CONTRACT["accepted_execution_contract_sha256"],
            "runner_sha256": CONTRACT["accepted_runner_sha256"],
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
            "manager_job_id": "synthetic-fixture-only",
        }
        write_json(resource_path, resource)
        review_hash, resource_hash = digest(review_path), digest(resource_path)
        dispatch = {
            "task": release_contract["dispatch_task"],
            "status": release_contract["dispatch_status"],
            "contract_sha256": CONTRACT["accepted_execution_contract_sha256"],
            "runner_sha256": CONTRACT["accepted_runner_sha256"],
            "review_complete_sha256": review_hash,
            "resource_receipt_sha256": resource_hash,
            "interpreter_path": source["expected_h5_interpreter"],
            "runner_path": source["expected_h5_runner_path"],
            "contract_path": source["expected_h5_contract_path"],
            "output_path": source["expected_h5_output_path"],
            "extra_kilosort_site": source["expected_h5_kilosort_site"],
            "review_complete_path": str(review_path),
            "resource_receipt_path": str(resource_path),
            "execution_mode": "execute-real",
        }
        write_json(dispatch_path, dispatch)
        dispatch_hash = digest(dispatch_path)
        result["verdict"] = "PASS_REAL_EQUALITY"
        result["binary_read_bytes"] = result["cached_input_bytes"]
        provenance_path = packet / "PROVENANCE.json"
        provenance = json.loads(provenance_path.read_text())
        provenance["mode"] = "real"
        provenance["python"] = source["expected_h5_interpreter"]
        provenance["release"] = {
            "review_complete": str(review_path), "review_complete_sha256": review_hash,
            "dispatch_receipt": str(dispatch_path), "dispatch_receipt_sha256": dispatch_hash,
            "resource_receipt": str(resource_path), "resource_receipt_sha256": resource_hash,
        }
        provenance["argv"] = [
            source["expected_h5_runner_path"],
            "--contract", source["expected_h5_contract_path"],
            "--expected-contract-sha256", CONTRACT["accepted_execution_contract_sha256"],
            "--output", source["expected_h5_output_path"],
            "--extra-kilosort-site", source["expected_h5_kilosort_site"],
            "--review-complete", str(review_path),
            "--expected-review-complete-sha256", review_hash,
            "--dispatch-receipt", str(dispatch_path),
            "--expected-dispatch-receipt-sha256", dispatch_hash,
            "--resource-receipt", str(resource_path),
            "--expected-resource-receipt-sha256", resource_hash,
            "--execute-real",
        ]
        write_json(provenance_path, provenance)
    else:
        raise AssertionError(mutation)
    write_json(result_path, result)
    reseal(packet, result["verdict"])
    return packet


@pytest.mark.parametrize(
    ("mutation", "category"),
    [
        ("success", "PASS_SYNTHETIC_FIXTURE_ONLY_NO_H5_DECISION"),
        ("tampered_complete", "REJECT_SEAL_OR_SCHEMA_STOP"),
        ("handled_failure", "UPSTREAM_HANDLED_FAILURE_STOP"),
        ("primary_mismatch", "REJECT_PRIMARY_MISMATCH_STOP"),
        ("positive_control_zero", "REJECT_POSITIVE_CONTROL_NONINFORMATIVE_STOP"),
        ("resource_excess", "REJECT_RESOURCE_OR_BOUNDS_STOP"),
        ("missing_hashes", "REJECT_PRIMARY_MISMATCH_STOP"),
        ("weak_real_relabel", "REJECT_PROVENANCE_STOP"),
        ("inconsistent_failure", "REJECT_SEAL_OR_SCHEMA_STOP"),
        ("valid_synthetic_real_packet", "PASS_REAL_EQUALITY_ADVANCE_TO_FROZEN_NEXT_GATE"),
    ],
)
def test_frozen_verdict_controls(tmp_path, mutation, category):
    assert evaluate(fixture(tmp_path, mutation), CONTRACT)["category"] == category
