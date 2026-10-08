import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "candidate3_h5_equality_result_evaluator",
    ROOT / "testing/candidate3_h5_equality_result_evaluator.py",
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
evaluate = MODULE.evaluate
BASE = ROOT / "review_work/candidate3_h5_real_wrapper_equality_contract_h1_20261008_v3/self_test_success"
CONTRACT = json.loads(
    (ROOT / "testing/contracts/candidate3_h5_equality_result_evaluator_20261008_v1.json").read_text()
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
    ],
)
def test_frozen_verdict_controls(tmp_path, mutation, category):
    assert evaluate(fixture(tmp_path, mutation), CONTRACT)["category"] == category
