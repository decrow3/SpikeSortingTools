#!/usr/bin/env python3
import hashlib
import importlib.util
import json
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "candidate3_h5_equality_result_evaluator_v2",
    ROOT / "testing/candidate3_h5_equality_result_evaluator_v2.py",
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
CONTRACT = json.loads(
    (ROOT / "testing/contracts/candidate3_h5_equality_result_evaluator_20261008_v2.json").read_text()
)
BASE = ROOT / "review_work/candidate3_h5_real_wrapper_equality_contract_h1_20261008_v3/self_test_success"


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


def observe(packet: Path) -> str:
    try:
        return MODULE.evaluate(packet, CONTRACT)["category"]
    except Exception as exc:
        return f"UNHANDLED_{type(exc).__name__}: {exc}"


with tempfile.TemporaryDirectory(prefix="candidate3_evaluator_v2_malformed_") as td:
    root = Path(td)
    malformed_failure = root / "malformed_failure"
    shutil.copytree(BASE, malformed_failure)
    (malformed_failure / "RESULT.json").unlink()
    (malformed_failure / "FAILURE.json").write_text("{not valid json\n")
    reseal(malformed_failure, "FAILED_PRESERVED_NO_RETRY")

    invalid_receipt_path_type = root / "invalid_receipt_path_type"
    shutil.copytree(BASE, invalid_receipt_path_type)
    result_path = invalid_receipt_path_type / "RESULT.json"
    result = json.loads(result_path.read_text())
    result["verdict"] = "PASS_REAL_EQUALITY"
    result["binary_read_bytes"] = result["cached_input_bytes"]
    write_json(result_path, result)
    provenance_path = invalid_receipt_path_type / "PROVENANCE.json"
    provenance = json.loads(provenance_path.read_text())
    execution = json.loads((invalid_receipt_path_type / "CONTRACT_COPY.json").read_text())
    provenance["mode"] = "real"
    provenance["python"] = execution["source"]["expected_h5_interpreter"]
    provenance["release"] = {
        "review_complete": 123,
        "review_complete_sha256": "a" * 64,
        "dispatch_receipt": "/tmp/absent-dispatch.json",
        "dispatch_receipt_sha256": "b" * 64,
        "resource_receipt": "/tmp/absent-resource.json",
        "resource_receipt_sha256": "c" * 64,
    }
    write_json(provenance_path, provenance)
    reseal(invalid_receipt_path_type, result["verdict"])

    print(json.dumps({
        "malformed_failure_json": observe(malformed_failure),
        "invalid_receipt_path_type": observe(invalid_receipt_path_type),
    }, indent=2, sort_keys=True))
