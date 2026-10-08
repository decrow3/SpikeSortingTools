#!/usr/bin/env python3
import hashlib
import importlib.util
import json
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "candidate3_h5_equality_result_evaluator",
    ROOT / "testing/candidate3_h5_equality_result_evaluator.py",
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
CONTRACT = json.loads(
    (ROOT / "testing/contracts/candidate3_h5_equality_result_evaluator_20261008_v1.json").read_text()
)
BASE = Path(
    "/mnt/NPX/Luke/DARTsort_motion_experiments/"
    "candidate3_h5_real_wrapper_equality_contract_h1_20261008_v3/self_test_success"
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


with tempfile.TemporaryDirectory(prefix="candidate3_evaluator_adversarial_") as td:
    root = Path(td)
    missing_digests = root / "missing_digests"
    shutil.copytree(BASE, missing_digests)
    result = json.loads((missing_digests / "RESULT.json").read_text())
    for row in result["requests"]:
        row.pop("lazy_sha256")
        row.pop("eager_sha256")
    write_json(missing_digests / "RESULT.json", result)
    reseal(missing_digests, result["verdict"])

    incomplete_real = root / "incomplete_real_release"
    shutil.copytree(BASE, incomplete_real)
    result = json.loads((incomplete_real / "RESULT.json").read_text())
    result["verdict"] = "PASS_REAL_EQUALITY"
    result["binary_read_bytes"] = CONTRACT["statistics"]["resources"].split("real binary_read_bytes=")[1].split(",")[0]
    result["binary_read_bytes"] = int(result["binary_read_bytes"])
    write_json(incomplete_real / "RESULT.json", result)
    provenance = json.loads((incomplete_real / "PROVENANCE.json").read_text())
    execution = json.loads((incomplete_real / "CONTRACT_COPY.json").read_text())
    source = execution["source"]
    provenance["mode"] = "real"
    provenance["python"] = source["expected_h5_interpreter"]
    provenance["argv"] = [
        source["expected_h5_contract_path"], source["expected_h5_output_path"],
        source["expected_h5_kilosort_site"], "--execute-real",
    ]
    provenance["release"] = {
        "review_complete_sha256": "not-a-sha256",
        "dispatch_receipt_sha256": "also-not-a-sha256",
        "resource_receipt_sha256": "still-not-a-sha256",
    }
    write_json(incomplete_real / "PROVENANCE.json", provenance)
    reseal(incomplete_real, result["verdict"])

    inconsistent_failure = root / "inconsistent_failure_status"
    shutil.copytree(BASE, inconsistent_failure)
    (inconsistent_failure / "RESULT.json").unlink()
    write_json(inconsistent_failure / "FAILURE.json", {
        "status": "NOT_THE_FROZEN_FAILURE_STATUS", "type": "RuntimeError", "message": "fixture",
    })
    reseal(inconsistent_failure, "PASS_REAL_EQUALITY")

    observed = {
        "missing_both_output_digests": MODULE.evaluate(missing_digests, CONTRACT)["category"],
        "incomplete_real_release_provenance": MODULE.evaluate(incomplete_real, CONTRACT)["category"],
        "inconsistent_failure_status": MODULE.evaluate(inconsistent_failure, CONTRACT)["category"],
    }
    print(json.dumps(observed, indent=2, sort_keys=True))
