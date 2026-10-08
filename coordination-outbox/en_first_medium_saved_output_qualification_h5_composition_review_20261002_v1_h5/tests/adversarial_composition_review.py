import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np


PACKET = Path("/mnt/NPX/Luke/DARTsort_motion_experiments/en_first_medium_saved_output_qualification_contract_20261002_v1_h1")
SOURCE = PACKET / "source"
CONTRACT = PACKET / "configs/en_first_medium_saved_output_qualification.execution_disabled.v1.json"
sys.path.insert(0, str(SOURCE))

fixture_spec = importlib.util.spec_from_file_location(
    "packet_fixture_helpers", PACKET / "tests/test_first_medium_saved_output_qualification.py"
)
fixture = importlib.util.module_from_spec(fixture_spec)
fixture_spec.loader.exec_module(fixture)

from testing.first_medium_saved_output_qualification import _cohort_digest, _verify_phase2_receipts


root = Path(tempfile.mkdtemp(prefix="en-h5-adversarial-"))
arm = fixture.write_ref_saved_format(root / "outside-packet-data")
cohort = np.array([1, 2], dtype=np.int64)
cohort_sha = _cohort_digest(cohort)
fixture_request = {
    "schema": "first-medium-saved-output-qualification-request-v1",
    "phase": "phase1_controls",
    "data_scope": "synthetic_saved_format_fixture",
    "fixture_token": "PACKAGED_SAVED_FORMAT_FIXTURE_ONLY_V1",
    "reference_cohort": cohort.tolist(),
    "reference_cohort_sha256": cohort_sha,
    "arms": [arm],
}
fixture_request_path = root / "fixture-request.json"
fixture_request_path.write_text(json.dumps(fixture_request))
fixture_output = root / "fixture-output"
fixture_result = fixture.run_cli(fixture_request_path, fixture_output)
if fixture_result.returncode:
    raise RuntimeError(fixture_result.stderr)

method_freeze = root / "method-freeze.json"
method_freeze.write_text("{}\n")
tokens = {
    "phase1_manifest_sha256": hashlib.sha256((fixture_output / "MANIFEST.sha256").read_bytes()).hexdigest(),
    "phase1_complete_sha256": hashlib.sha256((fixture_output / "COMPLETE.json").read_bytes()).hexdigest(),
    "fixed_ref_cohort_sha256": cohort_sha,
    "method_freeze_token": hashlib.sha256(method_freeze.read_bytes()).hexdigest(),
}
_verify_phase2_receipts(
    {"phase1_receipt_dir": str(fixture_output), "method_freeze_receipt": str(method_freeze)},
    tokens,
)

late_env = os.environ.copy()
late_env.update({
    "PYTHONPATH": str(SOURCE), "PYTHONDONTWRITEBYTECODE": "1",
    "NUMBA_CACHE_DIR": str(root / "numba-late"), "MPLCONFIGDIR": str(root / "mpl-late"),
    "OMP_NUM_THREADS": "5", "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1", "NUMBA_NUM_THREADS": "1",
})
late_output = root / "late-resource-output"
late_result = subprocess.run(
    [sys.executable, "-m", "testing.first_medium_saved_output_qualification",
     "--contract", str(CONTRACT), "--request", str(fixture_request_path),
     "--output", str(late_output)],
    cwd=root, env=late_env, capture_output=True, text=True,
)

enabled_contract = json.loads(CONTRACT.read_text())
enabled_contract["execution_enabled"] = True
enabled_contract["status"] = "ADVERSARIAL_SYNTHETIC_REPLAY_PROOF_ONLY"
phase1 = enabled_contract["phases"]["phase1_controls"]
phase1["real_execution_enabled"] = True
phase1["required_tokens"] = {
    "composition_review_token": "synthetic-review",
    "phase1_start_token": "same-replayed-start-token",
    "fixed_ref_cohort_sha256": cohort_sha,
}
enabled_contract_path = root / "enabled-contract.json"
enabled_contract_path.write_text(json.dumps(enabled_contract))
real_labeled_request = dict(fixture_request)
real_labeled_request.pop("fixture_token")
real_labeled_request["data_scope"] = "real_saved"
real_labeled_request["prerequisite_tokens"] = dict(phase1["required_tokens"])
real_request_path = root / "real-labeled-synthetic-request.json"
real_request_path.write_text(json.dumps(real_labeled_request))

replay_results = []
for index in (1, 2):
    output = root / f"replay-output-{index}"
    replay_env = os.environ.copy()
    replay_env.update({
        "PYTHONPATH": str(SOURCE), "PYTHONDONTWRITEBYTECODE": "1",
        "NUMBA_CACHE_DIR": str(root / "numba-replay"), "MPLCONFIGDIR": str(root / "mpl-replay"),
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1", "NUMBA_NUM_THREADS": "1",
    })
    replay = subprocess.run(
        [sys.executable, "-m", "testing.first_medium_saved_output_qualification",
         "--contract", str(enabled_contract_path), "--request", str(real_request_path),
         "--output", str(output)],
        cwd=root, env=replay_env, capture_output=True, text=True,
    )
    replay_results.append(replay.returncode)

print(json.dumps({
    "public_fixture_token_accepted_real_saved_schema_at_arbitrary_path": fixture_result.returncode == 0,
    "fixture_complete_status": json.loads((fixture_output / "COMPLETE.json").read_text())["status"],
    "phase2_receipt_verifier_accepted_fixture_complete": True,
    "resource_thread_violation_returncode_nonzero": late_result.returncode != 0,
    "resource_violation_detected_only_after_control_report": (late_output / "PHASE1_CONTROL_REPORT.json").is_file(),
    "resource_violation_detected_only_after_binding_receipt": (late_output / "ACTUAL_BINDING_RECEIPT.json").is_file(),
    "resource_failure": json.loads((late_output / "FAILURE.json").read_text()),
    "same_start_token_two_fresh_namespaces_returncodes": replay_results,
    "same_start_token_replay_both_completed": all(code == 0 for code in replay_results),
}, indent=2, sort_keys=True))
