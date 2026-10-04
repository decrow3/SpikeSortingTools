"""Adversarial lifecycle fixtures for the targeted orchestration repair."""
from concurrent.futures import ThreadPoolExecutor
import hashlib, hmac, json, os, subprocess, sys
from pathlib import Path
import pytest

PACKET = Path(__file__).resolve().parents[1]
SOURCE = PACKET / "source"
CONTRACT = PACKET / "configs" / "en_first_medium_saved_output_qualification.execution_disabled.v1.json"
sys.path.insert(0, str(SOURCE))
import testing.first_medium_saved_output_qualification as q
import testing.qualification_managed_launcher as launcher

def _run_module(module, *args):
    env = os.environ.copy(); env.update({"PYTHONPATH": str(SOURCE), "PYTHONDONTWRITEBYTECODE": "1"})
    return subprocess.run([sys.executable, "-m", module, *map(str, args)], env=env,
                          capture_output=True, text=True)

def test_dedicated_packaged_fixture_and_public_cli_isolation(tmp_path):
    output = tmp_path / "fixture-output"
    result = _run_module("testing.first_medium_saved_output_qualification_fixture", "--output", output)
    assert result.returncode == 0, result.stderr
    complete = json.loads((output / "COMPLETE.json").read_text())
    assert (complete["status"], complete["receipt_type"], complete["fixture"]) == (
        "COMPLETE_FIXTURE", "PACKAGED_FIXTURE", True)
    template = json.loads((PACKET / "fixtures" / "FIXTURE_REQUEST.template.json").read_text())
    template["arms"][0]["evidence_kind"] = "real_saved"
    template.update(data_scope="synthetic_saved_format_fixture", fixture_token="public-token")
    request = tmp_path / "arbitrary-real-saved.json"; request.write_text(json.dumps(template))
    rejected = _run_module("testing.first_medium_saved_output_qualification", "--contract", CONTRACT,
                           "--request", request, "--output", tmp_path / "must-not-exist")
    assert rejected.returncode != 0 and "production CLI has no fixture mode" in rejected.stderr
    assert not (tmp_path / "must-not-exist").exists()

def test_fixture_paths_are_packet_owned(tmp_path):
    contract = json.loads(CONTRACT.read_text())
    request = json.loads((PACKET / "fixtures" / "FIXTURE_REQUEST.template.json").read_text())
    request["arms"][0]["curated_output"] = str(tmp_path / "outside")
    with pytest.raises(RuntimeError, match="escaped"):
        q._validate_fixture_request(request, contract, PACKET)

def _phase1_receipt(root: Path, *, fixture=False, ref_identity="ref-id"):
    root.mkdir()
    control = {"label": "DIAGNOSTIC_CONTROLS_ONLY_NOT_EFFICACY", "cross_arm_scalar_emitted": False,
               "provenance": {"reference_cohort_sha256": "cohort"}}
    binding = {"REF384": {"curated_identity_digest": ref_identity,
                           "qc_request_digest": "qc", "spatial_identity": "space"}}
    (root / "PHASE1_CONTROL_REPORT.json").write_text(json.dumps(control))
    (root / "ACTUAL_BINDING_RECEIPT.json").write_text(json.dumps(binding))
    names = ["ACTUAL_BINDING_RECEIPT.json", "PHASE1_CONTROL_REPORT.json"]
    manifest = "".join(f"{hashlib.sha256((root/name).read_bytes()).hexdigest()}  {name}\n" for name in names)
    (root / "MANIFEST.sha256").write_text(manifest)
    complete = {"phase": "phase1_controls", "status": "COMPLETE_FIXTURE" if fixture else "COMPLETE_ONE_SHOT",
                "receipt_type": "PACKAGED_FIXTURE" if fixture else "REAL_CONTROL", "fixture": fixture,
                "manifest_sha256": hashlib.sha256(manifest.encode()).hexdigest(),
                "contract_sha256": "p1-contract", "request_sha256": "p1-request"}
    (root / "COMPLETE.json").write_text(json.dumps(complete))

def _phase2_request(tmp_path, receipt, expected_ref="ref-id"):
    freeze = tmp_path / (receipt.name + "-freeze.json"); freeze.write_text("{}")
    tokens = {"phase1_manifest_sha256": hashlib.sha256((receipt/"MANIFEST.sha256").read_bytes()).hexdigest(),
              "phase1_complete_sha256": hashlib.sha256((receipt/"COMPLETE.json").read_bytes()).hexdigest(),
              "phase1_contract_sha256": "p1-contract", "phase1_request_sha256": "p1-request",
              "fixed_ref_cohort_sha256": "cohort",
              "method_freeze_token": hashlib.sha256(freeze.read_bytes()).hexdigest()}
    request = {"phase1_receipt_dir": str(receipt), "method_freeze_receipt": str(freeze),
               "expected_phase1_ref_binding": {"curated_identity_digest": expected_ref,
                                               "qc_request_digest": "qc", "spatial_identity": "space"}}
    return request, tokens

def test_phase2_requires_real_phase1_and_exact_ref_binding(tmp_path):
    fixture = tmp_path / "fixture"; _phase1_receipt(fixture, fixture=True)
    request, tokens = _phase2_request(tmp_path, fixture)
    with pytest.raises(RuntimeError, match="not a successful bound phase1"):
        q._verify_phase2_receipts(request, tokens)
    real = tmp_path / "real"; _phase1_receipt(real)
    request, tokens = _phase2_request(tmp_path, real); q._verify_phase2_receipts(request, tokens)
    request["expected_phase1_ref_binding"]["curated_identity_digest"] = "other-ref"
    with pytest.raises(RuntimeError, match="REF binding mismatch"):
        q._verify_phase2_receipts(request, tokens)

def test_phase2_ref_mismatch_stops_before_b_load(monkeypatch, tmp_path):
    calls = []
    class Arm:
        provenance = {"curated_identity_digest": "wrong", "qc_request_digest": "qc", "spatial_identity": "space"}
    def load(spec, contract):
        calls.append(spec["arm_id"]); return Arm()
    monkeypatch.setattr(q, "_load_arm", load)
    request = {"arms": [{"arm_id": "REF384"}, {"arm_id": "existing_corrected_B384"}],
               "expected_phase1_ref_binding": {"curated_identity_digest": "right",
                                               "qc_request_digest": "qc", "spatial_identity": "space"}}
    with pytest.raises(RuntimeError, match="Phase-2 REF differs"):
        q._phase2_existing_b(request, {}, tmp_path)
    assert calls == ["REF384"]

def _token_case(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    registry = tmp_path / "registry"; registry.mkdir()
    contract = json.loads(CONTRACT.read_text()); contract["lifecycle"]["token_registry_root"] = str(registry)
    contract_path = tmp_path / "contract.json"; contract_path.write_text(json.dumps(contract))
    request = {"schema": q.REQUEST_SCHEMA, "phase": "phase1_controls", "arms": [],
               "prerequisite_tokens": {"phase1_start_token": "single-use-secret"}}
    return contract, contract_path, request

def _bind(request, contract_path, output):
    context = q._identity_context(request, contract_path, output, "phase1_controls")
    request["start_token_binding_sha256"] = hmac.new(
        request["prerequisite_tokens"]["phase1_start_token"].encode(),
        json.dumps(context, sort_keys=True, separators=(",", ":")).encode(), hashlib.sha256).hexdigest()

def test_token_namespace_binding_sequential_and_concurrent_replay(tmp_path):
    contract, contract_path, request = _token_case(tmp_path / "sequential")
    output = tmp_path / "run-a"; _bind(request, contract_path, output)
    q._claim_start_token(request, contract, contract_path, output, "phase1_controls")
    other = tmp_path / "run-b"; _bind(request, contract_path, other)
    with pytest.raises(FileExistsError): q._claim_start_token(request, contract, contract_path, other, "phase1_controls")
    contract, contract_path, request = _token_case(tmp_path / "concurrent")
    output = tmp_path / "run-c"; _bind(request, contract_path, output)
    def claim():
        try: q._claim_start_token(request, contract, contract_path, output, "phase1_controls"); return "claimed"
        except FileExistsError: return "rejected"
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(lambda _: claim(), range(2))) == ["claimed", "rejected"]

def test_token_cannot_move_to_another_namespace(tmp_path):
    contract, contract_path, request = _token_case(tmp_path)
    _bind(request, contract_path, tmp_path / "bound")
    with pytest.raises(RuntimeError, match="cryptographically bound"):
        q._claim_start_token(request, contract, contract_path, tmp_path / "other", "phase1_controls")

def test_resource_preconditions_and_output_quota(monkeypatch, tmp_path):
    contract = json.loads(CONTRACT.read_text()); request = tmp_path / "request.json"; request.write_text('{}')
    monkeypatch.setenv("OMP_NUM_THREADS", "5")
    for key in ("MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"): monkeypatch.setenv(key, "1")
    with pytest.raises(RuntimeError, match="before input loading"):
        q._resource_preflight("phase1_controls", contract, CONTRACT, request, tmp_path / "output")
    assert not (tmp_path / "output").exists()
    monkeypatch.setenv("OMP_NUM_THREADS", "1"); monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(RuntimeError, match="GPU visibility"):
        q._resource_preflight("phase1_controls", contract, CONTRACT, request, tmp_path / "output")
    output = tmp_path / "quota"; output.mkdir()
    monkeypatch.setattr(q, "_OUTPUT_ROOT", output); monkeypatch.setattr(q, "_OUTPUT_BYTES_MAX", 4)
    with pytest.raises(RuntimeError, match="output bound"): q._write_text(output / "too-large.txt", "12345")
    assert not (output / "too-large.txt").exists()

def test_memory_preflight_and_external_wall_timeout(monkeypatch, tmp_path):
    contract = json.loads(CONTRACT.read_text()); request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps({"phase": "phase1_controls"}))
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
        monkeypatch.setenv(key, "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", ""); monkeypatch.setenv("NVIDIA_VISIBLE_DEVICES", "void")
    monkeypatch.setattr(q.os, "sched_getaffinity", lambda _: {0})
    monkeypatch.setattr(q.resource, "getrlimit", lambda _: (q.resource.RLIM_INFINITY, q.resource.RLIM_INFINITY))
    with pytest.raises(RuntimeError, match="address-space limit"):
        q._resource_preflight("phase1_controls", contract, CONTRACT, request_path, tmp_path / "output")

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0] if args else "worker", 0.01)
    monkeypatch.setattr(launcher.subprocess, "run", timeout)
    with pytest.raises(RuntimeError, match="external wall timeout"):
        launcher.launch(CONTRACT, request_path, tmp_path / "timeout-output", "unused.worker")
