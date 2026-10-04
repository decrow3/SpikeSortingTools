"""Managed-entry known answers for the execution-disabled Phase-3 composition."""

import hashlib
import hmac
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest
import jsonschema

PACKET = Path(__file__).resolve().parents[1]
SOURCE = PACKET / "source"
PRODUCER = PACKET / "producer"
BASE_CONTRACT = PACKET / "configs/en_first_medium_saved_output_phase3.execution_disabled.v1.json"
sys.path.insert(0, str(SOURCE))
sys.path.insert(0, str(PRODUCER))

import testing.first_medium_saved_output_qualification as q
import testing.qualification_managed_launcher as launcher
from testing.first_medium_evaluator import (
    KILOSORT_SOURCE_DIGEST,
    KILOSORT_SOURCE_FILES,
    spatial_event_lineage_sha256,
)
from pipeline.config import fingerprint
from testing.luke_amplitude_dropout_audit import read_curated_arrays
import en_first_medium_trained_pair_launch as pair_producer


REGION = {
    "processing_depth_um": [0.0, 400.0],
    "scoring_depth_um": [100.0, 300.0],
    "minimum_edge_exclusion_um": 50.0,
    "source": "saved-format known-answer fixture",
}
CLOCK = {
    "saved_frame": "crop_local_samples",
    "crop_origin_global_frame": 0,
    "local_frames_half_open": [0, 4000],
    "global_frames_half_open": [0, 4000],
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def managed(contract: Path, request: Path, output: Path):
    """Launch from the frozen source root so cwd cannot select workspace code."""
    previous = Path.cwd()
    os.chdir(SOURCE)
    try:
        return launcher.launch(contract, request, output, "testing.first_medium_saved_output_qualification")
    finally:
        os.chdir(previous)


def _times_by_unit(count: int) -> tuple[np.ndarray, np.ndarray]:
    one = np.arange(1, 2000, 20, dtype=np.int64)[:count]
    two = np.arange(2001, 4000, 20, dtype=np.int64)[:count]
    return np.r_[one, two], np.r_[np.ones(len(one), dtype=np.int64), np.full(len(two), 2, dtype=np.int64)]


def write_sort(root: Path, arm_id: str, count: int, offset: int, *, output_name="sorter_output") -> tuple[Path, Path]:
    curated = root / arm_id / output_name
    qc = root / arm_id / "qc"
    curated.mkdir(parents=True)
    qc.mkdir(parents=True)
    times, clusters = _times_by_unit(count)
    clusters = clusters + offset
    full_st = np.c_[times, np.zeros(len(times)), np.full(len(times), 10.0)]
    np.save(curated / "spike_times.npy", times)
    np.save(curated / "spike_clusters.npy", clusters)
    np.save(curated / "full_st.npy", full_st)
    np.save(curated / "kept_spikes.npy", np.arange(len(times), dtype=np.int64))
    positions = np.c_[np.zeros(len(times)), np.full(len(times), 200.0)]
    np.save(curated / "spike_positions.npy", positions)
    ops = {
        "nblocks": 0,
        "dshift": None,
        "chanMap": np.arange(384),
        "xc": np.tile([0.0, 32.0], 192),
        "yc": np.arange(384, dtype=float),
    }
    np.save(curated / "ops.npy", ops, allow_pickle=True)
    labels = "cluster_id\tKSLabel\n" + "".join(f"{unit}\tgood\n" for unit in np.unique(clusters))
    (curated / "cluster_KSLabel.tsv").write_text(labels)
    # Deliberately do not write amplitudes.npy. The accepted loader must derive
    # amplitudes only from full_st[kept_spikes][:, 2].
    return curated, qc


def identity(curated: Path) -> str:
    _, hashes = read_curated_arrays(curated)
    hashes["spike_positions.npy"] = sha(curated / "spike_positions.npy")
    labels = (curated / "cluster_KSLabel.tsv").read_bytes()
    return fingerprint({"curated": str(curated.resolve()), "files": hashes,
                        "labels_sha256": hashlib.sha256(labels).hexdigest()})


def spatial(curated: Path) -> dict:
    positions = np.load(curated / "spike_positions.npy")
    times = np.load(curated / "spike_times.npy")
    clusters = np.load(curated / "spike_clusters.npy")
    full_st = np.load(curated / "full_st.npy")
    kept = np.load(curated / "kept_spikes.npy")
    ops = np.load(curated / "ops.npy", allow_pickle=True).item()
    geometry = np.stack([np.asarray(ops["xc"], dtype="<f8"), np.asarray(ops["yc"], dtype="<f8")], axis=1)
    dependency = PACKET / "dependencies/kilosort" / KILOSORT_SOURCE_DIGEST
    return {
        "schema": "evaluator-spatial-provenance-v1",
        "path": str((curated / "spike_positions.npy").resolve()),
        "sha256": sha(curated / "spike_positions.npy"),
        "ops_path": str((curated / "ops.npy").resolve()),
        "ops_sha256": sha(curated / "ops.npy"),
        "row_count": len(times),
        "event_lineage_sha256": spatial_event_lineage_sha256(times, clusters, full_st[kept, 2], positions),
        "reference_frame": "kilosort_probe_xy_feature_mass_unshifted_nblocks0",
        "units": "micrometers",
        "columns": [{"index": 0, "semantic": "x", "units": "micrometers"},
                    {"index": 1, "semantic": "physical_probe_depth_y", "units": "micrometers"}],
        "row_semantics": "exported_event_order_after_native_kept_spikes",
        "clock_binding": CLOCK,
        "spatial_region": REGION,
        "geometry_float64_sha256": hashlib.sha256(np.ascontiguousarray(geometry).tobytes()).hexdigest(),
        "source_generation": {"function": "kilosort.postprocessing.compute_spike_positions",
                              "dependency": {"schema": "content-addressed-python-source-root-v1",
                                             "root": str(dependency.resolve()),
                                             "digest_sha256": KILOSORT_SOURCE_DIGEST,
                                             "files": KILOSORT_SOURCE_FILES}},
    }


def arm_spec(arm_id: str, curated: Path, qc: Path) -> dict:
    return {
        "arm_id": arm_id,
        "evidence_kind": "real_saved",
        "curated_output": str(curated.resolve()),
        "qc_dir": str(qc.resolve()),
        "provenance": {"arm_contract_id": arm_id, "input_identity": "fixture-input",
                       "clock_identity": "fixture-clock", "geometry_identity": "fixture-geometry",
                       "runtime_identity": "fixture-runtime", "config_identity": "fixture-config"},
        "expected_identity_digest": identity(curated),
        "expected_qc_request_digest": "UNMEASURED_ABSENT_TRUNCATION_QC",
        "clock_normalization": CLOCK,
        "spatial_provenance": spatial(curated),
    }


def phase1_receipt(root: Path, cohort_sha: str, ref_binding: dict) -> tuple[Path, dict]:
    root.mkdir()
    control = {"label": "DIAGNOSTIC_CONTROLS_ONLY_NOT_EFFICACY", "cross_arm_scalar_emitted": False,
               "provenance": {"reference_cohort_sha256": cohort_sha}}
    binding = {"REF384": ref_binding}
    (root / "PHASE1_CONTROL_REPORT.json").write_text(json.dumps(control))
    (root / "ACTUAL_BINDING_RECEIPT.json").write_text(json.dumps(binding))
    manifest = "".join(f"{sha(root / name)}  {name}\n" for name in
                       ["ACTUAL_BINDING_RECEIPT.json", "PHASE1_CONTROL_REPORT.json"])
    (root / "MANIFEST.sha256").write_text(manifest)
    complete = {"phase": "phase1_controls", "status": "COMPLETE_ONE_SHOT",
                "receipt_type": "REAL_CONTROL", "fixture": False,
                "manifest_sha256": sha(root / "MANIFEST.sha256"),
                "contract_sha256": "phase1-contract", "request_sha256": "phase1-request"}
    (root / "COMPLETE.json").write_text(json.dumps(complete))
    tokens = {"phase1_manifest_sha256": sha(root / "MANIFEST.sha256"),
              "phase1_complete_sha256": sha(root / "COMPLETE.json"),
              "phase1_contract_sha256": "phase1-contract", "phase1_request_sha256": "phase1-request"}
    return root, tokens


def write_inventory(path: Path, curated: Path, *, repaired: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if repaired:
        native = {name: {"bytes": (curated / name).stat().st_size, "sha256": sha(curated / name)}
                  for name in q.CONSUMED_CURATED_FILES}
        payload = {"native": native, "pre_extraction_snapshot": {}}
    else:
        payload = {"native": {name: sha(curated / name) for name in q.CONSUMED_CURATED_FILES},
                   "snapshot": {}}
    path.write_text(json.dumps(payload, sort_keys=True))


def producer_receipts(tmp_path: Path, ref_curated: Path, repaired_curated: Path, *, valid: bool = True):
    packet = tmp_path / "producer_packet"
    config_dir = packet / "config"
    contract_dir = packet / "contract"
    source_dir = packet / "source"
    config_dir.mkdir(parents=True)
    contract_dir.mkdir()
    source_dir.mkdir()
    pair_root = ref_curated.parents[1]
    repaired_root = repaired_curated.parent
    ref_config = {"run_root": str((pair_root / "REF384_repeat").resolve())}
    repaired_config = {
        "trained": {
            "run_root": str(repaired_root.resolve()),
            "launch_evidence_dir": str((repaired_root / "launch_evidence").resolve()),
        },
        "native_invocation": {"results_dir": str(repaired_curated.resolve())},
    }
    config_paths = {}
    for arm, value in (("REF384_repeat", ref_config), ("repaired_B384", repaired_config),
                       ("historical_screen", {})):
        path = config_dir / f"{arm}.json"
        path.write_text(json.dumps(value, sort_keys=True))
        config_paths[arm] = path
    attempt = tmp_path / "producer_attempt"
    attempt.mkdir()
    (attempt / "VOLTAGE_CONTINUITY.json").write_text('{}\n')
    producer_source = source_dir / "en_first_medium_trained_pair_launch.py"
    producer_source.write_bytes((PRODUCER / "en_first_medium_trained_pair_launch.py").read_bytes())
    contract = {
        "schema": pair_producer.SCHEMA,
        "arm_order": list(pair_producer.ARM_ORDER),
        "configs": {arm: {"path": f"../config/{path.name}", "sha256": sha(path)}
                    for arm, path in config_paths.items()},
        "sources": {"pair_launcher": {
            "path": "../source/en_first_medium_trained_pair_launch.py",
            "sha256": sha(producer_source),
        }},
        "planned_paths": {"root": str(pair_root.resolve()), "logs": str((tmp_path / "logs").resolve()),
                          "attempt_evidence": str(attempt.resolve())},
        "resources": {"persistent_output_bytes_max": 20_000_000,
                      "live_output_bytes_max": 18_951_424,
                      "terminal_control_reserve_bytes": 1_048_576,
                      "counted_late_log_bytes_max": 262_144},
    }
    contract_path = contract_dir / "pair.json"
    contract_path.write_text(json.dumps(contract, sort_keys=True))
    contract["_contract_dir"] = str(contract_dir.resolve())
    ref_root = pair_root / "REF384_repeat"
    ref_inventory = ref_root / "launch_evidence/artifact_inventory.json"
    repaired_inventory = repaired_root / "launch_evidence/artifact_inventory.json"
    write_inventory(ref_inventory, ref_curated, repaired=False)
    write_inventory(repaired_inventory, repaired_curated, repaired=True)
    ref_complete = ref_root / "COMPLETE.json"
    ref_complete.write_text(json.dumps({"status": "complete", "arm": "REF384_repeat",
                                        "completion_written_last": True}))
    repaired_complete = repaired_root / "COMPLETE.json"
    repaired_complete.write_text(json.dumps({"status": "complete",
        "config_sha256": sha(config_paths["repaired_B384"]),
        "required_artifacts_validated": True, "completion_written_last": True}))
    states = {
        "REF384_repeat": pair_producer._execution_complete(
            {"status": "complete", "complete_path": str(ref_complete)}, "REF384_repeat", contract),
        "repaired_B384": pair_producer._execution_complete(
            {"status": "complete", "complete_path": str(repaired_complete)}, "repaired_B384", contract),
    }
    pair_producer.finalize_success(contract, {"execution_states": states})
    terminal_path = attempt / "ATTEMPT_TERMINAL.pending.json"
    complete_path = pair_root / "COMPLETE.json"
    if not valid:
        os.chmod(terminal_path, 0o600)
        terminal = json.loads(terminal_path.read_text())
        terminal["both_arms_and_ordinary_writers_stopped"] = False
        terminal_path.write_text(json.dumps(terminal, sort_keys=True))
        os.chmod(complete_path, 0o600)
        complete = json.loads(complete_path.read_text())
        complete["terminal_attempt_receipt_sha256"] = sha(terminal_path)
        complete_path.write_text(json.dumps(complete, sort_keys=True))
    receipts = {"pair_complete": str(complete_path), "attempt_terminal": str(terminal_path),
                "producer_contract": str(contract_path)}
    tokens = {"pair_complete_sha256": sha(complete_path),
              "pair_attempt_terminal_sha256": sha(terminal_path),
              "pair_contract_sha256": sha(contract_path)}
    return receipts, tokens


def build_case(
    tmp_path: Path, repaired_count: int, *, valid_pair: bool = True,
    reference_count: int = 20, existing_count: int = 18, repeat_count: int = 20,
):
    data = tmp_path / "data"
    pair_root = tmp_path / "producer_run"
    specs = []
    for arm_id, count, offset in [
        ("REF384", reference_count, 0),
        ("existing_corrected_B384", existing_count, 10),
    ]:
        curated, qc = write_sort(data, arm_id, count, offset)
        specs.append(arm_spec(arm_id, curated, qc))
    repaired_curated, repaired_qc = write_sort(
        pair_root, "repaired_B384", repaired_count, 20, output_name="native_results"
    )
    repeat_curated, repeat_qc = write_sort(pair_root, "REF384_repeat", repeat_count, 30)
    specs.extend([arm_spec("repaired_B384", repaired_curated, repaired_qc),
                  arm_spec("REF384_repeat", repeat_curated, repeat_qc)])
    contract = json.loads(BASE_CONTRACT.read_text())
    contract["execution_enabled"] = True
    phase = contract["phases"]["phase3_repaired_repeat"]
    phase["real_execution_enabled"] = True
    contract["recording_domain"].update(sampling_frequency_hz=1000.0, duration_s=4.0,
                                         crop_origin_global_frame=0,
                                         global_frames_half_open=[0, 4000],
                                         local_frames_half_open=[0, 4000],
                                         cell_edges_local_frames=[0, 1000, 2000, 3000, 4000],
                                         spatial_region=REGION)
    cohort = np.array([1, 2], dtype=np.int64)
    cohort_sha = q._cohort_digest(cohort)
    contract["fixed_ref_cohort"] = {"ordered_cluster_ids": cohort.tolist(), "cohort_sha256": cohort_sha}
    contract["implementation_boundary"]["active_module_sha256"][q.__name__] = sha(Path(q.__file__))
    registry = tmp_path / "registry"
    registry.mkdir()
    contract["lifecycle"]["token_registry_root"] = str(registry.resolve())
    loaded_ref = q._load_arm(specs[0], contract)
    ref_binding = {key: loaded_ref.provenance[key] for key in
                   ("curated_identity_digest", "qc_request_digest", "spatial_identity")}
    p1, p1_tokens = phase1_receipt(tmp_path / "phase1", cohort_sha, ref_binding)
    freeze = tmp_path / "METHOD_FREEZE.json"
    freeze.write_text("{}\n")
    pairs, pair_tokens = producer_receipts(
        tmp_path, repeat_curated, repaired_curated, valid=valid_pair
    )
    tokens = {"composition_review_token": "reviewed-composition", "method_freeze_token": sha(freeze),
              **p1_tokens, **pair_tokens, "phase3_start_token": "single-use-phase3-secret",
              "fixed_ref_cohort_sha256": cohort_sha}
    phase["required_tokens"] = tokens
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(contract, sort_keys=True))
    request = {"schema": q.REQUEST_SCHEMA, "phase": "phase3_repaired_repeat", "arms": specs,
               "prerequisite_tokens": tokens, "reference_cohort_sha256": cohort_sha,
               "expected_phase1_ref_binding": ref_binding, "phase1_receipt_dir": str(p1.resolve()),
               "method_freeze_receipt": str(freeze.resolve()), "pair_receipts": pairs}
    output = tmp_path / "output"
    context = q._identity_context(request, contract_path, output, "phase3_repaired_repeat")
    request["start_token_binding_sha256"] = hmac.new(
        tokens["phase3_start_token"].encode(),
        json.dumps(context, sort_keys=True, separators=(",", ":")).encode(), hashlib.sha256,
    ).hexdigest()
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(request, sort_keys=True))
    return contract_path, request_path, output


def mutate_pair_complete(contract_path: Path, request_path: Path, output: Path, mutation) -> None:
    request = json.loads(request_path.read_text())
    contract = json.loads(contract_path.read_text())
    complete_path = Path(request["pair_receipts"]["pair_complete"])
    os.chmod(complete_path, 0o600)
    complete = json.loads(complete_path.read_text())
    mutation(complete)
    complete_path.write_text(json.dumps(complete, sort_keys=True))
    request["prerequisite_tokens"]["pair_complete_sha256"] = sha(complete_path)
    contract["phases"]["phase3_repaired_repeat"]["required_tokens"] = request["prerequisite_tokens"]
    contract_path.write_text(json.dumps(contract, sort_keys=True))
    request.pop("start_token_binding_sha256", None)
    context = q._identity_context(request, contract_path, output, "phase3_repaired_repeat")
    request["start_token_binding_sha256"] = hmac.new(
        request["prerequisite_tokens"]["phase3_start_token"].encode(),
        json.dumps(context, sort_keys=True, separators=(",", ":")).encode(), hashlib.sha256,
    ).hexdigest()
    request_path.write_text(json.dumps(request, sort_keys=True))


@pytest.mark.parametrize("repaired_count,expected_delta,expected_pass", [(18, 0.0, False), (19, 0.05, False), (20, 0.10, True)])
def test_managed_phase3_strict_threshold_and_four_arm_known_answers(tmp_path, repaired_count, expected_delta, expected_pass):
    contract, request, output = build_case(tmp_path, repaired_count)
    result = managed(contract, request, output)
    assert result.returncode == 0, result.stderr
    report = json.loads((output / "PHASE3_COMPACT_REPORT.json").read_text())
    assert report["primary_retention"]["DeltaR"] == pytest.approx(expected_delta)
    assert report["primary_retention"]["lower95"] == pytest.approx(expected_delta)
    assert report["primary_retention"]["passes"] is expected_pass
    assert report["repeatability"]["R_repeat"] == 1.0
    assert report["repeatability"]["passes"] is True
    assert report["bindings"]["qc_status_by_arm"] == {arm: "UNMEASURED" for arm in
        ["REF384", "existing_corrected_B384", "repaired_B384", "REF384_repeat"]}
    assert report["waveform"]["status"] == "UNMEASURED"
    schema = json.loads((PACKET / "schemas/COMPACT_REPORT_SCHEMA.json").read_text())
    jsonschema.validate(report, schema)
    assert set(report["waveform"]["pair_coverage"].values()) == {"UNMEASURED"}
    assert report["decision"]["advance"] is False
    producer_binding = json.loads((output / "PRODUCER_BINDING_RECEIPT.json").read_text())
    assert set(producer_binding["prospective_arm_inventory_bindings"]) == {
        "REF384_repeat", "repaired_B384"
    }
    for state in producer_binding["pair_complete"]["execution_states"].values():
        assert set(("artifact_inventory_path", "artifact_inventory_sha256")) <= set(state)
    assert producer_binding["prospective_arm_inventory_bindings"]["REF384_repeat"]["curated_output"].endswith(
        "/REF384_repeat/sorter_output"
    )
    assert producer_binding["prospective_arm_inventory_bindings"]["repaired_B384"]["curated_output"].endswith(
        "/repaired_B384/native_results"
    )
    assert set(report["diagnostics"]["fixed_60s_cells"]) == {
        "existing_corrected_B384", "repaired_B384", "REF384_repeat"
    }
    assert not (Path(json.loads(request.read_text())["arms"][0]["curated_output"]) / "amplitudes.npy").exists()
    assert json.loads((output / "COMPLETE.json").read_text())["receipt_type"] == "REAL_PHASE3_COMPOSITION"


@pytest.mark.parametrize(
    "repaired_count,expected_delta,primary_pass",
    [(84, 0.04, False), (85, 0.05, False), (86, 0.06, True)],
)
@pytest.mark.parametrize(
    "repeat_count,expected_repeat,repeat_pass",
    [(97, 0.97, False), (98, 0.98, True), (99, 0.99, True)],
)
def test_managed_phase3_decision_precedence_grid_with_required_evidence_unmeasured(
    tmp_path, repaired_count, expected_delta, primary_pass,
    repeat_count, expected_repeat, repeat_pass,
):
    contract, request, output = build_case(
        tmp_path, repaired_count, reference_count=100,
        existing_count=80, repeat_count=repeat_count,
    )
    result = managed(contract, request, output)
    assert result.returncode == 0, result.stderr
    report = json.loads((output / "PHASE3_COMPACT_REPORT.json").read_text())

    # Numeric and component verdicts are invariant to overall-status mapping.
    assert report["primary_retention"]["DeltaR"] == pytest.approx(expected_delta)
    assert report["primary_retention"]["lower95"] == pytest.approx(expected_delta)
    assert report["primary_retention"]["passes"] is primary_pass
    assert report["repeatability"]["R_repeat"] == pytest.approx(expected_repeat)
    assert report["repeatability"]["passes"] is repeat_pass

    # The thin path has legitimate required-evidence missingness by design.
    assert report["waveform"]["status"] == "UNMEASURED"
    assert set(report["bindings"]["qc_status_by_arm"].values()) == {"UNMEASURED"}
    assert report["diagnostics"]["guardrails"]["status"] == "NOT_COMPOSED_BY_PHASE3_THIN_PATH"
    assert report["decision"]["status"] == "inconclusive"
    assert report["decision"]["advance"] is False

    reasons = set(report["decision"]["reason_codes"])
    assert ("DELTA_LOWER95_NOT_ABOVE_0_05" in reasons) is (not primary_pass)
    assert ("REF_REPEAT_RETENTION_BELOW_0_98" in reasons) is (not repeat_pass)
    assert {"QC_UNMEASURED", "WAVEFORM_UNMEASURED"} <= reasons


def test_unfinished_pair_receipt_rejected_before_start_or_output(tmp_path):
    contract, request, output = build_case(tmp_path, 20, valid_pair=False)
    result = managed(contract, request, output)
    assert result.returncode != 0
    assert "terminal receipt is not bound" in result.stderr
    assert not output.exists()


def test_input_mutation_rejected_before_scoring_and_never_completes(tmp_path):
    contract, request, output = build_case(tmp_path, 20)
    request_data = json.loads(request.read_text())
    repaired = Path(request_data["arms"][2]["curated_output"])
    times = np.load(repaired / "spike_times.npy")
    times[0] += 1
    np.save(repaired / "spike_times.npy", times)
    result = managed(contract, request, output)
    assert result.returncode != 0
    assert "differs from the bound inventory" in result.stderr
    assert not output.exists()
    assert not (output / "PHASE3_COMPACT_REPORT.json").exists()
    assert not (output / "COMPLETE.json").exists()


@pytest.mark.parametrize("fault", ["absent_state", "swapped_states", "wrong_completion_hash", "unrelated_same_basename"])
def test_managed_entry_rejects_producer_provenance_faults_before_scoring(tmp_path, fault):
    contract, request, output = build_case(tmp_path, 20)

    def mutation(complete):
        states = complete["execution_states"]
        if fault == "absent_state":
            states.pop("repaired_B384")
        elif fault == "swapped_states":
            states["REF384_repeat"], states["repaired_B384"] = (
                states["repaired_B384"], states["REF384_repeat"])
        elif fault == "wrong_completion_hash":
            states["repaired_B384"]["complete_sha256"] = "0" * 64
        else:
            unrelated = tmp_path / "unrelated" / "artifact_inventory.json"
            unrelated.parent.mkdir()
            source = Path(states["REF384_repeat"]["artifact_inventory_path"])
            unrelated.write_bytes(source.read_bytes())
            states["REF384_repeat"]["artifact_inventory_path"] = str(unrelated)
            states["REF384_repeat"]["artifact_inventory_sha256"] = sha(unrelated)

    mutate_pair_complete(contract, request, output, mutation)
    result = managed(contract, request, output)
    assert result.returncode != 0
    assert not output.exists()


def test_fixed_cohort_unmatched_unit_is_zero_and_label_permutation_is_bijective():
    rule = q.measurement.MatchRule(tolerance_frames=0, minimum_overlap=0.1, primary_retention=0.5)
    ref = {"st": np.array([1, 2, 10, 11]), "cl": np.array([1, 1, 2, 2])}
    permuted = {"st": np.array([1, 2]), "cl": np.array([101, 101])}
    result = q.measurement.reference_event_retention(ref, permuted, [1, 2], rule)
    assert result["R"] == 0.5
    table = result["unit_retention"].set_index("reference_unit")
    assert table.loc[2, "status"] == "unmatched_zero" and table.loc[2, "retention"] == 0.0
    assert result["primary_matches"].baseline_cluster.is_unique
    assert result["primary_matches"].candidate_cluster.is_unique


def test_phase3_ref_binding_failure_precedes_every_candidate_load(monkeypatch, tmp_path):
    calls = []

    class Ref:
        provenance = {"curated_identity_digest": "wrong", "qc_request_digest": "qc",
                      "spatial_identity": "space"}

    def load(spec, contract):
        calls.append(spec["arm_id"])
        return Ref()

    monkeypatch.setattr(q, "_load_arm", load)
    request = {"arms": [{"arm_id": arm} for arm in
                        ["REF384", "existing_corrected_B384", "repaired_B384", "REF384_repeat"]],
               "expected_phase1_ref_binding": {"curated_identity_digest": "right",
                                               "qc_request_digest": "qc", "spatial_identity": "space"}}
    with pytest.raises(RuntimeError, match="Phase-3 REF differs"):
        q._phase3_repaired_repeat(request, {}, tmp_path)
    assert calls == ["REF384"]
