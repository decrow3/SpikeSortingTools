from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

import en_first_medium_trained_pair_launch as pair


PACKET = Path(__file__).resolve().parents[1]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resources() -> dict:
    return {
        "wall_seconds_max": 2700,
        "ram_bytes_max": 68719476736,
        "cpu_threads_max": 16,
        "gpu_count_max": 1,
        "persistent_output_bytes_max": 17179869184,
        "minimum_free_bytes": 214748364800,
        "service_restart": "no",
        "service_start_limit": 1,
    }


def make_contract(tmp_path: Path, *, enabled: bool = False) -> tuple[Path, dict]:
    bound = tmp_path / "bound"
    bound.mkdir()
    configs = {}
    for name in ("REF384_repeat", "repaired_B384", "historical_screen"):
        path = bound / f"{name}.json"
        path.write_text("{}\n")
        configs[name] = {"path": str(path), "sha256": sha(path)}
    roles = (
        "pair_launcher", "historical_ref_runner", "trained_terminal",
        "support_candidate", "prewhitening", "kilosort_io",
        "kilosort_preprocessing", "kilosort_spikedetect",
        "kilosort_template_matching", "kilosort_run",
    )
    sources = {}
    for role in roles:
        path = bound / f"{role}.py"
        path.write_text(f"# {role}\n")
        sources[role] = {"path": str(path), "sha256": sha(path)}
    contract = {
        "schema": pair.SCHEMA,
        "status": "enabled_reviewed_once" if enabled else "execution_disabled_pending_h1_review",
        "execution_enabled": enabled,
        "approval": {"h1_review_manifest_sha256": "a" * 64 if enabled else None},
        "arm_order": list(pair.ARM_ORDER),
        "configs": configs,
        "sources": sources,
        "planned_paths": {
            "root": str(tmp_path / "pair_root"),
            "numba_cache": str(tmp_path / "numba_cache"),
            "logs": str(tmp_path / "logs"),
            "staged_source_root": str(tmp_path / "stage"),
        },
        "resources": resources(),
        "stop_conditions": ["test"],
        "checkpoint_policy": "arms atomic; no within-arm resume",
        "completion_condition": "both arms validated",
        "gates": {"clock_review_verdict": "GO_CLOCK_REPAIR_ONLY", "rf_holdout_sealed": True},
    }
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(contract, sort_keys=True) + "\n")
    return path, contract


def test_validate_disabled_contract_reads_only_bound_compact_files(tmp_path):
    path, contract = make_contract(tmp_path)
    forbidden = tmp_path / "must_not_open_voltage.raw"
    forbidden.write_bytes(b"voltage")
    os.chmod(forbidden, 0)
    observed = pair.validate_contract(path, sha(path))
    assert observed["execution_enabled"] is False
    assert not Path(contract["planned_paths"]["root"]).exists()


def test_execute_disabled_fails_before_namespace_or_runner(tmp_path, monkeypatch):
    _, contract = make_contract(tmp_path)
    monkeypatch.setenv(pair.PERSISTENT_MARKER, "1")
    called = []
    with pytest.raises(PermissionError, match="execution-disabled"):
        pair.execute_pair(contract, ref_executor=lambda _: called.append("REF"),
                          repaired_executor=lambda _: called.append("B"),
                          resource_probe=lambda _: None)
    assert called == []
    assert not Path(contract["planned_paths"]["root"]).exists()


class StateOnlyReceipt:
    def __init__(self, path: Path):
        self.path = path
    def get(self, key, default=None):
        if key == "status":
            return "complete"
        if key == "complete_path":
            return str(self.path)
        raise AssertionError(f"scientific field inspected: {key}")


def test_enabled_fixture_runs_ref_then_b_without_scientific_outcome_access(tmp_path, monkeypatch):
    _, contract = make_contract(tmp_path, enabled=True)
    monkeypatch.setenv(pair.PERSISTENT_MARKER, "1")
    order = []
    ref_complete = tmp_path / "ref.COMPLETE.json"
    b_complete = tmp_path / "b.COMPLETE.json"
    ref_complete.write_text('{"status":"complete","spikes":"forbidden"}\n')
    b_complete.write_text('{"status":"complete","score":"forbidden"}\n')
    def ref(_):
        order.append("REF384_repeat")
        return StateOnlyReceipt(ref_complete)
    def repaired(c):
        assert (Path(c["planned_paths"]["root"]) / "launch_evidence/REF384_repeat_EXECUTION_STATE.json").is_file()
        order.append("repaired_B384")
        return StateOnlyReceipt(b_complete)
    complete = pair.execute_pair(contract, ref_executor=ref, repaired_executor=repaired,
                                 resource_probe=lambda _: None)
    assert order == list(pair.ARM_ORDER)
    assert complete["scientific_outcomes_inspected_between_arms"] is False
    assert (Path(contract["planned_paths"]["root"]) / "COMPLETE.json").is_file()


def test_ref_failure_stops_before_b_and_preserves_failure(tmp_path, monkeypatch):
    _, contract = make_contract(tmp_path, enabled=True)
    monkeypatch.setenv(pair.PERSISTENT_MARKER, "1")
    called = []
    def fail(_):
        called.append("REF384_repeat")
        raise RuntimeError("injected REF failure")
    with pytest.raises(RuntimeError, match="injected REF failure"):
        pair.execute_pair(contract, ref_executor=fail,
                          repaired_executor=lambda _: called.append("repaired_B384"),
                          resource_probe=lambda _: None)
    root = Path(contract["planned_paths"]["root"])
    assert called == ["REF384_repeat"]
    assert json.loads((root / "launch_evidence/RUN_FAILURE.json").read_text())["stage"] == "REF384_repeat"
    assert not (root / "COMPLETE.json").exists()


def test_b_failure_preserves_ref_state_and_no_completion(tmp_path, monkeypatch):
    _, contract = make_contract(tmp_path, enabled=True)
    monkeypatch.setenv(pair.PERSISTENT_MARKER, "1")
    ref_complete = tmp_path / "ref.COMPLETE.json"
    ref_complete.write_text('{"status":"complete"}\n')
    def fail(_):
        raise RuntimeError("injected B failure")
    with pytest.raises(RuntimeError, match="injected B failure"):
        pair.execute_pair(contract, ref_executor=lambda _: StateOnlyReceipt(ref_complete),
                          repaired_executor=fail, resource_probe=lambda _: None)
    root = Path(contract["planned_paths"]["root"])
    assert (root / "launch_evidence/REF384_repeat_EXECUTION_STATE.json").is_file()
    assert json.loads((root / "launch_evidence/RUN_FAILURE.json").read_text())["stage"] == "repaired_B384"
    assert not (root / "COMPLETE.json").exists()


def test_existing_namespace_rejected_before_runners(tmp_path, monkeypatch):
    _, contract = make_contract(tmp_path, enabled=True)
    monkeypatch.setenv(pair.PERSISTENT_MARKER, "1")
    Path(contract["planned_paths"]["root"]).mkdir()
    called = []
    with pytest.raises(FileExistsError, match="namespace already exists"):
        pair.execute_pair(contract, ref_executor=lambda _: called.append("REF"),
                          repaired_executor=lambda _: called.append("B"),
                          resource_probe=lambda _: None)
    assert called == []


def test_contract_rejects_swapped_order_and_source_mutation(tmp_path):
    path, contract = make_contract(tmp_path)
    contract["arm_order"] = list(reversed(pair.ARM_ORDER))
    path.write_text(json.dumps(contract) + "\n")
    with pytest.raises(ValueError, match="arm order differs"):
        pair.validate_contract(path, sha(path))
    contract["arm_order"] = list(pair.ARM_ORDER)
    path.write_text(json.dumps(contract) + "\n")
    Path(contract["sources"]["trained_terminal"]["path"]).write_text("mutated\n")
    with pytest.raises(ValueError, match="source trained_terminal hash differs"):
        pair.validate_contract(path, sha(path))


def test_missing_persistent_marker_rejected(tmp_path, monkeypatch):
    _, contract = make_contract(tmp_path, enabled=True)
    monkeypatch.delenv(pair.PERSISTENT_MARKER, raising=False)
    with pytest.raises(PermissionError, match="persistent service marker"):
        pair.execute_pair(contract, resource_probe=lambda _: None)


def test_frozen_packet_contract_is_disabled_and_binds_distinct_arm_paths():
    path = PACKET / "contract/en_first_medium_pair.execution_disabled.v1.json"
    contract = pair.validate_contract(path, sha(path))
    ref = json.loads((PACKET / "config/REF384_repeat.execution_disabled.frozen.v2.json").read_text())
    repaired = json.loads((PACKET / "config/repaired_B384.execution_disabled.clock_v5.json").read_text())
    assert contract["execution_enabled"] is False
    assert contract["approval"]["h1_review_manifest_sha256"] is None
    assert contract["between_arm_policy"].startswith("inspect execution status")
    assert ref["execution_enabled"] is False
    assert ref["effective_requirements"]["support_policy_enabled"] is False
    assert ref["crop"] == {"imin": 208498882, "imax": 226498783}
    assert ref["output_clock"] == repaired["output_clock"]
    assert repaired["execution_enabled"] is False
    assert repaired["trained"]["allow_saved_smoke_bank"] is False
    assert contract["sources"]["historical_ref_runner"]["sha256"] != contract["sources"]["trained_terminal"]["sha256"]
