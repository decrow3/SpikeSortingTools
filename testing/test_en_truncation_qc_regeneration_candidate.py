import copy
import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from testing.en_truncation_qc_regeneration_candidate import prevalidate_structure

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/en_truncation_qc_regeneration.execution_disabled.v2.json"
CLI = ROOT / "testing/en_truncation_qc_regeneration_candidate.py"
PYTHON = ROOT / "environments/rescue-production/.venv/bin/python"


def config_digest():
    return hashlib.sha256(CONFIG.read_bytes()).hexdigest()


def invoke(output, *extra):
    return subprocess.run([
        str(PYTHON), str(CLI), "--config", str(CONFIG),
        "--expected-config-sha256", config_digest(), *extra,
        *([] if output is None else ["--output", str(output)]),
    ], cwd=output.parent if output else "/tmp", text=True, capture_output=True)


def fixture_gate(tmp_path, **changes):
    config = json.loads(CONFIG.read_text())
    source_hash = next(row["sha256"] for row in config["source_bindings"] if Path(row["path"]).name.startswith("en_truncation"))
    gate = {
        "schema": "truncation-qc-one-execution-gate-v1", "verdict": "ONE_EXECUTION_GO",
        "review_manifest_sha256": "1" * 64, "disabled_config_sha256": config_digest(),
        "source_sha256": source_hash,
        "frozen_real_output_root": config["activation"]["frozen_real_output_root"],
        "one_shot_token": "controlled-gate-transition-fixture", "synthetic_fixture": True,
    }
    gate.update(changes)
    path = tmp_path / "gate.json"
    path.write_text(json.dumps(gate))
    return path


def test_contract_is_disabled_and_phase_specific():
    config = json.loads(CONFIG.read_text())
    prevalidate_structure(config, phase="review_pending")
    with pytest.raises(ValueError, match="invalid for requested phase real"):
        prevalidate_structure(config, phase="real")


def test_actual_cli_exact_enabled_gate_transition(tmp_path):
    gate = fixture_gate(tmp_path)
    result = invoke(None, "--gate-fixture-check", "--gate", str(gate))
    assert result.returncode == 0, result.stderr
    receipt = json.loads(result.stdout)
    assert receipt["effective_status"] == "reviewed_one_execution_enabled"
    assert receipt["effective_execution_enabled"] is True
    assert receipt["synthetic_fixture"] is True


def test_actual_cli_wrong_gate_fails(tmp_path):
    gate = fixture_gate(tmp_path, verdict="NO_GO")
    result = invoke(None, "--gate-fixture-check", "--gate", str(gate))
    assert result.returncode != 0
    assert "wrong gate schema or verdict" in result.stderr


@pytest.mark.parametrize("mutation, message", [
    (lambda c: c.pop("resources"), "top-level config keys differ before file access"),
    (lambda c: c["resources"].update({"ram_bytes_max": "16GiB"}), "positive integer base units"),
    (lambda c: c["activation"].update({"allowed_config_mutations": ["anything"]}), "permits non-lifecycle mutation"),
    (lambda c: c["clock"].update({"saved_spike_times_frame": "unknown"}), "clock values or frame semantics"),
])
def test_missing_or_malformed_semantics_fail_before_bound_access(tmp_path, mutation, message):
    config = copy.deepcopy(json.loads(CONFIG.read_text()))
    mutation(config)
    config["source_bindings"] = [{"path": str(tmp_path / "must_not_open" / name), "sha256": "0" * 64}
        for name in ("en_truncation_qc_regeneration_candidate.py", "qc.py", "truncation.py", "refractory.py")]
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(config))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    result = subprocess.run([str(PYTHON), str(CLI), "--config", str(path), "--expected-config-sha256", digest,
                             "--validate-only"], text=True, capture_output=True)
    assert result.returncode != 0 and message in result.stderr
    assert not (tmp_path / "must_not_open").exists()


def test_global_token_reuse_refused_across_two_output_paths_and_preimport_limits(tmp_path):
    ledger = tmp_path / "global-ledger"
    first_output, second_output = tmp_path / "first", tmp_path / "second"
    first = invoke(first_output, "--synthetic", "--fixture-ledger-root", str(ledger))
    assert first.returncode == 0, first.stderr
    bootstrap = json.loads((first_output / "evidence/BOOTSTRAP.json").read_text())
    assert bootstrap["heavy_modules_present_before_limits"] == []
    assert bootstrap["limits_applied_before_runtime_imports"] is True
    assert bootstrap["enforced_limits"]["cpu_affinity"] == [0, 1, 2, 3]
    assert (first_output / "evidence/RUNTIME_ATTESTATION.json").is_file()
    second = invoke(second_output, "--synthetic", "--fixture-ledger-root", str(ledger))
    assert second.returncode != 0
    assert "FileExistsError" in second.stderr
    assert not second_output.exists()
    assert len(list(ledger.glob("*.consumed.json"))) == 1


@pytest.mark.parametrize("outcome, expected", [
    ("injected-failure", "injected synthetic worker failure"),
    ("corrupt-output", "Cannot load file containing pickled data"),
])
def test_failures_consume_global_token_and_preserve_evidence(tmp_path, outcome, expected):
    output, ledger = tmp_path / outcome, tmp_path / (outcome + "-ledger")
    result = invoke(output, "--synthetic", "--fixture-ledger-root", str(ledger), "--synthetic-outcome", outcome)
    assert result.returncode != 0
    assert expected in (output / "evidence/worker.stderr").read_text()
    assert (output / "evidence/FAILURE.json").is_file()
    assert (output / "evidence/AUTHORIZATION_CONSUMED.json").is_file()
    assert len(list(ledger.glob("*.consumed.json"))) == 1
    assert not (output / "COMPLETE.json").exists()


def test_real_destination_override_refused_before_gate_token_or_input(tmp_path):
    wrong = tmp_path / "wrong-destination"
    result = invoke(wrong, "--execute-real")
    assert result.returncode != 0
    assert "caller output override differs from frozen real destination" in result.stderr
    assert not wrong.exists()
