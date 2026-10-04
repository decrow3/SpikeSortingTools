import ast
import hashlib
import json
from pathlib import Path
import re


PACKET = Path(__file__).resolve().parents[1]
CONTRACT = PACKET / "configs" / "en_first_medium_saved_output_qualification.execution_disabled.v1.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def functions(path, names):
    tree = ast.parse(path.read_text())
    return {
        node.name: ast.dump(node, include_attributes=False)
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names
    }


def test_exact_active_source_and_dependency_boundary():
    contract = json.loads(CONTRACT.read_text())
    expected = contract["implementation_boundary"]["active_module_sha256"]
    paths = {
        "testing.first_medium_saved_output_qualification": PACKET / "source/testing/first_medium_saved_output_qualification.py",
        "testing.first_medium_measurement_redesign": PACKET / "source/testing/first_medium_measurement_redesign.py",
        "testing.first_medium_evaluator": PACKET / "source/testing/first_medium_evaluator.py",
        "testing.sort_comparison": PACKET / "source/testing/sort_comparison.py",
        "testing.luke_amplitude_dropout_audit": PACKET / "source/testing/luke_amplitude_dropout_audit.py",
        "pipeline.config": PACKET / "source/pipeline/config.py",
        "pipeline.truncation": PACKET / "source/pipeline/truncation.py",
    }
    assert {name: sha(path) for name, path in paths.items()} == expected
    dependency = contract["implementation_boundary"]["kilosort_source_dependency"]
    root = PACKET / "dependencies/kilosort" / dependency["aggregate_sha256"]
    observed = {name: sha(root / name) for name in sorted(dependency["files"])}
    assert observed == dependency["files"]
    aggregate = hashlib.sha256(
        json.dumps(observed, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert aggregate == dependency["aggregate_sha256"]


def test_active_matcher_preserves_reviewed_correspondence_ast():
    names = {"coincidence_counts", "exclusive_count", "correspondence"}
    reviewed = functions(PACKET / "snapshots/measurement_sort_comparison.py", names)
    active = functions(PACKET / "source/testing/sort_comparison.py", names)
    assert reviewed == active
    assert sha(PACKET / "snapshots/measurement_sort_comparison.py") == (
        "b4065ba7e6c326e74f5f283de9a6ee76df0dfd3b70b65fd534b1d4559a80de0c"
    )


def test_two_phase_contract_is_disabled_bounded_and_has_no_repaired_hash_binding():
    contract = json.loads(CONTRACT.read_text())
    assert contract["execution_enabled"] is False
    phase1 = contract["phases"]["phase1_controls"]
    phase2 = contract["phases"]["phase2_existing_b"]
    assert phase1["real_execution_enabled"] is False
    assert phase2["real_execution_enabled"] is False
    assert phase1["required_real_arms"] == ["REF384"]
    assert phase2["required_real_arms"] == ["REF384", "existing_corrected_B384"]
    assert "method_freeze_token" in phase2["required_tokens"]
    assert "phase1_manifest_sha256" in phase2["required_tokens"]
    assert contract["lifecycle"]["one_start_per_phase"] is True
    assert contract["lifecycle"]["automatic_retry"] is False
    for phase in (phase1, phase2):
        resources = phase["resources"]
        assert resources["wall_seconds_max"] > 0
        assert resources["peak_rss_bytes_max"] > 0
        assert resources["cpu_threads_max"] == 4
        assert resources["persistent_output_bytes_max"] > 0
    serialized = json.dumps(contract, sort_keys=True)
    for arm in ("repaired_B384", "REF384_repeat"):
        assert arm in serialized
    arm_hash_bindings = re.findall(
        r'(?i)(?:repaired_B384|REF384_repeat)[^\n]{0,120}\b[0-9a-f]{64}\b', serialized
    )
    assert arm_hash_bindings == []


def test_cli_has_no_delta_or_hidden_execution_path():
    path = PACKET / "source/testing/first_medium_saved_output_qualification.py"
    tree = ast.parse(path.read_text())
    calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                calls.append(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                calls.append(node.func.attr)
    assert "retention_difference" not in calls
    text = path.read_text()
    for forbidden in ("repaired_B384", "REF384_repeat", "waveform_evidence_npz"):
        assert forbidden not in text
    assert '"DeltaR": {"status": "UNAVAILABLE"' in text
