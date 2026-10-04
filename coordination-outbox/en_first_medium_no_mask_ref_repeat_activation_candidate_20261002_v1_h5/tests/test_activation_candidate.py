from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess


PACKET = Path(__file__).resolve().parents[1]
CONTRACT = PACKET / "contract/en_first_medium_pair.activation_candidate.execution_disabled.v1.json"
PREFLIGHT = PACKET / "source/activation_preflight.py"
RUNTIME = Path("/home/huklaban5/Documents/SpikeSortingTools/SpikeSortingTools/environments/rescue-production/.venv/bin/python")
PARENT = Path("/mnt/NPX/Luke/DARTsort_motion_experiments/en_first_medium_no_mask_ref_repeat_launcher_r2_finalization_repair_20261002_v3_h5")
H1 = Path("/mnt/NPX/Luke/DARTsort_motion_experiments/en_first_medium_no_mask_ref_repeat_launcher_r2_finalization_h1_review_20261002_v1_h1")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def preflight(mode: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(RUNTIME), str(PREFLIGHT), "--contract", str(CONTRACT),
         "--expected-contract-sha256", sha(CONTRACT), mode],
        text=True, capture_output=True, check=False,
    )


def test_exact_h1_go_and_parent_are_bound():
    contract = json.loads(CONTRACT.read_text())
    activation = contract["activation"]
    assert sha(PARENT / "MANIFEST.sha256") == "682b3c3cc6cbe9bf3ae9a2bd2a19313888ddacfae99e199e9bd884ff9ffda284"
    assert sha(H1 / "MANIFEST.sha256") == "7fde2fc6d93a85d3453471176ad66f6e3e8cb14349a449f38ea7e7e44d0fa6c5"
    assert activation["reviewed_launcher_packet"]["launcher_sha256"] == sha(PACKET / "source/en_first_medium_trained_pair_launch.py")
    assert contract["approval"]["h1_review_manifest_sha256"] == activation["h1_launcher_go_review"]["manifest_sha256"]


def test_disabled_inspection_is_fail_closed_and_reads_no_voltage_or_outcomes():
    result = preflight("--inspect-disabled")
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["exact_h1_launcher_go_bound"] is True
    assert report["all_namespaces_fresh"] is True
    assert report["phase3_review_bound"] is False
    assert report["activation_eligible"] is False
    assert report["recording_or_voltage_opened"] is False
    assert report["outcomes_opened"] is False


def test_require_activation_rejects_unresolved_phase3():
    result = preflight("--require-activation")
    assert result.returncode != 0
    assert "activation remains fail-closed" in result.stderr


def test_service_and_environment_are_proposed_only_with_fresh_namespaces():
    service = (PACKET / "service/PROPOSED_NOT_INSTALLED.service").read_text()
    assert "en_first_medium_trained_pair_20261002_v2" in service
    assert "ExecStartPre=" in service and "--require-activation" in service
    assert not (PACKET / "service/ENABLED_BINDING_NOT_PRESENT.env").exists()
    template = (PACKET / "service/PROPOSED_NOT_INSTALLED.env.template").read_text()
    assert "REQUIRED_PHASE3_REVIEW_MANIFEST_SHA256=UNRESOLVED" in template


def test_scientific_sources_are_byte_identical_to_reviewed_parent():
    excluded = {"activation_preflight.py", "en_first_medium_trained_pair_launch.py"}
    for child in (PACKET / "source").rglob("*.py"):
        if child.name in excluded:
            continue
        relative = child.relative_to(PACKET / "source")
        assert child.read_bytes() == (PARENT / "source" / relative).read_bytes()
    assert (PACKET / "source/en_first_medium_trained_pair_launch.py").read_bytes() == (PARENT / "source/en_first_medium_trained_pair_launch.py").read_bytes()


def test_resources_and_phase3_gate_remain_frozen():
    contract = json.loads(CONTRACT.read_text())
    resources = contract["resources"]
    assert resources["persistent_output_bytes_max"] == 17179869184
    assert resources["terminal_control_reserve_bytes"] == 1048576
    assert resources["live_output_bytes_max"] == 17178820608
    assert contract["activation"]["required_phase3_review"] == {
        "status": "UNRESOLVED_REQUIRED_BINDING", "path": None,
        "manifest_sha256": None, "complete_sha256": None,
    }
    assert contract["execution_enabled"] is False
