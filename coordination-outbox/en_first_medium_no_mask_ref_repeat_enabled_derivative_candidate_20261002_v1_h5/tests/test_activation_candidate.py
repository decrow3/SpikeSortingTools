from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import subprocess


PACKET = Path(__file__).resolve().parents[1]
CONTRACT = PACKET / "contract/en_first_medium_pair.enabled_candidate.v1.json"
PREFLIGHT = PACKET / "source/activation_preflight.py"
RUNTIME = Path("/home/huklaban5/Documents/SpikeSortingTools/SpikeSortingTools/environments/rescue-production/.venv/bin/python")
PARENT = Path("/mnt/NPX/Luke/DARTsort_motion_experiments/en_first_medium_no_mask_ref_repeat_launcher_r2_finalization_repair_20261002_v3_h5")
H1 = Path("/mnt/NPX/Luke/DARTsort_motion_experiments/en_first_medium_no_mask_ref_repeat_launcher_r2_finalization_h1_review_20261002_v1_h1")
PHASE3 = Path("/mnt/NPX/Luke/DARTsort_motion_experiments/en_first_medium_saved_output_phase3_provenance_repair_candidate_20261002_v3_h1")
PHASE3_REVIEW = Path("/mnt/NPX/Luke/DARTsort_motion_experiments/en_first_medium_saved_output_phase3_provenance_repair_h5_independent_review_20261002_v1_h5")
PHASE3_STATUS = Path("/mnt/NPX/Luke/DARTsort_motion_experiments/current_coordination_20261002_v1/H5_STATUS_PHASE3_V3_PROVENANCE_REPAIR_REVIEW_RESULT_V1.json")
DISABLED_BASE = Path("/mnt/NPX/Luke/DARTsort_motion_experiments/en_first_medium_no_mask_ref_repeat_activation_phase3_binding_candidate_20261002_v2_h5")
ACTIVATION_REVIEW = Path("/mnt/NPX/Luke/DARTsort_motion_experiments/en_first_medium_no_mask_ref_repeat_activation_phase3_binding_h1_review_20261002_v1_h1")
ACTIVATION_STATUS = Path("/mnt/NPX/Luke/DARTsort_motion_experiments/current_coordination_20261002_v1/H1_STATUS_ACTIVATION_PHASE3_BINDING_REVIEW_RESULT_V1.json")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def preflight(mode: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(RUNTIME), str(PREFLIGHT), "--contract", str(CONTRACT),
         "--expected-contract-sha256", sha(CONTRACT), mode],
        text=True, capture_output=True, check=False,
    )


def test_exact_launcher_parent_phase3_candidate_and_go_are_bound():
    contract = json.loads(CONTRACT.read_text())
    activation = contract["activation"]
    assert sha(PARENT / "MANIFEST.sha256") == "682b3c3cc6cbe9bf3ae9a2bd2a19313888ddacfae99e199e9bd884ff9ffda284"
    assert sha(H1 / "MANIFEST.sha256") == "7fde2fc6d93a85d3453471176ad66f6e3e8cb14349a449f38ea7e7e44d0fa6c5"
    assert activation["reviewed_launcher_packet"]["launcher_sha256"] == "238c9c997bad67415a13671deaa5569ee121f3a27aa07188fef51c424f2029f8"
    assert sha(PHASE3 / "MANIFEST.sha256") == "3199ddda56d1b03c6a3d77d87443fbf1c2956d4ab09c08ede52467ca4b09b1cd"
    assert sha(PHASE3_REVIEW / "MANIFEST.sha256") == "c0d3f09341dd8653551eaae205c80c35ba942467d3703ef74a7d643cf403ec33"
    assert sha(PHASE3_STATUS) == "7d439bb156c786e53300def540c7ecd543d0e81b948d3d92d531df765002ac74"
    assert sha(PACKET / "source/en_first_medium_trained_pair_launch.py") == "9e7290c182211ad5a71c952e9f178af764b6fe66ffe4d328cf3b2cbc694f84bc"
    assert contract["approval"]["h1_review_manifest_sha256"] == activation["h1_launcher_go_review"]["manifest_sha256"]
    assert contract["approval"]["phase3_review_manifest_sha256"] == "c0d3f09341dd8653551eaae205c80c35ba942467d3703ef74a7d643cf403ec33"
    assert sha(DISABLED_BASE / "MANIFEST.sha256") == "5b3b780e6ebc7a7d74f20bbea9505719bb8cccc4498249693672b6e654dd25c4"
    assert sha(ACTIVATION_REVIEW / "MANIFEST.sha256") == "d815d0eae6f36aebb3c7ad8e41a635127bce43e535bbd3612e54183475ff5709"
    assert sha(ACTIVATION_STATUS) == "96e4c4cad42579a59bc9eec369af585a3474d27f6962a810424ed7fbb79f88ed"
    assert contract["approval"]["h1_review_manifest_sha256"] != contract["approval"]["activation_review_manifest_sha256"]


def test_enabled_inspection_is_eligible_and_reads_no_voltage_or_outcomes():
    result = preflight("--inspect-disabled")
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["exact_h1_launcher_go_bound"] is True
    assert report["all_namespaces_fresh"] is True
    assert report["phase3_review_bound"] is True
    assert report["exact_phase3_candidate_bound"] is True
    assert report["exact_phase3_independent_go_bound"] is True
    assert report["exact_h1_activation_review_bound"] is True
    assert report["exact_hubplanner_decision_bound"] is True
    assert report["execution_enabled"] is True
    assert report["activation_eligible"] is True
    assert report["recording_or_voltage_opened"] is False
    assert report["outcomes_opened"] is False


def test_require_activation_accepts_exact_enabled_derivative_without_launching():
    result = preflight("--require-activation")
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["activation_eligible"] is True


def test_service_and_environment_are_enabled_but_not_installed_with_fresh_namespaces():
    service = (PACKET / "service/ENABLED_NOT_INSTALLED.service").read_text()
    assert "en_first_medium_trained_pair_20261002_v2" in service
    assert "ExecStartPre=" in service and "--require-activation" in service
    environment = (PACKET / "service/ENABLED_BINDING.env").read_text()
    assert "REQUIRED_PHASE3_REVIEW_MANIFEST_SHA256=c0d3f09341dd8653551eaae205c80c35ba942467d3703ef74a7d643cf403ec33" in environment
    assert "REQUIRED_PHASE3_MACHINE_STATUS_TOKEN=GO_PROVENANCE_REPAIR_DELTA" in environment
    assert "REQUIRED_H1_ACTIVATION_REVIEW_MANIFEST_SHA256=d815d0eae6f36aebb3c7ad8e41a635127bce43e535bbd3612e54183475ff5709" in environment


def test_scientific_sources_are_byte_identical_to_reviewed_parent():
    excluded = {"activation_preflight.py", "en_first_medium_trained_pair_launch.py"}
    for child in (PACKET / "source").rglob("*.py"):
        if child.name in excluded:
            continue
        relative = child.relative_to(PACKET / "source")
        assert child.read_bytes() == (PARENT / "source" / relative).read_bytes()


def test_launcher_delta_preserves_finalizer_and_one_start_implementation():
    spec = importlib.util.spec_from_file_location(
        "base_pair_launcher", PARENT / "source/en_first_medium_trained_pair_launch.py"
    )
    assert spec is not None and spec.loader is not None
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    spec = importlib.util.spec_from_file_location(
        "candidate_pair_launcher", PACKET / "source/en_first_medium_trained_pair_launch.py"
    )
    assert spec is not None and spec.loader is not None
    candidate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(candidate)
    for name in (
        "consume_start_claim", "terminal_reserve_derivation", "_final_complete_payload",
        "finalize_success", "run_single_attempt",
    ):
        assert inspect.getsource(getattr(candidate, name)) == inspect.getsource(getattr(base, name))


def test_resources_and_phase3_gate_remain_frozen():
    contract = json.loads(CONTRACT.read_text())
    resources = contract["resources"]
    assert resources["persistent_output_bytes_max"] == 17179869184
    assert resources["terminal_control_reserve_bytes"] == 1048576
    assert resources["live_output_bytes_max"] == 17178820608
    phase3 = contract["activation"]["required_phase3_review"]
    assert phase3["status"] == "BOUND_GO_EXECUTION_STILL_DISABLED"
    assert phase3["machine_status_token"] == "GO_PROVENANCE_REPAIR_DELTA"
    assert phase3["candidate"]["manifest_sha256"] == "3199ddda56d1b03c6a3d77d87443fbf1c2956d4ab09c08ede52467ca4b09b1cd"
    assert phase3["independent_go_review"]["manifest_sha256"] == "c0d3f09341dd8653551eaae205c80c35ba942467d3703ef74a7d643cf403ec33"
    assert contract["execution_enabled"] is True
    assert contract["resources"]["gpu_count_max"] == 1
    assert contract["resources"]["cpu_threads_max"] == 16
    assert contract["resources"]["ram_bytes_max"] == 68719476736
    assert contract["resources"]["wall_seconds_max"] == 2700
    assert contract["resources"]["minimum_free_bytes"] == 214748364800
    assert contract["resources"]["service_restart"] == "no"


def test_all_proposed_output_paths_share_fresh_v2_namespace():
    contract = json.loads(CONTRACT.read_text())
    ref = json.loads((PACKET / "config/REF384_repeat.enabled_candidate.frozen.v2.json").read_text())
    repaired = json.loads((PACKET / "config/repaired_B384.enabled_candidate.clock_v5.json").read_text())
    prefix = "/media/huklaban5/Data/en_first_medium_trained_pair_20261002_v2"
    paths = [
        contract["planned_paths"]["root"], contract["planned_paths"]["numba_cache"],
        contract["planned_paths"]["attempt_evidence"], ref["run_root"],
        ref["launch_evidence_dir"], ref["native_results_dir"],
        ref["pre_extraction_snapshot_dir"], repaired["trained"]["run_root"],
        repaired["trained"]["launch_evidence_dir"],
        repaired["trained"]["pre_extraction_snapshot_dir"],
        repaired["native_invocation"]["results_dir"],
    ]
    assert all(path.startswith(prefix) for path in paths)
    assert all(not Path(path).exists() for path in contract["planned_paths"].values()
               if path != contract["planned_paths"]["staged_source_root"])


def test_exact_delta_does_not_change_scientific_config_values():
    disabled = json.loads((DISABLED_BASE / "contract/en_first_medium_pair.activation_candidate.execution_disabled.v1.json").read_text())
    enabled = json.loads(CONTRACT.read_text())
    assert enabled["arm_order"] == disabled["arm_order"] == ["REF384_repeat", "repaired_B384"]
    assert enabled["resources"] == disabled["resources"]
    assert enabled["voltage_identity"] == disabled["voltage_identity"]
    assert enabled["between_arm_policy"] == disabled["between_arm_policy"]
    assert enabled["checkpoint_policy"] == disabled["checkpoint_policy"]
    for old_name, new_name in (
        ("REF384_repeat.execution_disabled.frozen.v2.json", "REF384_repeat.enabled_candidate.frozen.v2.json"),
        ("repaired_B384.execution_disabled.clock_v5.json", "repaired_B384.enabled_candidate.clock_v5.json"),
    ):
        old = json.loads((DISABLED_BASE / "config" / old_name).read_text())
        new = json.loads((PACKET / "config" / new_name).read_text())
        for payload in (old, new):
            payload.pop("execution_enabled")
            payload.pop("status")
            payload.pop("failure_reason", None)
        assert new == old
