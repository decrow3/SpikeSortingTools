#!/usr/bin/env python3
"""Read-only independent audit of the H5 waveform freeze and pair launcher candidates."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile


SHARED = Path("/mnt/NPX/Luke/DARTsort_motion_experiments")
WAVE = SHARED / "en_first_medium_waveform_method_h5_actual_freeze_candidate_20261002_v4_h5"
VOLTAGE = SHARED / "en_first_medium_voltage_current_full_hash_20261002_v1_h5"
CLOCK = SHARED / "en_minimum_training_support_mask_trained_entry_clock_v5_h5_review_20261002_v1_h5"
LAUNCH = SHARED / "en_first_medium_no_mask_ref_repeat_launcher_candidate_20261002_v1_h5"
STATUS = SHARED / "current_coordination_20261002_v1/H5_STATUS_REF_REPEAT_LAUNCHER_V1.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path):
    return json.loads(path.read_text())


def verify_manifest(root: Path) -> int:
    count = 0
    for line in (root / "MANIFEST.sha256").read_text().splitlines():
        expected, name = line.split("  ", 1)
        member = root / name
        if not member.is_file() or sha(member) != expected:
            raise RuntimeError(f"manifest mismatch: {root.name}/{name}")
        count += 1
    return count


def module_from(path: Path):
    spec = importlib.util.spec_from_file_location("audited_pair_launcher", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load audited pair launcher")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    expected = {
        "status": "526eb3676c76b85aecfffc7d67521bd9237119678b17e5e46a37f6e956deb59a",
        "wave_manifest": "50fa0589d6eb459b12a4ac8c5c997fa7b822a58d2b2c1f0cbd2de53f9c47f8a7",
        "wave_complete": "74bd795cee9a3e5c651210c2ed37e977e2476e75491a7e2a5e44ff87f0deb952",
        "voltage_manifest": "141a699e7396b62f67329ebba6a62dda3387db148f9ec1e56f18bb1f9e6070c4",
        "voltage_complete": "aaa5e9f5d372327baa612de5159f37afb11f2e9ed20e71223f00ede55bfcdfd6",
        "clock_manifest": "39fd736b838b81db39b96265a4fd51de0ebf6e102e6fa1876fd56f4132c74af4",
        "clock_complete": "74f63fe6880e50c5247f21df2ef16d3ea37f2e3206f60dce89495a69d493cd3d",
        "launcher_manifest": "920c628d58af5a583d8a9ace35379ef6c10f83bacec07f8f138e018875a13fb9",
        "launcher_complete": "af725225c267a62ad36f866c3490911b7e1ab85b8d199e403ad3a658d6a6f9f6",
    }
    observed = {
        "status": sha(STATUS),
        "wave_manifest": sha(WAVE / "MANIFEST.sha256"),
        "wave_complete": sha(WAVE / "COMPLETE.json"),
        "voltage_manifest": sha(VOLTAGE / "MANIFEST.sha256"),
        "voltage_complete": sha(VOLTAGE / "COMPLETE.json"),
        "clock_manifest": sha(CLOCK / "MANIFEST.sha256"),
        "clock_complete": sha(CLOCK / "COMPLETE.json"),
        "launcher_manifest": sha(LAUNCH / "MANIFEST.sha256"),
        "launcher_complete": sha(LAUNCH / "COMPLETE.json"),
    }
    if observed != expected:
        raise RuntimeError("durable status or candidate identities differ")
    members = {root.name: verify_manifest(root) for root in (WAVE, VOLTAGE, CLOCK, LAUNCH)}

    wave_complete = load(WAVE / "COMPLETE.json")
    frozen = load(WAVE / "FROZEN_INPUT_BINDINGS.json")
    full_hash = load(VOLTAGE / "CURRENT_VOLTAGE_FULL_HASH_RECEIPT.json")
    clock_review = load(CLOCK / "INDEPENDENT_REVIEW.json")
    launcher_complete = load(LAUNCH / "COMPLETE.json")
    contract_path = LAUNCH / "contract/en_first_medium_pair.execution_disabled.v1.json"
    contract = load(contract_path)
    ref = load(LAUNCH / "config/REF384_repeat.execution_disabled.frozen.v2.json")
    repaired = load(LAUNCH / "config/repaired_B384.execution_disabled.clock_v5.json")
    service_text = (LAUNCH / "service/PROPOSED_NOT_INSTALLED.service").read_text()
    pair_path = LAUNCH / "source/en_first_medium_trained_pair_launch.py"
    pair_text = pair_path.read_text()

    waveform_freeze = {
        "execution_enabled": wave_complete["execution_enabled"],
        "voltage_access_enabled": wave_complete["voltage_access_enabled"],
        "prospective_inputs_unbound": frozen["prospective_arm_and_anchor_bindings_unbound"],
        "runtime_differences": frozen["runtime"]["differences"],
        "runtime_members": frozen["runtime"]["member_count"],
        "runtime_bytes": frozen["runtime"]["member_bytes"],
        "current_voltage_match": frozen["voltage_identity"]["historic_and_current_match"],
        "compact_preflight_voltage_bytes": frozen["voltage_identity"]["compact_preflight_voltage_bytes_read"],
    }
    if waveform_freeze != {
        "execution_enabled": False,
        "voltage_access_enabled": False,
        "prospective_inputs_unbound": True,
        "runtime_differences": [],
        "runtime_members": 89,
        "runtime_bytes": 3333088832,
        "current_voltage_match": True,
        "compact_preflight_voltage_bytes": 0,
    }:
        raise RuntimeError("waveform freeze invariant differs")
    if not (
        full_hash["status"] == "PASS_EXACT_MATCH"
        and full_hash["bytes_read"] == full_hash["expected_bytes"] == 241309358592
        and full_hash["actual_sha256"] == full_hash["expected_sha256"]
        and full_hash["stat_unchanged"] is True
        and full_hash["binary_is_symlink"] is False
        and full_hash["no_retry"] is True
    ):
        raise RuntimeError("full-voltage identity invariant differs")
    if clock_review["verdict"] != "GO_CLOCK_REPAIR_ONLY":
        raise RuntimeError("clock repair is not independently accepted")

    if not (
        launcher_complete["execution_enabled"] is False
        and launcher_complete["service_installed"] is False
        and launcher_complete["scientific_execution_started"] is False
        and contract["execution_enabled"] is False
        and contract["approval"]["h1_review_manifest_sha256"] is None
        and ref["execution_enabled"] is False
        and repaired["execution_enabled"] is False
    ):
        raise RuntimeError("launcher is not fail-closed disabled")
    if contract["arm_order"] != ["REF384_repeat", "repaired_B384"]:
        raise RuntimeError("arm order differs")
    if not (
        ref["effective_requirements"]["support_policy_enabled"] is False
        and repaired["trained"]["allow_saved_smoke_bank"] is False
        and ref["crop"] == repaired["crop"]
        and ref["output_clock"] == repaired["output_clock"]
        and contract["sources"]["historical_ref_runner"]["sha256"]
            != contract["sources"]["trained_terminal"]["sha256"]
    ):
        raise RuntimeError("distinct entry/intended-factor/clock invariant differs")
    required_flow = [
        'ARM_ORDER = ("REF384_repeat", "repaired_B384")',
        "ref_state = _execution_complete(ref_executor(contract), stage)",
        'stage = "repaired_B384"',
        "repaired_state = _execution_complete(repaired_executor(contract), stage)",
        '"scientific_outcomes_inspected_between_arms": False',
        'raise PermissionError("pair contract is execution-disabled',
    ]
    if not all(item in pair_text for item in required_flow):
        raise RuntimeError("launcher flow differs")

    # Independent no-data fixture for the pre-root retry defect.
    pair = module_from(pair_path)
    with tempfile.TemporaryDirectory(prefix="h1-pair-retry-audit-") as temporary:
        root = Path(temporary)
        fixture_contract = {
            "execution_enabled": True,
            "planned_paths": {
                "root": str(root / "pair"),
                "numba_cache": str(root / "cache"),
                "logs": str(root / "logs"),
            },
            "resources": {"minimum_free_bytes": 0},
        }
        os.environ[pair.PERSISTENT_MARKER] = "1"
        attempts: list[str] = []
        def fail_preflight(_):
            attempts.append("preflight")
            raise RuntimeError("injected pre-root resource failure")
        for _ in range(2):
            try:
                pair.execute_pair(
                    fixture_contract,
                    ref_executor=lambda _: attempts.append("REF"),
                    repaired_executor=lambda _: attempts.append("B"),
                    resource_probe=fail_preflight,
                )
            except RuntimeError as error:
                if str(error) != "injected pre-root resource failure":
                    raise
            else:
                raise RuntimeError("injected preflight failure was not raised")
        retry_fixture = {
            "attempts": attempts,
            "root_exists": (root / "pair").exists(),
            "second_call_reached_preflight": attempts == ["preflight", "preflight"],
        }

    service_enforcement = {
        "restart_no": "Restart=no" in service_text,
        "start_limit_burst_present": "StartLimitBurst" in service_text,
        "persistent_single_use_claim_present": "claim" in service_text.lower() or "token" in service_text.lower(),
    }
    resource_enforcement = {
        "declared_output_cap_bytes": contract["resources"]["persistent_output_bytes_max"],
        "output_cap_referenced_after_contract_validation": pair_text.count("persistent_output_bytes_max") > 1,
        "systemd_disk_quota_present": any(
            key in service_text for key in ("LimitFSIZE=", "StateDirectoryQuota=", "RuntimeDirectoryQuota=")
        ),
    }
    voltage_launch_continuity = {
        "receipt_identity_declared_in_contract": "current_voltage_packet_manifest_sha256" in contract["gates"],
        "pair_source_invokes_lstat": "lstat" in pair_text,
        "pair_source_verifies_current_voltage_packet": "current_voltage_packet_manifest_sha256" in pair_text,
    }
    blockers = {
        "durable_one_start_missing": retry_fixture["second_call_reached_preflight"]
            and not service_enforcement["persistent_single_use_claim_present"],
        "pre_root_failure_not_preserved": retry_fixture["second_call_reached_preflight"]
            and not retry_fixture["root_exists"],
        "persistent_output_cap_unenforced": not resource_enforcement["output_cap_referenced_after_contract_validation"]
            and not resource_enforcement["systemd_disk_quota_present"],
        "current_voltage_continuity_not_checked_at_pair_launch":
            voltage_launch_continuity["receipt_identity_declared_in_contract"]
            and not voltage_launch_continuity["pair_source_invokes_lstat"]
            and not voltage_launch_continuity["pair_source_verifies_current_voltage_packet"],
    }
    if not all(blockers.values()):
        raise RuntimeError(f"expected load-bearing blocker fixture differs: {blockers}")

    print(json.dumps({
        "schema": "waveform-launcher-h1-independent-audit-v1",
        "overall_verdict": "TARGETED_LAUNCHER_REPAIR_REQUIRED",
        "identities": observed,
        "manifest_member_counts": members,
        "waveform_freeze": {**waveform_freeze, "verdict": "GO_METHOD_FREEZE_ONLY"},
        "full_voltage_receipt": {
            "verdict": "PASS_EXACT_MATCH", "bytes": full_hash["bytes_read"],
            "sha256": full_hash["actual_sha256"], "retry": False,
        },
        "clock_v5": {"verdict": clock_review["verdict"]},
        "launcher_accepted_structure": {
            "distinct_entry_paths": True,
            "intended_factor_difference": True,
            "ref_first_b_second": True,
            "between_arm_completion_state_only": True,
            "shared_clock_and_row_ancestry_validator": True,
            "fresh_namespace_checks": True,
            "disabled_fail_closed": True,
        },
        "retry_fixture": retry_fixture,
        "service_enforcement": service_enforcement,
        "resource_enforcement": resource_enforcement,
        "voltage_launch_continuity": voltage_launch_continuity,
        "blockers": blockers,
        "recording_or_voltage_opened_by_audit": False,
        "scientific_execution_started": False,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
