#!/usr/bin/env python3
"""Fail-closed serial launcher for one no-mask REF repeat and repaired-B arm.

The orchestration layer observes execution state only.  It never reads spike
counts, scores, evaluator fields, or other scientific outcomes between arms.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Callable, Mapping


SCHEMA = "en-first-medium-heterogeneous-pair-launch-contract-v1"
ARM_ORDER = ("REF384_repeat", "repaired_B384")
PERSISTENT_MARKER = "EN_FIRST_MEDIUM_PAIR_PERSISTENT_JOB"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: str | Path, payload: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _exact_file(binding: Mapping[str, Any], label: str, base: Path | None = None) -> Path:
    if set(binding) != {"path", "sha256"}:
        raise ValueError(f"{label} binding structure differs")
    path = Path(binding["path"])
    if not path.is_absolute() and base is not None:
        path = base / path
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} must be a non-symlink regular file")
    if sha256_file(path) != binding["sha256"]:
        raise ValueError(f"{label} hash differs")
    return path


def validate_contract(path: str | Path, expected_sha256: str) -> dict[str, Any]:
    """Validate compact contract/source/config bindings without opening voltage."""
    path = Path(path)
    if sha256_file(path) != expected_sha256:
        raise ValueError("pair contract hash differs")
    contract = json.loads(path.read_text())
    contract["_contract_dir"] = str(path.resolve().parent)
    base = path.resolve().parent
    for key in (
        "schema", "status", "execution_enabled", "approval", "arm_order",
        "configs", "sources", "planned_paths", "resources", "stop_conditions",
        "checkpoint_policy", "completion_condition", "gates",
    ):
        if key not in contract:
            raise ValueError(f"missing contract key before execution: {key}")
    if contract["schema"] != SCHEMA:
        raise ValueError("wrong pair contract schema")
    if type(contract["execution_enabled"]) is not bool:
        raise ValueError("execution_enabled must be an exact JSON boolean")
    if tuple(contract["arm_order"]) != ARM_ORDER:
        raise ValueError("arm order differs from frozen REF-then-repaired sequence")
    if set(contract["configs"]) != {"REF384_repeat", "repaired_B384", "historical_screen"}:
        raise ValueError("config binding set differs")
    for name, binding in contract["configs"].items():
        _exact_file(binding, f"config {name}", base)
    required_roles = {
        "pair_launcher", "historical_ref_runner", "trained_terminal",
        "support_candidate", "prewhitening", "kilosort_io",
        "kilosort_preprocessing", "kilosort_spikedetect",
        "kilosort_template_matching", "kilosort_run",
    }
    if set(contract["sources"]) != required_roles:
        raise ValueError("source role set differs")
    for role, binding in contract["sources"].items():
        _exact_file(binding, f"source {role}", base)
    paths = contract["planned_paths"]
    if set(paths) != {"root", "numba_cache", "logs", "staged_source_root"}:
        raise ValueError("planned path set differs")
    if Path(paths["root"]) == Path(paths["numba_cache"]) or Path(paths["root"]) == Path(paths["logs"]):
        raise ValueError("pair namespaces overlap")
    resources = contract["resources"]
    expected_resources = {
        "wall_seconds_max": 2700,
        "ram_bytes_max": 68719476736,
        "cpu_threads_max": 16,
        "gpu_count_max": 1,
        "persistent_output_bytes_max": 17179869184,
        "minimum_free_bytes": 214748364800,
        "service_restart": "no",
        "service_start_limit": 1,
    }
    if resources != expected_resources:
        raise ValueError("resource contract differs")
    if contract["gates"].get("clock_review_verdict") != "GO_CLOCK_REPAIR_ONLY":
        raise ValueError("clock GO is not bound")
    if contract["gates"].get("rf_holdout_sealed") is not True:
        raise ValueError("RF/holdout seal differs")
    if contract["execution_enabled"]:
        approval = contract["approval"].get("h1_review_manifest_sha256")
        if not isinstance(approval, str) or len(approval) != 64:
            raise PermissionError("enabled pair execution requires exact H1 review binding")
    return contract


def _bound_path(contract: Mapping[str, Any], binding: Mapping[str, Any]) -> Path:
    path = Path(binding["path"])
    if path.is_absolute():
        return path
    return Path(contract["_contract_dir"]) / path


def _load_file_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load reviewed module {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _trained_module(contract: Mapping[str, Any]):
    source_root = Path(contract["planned_paths"]["staged_source_root"])
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    importlib.invalidate_caches()
    return importlib.import_module("npx_preprocessing.motion.kilosort_support_mask_trained")


def _augment_ref_snapshot(trained, snapshot: Path) -> None:
    receipt_path = snapshot / "RECEIPT.json"
    receipt = json.loads(receipt_path.read_text())
    files = {
        name: sha256_file(snapshot / name)
        for name in trained.REQUIRED_SNAPSHOT_FILES if name.endswith(".npy")
    }
    receipt.update(
        wall3_axis0_binding="spike_detection_templates.npy",
        files=files,
        augmented_by_pair_wrapper=True,
        scientific_arrays_changed=False,
    )
    atomic_json(receipt_path, receipt)


def execute_ref_arm(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Run the exact historical no-mask path, then apply the reviewed terminal validator."""
    runner_path = _bound_path(contract, contract["sources"]["historical_ref_runner"])
    screen = _load_file_module("reviewed_en_common_support_crop_screen", runner_path)
    trained = _trained_module(contract)
    screen_path = _bound_path(contract, contract["configs"]["historical_screen"])
    screen_config = json.loads(screen_path.read_text())
    plan = screen.validate_plan(screen_path, screen_config)
    ref_path = _bound_path(contract, contract["configs"]["REF384_repeat"])
    ref = json.loads(ref_path.read_text())
    if ref["execution_enabled"] is not True:
        raise PermissionError("REF repeat config is not enabled by reviewed derivative")
    if ref["effective_requirements"]["support_policy_enabled"] is not False:
        raise ValueError("REF repeat unexpectedly enables support policy")
    if ref["input"] != screen_config["arms"]["REF384"]:
        raise ValueError("REF repeat input differs from frozen historical REF384")
    root = Path(contract["planned_paths"]["root"])
    receipt = screen.run_arm("REF384_repeat", ref["input"], screen_config, root)
    run_root = root / "REF384_repeat"
    snapshot = run_root / "pre_extraction_snapshot"
    results = run_root / "sorter_output"
    _augment_ref_snapshot(trained, snapshot)
    semantic = trained.validate_saved_artifacts(results, snapshot, ref)
    inventory = {
        "native": {name: sha256_file(results / name) for name in trained.REQUIRED_NATIVE_FILES},
        "snapshot": {name: sha256_file(snapshot / name) for name in trained.REQUIRED_SNAPSHOT_FILES},
    }
    atomic_json(run_root / "launch_evidence/terminal_validation.json", semantic)
    atomic_json(run_root / "launch_evidence/artifact_inventory.json", inventory)
    complete = {
        "status": "complete", "arm": "REF384_repeat", "no_support_policy": True,
        "historical_plan_config_sha256": plan["config_sha256"],
        "terminal_clock_validated": True, "artifact_inventory_validated": True,
        "completion_written_last": True,
    }
    atomic_json(run_root / "COMPLETE.json", complete)
    return {"status": "complete", "complete_path": str(run_root / "COMPLETE.json")}


def execute_repaired_arm(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Run repaired B through the reviewed support-aware trained entry."""
    trained = _trained_module(contract)
    config_binding = contract["configs"]["repaired_B384"]
    config_path = _bound_path(contract, config_binding)
    config = json.loads(config_path.read_text())
    if config["execution_enabled"] is not True:
        raise PermissionError("repaired-B config is not enabled by reviewed derivative")
    trained.run_native_trained(
        config_path=config_path, expected_config_sha256=config_binding["sha256"]
    )
    complete = Path(config["trained"]["run_root"]) / "COMPLETE.json"
    if not complete.is_file():
        raise RuntimeError("repaired-B trained entry returned without COMPLETE")
    return {"status": "complete", "complete_path": str(complete)}


def _execution_complete(receipt: Mapping[str, Any], arm: str) -> dict[str, str]:
    """Read only execution state, never scientific result fields."""
    if receipt.get("status") != "complete":
        raise RuntimeError(f"{arm} did not report complete execution state")
    complete_path = Path(receipt.get("complete_path", ""))
    if not complete_path.is_file():
        raise RuntimeError(f"{arm} completion file is absent")
    complete = json.loads(complete_path.read_text())
    if complete.get("status") != "complete":
        raise RuntimeError(f"{arm} completion file lacks complete execution state")
    return {"status": "complete", "complete_path": str(complete_path),
            "complete_sha256": sha256_file(complete_path)}


def resource_preflight(contract: Mapping[str, Any]) -> None:
    planned = contract["planned_paths"]
    root = Path(planned["root"])
    if shutil.disk_usage(root.parent).free < contract["resources"]["minimum_free_bytes"]:
        raise RuntimeError("free storage is below the frozen minimum")
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one visible CUDA GPU is required")


def execute_pair(
    contract: Mapping[str, Any], *,
    ref_executor: Callable[[Mapping[str, Any]], Mapping[str, Any]] = execute_ref_arm,
    repaired_executor: Callable[[Mapping[str, Any]], Mapping[str, Any]] = execute_repaired_arm,
    resource_probe: Callable[[Mapping[str, Any]], None] = resource_preflight,
) -> dict[str, Any]:
    if contract["execution_enabled"] is not True:
        raise PermissionError("pair contract is execution-disabled pending independent H1 review")
    if os.environ.get(PERSISTENT_MARKER) != "1":
        raise PermissionError("pair execution requires the reviewed persistent service marker")
    planned = contract["planned_paths"]
    for label in ("root", "numba_cache", "logs"):
        if Path(planned[label]).exists():
            raise FileExistsError(f"planned {label} namespace already exists")
    root = Path(planned["root"])
    resource_probe(contract)
    root.mkdir(parents=False, exist_ok=False)
    evidence = root / "launch_evidence"
    evidence.mkdir()
    atomic_json(evidence / "STARTED.json", {
        "status": "running", "arm_order": list(ARM_ORDER),
        "started_at": datetime.now(timezone.utc).isoformat(), "retry": False,
    })
    stage = "REF384_repeat"
    try:
        ref_state = _execution_complete(ref_executor(contract), stage)
        atomic_json(evidence / "REF384_repeat_EXECUTION_STATE.json", ref_state)
        stage = "repaired_B384"
        repaired_state = _execution_complete(repaired_executor(contract), stage)
        atomic_json(evidence / "repaired_B384_EXECUTION_STATE.json", repaired_state)
        complete = {
            "status": "complete", "arm_order": list(ARM_ORDER),
            "execution_states": {"REF384_repeat": ref_state, "repaired_B384": repaired_state},
            "scientific_outcomes_inspected_between_arms": False,
            "completion_written_last": True,
        }
        atomic_json(root / "COMPLETE.json", complete)
        return complete
    except Exception as exc:
        atomic_json(evidence / "RUN_FAILURE.json", {
            "status": "failed", "stage": stage, "exception_type": type(exc).__name__,
            "exception_message": str(exc), "completion_written": False,
        })
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--expected-contract-sha256", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--execute-reviewed-once", action="store_true")
    args = parser.parse_args()
    contract = validate_contract(args.contract, args.expected_contract_sha256)
    report = {
        "status": "validated", "execution_enabled": contract["execution_enabled"],
        "arm_order": list(ARM_ORDER), "recording_or_voltage_opened": False,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.validate_only:
        return 0
    execute_pair(contract)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
