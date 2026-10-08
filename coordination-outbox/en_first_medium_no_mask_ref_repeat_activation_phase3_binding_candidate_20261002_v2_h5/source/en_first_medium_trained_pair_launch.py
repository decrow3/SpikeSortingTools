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
import signal
import shutil
import subprocess
import sys
import time
from typing import Any, Callable, Mapping


SCHEMA = "en-first-medium-heterogeneous-pair-launch-contract-v1"
ARM_ORDER = ("REF384_repeat", "repaired_B384")
PERSISTENT_MARKER = "EN_FIRST_MEDIUM_PAIR_PERSISTENT_JOB"
CONTROL_RECEIPT_BYTES_MAX = 131072
FAILURE_TEXT_CHARS_MAX = 4096


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


def serialized_json(payload: object) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def durable_json(path: str | Path, payload: object) -> None:
    """Atomically replace a receipt, fsync the file, then fsync its directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    data = serialized_json(payload)
    if len(data) > CONTROL_RECEIPT_BYTES_MAX:
        raise ValueError(
            f"control receipt exceeds serialized bound: {len(data)} > "
            f"{CONTROL_RECEIPT_BYTES_MAX}"
        )
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        os.write(descriptor, data)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def consume_start_claim(
    attempt_dir: str | Path, *, contract_path: str | Path,
    expected_contract_sha256: str, expected_service_sha256: str,
) -> dict[str, Any]:
    """Consume the one permitted attempt before any contract/path/resource preflight."""
    attempt_dir = Path(attempt_dir)
    attempt_dir.mkdir(parents=True, exist_ok=True)
    claim_path = attempt_dir / "START_CLAIM.json"
    payload = {
        "schema": "en-first-medium-pair-single-start-claim-v1",
        "status": "START_CONSUMED",
        "contract_path": str(Path(contract_path).resolve()),
        "expected_contract_sha256": expected_contract_sha256,
        "expected_service_sha256": expected_service_sha256,
        "attempt_evidence_dir": str(attempt_dir.resolve()),
        "pid": os.getpid(),
        "invocation_id": os.environ.get("INVOCATION_ID"),
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    data = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    try:
        descriptor = os.open(claim_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except FileExistsError as exc:
        raise PermissionError("single start already consumed; second attempt rejected before preflight") from exc
    try:
        os.write(descriptor, data)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(attempt_dir)
    durable_json(attempt_dir / "STARTED.json", payload)
    return payload


def durable_attempt_failure(attempt_dir: str | Path, stage: str, exc: BaseException) -> None:
    durable_json(Path(attempt_dir) / "FAILURE.json", {
        "schema": "en-first-medium-pair-terminal-failure-v1",
        "status": "TERMINAL_FAILURE", "stage": str(stage)[:256],
        "exception_type": type(exc).__name__[:256],
        "exception_message": str(exc)[:FAILURE_TEXT_CHARS_MAX],
        "second_attempt_permitted": False,
        "completion_written": False,
        "failed_at": datetime.now(timezone.utc).isoformat(),
    })


def record_attempt_stage(attempt_dir: str | Path, stage: str) -> None:
    durable_json(Path(attempt_dir) / "CURRENT_STAGE.json", {
        "schema": "en-first-medium-pair-attempt-stage-v1", "stage": stage,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    })


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
    contract["_contract_path"] = str(path.resolve())
    contract["_expected_contract_sha256"] = expected_sha256
    base = path.resolve().parent
    for key in (
        "schema", "status", "execution_enabled", "approval", "arm_order",
        "configs", "sources", "planned_paths", "resources", "stop_conditions",
        "checkpoint_policy", "completion_condition", "gates", "service",
        "voltage_identity",
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
        "kilosort_template_matching", "kilosort_run", "voltage_preflight",
    }
    if set(contract["sources"]) != required_roles:
        raise ValueError("source role set differs")
    for role, binding in contract["sources"].items():
        _exact_file(binding, f"source {role}", base)
    paths = contract["planned_paths"]
    if set(paths) != {"root", "numba_cache", "logs", "staged_source_root", "attempt_evidence"}:
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
        "terminal_control_reserve_bytes": 1048576,
        "live_output_bytes_max": 17178820608,
        "counted_late_log_bytes_max": 262144,
        "minimum_free_bytes": 214748364800,
        "service_restart": "no",
        "service_start_limit": 1,
        "output_monitor_interval_seconds": 0.1,
    }
    if resources != expected_resources:
        raise ValueError("resource contract differs")
    if resources["live_output_bytes_max"] + resources["terminal_control_reserve_bytes"] != resources["persistent_output_bytes_max"]:
        raise ValueError("live threshold plus terminal reserve differs from total cap")
    if contract["gates"].get("clock_review_verdict") != "GO_CLOCK_REPAIR_ONLY":
        raise ValueError("clock GO is not bound")
    if contract["gates"].get("rf_holdout_sealed") is not True:
        raise ValueError("RF/holdout seal differs")
    service = contract["service"]
    _exact_file(service["unit_binding"], "service unit", base)
    if service.get("condition_on_start_claim_absence") is not True:
        raise ValueError("service does not condition on start-claim absence")
    if contract["gates"].get("waveform_method_h1_verdict") != "GO_METHOD_FREEZE_ONLY":
        raise ValueError("accepted waveform-method review is not bound")
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


def verify_voltage_continuity(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Run the reviewed compact receipt/lstat check; this reads zero voltage bytes."""
    module_path = _bound_path(contract, contract["sources"]["voltage_preflight"])
    module = _load_file_module("reviewed_voltage_identity_preflight", module_path)
    receipt = module.verify_voltage_identity(contract["voltage_identity"])
    if receipt.get("status") != "PASS":
        raise RuntimeError("compact voltage continuity did not pass")
    if receipt.get("voltage_bytes_read_by_this_compact_preflight") != 0:
        raise RuntimeError("compact voltage continuity unexpectedly read voltage bytes")
    if receipt.get("historic_and_current_match") is not True:
        raise RuntimeError("historic/current full voltage identities differ")
    return receipt


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


def _execution_complete(
    receipt: Mapping[str, Any], arm: str, contract: Mapping[str, Any],
) -> dict[str, str]:
    """Bind completion and the deterministic inventory without reading outcomes."""
    if receipt.get("status") != "complete":
        raise RuntimeError(f"{arm} did not report complete execution state")
    complete_path = Path(receipt.get("complete_path", ""))
    if not complete_path.is_file():
        raise RuntimeError(f"{arm} completion file is absent")
    complete = json.loads(complete_path.read_text())
    if complete.get("status") != "complete":
        raise RuntimeError(f"{arm} completion file lacks complete execution state")
    if arm == "REF384_repeat":
        inventory_path = (
            Path(contract["planned_paths"]["root"]) / arm
            / "launch_evidence" / "artifact_inventory.json"
        )
    elif arm == "repaired_B384":
        config_path = _bound_path(contract, contract["configs"]["repaired_B384"])
        config = json.loads(config_path.read_text())
        inventory_path = (
            Path(config["trained"]["run_root"])
            / "launch_evidence" / "artifact_inventory.json"
        )
    else:
        raise ValueError(f"unexpected arm {arm}")
    if inventory_path.is_symlink() or not inventory_path.is_file():
        raise RuntimeError(f"{arm} deterministic artifact inventory is absent or symlinked")
    return {"status": "complete", "complete_path": str(complete_path),
            "complete_sha256": sha256_file(complete_path),
            "artifact_inventory_path": str(inventory_path),
            "artifact_inventory_sha256": sha256_file(inventory_path)}


def resource_preflight(contract: Mapping[str, Any]) -> None:
    planned = contract["planned_paths"]
    root = Path(planned["root"])
    if shutil.disk_usage(root.parent).free < contract["resources"]["minimum_free_bytes"]:
        raise RuntimeError("free storage is below the frozen minimum")
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one visible CUDA GPU is required")


class OutputCapExceeded(RuntimeError):
    pass


def aggregate_output_bytes(contract: Mapping[str, Any]) -> int:
    """Count all persistent pair output, logs, and attempt evidence without following links."""
    planned = contract["planned_paths"]
    total = 0
    for root in (planned["root"], planned["logs"], planned["attempt_evidence"]):
        path = Path(root)
        if not path.exists():
            continue
        if path.is_symlink():
            raise OutputCapExceeded(f"persistent output root is a symlink: {path}")
        for directory, dirnames, filenames in os.walk(path, followlinks=False):
            base = Path(directory)
            for name in dirnames:
                if (base / name).is_symlink():
                    raise OutputCapExceeded(f"persistent output directory is a symlink: {base / name}")
            for name in filenames:
                member = base / name
                observed = os.lstat(member)
                if not os.path.isfile(member) or member.is_symlink():
                    raise OutputCapExceeded(f"persistent output member is not a regular file: {member}")
                total += int(observed.st_size)
    return total


def assert_live_output_budget(contract: Mapping[str, Any], stage: str) -> int:
    """Read-only ordinary-writer check; terminal reserve is never live data budget."""
    observed = aggregate_output_bytes(contract)
    limit = int(contract["resources"]["live_output_bytes_max"])
    if observed > limit:
        raise OutputCapExceeded(
            f"aggregate live-output threshold exceeded during {stage}: {observed} > {limit}"
        )
    return observed


def terminal_reserve_derivation(contract: Mapping[str, Any]) -> dict[str, int]:
    """Conservative peak allocation, including atomic replacement coexistence."""
    resources = contract["resources"]
    parts = {
        "counted_late_service_or_child_log_bytes": int(resources["counted_late_log_bytes_max"]),
        "mutable_control_final_plus_partial_bytes": 2 * CONTROL_RECEIPT_BYTES_MAX,
        "success_pending_receipts_bytes": 2 * CONTROL_RECEIPT_BYTES_MAX,
        "failure_receipts_and_partial_bytes": 2 * CONTROL_RECEIPT_BYTES_MAX,
    }
    parts["derived_terminal_control_reserve_bytes"] = sum(parts.values())
    return parts


def _write_new_durable(path: Path, payload: object) -> None:
    """Write a new bounded receipt without replacement or a hidden later write."""
    data = serialized_json(payload)
    if len(data) > CONTROL_RECEIPT_BYTES_MAX:
        raise ValueError(f"terminal receipt exceeds serialized bound: {len(data)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        os.write(descriptor, data)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)


def attempt_terminal_payload(contract: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": "en-first-medium-pair-attempt-terminal-pending-v3",
        "status": "PENDING_FINAL_READ_ONLY_AUDIT",
        "both_arms_and_ordinary_writers_stopped": True,
        "reserve_derivation": terminal_reserve_derivation(contract),
        "authoritative_success_path": str(Path(contract["planned_paths"]["root"]) / "COMPLETE.json"),
    }


def _final_complete_payload(
    contract: Mapping[str, Any], execution: Mapping[str, Any], total: int,
    attempt_receipt_sha256: str,
) -> dict[str, Any]:
    attempt = Path(contract["planned_paths"]["attempt_evidence"])
    return {
        "schema": "en-first-medium-pair-authoritative-complete-v3",
        "status": "complete",
        "arm_order": list(ARM_ORDER),
        "execution_states": execution["execution_states"],
        "scientific_outcomes_inspected_between_arms": False,
        "aggregate_persistent_output_bytes": total,
        "aggregate_persistent_output_limit_bytes": int(contract["resources"]["persistent_output_bytes_max"]),
        "live_output_bytes_max": int(contract["resources"]["live_output_bytes_max"]),
        "terminal_control_reserve_bytes": int(contract["resources"]["terminal_control_reserve_bytes"]),
        "terminal_attempt_receipt_sha256": attempt_receipt_sha256,
        "voltage_continuity_receipt_sha256": sha256_file(attempt / "VOLTAGE_CONTINUITY.json"),
        "final_audit_read_only": True,
        "authoritative_success_published_by_no_growth_rename": True,
        "no_later_counted_write_permitted": True,
    }


def finalize_success(
    contract: Mapping[str, Any], execution: Mapping[str, Any], *,
    before_final_audit_fixture: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Single outer finalizer: stage, read-only audit, then no-growth rename last."""
    root = Path(contract["planned_paths"]["root"])
    attempt = Path(contract["planned_paths"]["attempt_evidence"])
    pending_attempt = attempt / "ATTEMPT_TERMINAL.pending.json"
    pending_complete = root / "COMPLETE.pending.json"
    final_complete = root / "COMPLETE.json"
    if final_complete.exists() or pending_complete.exists() or pending_attempt.exists():
        raise FileExistsError("terminal finalization namespace is not fresh")
    derivation = terminal_reserve_derivation(contract)
    if derivation["derived_terminal_control_reserve_bytes"] != int(contract["resources"]["terminal_control_reserve_bytes"]):
        raise ValueError("derived terminal reserve differs from frozen reserve")
    pending_attempt_payload = attempt_terminal_payload(contract)
    _write_new_durable(pending_attempt, pending_attempt_payload)
    pending_attempt_sha256 = sha256_file(pending_attempt)
    base = aggregate_output_bytes(contract)
    total = base
    for _ in range(8):
        payload = _final_complete_payload(contract, execution, total, pending_attempt_sha256)
        candidate = base + len(serialized_json(payload))
        if candidate == total:
            break
        total = candidate
    else:
        raise RuntimeError("terminal receipt byte-count fixed point did not converge")
    payload = _final_complete_payload(contract, execution, total, pending_attempt_sha256)
    if base + len(serialized_json(payload)) != total:
        raise RuntimeError("terminal receipt byte-count fixed point is unstable")
    _write_new_durable(pending_complete, payload)
    if before_final_audit_fixture is not None:
        before_final_audit_fixture(contract)
    observed = aggregate_output_bytes(contract)  # the final aggregate audit is read-only
    limit = int(contract["resources"]["persistent_output_bytes_max"])
    if observed != total:
        raise OutputCapExceeded(f"final aggregate changed during finalization: {observed} != {total}")
    if observed > limit:
        raise OutputCapExceeded(f"final aggregate persistent output cap exceeded: {observed} > {limit}")
    os.replace(pending_complete, final_complete)  # same-directory rename; byte count cannot grow
    _fsync_directory(root)
    return payload


def run_monitored_command(
    command: list[str], *, contract: Mapping[str, Any], attempt_dir: Path,
    stage: str, environment: Mapping[str, str] | None = None,
) -> None:
    """Run an arm in its own process group and terminate it on aggregate cap crossing."""
    logs = Path(contract["planned_paths"]["logs"])
    logs.mkdir(parents=True, exist_ok=True)
    stdout_path, stderr_path = logs / f"{stage}.stdout.log", logs / f"{stage}.stderr.log"
    with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
        process = subprocess.Popen(
            command, stdout=stdout, stderr=stderr, env=dict(environment or os.environ),
            start_new_session=True,
        )
        interval = float(contract["resources"]["output_monitor_interval_seconds"])
        limit = int(contract["resources"]["live_output_bytes_max"])
        while process.poll() is None:
            observed = aggregate_output_bytes(contract)
            if observed >= limit:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                durable_json(attempt_dir / "OUTPUT_CAP_FAILURE.json", {
                    "schema": "en-first-medium-pair-output-cap-failure-v1",
                    "status": "TERMINAL_FAILURE", "stage": stage,
                    "observed_bytes": observed, "live_limit_bytes": limit,
                    "total_cap_bytes": int(contract["resources"]["persistent_output_bytes_max"]),
                    "process_group_terminated": True, "threshold_reached": True,
                })
                raise OutputCapExceeded(
                    f"aggregate persistent output cap reached during {stage}: {observed} >= {limit}"
                )
            time.sleep(interval)
        assert_live_output_budget(contract, stage)
        if process.returncode != 0:
            raise RuntimeError(f"{stage} child exited {process.returncode}")


def execute_arm_in_monitored_child(
    contract: Mapping[str, Any], arm: str, attempt_dir: Path,
) -> dict[str, str]:
    launcher = _bound_path(contract, contract["sources"]["pair_launcher"])
    mode = "--execute-ref-child" if arm == "REF384_repeat" else "--execute-repaired-child"
    command = [
        sys.executable, str(launcher), "--contract", contract["_contract_path"],
        "--expected-contract-sha256", contract["_expected_contract_sha256"], mode,
    ]
    environment = dict(os.environ)
    environment["EN_FIRST_MEDIUM_PAIR_CLAIM_SHA256"] = sha256_file(attempt_dir / "START_CLAIM.json")
    run_monitored_command(
        command, contract=contract, attempt_dir=attempt_dir, stage=arm,
        environment=environment,
    )
    if arm == "REF384_repeat":
        complete = Path(contract["planned_paths"]["root"]) / arm / "COMPLETE.json"
    else:
        repaired_path = _bound_path(contract, contract["configs"]["repaired_B384"])
        repaired = json.loads(repaired_path.read_text())
        complete = Path(repaired["trained"]["run_root"]) / "COMPLETE.json"
    return {"status": "complete", "complete_path": str(complete)}


def execute_pair(
    contract: Mapping[str, Any], *,
    ref_executor: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
    repaired_executor: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
    resource_probe: Callable[[Mapping[str, Any]], None] = resource_preflight,
    voltage_probe: Callable[[Mapping[str, Any]], Mapping[str, Any]] = verify_voltage_continuity,
) -> dict[str, Any]:
    if contract["execution_enabled"] is not True:
        raise PermissionError("pair contract is execution-disabled pending independent H1 review")
    planned = contract["planned_paths"]
    attempt_dir = Path(planned["attempt_evidence"])
    record_attempt_stage(attempt_dir, "persistent_service_marker")
    if os.environ.get(PERSISTENT_MARKER) != "1":
        raise PermissionError("pair execution requires the reviewed persistent service marker")
    if not (attempt_dir / "START_CLAIM.json").is_file():
        raise PermissionError("durable single-start claim is absent")
    record_attempt_stage(attempt_dir, "fresh_namespace_preflight")
    for label in ("root", "numba_cache", "logs"):
        if Path(planned[label]).exists():
            raise FileExistsError(f"planned {label} namespace already exists")
    record_attempt_stage(attempt_dir, "initial_output_cap_preflight")
    assert_live_output_budget(contract, "before_voltage_preflight")
    record_attempt_stage(attempt_dir, "compact_voltage_continuity")
    voltage_receipt = dict(voltage_probe(contract))
    if voltage_receipt.get("voltage_bytes_read_by_this_compact_preflight") != 0:
        raise RuntimeError("voltage continuity preflight read voltage")
    durable_json(attempt_dir / "VOLTAGE_CONTINUITY.json", voltage_receipt)
    root = Path(planned["root"])
    record_attempt_stage(attempt_dir, "gpu_and_free_space_preflight")
    resource_probe(contract)
    record_attempt_stage(attempt_dir, "create_pair_root")
    root.mkdir(parents=False, exist_ok=False)
    evidence = root / "launch_evidence"
    evidence.mkdir()
    atomic_json(evidence / "STARTED.json", {
        "status": "running", "arm_order": list(ARM_ORDER),
        "started_at": datetime.now(timezone.utc).isoformat(), "retry": False,
    })
    stage = "REF384_repeat"
    try:
        record_attempt_stage(attempt_dir, stage)
        ref_receipt = (
            ref_executor(contract) if ref_executor is not None
            else execute_arm_in_monitored_child(contract, stage, attempt_dir)
        )
        ref_state = _execution_complete(ref_receipt, stage, contract)
        assert_live_output_budget(contract, "after_REF384_repeat")
        atomic_json(evidence / "REF384_repeat_EXECUTION_STATE.json", ref_state)
        stage = "repaired_B384"
        record_attempt_stage(attempt_dir, stage)
        repaired_receipt = (
            repaired_executor(contract) if repaired_executor is not None
            else execute_arm_in_monitored_child(contract, stage, attempt_dir)
        )
        repaired_state = _execution_complete(repaired_receipt, stage, contract)
        atomic_json(evidence / "repaired_B384_EXECUTION_STATE.json", repaired_state)
        ordinary_output_bytes = assert_live_output_budget(contract, "after_repaired_B384")
        ready = {
            "status": "ready_for_outer_finalizer", "arm_order": list(ARM_ORDER),
            "execution_states": {"REF384_repeat": ref_state, "repaired_B384": repaired_state},
            "scientific_outcomes_inspected_between_arms": False,
            "ordinary_output_bytes_before_terminal_receipts": ordinary_output_bytes,
        }
        record_attempt_stage(attempt_dir, "outer_finalization_pending")
        return ready
    except Exception as exc:
        atomic_json(evidence / "RUN_FAILURE.json", {
            "status": "failed", "stage": stage, "exception_type": type(exc).__name__,
            "exception_message": str(exc)[:FAILURE_TEXT_CHARS_MAX], "completion_written": False,
        })
        raise


def run_single_attempt(
    *, contract_path: str | Path, expected_contract_sha256: str,
    expected_service_sha256: str, attempt_evidence_dir: str | Path,
    ref_executor: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
    repaired_executor: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
    resource_probe: Callable[[Mapping[str, Any]], None] = resource_preflight,
    voltage_probe: Callable[[Mapping[str, Any]], Mapping[str, Any]] = verify_voltage_continuity,
    contract_validator: Callable[[str | Path, str], dict[str, Any]] = validate_contract,
    pre_finalizer_fixture: Callable[[Mapping[str, Any], Mapping[str, Any]], None] | None = None,
    before_final_audit_fixture: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Consume one start, then preserve every later failure in the attempt namespace."""
    attempt_dir = Path(attempt_evidence_dir)
    consume_start_claim(
        attempt_dir, contract_path=contract_path,
        expected_contract_sha256=expected_contract_sha256,
        expected_service_sha256=expected_service_sha256,
    )
    stage = "contract_validation"
    try:
        contract = contract_validator(contract_path, expected_contract_sha256)
        service_path = _bound_path(contract, contract["service"]["unit_binding"])
        if sha256_file(service_path) != expected_service_sha256:
            raise ValueError("invoked service hash differs from single-start binding")
        if Path(contract["planned_paths"]["attempt_evidence"]).resolve() != attempt_dir.resolve():
            raise ValueError("attempt evidence path differs from frozen contract")
        stage = "pair_preflight_and_execution"
        execution = execute_pair(
            contract, ref_executor=ref_executor, repaired_executor=repaired_executor,
            resource_probe=resource_probe, voltage_probe=voltage_probe,
        )
        if pre_finalizer_fixture is not None:
            pre_finalizer_fixture(contract, execution)
            assert_live_output_budget(contract, "after_pre_finalizer_fixture")
        complete = finalize_success(
            contract, execution, before_final_audit_fixture=before_final_audit_fixture
        )
        return complete
    except BaseException as exc:
        current_stage = attempt_dir / "CURRENT_STAGE.json"
        if current_stage.is_file():
            stage = json.loads(current_stage.read_text()).get("stage", stage)
        durable_attempt_failure(attempt_dir, stage, exc)
        raise


def _verify_child_claim(contract: Mapping[str, Any]) -> None:
    claim = Path(contract["planned_paths"]["attempt_evidence"]) / "START_CLAIM.json"
    expected = os.environ.get("EN_FIRST_MEDIUM_PAIR_CLAIM_SHA256")
    if not expected or not claim.is_file() or sha256_file(claim) != expected:
        raise PermissionError("arm child lacks the exact parent single-start claim")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--expected-contract-sha256", required=True)
    parser.add_argument("--attempt-evidence-dir", type=Path)
    parser.add_argument("--expected-service-sha256")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--execute-reviewed-once", action="store_true")
    mode.add_argument("--execute-ref-child", action="store_true")
    mode.add_argument("--execute-repaired-child", action="store_true")
    args = parser.parse_args()
    if args.execute_reviewed_once:
        if args.attempt_evidence_dir is None or not args.expected_service_sha256:
            parser.error("reviewed execution requires attempt evidence and exact service hash")
        run_single_attempt(
            contract_path=args.contract, expected_contract_sha256=args.expected_contract_sha256,
            expected_service_sha256=args.expected_service_sha256,
            attempt_evidence_dir=args.attempt_evidence_dir,
        )
        return 0
    contract = validate_contract(args.contract, args.expected_contract_sha256)
    report = {
        "status": "validated", "execution_enabled": contract["execution_enabled"],
        "arm_order": list(ARM_ORDER), "recording_or_voltage_opened": False,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.validate_only:
        return 0
    _verify_child_claim(contract)
    if args.execute_ref_child:
        execute_ref_arm(contract)
    else:
        execute_repaired_arm(contract)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
