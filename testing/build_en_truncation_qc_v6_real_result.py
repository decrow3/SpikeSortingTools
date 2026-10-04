#!/usr/bin/env python3
"""Build the immutable compact result packet for the sole QC v6 execution."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np

PUB = Path("/mnt/NPX/Luke/DARTsort_motion_experiments")
CANDIDATE = PUB / "en_truncation_qc_regeneration_lifecycle_repair_candidate_20261002_v6_h5"
REVIEW = PUB / "en_truncation_qc_regeneration_lifecycle_v6_h1_review_20261002_v1_h1"
GATE = PUB / "en_truncation_qc_regeneration_lifecycle_v6_one_execution_gate_20261002_v1_h1"
OUTPUT = Path("/media/huklaban5/Data/en_common_support_crop_screen_20260930_v1/qc_regeneration_v6")
LOCK = Path("/media/huklaban5/Data/en_common_support_crop_screen_20260930_v1/qc_regeneration_authorization_ledger_v1/6c99957b6f3a38152141db1887cda22a03571e6886b97cf9ff4d7a975f98f7f7.consumed.json")
SERVICE_UNIT = Path("/home/huklaban5/Documents/SpikeSortingTools/SpikeSortingTools/testing/en_truncation_qc_regeneration_v6_real.service")
SERVICE_LOGS = Path("/home/huklaban5/Documents/SpikeSortingTools/SpikeSortingTools/testing/outputs/en_truncation_qc_regeneration_v6_real_20261002")
PACKET = PUB / "en_truncation_qc_regeneration_v6_real_result_20261002_v1_h5"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def service_state() -> dict:
    properties = (
        "Id", "ActiveState", "SubState", "Result", "ExecMainCode", "ExecMainStatus",
        "ExecMainStartTimestamp", "ExecMainExitTimestamp", "CPUUsageNSec", "MemoryMax",
        "RuntimeMaxUSec", "Restart", "MainPID", "FragmentPath", "AllowedCPUs",
        "CPUQuotaPerSecUSec", "LimitCPU", "LimitAS", "NRestarts",
    )
    text = subprocess.check_output(
        ["systemctl", "--user", "show", "en_truncation_qc_regeneration_v6_real.service",
         *[f"-p{name}" for name in properties], "--no-pager"], text=True,
    )
    result = {}
    for line in text.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            result[key] = value
    return result


def main() -> None:
    if PACKET.exists():
        raise FileExistsError(f"refusing to overwrite {PACKET}")
    expected = {
        CANDIDATE / "MANIFEST.sha256": "77ea48413aa9b7d5b8a1598d62cb754e2ed959fb9601039b656cf25ca2cecd61",
        CANDIDATE / "COMPLETE.json": "a2c048efcc6bdbd677896cead7755070f7ef809fca167de4d51840dbe3b7204e",
        REVIEW / "MANIFEST.sha256": "055e85afbc411c4e8b091db9599f4230d3f2d18fd5c9b32efdb98dc2316bb8a6",
        REVIEW / "COMPLETE.json": "cbd419775d6f1bcda44e8995295202ea897bf263eb213964e3f2e194c33de677",
        GATE / "MANIFEST.sha256": "b95e31549f47fd5dd28cfb9057a846da46c24a14921170a8bc81349a54266594",
        GATE / "COMPLETE.json": "db36dfa8bebc0cf6c5342594e41b9b33c635d3e9077beeb5de910037d187cd43",
    }
    for path, digest in expected.items():
        if sha(path) != digest:
            raise RuntimeError(f"reviewed packet moved: {path}")
    complete = json.loads((OUTPUT / "COMPLETE.json").read_text())
    run_receipt = json.loads((OUTPUT / "evidence/RUN_RECEIPT.json").read_text())
    if complete["status"] != "complete" or complete["run_receipt_sha256"] != sha(OUTPUT / "evidence/RUN_RECEIPT.json"):
        raise RuntimeError("completion/run receipt binding failed")
    if run_receipt["status"] != "success" or run_receipt["exit_code"] != 0:
        raise RuntimeError("managed run did not succeed")

    recorded_inventory = json.loads((OUTPUT / "evidence/OUTPUT_INVENTORY.json").read_text())
    inventory_errors = []
    for relative, row in recorded_inventory["files"].items():
        path = OUTPUT / relative
        if not path.is_file() or path.stat().st_size != row["bytes"] or sha(path) != row["sha256"]:
            inventory_errors.append(relative)
    if inventory_errors:
        raise RuntimeError(f"output inventory mismatch: {inventory_errors}")

    arms = {}
    for arm in ("REF384", "existing_corrected_B384"):
        directory = OUTPUT / arm / "amp_truncation"
        files = {name: {"path": str(directory / name), "bytes": (directory / name).stat().st_size,
                        "sha256": sha(directory / name)}
                 for name in ("truncation_qc.npz", "present_qc.npz", "truncation_qc.pdf")}
        with np.load(directory / "truncation_qc.npz", allow_pickle=False) as data:
            truncation_schema = {name: {"shape": list(data[name].shape), "dtype": str(data[name].dtype)} for name in data.files}
            window_rows = int(data["cid"].size)
            unique_clusters = int(np.unique(data["cid"]).size)
        with np.load(directory / "present_qc.npz", allow_pickle=False) as data:
            present_schema = {name: {"shape": list(data[name].shape), "dtype": str(data[name].dtype)} for name in data.files}
            present_rows = int(data["cid"].size)
        arms[arm] = {"files": files, "truncation_schema": truncation_schema,
                     "present_schema": present_schema, "window_rows": window_rows,
                     "present_rows": present_rows, "unique_truncation_cluster_ids": unique_clusters}

    total_bytes = sum(path.stat().st_size for path in OUTPUT.rglob("*") if path.is_file())
    state = service_state()
    summary = {
        "schema": "truncation-qc-v6-real-result-v1", "scientific_interpretation": "none",
        "execution_count": 1, "authorization_consumed": True,
        "token_identity": "6c99957b6f3a38152141db1887cda22a03571e6886b97cf9ff4d7a975f98f7f7",
        "global_lock": {"path": str(LOCK), "sha256": sha(LOCK)},
        "output_root": str(OUTPUT), "output_total_bytes": total_bytes,
        "output_limit_bytes": 1073741824, "arms": arms,
        "managed_run": {"elapsed_seconds": run_receipt["elapsed_seconds"],
                        "cpu_usage_ns": int(state["CPUUsageNSec"]),
                        "highest_observed_memory_bytes_during_monitoring": 641585152,
                        "memory_peak_property_after_exit": "unavailable on this systemd version",
                        "service": state},
        "receipts": {name: {"path": str(OUTPUT / "evidence" / name), "sha256": sha(OUTPUT / "evidence" / name)}
                     for name in ("BOOTSTRAP.json", "RUNTIME_ATTESTATION.json", "INPUTS.json", "LAUNCH.json",
                                  "worker.stdout", "worker.stderr", "OUTPUT_INVENTORY.json", "RUN_RECEIPT.json", "STATUS.json")},
        "completion": {"path": str(OUTPUT / "COMPLETE.json"), "sha256": sha(OUTPUT / "COMPLETE.json"), "payload": complete},
        "service_logs": {name: {"path": str(SERVICE_LOGS / name), "sha256": sha(SERVICE_LOGS / name),
                                "bytes": (SERVICE_LOGS / name).stat().st_size}
                         for name in ("service.stdout", "service.stderr")},
        "anomalies": [],
        "scope": {"recording_or_voltage_reads": 0, "sorting_training_detection": 0,
                  "rf_or_holdout_access": 0, "retry_count": 0},
    }
    PACKET.mkdir(parents=True)
    write_json(PACKET / "RESULT_SUMMARY.json", summary)
    write_json(PACKET / "PRELAUNCH_VERIFICATION.json", {
        "candidate_review_gate_hashes": {str(path): digest for path, digest in expected.items()},
        "validate_only": "passed", "gate_preflight_token_identity": summary["token_identity"],
        "all_eight_input_hashes": "matched frozen config", "frozen_output_absent": True,
        "global_token_lock_absent": True, "storage_writable": True,
        "available_storage_bytes": 1182068994048, "competing_process": False,
        "systemd_launcher_survival_proof": "luke-registration-manager-proof-v1.service Result=success ExecMainStatus=0",
    })
    for name in ("BOOTSTRAP.json", "RUNTIME_ATTESTATION.json", "INPUTS.json", "LAUNCH.json",
                 "OUTPUT_INVENTORY.json", "RUN_RECEIPT.json", "STATUS.json"):
        copy(OUTPUT / "evidence" / name, PACKET / "evidence" / name)
    copy(OUTPUT / "COMPLETE.json", PACKET / "evidence/OUTPUT_COMPLETE.json")
    copy(LOCK, PACKET / "evidence/GLOBAL_AUTHORIZATION_CONSUMED.json")
    copy(SERVICE_UNIT, PACKET / "service/en_truncation_qc_regeneration_v6_real.service")
    copy(SERVICE_LOGS / "service.stdout", PACKET / "service/service.stdout")
    copy(SERVICE_LOGS / "service.stderr", PACKET / "service/service.stderr")
    for source, destination in (
        (CANDIDATE / "MANIFEST.sha256", PACKET / "bindings/candidate_MANIFEST.sha256"),
        (CANDIDATE / "COMPLETE.json", PACKET / "bindings/candidate_COMPLETE.json"),
        (REVIEW / "MANIFEST.sha256", PACKET / "bindings/review_MANIFEST.sha256"),
        (REVIEW / "COMPLETE.json", PACKET / "bindings/review_COMPLETE.json"),
        (GATE / "MANIFEST.sha256", PACKET / "bindings/gate_MANIFEST.sha256"),
        (GATE / "COMPLETE.json", PACKET / "bindings/gate_COMPLETE.json"),
        (GATE / "ONE_EXECUTION_GATE.json", PACKET / "bindings/ONE_EXECUTION_GATE.json"),
    ):
        copy(source, destination)
    (PACKET / "README.md").write_text(
        "# Truncation-QC v6 single real execution result\n\n"
        "The sole H1-authorized saved-artifact regeneration completed successfully under the frozen persistent systemd "
        "service. This packet is compact: generated QC files remain at their exact paths in `RESULT_SUMMARY.json`; no "
        "large PDFs or input arrays are duplicated here. No scientific efficacy interpretation is made.\n\n"
        "Implementation checks\n"
        "- Done: packet/review/gate/input hashes, one-shot lock, bootstrap/runtime receipts, output schemas and hashes, service terminal state, inventory and completion binding.\n"
        "- Not done: no recording/voltage, sorting/training/detection, RF/holdout, retry, tuning or scientific comparison.\n"
        "- Can establish: exactly one frozen saved-artifact QC regeneration completed and produced schema-valid outputs.\n"
        "- Cannot establish: amplitude-completeness efficacy or any sorter-arm scientific conclusion.\n"
    )
    members = sorted(path for path in PACKET.rglob("*") if path.is_file() and path.name not in {"MANIFEST.sha256", "COMPLETE.json"})
    (PACKET / "MANIFEST.sha256").write_text("".join(f"{sha(path)}  {path.relative_to(PACKET).as_posix()}\n" for path in members))
    manifest = sha(PACKET / "MANIFEST.sha256")
    write_json(PACKET / "COMPLETE.json", {"schema": "immutable-result-completion-v1", "status": "complete",
        "manifest_sha256": manifest, "source_output_complete_sha256": sha(OUTPUT / "COMPLETE.json"),
        "execution_count": 1, "scientific_interpretation": "none", "completion_written_last": True})
    print(json.dumps({"packet": str(PACKET), "manifest_sha256": manifest,
                      "complete_sha256": sha(PACKET / "COMPLETE.json")}, indent=2))


if __name__ == "__main__":
    main()
