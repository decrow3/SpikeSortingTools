"""Metadata/source/resource-only preflight for the Candidate2 managed service."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import torch


STATIC_SYSTEMD_PROPERTIES = (
    "FragmentPath", "ExecStart", "ExecStartPre", "Environment",
    "TimeoutStartUSec", "OOMPolicy", "Restart", "MemoryHigh", "MemoryMax",
    "CPUQuotaPerSecUSec", "RuntimeMaxUSec", "Requires", "BindsTo", "After",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def metrics() -> dict:
    meminfo = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        meminfo[key] = int(value.strip().split()[0]) * 1024
    pressure = {}
    for line in Path("/proc/pressure/memory").read_text().splitlines():
        fields = line.split()
        pressure[fields[0]] = {k: float(v) for k, v in (item.split("=") for item in fields[1:])}
    free, total = torch.cuda.mem_get_info(0)
    compute_apps = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"],
        text=True, capture_output=True, check=True,
    ).stdout.strip().splitlines()
    return {
        "mem_available_bytes": meminfo["MemAvailable"], "pressure": pressure,
        "gpu": {"device": torch.cuda.get_device_name(0), "free_bytes": int(free), "total_bytes": int(total), "compute_apps": compute_apps},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--contract-sha256", required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--installed-unit", type=Path, required=True)
    parser.add_argument("--monitor-unit", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    args = parser.parse_args()
    if sha256(args.contract) != args.contract_sha256 or sha256(args.source_manifest) != args.source_manifest_sha256:
        raise RuntimeError("contract/source manifest hash mismatch")
    contract = json.loads(args.contract.read_text())
    manifest = json.loads(args.source_manifest.read_text())
    if contract.get("execution_enabled") is not True:
        raise RuntimeError("contract is execution-disabled")
    authorization = json.loads(args.authorization.read_text())
    expected_authorization = {
        "status": "AUTHORIZED_ONE_FRESH_CANDIDATE2_CACHE_MANAGED_RUN",
        "contract_sha256": args.contract_sha256,
        "source_manifest_sha256": args.source_manifest_sha256,
        "target_service_sha256": sha256(args.installed_unit),
        "monitor_service_sha256": sha256(args.monitor_unit),
    }
    if any(authorization.get(key) != value for key, value in expected_authorization.items()):
        raise RuntimeError("installed service/contract/source authorization differs")
    for relative, expected in manifest["files"].items():
        if sha256(args.source_root / relative) != expected:
            raise RuntimeError(f"source differs: {relative}")
    attempt = contract["attempt"]
    output = Path(attempt["output_root"])
    results = Path(attempt["results_dir"])
    partial = Path(attempt["partial_results_dir"])
    if results != output / "kilosort4" or partial != Path(str(results) + ".partial"):
        raise RuntimeError("attempt paths are not singly derived")
    if any(path.exists() for path in (output, results, partial)):
        raise RuntimeError("fresh output namespace is not absent")
    binary = Path(contract["input"]["recording_binary"])
    if binary.stat().st_size != contract["input"]["recording_binary_bytes"]:
        raise RuntimeError("recording metadata differs")
    if sha256(Path(contract["input"]["recording_manifest"])) != contract["input"]["recording_manifest_sha256"]:
        raise RuntimeError("recording manifest differs")
    if sha256(Path(contract["science_binding"]["candidate2_saved_request"])) != contract["science_binding"]["candidate2_saved_request_sha256"]:
        raise RuntimeError("Candidate2 saved request differs")
    observed = metrics()
    resources = contract["resource_conditions"]
    if shutil.disk_usage(output.parent).free < resources["minimum_output_free_bytes"]:
        raise RuntimeError("output free-space condition failed")
    if observed["mem_available_bytes"] < resources["minimum_mem_available_bytes"]:
        raise RuntimeError("host available-memory condition failed")
    if observed["pressure"]["some"]["avg10"] > resources["maximum_memory_psi_some_avg10"]:
        raise RuntimeError("memory PSI condition failed")
    if observed["gpu"]["free_bytes"] < resources["minimum_gpu_free_bytes"] or observed["gpu"]["compute_apps"]:
        raise RuntimeError("GPU idle/free-memory condition failed")
    process_rows = subprocess.run(["ps", "-eo", "pid=,args="], text=True, capture_output=True, check=True).stdout.splitlines()
    conflict_terms = ("kilosort.run_kilosort", "testing.candidate2_cache_managed_full", "testing.imec1_part2b_sort")
    process_conflicts = [row.strip() for row in process_rows if any(term in row for term in conflict_terms) and not row.lstrip().startswith(str(os.getpid()) + " ")]
    if process_conflicts:
        raise RuntimeError(f"competing sorter process found: {process_conflicts}")
    loaded_units = {}
    for role, unit_name in {
        "target": contract["attempt"]["service_name"],
        "monitor": contract["attempt"]["monitor_service_name"],
    }.items():
        output = subprocess.run(
            ["systemctl", "--user", "show", unit_name, *[f"--property={key}" for key in STATIC_SYSTEMD_PROPERTIES]],
            text=True, capture_output=True, check=True,
        ).stdout
        loaded_units[role] = dict(line.split("=", 1) for line in output.splitlines() if "=" in line)
    if loaded_units != authorization.get("loaded_units"):
        raise RuntimeError("manager-loaded unit properties differ from post-install authorization")
    systemd_values = loaded_units["target"]
    if systemd_values.get("FragmentPath") != str(args.installed_unit):
        raise RuntimeError("loaded target FragmentPath differs")
    if loaded_units["monitor"].get("FragmentPath") != str(args.monitor_unit):
        raise RuntimeError("loaded monitor FragmentPath differs")
    uncapped = ("MemoryHigh", "MemoryMax", "CPUQuotaPerSecUSec", "RuntimeMaxUSec")
    if (systemd_values.get("TimeoutStartUSec") != "infinity"
            or systemd_values.get("OOMPolicy") != "stop"
            or systemd_values.get("Restart") != "no"
            or any(systemd_values.get(key) != "infinity" for key in uncapped)):
        raise RuntimeError(f"installed target lifecycle/cap properties differ: {systemd_values}")
    if f"KILOSORT_BOUNDED_PREAD_PATH={binary}" not in systemd_values.get("Environment", ""):
        raise RuntimeError("installed service bounded-reader environment differs")
    if "-I -S -u" not in systemd_values.get("ExecStart", ""):
        raise RuntimeError("target bootstrap is not isolated before source verification")
    if "candidate2_cache_managed_monitor.py" not in loaded_units["monitor"].get("ExecStart", "") or "-I -S -u" not in loaded_units["monitor"].get("ExecStart", ""):
        raise RuntimeError("external monitor command differs")
    ready_path = Path(contract["attempt"]["monitor_output"]) / "READY.json"
    deadline = time.monotonic() + 30
    while not ready_path.is_file() and time.monotonic() < deadline:
        time.sleep(0.25)
    ready = json.loads(ready_path.read_text())
    if (ready.get("schema") != "candidate2-external-monitor-ready-v1"
            or ready.get("contract_sha256") != args.contract_sha256
            or ready.get("target_service_sha256") != expected_authorization["target_service_sha256"]
            or ready.get("monitor_service_sha256") != expected_authorization["monitor_service_sha256"]
            or not ready.get("first_sample_complete")
            or ready.get("ready_failures") != []
            or ready.get("systemctl_returncode") != "0"
            or ready.get("target_load_state") != "loaded"
            or ready.get("target_active_state") not in {"activating", "active"}
            or not ready.get("target_invocation_id")
            or not ready.get("target_control_group")
            or ready.get("gpu_returncode") != 0
            or ready.get("gpu_compute_returncode") != 0
            or time.time() - float(ready.get("first_sample_unix", 0)) > 60):
        raise RuntimeError("external monitor first-sample gate differs or is stale")
    installed_io = Path("/home/huklaban5/Documents/SpikeSortingTools/SpikeSortingTools/environments/rescue-production/.venv/lib/python3.12/site-packages/kilosort/io.py")
    if sha256(installed_io) != contract["cache_managed_mode"]["installed_kilosort_io_sha256"]:
        raise RuntimeError("installed bounded reader source differs")
    if os.path.realpath(os.environ.get("KILOSORT_BOUNDED_PREAD_PATH", "")) != os.path.realpath(binary):
        raise RuntimeError("bounded reader environment differs")
    output.parent.mkdir(parents=True, exist_ok=True)
    receipt = {
        "schema": "candidate2-cache-managed-preflight-v1", "pass": True,
        "contract_sha256": args.contract_sha256,
        "source_manifest_sha256": args.source_manifest_sha256,
        "input": {"binary": str(binary), "bytes": binary.stat().st_size},
        "resources": observed, "output_free_bytes": shutil.disk_usage(output.parent).free,
        "process_conflicts": process_conflicts, "loaded_units": loaded_units,
        "monitor_ready": ready,
        "finished_utc": datetime.now(timezone.utc).isoformat(), "finished_unix": time.time(),
        "recording_opened": False, "recording_bytes_read": 0,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.receipt, receipt)


if __name__ == "__main__":
    main()
