"""One authorized fresh cache-managed full-session Candidate2 run."""
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

from pipeline.downstream import pin_sort_identity, run_curation_stage, run_qc_stage, run_standard_qc_stage
from pipeline.kilosort_bounded_reader_compat import bounded_reader_receipt
from pipeline.preprocess import validate_accepted_recording
from pipeline.runtime import validate_production_environment
from testing.ladder_sorter import RESCUE, RESCUE_RIGID, _json_safe, check_effective_settings, run_sorter_config


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


def memory_metrics() -> dict:
    meminfo = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        meminfo[key] = int(value.strip().split()[0]) * 1024
    pressure = {}
    for line in Path("/proc/pressure/memory").read_text().splitlines():
        fields = line.split()
        pressure[fields[0]] = {k: float(v) for k, v in (item.split("=") for item in fields[1:])}
    return {"mem_available_bytes": meminfo["MemAvailable"], "pressure": pressure}


def gpu_metrics() -> dict:
    free, total = torch.cuda.mem_get_info(0)
    return {
        "device": torch.cuda.get_device_name(0),
        "free_bytes": int(free),
        "total_bytes": int(total),
        "allocated_bytes": int(torch.cuda.memory_allocated(0)),
        "reserved_bytes": int(torch.cuda.memory_reserved(0)),
    }


def cgroup_path() -> Path:
    row = next(line for line in Path("/proc/self/cgroup").read_text().splitlines() if line.startswith("0::"))
    return Path("/sys/fs/cgroup") / row.split("::", 1)[1].lstrip("/")


def snapshot() -> dict:
    value = {"utc": datetime.now(timezone.utc).isoformat(), **memory_metrics(), "gpu": gpu_metrics()}
    cg = cgroup_path()
    for name in ("memory.current", "memory.peak", "memory.events", "memory.pressure", "memory.stat"):
        target = cg / name
        value[name] = target.read_text().strip() if target.exists() else None
    return value


def verify_preflight(path: Path, contract_sha256: str, source_manifest_sha256: str) -> dict:
    value = json.loads(path.read_text())
    required = {
        "schema": "candidate2-cache-managed-preflight-v1",
        "pass": True,
        "contract_sha256": contract_sha256,
        "source_manifest_sha256": source_manifest_sha256,
    }
    if any(value.get(key) != expected for key, expected in required.items()):
        raise RuntimeError("fresh preflight binding differs")
    age = time.time() - float(value["finished_unix"])
    if age < 0 or age > 300:
        raise RuntimeError(f"fresh preflight receipt age is invalid: {age}")
    return value


def require_resource_gates(contract: dict) -> dict:
    resource = contract["resource_conditions"]
    metrics = snapshot()
    if shutil.disk_usage(Path(contract["attempt"]["output_root"]).parent).free < resource["minimum_output_free_bytes"]:
        raise RuntimeError("output free-space condition failed")
    if metrics["mem_available_bytes"] < resource["minimum_mem_available_bytes"]:
        raise RuntimeError("host available-memory condition failed")
    if metrics["pressure"]["some"]["avg10"] > resource["maximum_memory_psi_some_avg10"]:
        raise RuntimeError("host memory-pressure condition failed")
    if metrics["gpu"]["free_bytes"] < resource["minimum_gpu_free_bytes"]:
        raise RuntimeError("GPU free-memory condition failed")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--contract-sha256", required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    args = parser.parse_args()

    if sha256(args.contract) != args.contract_sha256:
        raise RuntimeError("execution contract hash mismatch")
    contract = json.loads(args.contract.read_text())
    if contract.get("execution_enabled") is not True:
        raise RuntimeError("contract is execution-disabled")
    if contract.get("parent_contract_sha256") != "1a68ab33ee49c7a5e2c729a709109f5a7dc8e90e778c86a7b7962fc8eb47d7dc":
        raise RuntimeError("wrong reviewed parent contract")

    attempt = contract["attempt"]
    output = Path(attempt["output_root"])
    results = Path(attempt["results_dir"])
    partial = Path(attempt["partial_results_dir"])
    if results != output / "kilosort4" or partial != Path(str(results) + ".partial"):
        raise RuntimeError("attempt output paths are not singly derived")
    if any(path.exists() for path in (output, results, partial)):
        raise RuntimeError("fresh Candidate2 output namespace is not absent")

    preflight = verify_preflight(args.preflight, args.contract_sha256, args.source_manifest_sha256)
    environment = validate_production_environment(require_cuda=True)
    binary = Path(contract["input"]["recording_binary"])
    if os.path.realpath(os.environ.get("KILOSORT_BOUNDED_PREAD_PATH", "")) != os.path.realpath(binary):
        raise RuntimeError("bounded reader environment is not bound to the frozen binary")
    reader = bounded_reader_receipt()
    if not reader["applied"] or reader["source_sha256"] != contract["cache_managed_mode"]["installed_kilosort_io_sha256"]:
        raise RuntimeError("installed bounded reader identity differs")
    if binary.stat().st_size != contract["input"]["recording_binary_bytes"]:
        raise RuntimeError("recording binary size differs")

    base = Path(contract["input"]["recording_dir"]).parent
    recording_dir = Path(contract["input"]["recording_dir"])
    manifest_path = Path(contract["input"]["recording_manifest"])
    baseline_manifest_path = base / "kilosort4/rescue_sort_manifest.json"
    if sha256(manifest_path) != contract["input"]["recording_manifest_sha256"]:
        raise RuntimeError("recording manifest changed")
    if sha256(baseline_manifest_path) != contract["science_binding"]["baseline_sort_manifest_sha256"]:
        raise RuntimeError("baseline sort manifest changed")
    recording_manifest = json.loads(manifest_path.read_text())
    baseline = json.loads(baseline_manifest_path.read_text())
    if recording_manifest.get("num_samples") != contract["input"]["num_samples"] or recording_manifest.get("num_channels") != contract["input"]["num_channels"]:
        raise RuntimeError("recording extent differs")
    if baseline.get("complete") is not True or baseline.get("recording_request_digest") != recording_manifest.get("request_digest"):
        raise RuntimeError("baseline identity differs")
    if baseline.get("sorter_params") != _json_safe(RESCUE.params()):
        raise RuntimeError("reference request differs from frozen rescue baseline")
    candidate_saved = json.loads(Path(contract["science_binding"]["candidate2_saved_request"]).read_text())["sorter_params"]
    if _json_safe(candidate_saved) != _json_safe(RESCUE_RIGID.params()):
        raise RuntimeError("complete Candidate2 sorter parameters differ")

    require_resource_gates(contract)
    # Operation 1: full accepted-content validation with exact-target per-range DONTNEED.
    validated = validate_accepted_recording(recording_dir)
    if validated.get("recording_content_sha256") != contract["input"]["recording_content_sha256"]:
        raise RuntimeError("accepted recording content identity differs")
    # Operation 2: clear any predecessor residency, settle, and reapply gates.
    before_evict = snapshot()
    with binary.open("rb") as stream:
        os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    time.sleep(5)
    after_evict = require_resource_gates(contract)

    output.mkdir(parents=False, exist_ok=False)
    atomic_json(output / "request.json", {
        "schema": "candidate2-native-rigid-cache-managed-imec0-v1",
        "contract": str(args.contract), "contract_sha256": args.contract_sha256,
        "attempt": attempt, "preflight": preflight,
        "candidate_settings": _json_safe(RESCUE_RIGID.params()),
        "required_effective_nblocks": 1,
    })
    atomic_json(output / "environment.json", environment)
    atomic_json(output / "cache_mode_receipt.json", {
        "schema": "candidate2-cache-managed-mode-v1", "reader": reader,
        "target_realpath": os.path.realpath(binary),
        "before_whole_file_advice": before_evict,
        "after_settle_and_recheck": after_evict,
        "operations": contract["cache_managed_mode"]["required_operations"],
    })

    def stage(name: str, **extra) -> None:
        atomic_json(output / "status.json", {"stage": name, "updated_utc": datetime.now(timezone.utc).isoformat(), **extra})
        print(name, flush=True)

    try:
        # Operation 3 is consumed inside installed Kilosort for every exact-target batch.
        stage("sorting_full_session_cache_managed")
        sort_manifest = run_sorter_config(recording_dir, results, RESCUE_RIGID)
        effective = check_effective_settings("rescue_rigid", sort_manifest)
        atomic_json(output / "effective_settings.json", effective)
        identity = pin_sort_identity(results, output / "sort_identity.json")
        stage("curation", effective_nblocks=effective["effective_nblocks"])
        run_curation_stage(results / "sorter_output", output / "cur", identity)
        stage("waveform_and_amplitude_qc")
        run_qc_stage(recording_dir, output / "cur/cur_output", output / "qc", identity)
        stage("standard_per_unit_qc")
        run_standard_qc_stage(recording_dir, output / "cur/cur_output", output / "qc", output / "qc/standard", identity)
        stage("candidate2_sort_and_qc_complete_comparison_pending", terminal=True)
    except BaseException as exc:
        stage("failed_preserved_no_retry", terminal=True, error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
