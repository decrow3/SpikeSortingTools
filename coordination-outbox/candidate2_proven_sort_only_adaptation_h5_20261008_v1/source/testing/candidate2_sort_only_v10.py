#!/usr/bin/env python3
"""Cache-managed, sort-only retry using the preserved imec1 materialization."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import math
from datetime import datetime, timezone
from pathlib import Path

from pipeline.kilosort_bounded_reader_compat import ensure_kilosort_bounded_reader
from pipeline.kilosort_cache_compat import ensure_kilosort_cache_compatibility
from pipeline.preprocess import validate_accepted_recording
from testing.ladder_sorter import RESCUE_RIGID, _json_safe, check_effective_settings, run_sorter_config
from testing.imec1_part2b_contract import atomic_json, sha256
from testing.imec1_part2b_sort_retry import memory_metrics


def cgroup_path() -> Path:
    row = next(line for line in Path("/proc/self/cgroup").read_text().splitlines() if line.startswith("0::"))
    return Path("/sys/fs/cgroup") / row.split("::", 1)[1].lstrip("/")


def telemetry(stop: threading.Event, path: Path) -> None:
    cg = cgroup_path()
    while not stop.wait(10):
        payload = {"utc": datetime.now(timezone.utc).isoformat(), **memory_metrics()}
        for name in (
            "memory.current", "memory.peak", "memory.events",
            "memory.pressure", "memory.stat",
        ):
            target = cg / name
            payload[name] = target.read_text().strip() if target.exists() else None
        with path.open("a") as stream:
            stream.write(json.dumps(payload, sort_keys=True) + "\n")


def verify_contract(config_path: Path, authorization_path: Path, installed_unit: Path, boundary_check: bool) -> tuple[dict, dict]:
    config = json.loads(config_path.read_text())
    authorization = json.loads(authorization_path.read_text())
    source_manifest = Path(config["source_manifest_path"])
    source_root = source_manifest.parent / "source"
    expected_sources = json.loads(source_manifest.read_text())["files"]
    observed_sources = {str(path.relative_to(source_root)): sha256(path)
                        for path in sorted(source_root.rglob("*")) if path.is_file()}
    if observed_sources != expected_sources:
        raise RuntimeError("frozen source closure differs")
    required = {
        "status": "BOUNDARY_FIXTURE_ONLY" if boundary_check else "AUTHORIZED_CACHE_MANAGED_REPLACEMENT",
        "runner_sha256": sha256(Path(__file__).resolve()),
        "source_manifest_sha256": sha256(source_manifest),
        "config_sha256": sha256(config_path),
        "service_sha256": sha256(installed_unit),
    }
    if any(authorization.get(key) != value for key, value in required.items()):
        raise RuntimeError("cache-managed replacement differs from direct authorization")
    verified_environment = {}
    for label, frozen in config["environment_files"].items():
        path = Path(frozen["path"])
        if sha256(path) != frozen["sha256"]:
            raise RuntimeError(f"frozen environment file differs: {label}")
        verified_environment[label] = {"path": str(path.resolve()), "sha256": frozen["sha256"]}
    return config, {**required, "authorization_sha256": sha256(authorization_path),
                    "interpreter": str(Path(sys.executable).resolve()),
                    "runner_path": str(Path(__file__).resolve()),
                    "ladder_sorter_path": str(Path(sys.modules["testing.ladder_sorter"].__file__).resolve()),
                    "verified_environment": verified_environment}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--installed-unit", type=Path, required=True)
    parser.add_argument("--boundary-check", action="store_true")
    parser.add_argument("--boundary-receipt", type=Path)
    args = parser.parse_args()
    config, binding = verify_contract(args.config, args.authorization, args.installed_unit, args.boundary_check)

    output = Path(config["output_root"])
    if output.exists():
        raise FileExistsError(output)
    for predecessor in config["preserved_failed_runs"]:
        path = Path(predecessor["path"])
        if sha256(path) != predecessor["sha256"]:
            raise RuntimeError(f"preserved failed-run evidence differs: {path}")

    recording = Path(config["recording_path"])
    manifest_path = recording / "rescue_recording_manifest.json"
    binary = recording / "traces_cached_seg0.raw"
    manifest = json.loads(manifest_path.read_text())
    if sha256(manifest_path) != config["recording_manifest_sha256"] or not manifest.get("complete"):
        raise RuntimeError("completed materialization manifest differs")
    if binary.stat().st_size != config["recording_binary_bytes"]:
        raise RuntimeError("materialized binary size differs")
    if manifest["recording_binary_files"][0]["sha256"] != config["recording_binary_sha256"]:
        raise RuntimeError("materialized binary identity differs")
    if (manifest.get("num_samples") != config["num_samples"]
            or manifest.get("num_channels") != config["num_channels"]
            or manifest.get("sampling_frequency_hz") != config["sampling_frequency_hz"]):
        raise RuntimeError("accepted recording clock/channel extent differs")
    saved_request = json.loads(Path(config["candidate2_saved_request"]).read_text())
    if sha256(Path(config["candidate2_saved_request"])) != config["candidate2_saved_request_sha256"]:
        raise RuntimeError("Candidate2 saved request identity differs")
    requested_params = saved_request.get("sorter_params")
    expected_params = RESCUE_RIGID.params()
    def canonical(value):
        if isinstance(value, float) and (math.isinf(value) or value == float.fromhex("0x1.fffffffffffffp+1023")):
            return "Infinity"
        return value
    if ({key: canonical(value) for key, value in expected_params.items()}
            != {key: canonical(value) for key, value in requested_params.items()}):
        raise RuntimeError("complete native-rigid request differs")
    if args.boundary_check:
        if args.boundary_receipt is None:
            raise RuntimeError("boundary receipt is required")
        atomic_json(args.boundary_receipt, {"status": "PASS_FINAL_ENTRYPOINT_CONFIG_BOUNDARY", "binding": binding,
                    "recording_bytes_read": 0, "sort_started": False})
        return
    validated = validate_accepted_recording(recording, manifest)
    if validated["recording_content_sha256"] != config["recording_content_sha256"]:
        raise RuntimeError("accepted recording content differs")
    reader_mode = config.get("reader_mode", "mapped_fadvise_v1")
    if reader_mode == "bounded_preadv_v1":
        target_environment = "KILOSORT_BOUNDED_PREAD_PATH"
        cache_patch = ensure_kilosort_bounded_reader()
    elif reader_mode == "mapped_fadvise_v1":
        target_environment = "KILOSORT_FADVISE_DONTNEED_PATH"
        cache_patch = ensure_kilosort_cache_compatibility()
    else:
        raise RuntimeError(f"unsupported reader mode: {reader_mode}")
    if os.path.realpath(os.environ.get(target_environment, "")) != str(binary.resolve()):
        raise RuntimeError("cache management is not bound to the frozen recording binary")
    if not cache_patch["applied"]:
        raise RuntimeError("cache compatibility patch is not applied")
    if shutil.disk_usage(output.parent).free < config["minimum_free_bytes"]:
        raise RuntimeError("free-space condition failed")
    before = memory_metrics()
    if before["mem_available_bytes"] < config["minimum_mem_available_bytes"]:
        raise RuntimeError("host available-memory condition failed")
    if before["pressure"]["some"]["avg10"] > config["maximum_prelaunch_memory_pressure_avg10"]:
        raise RuntimeError("prelaunch memory-pressure condition failed")
    with binary.open("rb") as stream:
        os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    time.sleep(5)
    after = memory_metrics()
    if after["pressure"]["some"]["avg10"] > config["maximum_prelaunch_memory_pressure_avg10"]:
        raise RuntimeError("post-eviction memory-pressure condition failed")

    output.mkdir(parents=True)
    atomic_json(output / "CACHE_EVICTION_RECEIPT.json", {
        "schema": "imec1-sort-retry-cache-eviction-v2", "before": before,
        "after": after, "binary": str(binary), "bytes": binary.stat().st_size,
        "cache_patch": cache_patch,
    })
    atomic_json(output / "STATUS.json", {
        "stage": "sorting_cache_managed_retry_full_once",
        "updated_utc": datetime.now(timezone.utc).isoformat(),
    })
    stop = threading.Event()
    watcher = threading.Thread(target=telemetry, args=(stop, output / "MEMORY_TELEMETRY.jsonl"), daemon=True)
    watcher.start()
    started = time.perf_counter()
    try:
        sort_manifest = run_sorter_config(recording, output / "sort", RESCUE_RIGID, requested_params)
    finally:
        stop.set()
        watcher.join(timeout=15)
    elapsed = time.perf_counter() - started
    effective = check_effective_settings("rescue_rigid", sort_manifest)
    if elapsed > config["sort_wall_cap_seconds"]:
        raise RuntimeError("cache-managed replacement exceeded frozen wall condition")
    with binary.open("rb") as stream:
        os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    atomic_json(output / "SORT_COMPLETE.json", {
        "schema": "en-full-session-part2b-imec1-sort-cache-retry-complete-v2",
        "status": "complete_sort_only_downstream_pending",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": sha256(args.config), "binding": binding,
        "recording_manifest_sha256": sha256(manifest_path),
        "sort_seconds": elapsed, "sort": sort_manifest,
        "sort_invocations": 1, "materialization_reused": True,
        "cache_patch": cache_patch,
        "effective_settings": effective,
    })
    atomic_json(output / "STATUS.json", {
        "stage": "sort_complete_downstream_pending",
        "updated_utc": datetime.now(timezone.utc).isoformat(),
    })


if __name__ == "__main__":
    main()
