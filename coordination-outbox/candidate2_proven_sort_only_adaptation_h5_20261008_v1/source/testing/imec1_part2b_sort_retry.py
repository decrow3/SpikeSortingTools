#!/usr/bin/env python3
"""One authorized sort-only retry using the preserved verified materialization."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from pipeline.sorting import run_kilosort4
from testing.imec1_part2b_contract import atomic_json, sha256


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


def cgroup_path() -> Path:
    row = next(line for line in Path("/proc/self/cgroup").read_text().splitlines() if line.startswith("0::"))
    return Path("/sys/fs/cgroup") / row.split("::", 1)[1].lstrip("/")


def telemetry(stop: threading.Event, path: Path) -> None:
    cg = cgroup_path()
    while not stop.wait(10):
        payload = {"utc": datetime.now(timezone.utc).isoformat(), **memory_metrics()}
        for name in ("memory.current", "memory.peak", "memory.events", "memory.pressure"):
            target = cg / name
            payload[name] = target.read_text().strip() if target.exists() else None
        with path.open("a") as stream:
            stream.write(json.dumps(payload, sort_keys=True) + "\n")


def verify_contract(config_path: Path, review_path: Path, installed_unit: Path) -> tuple[dict, dict]:
    config = json.loads(config_path.read_text())
    review = json.loads(review_path.read_text())
    repo = Path(__file__).resolve().parents[1]
    head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain=v1", "--untracked-files=all"], text=True)
    required = {
        "status": "PASS_FOR_ONE_REPLACEMENT_SORT",
        "source_commit": head,
        "config_sha256": sha256(config_path),
        "service_sha256": sha256(installed_unit),
    }
    if dirty or any(review.get(k) != v for k, v in required.items()):
        raise RuntimeError("replacement-sort runtime differs from independent review")
    if str(Path(config["output_root"])) == config["failed_output_root"]:
        raise RuntimeError("retry must use a new output namespace")
    return config, {**required, "review_sha256": sha256(review_path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--installed-unit", type=Path, required=True)
    args = parser.parse_args()
    config, binding = verify_contract(args.config, args.review, args.installed_unit)
    output = Path(config["output_root"])
    if output.exists():
        raise FileExistsError(output)
    failed = Path(config["failed_output_root"])
    status = failed / "STATUS.json"
    if sha256(status) != config["failed_status_sha256"] or json.loads(status.read_text()).get("stage") != "sorting_full_once":
        raise RuntimeError("preserved failed-run status differs")
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
    if sha256(Path(config["q0_receipt_path"])) != config["q0_receipt_sha256"]:
        raise RuntimeError("q0 receipt differs")
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
    atomic_json(output / "CACHE_EVICTION_RECEIPT.json", {"schema": "imec1-sort-retry-cache-eviction-v1", "before": before, "after": after, "binary": str(binary), "bytes": binary.stat().st_size})
    atomic_json(output / "STATUS.json", {"stage": "sorting_retry_full_once", "updated_utc": datetime.now(timezone.utc).isoformat()})
    stop = threading.Event()
    watcher = threading.Thread(target=telemetry, args=(stop, output / "MEMORY_TELEMETRY.jsonl"), daemon=True)
    watcher.start()
    started = time.perf_counter()
    try:
        sort_manifest = run_kilosort4(recording, output / "sort")
    finally:
        stop.set()
        watcher.join(timeout=15)
    elapsed = time.perf_counter() - started
    if elapsed > config["sort_wall_cap_seconds"]:
        raise RuntimeError("replacement sort exceeded frozen wall condition")
    atomic_json(output / "SORT_COMPLETE.json", {"schema": "en-full-session-part2b-imec1-sort-retry-complete-v1", "status": "complete_sort_only_downstream_pending", "completed_utc": datetime.now(timezone.utc).isoformat(), "config_sha256": sha256(args.config), "binding": binding, "recording_manifest_sha256": sha256(manifest_path), "sort_seconds": elapsed, "sort": sort_manifest, "sort_invocations": 1, "materialization_reused": True})
    atomic_json(output / "STATUS.json", {"stage": "sort_complete_downstream_pending", "updated_utc": datetime.now(timezone.utc).isoformat()})


if __name__ == "__main__":
    main()
