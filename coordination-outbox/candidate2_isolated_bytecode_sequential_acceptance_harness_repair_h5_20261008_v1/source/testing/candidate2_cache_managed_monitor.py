"""Out-of-process systemd/cgroup telemetry for the Candidate2 run."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


PROPERTIES = (
    "LoadState", "ActiveState", "SubState", "Result", "InvocationID",
    "ControlGroup", "ControlPID", "ExecMainPID", "ExecMainCode",
    "ExecMainStatus", "NRestarts", "MemoryCurrent", "MemoryPeak",
    "MemorySwapCurrent", "CPUUsageNSec",
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


def systemctl_show(unit: str) -> dict[str, str]:
    proc = subprocess.run(
        ["systemctl", "--user", "show", unit, *[f"--property={p}" for p in PROPERTIES]],
        text=True, capture_output=True, timeout=3, check=False,
    )
    values = {"returncode": str(proc.returncode)}
    for line in proc.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    return values


def parse_kv(path: Path) -> dict:
    result = {}
    try:
        for line in path.read_text().splitlines():
            key, value = line.split(maxsplit=1)
            result[key] = int(value)
    except (FileNotFoundError, PermissionError, ValueError):
        return {}
    return result


def parse_pressure(path: Path) -> dict:
    result = {}
    try:
        for line in path.read_text().splitlines():
            fields = line.split()
            result[fields[0]] = {
                key: (int(value) if key == "total" else float(value))
                for key, value in (part.split("=", 1) for part in fields[1:])
            }
    except (FileNotFoundError, PermissionError, ValueError, IndexError):
        return {}
    return result


def cgroup_snapshot(control_group: str) -> dict:
    if not control_group:
        return {}
    root = Path("/sys/fs/cgroup") / control_group.lstrip("/")
    value = {"path": str(root)}
    for name in ("memory.current", "memory.peak", "memory.swap.current"):
        try:
            value[name] = int((root / name).read_text().strip())
        except (FileNotFoundError, PermissionError, ValueError):
            value[name] = None
    value["memory.events"] = parse_kv(root / "memory.events")
    value["memory.stat"] = parse_kv(root / "memory.stat")
    value["memory.pressure"] = parse_pressure(root / "memory.pressure")
    pids = set()
    if root.exists():
        for child in root.rglob("cgroup.procs"):
            try:
                pids.update(int(item) for item in child.read_text().split())
            except (FileNotFoundError, PermissionError, ValueError):
                pass
    value["pids"] = sorted(pids)
    return value


def process_snapshot(pid: int) -> dict:
    value = {"pid": pid}
    root = Path("/proc") / str(pid)
    try:
        value["comm"] = (root / "comm").read_text().strip()
        value["cmdline"] = (root / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")[:2048]
        for line in (root / "status").read_text().splitlines():
            if line.startswith(("VmRSS:", "VmSize:", "VmSwap:")):
                key, raw = line.split(":", 1)
                value[key + "_bytes"] = int(raw.split()[0]) * 1024
    except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError, IndexError):
        value["unavailable"] = True
    return value


def host_memory() -> dict:
    value = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, raw = line.split(":", 1)
        if key in {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree", "Cached", "SReclaimable"}:
            value[key + "_bytes"] = int(raw.split()[0]) * 1024
    return value


def gpu_snapshot() -> dict:
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.total,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"],
        text=True, capture_output=True, timeout=3, check=False,
    )
    apps = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"],
        text=True, capture_output=True, timeout=3, check=False,
    )
    return {
        "gpu_returncode": gpu.returncode,
        "gpu_rows": [row.strip() for row in gpu.stdout.splitlines() if row.strip()],
        "compute_returncode": apps.returncode,
        "compute_rows": [row.strip() for row in apps.stdout.splitlines() if row.strip()],
    }


def log_tail(path: Path) -> list[str]:
    try:
        return path.read_text(errors="replace").splitlines()[-20:]
    except (FileNotFoundError, PermissionError, OSError):
        return []


def sample(unit: str, log: Path, index: int) -> dict:
    target = systemctl_show(unit)
    target_cgroup = cgroup_snapshot(target.get("ControlGroup", ""))
    pids = target_cgroup.get("pids", [])
    return {
        "sample_index": index,
        "utc": datetime.now(timezone.utc).isoformat(),
        "unix_time": time.time(),
        "target": target,
        "target_cgroup": target_cgroup,
        "target_processes": [process_snapshot(pid) for pid in pids],
        "user_service_cgroup": cgroup_snapshot("/user.slice/user-1000.slice/user@1000.service"),
        "user_slice_cgroup": cgroup_snapshot("/user.slice/user-1000.slice"),
        "host_memory": host_memory(),
        "host_pressure": parse_pressure(Path("/proc/pressure/memory")),
        "gpu": gpu_snapshot(),
        "service_log_tail": log_tail(log),
    }


def observed(target: dict) -> bool:
    if target.get("ActiveState") in {"activating", "active", "deactivating", "failed"}:
        return True
    return any(int(target.get(key, "0") or "0") > 0 for key in ("ControlPID", "ExecMainPID"))


def terminal(target: dict, seen: bool) -> bool:
    return seen and target.get("ActiveState") in {"inactive", "failed"}


def ready_failures(value: dict) -> list[str]:
    failures = []
    target = value["target"]
    if target.get("returncode") != "0":
        failures.append("systemctl_query")
    if target.get("LoadState") != "loaded":
        failures.append("target_not_loaded")
    if target.get("ActiveState") not in {"activating", "active"}:
        failures.append("target_not_started")
    if not target.get("InvocationID") or not target.get("ControlGroup"):
        failures.append("target_identity_or_cgroup")
    if not value["target_cgroup"].get("path"):
        failures.append("target_cgroup_snapshot")
    if value["gpu"].get("gpu_returncode") != 0 or not value["gpu"].get("gpu_rows"):
        failures.append("gpu_query")
    if value["gpu"].get("compute_returncode") != 0:
        failures.append("gpu_compute_query")
    if "MemAvailable_bytes" not in value["host_memory"] or not value["host_pressure"]:
        failures.append("host_memory_or_pressure")
    return failures


def classify(target: dict, memory_events: dict) -> str:
    if target.get("Result") == "success" and target.get("ExecMainStatus") == "0":
        return "success"
    if memory_events.get("oom_kill", 0) > 0 or target.get("Result") == "oom-kill":
        return "oomd_or_kernel_oom"
    if target.get("Result") in {"signal", "core-dump"}:
        return "external_or_signal_termination"
    if target.get("Result") == "exit-code":
        return "managed_job_failure_or_preflight_refusal"
    return "terminal_unclassified"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-sha256", required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--contract-sha256", required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--target-unit", required=True)
    parser.add_argument("--target-unit-file", type=Path, required=True)
    parser.add_argument("--monitor-unit-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--service-log", type=Path, required=True)
    args = parser.parse_args()
    if sha256(Path(__file__).resolve()) != args.self_sha256:
        raise RuntimeError("monitor self hash mismatch")
    if sha256(args.contract) != args.contract_sha256:
        raise RuntimeError("contract hash mismatch")
    authorization = json.loads(args.authorization.read_text())
    required = {
        "status": "AUTHORIZED_ONE_FRESH_CANDIDATE2_CACHE_MANAGED_RUN",
        "contract_sha256": args.contract_sha256,
        "target_service_sha256": sha256(args.target_unit_file),
        "monitor_service_sha256": sha256(args.monitor_unit_file),
    }
    if any(authorization.get(key) != expected for key, expected in required.items()):
        raise RuntimeError("monitor authorization binding differs")
    if args.output.exists():
        raise RuntimeError("fresh monitor output already exists")
    args.output.mkdir(parents=False)
    telemetry = args.output / "TELEMETRY.jsonl"
    seen = False
    post_terminal = 0
    final = None
    ready_written = False
    started = time.monotonic()
    with telemetry.open("x", buffering=1) as stream:
        index = 0
        while True:
            tick = time.monotonic()
            value = sample(args.target_unit, args.service_log, index)
            seen = seen or observed(value["target"])
            failures = ready_failures(value)
            value["ready_failures"] = failures
            stream.write(json.dumps(value, sort_keys=True) + "\n")
            final = value
            if not ready_written and not failures:
                atomic_json(args.output / "READY.json", {
                    "schema": "candidate2-external-monitor-ready-v1",
                    "contract_sha256": args.contract_sha256,
                    "target_service_sha256": required["target_service_sha256"],
                    "monitor_service_sha256": required["monitor_service_sha256"],
                    "first_sample_unix": value["unix_time"],
                    "sample_index": value["sample_index"],
                    "first_sample_complete": True,
                    "systemctl_returncode": value["target"]["returncode"],
                    "target_load_state": value["target"]["LoadState"],
                    "target_active_state": value["target"]["ActiveState"],
                    "target_invocation_id": value["target"]["InvocationID"],
                    "target_control_group": value["target"]["ControlGroup"],
                    "gpu_returncode": value["gpu"]["gpu_returncode"],
                    "gpu_compute_returncode": value["gpu"]["compute_returncode"],
                    "ready_failures": [],
                })
                ready_written = True
            if not ready_written and time.monotonic() - started > 30:
                atomic_json(args.output / "FAILURE.json", {
                    "status": "FIRST_COMPLETE_SAMPLE_NOT_OBSERVED",
                    "last_failures": failures,
                    "last_sample_index": index,
                })
                raise RuntimeError("first complete telemetry sample not observed")
            if terminal(value["target"], seen):
                post_terminal += 1
                if post_terminal >= 3:
                    break
            else:
                post_terminal = 0
            index += 1
            interval = 10.0 if ready_written else 1.0
            delay = interval - (time.monotonic() - tick)
            if delay > 0:
                time.sleep(delay)
    assert final is not None
    target = final["target"]
    classification = classify(target, final["target_cgroup"].get("memory.events", {}))
    atomic_json(args.output / "TERMINAL.json", {
        "schema": "candidate2-external-monitor-terminal-v1",
        "classification": classification,
        "samples": final["sample_index"] + 1,
        "telemetry_sha256": sha256(telemetry),
        "final_target": target,
        "final_memory_events": final["target_cgroup"].get("memory.events", {}),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    })


if __name__ == "__main__":
    main()
