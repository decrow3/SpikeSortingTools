#!/usr/bin/env python3
"""Managed, fail-closed launcher for the Candidate-3 synthetic transition arm."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def tree_file_bytes(root: Path) -> int:
    if not root.exists():
        return 0
    total = 0
    for directory, _, names in os.walk(root, followlinks=False):
        for name in names:
            path = Path(directory) / name
            try:
                if not path.is_symlink():
                    total += path.stat().st_size
            except FileNotFoundError:
                continue
    return total


def verify_tree(root: Path, manifest: dict, name: str) -> int:
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError(f"{name} root invalid")
    found = {}
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        base = Path(directory)
        for dirname in dirnames:
            if (base / dirname).is_symlink():
                raise RuntimeError(f"{name} contains symlink directory")
        for filename in filenames:
            path = base / filename
            if path.is_symlink() or not path.is_file():
                raise RuntimeError(f"{name} contains invalid file")
            found[str(path.relative_to(root))] = path
    if set(found) != set(manifest):
        missing = sorted(set(manifest) - set(found))
        extra = sorted(set(found) - set(manifest))
        raise RuntimeError(f"{name} membership mismatch missing={missing} extra={extra}")
    total = 0
    for relative, binding in manifest.items():
        path = found[relative]
        size = path.stat().st_size
        if size != binding["size"] or sha256(path) != binding["sha256"]:
            raise RuntimeError(f"{name} binding mismatch: {relative}")
        total += size
    return total


def load_contract(path: Path, expected_sha256: str) -> dict:
    if sha256(path) != expected_sha256:
        raise RuntimeError("run contract hash mismatch")
    contract = json.loads(path.read_text())
    if contract.get("schema") != "candidate3-mini-transition-managed-v2":
        raise ValueError("unexpected contract schema")
    return contract


def service_properties(unit: str, names: list[str]) -> dict:
    output = subprocess.check_output(
        ["systemctl", "--user", "show", unit, "--property=" + ",".join(names)], text=True
    )
    return dict(line.split("=", 1) for line in output.splitlines() if "=" in line)


def verify_manager(contract: dict) -> dict:
    unit = contract["service_unit"]
    cgroup = Path("/proc/self/cgroup").read_text()
    if unit not in cgroup:
        raise RuntimeError("runner is not inside expected systemd unit")
    invocation = os.environ.get("INVOCATION_ID")
    if not invocation:
        raise RuntimeError("missing systemd invocation id")
    expected = contract["manager_properties"]
    actual = service_properties(unit, list(expected) + ["InvocationID", "MainPID", "ActiveState", "SubState"])
    for key, value in expected.items():
        if actual.get(key) != value:
            raise RuntimeError(f"manager property mismatch: {key}={actual.get(key)!r}")
    if actual.get("InvocationID") != invocation:
        raise RuntimeError("manager invocation id mismatch")
    if actual.get("ActiveState") not in {"active", "activating"}:
        raise RuntimeError("manager unit is not active")
    # Configured properties are not sufficient evidence of effective cgroup
    # enforcement. Bind the live leaf files and CPU policy from inside the unit.
    unified = [line.split("::", 1)[1] for line in cgroup.splitlines() if "::" in line]
    if len(unified) != 1:
        raise RuntimeError("unexpected cgroup v2 membership")
    leaf = Path("/sys/fs/cgroup") / unified[0].lstrip("/")
    memory_max = (leaf / "memory.max").read_text().strip()
    pids_max = (leaf / "pids.max").read_text().strip()
    effective = contract["effective_resource_policy"]
    if memory_max != str(effective["memory_max_bytes"]):
        raise RuntimeError(f"effective memory.max mismatch: {memory_max!r}")
    if pids_max != str(effective["pids_max"]):
        raise RuntimeError(f"effective pids.max mismatch: {pids_max!r}")
    cpu_max_path = leaf / "cpu.max"
    cpu_max = cpu_max_path.read_text().strip() if cpu_max_path.is_file() else None
    if cpu_max is not None and not cpu_max.startswith("max "):
        raise RuntimeError(f"unexpected enforced CPU quota: {cpu_max!r}")
    allowed_cpus = sorted(os.sched_getaffinity(0))
    if len(allowed_cpus) < int(effective["minimum_allowed_cpu_count"]):
        raise RuntimeError("insufficient allowed CPUs")
    meminfo = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        meminfo[key] = int(value.strip().split()[0]) * 1024
    if meminfo["MemAvailable"] < int(effective["minimum_mem_available_bytes"]):
        raise RuntimeError("insufficient host memory headroom")
    actual["EffectiveCgroupPath"] = str(leaf)
    actual["EffectiveMemoryMax"] = memory_max
    actual["EffectivePidsMax"] = pids_max
    actual["EffectiveCpuMax"] = cpu_max
    actual["CpuPolicy"] = "NO_ENFORCED_QUOTA"
    actual["AllowedCpus"] = allowed_cpus
    actual["OnlineCpus"] = Path("/sys/devices/system/cpu/online").read_text().strip()
    actual["OnlineCpuCount"] = os.cpu_count()
    actual["MemAvailableBytes"] = meminfo["MemAvailable"]
    actual["LoadAverage"] = list(os.getloadavg())
    return actual


def render_command(values: list[str], *, packet: Path, contract_sha256: str) -> list[str]:
    return [
        value.format(packet=str(packet), contract_sha256=contract_sha256)
        for value in values
    ]


def launch(contract_path: Path, contract_sha256: str) -> int:
    contract = load_contract(contract_path, contract_sha256)
    packet = contract_path.parent.resolve(strict=True)
    outer = Path(contract["outer_launch_receipt"])
    for key in (
        "attempt_root", "preflight_failure_receipt", "preflight_manager_receipt", "outer_launch_receipt"
    ):
        path = Path(contract[key])
        if path.exists() or path.is_symlink():
            raise RuntimeError(f"launch freshness failure: {path}")
    state = service_properties(contract["service_unit"], ["LoadState", "ActiveState", "SubState"])
    if state != {"LoadState": "not-found", "ActiveState": "inactive", "SubState": "dead"}:
        raise RuntimeError(f"service unit is not fresh: {state}")
    command = render_command(contract["systemd_argv"], packet=packet, contract_sha256=contract_sha256)
    receipt = {
        "state": "launching",
        "started_at": now(),
        "launcher_pid": os.getpid(),
        "contract": str(contract_path),
        "contract_sha256": contract_sha256,
        "command": command,
        "unit": contract["service_unit"],
    }
    atomic_json(outer, receipt)
    result = subprocess.run(command, text=True, capture_output=True)
    receipt.update(
        state="started" if result.returncode == 0 else "launch_failed",
        returncode=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
        finished_at=now(),
    )
    atomic_json(outer, receipt)
    return result.returncode


def verify_preflight(contract_path: Path, contract: dict, *, require_fresh_attempt: bool = True) -> dict:
    packet = contract_path.parent.resolve(strict=True)
    if contract_path.is_symlink() or not contract_path.is_file():
        raise RuntimeError("contract must be a regular nonsymlink file")
    checks = {}
    for relative, expected in contract["packet_hashes"].items():
        path = packet / relative
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"packet member invalid: {relative}")
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"packet hash mismatch: {relative}")
        checks[str(path)] = actual
    input_bytes = verify_tree(
        Path(contract["input_root"]), contract["tree_manifests"]["inputs"], "inputs"
    )
    ordinary_bytes = verify_tree(
        Path(contract["ordinary_root"]), contract["tree_manifests"]["ordinary"], "ordinary"
    )
    if input_bytes + ordinary_bytes != contract["baseline_bytes"]:
        raise RuntimeError("baseline byte accounting mismatch")
    commit = subprocess.check_output(
        ["git", "-C", contract["dartsort_repo"], "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != contract["dartsort_commit"]:
        raise RuntimeError("DARTsort commit mismatch")
    dirty = subprocess.check_output(
        ["git", "-C", contract["dartsort_repo"], "status", "--porcelain=v1", "--untracked-files=all"],
        text=True,
    )
    if dirty:
        raise RuntimeError("DARTsort worktree is dirty")
    root = Path(contract["attempt_root"])
    failure = Path(contract["preflight_failure_receipt"])
    manager_failure = Path(contract["preflight_manager_receipt"])
    for path in (failure, manager_failure):
        if path.exists() or path.is_symlink():
            raise RuntimeError(f"freshness failure: {path}")
    if require_fresh_attempt:
        if root.exists() or root.is_symlink():
            raise RuntimeError(f"freshness failure: {root}")
    elif root.is_symlink() or not root.is_dir():
        raise RuntimeError("active attempt root invalid")
    parent = root.parent.resolve(strict=True)
    expected_parent = Path(contract["attempt_parent"]).resolve(strict=True)
    if root.resolve(strict=False).parent != parent or parent != expected_parent:
        raise RuntimeError("attempt root containment failure")
    free = os.statvfs(parent).f_bavail * os.statvfs(parent).f_frsize
    if free < contract["minimum_free_bytes"]:
        raise RuntimeError("insufficient free storage")
    return {
        "hashes": checks,
        "dartsort_commit": commit,
        "dartsort_clean": True,
        "input_bytes": input_bytes,
        "ordinary_bytes": ordinary_bytes,
        "baseline_bytes": input_bytes + ordinary_bytes,
        "free_bytes": free,
    }


def terminate_group(process: subprocess.Popen, grace_seconds: int = 30) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def run_child(command: list[str], *, cwd: Path, env: dict, stdout: Path, stderr: Path,
              root: Path, baseline_bytes: int, cap_bytes: int) -> tuple[int, str | None]:
    stop_reason = None
    signal_number = None

    def handle(signum, _frame):
        nonlocal signal_number
        signal_number = signum

    old_term = signal.signal(signal.SIGTERM, handle)
    old_int = signal.signal(signal.SIGINT, handle)
    try:
        with stdout.open("ab", buffering=0) as out, stderr.open("ab", buffering=0) as err:
            process = subprocess.Popen(
                command, cwd=cwd, env=env, stdout=out, stderr=err, start_new_session=True
            )
            while process.poll() is None:
                used = baseline_bytes + tree_file_bytes(root)
                if used > cap_bytes:
                    stop_reason = f"storage_cap_exceeded:{used}>{cap_bytes}"
                    terminate_group(process)
                    break
                if signal_number is not None:
                    stop_reason = f"manager_signal:{signal_number}"
                    terminate_group(process)
                    break
                time.sleep(2)
            return process.wait(), stop_reason
    finally:
        signal.signal(signal.SIGTERM, old_term)
        signal.signal(signal.SIGINT, old_int)


def run(contract_path: Path, contract_sha256: str) -> int:
    contract = load_contract(contract_path, contract_sha256)
    try:
        preflight = verify_preflight(contract_path, contract)
        outer = json.loads(Path(contract["outer_launch_receipt"]).read_text())
        if outer.get("state") not in {"launching", "started"}:
            raise RuntimeError("outer launch receipt state invalid")
        if outer.get("contract_sha256") != contract_sha256:
            raise RuntimeError("outer launch receipt contract mismatch")
        manager = verify_manager(contract)
    except BaseException as error:
        failure = Path(contract["preflight_failure_receipt"])
        if not failure.exists() and not failure.is_symlink():
            atomic_json(failure, {"state": "preflight_failed", "error": repr(error)})
        raise

    packet = contract_path.parent.resolve(strict=True)
    root = Path(contract["attempt_root"])
    root.mkdir(mode=0o700, exist_ok=False)
    job = root / "job"
    cache = root / "cache"
    runs = root / "runs"
    report = root / "report"
    combined = root / "combined"
    for path in (job, cache, runs, report, combined):
        path.mkdir()
    transition = runs / "transition"
    prelaunch = {
        "state": "running",
        "started_at": now(),
        "wrapper_pid": os.getpid(),
        "contract": str(contract_path),
        "preflight": preflight,
        "manager": manager,
        "transition_command": contract["transition_command"],
        "report_command": contract["report_command"],
        "cap_bytes": contract["cap_bytes"],
        "baseline_bytes": contract["baseline_bytes"],
    }
    atomic_json(job / "PRELAUNCH.json", prelaunch)
    env = os.environ.copy()
    env["NUMBA_CACHE_DIR"] = str(cache)
    transition_command = [
        value.format(packet=str(packet), transition=str(transition))
        for value in contract["transition_command"]
    ]
    code, reason = run_child(
        transition_command,
        cwd=Path(contract["cwd"]),
        env=env,
        stdout=job / "transition.stdout.log",
        stderr=job / "transition.stderr.log",
        root=root,
        baseline_bytes=contract["baseline_bytes"],
        cap_bytes=contract["cap_bytes"],
    )
    if code or reason:
        atomic_json(job / "FINAL_STATUS.json", {
            "state": "failed", "stage": "transition", "returncode": code, "reason": reason,
            "finished_at": now(),
        })
        return code or 70

    try:
        before_report = verify_preflight(contract_path, contract, require_fresh_attempt=False)
    except BaseException as error:
        atomic_json(job / "FINAL_STATUS.json", {
            "state": "failed", "stage": "pre_report_validation", "error": repr(error),
            "finished_at": now(),
        })
        return 72
    (combined / "ordinary").symlink_to(Path(contract["ordinary_root"]), target_is_directory=True)
    (combined / "transition").symlink_to(transition, target_is_directory=True)
    report_command = [
        value.format(packet=str(packet), combined=str(combined), report=str(report))
        for value in contract["report_command"]
    ]
    code, reason = run_child(
        report_command,
        cwd=Path(contract["cwd"]),
        env=env,
        stdout=job / "report.stdout.log",
        stderr=job / "report.stderr.log",
        root=root,
        baseline_bytes=contract["baseline_bytes"],
        cap_bytes=contract["cap_bytes"],
    )
    used = contract["baseline_bytes"] + tree_file_bytes(root)
    try:
        after_report = verify_preflight(contract_path, contract, require_fresh_attempt=False)
    except BaseException as error:
        atomic_json(job / "FINAL_STATUS.json", {
            "state": "failed", "stage": "post_report_validation", "error": repr(error),
            "finished_at": now(),
        })
        return 73
    state = "complete" if code == 0 and reason is None and used <= contract["cap_bytes"] else "failed"
    atomic_json(job / "FINAL_STATUS.json", {
        "state": state,
        "stage": "complete" if state == "complete" else "report",
        "returncode": code,
        "reason": reason,
        "finished_at": now(),
        "combined_bytes": used,
        "cap_bytes": contract["cap_bytes"],
        "accounting_json": str(report / "ACCOUNTING.json"),
        "accounting_npz": str(report / "ACCOUNTING.npz"),
        "before_report_preflight": before_report,
        "after_report_preflight": after_report,
    })
    return 0 if state == "complete" else (code or 71)


def finalize(contract_path: Path, contract_sha256: str) -> int:
    contract = load_contract(contract_path, contract_sha256)
    root = Path(contract["attempt_root"])
    target = root / "job" / "MANAGER_FINAL.json" if root.is_dir() else Path(contract["preflight_manager_receipt"])
    value = {
        "state": "manager_final",
        "finished_at": now(),
        "service_result": os.environ.get("SERVICE_RESULT"),
        "exit_code": os.environ.get("EXIT_CODE"),
        "exit_status": os.environ.get("EXIT_STATUS"),
        "final_status_present": (root / "job" / "FINAL_STATUS.json").is_file(),
    }
    try:
        value["manager"] = service_properties(
            contract["service_unit"],
            [
                "LoadState", "ActiveState", "SubState", "MainPID", "ExecMainStatus",
                "ExecMainCode", "Result", "NRestarts", "ExecMainStartTimestamp",
                "ExecMainExitTimestamp", "MemoryCurrent", "MemoryPeak", "CPUUsageNSec",
                "InvocationID",
            ],
        )
    except BaseException as error:
        value["manager_query_error"] = repr(error)
    if target.exists() or target.is_symlink():
        raise RuntimeError(f"refusing manager-final overwrite: {target}")
    atomic_json(target, value)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("launch", "run", "finalize"))
    parser.add_argument("contract", type=Path)
    parser.add_argument("contract_sha256")
    args = parser.parse_args(argv)
    if args.mode == "launch":
        return launch(args.contract, args.contract_sha256)
    if args.mode == "run":
        return run(args.contract, args.contract_sha256)
    return finalize(args.contract, args.contract_sha256)


if __name__ == "__main__":
    sys.exit(main())
