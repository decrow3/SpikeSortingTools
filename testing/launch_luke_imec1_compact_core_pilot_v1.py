#!/usr/bin/env python
"""Launch the bounded imec1 broad-vs-compact proposal pilot under systemd."""
from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import socket
import subprocess
from pathlib import Path

from testing.luke_full_session_medicine import save, sha
from testing.luke_imec1_compact_core_pilot_v1 import SCHEMA
from testing.luke_imec1_dots_raw_lighthouse_check import MANIFEST

ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/huklab/Documents/DARTsort/.venv/bin/python")
FILES = [
    "testing/__init__.py",
    "testing/managed_job.py",
    "testing/luke_imec1_dots_lighthouse_direct_check.py",
    "testing/luke_imec1_dots_raw_lighthouse_check.py",
    "testing/luke_imec1_dots_sorterfree_waveform_discovery.py",
    "testing/luke_imec1_sorterfree_phase_audit_v1.py",
    "testing/luke_imec1_dots_sorterfree_waveform_discovery_v3.py",
    "testing/luke_imec1_compact_core_pilot_v1.py",
]
SOURCE_BANK = ROOT / "testing/outputs/luke_imec1_dots_sorterfree_waveform_discovery_v3_s300"
EXTERNAL_BANK = ROOT / "testing/outputs/luke_imec1_dots_sorterfree_waveform_discovery_v3"
BANK_FILES = ["summary.json", "seed_cluster_events.csv", "families_after_depth_reveal.csv", "family_templates.npz"]


def valid_proof(path: Path) -> bool:
    try:
        receipt = json.loads((path / "receipt.json").read_text())
        disconnected = json.loads((path / "launcher_disconnected_state.json").read_text())
        result = json.loads((path / "service_result.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return False
    return (
        receipt.get("state") == "complete"
        and receipt.get("returncode") == 0
        and disconnected.get("active_after_launcher_exit") is True
        and result.get("SERVICE_RESULT") == "success"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit", required=True)
    parser.add_argument("--job-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--proof", type=Path, required=True)
    args = parser.parse_args()
    if args.job_dir.exists() or args.output.exists():
        raise RuntimeError("job/output exists; preserve evidence")
    if not valid_proof(args.proof):
        raise RuntimeError("valid launcher-disconnection proof required")

    job, output = args.job_dir.resolve(), args.output.resolve()
    job.mkdir(parents=True)
    bundle = job / "source"
    for name in FILES:
        target = bundle / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    save(job / "source_manifest.json", {name: sha(bundle / name) for name in FILES})

    manifest = json.loads(MANIFEST.read_text())
    binary = Path(manifest["binary_path"])
    stat = binary.stat()
    config = {
        "schema": SCHEMA,
        "output": str(output),
        "authorization": "User requested applying the old imec0 lighthouse lessons to improve imec1 proposals",
        "sorter_inputs": [],
        "motion_inputs": [],
        "source_bank": str(SOURCE_BANK.resolve()),
        "external_bank": str(EXTERNAL_BANK.resolve()),
        "cached_input_sha256": {
            str(path.resolve()): sha(path)
            for bank in (SOURCE_BANK, EXTERNAL_BANK)
            for name in BANK_FILES
            for path in [bank / name]
            if path.exists()
        },
        "raw_binary": str(binary),
        "raw_size": stat.st_size,
        "raw_mtime_ns": stat.st_mtime_ns,
        "train_interval_s": [300.0, 310.0],
        "qualification_interval_s": [310.0, 320.0],
        "method_contract": "Hold v3 assignments fixed; compare all-member templates with timing-aligned reciprocal compact cores; qualify both against the same s036 external bank.",
        "checkpoint": "This is one bounded 20 s stage; interruption requires preserving evidence and restarting under a new output version.",
        "no_automatic_restart": True,
    }
    save(job / "config.json", config)
    command = [
        str(PYTHON), "-u", "-m", "testing.managed_job",
        "--receipt", str(job / "receipt.json"), "--cwd", str(bundle), "--",
        str(PYTHON), "-u", "-m", "testing.luke_imec1_compact_core_pilot_v1",
        "--config", str(job / "config.json"),
    ]
    script = job / "launch.sh"
    script.write_text(
        "#!/bin/bash\nset -eu\ncd " + shlex.quote(str(bundle))
        + "\nexport OPENBLAS_NUM_THREADS=6 OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 NUMEXPR_NUM_THREADS=6\n"
        + "exec " + shlex.join(command) + "\n"
    )
    script.chmod(0o755)
    finish = job / "save_service_result.py"
    finish.write_text(
        "import json,os\nfrom pathlib import Path\n"
        "p=Path(__file__).with_name('service_result.json');q=p.with_suffix('.tmp');"
        "q.write_text(json.dumps({k:os.environ.get(k) for k in "
        "['SERVICE_RESULT','EXIT_CODE','EXIT_STATUS']}));os.replace(q,p)\n"
    )
    request = [
        "systemd-run", "--user", "--unit=" + args.unit, "--collect",
        "--property=Type=exec", "--property=CPUQuota=800%", "--property=MemoryMax=64G",
        "--property=TasksMax=160", "--property=KillMode=control-group",
        "--property=TimeoutStopSec=30",
        "--property=StandardOutput=append:" + str(job / "stdout.log"),
        "--property=StandardError=append:" + str(job / "stderr.log"),
        "--property=ExecStopPost=" + str(PYTHON) + " " + str(finish), str(script),
    ]
    save(job / "launch_request.json", {
        "command": request, "child_command": command, "unit": args.unit,
        "host": socket.gethostname(), "launcher_pid": os.getpid(),
        "proof": str(args.proof.resolve()),
    })
    result = subprocess.run(request, capture_output=True, text=True)
    save(job / "launcher_result.json", {
        "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr,
    })
    print(result.stdout + result.stderr, flush=True)
    result.check_returncode()
    shown = subprocess.check_output([
        "systemctl", "--user", "show", args.unit + ".service",
        "-p", "ActiveState", "-p", "SubState", "-p", "MainPID", "-p", "Result",
    ], text=True)
    state = dict(line.split("=", 1) for line in shown.splitlines() if "=" in line)
    save(job / "launcher_disconnected_state.json", {
        "active_after_launcher_exit": state.get("ActiveState") == "active"
        and state.get("SubState") == "running" and int(state.get("MainPID", "0")) > 0,
        **state,
    })


if __name__ == "__main__":
    main()
