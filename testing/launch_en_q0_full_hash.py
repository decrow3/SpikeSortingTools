#!/usr/bin/env python3
"""Launch EN's full accepted-binary hash gate as a durable user service."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / "environments/rescue-production/.venv/bin/python"


def write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit", default="en-q0-full-hash-20260929-v1.service")
    parser.add_argument("--job-dir", type=Path, default=ROOT / "testing/outputs/en_q0_full_hash_job_v1")
    parser.add_argument("--output", type=Path, default=ROOT / "testing/outputs/en_q0_full_hash_v1")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/en_rounded_field_ks129.v1.json")
    args = parser.parse_args()
    args.job_dir = args.job_dir.resolve()
    args.output = args.output.resolve()
    args.config = args.config.resolve()
    if args.job_dir.exists() or args.output.exists():
        raise FileExistsError("q0 job or output already exists")
    args.job_dir.mkdir(parents=True)
    command = [str(PYTHON), "-m", "testing.en_q0_full_hash", "--config", str(args.config), "--output", str(args.output)]
    shell = args.job_dir / "launch.sh"
    shell.write_text("#!/bin/bash\nset -eu\ncd " + shlex.quote(str(ROOT)) + "\nexec " + shlex.join(command) + "\n")
    shell.chmod(0o755)
    finish = args.job_dir / "save_service_result.py"
    finish.write_text(
        "import json,os\nfrom datetime import datetime,timezone\nfrom pathlib import Path\n"
        "p=Path(__file__).with_name('service_result.json');q=p.with_suffix('.partial')\n"
        "q.write_text(json.dumps({'finished_at':datetime.now(timezone.utc).isoformat(),"
        "**{k:os.environ.get(k) for k in ['SERVICE_RESULT','EXIT_CODE','EXIT_STATUS']}},indent=2)+'\\n')\n"
        "os.replace(q,p)\n"
    )
    request = [
        "systemd-run", "--user", f"--unit={args.unit}", "--collect",
        "--property=Type=exec", "--property=CPUQuota=100%", "--property=MemoryMax=2G",
        "--property=TasksMax=32", "--property=KillMode=control-group", "--property=TimeoutStopSec=30",
        "--property=ManagedOOMPreference=avoid",
        f"--property=StandardOutput=append:{args.job_dir / 'stdout.log'}",
        f"--property=StandardError=append:{args.job_dir / 'stderr.log'}",
        f"--property=ExecStopPost={PYTHON} {finish}", str(shell),
    ]
    write_json(args.job_dir / "launch_request.json", {"service_command": request, "child_command": command})
    completed = subprocess.run(request, capture_output=True, text=True)
    write_json(args.job_dir / "launcher_result.json", {"returncode": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr})
    print(completed.stdout + completed.stderr, end="")
    completed.check_returncode()


if __name__ == "__main__":
    main()
