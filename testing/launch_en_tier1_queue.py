#!/usr/bin/env python3
"""Launch the frozen EN Tier-1 watcher as a persistent systemd service."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / "environments/rescue-production/.venv/bin/python"


def write(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit", default="en-tier1-panel-20260929-v1.service")
    parser.add_argument(
        "--job-dir", type=Path, default=ROOT / "testing/outputs/en_tier1_panel_job_v1"
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("/media/huklaban5/Data/en_rounded_field_ks129_20260929_v1/tier1_v1"),
    )
    args = parser.parse_args()
    args.job_dir, args.output = args.job_dir.resolve(), args.output.resolve()
    if args.job_dir.exists() or args.output.exists() or args.output.with_name(args.output.name + ".partial").exists():
        raise FileExistsError("EN Tier-1 job or output exists")
    args.job_dir.mkdir(parents=True)
    command = [
        str(PYTHON), "-m", "testing.en_tier1_queue",
        "--config", str(ROOT / "configs/en_tier1_panel.v1.json"),
        "--output", str(args.output), "--status", str(args.job_dir / "STATUS.json"),
    ]
    shell = args.job_dir / "launch.sh"
    shell.write_text(
        "#!/bin/bash\nset -eu\ncd " + shlex.quote(str(ROOT))
        + "\nexport MPLCONFIGDIR=" + shlex.quote(str(args.job_dir / "matplotlib"))
        + "\nexport NUMBA_CACHE_DIR=/tmp/en-tier1-panel-numba\nexec "
        + shlex.join(command) + "\n"
    )
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
        "systemd-run", "--user", f"--unit={args.unit}", "--collect", "--property=Type=exec",
        "--property=CPUQuota=800%", "--property=MemoryHigh=80G", "--property=MemoryMax=120G",
        "--property=TasksMax=512", "--property=KillMode=control-group",
        "--property=TimeoutStopSec=60", "--property=ManagedOOMPreference=avoid",
        f"--property=StandardOutput=append:{args.job_dir / 'stdout.log'}",
        f"--property=StandardError=append:{args.job_dir / 'stderr.log'}",
        f"--property=ExecStopPost={PYTHON} {finish}", str(shell),
    ]
    write(args.job_dir / "launch_request.json", {"service_command": request, "child_command": command})
    result = subprocess.run(request, capture_output=True, text=True)
    write(
        args.job_dir / "launcher_result.json",
        {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr},
    )
    print(result.stdout + result.stderr, end="")
    result.check_returncode()


if __name__ == "__main__":
    main()
