#!/usr/bin/env python3
"""Launch EN arm A under a guarded persistent systemd user service."""

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
    parser.add_argument("--unit", default="en-rounded-ks129-arm-a-20260929-v1.service")
    parser.add_argument(
        "--job-dir", type=Path,
        default=ROOT / "testing/outputs/en_rounded_ks129_arm_a_job_v1",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("/media/huklaban5/Data/en_rounded_field_ks129_20260929_v1/arm_a_v1"),
    )
    parser.add_argument(
        "--q0-receipt", type=Path,
        default=ROOT / "testing/outputs/en_q0_full_hash_v3/RECEIPT.json",
    )
    args = parser.parse_args()
    args.job_dir = args.job_dir.resolve()
    args.output = args.output.resolve()
    args.q0_receipt = args.q0_receipt.resolve()
    if args.job_dir.exists() or args.output.exists():
        raise FileExistsError("EN arm-A job or output exists")
    args.job_dir.mkdir(parents=True)
    command = [
        str(PYTHON), "-m", "testing.en_rounded_ks129_arm_a_queue",
        "--config", str(ROOT / "configs/en_rounded_field_ks129.v1.json"),
        "--q0-receipt", str(args.q0_receipt), "--output", str(args.output),
    ]
    shell = args.job_dir / "launch.sh"
    shell.write_text(
        "#!/bin/bash\nset -eu\ncd " + shlex.quote(str(ROOT))
        + "\nexport MPLCONFIGDIR=" + shlex.quote(str(args.job_dir / "matplotlib"))
        + "\nexport NUMBA_CACHE_DIR=/tmp/en-rounded-ks129-arm-a-numba\nexec "
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
        "--property=CPUQuota=800%", "--property=MemoryHigh=150G", "--property=MemoryMax=180G",
        "--property=TasksMax=1024", "--property=KillMode=control-group",
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
