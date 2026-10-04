#!/usr/bin/env python3
"""Create, but do not silently broaden, the persistent EN crop-screen service."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / "environments/rescue-production/.venv/bin/python"


def save(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch", action="store_true", help="required; omitted means print-only")
    parser.add_argument("--unit", default="en-common-support-crop-screen-20260930-v1.service")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/en_common_support_crop_screen.v1.json")
    parser.add_argument("--job-dir", type=Path, default=ROOT / "testing/outputs/en_common_support_crop_screen_job_v1")
    parser.add_argument("--output", type=Path, default=Path("/media/huklaban5/Data/en_common_support_crop_screen_20260930_v1"))
    args = parser.parse_args()
    command = [str(PYTHON), "-m", "testing.en_common_support_crop_screen", "--config", str(args.config.resolve()), "--output", str(args.output.resolve()), "--execute-frozen-sorts"]
    if not args.launch:
        print(json.dumps({"status": "not_launched", "command": command}, indent=2))
        return
    if args.job_dir.exists() or args.output.exists():
        raise FileExistsError("job or output namespace already exists")
    args.job_dir.mkdir(parents=True)
    shell = args.job_dir / "launch.sh"
    shell.write_text("#!/bin/bash\nset -eu\ncd " + shlex.quote(str(ROOT)) + "\nexport EN_CROP_SCREEN_PERSISTENT_JOB=1\nexport MPLCONFIGDIR=" + shlex.quote(str(args.job_dir / "matplotlib")) + "\nexport NUMBA_CACHE_DIR=/tmp/en-common-support-crop-numba\nexec " + shlex.join(command) + "\n")
    shell.chmod(0o755)
    finish = args.job_dir / "save_service_result.py"
    finish.write_text("import json,os\nfrom datetime import datetime,timezone\nfrom pathlib import Path\np=Path(__file__).with_name('service_result.json');q=p.with_suffix('.partial')\nq.write_text(json.dumps({'finished_at':datetime.now(timezone.utc).isoformat(),**{k:os.environ.get(k) for k in ['SERVICE_RESULT','EXIT_CODE','EXIT_STATUS']}},indent=2)+'\\n');os.replace(q,p)\n")
    request = ["systemd-run", "--user", f"--unit={args.unit}", "--collect", "--property=Type=exec", "--property=CPUQuota=800%", "--property=MemoryHigh=150G", "--property=MemoryMax=180G", "--property=TasksMax=1024", "--property=KillMode=control-group", "--property=TimeoutStopSec=60", "--property=ManagedOOMPreference=avoid", f"--property=StandardOutput=append:{args.job_dir / 'stdout.log'}", f"--property=StandardError=append:{args.job_dir / 'stderr.log'}", f"--property=ExecStopPost={PYTHON} {finish}", str(shell)]
    save(args.job_dir / "launch_request.json", {"service_command": request, "child_command": command})
    result = subprocess.run(request, capture_output=True, text=True)
    save(args.job_dir / "launcher_result.json", {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr})
    print(result.stdout + result.stderr, end="")
    result.check_returncode()


if __name__ == "__main__":
    main()
