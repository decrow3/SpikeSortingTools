"""Launch the five-minute correction matrix in an independent user service."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import socket
import subprocess

from testing.luke_full_session_medicine import save, sha
from testing.luke_medicine_5min_correction_matrix import SCHEMA


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / "environments/rescue-production/.venv/bin/python"
BASE = Path("/mnt/NPX/Luke/20250804/rescue_pipeline_results_Luke0804_V2V1_g0_imec0")
FIELD = ROOT / "testing/outputs/luke_screened_medicine_300s_v1/fit_6sigma_full_10000/field.npz"


def valid_proof(path: Path) -> bool:
    try:
        receipt = json.loads((path / "receipt.json").read_text())
        disconnected = json.loads((path / "launcher_disconnected_state.json").read_text())
        service = json.loads((path / "service_result.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return False
    return (receipt.get("state") == "complete" and receipt.get("returncode") == 0
            and disconnected.get("active_after_launcher_exit") is True
            and service.get("SERVICE_RESULT") == "success")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unit", required=True)
    parser.add_argument("--job-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dummy", action="store_true")
    parser.add_argument("--proof", type=Path)
    args = parser.parse_args()
    if not socket.gethostname().startswith("huklaban1"):
        raise RuntimeError("five-minute matrix is assigned to huklaban1")
    if args.output.exists() or args.job_dir.exists():
        raise RuntimeError("output or job directory already exists; preserve prior evidence")
    if not args.dummy and (args.proof is None or not valid_proof(args.proof)):
        raise RuntimeError("successful launcher-disconnection dummy proof required")

    job = args.job_dir.resolve()
    job.mkdir(parents=True)
    bundle = job / "source"
    files = [p for package in ("pipeline", "testing") for p in (ROOT / package).glob("*.py")]
    files.append(ROOT / "environments/rescue-production/uv.lock")
    for path in files:
        target = bundle / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    source_manifest = BASE / "recording/rescue_recording_manifest.json"
    cfg = dict(
        schema=SCHEMA,
        output=str(args.output.resolve()),
        recording=str(BASE / "recording"),
        recording_manifest=str(source_manifest),
        recording_manifest_sha256=sha(source_manifest),
        recording_content_sha256="2d6fac755db3182841bf121d822754f963ebada3bcb8cd9e96e0b63e6f9b7372",
        field=str(FIELD),
        field_sha256=sha(FIELD),
        materialize_jobs=4,
        sort_scratch_reserve_bytes=80 * 1024**3,
        authorization="User requested direct iteration over voltage corrections and sorts on the five-minute snippet",
        no_automatic_restart=True,
        interruption_policy="Preserve evidence; an interrupted arm sort restarts from its beginning after investigation",
    )
    save(job / "config.json", cfg)
    save(job / "source_manifest.json", {str(p.relative_to(ROOT)): sha(bundle / p.relative_to(ROOT)) for p in files})
    command = [str(PYTHON), "-u", "-m", "testing.managed_job", "--receipt", str(job / "receipt.json"),
               "--cwd", str(bundle), "--", str(PYTHON), "-u", "-m",
               "testing.luke_medicine_5min_correction_matrix", "--config", str(job / "config.json")]
    if args.dummy:
        command.append("--dummy")
    script = job / "launch.sh"
    script.write_text("#!/bin/bash\nset -eu\ncd " + shlex.quote(str(bundle)) + "\n"
                      "export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4\n"
                      "export MPLCONFIGDIR=" + shlex.quote(str(job / "matplotlib")) + "\n"
                      "export NUMBA_CACHE_DIR=" + shlex.quote(str(job / "numba")) + "\n"
                      "exec " + shlex.join(command) + "\n")
    script.chmod(0o755)
    finish = job / "save_service_result.py"
    finish.write_text("import json,os\nfrom pathlib import Path\n"
                      "p=Path(__file__).with_name('service_result.json'); q=p.with_suffix('.tmp')\n"
                      "q.write_text(json.dumps({k:os.environ.get(k) for k in ['SERVICE_RESULT','EXIT_CODE','EXIT_STATUS']}))\n"
                      "os.replace(q,p)\n")
    request = ["systemd-run", "--user", "--unit=" + args.unit, "--collect", "--property=Type=exec",
               "--property=CPUQuota=800%", "--property=MemoryHigh=128G", "--property=MemoryMax=160G",
               "--property=TasksMax=256", "--property=KillMode=control-group", "--property=TimeoutStopSec=30",
               "--property=StandardOutput=append:" + str(job / "stdout.log"),
               "--property=StandardError=append:" + str(job / "stderr.log"),
               "--property=ExecStopPost=" + str(PYTHON) + " " + str(finish), str(script)]
    save(job / "launch_request.json", dict(command=request, child_command=command, unit=args.unit,
                                             host=socket.gethostname(), launcher_pid=os.getpid(), dummy=args.dummy,
                                             proof=None if args.proof is None else str(args.proof)))
    result = subprocess.run(request, capture_output=True, text=True)
    save(job / "launcher_result.json", dict(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr))
    print(result.stdout + result.stderr, flush=True)
    result.check_returncode()
    state_result = subprocess.run(
        ["systemctl", "--user", "show", args.unit + ".service", "-p", "ActiveState",
         "-p", "SubState", "-p", "MainPID", "-p", "Result"],
        capture_output=True, text=True, check=True,
    )
    state = dict(line.split("=", 1) for line in state_result.stdout.splitlines() if "=" in line)
    save(job / "launcher_disconnected_state.json", dict(
        active_after_launcher_exit=state.get("ActiveState") == "active"
        and state.get("SubState") == "running" and int(state.get("MainPID", "0")) > 0,
        **state,
    ))


if __name__ == "__main__":
    main()
