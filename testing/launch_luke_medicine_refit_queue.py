"""Freeze and independently launch the queued 30k full-session MEDiCINe refit."""
import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import socket
import subprocess

from testing.luke_full_session_medicine import MED_ARGS, save, sha


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / "environments/rescue-production/.venv/bin/python"
MEDPY = Path("/home/huklab/anaconda3/envs/spikeinterface/bin/python")
INPUT = Path("/media/huklab/Data/luke_full_session_medicine_20260909_v1/input")
BASELINE = Path("/media/huklab/Data/luke_full_session_medicine_20260909_v1/fit/field.npz")
DEP_JOB = Path("/media/huklab/Data/luke_medicine_rigid_sort_20260909_v1_job")
FILES = ["testing/__init__.py", "testing/managed_job.py", "testing/luke_full_session_medicine.py",
         "testing/luke_medicine_rigid_queue.py", "testing/luke_medicine_refit_queue.py"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--unit", required=True)
    ap.add_argument("--job-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--dummy", action="store_true")
    ap.add_argument("--proof", type=Path)
    args = ap.parse_args()
    if not socket.gethostname().startswith("huklaban1"):
        raise RuntimeError("Refit queue assigned to huklaban1")
    if not args.unit.startswith("luke-") or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-" for c in args.unit):
        raise ValueError("Invalid unit name")
    if args.output.exists():
        raise RuntimeError("Existing output; no automatic overwrite or retry")
    if not args.dummy:
        if args.proof is None:
            raise RuntimeError("Real queue requires launcher-disconnection proof")
        receipt = json.loads((args.proof / "receipt.json").read_text())
        state = json.loads((args.proof / "launcher_disconnected_state.json").read_text())
        result = json.loads((args.proof / "service_result.json").read_text())
        if not (receipt.get("state") == "complete" and receipt.get("returncode") == 0 and
                state.get("active_after_launcher_exit") is True and result.get("SERVICE_RESULT") == "success"):
            raise RuntimeError("Dummy survival/exit proof incomplete")
        linger = subprocess.check_output(["loginctl", "show-user", "huklab", "-p", "Linger", "--value"], text=True).strip()
        if linger != "yes":
            raise RuntimeError("User-manager lingering is not enabled")
        if shutil.disk_usage(args.output.parent).free < 2 * 1024**3:
            raise RuntimeError("Need 2 GiB free for diagnostic fit and evidence")

    job = args.job_dir.resolve()
    job.mkdir(parents=True, exist_ok=False)
    bundle = job / "source"
    for name in FILES:
        target = bundle / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    settings = dict(MED_ARGS)
    settings["training_steps"] = 30000
    pins = [INPUT / "complete.json", INPUT / "input.json", BASELINE,
            DEP_JOB / "launch_request.json"]
    cfg = dict(schema="luke-full-session-medicine-refit-queue-v1", output=str(args.output.resolve()),
        input=str(INPUT), baseline_field=str(BASELINE), medicine_python=str(MEDPY),
        medicine_settings=settings, dependency_unit="luke-medicine-rigid-sort-20260909-v1.service",
        dependency_job=str(DEP_JOB), pinned_inputs={str(p): sha(p) for p in pins},
        source_sha256={name: sha(bundle / name) for name in FILES},
        authorization="User requested cheap 30k full-session MEDiCINe refit queued after active rigid Kilosort",
        checkpoint="No within-fit optimizer checkpoint; interruption requires fit restart after investigation",
        no_automatic_restart=True)
    save(job / "config.json", cfg)
    save(job / "source_manifest.json", cfg["source_sha256"])
    command = [str(PYTHON), "-u", "-m", "testing.luke_medicine_refit_queue", "--config", str(job / "config.json")]
    if args.dummy:
        command.append("--dummy")
    wrapped = [str(PYTHON), "-u", "-m", "testing.managed_job", "--receipt", str(job / "receipt.json"),
               "--cwd", str(bundle), "--", *command]
    script = job / "launch.sh"
    script.write_text("#!/bin/bash\nset -eu\ncd " + shlex.quote(str(bundle)) + "\n"
        "export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4\n"
        "export MPLCONFIGDIR=" + shlex.quote(str(job / "matplotlib")) + "\n"
        "exec " + shlex.join(wrapped) + "\n")
    script.chmod(0o755)
    finish = job / "save_service_result.py"
    finish.write_text("import json,os\nfrom pathlib import Path\nfrom datetime import datetime,timezone\n"
        "p=Path(__file__).with_name('service_result.json');t=p.with_suffix('.tmp')\n"
        "t.write_text(json.dumps(dict(finished_at=datetime.now(timezone.utc).isoformat(),"
        "**{k:os.environ.get(k) for k in ['SERVICE_RESULT','EXIT_CODE','EXIT_STATUS']})))\n"
        "os.replace(t,p)\n")
    request = ["systemd-run", "--user", "--unit=" + args.unit, "--collect", "--property=Type=exec",
        "--property=CPUQuota=400%", "--property=MemoryMax=32G", "--property=TasksMax=128",
        "--property=KillMode=control-group", "--property=TimeoutStopSec=30",
        "--property=StandardOutput=append:" + str(job / "stdout.log"),
        "--property=StandardError=append:" + str(job / "stderr.log"),
        "--property=ExecStopPost=" + str(PYTHON) + " " + str(finish), str(script)]
    save(job / "launch_request.json", dict(command=request, child_command=wrapped, launcher_pid=os.getpid(),
        dummy=args.dummy, unit=args.unit, host=socket.gethostname(), proof=str(args.proof) if args.proof else None))
    result = subprocess.run(request, capture_output=True, text=True)
    save(job / "launcher_result.json", dict(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr))
    print(result.stdout + result.stderr, flush=True)
    result.check_returncode()


if __name__ == "__main__":
    main()
