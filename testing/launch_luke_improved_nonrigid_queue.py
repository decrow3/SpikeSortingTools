"""Launch the huklaban5 correction/sort queue as an independent user service."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / "environments/rescue-production/.venv/bin/python"
SOURCE = Path("/mnt/NPX/Luke/20250804/rescue_pipeline_results_Luke0804_V2V1_g0_imec0/recording")
SHARED = Path("/mnt/NPX/Luke/20250804/shared_analysis/luke_improved_motion_two_machine_20260909_v1")
OUTPUT = Path("/media/huklaban5/Data/luke_improved_motion_20260909_v1")
JOB = Path("/media/huklaban5/Data/luke_improved_motion_20260909_v1_job")
FILES = (
    "testing/luke_improved_nonrigid_queue.py",
    "testing/checked_int16_recording.py",
    "testing/managed_job.py",
    "testing/luke_external_warp_pipeline.py",
    "pipeline/config.py",
    "pipeline/preprocess.py",
    "pipeline/runtime.py",
    "pipeline/sorting.py",
    "pipeline/kilosort_compat.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unit", default="luke-improved-nonrigid-20260909-v1")
    parser.add_argument("--job-dir", type=Path, default=JOB)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--proof", required=True, type=Path)
    parser.add_argument("--reuse-completed-correction", action="store_true")
    args = parser.parse_args()
    if not socket.gethostname().startswith("huklaban5"):
        raise RuntimeError("the nonrigid arm is assigned only to huklaban5")
    proof_receipt = json.loads((args.proof / "receipt.json").read_text())
    proof_state = json.loads((args.proof / "launcher_disconnected_state.json").read_text())
    proof_result = json.loads((args.proof / "service_result.json").read_text())
    if not (
        proof_receipt.get("state") == "complete"
        and proof_receipt.get("returncode") == 0
        and proof_state.get("active_after_launcher_exit") is True
        and proof_result.get("SERVICE_RESULT") == "success"
    ):
        raise RuntimeError("independent user-service proof is incomplete")
    if args.job_dir.exists():
        raise RuntimeError("fresh queue requires an unused job directory")
    if args.output.exists():
        completed = args.output / "corrected_recording" / "rescue_recording_manifest.json"
        if not args.reuse_completed_correction or not completed.is_file():
            raise RuntimeError("existing output is not an explicitly reusable completed correction")
    if not SOURCE.is_dir():
        raise FileNotFoundError(SOURCE)
    args.job_dir.mkdir(parents=True)
    source_hashes = {name: _sha256(ROOT / name) for name in FILES}
    _write_json(
        args.job_dir / "resolved_request.json",
        {
            "schema": "luke-improved-nonrigid-launch-v1",
            "host": socket.gethostname(),
            "source_recording": str(SOURCE),
            "shared_exchange": str(SHARED),
            "output_root": str(args.output),
            "proof": str(args.proof),
            "source_sha256": source_hashes,
            "repository_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "no_automatic_restart": True,
        },
    )
    command = [
        str(PYTHON),
        "-u",
        "-m",
        "testing.managed_job",
        "--receipt",
        str(args.job_dir / "receipt.json"),
        "--cwd",
        str(ROOT),
        "--",
        str(PYTHON),
        "-u",
        "-m",
        "testing.luke_improved_nonrigid_queue",
        "--shared",
        str(SHARED),
        "--source",
        str(SOURCE),
        "--output",
        str(args.output),
    ]
    if args.reuse_completed_correction:
        command.append("--reuse-completed-correction")
    shell = args.job_dir / "launch.sh"
    shell.write_text(
        "#!/bin/bash\nset -eu\n"
        f"cd {shlex.quote(str(ROOT))}\n"
        f"export MPLCONFIGDIR={shlex.quote(str(args.job_dir / 'matplotlib'))}\n"
        "export NUMBA_CACHE_DIR=/tmp/luke-improved-nonrigid-numba\n"
        f"exec {shlex.join(command)}\n"
    )
    shell.chmod(0o755)
    finish = args.job_dir / "save_service_result.py"
    finish.write_text(
        "import json,os\nfrom datetime import datetime,timezone\nfrom pathlib import Path\n"
        "p=Path(__file__).with_name('service_result.json'); q=p.with_suffix('.tmp')\n"
        "q.write_text(json.dumps(dict(finished_at=datetime.now(timezone.utc).isoformat(),"
        "**{k:os.environ.get(k) for k in ['SERVICE_RESULT','EXIT_CODE','EXIT_STATUS']}),indent=2)+'\\n')\n"
        "os.replace(q,p)\n"
    )
    request = [
        "systemd-run",
        "--user",
        f"--unit={args.unit}",
        "--collect",
        "--property=Type=exec",
        "--property=CPUQuota=800%",
        "--property=MemoryMax=180G",
        "--property=TasksMax=1024",
        "--property=KillMode=control-group",
        "--property=TimeoutStopSec=60",
        f"--property=StandardOutput=append:{args.job_dir / 'stdout.log'}",
        f"--property=StandardError=append:{args.job_dir / 'stderr.log'}",
        f"--property=ExecStopPost={PYTHON} {finish}",
        str(shell),
    ]
    _write_json(args.job_dir / "launch_request.json", {"command": request, "child_command": command})
    completed = subprocess.run(request, capture_output=True, text=True)
    _write_json(
        args.job_dir / "launcher_result.json",
        {"returncode": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr},
    )
    print(completed.stdout + completed.stderr, flush=True)
    completed.check_returncode()


if __name__ == "__main__":
    main()
