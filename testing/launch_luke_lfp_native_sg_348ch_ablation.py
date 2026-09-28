"""Launch the matched native-LFP versus SG25 ablation in a user service."""

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
from testing.luke_lfp_native_sg_348ch_ablation import SCHEMA


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / "environments/rescue-production/.venv/bin/python"
BASE = Path("/mnt/NPX/Luke/20250804/rescue_pipeline_results_Luke0804_V2V1_g0_imec0")
LFP = ROOT / "testing/outputs/luke_lfp_lighthouse_validation_v1/lfp_package/930_1030_band_0p5_8_motion.npz"
BASELINE = Path("/media/huklab/Data/luke_lfp_ap_100s_sort_matrix_v1")
PROOF = Path("/media/huklab/Data/luke_lfp_native_250hz_100s_v1_job")
EVENTS = ROOT / "testing/outputs/luke_waveform_only_expansion_v1/overlay_events.csv"
FAMILIES = ROOT / "testing/outputs/luke_lighthouse_method_audit_v1/d_candidate_families.csv"
REGIMES = ROOT / "testing/outputs/luke_motion_candidate_lighthouse_comparison_v1/candidate_increments.csv"


def valid_dummy(path: Path) -> bool:
    try:
        receipt = json.loads((path / "receipt.json").read_text())
        disconnected = json.loads((path / "launcher_disconnected_state.json").read_text())
        service = json.loads((path / "service_result.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return False
    return receipt.get("state") == "complete" and receipt.get("returncode") == 0 and disconnected.get("active_after_launcher_exit") is True and service.get("SERVICE_RESULT") == "success"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unit", required=True)
    parser.add_argument("--job-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dummy", action="store_true")
    parser.add_argument("--proof", type=Path)
    args = parser.parse_args()
    if not socket.gethostname().startswith("huklaban1"):
        raise RuntimeError("native/SG ablation is assigned to huklaban1")
    if not args.unit.startswith("luke-") or any(char not in "abcdefghijklmnopqrstuvwxyz0123456789-" for char in args.unit):
        raise ValueError("invalid systemd unit name")
    if args.output.exists() or args.job_dir.exists():
        raise RuntimeError("output or job directory already exists; preserve prior evidence")
    if not args.dummy and (args.proof is None or not valid_dummy(args.proof)):
        raise RuntimeError("successful launcher-disconnection dummy proof required")
    if not args.dummy:
        linger = subprocess.check_output(["loginctl", "show-user", "huklab", "-p", "Linger", "--value"], text=True).strip()
        if linger != "yes":
            raise RuntimeError("independent user-manager lifetime is not configured")
    job = args.job_dir.resolve()
    job.mkdir(parents=True)
    bundle = job / "source"
    files = [path for package in ("pipeline", "testing") for path in (ROOT / package).glob("*.py")]
    files.append(ROOT / "environments/rescue-production/uv.lock")
    for path in files:
        target = bundle / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    from pipeline.runtime import production_environment_contract, validate_production_environment
    environment = validate_production_environment(require_cuda=not args.dummy)
    from pipeline.kilosort_compat import kilosort_compatibility_receipt
    compatibility = kilosort_compatibility_receipt()
    if not args.dummy and not compatibility["applied"]:
        raise RuntimeError("known KS4 compatibility repair is not installed")
    source_manifest = BASE / "recording/rescue_recording_manifest.json"
    binary = BASE / "recording/traces_cached_seg0.raw"
    proof_receipt = PROOF / "receipt.json"
    proof_config = PROOF / "config.json"
    proof_stderr = PROOF / "stderr.log"
    proof_worker = PROOF / "source/testing/luke_lfp_native_100s_sort.py"
    cfg = {
        "schema": SCHEMA,
        "output": str(args.output.resolve()),
        "recording": str(BASE / "recording"),
        "recording_manifest": str(source_manifest),
        "recording_manifest_sha256": sha(source_manifest),
        "recording_content_sha256": "2d6fac755db3182841bf121d822754f963ebada3bcb8cd9e96e0b63e6f9b7372",
        "recording_binary_mtime_ns": binary.stat().st_mtime_ns,
        "lfp_native_motion": str(LFP),
        "lfp_native_motion_sha256": sha(LFP),
        "baseline_contract": str(BASELINE / "contract.json"),
        "baseline_contract_sha256": sha(BASELINE / "contract.json"),
        "source_proof_receipt": str(proof_receipt),
        "source_proof_receipt_sha256": sha(proof_receipt),
        "source_proof_config": str(proof_config),
        "source_proof_config_sha256": sha(proof_config),
        "source_proof_stderr": str(proof_stderr),
        "source_proof_stderr_sha256": sha(proof_stderr),
        "source_proof_worker": str(proof_worker),
        "source_proof_worker_sha256": sha(proof_worker),
        "lighthouse_events": str(EVENTS),
        "lighthouse_events_sha256": sha(EVENTS),
        "lighthouse_families": str(FAMILIES),
        "lighthouse_families_sha256": sha(FAMILIES),
        "lighthouse_regimes": str(REGIMES),
        "lighthouse_regimes_sha256": sha(REGIMES),
        "materialize_jobs": 4,
        "sort_scratch_reserve_bytes": 100 * 1024**3,
        "authorization": "User requested the decisive unfiltered native-250-Hz LFP control; channel-gate diagnosis requires matched native and SG reruns",
        "source_validation_reuse": "Reuse the recent complete 241 GB hash from failed v1 after frozen proof, size, mtime, and post-hash gate verification",
        "no_automatic_restart": True,
        "interruption_policy": "Preserve evidence; an interrupted arm restarts from its beginning only after investigation",
        "production_environment": production_environment_contract(),
        "preflight_environment": environment,
        "compatibility": compatibility,
    }
    save(job / "config.json", cfg)
    save(job / "source_manifest.json", {str(path.relative_to(ROOT)): sha(bundle / path.relative_to(ROOT)) for path in files})
    command = [str(PYTHON), "-u", "-m", "testing.managed_job", "--receipt", str(job / "receipt.json"), "--cwd", str(bundle), "--", str(PYTHON), "-u", "-m", "testing.luke_lfp_native_sg_348ch_ablation", "--config", str(job / "config.json")]
    if args.dummy:
        command.append("--dummy")
    script = job / "launch.sh"
    script.write_text("#!/bin/bash\nset -eu\ncd " + shlex.quote(str(bundle)) + "\nexport OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4\nexport MPLCONFIGDIR=" + shlex.quote(str(job / "matplotlib")) + "\nexport NUMBA_CACHE_DIR=" + shlex.quote(str(job / "numba")) + "\nexec " + shlex.join(command) + "\n")
    script.chmod(0o755)
    finish = job / "save_service_result.py"
    finish.write_text("import json,os\nfrom pathlib import Path\np=Path(__file__).with_name('service_result.json'); q=p.with_suffix('.tmp')\nq.write_text(json.dumps({k:os.environ.get(k) for k in ['SERVICE_RESULT','EXIT_CODE','EXIT_STATUS']}))\nos.replace(q,p)\n")
    request = ["systemd-run", "--user", "--unit=" + args.unit, "--collect", "--property=Type=exec", "--property=CPUQuota=800%", "--property=MemoryHigh=128G", "--property=MemoryMax=160G", "--property=TasksMax=256", "--property=KillMode=control-group", "--property=TimeoutStopSec=30", "--property=StandardOutput=append:" + str(job / "stdout.log"), "--property=StandardError=append:" + str(job / "stderr.log"), "--property=ExecStopPost=" + str(PYTHON) + " " + str(finish), str(script)]
    save(job / "launch_request.json", {"command": request, "child_command": command, "unit": args.unit, "host": socket.gethostname(), "launcher_pid": os.getpid(), "dummy": args.dummy, "proof": None if args.proof is None else str(args.proof)})
    result = subprocess.run(request, capture_output=True, text=True)
    save(job / "launcher_result.json", {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr})
    print(result.stdout + result.stderr, flush=True)
    result.check_returncode()
    shown = subprocess.check_output(["systemctl", "--user", "show", args.unit + ".service", "-p", "ActiveState", "-p", "SubState", "-p", "MainPID", "-p", "Result"], text=True)
    state = dict(line.split("=", 1) for line in shown.splitlines() if "=" in line)
    save(job / "launcher_disconnected_state.json", {"active_after_launcher_exit": state.get("ActiveState") == "active" and state.get("SubState") == "running" and int(state.get("MainPID", "0")) > 0, **state})


if __name__ == "__main__":
    main()
