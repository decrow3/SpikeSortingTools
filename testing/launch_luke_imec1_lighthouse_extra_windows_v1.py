#!/usr/bin/env python
"""Launch the frozen extra-window tracker under a persistent user service."""
from __future__ import annotations

import argparse, json, os, shlex, shutil, socket, subprocess
from pathlib import Path

from testing.luke_full_session_medicine import save, sha
from testing.luke_imec1_dots_raw_lighthouse_check import MANIFEST

ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/huklab/Documents/DARTsort/.venv/bin/python")
WINDOWS = [[120.0, 150.0], [320.0, 350.0], [484.0, 514.0], [696.0, 726.0], [941.0, 971.0], [1120.0, 1150.0]]
FILES = [
    "testing/__init__.py", "testing/managed_job.py", "testing/luke_full_session_medicine.py",
    "testing/luke_imec1_dots_lighthouse_direct_check.py", "testing/luke_imec1_dots_raw_lighthouse_check.py",
    "testing/luke_imec1_dots_sorterfree_waveform_discovery.py", "testing/luke_imec1_sorterfree_phase_audit_v1.py",
    "testing/luke_imec1_dots_sorterfree_waveform_discovery_v3.py", "testing/luke_imec1_compact_core_pilot_v1.py",
    "testing/luke_imec1_compact_alignment_pilot_v2.py", "testing/luke_imec1_lighthouse_extra_windows_v1.py",
]


def valid_proof(path: Path) -> bool:
    try:
        receipt=json.loads((path/"receipt.json").read_text()); detached=json.loads((path/"launcher_disconnected_state.json").read_text()); result=json.loads((path/"service_result.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError): return False
    return receipt.get("state")=="complete" and receipt.get("returncode")==0 and detached.get("active_after_launcher_exit") is True and result.get("SERVICE_RESULT")=="success"


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--unit",required=True); parser.add_argument("--job-dir",type=Path,required=True); parser.add_argument("--output",type=Path,required=True); parser.add_argument("--proof",type=Path,required=True); args=parser.parse_args()
    if args.job_dir.exists() or args.output.exists(): raise RuntimeError("job/output exists; preserve evidence")
    if not valid_proof(args.proof): raise RuntimeError("valid launcher-disconnection proof required")
    job=args.job_dir.resolve(); output=args.output.resolve(); job.mkdir(parents=True); bundle=job/"source"
    for name in FILES:
        target=bundle/name; target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(ROOT/name,target)
    save(job/"source_manifest.json",{name:sha(bundle/name) for name in FILES})
    manifest=json.loads(MANIFEST.read_text()); binary=Path(manifest["binary_path"]); stat=binary.stat()
    config={
        "schema":"luke0804-imec1-lighthouse-extra-windows-v1", "output":str(output), "windows_s":WINDOWS,
        "templates":str((ROOT/"testing/outputs/luke_imec1_compact_alignment_pilot_v2/frozen_training_templates.npz").resolve()),
        "external_bank":str((ROOT/"testing/outputs/luke_imec1_dots_sorterfree_waveform_discovery_v3").resolve()),
        "window_derivation":"Six duration-only evenly spaced 30-s proposals, deterministically shifted forward past all earlier seed/training/qualification/gentle/motion-review intervals; half-open boundaries.",
        "excluded_intervals_s":[[36.457968473,56.457968473],[70,80],[110,120],[300,320],[420,430],[560,570],[820,830],[850,860],[935,941],[974,986],[998,1002],[1014,1025],[1071,1075],[1097,1112]],
        "rf_policy":"No stimulus events, RF trial identities, development responses, or RF holdout outcomes are opened. This job reads AP voltage only.",
        "motion_inputs":[], "sorter_inputs":[], "raw_binary":str(binary), "raw_size":stat.st_size, "raw_mtime_ns":stat.st_mtime_ns,
        "checkpoint":"Each completed 30-s window has an immutable hash receipt; an incomplete window is preserved and requires a new version.",
    }
    save(job/"config.json",config)
    command=[str(PYTHON),"-u","-m","testing.managed_job","--receipt",str(job/"receipt.json"),"--cwd",str(bundle),"--",str(PYTHON),"-u","-m","testing.luke_imec1_lighthouse_extra_windows_v1","--config",str(job/"config.json")]
    script=job/"launch.sh"; script.write_text("#!/bin/bash\nset -eu\ncd "+shlex.quote(str(bundle))+"\nexport OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8\nexport MPLCONFIGDIR="+shlex.quote(str(job/"matplotlib"))+"\nexec "+shlex.join(command)+"\n"); script.chmod(0o755)
    finish=job/"save_service_result.py"; finish.write_text("import json,os\nfrom pathlib import Path\np=Path(__file__).with_name('service_result.json');q=p.with_suffix('.tmp');q.write_text(json.dumps({k:os.environ.get(k) for k in ['SERVICE_RESULT','EXIT_CODE','EXIT_STATUS']}));os.replace(q,p)\n")
    request=["systemd-run","--user","--unit="+args.unit,"--collect","--property=Type=exec","--property=CPUQuota=800%","--property=MemoryMax=64G","--property=TasksMax=160","--property=KillMode=control-group","--property=TimeoutStopSec=30","--property=StandardOutput=append:"+str(job/"stdout.log"),"--property=StandardError=append:"+str(job/"stderr.log"),"--property=ExecStopPost="+str(PYTHON)+" "+str(finish),str(script)]
    save(job/"launch_request.json",{"command":request,"child_command":command,"unit":args.unit,"host":socket.gethostname(),"proof":str(args.proof.resolve())})
    result=subprocess.run(request,capture_output=True,text=True); save(job/"launcher_result.json",{"returncode":result.returncode,"stdout":result.stdout,"stderr":result.stderr}); print(result.stdout+result.stderr,flush=True); result.check_returncode()
    shown=subprocess.check_output(["systemctl","--user","show",args.unit+".service","-p","ActiveState","-p","SubState","-p","MainPID","-p","Result"],text=True); state=dict(line.split("=",1) for line in shown.splitlines() if "=" in line)
    save(job/"launcher_disconnected_state.json",{"active_after_launcher_exit":state.get("ActiveState")=="active" and state.get("SubState")=="running" and int(state.get("MainPID","0"))>0,**state})


if __name__=="__main__": main()
