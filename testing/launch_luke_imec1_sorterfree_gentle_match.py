#!/usr/bin/env python
"""Persistent launcher for the frozen-bank imec1 gentle-window match."""
from __future__ import annotations

import argparse, json, os, shlex, shutil, socket, subprocess
from pathlib import Path

from testing.luke_full_session_medicine import save, sha
from testing.luke_imec1_dots_raw_lighthouse_check import MANIFEST
from testing.luke_imec1_sorterfree_gentle_match import SCHEMA

ROOT=Path(__file__).resolve().parents[1]
PYTHON=Path("/home/huklab/Documents/DARTsort/.venv/bin/python")
WINDOWS=[[70,80],[110,120],[420,430],[560,570],[820,830],[850,860]]
FILES=["testing/__init__.py","testing/managed_job.py","testing/luke_imec1_dots_lighthouse_direct_check.py","testing/luke_imec1_dots_raw_lighthouse_check.py","testing/luke_imec1_dots_sorterfree_waveform_discovery.py","testing/luke_imec1_sorterfree_gentle_match.py"]


def valid(path):
    try:
        r=json.loads((path/"receipt.json").read_text()); d=json.loads((path/"launcher_disconnected_state.json").read_text()); s=json.loads((path/"service_result.json").read_text())
    except (FileNotFoundError,json.JSONDecodeError): return False
    return r.get("state")=="complete" and r.get("returncode")==0 and d.get("active_after_launcher_exit") is True and s.get("SERVICE_RESULT")=="success"


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--unit",required=True); ap.add_argument("--job-dir",type=Path,required=True); ap.add_argument("--output",type=Path,required=True); ap.add_argument("--discovery",type=Path,required=True); ap.add_argument("--proof",type=Path,required=True); args=ap.parse_args()
    if args.job_dir.exists() or args.output.exists(): raise RuntimeError("job/output exists; preserve evidence")
    if not valid(args.proof): raise RuntimeError("valid persistent-manager proof required")
    job=args.job_dir.resolve(); output=args.output.resolve(); discovery=args.discovery.resolve(); job.mkdir(parents=True); bundle=job/"source"
    for name in FILES:
        target=bundle/name; target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(ROOT/name,target)
    save(job/"source_manifest.json",{name:sha(bundle/name) for name in FILES})
    manifest=json.loads(MANIFEST.read_text()); binary=Path(manifest["binary_path"]); stat=binary.stat()
    config={"schema":SCHEMA,"output":str(output),"frozen_discovery":str(discovery),"gentle_windows_s":WINDOWS,"window_selection":"Within each third of the dots-RF epoch, the two nonoverlapping 10-s windows with smallest maximum across DREDGE, KS-sidecar, decentralized, and MEDiCINe P90 depthwise temporal ranges; selected before candidate tracks were read.","selection_is_conditional_on_estimators":True,"sorter_inputs":[],"motion_inputs_used_for_matching":[],"raw_binary":str(binary),"raw_size":stat.st_size,"raw_mtime_ns":stat.st_mtime_ns,"checkpoint":"Each completed window is hash-sealed; interrupted unsealed work requires preservation and a new-version restart.","no_automatic_restart":True}
    save(job/"config.json",config)
    command=[str(PYTHON),"-u","-m","testing.managed_job","--receipt",str(job/"receipt.json"),"--cwd",str(bundle),"--",str(PYTHON),"-u","-m","testing.luke_imec1_sorterfree_gentle_match","--config",str(job/"config.json")]
    script=job/"launch.sh"; script.write_text("#!/bin/bash\nset -eu\ncd "+shlex.quote(str(bundle))+"\nexport OPENBLAS_NUM_THREADS=6 OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 NUMEXPR_NUM_THREADS=6\nexport MPLCONFIGDIR="+shlex.quote(str(job/"matplotlib"))+"\nexec "+shlex.join(command)+"\n"); script.chmod(0o755)
    finish=job/"save_service_result.py"; finish.write_text("import json,os\nfrom pathlib import Path\np=Path(__file__).with_name('service_result.json');q=p.with_suffix('.tmp');q.write_text(json.dumps({k:os.environ.get(k) for k in ['SERVICE_RESULT','EXIT_CODE','EXIT_STATUS']}));os.replace(q,p)\n")
    request=["systemd-run","--user","--unit="+args.unit,"--collect","--property=Type=exec","--property=CPUQuota=800%","--property=MemoryMax=64G","--property=TasksMax=160","--property=KillMode=control-group","--property=StandardOutput=append:"+str(job/"stdout.log"),"--property=StandardError=append:"+str(job/"stderr.log"),"--property=ExecStopPost="+str(PYTHON)+" "+str(finish),str(script)]
    save(job/"launch_request.json",{"command":request,"child_command":command,"unit":args.unit,"host":socket.gethostname(),"proof":str(args.proof)})
    result=subprocess.run(request,capture_output=True,text=True); save(job/"launcher_result.json",{"returncode":result.returncode,"stdout":result.stdout,"stderr":result.stderr}); print(result.stdout+result.stderr,flush=True); result.check_returncode()
    shown=subprocess.check_output(["systemctl","--user","show",args.unit+".service","-p","ActiveState","-p","SubState","-p","MainPID","-p","Result"],text=True); state=dict(line.split("=",1) for line in shown.splitlines() if "=" in line)
    save(job/"launcher_disconnected_state.json",{"active_after_launcher_exit":state.get("ActiveState")=="active" and state.get("SubState")=="running" and int(state.get("MainPID","0"))>0,**state})


if __name__=="__main__": main()
