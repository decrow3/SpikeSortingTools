#!/usr/bin/env python
"""Launch the bounded raw imec1 lighthouse search as a persistent user service."""
from __future__ import annotations
import argparse, json, os, shlex, shutil, socket, subprocess
from pathlib import Path
from testing.luke_full_session_medicine import save, sha
from testing.luke_imec1_dots_raw_lighthouse_check import DEFAULT_OUTPUT, MANIFEST, SCHEMA, SORTING, SORT_RESULT, DIRECT

ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/huklab/Documents/DARTsort/.venv/bin/python")
FILES = ["testing/__init__.py", "testing/managed_job.py", "testing/luke_imec1_dots_lighthouse_direct_check.py", "testing/luke_imec1_dots_raw_lighthouse_check.py"]

def valid_proof(path: Path) -> bool:
    try:
        receipt=json.loads((path/"receipt.json").read_text()); disconnected=json.loads((path/"launcher_disconnected_state.json").read_text()); result=json.loads((path/"service_result.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError): return False
    return receipt.get("state")=="complete" and receipt.get("returncode")==0 and disconnected.get("active_after_launcher_exit") is True and result.get("SERVICE_RESULT")=="success"

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--unit",required=True); ap.add_argument("--job-dir",type=Path,required=True); ap.add_argument("--output",type=Path); ap.add_argument("--dummy",action="store_true"); ap.add_argument("--proof",type=Path); ap.add_argument("--recovery-source",type=Path); ap.add_argument("--seed-audit-source",type=Path); ap.add_argument("--candidate-extraction-source",type=Path); ap.add_argument("--candidate-audit-source",type=Path); ap.add_argument("--candidate-unit-id",type=int); ap.add_argument("--source-failed-job",type=Path); args=ap.parse_args()
    if not args.unit.startswith("luke-") or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-" for c in args.unit): ap.error("invalid unit")
    if args.job_dir.exists(): raise RuntimeError("job directory exists; preserve evidence")
    if not args.dummy and (args.output is None or args.proof is None or not valid_proof(args.proof)): raise RuntimeError("valid launcher-disconnection dummy proof required")
    output=(args.output or (ROOT/"testing/outputs/luke_imec1_dots_raw_lighthouse_dummy_v1")).resolve()
    if output.exists(): raise RuntimeError("output exists; preserve evidence")
    job=args.job_dir.resolve(); job.mkdir(parents=True); bundle=job/"source"
    for name in FILES:
        target=bundle/name; target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(ROOT/name,target)
    save(job/"source_manifest.json",{name:sha(bundle/name) for name in FILES})
    manifest=json.loads(MANIFEST.read_text()); binary=Path(manifest["binary_path"]); stat=binary.stat()
    config={"schema":SCHEMA,"output":str(output),"authorization":"User requested continued imec1 dots-RF lighthouse discovery", "seed_interval_s":[36.457968473,56.457968473],"motion_windows_s":[[935,941],[974,986],[998,1002],[1014,1025],[1071,1075],[1097,1112]],"raw_binary":str(binary),"raw_size":stat.st_size,"raw_mtime_ns":stat.st_mtime_ns,"candidate_inventory":str((DIRECT/"candidate_inventory.csv").resolve()),"whole_probe_rivals":True,"all_eligible_targets":True,"input_sha256":{str(p):sha(p) for p in [MANIFEST,SORTING,SORT_RESULT,DIRECT/"candidate_inventory.csv"]},"checkpoint":"Completed intervals are sealed, but this v2 worker deliberately refuses in-place retry; preserve any partial evidence and launch a new version after investigation.","no_automatic_restart":True}
    if args.recovery_source:
        if args.source_failed_job is None: ap.error("--recovery-source requires --source-failed-job")
        config["recovery_source"]=str(args.recovery_source.resolve()); config["source_failed_job"]=str(args.source_failed_job.resolve())
        config["input_sha256"].update({str((args.recovery_source/name).resolve()):sha(args.recovery_source/name) for name in ["events.csv","track_plausibility.csv","interval_counts.csv","seed_phase_templates.csv"]})
        config["input_sha256"][str((args.source_failed_job/"receipt.json").resolve())]=sha(args.source_failed_job/"receipt.json")
    if args.seed_audit_source:
        config["seed_audit_source"]=str(args.seed_audit_source.resolve())
        config["input_sha256"].update({str((args.seed_audit_source/name).resolve()):sha(args.seed_audit_source/name) for name in ["events.csv","seed_phase_templates.csv","seed_phase_waveforms.npz"]})
    if args.candidate_extraction_source:
        if args.candidate_audit_source is None or args.candidate_unit_id is None: ap.error("candidate report requires audit source and unit id")
        config.update(candidate_extraction_source=str(args.candidate_extraction_source.resolve()),candidate_audit_source=str(args.candidate_audit_source.resolve()),candidate_unit_id=args.candidate_unit_id)
        for path in [args.candidate_extraction_source/"events.csv",args.candidate_audit_source/"seed_source_event_competition.csv"]: config["input_sha256"][str(path.resolve())]=sha(path)
    save(job/"config.json",config)
    command=[str(PYTHON),"-u","-m","testing.managed_job","--receipt",str(job/"receipt.json"),"--cwd",str(bundle),"--",str(PYTHON),"-u","-m","testing.luke_imec1_dots_raw_lighthouse_check","--config",str(job/"config.json")]
    if args.dummy: command.append("--dummy")
    script=job/"launch.sh"; script.write_text("#!/bin/bash\nset -eu\ncd "+shlex.quote(str(bundle))+"\nexport OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4\nexport MPLCONFIGDIR="+shlex.quote(str(job/"matplotlib"))+"\nexport NUMBA_CACHE_DIR="+shlex.quote(str(job/"numba"))+"\nexec "+shlex.join(command)+"\n"); script.chmod(0o755)
    finish=job/"save_service_result.py"; finish.write_text("import json,os\nfrom pathlib import Path\np=Path(__file__).with_name('service_result.json'); q=p.with_suffix('.tmp'); q.write_text(json.dumps({k:os.environ.get(k) for k in ['SERVICE_RESULT','EXIT_CODE','EXIT_STATUS']})); os.replace(q,p)\n")
    request=["systemd-run","--user","--unit="+args.unit,"--collect","--property=Type=exec","--property=CPUQuota=600%","--property=MemoryMax=48G","--property=TasksMax=128","--property=KillMode=control-group","--property=TimeoutStopSec=30","--property=StandardOutput=append:"+str(job/"stdout.log"),"--property=StandardError=append:"+str(job/"stderr.log"),"--property=ExecStopPost="+str(PYTHON)+" "+str(finish),str(script)]
    save(job/"launch_request.json",{"command":request,"child_command":command,"unit":args.unit,"host":socket.gethostname(),"launcher_pid":os.getpid(),"dummy":args.dummy,"proof":None if args.proof is None else str(args.proof)})
    result=subprocess.run(request,capture_output=True,text=True); save(job/"launcher_result.json",{"returncode":result.returncode,"stdout":result.stdout,"stderr":result.stderr}); print(result.stdout+result.stderr,flush=True); result.check_returncode()
    shown=subprocess.check_output(["systemctl","--user","show",args.unit+".service","-p","ActiveState","-p","SubState","-p","MainPID","-p","Result"],text=True); state=dict(line.split("=",1) for line in shown.splitlines() if "=" in line)
    save(job/"launcher_disconnected_state.json",{"active_after_launcher_exit":state.get("ActiveState")=="active" and state.get("SubState")=="running" and int(state.get("MainPID","0"))>0,**state})

if __name__=="__main__": main()
