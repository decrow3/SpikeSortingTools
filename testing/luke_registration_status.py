"""Publish only huklaban5's status, using freshly queried systemd/process state."""
import argparse
import json
import subprocess
import time
from pathlib import Path
from testing.managed_job import _atomic_json

COORD=Path('/mnt/NPX/Luke/20250804/shared_analysis/luke_next_stage_coordination_20260907_v1')

def publish(root,service,phase=None):
    p=root/'huklaban5_status.json';s=json.loads(p.read_text())
    if service:
        r=subprocess.run(['systemctl','--user','show',service,'--property=ActiveState,SubState,MainPID,ExecMainStatus,Result,MemoryCurrent,MemoryPeak,CPUUsageNSec'],text=True,capture_output=True,check=True)
        state=dict(line.split('=',1) for line in r.stdout.splitlines() if '=' in line)
        s['service']=service;s['systemd']=state;s['last_verified_main_pid']=int(state.get('MainPID',0))
        s['job_state']=state.get('ActiveState','unknown')+'/'+state.get('SubState','unknown')
        pid=s['last_verified_main_pid']
        s['process_exists']=bool(pid and Path(f'/proc/{pid}').exists())
    if phase:s['phase']=phase
    if (root/'progress.json').exists():s['progress']=json.loads((root/'progress.json').read_text())
    if (root/'job.json').exists():s['job_receipt']=json.loads((root/'job.json').read_text())
    s['updated_unix']=time.time();s['last_verified_unix']=time.time()
    _atomic_json(p,s);_atomic_json(COORD/'huklaban5_status.json',s)
    print(json.dumps(s,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--service');p.add_argument('--phase')
    a=p.parse_args();publish(a.root,a.service,a.phase)
