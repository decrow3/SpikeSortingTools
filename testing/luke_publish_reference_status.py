"""Independent status publisher for the authorized two-machine workflow.

Reads actual systemd state; never launches/restarts a sorter or edits inputs.
Run under systemd, so status delivery also survives the chat closing.
"""
from pathlib import Path
import hashlib
import json
import subprocess
import time

from testing.managed_job import _atomic_json

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'testing/outputs/luke_full_reference_reuse_audit_v1'
SHARED=Path('/mnt/NPX/Luke/20250804/shared_analysis/luke_next_stage_coordination_20260907_v1')
UNIT='luke-full-reference-reuse-audit-v1.service'


def main():
    deadline=time.monotonic()+24*3600
    while time.monotonic()<deadline:
        result=subprocess.run(['systemctl','--user','show',UNIT,
            '--property=LoadState,ActiveState,SubState,MainPID,ExecMainStatus'],
            text=True,capture_output=True,check=True)
        state=dict(line.split('=',1) for line in result.stdout.splitlines() if '=' in line)
        status=json.loads((SHARED/'huklaban1_status.json').read_text())
        status.update(last_verified_unix=time.time(),systemd=state)
        if state.get('LoadState')!='loaded':
            status.update(reference_status='job_handle_missing_investigate',bulk_shared_reads='unresolved_do_not_assume_released')
            _atomic_json(SHARED/'huklaban1_status.json',status)
            raise RuntimeError('Reference job handle missing; no restart attempted')
        if state.get('MainPID')=='0':
            receipt=json.loads((OUT.parent/'luke_full_reference_reuse_audit_v1.job.json').read_text())
            if state.get('ExecMainStatus')=='0' and receipt.get('state')=='complete' and receipt.get('returncode')==0:
                raw=(OUT/'summary.json').read_bytes();summary=json.loads(raw)
                assert summary['status']=='reference_reusable'
                compact={k:summary[k] for k in ['status','curated_spikes','curated_units','effective_motion_off',
                    'recording_request_digest','recording_content_sha256','verified_recording_bytes','limitations']}
                compact.update(sort_identity_digest=summary['sort_identity']['identity_digest'],
                    source_summary_path=str(OUT/'summary.json'),source_summary_sha256=hashlib.sha256(raw).hexdigest(),
                    service=UNIT,terminal_systemd=state)
                _atomic_json(SHARED/'huklaban1_reference_summary.json',compact)
                status.update(reference_status='reference_reusable',bulk_shared_reads='released',
                    next_local_work='fixed-time diagnostic panel; no repeat sort required',returncode=0)
                _atomic_json(SHARED/'huklaban1_status.json',status)
                print(json.dumps(compact),flush=True)
                return
            status.update(reference_status='audit_failed_investigate_before_reuse',
                bulk_shared_reads='released_after_audit_termination',returncode=receipt.get('returncode'))
            _atomic_json(SHARED/'huklaban1_status.json',status)
            raise RuntimeError('Reference audit terminated unsuccessfully; no restart attempted')
        progress=OUT/'progress.json'
        if progress.exists():status['progress']=json.loads(progress.read_text())
        status.update(reference_status='audit_running',bulk_shared_reads='reserved_for_huklaban1')
        _atomic_json(SHARED/'huklaban1_status.json',status)
        time.sleep(30)
    raise RuntimeError('Publisher observation deadline exceeded; inspect live job, do not assume termination')


if __name__=='__main__':
    main()
