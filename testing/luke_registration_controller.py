"""Independent-manager child: registration, then CPU field audit, then stop."""
import json
import subprocess
import sys
from pathlib import Path
from testing.managed_job import _atomic_json
from testing.luke_registration_status import publish

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'testing/outputs/luke_full_probe_rigid_registration_v1'
SERVICE='luke-full-probe-rigid-registration-v1.service'
STRIP=ROOT/'testing/outputs/luke_saved_motion_field_audit_v1/native_rigid_field.npz'
MOTION=Path('/mnt/NPX/Luke/20250804/dredge_pipeline_results_Luke0804_V2V1_g0_imec0/motion')


def main():
    commands=[('registration',[sys.executable,'-u','-m','testing.luke_full_probe_registration','execute',
        '--output-root',str(OUT),'--qualification',str(OUT/'qualification.json')]),
        ('field_audit',[sys.executable,'-u','-m','testing.luke_full_probe_registration_audit',
        '--registration',str(OUT),'--strip',str(STRIP),'--motion',str(MOTION),'--output',str(OUT/'audit')])]
    _atomic_json(OUT/'controller_commands.json',{'commands':commands,'scope':'registration and CPU audit only; no retries'})
    for phase,command in commands:
        publish(OUT,SERVICE,phase)
        wrapper=[sys.executable,'-u','-m','testing.managed_job','--receipt',str(OUT/(phase+'_job.json')),
                 '--cwd',str(ROOT),'--']+command
        child=subprocess.Popen(wrapper,cwd=ROOT)
        while True:
            try:code=child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                publish(OUT,SERVICE,phase);continue
            break
        if code:
            publish(OUT,SERVICE,phase+'_failed_no_retry')
            return code
    publish(OUT,SERVICE,'registration_and_numeric_audit_complete_review_pending')
    return 0

if __name__=='__main__':raise SystemExit(main())
