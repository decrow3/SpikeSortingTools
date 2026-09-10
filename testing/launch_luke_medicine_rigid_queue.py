"""Freeze a persistent rigid-correction/KS4 queue after the current MEDiCINe job."""
import argparse
import json
from pathlib import Path
import shlex
import shutil
import socket
import subprocess

from testing.luke_full_session_medicine import save, sha

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT/'environments/rescue-production/.venv/bin/python'
DEP_JOB = Path('/media/huklab/Data/luke_full_session_medicine_20260909_v1_job')
DEP_OUTPUT = Path('/media/huklab/Data/luke_full_session_medicine_20260909_v1')
BASE = Path('/mnt/NPX/Luke/20250804/rescue_pipeline_results_Luke0804_V2V1_g0_imec0')
SHARED = Path('/mnt/NPX/Luke/20250804/shared_analysis/luke_improved_motion_two_machine_20260909_v1')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--unit', required=True)
    ap.add_argument('--job-dir', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--dummy', action='store_true')
    ap.add_argument('--proof', type=Path)
    args = ap.parse_args()
    if not socket.gethostname().startswith('huklaban1'):
        raise RuntimeError('Rigid queue assigned to huklaban1')
    if not args.unit.startswith('luke-') or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-' for c in args.unit):
        raise ValueError('Invalid unit name')
    if args.output.exists():
        raise RuntimeError('Existing output; no automatic retry or overwrite')
    if not args.dummy:
        if args.proof is None:
            raise RuntimeError('Queue requires launcher-disconnection proof')
        receipt = json.loads((args.proof/'receipt.json').read_text())
        state = json.loads((args.proof/'launcher_disconnected_state.json').read_text())
        finish = json.loads((args.proof/'service_result.json').read_text())
        if not (receipt.get('state') == 'complete' and receipt.get('returncode') == 0 and
                state.get('active_after_launcher_exit') is True and finish.get('SERVICE_RESULT') == 'success'):
            raise RuntimeError('Dummy survival/exit checks incomplete')
        linger = subprocess.check_output(['loginctl', 'show-user', 'huklab', '-p', 'Linger', '--value'], text=True).strip()
        if linger != 'yes':
            raise RuntimeError('Independent user-manager lifetime not configured')
    job = args.job_dir.resolve(); job.mkdir(parents=True, exist_ok=False)
    bundle = job/'source'
    # Freeze executable source and lockfile so future pulls cannot alter queued work.
    files = [p for package in ['pipeline', 'testing'] for p in (ROOT/package).glob('*.py')]
    files += [ROOT/'environments/rescue-production/uv.lock']
    for path in files:
        target = bundle/path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    hashes = {str(p.relative_to(ROOT)): sha(bundle/p.relative_to(ROOT)) for p in files}
    from pipeline.runtime import production_environment_contract, validate_production_environment
    environment = validate_production_environment(require_cuda=not args.dummy)
    from pipeline.kilosort_compat import kilosort_compatibility_receipt
    compatibility = kilosort_compatibility_receipt()
    if not args.dummy and not compatibility['applied']:
        raise RuntimeError('Known KS4 compatibility repair must already be installed before queueing')
    pins = [BASE/'recording/rescue_recording_manifest.json', BASE/'recording/binary.json',
            BASE/'kilosort4/rescue_sort_manifest.json', DEP_JOB/'config.json', DEP_JOB/'source_manifest.json',
            DEP_JOB/'launch_request.json']
    cfg = dict(schema='luke-medicine-rigid-queue-v1', output=str(args.output.resolve()),
        estimation_output=str(DEP_OUTPUT), estimation_job=str(DEP_JOB),
        estimation_unit='luke-full-session-medicine-20260909-v1.service',
        recording=str(BASE/'recording'), baseline_manifest=str(BASE/'kilosort4/rescue_sort_manifest.json'),
        recording_content_sha256='2d6fac755db3182841bf121d822754f963ebada3bcb8cd9e96e0b63e6f9b7372',
        shared=str(SHARED), pinned_inputs={str(p):sha(p) for p in pins}, source_sha256=hashes,
        production_environment=production_environment_contract(), preflight_environment=environment,
        compatibility=compatibility, terminal_margin_limit_s=1., sort_scratch_reserve_bytes=150*1024**3,
        application=dict(operator='SpikeInterface 0.102.1 InterpolateMotionRecording',
                         method='kriging', sigma_um=20., p=1, support_margin_um=60.,
                         border_mode='remove_channels', interpolation_time_bin_s=.25,
                         input_dtype='float32', output_dtype='int16', rounding='nearest',
                         overflow='fail without clipping or wrapping',
                         spatial_application='Full source geometry before common interior selection',
                         time_policy='Recording-relative field plus acquisition origin; nearest bin at terminal margins <=1s'),
        hold_paths=[str(job/'HOLD.json'), str(args.output.resolve()/'HOLD.json')],
        authorization='User explicitly requested queued rigid correction and Kilosort after estimation, independent of chat; new development run, not cancelled native-rigid v1',
        no_automatic_restart=True, checkpoint='No within-sort checkpoint; interruption requires whole-sort restart after investigation',
        scientific_policy='Proceed as explicitly authorized development experiment after implementation-integrity checks; preserve requires_review labels and do not claim field qualification')
    save(job/'config.json', cfg)
    save(job/'source_manifest.json', hashes)
    command = [str(PYTHON), '-u', '-m', 'testing.luke_medicine_rigid_queue', '--config', str(job/'config.json')]
    if args.dummy:
        command.append('--dummy')
    wrapped = [str(PYTHON), '-u', '-m', 'testing.managed_job', '--receipt', str(job/'receipt.json'),
               '--cwd', str(bundle), '--', *command]
    script = job/'launch.sh'
    script.write_text('#!/bin/bash\nset -eu\ncd '+shlex.quote(str(bundle))+'\n'
        'export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4\n'
        'export MPLCONFIGDIR='+shlex.quote(str(job/'matplotlib'))+'\n'
        'export NUMBA_CACHE_DIR='+shlex.quote(str(job/'numba'))+'\n'
        'exec '+shlex.join(wrapped)+'\n')
    script.chmod(0o755)
    finish = job/'save_service_result.py'
    finish.write_text('import json,os\nfrom pathlib import Path\nfrom datetime import datetime,timezone\n'
        'p=Path(__file__).with_name("service_result.json"); t=p.with_suffix(".tmp")\n'
        't.write_text(json.dumps(dict(finished_at=datetime.now(timezone.utc).isoformat(),'
        '**{k:os.environ.get(k) for k in ["SERVICE_RESULT","EXIT_CODE","EXIT_STATUS"]})))\n'
        'os.replace(t,p)\n')
    request = ['systemd-run','--user','--unit='+args.unit,'--collect','--property=Type=exec',
               '--property=CPUQuota=800%','--property=MemoryHigh=128G','--property=MemoryMax=160G',
               '--property=TasksMax=256','--property=KillMode=control-group','--property=TimeoutStopSec=30',
               '--property=StandardOutput=append:'+str(job/'stdout.log'),
               '--property=StandardError=append:'+str(job/'stderr.log'),
               '--property=ExecStopPost='+str(PYTHON)+' '+str(finish),str(script)]
    import os
    save(job/'launch_request.json',dict(command=request,child_command=wrapped,launcher_pid=os.getpid(),dummy=args.dummy,
                                      unit=args.unit,host=socket.gethostname(),proof=str(args.proof)))
    result = subprocess.run(request, capture_output=True, text=True)
    save(job/'launcher_result.json',dict(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr))
    print(result.stdout+result.stderr,flush=True); result.check_returncode()


if __name__ == '__main__':
    main()
