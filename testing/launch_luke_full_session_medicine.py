"""Freeze and launch the authorized huklaban1-only estimate under systemd."""
import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import socket
import subprocess

from testing.luke_full_session_medicine import MED_ARGS, sha, save

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT/'environments/rescue-production/.venv/bin/python'
MEDPY = Path('/home/huklab/anaconda3/envs/spikeinterface/bin/python')
SOURCE = Path('/mnt/NPX/Luke/20250804/rescue_pipeline_results_Luke0804_V2V1_g0_imec0/recording')
SHARED = Path('/mnt/NPX/Luke/20250804/shared_analysis/luke_improved_motion_two_machine_20260909_v1')
FILES = ['testing/__init__.py', 'testing/managed_job.py', 'testing/luke_full_session_medicine.py',
         'testing/luke_screen_sweep.py', 'testing/luke_epoch_corroboration.py', 'testing/luke_dredge_bounded.py']


def prepare(job, output):
    bundle = job/'source'
    for name in FILES:
        target = bundle/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/name, target)
    save(job/'source_manifest.json', {name: sha(bundle/name) for name in FILES})
    src = ROOT/'testing/outputs'
    control = src/'luke_screened_medicine_300s_v1/control_input'
    field = src/'luke_ap_methods_sweep_v1/arms/5sigma_relaxed_bound250__medicine/field.npz'
    paths = [SOURCE/'rescue_recording_manifest.json',
             src/'luke_common_event_screen_v1/shared_response_model.npz',
             src/'luke_peak_threshold_screen_v1/noise_uv.npy', field,
             *[control/name for name in ['input.json', 'peaks.npy', 'locations.npy']],
             src/'luke_screened_medicine_300s_v1/settings.json']
    cfg = dict(schema='luke-full-session-screened-medicine-v1', estimation_owner='huklaban1',
               output=str(output), medicine_python=str(MEDPY), medicine_settings=MED_ARGS,
               recording_manifest=str(paths[0]), recording_binary=str(SOURCE/'traces_cached_seg0.raw'),
               recording_content_sha256='2d6fac755db3182841bf121d822754f963ebada3bcb8cd9e96e0b63e6f9b7372',
               model=str(paths[1]), noise=str(paths[2]), control_input=str(control), control_field=str(field),
               cache=str(src/'luke_screened_medicine_300s_v1'), shared_output=str(SHARED),
               input_sha256={str(path): sha(path) for path in paths},
               source_manifest_sha256=sha(job/'source_manifest.json'),
               authorization='User requested one shared full-session MEDiCINe estimate here on huklaban1; no sort in this job',
               restart='No optimizer checkpoint. Reuse sealed stages only; unsealed stages require investigation.')
    save(job/'config.json', cfg)
    return bundle


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--unit', required=True)
    ap.add_argument('--job-dir', required=True, type=Path)
    ap.add_argument('--output', type=Path)
    ap.add_argument('--dummy', action='store_true')
    ap.add_argument('--proof', type=Path)
    args = ap.parse_args()
    if not args.unit.startswith('luke-') or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-' for c in args.unit):
        ap.error('Use a simple luke- service name')
    if not socket.gethostname().startswith('huklaban1'):
        raise RuntimeError('Estimation is assigned only to huklaban1')
    if not args.dummy:
        if args.output is None or args.proof is None:
            ap.error('Actual estimation requires --output and --proof')
        proof = args.proof
        receipt = json.loads((proof/'receipt.json').read_text())
        disconnected = json.loads((proof/'launcher_disconnected_state.json').read_text())
        result = json.loads((proof/'service_result.json').read_text())
        if not (receipt.get('returncode') == 0 and receipt.get('state') == 'complete'
                and disconnected.get('active_after_launcher_exit') is True
                and result.get('SERVICE_RESULT') == 'success'):
            raise RuntimeError('Launcher-disconnection proof incomplete')
        if args.output.exists():
            raise RuntimeError('Fresh launch requires a fresh output directory; investigate retries explicitly')
        if shutil.disk_usage(args.output.parent).free < 30*1024**3:
            raise RuntimeError('Need at least 30 GiB free for estimation artifacts')
    job = args.job_dir.resolve()
    job.mkdir(parents=True, exist_ok=False)
    if args.dummy:
        cwd = ROOT
        command = [str(PYTHON), '-u', '-c', "import time; print('dummy started', flush=True); time.sleep(20); print('dummy completed', flush=True)"]
    else:
        cwd = prepare(job, args.output.resolve())
        command = [str(PYTHON), '-u', '-m', 'testing.luke_full_session_medicine', '--config', str(job/'config.json')]
    wrapped = [str(PYTHON), '-u', '-m', 'testing.managed_job', '--receipt', str(job/'receipt.json'),
               '--cwd', str(cwd), '--', *command]
    finish = job/'save_service_result.py'
    finish.write_text('import json,os\nfrom pathlib import Path\nfrom datetime import datetime,timezone\n'
                     'p=Path(__file__).with_name("service_result.json"); t=p.with_suffix(".tmp")\n'
                     't.write_text(json.dumps(dict(finished_at=datetime.now(timezone.utc).isoformat(),'
                     '**{k:os.environ.get(k) for k in ["SERVICE_RESULT","EXIT_CODE","EXIT_STATUS"]})))\n'
                     'os.replace(t,p)\n')
    shell = job/'launch.sh'
    shell.write_text('#!/bin/bash\nset -eu\ncd '+shlex.quote(str(cwd))+'\n'
                     'export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4\n'
                     'export MPLCONFIGDIR='+shlex.quote(str(job/'matplotlib'))+'\n'
                     'exec '+shlex.join(wrapped)+'\n')
    shell.chmod(0o755)
    request = ['systemd-run', '--user', '--unit='+args.unit, '--collect', '--property=Type=exec',
               '--property=CPUQuota=400%', '--property=MemoryMax=64G', '--property=TasksMax=128',
               '--property=KillMode=control-group', '--property=TimeoutStopSec=30',
               '--property=StandardOutput=append:'+str(job/'stdout.log'),
               '--property=StandardError=append:'+str(job/'stderr.log'),
               '--property=ExecStopPost='+str(PYTHON)+' '+str(finish), str(shell)]
    save(job/'launch_request.json', dict(command=request, child_command=wrapped, cwd=str(cwd),
         launcher_pid=os.getpid(), host=socket.gethostname(), dummy=args.dummy,
         proof=str(args.proof) if args.proof else None, no_automatic_restart=True))
    completed = subprocess.run(request, capture_output=True, text=True)
    save(job/'launcher_result.json', dict(returncode=completed.returncode, stdout=completed.stdout, stderr=completed.stderr))
    print(completed.stdout+completed.stderr, flush=True)
    completed.check_returncode()


if __name__ == '__main__':
    main()
