"""Launch an authorized motion diagnostic in an independent systemd user service."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / 'environments/rescue-production/.venv/bin/python'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--unit', required=True)
    parser.add_argument('--module', required=True, choices=[
        'testing.luke_shallow_reference_overnight_v1',
        'testing.luke_motion_transfer_overnight_v1'])
    parser.add_argument('--post-module', choices=['testing.luke_shallow_reference_overnight_report_v1'])
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not args.unit.startswith('luke-') or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-' for c in args.unit):
        parser.error('unit must be a simple luke- diagnostic name')
    state = subprocess.run(['systemctl', '--user', 'list-units', '--type=service',
                            '--state=running,activating', '--plain', '--no-legend',
                            'luke-*-overnight*'], text=True, capture_output=True, check=True)
    if len(state.stdout.strip().splitlines()) >= 2:
        raise RuntimeError('Two overnight services already active; defer this launch')
    job = ROOT / 'testing/outputs' / (args.unit.replace('-', '_') + '_job')
    job.mkdir(exist_ok=False)
    extra = args.arguments[1:] if args.arguments[:1] == ['--'] else args.arguments
    command = [str(PYTHON), '-m', 'testing.managed_job', '--receipt', str(job / 'receipt.json'),
               '--cwd', str(ROOT), '--', str(PYTHON), '-m', args.module, *extra]
    post_command = ([str(PYTHON), '-m', 'testing.managed_job', '--receipt', str(job / 'report_receipt.json'),
                     '--cwd', str(ROOT), '--', str(PYTHON), '-m', args.post_module] if args.post_module else None)
    launch = job / 'launch.sh'
    launch.write_text('#!/bin/bash\nset -eu\ncd ' + shlex.quote(str(ROOT)) + '\n'
                      'export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1\n'
                      'export MPLCONFIGDIR=' + shlex.quote('/tmp/' + args.unit) + '\n' +
                      ((shlex.join(command) + '\nexec ' + shlex.join(post_command)) if post_command
                       else ('exec ' + shlex.join(command))) + '\n')
    launch.chmod(0o755)
    # systemd calls this even when its main process is terminated, independently of chat.
    finish = job / 'save_service_result.py'
    finish.write_text('import json, os\nfrom pathlib import Path\nfrom datetime import datetime, timezone\n'
                      'p = Path(__file__).with_name("service_result.json")\n'
                      'tmp = p.with_suffix(".tmp")\n'
                      'tmp.write_text(json.dumps(dict(finished_at=datetime.now(timezone.utc).isoformat(), '
                      '**{k:os.environ.get(k) for k in ["SERVICE_RESULT","EXIT_CODE","EXIT_STATUS"]}), indent=2))\n'
                      'os.replace(tmp, p)\n')
    request = ['systemd-run', '--user', '--unit=' + args.unit, '--collect',
               '--property=Type=exec', '--property=CPUQuota=400%', '--property=MemoryMax=48G',
               '--property=TasksMax=96', '--property=KillMode=control-group', '--property=TimeoutStopSec=30',
               '--property=WorkingDirectory=' + str(ROOT),
               '--property=StandardOutput=append:' + str(job / 'stdout.log'),
               '--property=StandardError=append:' + str(job / 'stderr.log'),
               '--property=ExecStopPost=' + str(PYTHON) + ' ' + str(finish), str(launch)]
    (job / 'launch_request.json').write_text(json.dumps(dict(command=request, child_command=command, post_command=post_command,
        parent_pid=os.getpid(), resume='Explicit relaunch with a fresh unit/receipt after investigating interruption; module validates completed checkpoints.'), indent=2))
    result = subprocess.run(request, text=True, capture_output=True)
    (job / 'launcher_result.json').write_text(json.dumps(dict(returncode=result.returncode,
        stdout=result.stdout, stderr=result.stderr), indent=2))
    print(result.stdout + result.stderr, flush=True)
    result.check_returncode()


if __name__ == '__main__':
    main()
