"""Independent systemd launcher for the cross-dataset estimation-only pilot."""
import argparse
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--unit', required=True)
    ap.add_argument('--job', type=Path, required=True)
    ap.add_argument('--output', type=Path)
    ap.add_argument('--proof', type=Path)
    ap.add_argument('--dummy', action='store_true')
    a = ap.parse_args()
    assert a.unit.startswith('cross-motion-') and all(c in 'abcdefghijklmnopqrstuvwxyz0123456789-' for c in a.unit)
    if not a.dummy:
        assert a.proof and a.output
        proof = json.loads((a.proof/'receipt.json').read_text())
        disconnection = json.loads((a.proof/'disconnection_state.json').read_text())
        assert proof['state'] == 'complete' and proof['returncode'] == 0
        assert 'ActiveState=active' in disconnection['stdout'] and 'SubState=running' in disconnection['stdout']
        assert (a.proof/'dummy_done.txt').read_text() == 'complete\n'
    job = a.job.resolve()
    job.mkdir(parents=True, exist_ok=False)
    for name in ['cross_dataset_fast_motion.py', 'managed_job.py']:
        shutil.copy2(ROOT/'testing'/name, job/name)
    if a.dummy:
        child = [sys.executable, '-c', "import time;from pathlib import Path;time.sleep(25);Path(" + repr(str(job/'dummy_done.txt')) + ").write_text('complete\\n')"]
    else:
        child = [sys.executable, str(job/'cross_dataset_fast_motion.py'), 'run', '--output', str(a.output.resolve())]
    command = [sys.executable,str(job/'managed_job.py'),'--receipt',str(job/'receipt.json'),'--cwd',str(ROOT),'--',*child]
    stop = job/'save_service_result.py'
    stop.write_text('import json,os,time\nfrom pathlib import Path\nPath(__file__).with_name("service_result.json").write_text(json.dumps(dict(finished_at=time.time(),**{k:os.environ.get(k) for k in ["SERVICE_RESULT","EXIT_CODE","EXIT_STATUS"]}),indent=2))\n')
    request = ['systemd-run','--user','--unit='+a.unit,
        '--property=Type=exec','--property=RemainAfterExit=yes','--property=CPUQuota=400%',
        '--property=MemoryMax=32G','--property=TasksMax=128','--property=KillMode=control-group',
        '--property=WorkingDirectory='+str(ROOT),'--property=TimeoutStopSec=30',
        '--property=StandardOutput=append:'+str(job/'stdout.log'),
        '--property=StandardError=append:'+str(job/'stderr.log'),
        '--property=ExecStopPost='+shlex.join([sys.executable,str(stop)]),
        '--setenv=OPENBLAS_NUM_THREADS=4','--setenv=OMP_NUM_THREADS=4','--setenv=MKL_NUM_THREADS=4',
        '--setenv=NUMBA_CACHE_DIR=/tmp/crossmotion-numba','--setenv=MPLCONFIGDIR=/tmp/crossmotion-mpl',
        *command]
    (job/'launch_request.json').write_text(json.dumps(dict(command=request,child_command=child,
        output=str(a.output),proof=str(a.proof),no_optimizer_checkpoint=True),indent=2))
    result = subprocess.run(request,text=True,capture_output=True)
    (job/'launcher_result.json').write_text(json.dumps(dict(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr),indent=2))
    print(result.stdout+result.stderr,flush=True)
    result.check_returncode()


if __name__ == '__main__':
    main()
