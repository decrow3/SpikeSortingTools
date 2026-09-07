"""Run the authorized Group 2 smoke and long-strip stages under a job manager."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

from testing.development_ladder import load_contract
from testing.development_runner import _atomic_json
from testing.ladder_sorter import NAMED_CONFIGS
from testing.managed_job import run_managed_command
from pipeline.runtime import validate_production_environment

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / 'testing/configs/luke0804_group2_huklaban1_v1.json'
EXPECTED = ('rescue_10_9_motion_off', 'rescue_9_9_motion_off', 'rescue_9_8_motion_off')


def commands(plan):
    contract = load_contract(ROOT / plan['contract_path'])
    if contract.digest != plan['contract_digest'] or tuple(plan['arms']) != EXPECTED:
        raise RuntimeError('Group 2 membership or contract changed')
    by_name = {c['name']: c for c in contract.candidates}
    for name, thresholds in zip(EXPECTED, [(10,9),(9,9),(9,8)]):
        params = NAMED_CONFIGS[by_name[name]['sorter_config']].params()
        if params['do_correction'] or (params['Th_universal'],params['Th_learned']) != thresholds:
            raise RuntimeError('Group 2 must vary thresholds with motion off')
    root = Path(plan['output_root'])
    prefix = [sys.executable, '-u', '-m', 'testing.run_development_ladder']
    cfg = ['--config', str(ROOT / plan['contract_path'])]
    arms = [value for name in EXPECTED for value in ('--arm',name)]
    smoke = ['--smoke-start-s',str(plan['smoke_start_s']), '--smoke-duration-s',str(plan['smoke_duration_s'])]
    return [
        ('prepare_smoke', prefix+['prepare-smoke']+cfg+arms+smoke+['--output-root',str(root/'smoke/recording')]),
        ('run_smoke', prefix+['run-smoke']+cfg+arms+smoke+['--recording-dir',str(root/'smoke/recording'),'--output-root',str(root/'smoke/arms'),'--group-id','group_2_smoke']),
        ('prepare_long', prefix+['prepare-strip']+cfg+['--output-root',str(root/'long/recording')]),
        ('run_long', prefix+['run-arms']+cfg+arms+['--recording-dir',str(root/'long/recording'),'--output-root',str(root/'long/arms'),'--group-id',plan['group_id']]),
        ('finalize_long', prefix+['finalize-arms']+cfg+arms+['--recording-dir',str(root/'long/recording'),'--output-root',str(root/'long/arms')]),
    ]


def execute(stages, root):
    for stage, command in stages:
        _atomic_json(root/'status.json', dict(state='running',stage=stage))
        code = run_managed_command(command,receipt_path=root/'jobs'/f'{stage}.json',cwd=ROOT)
        if code:
            _atomic_json(root/'status.json',dict(state='failed',stage=stage,returncode=code))
            return code
    _atomic_json(root/'status.json',dict(state='complete',comparison='Awaiting shared Group 1 reference'))
    return 0


def main():
    plan = json.loads(PLAN.read_text())
    stages = commands(plan)
    root = Path(plan['output_root'])
    # Refuse an absent mount rather than filling the system disk underneath it.
    if not Path('/media/huklab/Data').is_mount():
        raise RuntimeError('selected local data disk is not mounted')
    if shutil.disk_usage('/media/huklab/Data').free < plan['minimum_free_gib']*1024**3:
        raise RuntimeError('insufficient local output capacity')
    for label, code in [('success',0),('failure',7)]:
        evidence = json.loads((ROOT/f'testing/outputs/group2_preparation/{label}.json').read_text())
        if evidence.get('returncode') != code or evidence.get('state') != ('complete' if code==0 else 'failed'):
            raise RuntimeError('independent-manager test has not passed')
    environment = validate_production_environment(require_cuda=True)
    root.mkdir(parents=True,exist_ok=True)
    if (root/'launch.json').exists():
        raise RuntimeError('launch already exists; inspect prior attempt rather than retry automatically')
    _atomic_json(root/'launch.json',dict(plan=plan,stages=stages,environment=environment))
    return execute(stages,root)


if __name__=='__main__':
    sys.exit(main())
