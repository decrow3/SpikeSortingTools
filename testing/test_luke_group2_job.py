import json
from unittest.mock import patch

import pytest

from testing.luke_group2_job import PLAN, EXPECTED, commands, execute


def test_group2_commands_keep_axes_separate():
    plan=json.loads(PLAN.read_text())
    stages=commands(plan)
    assert [s for s,_ in stages]==['prepare_smoke','run_smoke','prepare_long','run_long','finalize_long']
    for name,cmd in stages:
        assert 'legacy_style' not in cmd
        if name!='prepare_long':
            assert [cmd[i+1] for i,v in enumerate(cmd) if v=='--arm']==list(EXPECTED)


def test_group1_or_stale_contract_refused():
    plan=json.loads(PLAN.read_text())
    for change in [dict(arms=['rescue_12_9_motion_off']),dict(contract_digest='wrong')]:
        with pytest.raises(RuntimeError): commands({**plan,**change})


def test_failed_smoke_never_launches_long_run(tmp_path):
    with patch('testing.luke_group2_job.run_managed_command',side_effect=[0,7]) as run:
        assert execute([('prepare_smoke',['a']),('run_smoke',['b']),('run_long',['c'])],tmp_path)==7
    assert run.call_count==2
    assert json.loads((tmp_path/'status.json').read_text())['stage']=='run_smoke'
