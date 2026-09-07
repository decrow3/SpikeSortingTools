import json
from unittest.mock import patch

import pytest

from testing.luke_group3_job import EXPECTED, PLAN, REFERENCE, commands, execute


def test_group3_commands_are_serial_and_reuse_the_long_recording():
    plan = json.loads(PLAN.read_text())
    stages = commands(plan)
    assert [stage for stage, _ in stages] == [
        "prepare_smoke", "run_smoke", "run_long", "finalize_long"
    ]
    for stage, command in stages:
        arms = [command[index + 1] for index, value in enumerate(command) if value == "--arm"]
        assert arms == (list(EXPECTED) if "smoke" in stage else [REFERENCE, *EXPECTED])
    run_long = dict(stages)["run_long"]
    assert plan["reused_long_recording"] in run_long


def test_group3_rejects_stale_contract_or_membership():
    plan = json.loads(PLAN.read_text())
    for change in (
        {"arms": ["rescue_9_7_motion_off"]},
        {"contract_digest": "wrong"},
        {"reference_arm": "rescue_9_9_motion_off"},
    ):
        with pytest.raises(RuntimeError):
            commands({**plan, **change})


def test_failed_smoke_prevents_group3_long_run(tmp_path):
    with patch("testing.luke_group3_job.run_managed_command", side_effect=[0, 9]) as run:
        assert execute(
            [("prepare_smoke", ["a"]), ("run_smoke", ["b"]), ("run_long", ["c"])],
            tmp_path,
        ) == 9
    assert run.call_count == 2
    status = json.loads((tmp_path / "status.json").read_text())
    assert status == {"state": "failed", "stage": "run_smoke", "returncode": 9}
