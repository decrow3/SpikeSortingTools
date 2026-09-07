import json
from unittest.mock import patch

import pytest

from testing.luke_group1_job import EXPECTED, PLAN, commands, execute


def test_group1_commands_keep_motion_axis_and_skip_completed_smoke():
    stages = commands(json.loads(PLAN.read_text()))
    assert [stage for stage, _ in stages] == ["prepare_long", "run_long", "finalize_long"]
    for stage, command in stages:
        assert "legacy_style" not in command
        assert not any(name.startswith("rescue_10_9") or name.startswith("rescue_9_") for name in command)
        if stage != "prepare_long":
            assert [command[i + 1] for i, value in enumerate(command) if value == "--arm"] == list(EXPECTED)


def test_group2_or_stale_contract_is_refused():
    plan = json.loads(PLAN.read_text())
    for change in [
        {"arms": ["rescue_10_9_motion_off"]},
        {"contract_digest": "wrong"},
    ]:
        with pytest.raises(RuntimeError):
            commands({**plan, **change})


def test_failed_long_preparation_never_launches_sort(tmp_path):
    with patch("testing.luke_group1_job.run_managed_command", side_effect=[7]) as run:
        assert execute(
            [("prepare_long", ["a"]), ("run_long", ["b"]), ("finalize_long", ["c"])],
            tmp_path,
        ) == 7
    assert run.call_count == 1
    status = json.loads((tmp_path / "status.json").read_text())
    assert status == {"state": "failed", "stage": "prepare_long", "returncode": 7}
