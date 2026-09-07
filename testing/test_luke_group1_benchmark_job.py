import json
from unittest.mock import patch

from testing.luke_group1_benchmark_job import BASELINE, CANDIDATES, commands, execute
from testing.luke_group1_job import EXPECTED, PLAN


def test_benchmark_commands_compare_only_group1_candidates_to_frozen_baseline():
    stages = commands(json.loads(PLAN.read_text()))
    assert len(stages) == 2
    assert [name for name, _ in stages] == [f"{name}_vs_{BASELINE}" for name in CANDIDATES]
    for (_, command), candidate in zip(stages, CANDIDATES):
        assert command[command.index("--baseline") + 1] == EXPECTED[0]
        assert command[command.index("--candidate") + 1] == candidate
        assert all("rescue_10_9" not in value and "rescue_9_" not in value for value in command)


def test_benchmark_failure_stops_before_second_comparison(tmp_path):
    with patch(
        "testing.luke_group1_benchmark_job.run_managed_command", side_effect=[9]
    ) as run:
        assert execute([("first", ["a"]), ("second", ["b"])], tmp_path) == 9
    assert run.call_count == 1
    assert json.loads((tmp_path / "status.json").read_text()) == {
        "state": "failed", "stage": "first", "returncode": 9,
    }


def test_benchmark_success_records_both_comparisons(tmp_path):
    with patch(
        "testing.luke_group1_benchmark_job.run_managed_command", side_effect=[0, 0]
    ) as run:
        assert execute([("first", ["a"]), ("second", ["b"])], tmp_path) == 0
    assert run.call_count == 2
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["state"] == "complete"
    assert status["comparisons"] == [f"{candidate}_vs_{BASELINE}" for candidate in CANDIDATES]
