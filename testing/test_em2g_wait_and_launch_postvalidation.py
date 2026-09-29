import json
from pathlib import Path

from testing.em2g_wait_and_launch_postvalidation import (
    BOUND_RELATIVE_INPUTS,
    absolute_preserving_symlinks,
    bound_inputs,
    terminal_decision,
    waiter_command,
    waiter_unit_name,
)


def test_waiter_is_persistent_bounded_and_names_handoff_inputs():
    command = waiter_command(
        Path("/venv/python"),
        Path("/state"),
        Path("/validation"),
        Path("/job"),
        Path("/sort"),
        "sort.service",
        "wait.service",
        65000,
    )
    assert command[:4] == [
        "systemd-run",
        "--user",
        "--unit=wait.service",
        "--property=Type=exec",
    ]
    assert "--property=RuntimeMaxSec=65300" in command
    assert "--property=CPUQuota=50%" in command
    assert "--property=MemoryMax=1G" in command
    assert "--property=KillMode=control-group" in command
    assert command[-2:] == ["--timeout-seconds", "65000"]
    assert command.count("wait") == 1


def test_handoff_binds_every_executed_source_and_contract():
    observed = bound_inputs()
    assert set(observed) == set(BOUND_RELATIVE_INPUTS)
    assert all(len(digest) == 64 for digest in observed.values())
    assert "testing/em2g_full_postvalidation.py" in observed
    assert "configs/em2g_full_slice_validation.v1.json" in observed


def test_interpreter_path_keeps_virtualenv_symlink(tmp_path):
    base = tmp_path / "base-python"
    base.write_text("")
    venv_python = tmp_path / "venv-python"
    venv_python.symlink_to(base)
    observed = absolute_preserving_symlinks(venv_python)
    assert observed == venv_python
    assert observed.is_symlink()


def test_replacement_handoff_uses_a_fresh_service_identity(tmp_path):
    output = tmp_path / "validation"
    first = waiter_unit_name(tmp_path / "handoff-v1", output)
    second = waiter_unit_name(tmp_path / "handoff-v2", output)
    assert first.startswith("em2g-wait-")
    assert second.startswith("em2g-wait-")
    assert first != second


def test_terminal_decision_requires_successful_receipt(tmp_path):
    active = {"ActiveState": "active", "SubState": "running"}
    assert terminal_decision(tmp_path, active) == ("wait", None)

    (tmp_path / "receipt.json").write_text(
        json.dumps({"status": "complete", "exit_status": 0})
    )
    decision, receipt = terminal_decision(tmp_path, active)
    assert decision == "launch"
    assert receipt == {"status": "complete", "exit_status": 0}


def test_terminal_decision_preserves_failure_evidence(tmp_path):
    (tmp_path / "failure.json").write_text(json.dumps({"error": "boom"}))
    decision, evidence = terminal_decision(
        tmp_path, {"ActiveState": "inactive", "Result": "exit-code"}
    )
    assert decision == "fail"
    assert evidence["reason"] == "sort failure receipt"
    assert evidence["failure"]["error"] == "boom"
