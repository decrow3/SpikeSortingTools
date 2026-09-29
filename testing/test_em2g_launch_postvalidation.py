from pathlib import Path

from testing.em2g_launch_postvalidation import (
    ROOT,
    absolute_preserving_symlinks,
    service_command,
)


def test_postvalidation_service_is_persistent_and_resource_bounded():
    command = service_command(
        Path("/venv/python"), Path("/science"), Path("/job"), "em2g-test"
    )
    assert command[:4] == [
        "systemd-run",
        "--user",
        "--unit=em2g-test",
        "--property=Type=exec",
    ]
    assert "--property=RuntimeMaxSec=3600" in command
    assert "--property=CPUQuota=200%" in command
    assert "--property=MemoryMax=24G" in command
    assert "--property=KillMode=control-group" in command
    assert "--property=StandardOutput=append:/job/stdout.log" in command
    assert str(ROOT / "testing/em2g_full_postvalidation.py") in command
    assert command[-2:] == ["--output", "/science"]


def test_interpreter_path_keeps_virtualenv_symlink(tmp_path):
    base = tmp_path / "base-python"
    base.write_text("")
    venv_python = tmp_path / "venv-python"
    venv_python.symlink_to(base)
    observed = absolute_preserving_symlinks(venv_python)
    assert observed == venv_python
    assert observed.is_symlink()
