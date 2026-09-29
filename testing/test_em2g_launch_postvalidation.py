from pathlib import Path

from testing.em2g_launch_postvalidation import ROOT, service_command


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
