from pathlib import Path

from testing.em2f_resume_after_oomd import service_command


def test_recovery_service_is_persistent_bounded_and_oom_protected():
    config = {
        "repo": "/home/huklaban5/Documents/DARTsort",
        "budgets_seconds": {"sort": 21600, "qc_phy": 3600},
    }
    command = service_command(config, Path("/tmp/output"), Path("/tmp/job"), "recover")
    for expected in (
        "--collect",
        "--property=Type=exec",
        "--property=RuntimeMaxSec=25800",
        "--property=CPUQuota=400%",
        "--property=MemoryHigh=128G",
        "--property=MemoryMax=160G",
        "--property=ManagedOOMPreference=avoid",
        "--property=TasksMax=128",
        "--property=KillMode=control-group",
        "--property=StandardOutput=append:/tmp/job/stdout.log",
        "--property=StandardError=append:/tmp/job/stderr.log",
    ):
        assert expected in command
    assert command[-5:] == [
        "run",
        "--config",
        "/tmp/output/config.json",
        "--output",
        "/tmp/output",
    ]
