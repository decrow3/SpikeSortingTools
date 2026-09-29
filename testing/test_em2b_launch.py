from pathlib import Path

from testing.em2b_launch import PROJECT_ROOT, service_command


def test_service_command_exposes_frozen_extractor_without_extending_wall_cap():
    config = {
        "repo": "/home/huklaban5/Documents/DARTsort",
        "budgets_seconds": {
            "preprocess": 1200,
            "detect": 720,
            "motion": 60,
            "sort": 1080,
            "qc_phy": 240,
        },
    }
    command = service_command(config, Path("/tmp/arm"), "unit", preflight=False)
    assert "--property=RuntimeMaxSec=3480" in command
    expected = (
        "--setenv=PYTHONPATH=/home/huklaban5/Documents/DARTsort:" + str(PROJECT_ROOT)
    )
    assert expected in command
    assert command[-5:] == [
        "run",
        "--config",
        "/tmp/arm/config.json",
        "--output",
        "/tmp/arm",
    ]
