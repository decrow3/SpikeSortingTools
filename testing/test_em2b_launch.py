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


def test_full_session_runtime_cap_covers_all_stage_budgets():
    import json

    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs/em2f_full_rounded_kriging.v1.json").read_text())
    command = service_command(config, Path("/tmp/full"), "full", preflight=False)
    expected = sum(config["budgets_seconds"].values()) + 180
    assert f"--property=RuntimeMaxSec={expected}" in command
    assert config["free_space_reserve_gib"] == 450
