#!/usr/bin/env python3
"""Launch EM.2b arms with the frozen runner and explicit extractor import path."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER_SOURCE = Path(
    "/mnt/NPX/Luke/DARTsort_motion_experiments/"
    "luke0804-imec1-ao-d2l-v2/windows/W2/runs/D2L/source"
)
sys.path.insert(0, str(RUNNER_SOURCE))
import pipeline as production  # noqa: E402


def service_command(
    config: dict[str, object], output: Path, unit: str, *, preflight: bool
) -> list[str]:
    repo = Path(config["repo"]).resolve()
    python = repo / ".venv/bin/python"
    pythonpath = os.pathsep.join((str(repo), str(PROJECT_ROOT)))
    runtime = 120 if preflight else sum(config["budgets_seconds"].values()) + 180
    worker = (
        [
            str(python),
            str(PROJECT_ROOT / "testing/em2b_runtime_preflight.py"),
            "--config",
            str(output / "config.json"),
            "--output",
            str(output),
        ]
        if preflight
        else [
            str(python),
            "-u",
            str(output / "source/pipeline.py"),
            "run",
            "--config",
            str(output / "config.json"),
            "--output",
            str(output),
        ]
    )
    return [
        "systemd-run",
        "--user",
        "--unit=" + unit,
        "--property=RuntimeMaxSec=" + str(runtime),
        "--property=TimeoutStopSec=15",
        "--property=WorkingDirectory=" + str(repo),
        "--setenv=PYTHONPATH=" + pythonpath,
        "--setenv=OMP_NUM_THREADS=1",
        "--setenv=OPENBLAS_NUM_THREADS=1",
        *worker,
    ]


def prepare(config_path: Path, output: Path, *, resume: bool) -> dict[str, object]:
    config = production.launch_config(config_path)
    production.validate(config)
    repo = Path(config["repo"]).resolve()
    python = repo / ".venv/bin/python"
    if not python.is_file():
        raise FileNotFoundError(python)
    if output.exists():
        if (output / "receipt.json").exists():
            raise FileExistsError("completed output is immutable")
        if not resume:
            raise FileExistsError("use a fresh output or explicit --resume")
        if production.read(output / "config.json") != config:
            raise RuntimeError("resume configuration changed")
        if production.read(output / "provenance.json")["repo_state"] != production.repo_state(repo):
            raise RuntimeError("resume repository code or lock changed")
        for source in production.HERE.glob("*.py"):
            if production.digest(source) != production.digest(output / "source" / source.name):
                raise RuntimeError("resume runner source changed")
    else:
        output.mkdir(parents=True)
        (output / "source").mkdir()
        for source in production.HERE.glob("*.py"):
            shutil.copy2(source, output / "source" / source.name)
        production.write(output / "config.json", config)
        production.write(
            output / "provenance.json",
            {
                "source_hashes": {
                    path.name: production.digest(path)
                    for path in (output / "source").glob("*.py")
                },
                "repo_state": production.repo_state(repo),
                "git_status": subprocess.check_output(
                    ["git", "-C", str(repo), "status", "--short"], text=True
                ),
                "lock_sha256": production.digest(repo / "uv.lock"),
                "analysis_config_sha256": None,
                "extractor_project_root": str(PROJECT_ROOT),
                "extractor_source_sha256": production.digest(
                    PROJECT_ROOT / "npx_preprocessing/motion/interpolated_motion_si.py"
                ),
                "launcher_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            },
        )
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("preflight", "launch"))
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    config = prepare(args.config.resolve(), output, resume=args.resume)
    suffix = "-preflight" if args.action == "preflight" else ""
    unit = "em2b-" + hashlib.sha256((str(output) + suffix).encode()).hexdigest()[:16]
    state = subprocess.run(
        ["systemctl", "--user", "show", unit, "--property=ActiveState", "--value"],
        capture_output=True,
        text=True,
    )
    if state.stdout.strip() in ("active", "activating", "deactivating"):
        raise RuntimeError("existing service live; do not restart")
    subprocess.run(["systemctl", "--user", "reset-failed", unit], capture_output=True)
    command = service_command(config, output, unit, preflight=args.action == "preflight")
    subprocess.run(command, check=True)
    dispatch = {
        "schema": "em2b-explicit-extractor-launch-v1",
        "service": unit,
        "action": args.action,
        "command": command,
        "started_epoch": time.time(),
    }
    production.write(output / f"{args.action}-dispatch.json", dispatch)
    print(json.dumps({"service": unit, "output": str(output), "action": args.action}))


if __name__ == "__main__":
    main()
