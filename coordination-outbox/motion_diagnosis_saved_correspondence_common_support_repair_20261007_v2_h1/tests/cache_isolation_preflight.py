#!/usr/bin/env python3
"""Fail-closed launch/cache isolation check for the H5 managed execution."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


SOURCE_SHA256 = "3e40224aaed11429725c1c081ed68b5cbd3debf9c7c518128345b2e860ff0728"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def regular_files(path: Path) -> list[str]:
    if not path.is_dir():
        return []
    return sorted(str(p.relative_to(path)) for p in path.rglob("*") if p.is_file())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--fixture-cache", type=Path, required=True)
    parser.add_argument("--execution-cache", type=Path, required=True)
    parser.add_argument("--pythonpath", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()

    python = args.python.resolve()
    source = args.source.resolve()
    fixture_cache = args.fixture_cache.resolve()
    execution_cache = args.execution_cache.resolve()
    if fixture_cache == execution_cache:
        raise RuntimeError("fixture and execution NUMBA cache paths are identical")
    if not python.is_file() or not os.access(python, os.X_OK):
        raise RuntimeError(f"service interpreter is not executable: {python}")
    if sha256(source) != SOURCE_SHA256:
        raise RuntimeError("exact service source hash differs")
    fixture_members = regular_files(fixture_cache)
    if not fixture_members:
        raise RuntimeError("fixture cache is absent/empty; run the exact fixture tests there first")
    if execution_cache.exists() and any(execution_cache.iterdir()):
        raise RuntimeError("execution cache must be fresh and empty")
    execution_cache.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update({
        "PYTHONDONTWRITEBYTECODE": "1",
        "NUMBA_CACHE_DIR": str(execution_cache),
        "PYTHONPATH": str(args.pythonpath.resolve()),
    })
    proc = subprocess.run(
        [str(python), str(source), "--help"],
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=60,
        check=False,
    )
    if proc.returncode != 0 or "--contract-sha256" not in proc.stdout:
        raise RuntimeError(f"exact interpreter/source-path preflight failed: rc={proc.returncode}\n{proc.stdout}")
    receipt = {
        "schema": "corrected-correspondence-cache-isolation-preflight-v1",
        "status": "PASS",
        "python": str(python),
        "source": str(source),
        "source_sha256": sha256(source),
        "fixture_cache": str(fixture_cache),
        "fixture_cache_file_count": len(fixture_members),
        "execution_cache": str(execution_cache),
        "execution_cache_file_count_after_help": len(regular_files(execution_cache)),
        "pythonpath": str(args.pythonpath.resolve()),
        "exact_argv_check": [str(python), str(source), "--help"],
        "service_requirement": "launch with this exact execution cache; do not run dynamic-import tests in it",
    }
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
