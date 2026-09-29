#!/usr/bin/env python3
"""Resume EM.2f while validating and reusing already-materialized link inputs."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys


def read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def validate_existing_link_inputs(source: Path, target: Path, link_step: str) -> dict[str, object]:
    receipt_path = target / "link-input-copy.json"
    receipt = read(receipt_path)
    if receipt.get("status") != "complete" or receipt.get("link_step") != link_step:
        raise RuntimeError("existing link-input receipt is incompatible")
    if Path(str(receipt.get("source"))).resolve() != source.resolve():
        raise RuntimeError("existing link-input source changed")
    for item in receipt["files"]:
        rel = Path(str(item["path"]))
        src, dst = source / rel, target / rel
        if item["kind"] == "directory":
            dst_bytes = sum(p.stat().st_size for p in dst.rglob("*") if p.is_file())
            src_bytes = sum(p.stat().st_size for p in src.rglob("*") if p.is_file())
        else:
            dst_bytes, src_bytes = dst.stat().st_size, src.stat().st_size
        if dst_bytes != item["bytes"] or src_bytes != item["bytes"]:
            raise RuntimeError(f"materialized link input size changed: {rel}")
    return receipt


def load_frozen_pipeline(output: Path):
    path = output / "source/pipeline.py"
    source_dir = str(path.parent)
    if source_dir not in sys.path:
        sys.path.insert(0, source_dir)
    spec = importlib.util.spec_from_file_location("em2f_frozen_pipeline", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--job", required=True, type=Path)
    args = parser.parse_args()
    output, job = args.output.resolve(), args.job.resolve()
    config = read(args.config.resolve())
    frozen = load_frozen_pipeline(output)

    failure = output / "failure.json"
    if failure.exists():
        preserved = job / "PREVIOUS_FAILURE.json"
        if preserved.exists():
            raise FileExistsError(preserved)
        shutil.move(failure, preserved)

    frozen.materialize_link_inputs = validate_existing_link_inputs
    frozen.stage_worker("sort", config, output)

    command = [
        sys.executable,
        "-u",
        str(output / "source/pipeline.py"),
        "run",
        "--config",
        str(args.config.resolve()),
        "--output",
        str(output),
    ]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
