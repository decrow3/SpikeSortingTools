"""Receipt-writing wrapper for commands launched by an independent job manager."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path


JOB_SCHEMA = "spikesorting-managed-job-v1"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def run_managed_command(command, *, receipt_path: Path | str, cwd: Path | str) -> int:
    """Run one command and persist its exact request and final return code."""
    command = [str(value) for value in command]
    if not command:
        raise ValueError("managed command must not be empty")
    receipt_path = Path(receipt_path).resolve()
    cwd = Path(cwd).resolve()
    request = {
        "schema_version": JOB_SCHEMA,
        "command": command,
        "cwd": str(cwd),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    if receipt_path.exists():
        existing = json.loads(receipt_path.read_text())
        raise RuntimeError(
            f"managed job receipt already exists with state={existing.get('state')!r}; "
            "preserve it and choose a new receipt path"
        )
    receipt = {
        **request,
        "state": "running",
        "wrapper_pid": os.getpid(),
        "started_at": _now(),
    }
    _atomic_json(receipt_path, receipt)
    try:
        result = subprocess.run(command, cwd=cwd)
    except BaseException as error:
        receipt.update(state="wrapper_error", error=repr(error), finished_at=_now())
        _atomic_json(receipt_path, receipt)
        raise
    receipt.update(
        state="complete" if result.returncode == 0 else "failed",
        returncode=result.returncode,
        finished_at=_now(),
    )
    _atomic_json(receipt_path, receipt)
    return result.returncode


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--cwd", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    return run_managed_command(command, receipt_path=args.receipt, cwd=args.cwd)


if __name__ == "__main__":
    sys.exit(main())
