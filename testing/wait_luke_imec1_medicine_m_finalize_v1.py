#!/usr/bin/env python
"""Persistently wait for stage 1, then run correction-M references and scoring."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def service_state(unit: str) -> dict[str, str]:
    result = subprocess.run(
        ["systemctl", "--user", "show", unit,
         "--property=ActiveState,SubState,MainPID,ExecMainStatus"],
        check=True, capture_output=True, text=True,
    )
    return dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--upstream-unit", required=True)
    parser.add_argument("--timeout-hours", type=float, default=12.0)
    args = parser.parse_args()
    output = args.output.resolve()
    job = output / "job-m-finalize"
    job.mkdir(parents=True, exist_ok=False)
    status_path = job / "status.json"
    started = time.time()
    try:
        while True:
            state = service_state(args.upstream_unit)
            write_json(status_path, {"status": "waiting", "upstream": state,
                                     "elapsed_s": time.time() - started})
            if state.get("ActiveState") not in {"active", "activating", "reloading"}:
                break
            if time.time() - started > args.timeout_hours * 3600:
                raise TimeoutError(f"Upstream did not finish within {args.timeout_hours} hours")
            time.sleep(30)
        if state.get("ExecMainStatus") != "0":
            raise RuntimeError(f"Upstream failed: {state}")
        for required in ("sweep_complete.json", "validation.json", "selection.json"):
            if not (output / required).exists():
                raise RuntimeError(f"Upstream ended without {required}")
        script = Path(__file__).with_name("luke_imec1_medicine_m_reference_v1.py")
        command = [sys.executable, str(script), "--output", str(output)]
        write_json(status_path, {"status": "running_m", "upstream": state,
                                 "command": command, "elapsed_s": time.time() - started})
        result = subprocess.run(command)
        if result.returncode:
            raise RuntimeError(f"M postprocessor failed with {result.returncode}")
        write_json(status_path, {"status": "complete", "upstream": state,
                                 "elapsed_s": time.time() - started,
                                 "m_validation": json.loads((output / "m_validation.json").read_text())})
    except Exception as error:
        write_json(status_path, {"status": "failed", "error": repr(error),
                                 "traceback": traceback.format_exc(),
                                 "elapsed_s": time.time() - started})
        raise


if __name__ == "__main__":
    main()
