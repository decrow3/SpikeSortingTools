#!/usr/bin/env python3
"""Persistently wait for all EN raw sort inputs, then run frozen Tier 1."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

from testing.en_tier1_panel import audit_inputs, run


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def wait_for_inputs(config_path: Path, status_path: Path, poll_seconds: float = 60.0) -> dict:
    while True:
        config = json.loads(config_path.read_text())
        audit = audit_inputs(config)
        if audit["all_ready"]:
            return audit
        save(
            status_path,
            {
                "stage": "waiting_inputs",
                "audit": audit,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        time.sleep(poll_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    args = parser.parse_args()
    audit = wait_for_inputs(args.config, args.status, args.poll_seconds)
    save(
        args.status,
        {"stage": "running", "audit": audit, "updated_at": datetime.now(timezone.utc).isoformat()},
    )
    result = run(args.config, args.output)
    save(
        args.status,
        {
            "stage": "complete", "result_status": result["status"],
            "updated_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
