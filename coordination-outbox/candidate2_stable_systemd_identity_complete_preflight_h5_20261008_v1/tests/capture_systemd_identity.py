from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


DEFAULT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--target-unit", required=True)
    parser.add_argument("--monitor-unit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(args.root.resolve() / "source"))
    from testing.candidate2_cache_managed_preflight import (
        STATIC_SYSTEMD_PROPERTIES, capture_typed_exec_identity,
    )

    units = {"target": args.target_unit, "monitor": args.monitor_unit}
    loaded = {}
    for role, unit in units.items():
        completed = subprocess.run(
            ["systemctl", "--user", "show", unit,
             *[f"--property={key}" for key in STATIC_SYSTEMD_PROPERTIES]],
            text=True, capture_output=True, check=True,
        )
        loaded[role] = dict(
            line.split("=", 1) for line in completed.stdout.splitlines() if "=" in line
        )
    value = {
        "schema": "candidate2-systemd-authorization-identity-v1",
        "loaded_units": loaded,
        "typed_exec_identity": capture_typed_exec_identity(units),
    }
    args.output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
