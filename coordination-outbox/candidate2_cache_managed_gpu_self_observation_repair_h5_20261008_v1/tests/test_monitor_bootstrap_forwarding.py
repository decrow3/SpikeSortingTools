from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BOOTSTRAP = ROOT / "monitor_bootstrap.py"
DEFAULT_UNIT = ROOT / "luke-candidate2-native-rigid-cache-managed-monitor-imec0-20261008-v7.service"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def service_argv(unit: Path) -> list[str]:
    line = next(line for line in unit.read_text().splitlines() if line.startswith("ExecStart="))
    return shlex.split(line.removeprefix("ExecStart="))


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--bootstrap", type=Path, default=DEFAULT_BOOTSTRAP)
    parser.add_argument("--unit", type=Path, default=DEFAULT_UNIT)
    args = parser.parse_args()
    bootstrap = args.bootstrap.resolve()
    unit = args.unit.resolve()
    with tempfile.TemporaryDirectory(prefix="candidate2-monitor-forwarding-") as tmp:
        tmp_path = Path(tmp)
        child = tmp_path / "inert_monitor.py"
        child.write_text(
            """from __future__ import annotations
import argparse
import json

p = argparse.ArgumentParser(allow_abbrev=False)
p.add_argument('--self-sha256', required=True)
p.add_argument('--contract', required=True)
p.add_argument('--contract-sha256', required=True)
p.add_argument('--authorization', required=True)
p.add_argument('--target-unit-file', required=True)
p.add_argument('--monitor-unit-file', required=True)
p.add_argument('--target-unit', required=True)
p.add_argument('--output', required=True)
p.add_argument('--service-log', required=True)
a = p.parse_args()
print(json.dumps(vars(a), sort_keys=True))
"""
        )
        contract = tmp_path / "CONTRACT.json"
        contract.write_text("{}\n")
        target_unit = tmp_path / "target.service"
        monitor_unit = tmp_path / "monitor.service"
        target_unit.write_text("target\n")
        monitor_unit.write_text("monitor\n")
        authorization = tmp_path / "AUTHORIZATION.json"
        authorization.write_text(
            json.dumps(
                {
                    "status": "AUTHORIZED_ONE_FRESH_CANDIDATE2_CACHE_MANAGED_RUN",
                    "contract_sha256": sha256(contract),
                    "target_service_sha256": sha256(target_unit),
                    "monitor_service_sha256": sha256(monitor_unit),
                }
            )
            + "\n"
        )

        raw = service_argv(unit)
        expected_option_order = [
            "--bootstrap-sha256",
            "--monitor-source",
            "--self-sha256",
            "--contract",
            "--contract-sha256",
            "--authorization",
            "--target-unit",
            "--target-unit-file",
            "--monitor-unit-file",
            "--output",
            "--service-log",
        ]
        observed_option_order = [token for token in raw if token.startswith("--")]
        assert observed_option_order == expected_option_order
        values = {
            "--bootstrap-sha256": sha256(bootstrap),
            "--monitor-source": str(child),
            "--self-sha256": sha256(child),
            "--contract": str(contract),
            "--contract-sha256": sha256(contract),
            "--authorization": str(authorization),
            "--target-unit": "synthetic-target.service",
            "--target-unit-file": str(target_unit),
            "--monitor-unit-file": str(monitor_unit),
            "--output": str(tmp_path / "output"),
            "--service-log": str(tmp_path / "service.log"),
        }
        command = [sys.executable, "-I", "-S", "-u", str(bootstrap)]
        for option in observed_option_order:
            command.extend((option, values[option]))
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
        assert completed.returncode == 0, completed.stderr
        child_args = json.loads(completed.stdout)
        assert child_args["target_unit"] == "synthetic-target.service"
        assert child_args["target_unit_file"] == str(target_unit)
        assert child_args["monitor_unit_file"] == str(monitor_unit)
        assert child_args["output"] == str(tmp_path / "output")
        assert child_args["service_log"] == str(tmp_path / "service.log")
        print("actual service argv crossed monitor bootstrap boundary with --target-unit intact")


if __name__ == "__main__":
    main()
