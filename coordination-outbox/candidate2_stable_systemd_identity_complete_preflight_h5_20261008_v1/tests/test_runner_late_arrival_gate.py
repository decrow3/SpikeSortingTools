from __future__ import annotations

import argparse
from pathlib import Path
import subprocess


EXPECTED_INTERPRETER = Path(
    "/home/huklaban5/Documents/SpikeSortingTools/SpikeSortingTools/"
    "environments/rescue-production/.venv/bin/python"
)
DEFAULT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    source = args.root.resolve() / "source"
    code = r'''
import sys
sys.dont_write_bytecode = True
root = sys.argv.pop(1)
sys.path.insert(0, root)
import testing.candidate2_cache_managed_full as full

Disk = type("Disk", (), {"free": 10**15})
full.shutil.disk_usage = lambda path: Disk()
base = {
    "mem_available_bytes": 10**12,
    "pressure": {"some": {"avg10": 0.0}},
    "gpu": {"free_bytes": 30 * 2**30, "external_compute_apps": []},
}
contract = {
    "attempt": {"output_root": "/tmp/candidate2-late-arrival-fixture"},
    "resource_conditions": {
        "minimum_output_free_bytes": 1,
        "minimum_mem_available_bytes": 1,
        "maximum_memory_psi_some_avg10": 1.0,
        "minimum_gpu_free_bytes": 20 * 2**30,
    },
}

# Clean first gate, then a non-self GPU PID arrives before the final gate.
states = [base, {**base, "gpu": {**base["gpu"], "external_compute_apps": [{"pid": 424242}]}}]
full.snapshot = lambda: states.pop(0)
full.competing_processes = lambda: []
full.require_resource_gates(contract)
try:
    full.require_resource_gates(contract)
except RuntimeError as exc:
    assert "external_gpu_compute_apps" in str(exc)
else:
    raise AssertionError("late external GPU application passed final runner gate")

# A late competing sorter is rejected independently of free GPU bytes.
full.snapshot = lambda: base
full.competing_processes = lambda: ["424243 python -m kilosort.run_kilosort"]
try:
    full.require_resource_gates(contract)
except RuntimeError as exc:
    assert "competing sorter process" in str(exc)
else:
    raise AssertionError("late competing sorter passed final runner gate")
print("actual isolated runner rejected late GPU-app and sorter arrivals")
'''
    completed = subprocess.run(
        [str(EXPECTED_INTERPRETER), "-I", "-u", "-c", code, str(source)],
        text=True, capture_output=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    print(completed.stdout.strip())


if __name__ == "__main__":
    main()
