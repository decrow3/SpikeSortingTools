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
    code = r'''
import sys
sys.dont_write_bytecode = True
root = sys.argv.pop(1)
sys.path.insert(0, root)
from testing.candidate2_cache_managed_preflight import canonicalize_loaded_units

left = {"target": {"After": "b a", "Requires": "z y", "BindsTo": "m", "ExecStart": "exact"}}
reordered = {"target": {"After": "a b", "Requires": "y z", "BindsTo": "m", "ExecStart": "exact"}}
changed_member = {"target": {"After": "a c", "Requires": "y z", "BindsTo": "m", "ExecStart": "exact"}}
changed_scalar = {"target": {"After": "a b", "Requires": "y z", "BindsTo": "m", "ExecStart": "different"}}
assert canonicalize_loaded_units(left) == canonicalize_loaded_units(reordered)
assert canonicalize_loaded_units(left) != canonicalize_loaded_units(changed_member)
assert canonicalize_loaded_units(left) != canonicalize_loaded_units(changed_scalar)
print("systemd set order canonicalized while membership and scalar bytes remain strict")
'''
    completed = subprocess.run(
        [str(EXPECTED_INTERPRETER), "-I", "-u", "-c", code, str(args.root.resolve() / "source")],
        text=True, capture_output=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    print(completed.stdout.strip())


if __name__ == "__main__":
    main()
