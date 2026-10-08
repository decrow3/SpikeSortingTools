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
import json
import sys
from types import SimpleNamespace
sys.dont_write_bytecode = True
root = sys.argv.pop(1)
sys.path.insert(0, root)
from testing.candidate2_cache_managed_preflight import (
    TYPED_EXEC_SIGNATURE, capture_typed_exec_identity, typed_exec_property,
)

def completed(payload, returncode=0, stderr=""):
    return SimpleNamespace(returncode=returncode, stdout=json.dumps(payload), stderr=stderr)

path = "/org/freedesktop/systemd1/unit/demo_2eservice"
rows = [["/bin/tool", ["/bin/tool", "--flag", "two words"], False, 1, 2, 3, 4, 5, 0, 0]]
responses = [completed({"type": "o", "data": [path]}),
             completed({"type": TYPED_EXEC_SIGNATURE, "data": rows}),
             completed({"type": TYPED_EXEC_SIGNATURE, "data": []})]
identity = capture_typed_exec_identity({"target": "demo.service"}, run=lambda *a, **k: responses.pop(0))
assert identity == {"target": {
    "ExecStart": [{"executable_path": "/bin/tool", "argv": ["/bin/tool", "--flag", "two words"], "ignore_errors": False}],
    "ExecStartPre": [],
}}

bad_values = [
    {"type": "s", "data": rows},
    {"type": TYPED_EXEC_SIGNATURE, "data": [["/bin/tool", ["/bin/tool"], False]]},
    {"type": TYPED_EXEC_SIGNATURE, "data": [["/bin/tool", ["/bin/tool", 7], False, 1, 2, 3, 4, 5, 0, 0]]},
    {"type": TYPED_EXEC_SIGNATURE, "data": [["/bin/tool", [], 0, 1, 2, 3, 4, 5, 0, 0]]},
]
for bad in bad_values:
    try:
        typed_exec_property(path, "ExecStart", run=lambda *a, value=bad, **k: completed(value))
    except RuntimeError:
        pass
    else:
        raise AssertionError(f"unsupported typed property schema passed: {bad}")
try:
    typed_exec_property(path, "ExecStart", run=lambda *a, **k: completed({}, 1, "denied"))
except RuntimeError as exc:
    assert "query failed" in str(exc)
else:
    raise AssertionError("structured query failure passed")
print("typed systemd identity preserves exact path/argv/flags, accepts empty arrays, and rejects bad schemas/query failure")
'''
    completed_process = subprocess.run(
        [str(EXPECTED_INTERPRETER), "-I", "-u", "-c", code, str(args.root.resolve() / "source")],
        text=True, capture_output=True, check=False,
    )
    assert completed_process.returncode == 0, completed_process.stderr
    print(completed_process.stdout.strip())


if __name__ == "__main__":
    main()
