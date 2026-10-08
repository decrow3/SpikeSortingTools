#!/usr/bin/env python3
import json
import os
import subprocess
import tempfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
DARTSORT = Path("/home/huklab/Documents/DARTsort")
KS_PY = PROJECT / "environments/rescue-production/.venv/bin/python"
DS_PY = DARTSORT / ".venv/bin/python"


def run(cmd, cwd):
    env = os.environ.copy()
    env["NUMBA_CACHE_DIR"] = "/tmp/candidate3_numba_cache"
    subprocess.run([str(x) for x in cmd], cwd=cwd, env=env, check=True)


def main():
    with tempfile.TemporaryDirectory(prefix="candidate3-fixtures-", dir="/tmp") as td:
        out = Path(td)
        run([DS_PY, HERE / "checkpoint_fault_fixture.py", out], DARTSORT)
        for kind in ("ordinary", "transition"):
            run([KS_PY, HERE / "kilosort_boundary_producer.py", kind, out], PROJECT)
            run([DS_PY, HERE / "dartsort_boundary_consumer.py", kind, out], DARTSORT)
        receipt = {
            "schema": "candidate3-checkpoint-boundary-fixture-receipt-v1",
            "checkpoint": json.loads((out / "checkpoint_result.json").read_text()),
            "ordinary": json.loads((out / "ordinary_consumer.json").read_text()),
            "transition": json.loads((out / "transition_consumer.json").read_text()),
            "temporary_hdf5_preserved": False,
            "raw_or_real_voltage_read": False,
            "sort_launched": False,
        }
    (HERE / "FIXTURE_RECEIPT.json").write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main()
