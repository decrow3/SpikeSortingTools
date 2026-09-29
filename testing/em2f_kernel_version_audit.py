#!/usr/bin/env python3
"""Compare the full-session-only +80 um kriging kernel across SI versions."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from testing.em2a_operator_screen import ARM_SPECS, effective_kernel, interpolation_insertion
from testing.em_huklaban5_voltage_equivalence import TARGET_IDS
from testing.em_huklaban5_voltage_equivalence_arms import (
    BASE,
    EXPECTED,
    array_sha256,
    find_channel_slice,
)


def worker() -> dict[str, object]:
    import spikeinterface as si
    from spikeinterface.preprocessing import get_spatial_interpolation_kernel
    from spikeinterface.preprocessing.preprocessing_tools import get_kriging_channel_weights

    provenance = json.loads((BASE / "provenance.json").read_text())
    parent = si.load(provenance["kwargs"]["parent_recording"], base_folder=BASE)
    parent_ids = list(map(str, parent.get_channel_ids()))
    source_geometry = np.asarray(parent.get_channel_locations(), dtype=np.float64)[:, :2]
    target_indices = np.asarray([parent_ids.index(channel) for channel in TARGET_IDS])
    target_geometry = source_geometry[target_indices]
    if array_sha256(source_geometry) != EXPECTED["source_geometry"]:
        raise RuntimeError("source geometry hash differs")
    if array_sha256(target_geometry) != EXPECTED["target_geometry"]:
        raise RuntimeError("target geometry hash differs")
    full_slice = find_channel_slice(provenance, 384)
    full_ids = list(map(str, full_slice["kwargs"]["channel_ids"]))
    full_geometry = np.asarray(full_slice["properties"]["location"], dtype=np.float64)[:, :2]
    insertion, _ = interpolation_insertion(
        parent_ids,
        full_ids,
        source_geometry,
        full_geometry,
        get_kriging_channel_weights,
    )
    kernel, structural_zero = effective_kernel(
        get_spatial_interpolation_kernel,
        insertion,
        full_geometry,
        target_geometry,
        80.0,
        ARM_SPECS["rounded_kriging"],
    )
    return {
        "spikeinterface": si.__version__,
        "state_um": 80.0,
        "shape": list(kernel.shape),
        "dtype": str(kernel.dtype),
        "kernel_sha256": array_sha256(kernel),
        "structural_zero_sha256": array_sha256(structural_zero),
        "structural_zero_targets": int(np.count_nonzero(structural_zero)),
        "finite": bool(np.isfinite(kernel).all()),
        "column_sum_min": float(kernel.sum(axis=0).min()),
        "column_sum_max": float(kernel.sum(axis=0).max()),
    }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--python-1047", type=Path)
    parser.add_argument("--python-1048", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(), sort_keys=True))
        return
    if args.python_1047 is None or args.python_1048 is None or args.output is None:
        parser.error("controller mode requires both interpreters and --output")
    if args.output.exists():
        raise FileExistsError(args.output)

    rows = []
    for interpreter in (args.python_1047, args.python_1048):
        completed = subprocess.run(
            [str(interpreter), str(Path(__file__).resolve()), "--worker"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        rows.append(json.loads(completed.stdout))
    versions = [row["spikeinterface"] for row in rows]
    if versions != ["0.104.7", "0.104.8"]:
        raise RuntimeError(f"unexpected SpikeInterface versions: {versions}")
    identical = all(
        rows[0][key] == rows[1][key]
        for key in (
            "kernel_sha256",
            "structural_zero_sha256",
            "shape",
            "dtype",
            "structural_zero_targets",
            "column_sum_min",
            "column_sum_max",
        )
    )
    result = {
        "schema": "em2f-full-new-state-kernel-version-audit-v1",
        "status": "pass" if identical and all(row["finite"] for row in rows) else "fail",
        "voltage_read": False,
        "sort_launched": False,
        "state_um": 80.0,
        "versions": versions,
        "byte_identical": identical,
        "results": rows,
    }
    args.output.mkdir(parents=True)
    result_path = args.output / "RESULT.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (args.output / "COMPLETE.json").write_text(
        json.dumps(
            {"status": result["status"], "result_sha256": sha256(result_path)},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    if result["status"] != "pass":
        raise RuntimeError("+80 um kernel differs across SpikeInterface versions")


if __name__ == "__main__":
    main()
