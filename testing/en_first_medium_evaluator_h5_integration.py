#!/usr/bin/env python3
"""Read-only H5 integration inventory for the H1 four-arm evaluator."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
H1 = Path("/mnt/NPX/Luke/DARTsort_motion_experiments/en_first_medium_four_arm_evaluator_review_candidate_20261002_v2_h1")
SCREEN = Path("/media/huklaban5/Data/en_common_support_crop_screen_20260930_v1")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.config import fingerprint
from testing.luke_amplitude_dropout_audit import read_curated_arrays, curated_arrays_from_raw


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def arm_inventory(request_id: str, disk_id: str) -> dict:
    root = SCREEN / disk_id
    curated = root / "sorter_output"
    raw, hashes = read_curated_arrays(curated)
    validation_error = None
    try:
        arrays = curated_arrays_from_raw(request_id, raw)
    except ValueError as exc:
        arrays = None
        validation_error = f"{type(exc).__name__}: {exc}"
    saved_times = np.asarray(raw["spike_times.npy"]).reshape(-1)
    full_st = np.asarray(raw["full_st.npy"])
    kept = np.asarray(raw["kept_spikes.npy"])
    kept_st = full_st[kept]
    delta = saved_times.astype(np.int64) - kept_st[:, 0].astype(np.int64)
    label_path = curated / "cluster_KSLabel.tsv"
    label_bytes = label_path.read_bytes()
    labels = pd.read_csv(io.BytesIO(label_bytes), sep="\t")
    positions_path = curated / "spike_positions.npy"
    positions = np.load(positions_path, mmap_mode="r", allow_pickle=False)
    channel_positions_path = curated / "channel_positions.npy"
    channel_positions = np.load(channel_positions_path, allow_pickle=False)
    ops_path = curated / "ops.npy"
    ops = np.load(ops_path, allow_pickle=True).item()
    qc_path = root / "qc/amp_truncation/truncation_qc.npz"
    curated_identity = fingerprint({
        "curated": str(curated.resolve()), "files": hashes,
        "labels_sha256": hashlib.sha256(label_bytes).hexdigest(),
    })
    geometry_float64_sha = hashlib.sha256(
        np.ascontiguousarray(channel_positions, dtype=np.float64).tobytes()
    ).hexdigest()
    clock = {
        "saved_time_frame": "global acquisition samples",
        "saved_min": int(saved_times.min()), "saved_max": int(saved_times.max()),
        "expected_evaluator_frame": "crop-local samples",
        "expected_local_half_open": [0, 17999901],
        "global_half_open": [208498882, 226498783],
        "required_transform": "local = saved_global - 208498882",
        "current_adapter_applies_transform": False,
        "compatible_with_current_adapter": False,
    }
    runtime_payload = {
        "kilosort_version": "4.0.27",
        "compatibility_patch_sha256": "19ba98a8cc752889a4eb2833410f8d69da6a275688fb9580b840075092fde259",
        "ops_sha256": sha(ops_path),
        "consumed_settings": json.loads((root / "RECEIPT.json").read_text())["consumed_settings"],
    }
    return {
        "arm_contract_id": request_id, "disk_arm_id": disk_id,
        "root": str(root), "curated_output": str(curated),
        "loader_consumed": {
            name: {"path": str(curated / name), "sha256": hashes[name],
                   "shape": list(np.asarray(raw[name]).shape), "dtype": str(np.asarray(raw[name]).dtype)}
            for name in ("spike_times.npy", "spike_clusters.npy", "full_st.npy", "kept_spikes.npy")
        } | {
            "cluster_KSLabel.tsv": {"path": str(label_path), "sha256": hashlib.sha256(label_bytes).hexdigest(),
                                      "rows": int(len(labels))},
            "spike_positions.npy": {"path": str(positions_path), "sha256": sha(positions_path),
                                      "shape": list(positions.shape), "dtype": str(positions.dtype)},
        },
        "qc": {"path": str(qc_path), "exists": qc_path.is_file(),
               "sha256": sha(qc_path) if qc_path.is_file() else None,
               "status": "BOUND" if qc_path.is_file() else "UNMEASURED_ABSENT"},
        "identity_digest": curated_identity,
        "input_identity": fingerprint({"curated_identity_digest": curated_identity,
                                         "receipt_sha256": sha(root / "RECEIPT.json")}),
        "clock_identity": fingerprint(clock), "clock": clock,
        "geometry_identity": fingerprint({"channel_positions_float64_sha256": geometry_float64_sha,
                                             "chanMap": np.asarray(ops["chanMap"]).astype(int).tolist()}),
        "geometry": {"channel_positions_path": str(channel_positions_path),
                     "channel_positions_file_sha256": sha(channel_positions_path),
                     "geometry_float64_sha256": geometry_float64_sha,
                     "shape": list(channel_positions.shape),
                     "identity_chanMap": bool(np.array_equal(np.asarray(ops["chanMap"]), np.arange(384)))},
        "runtime_identity": fingerprint(runtime_payload), "runtime": runtime_payload,
        "config_identity": "bb6dd8f21539e5ee57f4216702b7c9d4c36d1a0feeefe3859ba25449af52ad17",
        "receipt_sha256": sha(root / "RECEIPT.json"), "ops_sha256": sha(ops_path),
        "validation": {
            "curated_alignment_passed": validation_error is None,
            "curated_alignment_error": validation_error,
            "was_time_ordered": bool(saved_times.size < 2 or np.all(np.diff(saved_times) >= 0)),
            "spike_positions_aligned": positions.ndim == 2 and positions.shape[0] == len(saved_times),
            "full_st_kept_rows": int(len(kept_st)), "spike_times_rows": int(len(saved_times)),
            "spike_times_minus_full_st_kept_unique": np.unique(delta).astype(int).tolist(),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    ref = arm_inventory("REF384", "REF384")
    b384 = arm_inventory("existing_corrected_B384", "B384")
    inventory = {
        "schema": "en-first-medium-evaluator-h5-inventory-v1",
        "candidate_manifest_sha256": sha(H1 / "MANIFEST.sha256"),
        "candidate_complete_sha256": sha(H1 / "COMPLETE.json"),
        "arms": {"REF384": ref, "existing_corrected_B384": b384},
        "raw_recording_reads": 0,
    }
    (output / "INVENTORY.json").write_text(json.dumps(inventory, indent=2) + "\n")
    request = json.loads((H1 / "source/configs/en_first_medium_four_arm_evaluator.execution_disabled.v1.json").read_text())
    for spec, info in zip(request["arms"][:2], (ref, b384)):
        spec["expected_identity_digest"] = info["identity_digest"]
        spec["expected_qc_request_digest"] = "UNMEASURED_ABSENT_TRUNCATION_QC"
        spec["provenance"] = {
            "arm_contract_id": info["arm_contract_id"], "input_identity": info["input_identity"],
            "clock_identity": info["clock_identity"], "geometry_identity": info["geometry_identity"],
            "runtime_identity": info["runtime_identity"], "config_identity": info["config_identity"],
        }
    request_path = output / "BOUND_REQUEST.json"
    request_path.write_text(json.dumps(request, indent=2) + "\n")
    traversal = output / "traversal_output"
    runtime = output / "isolated_runtime/testing"
    runtime.mkdir(parents=True)
    shutil.copyfile(H1 / "source/testing/first_medium_evaluator.py", runtime / "first_medium_evaluator.py")
    shutil.copyfile(H1 / "source/testing/sort_comparison.py", runtime / "sort_comparison.py")
    (runtime / "__init__.py").write_text(
        "# orchestration-only package binding for byte-identical H1 snapshots\n"
        f"__path__.append({str(ROOT / 'testing')!r})\n"
    )
    command = [
        sys.executable, "-m", "testing.first_medium_evaluator",
        "--request", str(request_path), "--output", str(traversal),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(output / "isolated_runtime") + os.pathsep + str(ROOT)
    completed = subprocess.run(command, cwd="/tmp", env=env, text=True, capture_output=True, check=False)
    (output / "stdout.log").write_text(completed.stdout)
    (output / "stderr.log").write_text(completed.stderr)
    (output / "COMMAND.json").write_text(json.dumps({"command": command, "cwd": "/tmp",
        "python": sys.executable, "python_sha256": sha(Path(sys.executable)),
        "candidate_source_sha256": sha(H1 / "source/testing/first_medium_evaluator.py"),
        "candidate_adapter_sha256": sha(H1 / "source/testing/sort_comparison.py"),
        "runtime_source_sha256": sha(runtime / "first_medium_evaluator.py"),
        "runtime_adapter_sha256": sha(runtime / "sort_comparison.py"),
        "package_shim_sha256": sha(runtime / "__init__.py"),
        "dependency_path_extension": str(ROOT / "testing"),
        "bound_request_sha256": sha(request_path)}, indent=2) + "\n")
    result = {
        "schema": "en-first-medium-evaluator-h5-traversal-result-v1",
        "returncode": completed.returncode,
        "status": "UNMEASURED_BLOCKED",
        "blocker": "saved spike_times/global clock is incompatible with full_st[kept]/crop-local clock in the byte-identical loader",
        "secondary_blocker": "required cached truncation QC absent for both REF384 and existing_corrected_B384",
        "first_failure": completed.stderr.strip().splitlines()[-1] if completed.stderr.strip() else None,
        "four_arm_report_exists": (traversal / "FOUR_ARM_REPORT.json").is_file(),
        "traversal_output_exists": traversal.exists(),
        "required_adapter_repair": "normalize exactly one clock at load: full_st[kept] + 208498882 must equal saved global spike_times, then emit crop-local times for evaluation; independently bind real QC",
        "scientific_rows_emitted": 0,
        "raw_recording_reads": 0,
    }
    (output / "TRAVERSAL_RESULT.json").write_text(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
