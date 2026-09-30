#!/usr/bin/env python3
"""Durable EN arm-B smoke then full-session materialization and KS4 sort."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from npx_preprocessing.motion import ExactLatticeRemapRecording
from npx_preprocessing.motion.lattice_remap_si import audit_spikeinterface_nearest_kernel
from pipeline.config import fingerprint
from pipeline.sorting import run_kilosort4
from testing.en_rounded_field import rounded_rigid_field
from testing.luke_external_warp_pipeline import _materialize_arm


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def save(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(temporary, path)


def gpu_busy() -> bool:
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    )
    ignored = {"snapd-desktop-integration", "Xorg", "gnome-shell"}
    for line in result.stdout.splitlines():
        fields = [value.strip() for value in line.split(",")]
        if len(fields) >= 2 and Path(fields[1]).name not in ignored and fields[0].isdigit() and int(fields[0]) != os.getpid():
            return True
    return False


def wait_for_q0(path: Path, status_path: Path) -> dict:
    while not path.is_file():
        save(status_path, {"stage": "waiting_q0_full_hash", "updated_at": datetime.now(timezone.utc).isoformat()})
        time.sleep(30)
    receipt = json.loads(path.read_text())
    if receipt.get("status") != "pass":
        raise RuntimeError("q0 full binary hash gate failed")
    return receipt


def materialization_request(
    contract: dict, field_sha256: str, scope: str, source_frames: list[int]
) -> dict:
    """Keep experiment metadata from overwriting the accepted-manifest schema."""
    if scope not in {"smoke", "full_session"}:
        raise ValueError("unsupported EN materialization scope")
    return {
        "schema": "en-rounded-field-ks129-materialization-v1",
        "contract_digest": contract["digest"],
        "arm": "B",
        "field_sha256": field_sha256,
        "q0_receipt_sha256": contract["q0_receipt_sha256"],
        "adapter": contract["adapter"],
        "scope": scope,
        "source_frames": source_frames,
    }


def run_sort(recording_dir: Path, sort_dir: Path, status_path: Path, label: str) -> tuple[dict, float]:
    while gpu_busy():
        save(status_path, {"stage": f"waiting_gpu_{label}", "updated_at": datetime.now(timezone.utc).isoformat()})
        time.sleep(60)
    save(status_path, {"stage": f"sorting_{label}", "updated_at": datetime.now(timezone.utc).isoformat()})
    started = time.perf_counter()
    manifest = run_kilosort4(recording_dir, sort_dir)
    return manifest, time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/en_rounded_field_ks129.v1.json"))
    parser.add_argument("--q0-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke-seconds", type=float, default=120.0)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    status_path = args.output / "STATUS.json"
    started_all = time.perf_counter()
    q0 = wait_for_q0(args.q0_receipt, status_path)
    config = json.loads(args.config.read_text())
    ref, spec = config["reference"], config["arm_b"]
    if q0.get("actual_sha256") != ref["binary_sha256"]:
        raise RuntimeError("q0 receipt names a different accepted binary")
    field_path = Path(spec["field_path"])
    if sha256(field_path) != spec["field_sha256"]:
        raise RuntimeError("arm B field hash differs")

    import spikeinterface as si
    source_dir = Path(ref["recording_dir"])
    source_manifest = json.loads((source_dir / "rescue_recording_manifest.json").read_text())
    source = si.load(source_dir)
    with np.load(field_path, allow_pickle=False) as data:
        time_s = np.asarray(data[spec["time_key"]], dtype=np.float64)
        displacement = np.asarray(data[spec["displacement_key"]], dtype=np.float64)
    rounded, reference_um = rounded_rigid_field(displacement, config["adapter"]["period_um"])
    dt = float(np.median(np.diff(time_s)))
    if not np.allclose(np.diff(time_s), dt, rtol=0, atol=1e-6):
        raise RuntimeError("arm B field time grid is not uniform")
    mapping = audit_spikeinterface_nearest_kernel(source.get_channel_locations()[:, :2], np.unique(rounded))
    if not all(row["stock_matches_exact"] for row in mapping):
        raise RuntimeError("fast exact lattice adapter is not equivalent to installed SI nearest")
    corrected = ExactLatticeRemapRecording(
        source, time_s, rounded, cell_width_s=dt, direction="y", time_origins_s=[0.0]
    )
    contract = {
        "schema_version": "en-rounded-field-ks129-arm-b-contract-v1",
        "arm": "B",
        "config_sha256": sha256(args.config),
        "q0_receipt_sha256": sha256(args.q0_receipt),
        "field_sha256": spec["field_sha256"],
        "reference_median_um": reference_um,
        "rounded_states_um": np.unique(rounded).tolist(),
        "adapter": "ExactLatticeRemapRecording, proven pairwise equal to SI 0.102.1 nearest/force_zeros on every arm-B state",
        "checkpoint_policy": "Materialization and each sort are atomic stages. Kilosort has no within-sort checkpoint; interruption requires preserving evidence and restarting that sort.",
    }
    contract["digest"] = fingerprint(contract)
    save(args.output / "CONTRACT.json", contract)

    fs = float(source.get_sampling_frequency())
    extreme_index = int(np.argmax(np.abs(rounded)))
    center_s = float(time_s[extreme_index])
    smoke_start_s = max(0.0, min(center_s - args.smoke_seconds / 2, source.get_num_samples() / fs - args.smoke_seconds))
    smoke_start = int(round(smoke_start_s * fs))
    smoke_stop = min(source.get_num_samples(), smoke_start + int(round(args.smoke_seconds * fs)))
    smoke = corrected.frame_slice(start_frame=smoke_start, end_frame=smoke_stop)
    smoke_request = materialization_request(
        contract, spec["field_sha256"], "smoke", [smoke_start, smoke_stop]
    )
    save(status_path, {"stage": "materializing_smoke", "updated_at": datetime.now(timezone.utc).isoformat()})
    smoke_t0 = time.perf_counter()
    smoke_manifest = _materialize_arm(
        smoke,
        args.output / "smoke/recording",
        source_manifest=source_manifest,
        request=smoke_request,
        n_jobs=8,
        before_accept=lambda: save(
            status_path,
            {"stage": "verifying_smoke_recording", "updated_at": datetime.now(timezone.utc).isoformat()},
        ),
    )
    smoke_materialize_s = time.perf_counter() - smoke_t0
    smoke_sort, smoke_sort_s = run_sort(args.output / "smoke/recording", args.output / "smoke/sort", status_path, "smoke")
    save(args.output / "SMOKE.json", {
        "status": "pass", "interval_s": [smoke_start / fs, smoke_stop / fs],
        "extreme_state_um": float(rounded[extreme_index]), "materialize_seconds": smoke_materialize_s,
        "sort_seconds": smoke_sort_s, "recording": smoke_manifest, "sort": smoke_sort,
    })

    required = int(ref["num_samples"] * ref["num_channels"] * 2 + 300 * 1024**3)
    if shutil.disk_usage(args.output).free < required:
        raise RuntimeError("insufficient local space for full arm-B materialization and sort reserve")
    save(status_path, {"stage": "materializing_full", "updated_at": datetime.now(timezone.utc).isoformat()})
    full_t0 = time.perf_counter()
    full_request = materialization_request(
        contract,
        spec["field_sha256"],
        "full_session",
        [0, int(source.get_num_samples())],
    )
    full_manifest = _materialize_arm(
        corrected,
        args.output / "full/recording",
        source_manifest=source_manifest,
        request=full_request,
        n_jobs=8,
        before_accept=lambda: save(
            status_path,
            {"stage": "verifying_full_recording", "updated_at": datetime.now(timezone.utc).isoformat()},
        ),
    )
    full_materialize_s = time.perf_counter() - full_t0
    full_sort, full_sort_s = run_sort(args.output / "full/recording", args.output / "full/sort", status_path, "full")
    save(args.output / "COMPLETE.json", {
        "schema_version": "en-rounded-field-ks129-arm-b-complete-v1", "status": "complete",
        "full_materialize_seconds": full_materialize_s, "full_sort_seconds": full_sort_s,
        "total_wall_seconds": time.perf_counter() - started_all, "recording": full_manifest, "sort": full_sort,
    })
    save(status_path, {"stage": "complete", "updated_at": datetime.now(timezone.utc).isoformat()})


if __name__ == "__main__":
    main()
