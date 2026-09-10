"""Run a bounded 930--1230 s MEDiCINe voltage-correction sort matrix.

This is a development comparison.  It consumes one frozen five-minute field,
applies every correction on the full source geometry, restricts all arms to the
same surviving channels, materializes checked int16 voltage, and runs KS4 with
its internal motion correction disabled.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np

from pipeline.config import fingerprint
from pipeline.preprocess import validate_accepted_recording
from pipeline.sorting import run_kilosort4
from testing.luke_external_warp_pipeline import _materialize_arm
from testing.luke_full_session_medicine import save, sha
from testing.luke_medicine_rigid_queue import checked_int16


SCHEMA = "luke-medicine-5min-correction-matrix-v1"
INTERVAL_S = (930.0, 1230.0)
ARMS = {
    "unwarped": None,
    "rigid_g025_p2_s10": dict(spatial="rigid", gain=0.25, p=2, sigma_um=10.0),
    "rigid_g100_p2_s10": dict(spatial="rigid", gain=1.0, p=2, sigma_um=10.0),
    "nonrigid_g025_p2_s10": dict(spatial="nonrigid", gain=0.25, p=2, sigma_um=10.0),
    "nonrigid_g100_p2_s10": dict(spatial="nonrigid", gain=1.0, p=2, sigma_um=10.0),
    "nonrigid_g100_p2_s20": dict(spatial="nonrigid", gain=1.0, p=2, sigma_um=20.0),
    "nonrigid_g100_p1_s20_historical": dict(spatial="nonrigid", gain=1.0, p=1, sigma_um=20.0),
}


def load_field(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    result = {name: np.asarray(data[name], dtype=np.float64) for name in
              ("time_s", "depth_um", "displacement_um")}
    if result["displacement_um"].shape != (result["time_s"].size, result["depth_um"].size):
        raise ValueError("field is not time by depth")
    if not all(np.all(np.isfinite(value)) for value in result.values()):
        raise ValueError("field contains nonfinite values")
    if not np.all(np.diff(result["time_s"]) > 0) or not np.all(np.diff(result["depth_um"]) > 0):
        raise ValueError("field axes are not strictly increasing")
    if result["time_s"][0] - INTERVAL_S[0] > 1.0 or INTERVAL_S[1] - result["time_s"][-1] > 1.0:
        raise ValueError("field terminal support is more than one second from requested interval")
    return result


def motion_for(recording, field: dict[str, np.ndarray], spec: dict):
    from spikeinterface.core.motion import Motion

    displacement = field["displacement_um"]
    if spec["spatial"] == "rigid":
        displacement = np.median(displacement, axis=1)[:, None]
        depths = np.array([np.mean(recording.get_channel_locations()[:, 1])])
    else:
        depths = field["depth_um"]
    return Motion(
        displacement=[displacement * float(spec["gain"])],
        temporal_bins_s=[field["time_s"] + float(recording.sample_index_to_time(0))],
        spatial_bins_um=depths,
        direction="y",
        interpolation_method="linear",
    )


def corrected(recording, field: dict[str, np.ndarray], spec: dict):
    from spikeinterface.preprocessing import astype
    from spikeinterface.sortingcomponents.motion import InterpolateMotionRecording

    return InterpolateMotionRecording(
        astype(recording, "float32"),
        motion_for(recording, field, spec),
        border_mode="remove_channels",
        spatial_interpolation_method="kriging",
        sigma_um=float(spec["sigma_um"]),
        p=float(spec["p"]),
        dtype="float32",
    )


def common_channel_ids(recording, field: dict[str, np.ndarray]) -> list:
    common = set(recording.get_channel_ids().tolist())
    for spec in ARMS.values():
        if spec is not None:
            common &= set(corrected(recording, field, spec).get_channel_ids().tolist())
    ordered = [value for value in recording.get_channel_ids().tolist() if value in common]
    if len(ordered) < 64:
        raise RuntimeError("correction matrix leaves fewer than 64 common channels")
    return ordered


def verify_config(cfg: dict) -> None:
    if cfg.get("schema") != SCHEMA:
        raise ValueError("wrong correction-matrix schema")
    if sha(Path(cfg["field"])) != cfg["field_sha256"]:
        raise RuntimeError("frozen MEDiCINe field changed")
    if sha(Path(cfg["recording_manifest"])) != cfg["recording_manifest_sha256"]:
        raise RuntimeError("accepted recording manifest changed")


def execute(cfg: dict, output: Path) -> None:
    from spikeinterface.core import load

    verify_config(cfg)
    source_dir = Path(cfg["recording"])
    source_manifest = validate_accepted_recording(source_dir)
    if source_manifest["recording_content_sha256"] != cfg["recording_content_sha256"]:
        raise RuntimeError("accepted recording content changed")
    source = load(source_dir)
    field = load_field(Path(cfg["field"]))
    ids = common_channel_ids(source, field)
    positions = np.asarray(source.get_channel_locations())
    id_to_index = {value: index for index, value in enumerate(source.get_channel_ids().tolist())}
    common_positions = positions[[id_to_index[value] for value in ids]]
    fs = float(source.get_sampling_frequency())
    start, stop = [int(round(value * fs)) for value in INTERVAL_S]
    contract = {
        "schema": SCHEMA,
        "development_only": True,
        "field": str(Path(cfg["field"])),
        "field_sha256": cfg["field_sha256"],
        "interval_s": list(INTERVAL_S),
        "field_terminal_support_s": [float(field["time_s"][0]), float(field["time_s"][-1])],
        "terminal_nearest_extension_s": [float(field["time_s"][0] - INTERVAL_S[0]),
                                           float(INTERVAL_S[1] - field["time_s"][-1])],
        "source_channels": int(source.get_num_channels()),
        "common_channels": len(ids),
        "common_channel_ids": [str(value) for value in ids],
        "common_depth_um": [float(common_positions[:, 1].min()), float(common_positions[:, 1].max())],
        "crop_order": "correct full source geometry, then crop every arm to the common channel set",
        "dtype_policy": "float32 interpolation; round-nearest checked int16 materialization",
        "internal_kilosort_motion": False,
        "arms": ARMS,
    }
    contract["digest"] = fingerprint(contract)
    save(output / "contract.json", contract)

    summaries = {}
    for arm, spec in ARMS.items():
        save(output / "status.json", dict(stage="prepare_arm", arm=arm, updated_unix=time.time(), pid=os.getpid()))
        arm_root = output / "arms" / arm
        arm_root.mkdir(parents=True, exist_ok=True)
        base = source if spec is None else corrected(source, field, spec)
        view = base.frame_slice(start_frame=start, end_frame=stop).channel_slice(channel_ids=ids)
        if spec is not None:
            view = checked_int16(view)
        request = dict(schema=SCHEMA, contract_digest=contract["digest"], arm=arm,
                       correction=spec, source_content_sha256=cfg["recording_content_sha256"])
        expected_bytes = (stop - start) * len(ids) * 2
        if shutil.disk_usage(output).free < expected_bytes + int(cfg["sort_scratch_reserve_bytes"]):
            raise RuntimeError(f"insufficient disk before {arm}")
        manifest = _materialize_arm(view, arm_root / "recording", source_manifest=source_manifest,
                                    request=request, n_jobs=int(cfg["materialize_jobs"]))
        if manifest["num_samples"] != stop - start or manifest["num_channels"] != len(ids):
            raise RuntimeError(f"materialized extent changed for {arm}")
        save(output / "status.json", dict(stage="sort", arm=arm, updated_unix=time.time(), pid=os.getpid()))
        sort_manifest = run_kilosort4(arm_root / "recording", arm_root / "kilosort4")
        ops = np.load(arm_root / "kilosort4/sorter_output/ops.npy", allow_pickle=True).item()
        if int(ops["nblocks"]) != 0 or ops["dshift"] is not None:
            raise RuntimeError(f"internal Kilosort motion unexpectedly active for {arm}")
        summaries[arm] = sort_manifest["summary"]
        save(arm_root / "summary.json", dict(status="complete", arm=arm, correction=spec,
                                               sort_summary=sort_manifest["summary"]))
    save(output / "summary.json", dict(status="complete", contract_digest=contract["digest"],
                                         arms=summaries, scientific_status="comparison_pending"))
    save(output / "status.json", dict(stage="complete", updated_unix=time.time(), pid=os.getpid()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dummy", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    output = Path(cfg["output"])
    output.mkdir(parents=True, exist_ok=False)
    save(output / "request.json", cfg)
    if args.dummy:
        save(output / "status.json", dict(stage="dummy_running", pid=os.getpid()))
        time.sleep(20)
        save(output / "summary.json", dict(status="dummy_complete"))
        return
    execute(cfg, output)


if __name__ == "__main__":
    main()
