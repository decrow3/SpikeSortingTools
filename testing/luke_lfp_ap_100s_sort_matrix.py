"""Run a matched 930--1030 s LFP/AP voltage-correction sort matrix.

This bounded development experiment materializes four recordings on one common
channel set and runs KS4 with internal motion correction disabled. Candidate
fields are frozen and identically referenced before voltage interpolation.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd

from pipeline.config import fingerprint
from pipeline.preprocess import validate_accepted_recording
from pipeline.sorting import run_kilosort4
from testing.luke_external_warp_pipeline import _materialize_arm
from testing.luke_full_session_medicine import save, sha
from testing.luke_medicine_rigid_queue import checked_int16


SCHEMA = "luke-lfp-ap-100s-sort-matrix-v1"
INTERVAL_S = (930.0, 1030.0)
REFERENCE_S = (970.0, 975.0)
ARMS = {
    "unwarped": None,
    "lfp_rigid": {"field": "lfp_rigid", "spatial": "rigid"},
    "ap_rigid": {"field": "ap_rigid", "spatial": "rigid"},
    "ap_nonrigid": {"field": "ap_nonrigid", "spatial": "nonrigid"},
}
INTERPOLATION = {"gain": 1.0, "p": 2.0, "sigma_um": 10.0}


def _reference(time_s: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = (time_s >= REFERENCE_S[0]) & (time_s < REFERENCE_S[1])
    if mask.sum() < 2:
        raise RuntimeError("candidate lacks common reference support")
    offset = np.median(values[mask], axis=0)
    return values - offset, np.atleast_1d(offset)


def load_candidate_fields(lfp_csv: Path, ap_path: Path) -> tuple[dict, dict]:
    traces = pd.read_csv(lfp_csv)
    lfp = traces.loc[
        traces.window.eq("930_1030") & traces.arm.eq("band_0p5_8")
    ].sort_values("time_s")
    if len(lfp) != 400 or not np.allclose(np.diff(lfp.time_s), 0.25):
        raise RuntimeError("unexpected frozen LFP validation grid")
    supported = lfp.supported.to_numpy(bool) & np.isfinite(lfp.pred_um)
    bad = np.flatnonzero(~supported)
    if len(bad) and (np.any(bad == 0) or np.any(bad == len(lfp) - 1) or np.any(np.diff(bad) == 1)):
        raise RuntimeError("LFP gap policy allows only bounded isolated samples")
    lfp_values = lfp.pred_um.to_numpy(float)
    lfp_values[~supported] = np.interp(
        lfp.time_s.to_numpy(float)[~supported],
        lfp.time_s.to_numpy(float)[supported],
        lfp_values[supported],
    )
    lfp_centered, lfp_offset = _reference(lfp.time_s.to_numpy(float), lfp_values[:, None])

    ap = np.load(ap_path, allow_pickle=False)
    required = {"time_s", "depth_um", "nonrigid_displacement_um", "rigid_displacement_um"}
    if not required.issubset(ap.files):
        raise RuntimeError("AP candidate package is missing arrays")
    ap_time = np.asarray(ap["time_s"], float)
    ap_depth = np.asarray(ap["depth_um"], float)
    ap_nonrigid = np.asarray(ap["nonrigid_displacement_um"], float)
    ap_rigid = np.asarray(ap["rigid_displacement_um"], float)[:, None]
    if ap_nonrigid.shape != (len(ap_time), len(ap_depth)):
        raise RuntimeError("AP nonrigid field shape mismatch")
    if not np.all(np.diff(ap_time) > 0) or not np.all(np.diff(ap_depth) > 0):
        raise RuntimeError("AP field axes are not increasing")
    ap_nonrigid, ap_nonrigid_offset = _reference(ap_time, ap_nonrigid)
    ap_rigid, ap_rigid_offset = _reference(ap_time, ap_rigid)

    fields = {
        "lfp_rigid": {
            "time_s": lfp.time_s.to_numpy(float),
            "depth_um": np.array([0.0]),
            "displacement_um": lfp_centered,
        },
        "ap_rigid": {
            "time_s": ap_time,
            "depth_um": np.array([0.0]),
            "displacement_um": ap_rigid,
        },
        "ap_nonrigid": {
            "time_s": ap_time,
            "depth_um": ap_depth,
            "displacement_um": ap_nonrigid,
        },
    }
    for name, field in fields.items():
        values = field["displacement_um"]
        if not np.all(np.isfinite(values)) or not np.all(np.diff(field["time_s"]) > 0):
            raise RuntimeError(f"invalid candidate field: {name}")
        if field["time_s"][0] - INTERVAL_S[0] > 1 or INTERVAL_S[1] - field["time_s"][-1] > 1:
            raise RuntimeError(f"candidate terminal support too far from interval: {name}")
    audit = {
        "reference_window_s": list(REFERENCE_S),
        "lfp_grid_s": 0.25,
        "lfp_supported_samples": int(supported.sum()),
        "lfp_total_samples": int(len(supported)),
        "lfp_gap_samples_filled": int((~supported).sum()),
        "lfp_gap_policy": "linear interpolation of bounded isolated 0.25 s samples only",
        "lfp_reference_offset_um": float(lfp_offset[0]),
        "ap_rigid_reference_offset_um": float(ap_rigid_offset[0]),
        "ap_nonrigid_reference_offsets_um": ap_nonrigid_offset.tolist(),
        "candidate_ranges_um": {
            name: [float(np.min(field["displacement_um"])), float(np.max(field["displacement_um"]))]
            for name, field in fields.items()
        },
    }
    return fields, audit


def motion_for(recording, field: dict):
    from spikeinterface.core.motion import Motion

    depths = field["depth_um"]
    if len(depths) == 1:
        depths = np.array([np.mean(recording.get_channel_locations()[:, 1])])
    return Motion(
        displacement=[field["displacement_um"] * INTERPOLATION["gain"]],
        temporal_bins_s=[field["time_s"] + float(recording.sample_index_to_time(0))],
        spatial_bins_um=depths,
        direction="y",
        interpolation_method="linear",
    )


def corrected(recording, field: dict):
    from spikeinterface.preprocessing import astype
    from spikeinterface.sortingcomponents.motion import InterpolateMotionRecording

    return InterpolateMotionRecording(
        astype(recording, "float32"),
        motion_for(recording, field),
        border_mode="remove_channels",
        spatial_interpolation_method="kriging",
        sigma_um=INTERPOLATION["sigma_um"],
        p=INTERPOLATION["p"],
        dtype="float32",
    )


def common_channel_ids(recording, fields: dict) -> list:
    common = set(recording.get_channel_ids().tolist())
    for name in ("lfp_rigid", "ap_rigid", "ap_nonrigid"):
        common &= set(corrected(recording, fields[name]).get_channel_ids().tolist())
    ordered = [value for value in recording.get_channel_ids().tolist() if value in common]
    if len(ordered) < 64:
        raise RuntimeError("correction matrix leaves fewer than 64 common channels")
    return ordered


def verify_config(cfg: dict) -> None:
    if cfg.get("schema") != SCHEMA:
        raise ValueError("wrong matrix schema")
    for key, hash_key in [
        ("recording_manifest", "recording_manifest_sha256"),
        ("lfp_validation_traces", "lfp_validation_traces_sha256"),
        ("ap_candidate_field", "ap_candidate_field_sha256"),
        ("lighthouse_events", "lighthouse_events_sha256"),
        ("lighthouse_families", "lighthouse_families_sha256"),
        ("lighthouse_regimes", "lighthouse_regimes_sha256"),
    ]:
        if sha(Path(cfg[key])) != cfg[hash_key]:
            raise RuntimeError(f"frozen input changed: {key}")


def save_resolved_fields(output: Path, fields: dict, audit: dict) -> None:
    np.savez_compressed(
        output / "resolved_candidate_fields.npz",
        lfp_time_s=fields["lfp_rigid"]["time_s"],
        lfp_displacement_um=fields["lfp_rigid"]["displacement_um"][:, 0],
        ap_time_s=fields["ap_rigid"]["time_s"],
        ap_rigid_displacement_um=fields["ap_rigid"]["displacement_um"][:, 0],
        ap_depth_um=fields["ap_nonrigid"]["depth_um"],
        ap_nonrigid_displacement_um=fields["ap_nonrigid"]["displacement_um"],
    )
    save(output / "field_resolution_audit.json", audit)


def execute(cfg: dict, output: Path) -> None:
    from spikeinterface.core import load

    verify_config(cfg)
    source_dir = Path(cfg["recording"])
    source_manifest = validate_accepted_recording(source_dir)
    if source_manifest["recording_content_sha256"] != cfg["recording_content_sha256"]:
        raise RuntimeError("accepted recording content changed")
    source = load(source_dir)
    fields, field_audit = load_candidate_fields(
        Path(cfg["lfp_validation_traces"]), Path(cfg["ap_candidate_field"])
    )
    ids = common_channel_ids(source, fields)
    positions = np.asarray(source.get_channel_locations())
    id_to_index = {value: index for index, value in enumerate(source.get_channel_ids().tolist())}
    common_positions = positions[[id_to_index[value] for value in ids]]
    fs = float(source.get_sampling_frequency())
    start, stop = [int(round(value * fs)) for value in INTERVAL_S]

    contract = {
        "schema": SCHEMA,
        "development_only": True,
        "interval_s": list(INTERVAL_S),
        "reference_window_s": list(REFERENCE_S),
        "source_channels": int(source.get_num_channels()),
        "common_channels": len(ids),
        "common_channel_ids": [str(value) for value in ids],
        "common_depth_um": [float(common_positions[:, 1].min()), float(common_positions[:, 1].max())],
        "crop_order": "correct full source geometry, then crop every arm to one common channel set",
        "dtype_policy": "float32 interpolation; round-nearest checked int16 materialization",
        "temporal_interpolation": "linear",
        "spatial_interpolation": "kriging",
        "interpolation_parameters": INTERPOLATION,
        "internal_kilosort_motion": False,
        "arms": ARMS,
        "field_resolution_audit": field_audit,
        "source_hashes": {
            "recording_manifest": cfg["recording_manifest_sha256"],
            "lfp_validation_traces": cfg["lfp_validation_traces_sha256"],
            "ap_candidate_field": cfg["ap_candidate_field_sha256"],
        },
        "checkpoint_policy": (
            "No within-sort checkpoint. Interruption requires preserving evidence and restarting the interrupted "
            "arm from its beginning in a new reviewed run; no automatic restart."
        ),
    }
    contract["digest"] = fingerprint(contract)
    save(output / "contract.json", contract)
    save_resolved_fields(output, fields, field_audit)

    summaries = {}
    for arm, spec in ARMS.items():
        save(output / "status.json", {"stage": "prepare_arm", "arm": arm, "updated_unix": time.time(), "pid": os.getpid()})
        arm_root = output / "arms" / arm
        arm_root.mkdir(parents=True, exist_ok=True)
        base = source if spec is None else corrected(source, fields[spec["field"]])
        view = base.frame_slice(start_frame=start, end_frame=stop).channel_slice(channel_ids=ids)
        if spec is not None:
            view = checked_int16(view)
        request = {
            "schema": SCHEMA,
            "contract_digest": contract["digest"],
            "arm": arm,
            "correction": spec,
            "source_content_sha256": cfg["recording_content_sha256"],
        }
        expected_bytes = (stop - start) * len(ids) * 2
        if shutil.disk_usage(output).free < expected_bytes + int(cfg["sort_scratch_reserve_bytes"]):
            raise RuntimeError(f"insufficient disk before {arm}")
        manifest = _materialize_arm(
            view,
            arm_root / "recording",
            source_manifest=source_manifest,
            request=request,
            n_jobs=int(cfg["materialize_jobs"]),
        )
        if manifest["num_samples"] != stop - start or manifest["num_channels"] != len(ids):
            raise RuntimeError(f"materialized extent changed for {arm}")
        save(output / "status.json", {"stage": "sort", "arm": arm, "updated_unix": time.time(), "pid": os.getpid()})
        sort_manifest = run_kilosort4(arm_root / "recording", arm_root / "kilosort4")
        ops = np.load(arm_root / "kilosort4/sorter_output/ops.npy", allow_pickle=True).item()
        if int(ops["nblocks"]) != 0 or ops["dshift"] is not None:
            raise RuntimeError(f"internal Kilosort motion unexpectedly active for {arm}")
        summaries[arm] = sort_manifest["summary"]
        save(arm_root / "summary.json", {"status": "complete", "arm": arm, "correction": spec, "sort_summary": sort_manifest["summary"]})

    save(output / "summary.json", {
        "status": "complete",
        "contract_digest": contract["digest"],
        "arms": summaries,
        "scientific_status": "comparison_pending",
    })
    from testing.luke_lfp_ap_100s_matrix_analysis import run as analyze
    save(output / "status.json", {"stage": "analysis", "updated_unix": time.time(), "pid": os.getpid()})
    analysis = analyze(
        output,
        Path(cfg["lighthouse_events"]),
        Path(cfg["lighthouse_families"]),
        Path(cfg["lighthouse_regimes"]),
    )
    save(output / "summary.json", {
        "status": "complete",
        "contract_digest": contract["digest"],
        "arms": summaries,
        "analysis_status": analysis["status"],
        "scientific_status": "development_comparison_complete_requires_review",
    })
    save(output / "status.json", {"stage": "complete", "updated_unix": time.time(), "pid": os.getpid()})


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
        save(output / "status.json", {"stage": "dummy_running", "pid": os.getpid()})
        time.sleep(20)
        save(output / "summary.json", {"status": "dummy_complete"})
        return
    execute(cfg, output)


if __name__ == "__main__":
    main()
