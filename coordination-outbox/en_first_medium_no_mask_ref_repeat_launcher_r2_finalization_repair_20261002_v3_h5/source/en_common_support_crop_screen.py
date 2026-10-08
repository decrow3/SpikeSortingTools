#!/usr/bin/env python3
"""Frozen zero-copy EN crop screen runner; execution requires an explicit flag."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


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


def validate_plan(config_path: Path, config: dict) -> dict:
    if config["status"] != "frozen_preoutcome":
        raise RuntimeError("screen config is not frozen")
    for label in ("runner", "reader_fixture", "crop_panel", "crop_panel_fixture", "wiring_fixture", "launcher", "kilosort_run", "kilosort_io", "kilosort_template_matching", "tier1_panel"):
        path = Path(config["source"][f"{label}_path"])
        if sha256(path) != config["source"][f"{label}_sha256"]:
            raise RuntimeError(f"{label} source hash differs from frozen config")
    from kilosort.io import load_probe, remove_bad_channels
    from kilosort.parameters import DEFAULT_SETTINGS
    from pipeline.kilosort_compat import ensure_kilosort_compatibility
    from pipeline.sorting import build_kilosort4_params
    patch = ensure_kilosort_compatibility()
    if patch["expected_patched_sha256"] != config["sorter"]["compatibility_patch_sha256"]:
        raise RuntimeError("Kilosort compatibility patch differs")
    production = build_kilosort4_params()
    consumed = {key: value for key, value in production.items() if key in DEFAULT_SETTINGS}
    consumed.update(
        nblocks=0, n_chan_bin=384,
        fs=float(config["crop"]["sampling_frequency_hz"]),
        tmin=float(config["crop"]["tmin_s"]), tmax=float(config["crop"]["tmax_s"]),
    )
    for key, expected in config["sorter"]["consumed_settings"].items():
        observed = consumed[key]
        if expected == "Infinity":
            if not np.isinf(observed):
                raise RuntimeError(f"effective setting {key} differs")
        elif observed != expected:
            raise RuntimeError(f"effective setting {key} differs: {observed!r} != {expected!r}")
    fs = float(config["crop"]["sampling_frequency_hz"])
    tmin, tmax = float(config["crop"]["tmin_s"]), float(config["crop"]["tmax_s"])
    effective_start, effective_end = np.int64(tmin * fs), np.int64(tmax * fs)
    if [int(effective_start), int(effective_end)] != config["crop"]["effective_frames_half_open"]:
        raise RuntimeError("Kilosort floor-frame crop differs from frozen bounds")
    if effective_end <= effective_start:
        raise RuntimeError("empty crop")
    arms = config["arms"]
    if list(arms) != ["REF384", "B384", "REF368", "B368"]:
        raise RuntimeError("unexpected arm order")
    for name, arm in arms.items():
        binary = Path(arm["binary_path"])
        manifest = Path(arm["recording_manifest_path"])
        probe = Path(arm["probe_path"])
        if binary.stat().st_size != int(arm["binary_bytes"]):
            raise RuntimeError(f"{name} binary size differs")
        if sha256(manifest) != arm["recording_manifest_sha256"]:
            raise RuntimeError(f"{name} recording manifest hash differs")
        if sha256(probe) != arm["probe_sha256"]:
            raise RuntimeError(f"{name} probe hash differs")
        manifest_payload = json.loads(manifest.read_text())
        saved_files = manifest_payload.get("recording_binary_files", [])
        if len(saved_files) != 1 or saved_files[0].get("sha256") != arm["accepted_binary_sha256"]:
            raise RuntimeError(f"{name} accepted binary receipt differs")
        bad = [int(value) for value in arm["bad_channels"]]
        expected_bad = [] if name.endswith("384") else list(range(12)) + list(range(380, 384))
        if bad != expected_bad:
            raise RuntimeError(f"{name} support mask differs")
        loaded_probe = load_probe(probe_path=probe)
        expected_from_geometry = np.flatnonzero(
            (np.asarray(loaded_probe["yc"]) >= 120.0) & (np.asarray(loaded_probe["yc"]) <= 3780.0)
        )
        geometry_bad = np.setdiff1d(np.arange(384), expected_from_geometry).tolist()
        if name.endswith("368") and bad != geometry_bad:
            raise RuntimeError(f"{name} bad_channels are not derived from actual geometry")
        selected_probe = remove_bad_channels(loaded_probe, bad) if bad else loaded_probe
        map_start, map_stop = arm["retained_channel_map_half_open"]
        if not np.array_equal(selected_probe["chanMap"], np.arange(map_start, map_stop)):
            raise RuntimeError(f"{name} actual selected channel order differs")
    return {
        "config_path": str(config_path.resolve()),
        "config_sha256": sha256(config_path),
        "effective_start_frame": int(effective_start),
        "effective_end_frame_exclusive": int(effective_end),
        "effective_samples": int(effective_end - effective_start),
        "arms": list(arms),
        "compatibility_patch": patch,
        "consumed_settings": config["sorter"]["consumed_settings"],
    }


def run_arm(name: str, arm: dict, config: dict, root: Path) -> dict:
    import torch
    from kilosort import template_matching
    from kilosort.io import load_probe
    from kilosort.parameters import DEFAULT_SETTINGS
    from kilosort.run_kilosort import run_kilosort
    from pipeline.sorting import build_kilosort4_params
    from pipeline.kilosort_compat import ensure_kilosort_compatibility

    final = root / name
    partial = root / f"{name}.partial"
    if final.exists() or partial.exists():
        raise FileExistsError(f"refusing existing arm namespace: {name}")
    partial.mkdir(parents=True)
    status = partial / "STATUS.json"
    save(status, {"stage": "starting", "arm": name, "updated_at": datetime.now(timezone.utc).isoformat()})

    compatibility_patch = ensure_kilosort_compatibility()
    params = build_kilosort4_params()
    settings = {key: value for key, value in params.items() if key in DEFAULT_SETTINGS}
    settings.update(
        n_chan_bin=384,
        fs=float(config["crop"]["sampling_frequency_hz"]),
        tmin=float(config["crop"]["tmin_s"]),
        tmax=float(config["crop"]["tmax_s"]),
        nblocks=0,
    )
    probe = load_probe(probe_path=Path(arm["probe_path"]))
    snapshots = partial / "pre_extraction_snapshot"
    snapshots.mkdir()
    original_extract = template_matching.extract
    snapshot_written = False

    def snapshot_extract(ops, bfile, U, device=torch.device("cuda"), progress_bar=None):
        nonlocal snapshot_written
        if snapshot_written:
            raise RuntimeError("pre-extraction matching bank encountered more than once")
        np.save(snapshots / "Wall3.npy", U.detach().cpu().numpy())
        np.save(snapshots / "wPCA.npy", ops["wPCA"].detach().cpu().numpy())
        np.save(snapshots / "wTEMP.npy", ops["wTEMP"].detach().cpu().numpy())
        snapshot_written = True
        result = original_extract(ops, bfile, U, device=device, progress_bar=progress_bar)
        returned_ops = result[2]
        for key in ("iU", "iCC", "iCC_mask"):
            value = returned_ops[key]
            if hasattr(value, "detach"):
                value = value.detach().cpu().numpy()
            np.save(snapshots / f"{key}.npy", np.asarray(value))
        np.save(snapshots / "chanMap.npy", np.asarray(returned_ops["chanMap"], dtype=np.int64))
        selected_iU = np.asarray(returned_ops["iU"].detach().cpu().numpy(), dtype=np.int64)
        physical_iU = np.asarray(returned_ops["chanMap"], dtype=np.int64)[selected_iU]
        np.save(snapshots / "iU_physical_binary_channel.npy", physical_iU)
        save(snapshots / "RECEIPT.json", {
            "schema": "en-pre-extraction-bank-snapshot-v1",
            "Wall3_shape": list(U.shape),
            "wPCA_shape": list(ops["wPCA"].shape),
            "wTEMP_shape": list(ops["wTEMP"].shape),
            "template_index_ancestry": "Wall3 axis 0 is the exact template index written to st[:,1]/spike_detection_templates. iU is the selected-probe channel index; iU_physical_binary_channel equals ops.chanMap[iU] and maps it to the 384-channel binary row.",
            "consumer": "template_matching.extract exact U argument",
        })
        return result

    template_matching.extract = snapshot_extract
    started = time.perf_counter()
    try:
        save(status, {"stage": "sorting", "arm": name, "updated_at": datetime.now(timezone.utc).isoformat()})
        run_kilosort(
            settings=settings,
            probe=probe,
            filename=Path(arm["binary_path"]),
            results_dir=partial / "sorter_output",
            data_dtype="int16",
            do_CAR=True,
            invert_sign=False,
            device=torch.device("cuda"),
            save_extra_vars=True,
            clear_cache=True,
            save_preprocessed_copy=False,
            bad_channels=arm["bad_channels"] or None,
            verbose_console=True,
            verbose_log=True,
        )
    finally:
        template_matching.extract = original_extract
    if not snapshot_written:
        raise RuntimeError("exact pre-extraction bank was not captured")

    sorter = partial / "sorter_output"
    ops = np.load(sorter / "ops.npy", allow_pickle=True).item()
    expected_channels = 384 - len(arm["bad_channels"])
    if int(ops["n_chan_bin"]) != 384 or int(ops["Nchan"]) != expected_channels:
        raise RuntimeError("saved binary stride or selected channel count differs")
    if int(ops["nblocks"]) != 0 or ops.get("dshift") is not None:
        raise RuntimeError("internal motion correction was not effectively disabled")
    map_start, map_stop = arm["retained_channel_map_half_open"]
    expected_map = np.arange(map_start, map_stop, dtype=np.int64)
    if not np.array_equal(np.asarray(ops["chanMap"]), expected_map):
        raise RuntimeError("saved chanMap differs from frozen physical-channel order")
    start, stop = config["crop"]["effective_frames_half_open"]
    spike_times = np.load(sorter / "spike_times.npy", mmap_mode="r").reshape(-1)
    if len(spike_times) and (int(spike_times.min()) < start or int(spike_times.max()) >= stop):
        raise RuntimeError("exported spike time lies outside frozen global crop")
    receipt = {
        "schema": "en-common-support-crop-arm-v1",
        "status": "complete",
        "arm": name,
        "wall_seconds": time.perf_counter() - started,
        "n_chan_bin": int(ops["n_chan_bin"]),
        "Nchan": int(ops["Nchan"]),
        "chanMap": np.asarray(ops["chanMap"]).astype(int).tolist(),
        "tmin": float(ops["tmin"]),
        "tmax": float(ops["tmax"]),
        "global_spike_time_bounds": (
            [int(spike_times.min()), int(spike_times.max())] if len(spike_times) else None
        ),
        "spikes": int(len(spike_times)),
        "pre_extraction_snapshot": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in sorted(snapshots.glob("*.npy"))
        },
        "algorithm_change": False,
        "compatibility_patch": compatibility_patch,
        "consumed_settings": config["sorter"]["consumed_settings"],
    }
    save(partial / "RECEIPT.json", receipt)
    save(status, {"stage": "complete", "arm": name, "updated_at": datetime.now(timezone.utc).isoformat()})
    os.replace(partial, final)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute-frozen-sorts", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    plan = validate_plan(args.config, config)
    if not args.execute_frozen_sorts:
        print(json.dumps({"status": "preflight_only", **plan}, indent=2, sort_keys=True))
        return
    if os.environ.get("EN_CROP_SCREEN_PERSISTENT_JOB") != "1":
        raise RuntimeError("sort execution is allowed only inside the frozen persistent launcher")
    if args.output.exists():
        raise FileExistsError(args.output)
    if shutil.disk_usage(args.output.parent).free < int(config["resources"]["minimum_free_bytes"]):
        raise RuntimeError("insufficient free space for four sorter outputs")
    args.output.mkdir(parents=True)
    save(args.output / "PLAN_RECEIPT.json", plan)
    results = {}
    for name, arm in config["arms"].items():
        results[name] = run_arm(name, arm, config, args.output)
    save(args.output / "COMPLETE.json", {"schema": "en-common-support-crop-screen-complete-v1", "status": "complete", "arms": results})


if __name__ == "__main__":
    main()
