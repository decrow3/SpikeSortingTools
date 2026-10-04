#!/usr/bin/env python3
"""Audit effective saved Kilosort parameters for the four-arm crop screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from testing.en_tier1_panel import sha256


ARMS = ("REF384", "B384", "REF368", "B368")
SCALARS = (
    "fs", "n_chan_bin", "Nchan", "tmin", "tmax", "nblocks", "do_CAR",
    "Th_universal", "Th_learned", "Th_single_ch", "duplicate_spike_ms",
    "nearest_chans", "nearest_templates", "max_channel_distance", "nt", "nskip",
    "whitening_range",
)
ARRAYS = ("chanMap", "xc", "yc")


def clean(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def effective(ops: dict[str, Any], key: str) -> Any:
    """Prefer the effective top-level value over the nested input settings."""
    return ops[key] if key in ops else ops.get("settings", {})[key]


def array_receipt(value: Any) -> dict[str, Any]:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256(array.view(np.uint8)).hexdigest()
    return {"shape": list(array.shape), "dtype": str(array.dtype), "sha256_memory": digest,
            "values": array.tolist()}


def scalar_receipt(ops: dict[str, Any], key: str, expected: Any) -> dict[str, Any]:
    nested = ops.get("settings", {})
    source = "top_level" if key in ops else "nested_settings"
    actual = clean(effective(ops, key))
    return {"top_level": clean(ops.get(key)), "nested_settings": clean(nested.get(key)),
            "effective": actual, "source_key": source, "expected_frozen": clean(expected),
            "matches_frozen": equal(actual, expected)}


def audit_arm(root: Path, arm: str, config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    arm_root = root / arm
    receipt = json.loads((arm_root / "RECEIPT.json").read_text())
    if receipt.get("status") != "complete":
        raise RuntimeError(f"{arm}: incomplete receipt")
    ops_path = arm_root / "sorter_output/ops.npy"
    ops = np.load(ops_path, allow_pickle=True).item()
    expected_settings = config["sorter"]["consumed_settings"]
    expected_scalars = {key: expected_settings.get(key) for key in SCALARS}
    expected_scalars["Nchan"] = 384 if arm.endswith("384") else 368
    expected_scalars["do_CAR"] = config["sorter"]["do_CAR"]
    values = {key: scalar_receipt(ops, key, expected_scalars[key]) for key in SCALARS}
    if not all(row["matches_frozen"] for row in values.values()):
        raise RuntimeError(f"{arm}: effective scalar differs from frozen config")
    values["dshift"] = None if ops.get("dshift") is None else array_receipt(ops["dshift"])
    values.update({key: array_receipt(effective(ops, key)) for key in ARRAYS})
    snapshot_root = arm_root / "pre_extraction_snapshot"
    snapshots = {}
    for name in ("Wall3.npy", "wPCA.npy", "wTEMP.npy", "chanMap.npy", "iU.npy",
                 "iU_physical_binary_channel.npy", "iCC.npy", "iCC_mask.npy"):
        path = snapshot_root / name
        array = np.load(path, mmap_mode="r")
        snapshots[name] = {"shape": list(array.shape), "dtype": str(array.dtype),
                           "bytes": path.stat().st_size, "sha256": sha256(path)}
    detection_path = arm_root / "sorter_output/spike_detection_templates.npy"
    detection = np.load(detection_path, mmap_mode="r").reshape(-1)
    spike_path = arm_root / "sorter_output/spike_times.npy"
    spikes = np.load(spike_path, mmap_mode="r").reshape(-1)
    start, stop = map(int, config["crop"]["effective_frames_half_open"])
    spike_summary = {"events": int(len(spikes)), "minimum": int(spikes.min()) if len(spikes) else None,
                     "maximum": int(spikes.max()) if len(spikes) else None,
                     "bytes": spike_path.stat().st_size, "sha256": sha256(spike_path),
                     "inside_frozen_half_open_crop": bool(not len(spikes) or (spikes.min() >= start and spikes.max() < stop))}
    if not spike_summary["inside_frozen_half_open_crop"]:
        raise RuntimeError(f"{arm}: saved spike time outside frozen crop")
    detection_summary = {"events": int(len(detection)),
                         "minimum": int(detection.min()) if len(detection) else None,
                         "maximum": int(detection.max()) if len(detection) else None,
                         "unique": int(len(np.unique(detection))), "sha256": sha256(detection_path)}
    wall_count = snapshots["Wall3.npy"]["shape"][0]
    if len(detection) and (detection_summary["minimum"] < 0 or detection_summary["maximum"] >= wall_count):
        raise RuntimeError(f"{arm}: detection-template ID lies outside captured Wall3")
    if len(detection) != len(spikes):
        raise RuntimeError(f"{arm}: detection-template and spike-time lengths differ")
    snapshot_chan_map = np.load(snapshot_root / "chanMap.npy")
    snapshot_iu = np.load(snapshot_root / "iU.npy")
    physical = np.load(snapshot_root / "iU_physical_binary_channel.npy")
    if len(snapshot_iu) != wall_count:
        raise RuntimeError(f"{arm}: iU length differs from Wall3 axis 0")
    if not np.array_equal(snapshot_chan_map, np.asarray(ops["chanMap"])):
        raise RuntimeError(f"{arm}: snapshot chanMap differs from effective ops")
    if not np.array_equal(physical, snapshot_chan_map[snapshot_iu.astype(np.int64)]):
        raise RuntimeError(f"{arm}: physical template channel ancestry differs")
    return ({"arm": arm, "receipt_wall_seconds": receipt["wall_seconds"],
             "ops": {"path": str(ops_path), "bytes": ops_path.stat().st_size, "sha256": sha256(ops_path)},
             "effective": values, "snapshots": snapshots,
             "saved_spike_times": spike_summary, "detection_template_ids": detection_summary,
             "ancestry_checks": {"detection_length_equals_spike_times": True,
                                 "iU_length_equals_Wall3_axis0": True,
                                 "snapshot_chanMap_equals_effective_ops": True,
                                 "physical_iU_equals_chanMap_index_iU": True}}, ops)


def equal(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return left is right
    return bool(np.array_equal(np.asarray(left), np.asarray(right), equal_nan=True))


def run(root: Path, config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text())
    rows, raw = {}, {}
    for arm in ARMS:
        rows[arm], raw[arm] = audit_arm(root, arm, config)
    invariant = ("fs", "n_chan_bin", "tmin", "tmax", "nblocks", "do_CAR", "Th_universal",
                 "Th_learned", "Th_single_ch", "duplicate_spike_ms", "nearest_chans",
                 "nearest_templates", "max_channel_distance", "nt", "nskip", "whitening_range")
    invariant_checks = {key: all(equal(effective(raw[ARMS[0]], key), effective(raw[arm], key))
                                     for arm in ARMS[1:]) for key in invariant}
    invariant_checks["dshift_none_all_arms"] = all(raw[arm].get("dshift") is None for arm in ARMS)
    within_support = {}
    for key in ("Nchan", "chanMap", "xc", "yc"):
        within_support[key] = {
            "REF384_equals_B384": equal(effective(raw["REF384"], key), effective(raw["B384"], key)),
            "REF368_equals_B368": equal(effective(raw["REF368"], key), effective(raw["B368"], key)),
        }
    expected = {
        "Nchan": {arm: int(effective(raw[arm], "Nchan")) for arm in ARMS},
        "chanMap_384": np.array_equal(effective(raw["REF384"], "chanMap"), np.arange(384)),
        "chanMap_368": np.array_equal(effective(raw["REF368"], "chanMap"), np.arange(12, 380)),
    }
    if not all(invariant_checks.values()) or not all(all(value.values()) for value in within_support.values()):
        raise RuntimeError("unintended effective-parameter difference across arms")
    if expected["Nchan"] != {"REF384": 384, "B384": 384, "REF368": 368, "B368": 368}:
        raise RuntimeError("unexpected effective Nchan")
    if not expected["chanMap_384"] or not expected["chanMap_368"]:
        raise RuntimeError("unexpected effective chanMap")
    return {"schema": "en-common-support-crop-ops-audit-v1", "status": "pass",
            "source": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
            "config": {"path": str(config_path.resolve()), "sha256": sha256(config_path)},
            "sort_root": str(root.resolve()), "arms": rows,
            "invariant_effective_values_equal": invariant_checks,
            "support_matched_geometry_equal": within_support,
            "intended_support_differences": expected,
            "adaptive_refits": "Wrot, wPCA, wTEMP, Wall3, templates and cluster assignments may differ by arm and are preserved, not treated as parity failures.",
            "raw_voltage_bytes_read": 0, "sorts_launched": 0}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/media/huklaban5/Data/en_common_support_crop_screen_20260930_v1"))
    parser.add_argument("--config", type=Path, default=Path("configs/en_common_support_crop_screen.v1.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = run(args.root, args.config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
