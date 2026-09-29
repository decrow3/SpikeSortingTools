"""Check whether cluster 553 anchor events escaped the prior spatial full_st audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from testing.luke_cluster21_detection_lineage import detection_template_depths
from testing.luke_cluster452_identity_replay import exclusive_pairs, interval, sha256, template_depths


def spatial_escape_summary(
    anchor: np.ndarray,
    all_times: np.ndarray,
    all_depths: np.ndarray,
    target_depth: float,
    radius_um: float,
    tolerance: int,
) -> dict:
    """Compare spatially bounded and whole-probe one-to-one event matches."""
    anchor = np.asarray(anchor, dtype=np.int64)
    all_times = np.asarray(all_times, dtype=np.int64)
    all_depths = np.asarray(all_depths, dtype=float)
    if len(all_times) != len(all_depths):
        raise ValueError("detection times and depths must align")
    near = np.abs(all_depths - target_depth) <= radius_um
    spatial_anchor_hit, _ = exclusive_pairs(anchor, all_times[near], tolerance)
    whole_anchor_hit, whole_detection_hit = exclusive_pairs(anchor, all_times, tolerance)
    spatial_mask = np.zeros(len(anchor), dtype=bool)
    spatial_mask[spatial_anchor_hit] = True
    whole_mask = np.zeros(len(anchor), dtype=bool)
    whole_mask[whole_anchor_hit] = True
    escaped_anchor = whole_mask & ~spatial_mask
    unmatched = ~spatial_mask
    depth_by_anchor = np.full(len(anchor), np.nan)
    depth_by_anchor[whole_anchor_hit] = all_depths[whole_detection_hit]
    escaped_offsets = depth_by_anchor[escaped_anchor] - target_depth
    return {
        "anchor_events": int(len(anchor)),
        "spatial_matched_events": int(np.count_nonzero(spatial_mask)),
        "spatial_matched_fraction": float(np.mean(spatial_mask)) if len(anchor) else 0.0,
        "whole_probe_matched_events": int(np.count_nonzero(whole_mask)),
        "whole_probe_matched_fraction": float(np.mean(whole_mask)) if len(anchor) else 0.0,
        "spatially_unmatched_events": int(np.count_nonzero(unmatched)),
        "spatial_escape_events": int(np.count_nonzero(escaped_anchor)),
        "temporal_match_fraction_among_spatially_unmatched": float(
            np.count_nonzero(escaped_anchor) / max(np.count_nonzero(unmatched), 1)
        ),
        "escaped_depth_offset_um": {
            "median": float(np.median(escaped_offsets)) if len(escaped_offsets) else None,
            "minimum": float(np.min(escaped_offsets)) if len(escaped_offsets) else None,
            "maximum": float(np.max(escaped_offsets)) if len(escaped_offsets) else None,
        },
    }


def run(config_path: Path, output: Path) -> dict:
    if output.exists():
        raise RuntimeError(f"output exists; preserve prior evidence: {output}")
    config = json.loads(config_path.read_text())
    sorter = Path(config["sorter_output"])
    fs = float(config["sampling_frequency_hz"])
    tolerance = int(round(float(config["tolerance_ms"]) * 1e-3 * fs))
    target_depth = float(template_depths(Path(config["curated_output"]))[int(config["target_cluster"])])

    ops = np.load(sorter / "ops.npy", allow_pickle=True).item()
    detection_depths = detection_template_depths(ops)
    full_st = np.load(sorter / "full_st.npy", mmap_mode="r")
    full_times = np.asarray(full_st[:, 0], dtype=np.int64)
    template_ids = np.asarray(full_st[:, 1], dtype=np.int64)
    if np.any(template_ids < 0) or np.any(template_ids >= len(detection_depths)):
        raise RuntimeError("full_st contains a detection-template id outside ops.iU")
    full_depths = detection_depths[template_ids]
    order = np.argsort(full_times, kind="stable")
    full_times, full_depths = full_times[order], full_depths[order]

    legacy = Path(config["legacy_curated"])
    legacy_times = np.load(legacy / "spike_times.npy", mmap_mode="r").reshape(-1)
    legacy_labels = np.load(legacy / "spike_clusters.npy", mmap_mode="r").reshape(-1)
    anchor_times = np.array(
        legacy_times[legacy_labels == int(config["legacy_anchor_cluster"])],
        dtype=np.int64,
        copy=True,
    )
    anchor_times.sort(kind="stable")

    windows = []
    for window in config["windows"]:
        start = int(round(float(window["start_s"]) * fs))
        stop = int(round(float(window["stop_s"]) * fs))
        a = anchor_times[interval(anchor_times, start, stop)]
        window_slice = interval(full_times, start, stop)
        summary = spatial_escape_summary(
            a, full_times[window_slice], full_depths[window_slice], target_depth,
            float(config["spatial_radius_um"]), tolerance,
        )
        summary.update({"name": window["name"], "role": window["role"]})
        windows.append(summary)

    failing = next(row for row in windows if row["role"] == "failing")
    boundary = float(config["spatial_escape_support_rule"][
        "minimum_temporal_match_fraction_among_spatially_unmatched"
    ])
    verdict = (
        "whole_probe_spatial_escape_supported"
        if failing["temporal_match_fraction_among_spatially_unmatched"] >= boundary
        else "universal_detection_absence_supported"
    )
    result = {
        "schema": config["schema"], "status": "complete", "verdict": verdict,
        "config_sha256": sha256(config_path), "target_cluster": int(config["target_cluster"]),
        "legacy_anchor_cluster": int(config["legacy_anchor_cluster"]),
        "target_depth_um": target_depth, "windows": windows,
        "spatial_escape_support_boundary": boundary,
        "full_st_semantics": "Saved Kilosort universal-template detections; whole-probe matching removes the prior depth gate but retains the frozen 0.5 ms event tolerance.",
        "sort_launched": False, "raw_voltage_read": False, "production_changed": False,
    }
    output.mkdir(parents=True)
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/luke_cluster553_whole_probe_detection.v1.json"))
    parser.add_argument("--output", type=Path, default=Path("testing/outputs/luke_cluster553_whole_probe_detection_v1"))
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.output), indent=2))


if __name__ == "__main__":
    main()
