"""Trace the cluster 21 independent anchor through cached Kilosort stages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from testing.luke_cluster452_identity_replay import (
    exclusive_pairs,
    interval,
    sha256,
    template_depths,
)


def stage_verdict(full_loss: float, retained_loss: float, threshold: float) -> str:
    if full_loss >= threshold:
        return "loss_at_or_before_full_st_detection"
    if retained_loss >= threshold:
        return "loss_between_full_st_and_retained_sorter_output"
    return "stage_attribution_unresolved"


def matched_fraction(anchor: np.ndarray, candidates: np.ndarray, tolerance: int) -> float:
    hits, _ = exclusive_pairs(anchor, candidates, tolerance)
    return float(len(hits) / max(len(anchor), 1))


def detection_template_depths(ops: dict) -> np.ndarray:
    channels = np.asarray(ops["iU"], dtype=np.int64)
    channel_depths = np.asarray(ops["yc"], dtype=float)
    if channels.ndim != 1 or np.any(channels < 0) or np.any(channels >= len(channel_depths)):
        raise RuntimeError("ops iU is not a valid detection-template to channel map")
    return channel_depths[channels]


def run(config_path: Path, output: Path) -> dict:
    if output.exists():
        raise RuntimeError(f"output exists; preserve prior evidence: {output}")
    config = json.loads(config_path.read_text())
    sorter = Path(config["sorter_output"])
    curated = Path(config["curated_output"])
    legacy = Path(config["legacy_curated"])
    fs = float(config["sampling_frequency_hz"])
    tolerance = int(round(float(config["tolerance_ms"]) * 1e-3 * fs))
    target = int(config["target_cluster"])
    anchor_id = int(config["legacy_anchor_cluster"])
    radius = float(config["maximum_depth_difference_um"])

    final_template_depth = template_depths(curated)
    target_depth = float(final_template_depth[target])
    ops = np.load(sorter / "ops.npy", allow_pickle=True).item()
    detection_depth = detection_template_depths(ops)
    template_ids = np.flatnonzero(np.abs(detection_depth - target_depth) <= radius)
    full_st = np.load(sorter / "full_st.npy", mmap_mode="r")
    full_template = np.asarray(full_st[:, 1], dtype=np.int64)
    full_mask = np.isin(full_template, template_ids)
    full_times = np.asarray(full_st[full_mask, 0], dtype=np.int64)
    full_order = np.argsort(full_times, kind="stable")
    full_times = full_times[full_order]

    kept = np.load(sorter / "kept_spikes.npy", mmap_mode="r").reshape(-1)
    if kept.dtype.kind != "b" or len(kept) != len(full_st):
        raise RuntimeError("expected boolean kept_spikes aligned one-to-one with full_st")
    retained_times_all = np.load(sorter / "spike_times.npy", mmap_mode="r").reshape(-1)
    retained_templates = np.load(sorter / "spike_detection_templates.npy", mmap_mode="r").reshape(-1)
    if int(np.count_nonzero(kept)) != len(retained_times_all):
        raise RuntimeError("kept_spikes true count differs from sorter_output row count")
    retained_mask = np.isin(retained_templates, template_ids)
    retained_times = np.array(retained_times_all[retained_mask], dtype=np.int64, copy=True)
    retained_times.sort(kind="stable")

    curated_times_all = np.load(curated / "spike_times.npy", mmap_mode="r").reshape(-1)
    curated_templates = np.load(curated / "spike_detection_templates.npy", mmap_mode="r").reshape(-1)
    curated_mask = np.isin(curated_templates, template_ids)
    curated_times = np.array(curated_times_all[curated_mask], dtype=np.int64, copy=True)
    curated_times.sort(kind="stable")

    legacy_times_all = np.load(legacy / "spike_times.npy", mmap_mode="r").reshape(-1)
    legacy_labels = np.load(legacy / "spike_clusters.npy", mmap_mode="r").reshape(-1)
    anchor_times = np.array(legacy_times_all[legacy_labels == anchor_id], dtype=np.int64, copy=True)
    anchor_times.sort(kind="stable")

    stage_times = {"full_st": full_times, "sorter_output": retained_times, "curated_output": curated_times}
    rows = []
    for window in config["windows"]:
        start = int(round(float(window["start_s"]) * fs))
        stop = int(round(float(window["stop_s"]) * fs))
        anchor = anchor_times[interval(anchor_times, start, stop)]
        row = {"name": window["name"], "role": window["role"], "anchor_events": int(len(anchor))}
        for stage, times in stage_times.items():
            candidates = times[interval(times, start, stop)]
            row[f"{stage}_candidate_events"] = int(len(candidates))
            row[f"{stage}_matched_fraction"] = matched_fraction(anchor, candidates, tolerance)
        rows.append(row)
    reference = next(row for row in rows if row["role"] == "reference")
    failing = next(row for row in rows if row["role"] == "failing")
    losses = {
        stage: float(reference[f"{stage}_matched_fraction"] - failing[f"{stage}_matched_fraction"])
        for stage in stage_times
    }
    threshold = float(config["material_loss_fraction"])
    verdict = stage_verdict(losses["full_st"], losses["sorter_output"], threshold)
    result = {
        "schema": config["schema"], "status": "complete", "verdict": verdict,
        "config_sha256": sha256(config_path), "target_cluster": target,
        "legacy_anchor_cluster": anchor_id, "target_template_depth_um": target_depth,
        "spatial_detection_template_count": int(len(template_ids)), "windows": rows,
        "reference_minus_failing_matched_fraction": losses,
        "material_loss_fraction": threshold,
        "full_st_semantics": "Kilosort full_st universal-template detections before kept_spikes; stage claim remains version-qualified to these saved arrays.",
        "sort_launched": False, "raw_voltage_read": False, "production_changed": False,
    }
    output.mkdir(parents=True)
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/luke_cluster21_detection_lineage.v1.json"))
    parser.add_argument("--output", type=Path, default=Path("testing/outputs/luke_cluster21_detection_lineage_v1"))
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.output), indent=2))


if __name__ == "__main__":
    main()
