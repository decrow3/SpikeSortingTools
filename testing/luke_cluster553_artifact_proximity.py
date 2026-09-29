"""Artifact-sidecar proximity at matched and missed cluster 553 anchor events."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from testing.luke_cluster21_detection_lineage import detection_template_depths
from testing.luke_cluster452_identity_replay import exclusive_pairs, interval, sha256, template_depths


def nearest_distance_frames(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.int64)
    reference = np.asarray(reference, dtype=np.int64)
    if not len(reference):
        return np.full(values.shape, np.iinfo(np.int64).max, dtype=np.int64)
    insertion = np.searchsorted(reference, values)
    left = np.maximum(insertion - 1, 0)
    right = np.minimum(insertion, len(reference) - 1)
    return np.minimum(np.abs(values - reference[left]), np.abs(reference[right] - values))


def proximity(distances: np.ndarray, fs: float) -> dict[str, float]:
    return {
        label: float(np.mean(distances <= int(round(ms * 1e-3 * fs)))) if len(distances) else 0.0
        for label, ms in (("0p5ms", 0.5), ("1ms", 1.0), ("2ms", 2.0), ("5ms", 5.0))
    }


def artifact_supported(unmatched: float, matched: float, rule: dict) -> bool:
    return bool(
        unmatched >= float(rule["minimum_absolute_fraction"])
        or (
            unmatched >= matched + float(rule["minimum_increase_over_failing_matched"])
            and unmatched >= float(rule["minimum_ratio_over_failing_matched"]) * max(matched, 1e-12)
        )
    )


def run(config_path: Path, output: Path) -> dict:
    if output.exists():
        raise RuntimeError(f"output exists; preserve prior evidence: {output}")
    try:
        import h5py
    except ModuleNotFoundError as exc:
        raise RuntimeError("artifact proximity requires h5py") from exc
    config = json.loads(config_path.read_text())
    sorter, curated = Path(config["sorter_output"]), Path(config["curated_output"])
    legacy = Path(config["legacy_curated"])
    fs = float(config["sampling_frequency_hz"])
    tolerance = int(round(float(config["tolerance_ms"]) * 1e-3 * fs))
    target = int(config["target_cluster"])
    target_depth = float(template_depths(curated)[target])
    detection_depth = detection_template_depths(np.load(sorter / "ops.npy", allow_pickle=True).item())
    detection_ids = np.flatnonzero(
        np.abs(detection_depth - target_depth) <= float(config["maximum_depth_difference_um"])
    )
    full_st = np.load(sorter / "full_st.npy", mmap_mode="r")
    full_mask = np.isin(np.asarray(full_st[:, 1], dtype=np.int64), detection_ids)
    detected = np.array(full_st[full_mask, 0], dtype=np.int64, copy=True)
    detected.sort(kind="stable")
    legacy_times = np.load(legacy / "spike_times.npy", mmap_mode="r").reshape(-1)
    legacy_labels = np.load(legacy / "spike_clusters.npy", mmap_mode="r").reshape(-1)
    anchor = np.array(
        legacy_times[legacy_labels == int(config["legacy_anchor_cluster"])], dtype=np.int64, copy=True
    )
    anchor.sort(kind="stable")
    with h5py.File(config["artifact_sidecar"], "r") as handle:
        if not bool(handle.attrs["complete"]):
            raise RuntimeError("artifact sidecar is incomplete")
        sidecar_fs = float(handle.attrs["sampling_frequency_hz"])
        if not np.isclose(sidecar_fs, fs, rtol=0, atol=1e-9):
            raise RuntimeError("artifact sidecar sampling frequency differs")
        artifact_samples = np.asarray(handle["claim_active_sample_index"], dtype=np.int64)
        sidecar = {
            "source_stage": str(handle.attrs["source_stage"]),
            "threshold_uv": float(handle.attrs["threshold_uv"]),
            "claim_active_sample_count": int(handle.attrs["claim_active_sample_count"]),
        }
    partitions = {}
    for window in config["windows"]:
        start, stop = int(round(window["start_s"] * fs)), int(round(window["stop_s"] * fs))
        a = anchor[interval(anchor, start, stop)]
        d = detected[interval(detected, start, stop)]
        anchor_hit, _ = exclusive_pairs(a, d, tolerance)
        matched = np.zeros(len(a), dtype=bool); matched[anchor_hit] = True
        for name, values in (("matched", a[matched]), ("unmatched", a[~matched])):
            key = f"{window['role']}_{name}"
            distances = nearest_distance_frames(values, artifact_samples)
            partitions[key] = {"events": int(len(values)), **proximity(distances, fs)}
    rule = config["artifact_support_rule"]
    supported = artifact_supported(
        partitions["failing_unmatched"]["2ms"], partitions["failing_matched"]["2ms"], rule
    )
    result = {
        "schema": config["schema"], "status": "complete",
        "verdict": "artifact_association_supported" if supported else "artifact_association_not_supported",
        "config_sha256": sha256(config_path), "target_cluster": target,
        "legacy_anchor_cluster": int(config["legacy_anchor_cluster"]),
        "target_depth_um": target_depth, "partitions": partitions,
        "sidecar": sidecar, "sort_launched": False, "raw_voltage_read": False,
        "production_changed": False,
    }
    output.mkdir(parents=True)
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/luke_cluster553_artifact_proximity.v1.json"))
    parser.add_argument("--output", type=Path, default=Path("testing/outputs/luke_cluster553_artifact_proximity_v1"))
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.output), indent=2))


if __name__ == "__main__":
    main()
