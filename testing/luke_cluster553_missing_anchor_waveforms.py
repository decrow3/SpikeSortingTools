"""Sparse voltage check at independently anchored events missed by full_st."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from testing.luke_cluster21_detection_lineage import detection_template_depths
from testing.luke_cluster452_identity_replay import (
    exclusive_pairs,
    interval,
    sha256,
    template_cosine,
    template_depths,
)


def evenly_bounded(values: np.ndarray, maximum: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.int64)
    if len(values) <= maximum:
        return values
    return values[np.linspace(0, len(values) - 1, maximum, dtype=int)]


def waveform_metrics(waves: np.ndarray, reference: np.ndarray | None, max_lag: int) -> tuple[dict, np.ndarray | None]:
    if not len(waves):
        return {"sampled_events": 0, "median_waveform_ptp_uv": None,
                "reference_cosine": None, "reference_best_lag_samples": None}, None
    median = np.median(waves.astype(np.float64), axis=0)
    ptp = float(np.max(np.ptp(median, axis=0)))
    cosine, lag = template_cosine(reference, median, max_lag) if reference is not None else (1.0, 0)
    return {"sampled_events": int(len(waves)), "median_waveform_ptp_uv": ptp,
            "reference_cosine": cosine, "reference_best_lag_samples": lag}, median


def extract(raw: np.memmap, times: np.ndarray, channels: np.ndarray, pre: int, post: int,
            baseline: int, gain: float) -> np.ndarray:
    times = np.asarray(times, dtype=np.int64)
    times = times[(times >= pre) & (times < raw.shape[0] - post)]
    waves = np.empty((len(times), pre + post, len(channels)), dtype=np.float32)
    for index, time in enumerate(times):
        wave = np.array(raw[time - pre:time + post, channels], dtype=np.float32, copy=True)
        wave *= gain
        wave -= np.median(wave[:baseline], axis=0, keepdims=True)
        waves[index] = wave
    return waves


def run(config_path: Path, output: Path) -> dict:
    if output.exists():
        raise RuntimeError(f"output exists; preserve prior evidence: {output}")
    config = json.loads(config_path.read_text())
    sorter = Path(config["sorter_output"])
    legacy = Path(config["legacy_curated"])
    target = int(config["target_cluster"])
    anchor_id = int(config["legacy_anchor_cluster"])
    fs = float(config["sampling_frequency_hz"])
    tolerance = int(round(float(config["tolerance_ms"]) * 1e-3 * fs))
    target_depth = float(template_depths(Path(config["sorter_output"]).parent.parent / "cur/cur_output")[target])
    ops = np.load(sorter / "ops.npy", allow_pickle=True).item()
    detection_depth = detection_template_depths(ops)
    detection_ids = np.flatnonzero(
        np.abs(detection_depth - target_depth) <= float(config["maximum_depth_difference_um"])
    )
    full_st = np.load(sorter / "full_st.npy", mmap_mode="r")
    mask = np.isin(np.asarray(full_st[:, 1], dtype=np.int64), detection_ids)
    detected_times = np.array(full_st[mask, 0], dtype=np.int64, copy=True)
    detected_times.sort(kind="stable")
    legacy_times = np.load(legacy / "spike_times.npy", mmap_mode="r").reshape(-1)
    legacy_labels = np.load(legacy / "spike_clusters.npy", mmap_mode="r").reshape(-1)
    anchor_times = np.array(legacy_times[legacy_labels == anchor_id], dtype=np.int64, copy=True)
    anchor_times.sort(kind="stable")

    binary = json.loads(Path(config["binary_manifest"]).read_text())
    kwargs = binary["kwargs"]
    n_channels = int(kwargs["num_channels"])
    dtype = np.dtype(kwargs["dtype"])
    raw_path = Path(config["raw_recording"])
    n_frames = raw_path.stat().st_size // (dtype.itemsize * n_channels)
    raw = np.memmap(raw_path, mode="r", dtype=dtype, shape=(n_frames, n_channels))
    wf = config["waveform"]
    channels = np.asarray(wf["channels"], dtype=int)
    partitions: dict[str, np.ndarray] = {}
    counts = {}
    for window in config["windows"]:
        start, stop = int(round(window["start_s"] * fs)), int(round(window["stop_s"] * fs))
        anchor = anchor_times[interval(anchor_times, start, stop)]
        detected = detected_times[interval(detected_times, start, stop)]
        anchor_hit, _ = exclusive_pairs(anchor, detected, tolerance)
        matched = np.zeros(len(anchor), dtype=bool); matched[anchor_hit] = True
        for name, values in (("matched", anchor[matched]), ("unmatched", anchor[~matched])):
            key = f"{window['role']}_{name}"
            counts[key] = int(len(values))
            partitions[key] = evenly_bounded(values, int(wf["maximum_events_per_partition"]))
    waves = {
        key: extract(raw, values, channels, int(wf["pre_samples"]), int(wf["post_samples"]),
                     int(wf["baseline_samples"]), float(wf["gain_uv_per_count"]))
        for key, values in partitions.items()
    }
    _, reference = waveform_metrics(waves["reference_matched"], None, int(wf["maximum_alignment_lag_samples"]))
    metrics = {}
    for key, stack in waves.items():
        metrics[key], _ = waveform_metrics(stack, reference, int(wf["maximum_alignment_lag_samples"]))
        metrics[key]["available_events"] = counts[key]
    failure = metrics["failing_unmatched"]
    reference_ptp = metrics["reference_matched"]["median_waveform_ptp_uv"]
    ratio = float(failure["median_waveform_ptp_uv"] / reference_ptp) if reference_ptp else None
    cosine_pass = failure["reference_cosine"] is not None and failure["reference_cosine"] >= float(wf["minimum_reference_cosine"])
    attenuation_pass = ratio is not None and ratio <= float(wf["maximum_unmatched_to_reference_ptp_ratio_for_attenuation"])
    verdict = "subthreshold_like_waveform_supported" if cosine_pass and attenuation_pass else (
        "missed_similar_waveform_not_attenuated" if cosine_pass else "unmatched_anchor_waveform_not_supported"
    )
    result = {
        "schema": config["schema"], "status": "complete", "verdict": verdict,
        "config_sha256": sha256(config_path), "target_cluster": target,
        "legacy_anchor_cluster": anchor_id, "target_depth_um": target_depth,
        "partition_metrics": metrics, "failing_unmatched_to_reference_ptp_ratio": ratio,
        "raw_samples_requested": int(sum(len(v) for v in partitions.values()) * (int(wf["pre_samples"]) + int(wf["post_samples"])) * len(channels)),
        "sort_launched": False, "production_changed": False,
    }
    output.mkdir(parents=True)
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/luke_cluster553_missing_anchor_waveforms.v1.json"))
    parser.add_argument("--output", type=Path, default=Path("testing/outputs/luke_cluster553_missing_anchor_waveforms_v1"))
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.output), indent=2))


if __name__ == "__main__":
    main()
