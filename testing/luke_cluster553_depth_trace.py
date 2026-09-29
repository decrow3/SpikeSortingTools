"""Trace the independently anchored cluster 553 waveform through physical depth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from testing.luke_cluster21_detection_lineage import detection_template_depths, matched_fraction
from testing.luke_cluster452_identity_replay import interval, sha256, template_depths
from testing.luke_cluster553_missing_anchor_waveforms import evenly_bounded, extract


def depth_profile(median_waveform: np.ndarray, channel_depths: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Maximum channel PTP at each unique physical depth."""
    depths = np.unique(np.asarray(channel_depths, dtype=float))
    channel_ptp = np.ptp(np.asarray(median_waveform, dtype=float), axis=0)
    profile = np.array([np.max(channel_ptp[channel_depths == depth]) for depth in depths])
    return depths, profile


def shifted_profile_match(reference: np.ndarray, candidate: np.ndarray,
                          maximum_lag: int) -> tuple[float, int]:
    best = (-np.inf, 0)
    for lag in range(-maximum_lag, maximum_lag + 1):
        if lag < 0:
            first, second = reference[-lag:], candidate[:lag]
        elif lag > 0:
            first, second = reference[:-lag], candidate[lag:]
        else:
            first, second = reference, candidate
        denominator = np.linalg.norm(first) * np.linalg.norm(second)
        cosine = float(np.dot(first, second) / denominator) if denominator else np.nan
        if np.isfinite(cosine) and cosine > best[0]:
            best = (cosine, lag)
    return best


def movement_supported(rows: list[dict], reference_coverage: float, rule: dict,
                       minimum_cosine: float) -> tuple[bool, list[int]]:
    indices = [
        index for index, row in enumerate(rows)
        if row["role"] == "failing"
        and row["profile_cosine"] >= minimum_cosine
        and abs(row["profile_shift_um"]) >= float(rule["minimum_absolute_shift_um"])
        and reference_coverage - row["local_full_st_coverage"] >= float(rule["minimum_coverage_loss_from_reference"])
    ]
    return bool(indices), indices


def run(config_path: Path, output: Path) -> dict:
    if output.exists():
        raise RuntimeError(f"output exists; preserve prior evidence: {output}")
    config = json.loads(config_path.read_text())
    trace = config["trace"]
    fs = float(config["sampling_frequency_hz"])
    sorter, curated = Path(config["sorter_output"]), Path(config["curated_output"])
    target_depth = float(template_depths(curated)[int(config["target_cluster"])])
    geometry = np.load(curated / "channel_positions.npy")
    channel_depth_all = np.asarray(geometry[:, 1], dtype=float)
    channels = np.flatnonzero(
        np.abs(channel_depth_all - target_depth) <= float(trace["depth_half_width_um"])
    )
    channel_depths = channel_depth_all[channels]
    unique_depths = np.unique(channel_depths)
    depth_step = float(np.median(np.diff(unique_depths)))
    maximum_lag = int(round(float(trace["maximum_profile_shift_um"]) / depth_step))

    ops = np.load(sorter / "ops.npy", allow_pickle=True).item()
    detection_depth = detection_template_depths(ops)
    local_ids = np.flatnonzero(
        np.abs(detection_depth - target_depth) <= float(trace["local_detection_radius_um"])
    )
    full = np.load(sorter / "full_st.npy", mmap_mode="r")
    local_times = np.sort(np.array(
        full[np.isin(np.asarray(full[:, 1], dtype=np.int64), local_ids), 0], dtype=np.int64
    ))
    legacy = Path(config["legacy_curated"])
    legacy_times = np.load(legacy / "spike_times.npy", mmap_mode="r").reshape(-1)
    legacy_labels = np.load(legacy / "spike_clusters.npy", mmap_mode="r").reshape(-1)
    anchor = np.sort(np.array(
        legacy_times[legacy_labels == int(config["legacy_anchor_cluster"])], dtype=np.int64
    ))

    manifest = json.loads(Path(config["binary_manifest"]).read_text())["kwargs"]
    dtype, n_channels = np.dtype(manifest["dtype"]), int(manifest["num_channels"])
    raw_path = Path(config["raw_recording"])
    frames = raw_path.stat().st_size // (dtype.itemsize * n_channels)
    raw = np.memmap(raw_path, mode="r", dtype=dtype, shape=(frames, n_channels))
    tolerance = int(round(0.5e-3 * fs))
    bin_frames = int(round(float(trace["bin_duration_s"]) * fs))
    row_inputs = []
    reference_samples = []
    for window in config["windows"]:
        start = int(round(float(window["start_s"]) * fs))
        stop = int(round(float(window["stop_s"]) * fs))
        for bin_start in range(start, stop, bin_frames):
            bin_stop = min(bin_start + bin_frames, stop)
            values = anchor[interval(anchor, bin_start, bin_stop)]
            if len(values) < int(trace["minimum_anchor_events_per_bin"]):
                continue
            sampled = evenly_bounded(values, int(trace["maximum_events_per_bin"]))
            row_inputs.append((window["role"], bin_start, bin_stop, values, sampled))
            if window["role"] == "reference":
                reference_samples.append(sampled)
    reference_times = np.concatenate(reference_samples)
    reference_waves = extract(
        raw, reference_times, channels, int(trace["pre_samples"]), int(trace["post_samples"]),
        int(trace["baseline_samples"]), float(trace["gain_uv_per_count"]),
    )
    reference_median = np.median(reference_waves, axis=0)
    profile_depths, reference_profile = depth_profile(reference_median, channel_depths)
    rows = []
    profiles = []
    for role, start, stop, values, sampled in row_inputs:
        waves = extract(
            raw, sampled, channels, int(trace["pre_samples"]), int(trace["post_samples"]),
            int(trace["baseline_samples"]), float(trace["gain_uv_per_count"]),
        )
        median = np.median(waves, axis=0)
        depths, profile = depth_profile(median, channel_depths)
        if not np.array_equal(depths, profile_depths):
            raise RuntimeError("depth profile grid changed")
        profiles.append(profile)
        cosine, lag = shifted_profile_match(reference_profile, profile, maximum_lag)
        local = local_times[interval(local_times, start, stop)]
        rows.append({
            "role": role, "start_s": start / fs, "stop_s": stop / fs,
            "anchor_events": int(len(values)), "sampled_events": int(len(sampled)),
            "local_full_st_coverage": matched_fraction(values, local, tolerance),
            "profile_cosine": cosine, "profile_shift_um": float(lag * depth_step),
            "peak_depth_um": float(depths[np.argmax(profile)]),
            "weighted_depth_um": float(np.sum(depths * profile) / np.sum(profile)),
            "median_waveform_ptp_uv": float(np.max(profile)),
        })
    reference_coverage = float(np.mean([row["local_full_st_coverage"] for row in rows if row["role"] == "reference"]))
    supported, supporting_rows = movement_supported(
        rows, reference_coverage, config["movement_support_rule"],
        float(trace["minimum_shifted_profile_cosine"]),
    )
    qualified = [row for row in rows if np.isfinite(row["profile_shift_um"])]
    correlation = spearmanr(
        [abs(row["profile_shift_um"]) for row in qualified],
        [row["local_full_st_coverage"] for row in qualified],
    ) if len(qualified) >= 3 else None
    result = {
        "schema": config["schema"], "status": "complete",
        "verdict": "spatial_movement_supported" if supported else "spatial_movement_not_supported",
        "config_sha256": sha256(config_path), "target_depth_um": target_depth,
        "channels": channels.tolist(), "depth_step_um": depth_step,
        "reference_local_full_st_coverage": reference_coverage, "rows": rows,
        "supporting_row_indices": supporting_rows,
        "absolute_shift_vs_coverage_spearman": (
            {"statistic": float(correlation.statistic), "pvalue": float(correlation.pvalue)}
            if correlation is not None else None
        ),
        "raw_samples_requested": int(sum(len(item[4]) for item in row_inputs)
                                     * (int(trace["pre_samples"]) + int(trace["post_samples"]))
                                     * len(channels)),
        "sort_launched": False, "production_changed": False,
    }
    output.mkdir(parents=True)
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    np.savez(output / "profiles.npz", depths_um=profile_depths,
             reference_profile_uv=reference_profile,
             profiles_uv=np.stack(profiles))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/luke_cluster553_depth_trace.v1.json"))
    parser.add_argument("--output", type=Path, default=Path("testing/outputs/luke_cluster553_depth_trace_v1"))
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.output), indent=2))


if __name__ == "__main__":
    main()
