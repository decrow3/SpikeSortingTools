"""Diagnose the completed full-duration, no-motion Luke core-strip sort.

This audit does not select sorter parameters.  It localizes time/depth rate
changes, strip-edge burden, near-coincident unit pairs, and within-unit
amplitude/depth/PC-feature continuity.  Existing reviewed events are not used.
Voltage residual reconstruction is intentionally deferred because this sort did
not save a residual binary.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from testing.luke_preprocessing_family_continuity_audit import count_one_to_one_matches


SORTER = Path(
    "/media/huklab/Data/NPX/Ryansorting/Luke/"
    "Luke0804_two_axis_pilot_imec1/sorts/core_depth_strip/"
    "single_ks_preprocessing_claim_off/sorter_output"
)
MOTION = Path(
    "/mnt/NPX/Luke/20250804/"
    "dredge_pipeline_results_Luke0804_V2V1_g0_imec1/motion/dredge-motion"
)
OUTPUT = Path("testing/outputs/luke_full_strip_diagnostic_audit")
TIME_BIN_S = 300.0
FEATURE_BIN_COUNT = 12
FEATURE_SAMPLES_PER_BIN = 50
COINCIDENCE_MS = 0.5
PAIR_DEPTH_UM = 100.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sorter", type=Path, default=SORTER)
    parser.add_argument("--motion-dir", type=Path, default=MOTION)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--time-bin-s", type=float, default=TIME_BIN_S)
    return parser.parse_args()


def cosine(first: np.ndarray, second: np.ndarray) -> float:
    a = np.asarray(first, dtype=float).ravel()
    b = np.asarray(second, dtype=float).ravel()
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denominator) if denominator else np.nan


def linear_slope_per_hour(seconds: np.ndarray, values: np.ndarray) -> float:
    keep = np.isfinite(seconds) & np.isfinite(values)
    if np.sum(keep) < 2 or np.ptp(seconds[keep]) == 0:
        return np.nan
    return float(np.polyfit(seconds[keep] / 3600.0, values[keep], 1)[0])


def shifted_coincidence_null(
    times: np.ndarray,
    clusters: np.ndarray,
    depths: np.ndarray,
    duration_frames: int,
    tolerance_frames: int,
    seed: int,
) -> float:
    from testing.luke_claimmask_window_sweep import (
        cross_unit_near_coincident_fraction,
    )

    rng = np.random.default_rng(seed)
    shifted = times.copy()
    for unit in np.unique(clusters):
        keep = clusters == unit
        if np.any(keep):
            offset = int(rng.integers(tolerance_frames + 1, duration_frames))
            shifted[keep] = (shifted[keep] + offset) % duration_frames
    return cross_unit_near_coincident_fraction(
        shifted, clusters, depths, tolerance_frames
    )


def load_motion(motion_dir: Path, depth_min: float, depth_max: float) -> pd.DataFrame:
    times = np.load(motion_dir / "time_bins.npy").astype(float)
    depths = np.load(motion_dir / "depth_bins.npy").astype(float)
    field = np.load(motion_dir / "motion.npy", mmap_mode="r")
    local_times = times - times[0] + 1.0
    keep = (depths >= depth_min) & (depths <= depth_max)
    selected = np.asarray(field[:, keep], dtype=float)
    rigid = np.nanmedian(selected, axis=1)
    spread = np.nanpercentile(selected, 95, axis=1) - np.nanpercentile(
        selected, 5, axis=1
    )
    step = np.r_[np.nan, np.abs(np.diff(rigid))]
    return pd.DataFrame(
        {
            "time_s": local_times,
            "rigid_um": rigid,
            "nonrigid_spread_um": spread,
            "abs_rigid_step_um": step,
        }
    )


def run_audit(
    sorter: Path, motion_dir: Path, output_dir: Path, time_bin_s: float
) -> dict:
    if time_bin_s <= 0:
        raise ValueError("time-bin duration must be positive")
    required = [
        "spike_times.npy",
        "spike_clusters.npy",
        "spike_positions.npy",
        "amplitudes.npy",
        "spike_detection_templates.npy",
        "pc_features.npy",
        "templates.npy",
        "similar_templates.npy",
        "channel_positions.npy",
        "cluster_KSLabel.tsv",
        "cluster_ContamPct.tsv",
        "ops.npy",
    ]
    missing = [name for name in required if not (sorter / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing sorter files: {missing}")
    ops = np.load(sorter / "ops.npy", allow_pickle=True).item()
    fs = float(ops["fs"])
    duration_s = float(ops["tmax"])
    if not np.isfinite(duration_s):
        binary_path = Path(ops["filename"])
        dtype = np.dtype(ops.get("data_dtype", "int16"))
        n_channels_binary = int(ops.get("n_chan_bin", ops["Nchan"]))
        duration_s = (
            binary_path.stat().st_size
            / dtype.itemsize
            / n_channels_binary
            / fs
        )
    times = np.load(sorter / "spike_times.npy", mmap_mode="r").reshape(-1)
    clusters = np.load(sorter / "spike_clusters.npy", mmap_mode="r").reshape(-1)
    positions = np.load(sorter / "spike_positions.npy", mmap_mode="r")
    amplitudes = np.load(sorter / "amplitudes.npy", mmap_mode="r").reshape(-1)
    detection_templates = np.load(
        sorter / "spike_detection_templates.npy", mmap_mode="r"
    ).reshape(-1)
    pc_features = np.load(sorter / "pc_features.npy", mmap_mode="r")
    templates = np.load(sorter / "templates.npy", mmap_mode="r")
    similarity = np.load(sorter / "similar_templates.npy").astype(float)
    channel_positions = np.load(sorter / "channel_positions.npy").astype(float)
    labels = pd.read_csv(sorter / "cluster_KSLabel.tsv", sep="\t").set_index(
        "cluster_id"
    )
    contamination = pd.read_csv(
        sorter / "cluster_ContamPct.tsv", sep="\t"
    ).set_index("cluster_id")
    label_column = labels.columns[0]
    contamination_column = contamination.columns[0]
    unit_ids = np.unique(clusters).astype(int)
    depth = np.asarray(positions[:, 1], dtype=float)
    depth_min = float(channel_positions[:, 1].min())
    depth_max = float(channel_positions[:, 1].max())
    edge_distance = np.minimum(depth - depth_min, depth_max - depth)
    edge = edge_distance <= 40.0
    n_time_bins = int(np.ceil(duration_s / time_bin_s))
    time_edges = np.minimum(
        np.arange(n_time_bins + 1, dtype=float) * time_bin_s, duration_s
    )
    time_bin = np.minimum((np.asarray(times) / fs / time_bin_s).astype(int), n_time_bins - 1)
    tolerance = int(round(COINCIDENCE_MS * fs / 1000.0))

    motion = load_motion(motion_dir, depth_min, depth_max)
    time_rows = []
    for bin_index in range(n_time_bins):
        keep = time_bin == bin_index
        bin_start = bin_index * time_bin_s
        bin_stop = min(duration_s, (bin_index + 1) * time_bin_s)
        local_times = np.asarray(times[keep], dtype=np.int64) - int(
            round(bin_start * fs)
        )
        local_clusters = np.asarray(clusters[keep], dtype=int)
        local_depths = depth[keep]
        if len(local_times):
            from testing.luke_claimmask_window_sweep import (
                cross_unit_near_coincident_fraction,
            )

            observed = cross_unit_near_coincident_fraction(
                local_times, local_clusters, local_depths, tolerance
            )
            null = shifted_coincidence_null(
                local_times,
                local_clusters,
                local_depths,
                max(tolerance + 2, int(round((bin_stop - bin_start) * fs))),
                tolerance,
                seed=19 + bin_index,
            )
        else:
            observed = null = np.nan
        motion_keep = motion.time_s.between(bin_start, bin_stop, inclusive="left")
        motion_bin = motion[motion_keep]
        time_rows.append(
            {
                "time_bin": bin_index,
                "start_s": bin_start,
                "stop_s": bin_stop,
                "spike_count": int(np.sum(keep)),
                "spikes_per_s": float(np.sum(keep) / (bin_stop - bin_start)),
                "active_units": int(np.unique(local_clusters).size),
                "edge_spike_fraction": float(np.mean(edge[keep])) if np.any(keep) else np.nan,
                "median_depth_um": float(np.median(local_depths)) if len(local_depths) else np.nan,
                "median_amplitude": float(np.median(np.asarray(amplitudes[keep]))) if np.any(keep) else np.nan,
                "coincidence_fraction": observed,
                "coincidence_shift_null": null,
                "coincidence_excess": observed - null,
                "rigid_um": float(motion_bin.rigid_um.median()) if len(motion_bin) else np.nan,
                "rigid_excursion_um": float(motion_bin.rigid_um.quantile(0.95) - motion_bin.rigid_um.quantile(0.05)) if len(motion_bin) else np.nan,
                "median_nonrigid_spread_um": float(motion_bin.nonrigid_spread_um.median()) if len(motion_bin) else np.nan,
                "p99_abs_rigid_step_um": float(motion_bin.abs_rigid_step_um.quantile(0.99)) if len(motion_bin) else np.nan,
            }
        )
    time_metrics = pd.DataFrame(time_rows)

    depth_edges = np.arange(np.floor(depth_min / 100) * 100, np.ceil(depth_max / 100) * 100 + 101, 100)
    depth_bin = np.clip(np.digitize(depth, depth_edges) - 1, 0, len(depth_edges) - 2)
    depth_rows = []
    for tbin in range(n_time_bins):
        for dbin in range(len(depth_edges) - 1):
            keep = (time_bin == tbin) & (depth_bin == dbin)
            depth_rows.append(
                {
                    "time_bin": tbin,
                    "start_s": tbin * time_bin_s,
                    "depth_start_um": depth_edges[dbin],
                    "depth_stop_um": depth_edges[dbin + 1],
                    "spike_count": int(np.sum(keep)),
                    "active_units": int(np.unique(np.asarray(clusters[keep])).size),
                    "edge_spike_fraction": float(np.mean(edge[keep])) if np.any(keep) else np.nan,
                    "median_amplitude": float(np.median(np.asarray(amplitudes[keep]))) if np.any(keep) else np.nan,
                }
            )
    time_depth_metrics = pd.DataFrame(depth_rows)

    order = np.argsort(np.asarray(clusters), kind="stable")
    sorted_clusters = np.asarray(clusters)[order]
    split = np.flatnonzero(np.diff(sorted_clusters)) + 1
    groups = np.split(order, split)
    grouped_indices = {int(np.asarray(clusters)[group[0]]): group for group in groups}
    feature_edges = np.linspace(0, duration_s, FEATURE_BIN_COUNT + 1)
    unit_rows = []
    times_by_unit: dict[int, np.ndarray] = {}
    template_peak_channels = np.argmax(np.max(np.abs(templates), axis=1), axis=1)
    template_depths = channel_positions[template_peak_channels, 1]
    for unit in unit_ids:
        indices = grouped_indices[int(unit)]
        unit_times = np.asarray(times[indices], dtype=np.int64)
        times_by_unit[int(unit)] = np.sort(unit_times)
        seconds = unit_times / fs
        unit_depth = depth[indices]
        unit_amp = np.asarray(amplitudes[indices], dtype=float)
        unit_edge = edge[indices]
        bins = np.minimum((seconds / time_bin_s).astype(int), n_time_bins - 1)
        bin_counts = np.bincount(bins, minlength=n_time_bins)
        rates = bin_counts / np.diff(time_edges)
        assigned_templates = np.asarray(detection_templates[indices], dtype=int)
        template_values, template_counts = np.unique(
            assigned_templates, return_counts=True
        )
        dominant_template = int(template_values[np.argmax(template_counts)])
        dominant_fraction = float(np.max(template_counts) / len(indices))
        feature_vectors = []
        feature_times = []
        dominant_indices = indices[assigned_templates == dominant_template]
        dominant_seconds = np.asarray(times[dominant_indices], dtype=float) / fs
        for feature_bin in range(FEATURE_BIN_COUNT):
            keep = (dominant_seconds >= feature_edges[feature_bin]) & (
                dominant_seconds < feature_edges[feature_bin + 1]
            )
            candidates = dominant_indices[keep]
            if len(candidates) < 5:
                feature_vectors.append(None)
                feature_times.append(np.nan)
                continue
            take = candidates[
                np.linspace(
                    0,
                    len(candidates) - 1,
                    min(FEATURE_SAMPLES_PER_BIN, len(candidates)),
                    dtype=int,
                )
            ]
            feature_vectors.append(
                np.median(np.asarray(pc_features[take], dtype=float), axis=0)
            )
            feature_times.append(float(np.median(np.asarray(times[take])) / fs))
        available = [i for i, value in enumerate(feature_vectors) if value is not None]
        consecutive_cosines = [
            cosine(feature_vectors[first], feature_vectors[second])
            for first, second in zip(available[:-1], available[1:])
        ]
        first_last_cosine = (
            cosine(feature_vectors[available[0]], feature_vectors[available[-1]])
            if len(available) >= 2
            else np.nan
        )
        coarse_depth = []
        coarse_amp = []
        coarse_time = []
        for feature_bin in range(FEATURE_BIN_COUNT):
            keep = (seconds >= feature_edges[feature_bin]) & (
                seconds < feature_edges[feature_bin + 1]
            )
            if np.sum(keep) < 5:
                coarse_depth.append(np.nan)
                coarse_amp.append(np.nan)
                coarse_time.append(np.nan)
            else:
                coarse_depth.append(float(np.median(unit_depth[keep])))
                coarse_amp.append(float(np.median(unit_amp[keep])))
                coarse_time.append(float(np.median(seconds[keep])))
        coarse_depth_array = np.asarray(coarse_depth)
        coarse_amp_array = np.asarray(coarse_amp)
        coarse_time_array = np.asarray(coarse_time)
        valid_amp = coarse_amp_array[np.isfinite(coarse_amp_array)]
        unit_rows.append(
            {
                "unit_id": int(unit),
                "ks_good": str(labels[label_column].get(int(unit), "")).lower() == "good",
                "contamination_pct": float(contamination[contamination_column].get(int(unit), np.nan)),
                "spike_count": len(indices),
                "mean_rate_hz": len(indices) / duration_s,
                "presence_fraction_300s": float(np.mean(bin_counts > 0)),
                "rate_cv_300s": float(np.std(rates) / np.mean(rates)) if np.mean(rates) else np.nan,
                "first_spike_s": float(seconds.min()),
                "last_spike_s": float(seconds.max()),
                "lifetime_s": float(seconds.max() - seconds.min()),
                "template_depth_um": float(template_depths[int(unit)]),
                "median_spike_depth_um": float(np.median(unit_depth)),
                "depth_excursion_p95_p5_um": float(np.quantile(unit_depth, 0.95) - np.quantile(unit_depth, 0.05)),
                "coarse_depth_slope_um_per_hour": linear_slope_per_hour(coarse_time_array, coarse_depth_array),
                "median_amplitude": float(np.median(unit_amp)),
                "coarse_amplitude_cv": float(np.std(valid_amp) / np.mean(valid_amp)) if len(valid_amp) and np.mean(valid_amp) else np.nan,
                "late_early_amplitude_ratio": float(valid_amp[-1] / valid_amp[0]) if len(valid_amp) >= 2 and valid_amp[0] else np.nan,
                "edge_spike_fraction": float(np.mean(unit_edge)),
                "dominant_detection_template": dominant_template,
                "dominant_detection_template_fraction": dominant_fraction,
                "feature_bins_available": len(available),
                "median_consecutive_pc_cosine": float(np.nanmedian(consecutive_cosines)) if consecutive_cosines else np.nan,
                "minimum_consecutive_pc_cosine": float(np.nanmin(consecutive_cosines)) if consecutive_cosines else np.nan,
                "first_last_pc_cosine": first_last_cosine,
            }
        )
    unit_metrics = pd.DataFrame(unit_rows)

    pair_rows = []
    for first_index, first_unit in enumerate(unit_ids):
        for second_unit in unit_ids[first_index + 1 :]:
            depth_difference = abs(
                float(template_depths[int(first_unit)])
                - float(template_depths[int(second_unit)])
            )
            template_similarity = float(similarity[int(first_unit), int(second_unit)])
            if depth_difference > PAIR_DEPTH_UM and template_similarity < 0.8:
                continue
            first_times = times_by_unit[int(first_unit)]
            second_times = times_by_unit[int(second_unit)]
            matches = count_one_to_one_matches(first_times, second_times, tolerance)
            if matches == 0:
                continue
            expected = (
                2
                * COINCIDENCE_MS
                / 1000.0
                * len(first_times)
                * len(second_times)
                / duration_s
            )
            pair_rows.append(
                {
                    "first_unit": int(first_unit),
                    "second_unit": int(second_unit),
                    "first_ks_good": bool(unit_metrics.set_index("unit_id").loc[int(first_unit), "ks_good"]),
                    "second_ks_good": bool(unit_metrics.set_index("unit_id").loc[int(second_unit), "ks_good"]),
                    "depth_difference_um": depth_difference,
                    "template_similarity": template_similarity,
                    "matched_spikes": matches,
                    "expected_chance_matches": expected,
                    "observed_expected_ratio": matches / max(expected, 1e-12),
                    "smaller_unit_match_fraction": matches / min(len(first_times), len(second_times)),
                    "jaccard_agreement": matches / (len(first_times) + len(second_times) - matches),
                }
            )
    pair_metrics = pd.DataFrame(pair_rows)
    if len(pair_metrics):
        pair_metrics["excess_coincident_pair"] = (
            (pair_metrics.matched_spikes >= 50)
            & (pair_metrics.observed_expected_ratio >= 3.0)
            & (pair_metrics.smaller_unit_match_fraction >= 0.05)
        )
        pair_metrics["high_priority_pair"] = (
            pair_metrics.excess_coincident_pair
            & (pair_metrics.matched_spikes >= 100)
            & (pair_metrics.smaller_unit_match_fraction >= 0.5)
            & (pair_metrics.jaccard_agreement >= 0.05)
            & (pair_metrics.depth_difference_um <= 40.0)
        )
    else:
        pair_metrics["excess_coincident_pair"] = pd.Series(dtype=bool)
        pair_metrics["high_priority_pair"] = pd.Series(dtype=bool)

    output_dir.mkdir(parents=True, exist_ok=True)
    time_metrics.to_csv(output_dir / "time_bin_metrics.csv", index=False)
    time_depth_metrics.to_csv(output_dir / "time_depth_metrics.csv", index=False)
    unit_metrics.to_csv(output_dir / "unit_continuity_metrics.csv", index=False)
    pair_metrics.to_csv(output_dir / "near_coincident_unit_pairs.csv", index=False)
    edge_contribution = (
        unit_metrics.assign(edge_spikes=unit_metrics.spike_count * unit_metrics.edge_spike_fraction)
        .sort_values("edge_spikes", ascending=False)
    )
    total_edge_spikes = float(edge_contribution.edge_spikes.sum())
    correlations = {}
    for motion_field in (
        "rigid_excursion_um",
        "median_nonrigid_spread_um",
        "p99_abs_rigid_step_um",
    ):
        for outcome in (
            "spikes_per_s",
            "edge_spike_fraction",
            "coincidence_excess",
            "median_amplitude",
        ):
            correlations[f"{outcome}_vs_{motion_field}"] = float(
                time_metrics[[outcome, motion_field]].corr(method="spearman").iloc[0, 1]
            )
    summary = {
        "sorter": str(sorter),
        "duration_s": duration_s,
        "n_spikes": int(len(times)),
        "n_units": int(len(unit_ids)),
        "n_ks_good": int(unit_metrics.ks_good.sum()),
        "time_bin_s": time_bin_s,
        "depth_range_um": [depth_min, depth_max],
        "overall_edge_spike_fraction": float(np.mean(edge)),
        "edge_spike_fraction_top_5_units": float(edge_contribution.head(5).edge_spikes.sum() / total_edge_spikes) if total_edge_spikes else np.nan,
        "units_edge_dominated_gt50pct": int((unit_metrics.edge_spike_fraction > 0.5).sum()),
        "maximum_time_bin_edge_fraction": float(time_metrics.edge_spike_fraction.max()),
        "maximum_time_bin_coincidence_excess": float(time_metrics.coincidence_excess.max()),
        "median_time_bin_coincidence_excess": float(time_metrics.coincidence_excess.median()),
        "excess_coincident_pairs_broad_screen": int(pair_metrics.excess_coincident_pair.sum()),
        "high_priority_pairs_for_ccg_template_residual_review": int(pair_metrics.high_priority_pair.sum()),
        "high_priority_pairs_with_any_ks_good": int((pair_metrics.high_priority_pair & (pair_metrics.first_ks_good | pair_metrics.second_ks_good)).sum()),
        "median_unit_depth_excursion_um": float(unit_metrics.depth_excursion_p95_p5_um.median()),
        "p90_unit_depth_excursion_um": float(unit_metrics.depth_excursion_p95_p5_um.quantile(0.9)),
        "median_first_last_pc_cosine": float(unit_metrics.first_last_pc_cosine.median()),
        "units_first_last_pc_cosine_lt_0_8": int((unit_metrics.first_last_pc_cosine < 0.8).sum()),
        "median_late_early_amplitude_ratio": float(unit_metrics.late_early_amplitude_ratio.median()),
        "units_late_early_amplitude_ratio_outside_0_5_2": int(((unit_metrics.late_early_amplitude_ratio < 0.5) | (unit_metrics.late_early_amplitude_ratio > 2.0)).sum()),
        "motion_spearman_correlations": correlations,
        "residual_status": "not_computed; no residual binary was saved; reconstruct only for prioritized units/events",
        "interpretation_guardrail": "PC-feature, depth and amplitude continuity are diagnostic and do not establish biological unit identity. Excess coincidence does not establish duplicate peeling; high-priority pairs require CCG, template, refractory, waveform and residual review.",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    args = parse_args()
    summary = run_audit(args.sorter, args.motion_dir, args.output_dir, args.time_bin_s)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
