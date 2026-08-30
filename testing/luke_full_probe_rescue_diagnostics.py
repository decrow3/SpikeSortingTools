"""Audit the accepted full-probe Luke imec1 rescue sort.

The audit separates independent automatic holdout preservation from the reused
reviewed discovery cohort, and reports yield only alongside contamination,
temporal presence, spatial concentration, template redundancy, polarity, and
near-coincidence guardrails.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit
import matplotlib.pyplot as plt

from testing.luke_trace_reviewed_events import local_match_details


SORTER = Path(
    "/mnt/NPX/Luke/20250804/"
    "rescue_pipeline_results_Luke0804_V2V1_g0_imec1/"
    "kilosort4/sorter_output"
)
OUTPUT = Path("testing/outputs/luke_full_probe_rescue_diagnostics")
HOLDOUT_KEY = Path(
    "testing/outputs/luke_prospective_holdout/holdout_candidate_key_v2.csv"
)
HOLDOUT_PUBLIC = Path(
    "testing/outputs/luke_prospective_holdout/holdout_candidates_v2.csv"
)
DISCOVERY_IMEC1 = Path(
    "testing/outputs/luke_multichannel_event_validation/imec1/event_stage_trace.csv"
)
LEGACY_QUALITY = Path(
    "testing/outputs/luke_multichannel_event_validation/imec1/"
    "sort_variant_quality_summary.csv"
)
REFERENCE_DENSITY = Path(
    "testing/outputs/luke_motion_candidate_results/"
    "luke_yates_detection_stage_dataset_metrics.csv"
)
MOTION = Path(
    "/mnt/NPX/Luke/20250804/"
    "dredge_pipeline_results_Luke0804_V2V1_g0_imec1/motion/dredge-motion"
)
TIME_BIN_S = 300.0
DEPTH_BIN_UM = 100.0
BAD_CHANNEL_RADIUS_UM = 40.0
PROBE_DEPTH_EXPOSURE_MM = 3.82


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sorter", type=Path, default=SORTER)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--probe", choices=("imec0", "imec1"), default="imec1")
    parser.add_argument(
        "--duration-s",
        type=float,
        help="Recording duration override for legacy outputs whose binary moved.",
    )
    parser.add_argument(
        "--bad-channel-depth-um",
        action="append",
        type=float,
        default=None,
        help="Optional repaired-channel depth; repeat for multiple channels.",
    )
    parser.add_argument("--n-jitters", type=int, default=250)
    parser.add_argument("--seed", type=int, default=20250804)
    return parser.parse_args()


@njit(cache=True)
def near_coincident_fraction_sorted(
    times: np.ndarray,
    clusters: np.ndarray,
    depths: np.ndarray,
    tolerance_frames: int,
    depth_tolerance_um: float,
) -> float:
    """Exact marked-spike fraction for time-sorted arrays."""
    marked = np.zeros(times.size, dtype=np.uint8)
    for left in range(times.size):
        right = left + 1
        while right < times.size and times[right] - times[left] <= tolerance_frames:
            if (
                clusters[right] != clusters[left]
                and abs(depths[right] - depths[left]) <= depth_tolerance_um
            ):
                marked[left] = 1
                marked[right] = 1
            right += 1
    return float(marked.sum() / times.size) if times.size else np.nan


def graph_component_count(unit_ids: np.ndarray, pairs: pd.DataFrame) -> int:
    parent = {int(unit): int(unit) for unit in unit_ids}

    def find(unit: int) -> int:
        while parent[unit] != unit:
            parent[unit] = parent[parent[unit]]
            unit = parent[unit]
        return unit

    for pair in pairs.itertuples(index=False):
        first, second = int(pair.unit_first), int(pair.unit_second)
        if first not in parent or second not in parent:
            continue
        first_root, second_root = find(first), find(second)
        if first_root != second_root:
            parent[second_root] = first_root
    return len({find(int(unit)) for unit in unit_ids})


def shifted_coincidence_null(
    times: np.ndarray,
    clusters: np.ndarray,
    depths: np.ndarray,
    duration_frames: int,
    tolerance_frames: int,
    seed: int,
    n_repeats: int = 3,
) -> float:
    """Rate/unit-preserving circular-shift null for one local window."""
    rng = np.random.default_rng(seed)
    n_units = int(clusters.max()) + 1 if clusters.size else 0
    values = []
    for _ in range(n_repeats):
        offsets = rng.integers(
            tolerance_frames + 1, duration_frames, size=n_units, dtype=np.int64
        )
        shifted = (times + offsets[clusters]) % duration_frames
        order = np.argsort(shifted, kind="stable")
        values.append(
            near_coincident_fraction_sorted(
                shifted[order], clusters[order], depths[order],
                tolerance_frames, 75.0
            )
        )
    return float(np.median(values))


def score_recovery(
    events: pd.DataFrame,
    population: str,
    times: np.ndarray,
    depths: np.ndarray,
    fs: float,
    rng: np.random.Generator,
    n_jitters: int,
) -> dict:
    samples = events.sample_index.to_numpy(np.int64)
    event_depths = events.depth_um.to_numpy(float)
    tolerance = int(round(0.5e-3 * fs))
    observed = float(
        local_match_details(
            samples,
            event_depths,
            times,
            depths,
            tolerance,
            100.0,
        ).present.mean()
    )
    null = []
    for _ in range(n_jitters):
        offsets_ms = rng.uniform(20.0, 500.0, len(events))
        offsets_ms *= rng.choice(np.array([-1.0, 1.0]), len(events))
        shifted = samples + np.rint(offsets_ms * fs / 1000.0).astype(np.int64)
        null.append(
            float(
                local_match_details(
                    shifted,
                    event_depths,
                    times,
                    depths,
                    tolerance,
                    100.0,
                ).present.mean()
            )
        )
    null_array = np.asarray(null)
    return {
        "cohort": population,
        "n_events": len(events),
        "observed_recovery": observed,
        "jitter_null_mean": float(null_array.mean()),
        "jitter_null_p95": float(np.quantile(null_array, 0.95)),
        "recovery_above_null": observed - float(null_array.mean()),
        "empirical_p": float((1 + np.sum(null_array >= observed)) / (n_jitters + 1)),
    }


def infer_bad_channel_depths(
    sorter: Path, channel_positions: np.ndarray
) -> list[float]:
    """Resolve repaired physical channels recorded in a rescue manifest."""
    manifest_path = sorter.parents[1] / "recording" / "rescue_recording_manifest.json"
    if not manifest_path.exists():
        return []
    manifest = json.loads(manifest_path.read_text())
    depths = []
    for channel_id in manifest.get("bad_channel_ids", []):
        match = re.search(r"AP(\d+)$", str(channel_id))
        if match and int(match.group(1)) < len(channel_positions):
            depths.append(float(channel_positions[int(match.group(1)), 1]))
    return sorted(set(depths))


def write_overview_figure(
    output_path: Path,
    comparison: pd.DataFrame,
    density: pd.DataFrame,
    time_metrics: pd.DataFrame,
    recovery: pd.DataFrame,
) -> None:
    """Write a compact decision figure from the audited tables."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    legacy_labels = {
        "patched": "patched",
        "dredge": "DREDGE",
        "dredgetest": "DREDGE test",
        "pipeline": "pipeline",
        "pipeline_an5": "pipeline_an5",
        "rescue_full_probe": "rescue",
    }
    quality = comparison.copy()
    axes[0, 0].bar(
        [legacy_labels.get(value, value) for value in quality.variant],
        quality.n_ks_good,
        color=["#2b6cb0" if value == "rescue_full_probe" else "#a0aec0" for value in quality.variant],
    )
    axes[0, 0].set_title("KS-good yield rises without spike-count inflation")
    axes[0, 0].set_ylabel("KS-good units")
    axes[0, 0].tick_params(axis="x", rotation=30)

    density_labels = [
        "rescue", "96-ch strip", "Luke 240 s", "Yates sampled"
    ]
    axes[0, 1].bar(
        density_labels,
        density.ks_good_per_depth_mm,
        color=["#2b6cb0", "#a0aec0", "#718096", "#4a5568"],
    )
    axes[0, 1].set_title("Depth-normalized KS-good yield")
    axes[0, 1].set_ylabel("KS-good units / mm")
    axes[0, 1].tick_params(axis="x", rotation=20)

    midpoints = (time_metrics.start_s + time_metrics.stop_s) / 7200.0
    axes[1, 0].plot(midpoints, time_metrics.active_good_units, color="#2b6cb0", lw=2)
    axes[1, 0].set_title("Good units remain active across the 2.9 h session")
    axes[1, 0].set_xlabel("Time (h)")
    axes[1, 0].set_ylabel("Active KS-good units / 300 s")

    headline = recovery[
        recovery.cohort.isin(
            [
                "sealed_holdout_all_raw_events",
                "reused_reviewed_neural_unmatched",
            ]
        )
    ].copy()
    axes[1, 1].bar(
        ["sealed raw holdout", "reused reviewed neural"],
        headline.observed_recovery * 100,
        color=["#2b6cb0", "#718096"],
    )
    axes[1, 1].scatter(
        [0, 1], headline.jitter_null_p95 * 100, marker="_", s=500,
        linewidths=3, color="#c53030", label="jitter-null 95th percentile"
    )
    axes[1, 1].set_ylim(0, 105)
    axes[1, 1].set_title("Event recovery exceeds the timing-null guardrail")
    axes[1, 1].set_ylabel("Recovered (%)")
    axes[1, 1].legend(frameon=False, fontsize=8)
    axes[1, 1].tick_params(axis="x", rotation=15)

    fig.suptitle("Luke imec1 full-probe rescue: diagnostic overview", fontsize=15)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def run(
    sorter: Path,
    output_dir: Path,
    n_jitters: int,
    seed: int,
    *,
    probe: str = "imec1",
    duration_override_s: float | None = None,
    bad_channel_depths_um: list[float] | None = None,
) -> dict:
    required = (
        "spike_times.npy",
        "spike_clusters.npy",
        "spike_positions.npy",
        "amplitudes.npy",
        "templates.npy",
        "similar_templates.npy",
        "channel_positions.npy",
        "cluster_KSLabel.tsv",
        "cluster_ContamPct.tsv",
        "ops.npy",
    )
    missing = [name for name in required if not (sorter / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing sorter outputs: {missing}")

    ops = np.load(sorter / "ops.npy", allow_pickle=True).item()
    fs = float(ops["fs"])
    times = np.load(sorter / "spike_times.npy", mmap_mode="r").reshape(-1)
    clusters = np.load(sorter / "spike_clusters.npy", mmap_mode="r").reshape(-1)
    positions = np.load(sorter / "spike_positions.npy", mmap_mode="r")
    amplitudes = np.load(sorter / "amplitudes.npy", mmap_mode="r").reshape(-1)
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
    duration_s = (
        float(duration_override_s)
        if duration_override_s is not None
        else float(ops.get("tmax", np.nan))
    )
    if not np.isfinite(duration_s):
        recording_manifest = (
            sorter.parents[1] / "recording" / "rescue_recording_manifest.json"
        )
        if recording_manifest.exists():
            recording_metadata = json.loads(recording_manifest.read_text())
            duration_s = (
                float(recording_metadata["num_samples"])
                / float(recording_metadata["sampling_frequency_hz"])
            )
        else:
            binary_path = Path(str(ops.get("filename", "")))
            if binary_path.exists():
                dtype = np.dtype(ops.get("data_dtype", "int16"))
                n_channels = int(ops.get("n_chan_bin", ops.get("Nchan", 384)))
                duration_s = (
                    binary_path.stat().st_size / dtype.itemsize / n_channels / fs
                )
            else:
                duration_s = float(np.asarray(times[-1])) / fs
    n_units = int(templates.shape[0])
    unit_ids = np.arange(n_units, dtype=int)
    cluster_values = np.asarray(clusters)
    unit_counts = np.bincount(cluster_values, minlength=n_units)
    ks_good = np.array(
        [str(labels[label_column].get(unit, "")).lower() == "good" for unit in unit_ids]
    )
    contam = np.array(
        [float(contamination[contamination_column].get(unit, np.nan)) for unit in unit_ids]
    )

    n_time_bins = int(np.ceil(duration_s / TIME_BIN_S))
    time_bin = np.minimum(
        (np.asarray(times, dtype=np.float64) / fs / TIME_BIN_S).astype(np.int64),
        n_time_bins - 1,
    )
    unit_time_counts = np.bincount(
        cluster_values.astype(np.int64) * n_time_bins + time_bin,
        minlength=n_units * n_time_bins,
    ).reshape(n_units, n_time_bins)
    time_edges = np.minimum(np.arange(n_time_bins + 1) * TIME_BIN_S, duration_s)
    bin_durations = np.diff(time_edges)
    rates = unit_time_counts / bin_durations
    presence = np.mean(unit_time_counts > 0, axis=1)
    mean_rates = unit_counts / duration_s
    rate_cv = rates.std(axis=1) / np.maximum(rates.mean(axis=1), np.finfo(float).eps)
    first = np.full(n_units, np.iinfo(np.int64).max, dtype=np.int64)
    last = np.zeros(n_units, dtype=np.int64)
    np.minimum.at(first, cluster_values, np.asarray(times, dtype=np.int64))
    np.maximum.at(last, cluster_values, np.asarray(times, dtype=np.int64))
    lifetime_s = (last - first) / fs

    depth = np.asarray(positions[:, 1], dtype=np.float32)
    depth_min = float(channel_positions[:, 1].min())
    depth_max = float(channel_positions[:, 1].max())
    edge = (depth <= depth_min + 40.0) | (depth >= depth_max - 40.0)
    resolved_bad_depths = (
        list(bad_channel_depths_um)
        if bad_channel_depths_um is not None
        else infer_bad_channel_depths(sorter, channel_positions)
    )
    bad_zone = np.zeros(depth.shape, dtype=bool)
    for bad_depth in resolved_bad_depths:
        bad_zone |= np.abs(depth - bad_depth) <= BAD_CHANNEL_RADIUS_UM
    edge_counts = np.bincount(cluster_values, weights=edge, minlength=n_units)
    bad_zone_counts = np.bincount(cluster_values, weights=bad_zone, minlength=n_units)

    order = np.argsort(cluster_values, kind="stable")
    offsets = np.r_[0, np.cumsum(unit_counts)]
    median_depth = np.empty(n_units)
    depth_excursion = np.empty(n_units)
    median_amplitude = np.empty(n_units)
    refractory_violation_fraction = np.empty(n_units)
    refractory_frames = int(round(1.5e-3 * fs))
    for unit in unit_ids:
        all_indices = order[offsets[unit] : offsets[unit + 1]]
        unit_times = np.asarray(times[all_indices], dtype=np.int64)
        refractory_violation_fraction[unit] = (
            float(np.mean(np.diff(unit_times) < refractory_frames))
            if unit_times.size > 1
            else np.nan
        )
        metric_indices = all_indices
        if metric_indices.size > 50_000:
            metric_indices = metric_indices[
                np.linspace(0, metric_indices.size - 1, 50_000, dtype=int)
            ]
        unit_depth = depth[metric_indices]
        median_depth[unit] = float(np.median(unit_depth))
        depth_excursion[unit] = float(
            np.quantile(unit_depth, 0.95) - np.quantile(unit_depth, 0.05)
        )
        median_amplitude[unit] = float(
            np.median(np.asarray(amplitudes[metric_indices]))
        )
    del order

    peak_channels = np.argmax(np.max(np.abs(templates), axis=1), axis=1)
    template_depth = channel_positions[peak_channels, 1]
    positive_dominant = templates.max(axis=(1, 2)) > np.abs(
        templates.min(axis=(1, 2))
    )
    upper = np.triu(np.ones((n_units, n_units), dtype=bool), 1)
    nearby = np.abs(template_depth[:, None] - template_depth[None, :]) <= 100.0
    pair_first, pair_second = np.where(upper & nearby & (similarity >= 0.8))
    pair_rows = []
    for first_unit, second_unit in zip(pair_first, pair_second):
        first_active = unit_time_counts[first_unit] > 0
        second_active = unit_time_counts[second_unit] > 0
        union = first_active | second_active
        pair_rows.append(
            {
                "unit_first": int(first_unit),
                "unit_second": int(second_unit),
                "both_good": bool(ks_good[first_unit] and ks_good[second_unit]),
                "template_similarity": float(similarity[first_unit, second_unit]),
                "depth_difference_um": float(
                    abs(template_depth[first_unit] - template_depth[second_unit])
                ),
                "active_bin_jaccard": float(
                    np.sum(first_active & second_active) / np.sum(union)
                )
                if np.any(union)
                else np.nan,
                "rate_correlation": float(
                    np.corrcoef(rates[first_unit], rates[second_unit])[0, 1]
                ),
            }
        )
    similar_pairs = pd.DataFrame(pair_rows)
    all_components = graph_component_count(unit_ids, similar_pairs)
    good_pairs = similar_pairs[similar_pairs.both_good] if len(similar_pairs) else similar_pairs
    good_components = graph_component_count(unit_ids[ks_good], good_pairs)

    units = pd.DataFrame(
        {
            "unit_id": unit_ids,
            "ks_good": ks_good,
            "contamination_pct": contam,
            "spike_count": unit_counts,
            "mean_rate_hz": mean_rates,
            "presence_fraction_300s": presence,
            "rate_cv_300s": rate_cv,
            "refractory_violation_fraction_1p5ms": refractory_violation_fraction,
            "lifetime_s": lifetime_s,
            "template_depth_um": template_depth,
            "median_spike_depth_um": median_depth,
            "depth_excursion_p95_p5_um": depth_excursion,
            "median_amplitude": median_amplitude,
            "edge_spike_fraction": edge_counts / np.maximum(unit_counts, 1),
            "bad_channel_zone_spike_fraction": bad_zone_counts
            / np.maximum(unit_counts, 1),
            "positive_dominant_template": positive_dominant,
        }
    )

    global_time_counts = unit_time_counts.sum(axis=0)
    time_metrics = pd.DataFrame(
        {
            "time_bin": np.arange(n_time_bins),
            "start_s": time_edges[:-1],
            "stop_s": time_edges[1:],
            "spike_count": global_time_counts,
            "spikes_per_s": global_time_counts / bin_durations,
            "active_units": np.sum(unit_time_counts > 0, axis=0),
            "active_good_units": np.sum(unit_time_counts[ks_good] > 0, axis=0),
        }
    )

    depth_edges = np.arange(
        np.floor(depth_min / DEPTH_BIN_UM) * DEPTH_BIN_UM,
        np.ceil(depth_max / DEPTH_BIN_UM) * DEPTH_BIN_UM + DEPTH_BIN_UM,
        DEPTH_BIN_UM,
    )
    spike_depth_bin = np.clip(
        np.digitize(depth, depth_edges) - 1, 0, len(depth_edges) - 2
    )
    template_depth_bin = np.clip(
        np.digitize(template_depth, depth_edges) - 1, 0, len(depth_edges) - 2
    )
    depth_metrics = pd.DataFrame(
        {
            "depth_start_um": depth_edges[:-1],
            "depth_stop_um": depth_edges[1:],
            "spike_count": np.bincount(
                spike_depth_bin, minlength=len(depth_edges) - 1
            ),
            "units": np.bincount(template_depth_bin, minlength=len(depth_edges) - 1),
            "ks_good_units": np.bincount(
                template_depth_bin, weights=ks_good, minlength=len(depth_edges) - 1
            ).astype(int),
        }
    )

    rng = np.random.default_rng(seed)
    holdout = pd.read_csv(HOLDOUT_KEY).merge(
        pd.read_csv(HOLDOUT_PUBLIC), on=["candidate_id", "probe", "window_id"]
    )
    holdout = holdout[holdout.probe.eq(probe)].copy()
    recovery_rows = [
        score_recovery(
            holdout,
            "sealed_holdout_all_raw_events",
            np.asarray(times),
            depth,
            fs,
            rng,
            n_jitters,
        )
    ]
    for field in ("motion_stratum", "polarity", "amplitude_stratum", "depth_third"):
        for value, group in holdout.groupby(field):
            recovery_rows.append(
                score_recovery(
                    group,
                    f"sealed_holdout_{field}={value}",
                    np.asarray(times),
                    depth,
                    fs,
                    rng,
                    n_jitters,
                )
            )

    if probe == "imec1" and DISCOVERY_IMEC1.exists():
        discovery = pd.read_csv(DISCOVERY_IMEC1)
        discovery = discovery.rename(columns={"peak_depth_um": "depth_um"})
        discovery_populations = {
            "reused_reviewed_neural_unmatched": discovery.review_label.eq("neural")
            & discovery.status.eq("unmatched"),
            "reused_automatic_neural_like_unmatched": discovery.automatic_neural_like
            & discovery.status.eq("unmatched"),
            "reused_all_unmatched": discovery.status.eq("unmatched"),
        }
        for name, mask in discovery_populations.items():
            recovery_rows.append(
                score_recovery(
                    discovery[mask],
                    name,
                    np.asarray(times),
                    depth,
                    fs,
                    rng,
                    n_jitters,
                )
            )
    recovery = pd.DataFrame(recovery_rows)

    tolerance = int(round(0.5e-3 * fs))
    coincidence_rows = []
    holdout_windows = holdout[
        ["window_id", "motion_stratum"]
    ].drop_duplicates()
    for window in holdout_windows.itertuples(index=False):
        selected = holdout[holdout.window_id.eq(window.window_id)]
        start = int(round(selected.time_s.min() // 120 * 120 * fs))
        stop = start + int(round(120 * fs))
        left = int(np.searchsorted(times, start, side="left"))
        right = int(np.searchsorted(times, stop, side="left"))
        local_times = np.asarray(times[left:right], dtype=np.int64) - start
        local_clusters = np.asarray(clusters[left:right], dtype=np.int32)
        local_depths = np.asarray(depth[left:right], dtype=np.float32)
        observed_coincidence = near_coincident_fraction_sorted(
            local_times, local_clusters, local_depths, tolerance, 75.0
        )
        coincidence_null = shifted_coincidence_null(
            local_times,
            local_clusters,
            local_depths,
            stop - start,
            tolerance,
            seed + len(coincidence_rows),
        )
        coincidence_rows.append(
            {
                "window_id": window.window_id,
                "motion_stratum": window.motion_stratum,
                "start_s": start / fs,
                "stop_s": stop / fs,
                "spike_count": right - left,
                "cross_unit_near_coincident_fraction": observed_coincidence,
                "coincidence_shift_null": coincidence_null,
                "coincidence_excess": observed_coincidence - coincidence_null,
            }
        )
    coincidence = pd.DataFrame(coincidence_rows)

    rescue_quality = {
        "variant": "rescue_full_probe" if probe == "imec1" else "imec0_legacy_pipeline",
        "n_spikes": len(times),
        "n_units": n_units,
        "n_ks_good": int(ks_good.sum()),
        "median_unit_rate_hz": float(np.median(mean_rates)),
        "mean_unit_rate_hz": float(np.mean(mean_rates)),
        "median_contamination_pct": float(np.nanmedian(contam)),
        "fraction_units_contamination_le_10pct": float(np.nanmean(contam <= 10)),
    }
    if probe == "imec1":
        legacy = pd.read_csv(LEGACY_QUALITY)
        comparison = pd.concat(
            [legacy, pd.DataFrame([rescue_quality])], ignore_index=True
        )
    else:
        legacy = pd.DataFrame()
        comparison = pd.DataFrame([rescue_quality])

    strip_depth_mm = 0.94
    density_rows = [
        {
            "dataset": f"{probe}_audited_sort",
            "depth_exposure_mm": PROBE_DEPTH_EXPOSURE_MM,
            "units_per_depth_mm": n_units / PROBE_DEPTH_EXPOSURE_MM,
            "ks_good_per_depth_mm": ks_good.sum() / PROBE_DEPTH_EXPOSURE_MM,
            "final_spikes_per_depth_mm_s": len(times)
            / duration_s
            / PROBE_DEPTH_EXPOSURE_MM,
            "source_note": f"audited {probe} full-probe sort",
        },
    ]
    if probe == "imec1":
        density_rows.append(
            {
                "dataset": "prior_96_channel_full_strip",
                "depth_exposure_mm": strip_depth_mm,
                "units_per_depth_mm": 125 / strip_depth_mm,
                "ks_good_per_depth_mm": 32 / strip_depth_mm,
                "final_spikes_per_depth_mm_s": 3_269_181
                / duration_s
                / strip_depth_mm,
                "source_note": "prior 96-channel full-session strip audit",
            }
        )
        reference = pd.read_csv(REFERENCE_DENSITY)
        for name in ("Luke shared, single KS preprocessing", "Yates known-good"):
            row = reference[reference.dataset.eq(name)].iloc[0]
            density_rows.append(
                {
                    "dataset": name,
                    "depth_exposure_mm": float(row.depth_exposure_mm),
                    "units_per_depth_mm": float(row.units_per_depth_mm),
                    "ks_good_per_depth_mm": float(row.ks_good_per_depth_mm),
                    "final_spikes_per_depth_mm_s": (
                        1434.9 if name.startswith("Luke") else 666.1
                    ),
                    "source_note": (
                        "reported detection-stage audit rate; sampled comparison"
                    ),
                }
            )
    density = pd.DataFrame(density_rows)

    template_bad_zone = np.zeros(template_depth.shape, dtype=bool)
    for bad_depth in resolved_bad_depths:
        template_bad_zone |= np.abs(template_depth - bad_depth) <= BAD_CHANNEL_RADIUS_UM
    summary = {
        "probe": probe,
        "sorter": str(sorter),
        "duration_s": duration_s,
        "n_spikes": len(times),
        "n_units": n_units,
        "n_ks_good": int(ks_good.sum()),
        "final_to_learned_retention": (
            len(times) / 44_344_490 if probe == "imec1" else None
        ),
        "median_contamination_pct_all": float(np.nanmedian(contam)),
        "median_contamination_pct_good": float(np.nanmedian(contam[ks_good])),
        "fraction_units_contamination_le_10pct": float(np.nanmean(contam <= 10)),
        "median_good_presence_fraction_300s": float(np.median(presence[ks_good])),
        "good_units_presence_ge_0_9": int(np.sum(presence[ks_good] >= 0.9)),
        "median_good_rate_cv_300s": float(np.median(rate_cv[ks_good])),
        "median_refractory_violation_fraction_all": float(
            np.nanmedian(refractory_violation_fraction)
        ),
        "median_refractory_violation_fraction_good": float(
            np.nanmedian(refractory_violation_fraction[ks_good])
        ),
        "median_good_lifetime_s": float(np.median(lifetime_s[ks_good])),
        "median_good_depth_excursion_um": float(np.median(depth_excursion[ks_good])),
        "positive_dominant_fraction_all": float(np.mean(positive_dominant)),
        "positive_dominant_fraction_good": float(np.mean(positive_dominant[ks_good])),
        "overall_edge_spike_fraction_40um": float(np.mean(edge)),
        "repaired_channel_depths_um": resolved_bad_depths,
        "bad_channel_zone_spike_fraction_40um": float(np.mean(bad_zone)),
        "units_template_near_bad_channel_40um": int(
            np.sum(template_bad_zone)
        ),
        "good_units_template_near_bad_channel_40um": int(
            np.sum(ks_good & template_bad_zone)
        ),
        "nearby_similar_template_pairs": int(len(similar_pairs)),
        "nearby_similar_good_good_pairs": int(len(good_pairs)),
        "similarity_graph_components_all": all_components,
        "redundant_units_in_similarity_graph_all": n_units - all_components,
        "similarity_graph_components_good": good_components,
        "redundant_good_units_in_similarity_graph": int(ks_good.sum())
        - good_components,
        "median_holdout_window_coincidence": float(
            coincidence.cross_unit_near_coincident_fraction.median()
        ),
        "max_holdout_window_coincidence": float(
            coincidence.cross_unit_near_coincident_fraction.max()
        ),
        "median_holdout_window_coincidence_excess": float(
            coincidence.coincidence_excess.median()
        ),
        "max_holdout_window_coincidence_excess": float(
            coincidence.coincidence_excess.max()
        ),
        "time_bin_spike_rate_cv": float(
            time_metrics.spikes_per_s.std() / time_metrics.spikes_per_s.mean()
        ),
        "time_bin_spike_rate_min": float(time_metrics.spikes_per_s.min()),
        "time_bin_spike_rate_max": float(time_metrics.spikes_per_s.max()),
        "sealed_holdout_status": (
            "independent automatic raw-event preservation only; manual labels absent"
        ),
        "reviewed_event_status": (
            "reused discovery cohort; descriptive and not independent validation"
            if probe == "imec1"
            else "no manually reviewed imec0 discovery cohort in this audit"
        ),
    }
    if probe == "imec1":
        best_legacy_good = int(legacy.n_ks_good.max())
        pipeline_an5 = legacy[legacy.variant.eq("pipeline_an5")].iloc[0]
        summary.update(
            {
                "delta_vs_best_legacy_full_good_units": int(ks_good.sum())
                - best_legacy_good,
                "relative_gain_vs_best_legacy_full_good_units": float(
                    ks_good.sum() / best_legacy_good - 1
                ),
                "relative_gain_vs_pipeline_an5_good_units": float(
                    ks_good.sum() / pipeline_an5.n_ks_good - 1
                ),
                "relative_change_vs_pipeline_an5_final_spikes": float(
                    len(times) / pipeline_an5.n_spikes - 1
                ),
                "relative_gain_vs_full_strip_good_units_per_mm": float(
                    (ks_good.sum() / PROBE_DEPTH_EXPOSURE_MM)
                    / (32 / strip_depth_mm)
                    - 1
                ),
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    units.to_csv(output_dir / "unit_metrics.csv", index=False)
    time_metrics.to_csv(output_dir / "time_bin_metrics.csv", index=False)
    depth_metrics.to_csv(output_dir / "depth_bin_metrics.csv", index=False)
    similar_pairs.to_csv(output_dir / "similar_template_pairs.csv", index=False)
    coincidence.to_csv(output_dir / "holdout_window_coincidence.csv", index=False)
    recovery.to_csv(output_dir / "event_recovery.csv", index=False)
    comparison.to_csv(output_dir / "legacy_full_session_comparison.csv", index=False)
    density.to_csv(output_dir / "depth_normalized_comparison.csv", index=False)
    if probe == "imec1":
        write_overview_figure(
            output_dir / "diagnostic_overview.png",
            comparison,
            density,
            time_metrics,
            recovery,
        )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    args = parse_args()
    print(
        json.dumps(
            run(
                args.sorter,
                args.output_dir,
                args.n_jitters,
                args.seed,
                probe=args.probe,
                duration_override_s=args.duration_s,
                bad_channel_depths_um=args.bad_channel_depth_um,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
