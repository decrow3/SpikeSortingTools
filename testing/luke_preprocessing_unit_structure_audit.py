"""Check whether single-preprocessing creates stable units or fragments.

Compares the independent 240 s shared-window sorts using current upstream
conditioning versus saturation/bad-channel handling followed by Kilosort's
single high-pass/CAR pass.  The analysis is intentionally within-sort: unit
presence, firing stability, nearby template similarity, and temporal overlap
of similar-template pairs do not require questionable cross-condition cluster
identity matching.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(
    "/mnt/NPX/Luke/20250804/dredge_pipeline_results_"
    "Luke0804_V2V1_g0_imec1/motion_candidate_replication/shared_template"
)
OUT = Path("testing/outputs/luke_motion_candidate_results")
CONDITIONS = {
    "Current conditioning": ROOT / "sorts/no_external_correction/sorter_output",
    "Single Kilosort preprocessing": ROOT / "sorts/single_ks_preprocessing/sorter_output",
}
DURATION_S = 240.0
BIN_S = 10.0


def unit_bin_counts(
    times: np.ndarray, clusters: np.ndarray, unit_ids: np.ndarray, fs: float
) -> np.ndarray:
    n_bins = int(np.ceil(DURATION_S / BIN_S))
    result = np.zeros((len(unit_ids), n_bins), dtype=np.int64)
    lookup = {int(unit): i for i, unit in enumerate(unit_ids)}
    bins = np.minimum((times / fs / BIN_S).astype(int), n_bins - 1)
    for cluster, bin_index in zip(clusters, bins):
        if int(cluster) in lookup:
            result[lookup[int(cluster)], bin_index] += 1
    return result


def template_peak_depths(templates: np.ndarray, positions: np.ndarray) -> np.ndarray:
    peak_channels = np.argmax(np.max(np.abs(templates), axis=1), axis=1)
    return positions[peak_channels, 1]


def unit_metrics(condition: str, sorter: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    times = np.load(sorter / "spike_times.npy").reshape(-1).astype(np.int64)
    clusters = np.load(sorter / "spike_clusters.npy").reshape(-1)
    positions = np.load(sorter / "channel_positions.npy").astype(float)
    templates = np.load(sorter / "templates.npy").astype(float)
    similarity = np.load(sorter / "similar_templates.npy").astype(float)
    labels = pd.read_csv(sorter / "cluster_KSLabel.tsv", sep="\t").set_index("cluster_id")
    fs = float(np.load(sorter / "ops.npy", allow_pickle=True).item()["fs"])
    valid = (times >= 0) & (times < int(round(DURATION_S * fs)))
    times, clusters = times[valid], clusters[valid]
    unit_ids = np.arange(templates.shape[0], dtype=int)
    counts = unit_bin_counts(times, clusters, unit_ids, fs)
    totals = counts.sum(axis=1)
    rates = counts / BIN_S
    mean_rate = rates.mean(axis=1)
    rate_cv = rates.std(axis=1) / np.maximum(mean_rate, np.finfo(float).eps)
    presence = np.mean(counts > 0, axis=1)
    peak_depth = template_peak_depths(templates, positions)
    max_pos = templates.max(axis=(1, 2))
    max_neg = np.abs(templates.min(axis=(1, 2)))
    np.fill_diagonal(similarity, -np.inf)
    nearby = np.abs(peak_depth[:, None] - peak_depth[None, :]) <= 100.0
    nearest_similarity = np.where(nearby, similarity, -np.inf).max(axis=1)

    units = pd.DataFrame(
        {
            "condition": condition,
            "unit_id": unit_ids,
            "ks_good": [str(labels.KSLabel.get(unit, "")).lower() == "good" for unit in unit_ids],
            "spike_count": totals,
            "mean_rate_hz": totals / DURATION_S,
            "presence_ratio_10s": presence,
            "rate_cv_10s": rate_cv,
            "peak_depth_um": peak_depth,
            "positive_dominant_template": max_pos > max_neg,
            "max_nearby_template_similarity": nearest_similarity,
        }
    )

    pair_rows = []
    for first in range(len(unit_ids)):
        for second in range(first + 1, len(unit_ids)):
            if abs(peak_depth[first] - peak_depth[second]) > 100.0:
                continue
            sim = similarity[first, second]
            if sim < 0.8:
                continue
            active_first = counts[first] > 0
            active_second = counts[second] > 0
            union = active_first | active_second
            correlation = np.corrcoef(rates[first], rates[second])[0, 1]
            pair_rows.append(
                {
                    "condition": condition,
                    "unit_first": first,
                    "unit_second": second,
                    "template_similarity": sim,
                    "depth_difference_um": abs(peak_depth[first] - peak_depth[second]),
                    "both_good": bool(units.loc[first, "ks_good"] and units.loc[second, "ks_good"]),
                    "active_bin_jaccard": float(np.sum(active_first & active_second) / np.sum(union))
                    if np.any(union)
                    else np.nan,
                    "rate_correlation": float(correlation) if np.isfinite(correlation) else np.nan,
                    "combined_presence_ratio": float(np.mean(union)),
                }
            )
    return units, pd.DataFrame(pair_rows)


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
        root_first, root_second = find(first), find(second)
        if root_first != root_second:
            parent[root_second] = root_first
    return len({find(unit) for unit in parent})


def summarize(units: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for condition, group in units.groupby("condition", sort=False):
        pair_group = pairs[pairs.condition.eq(condition)] if len(pairs) else pairs
        for population, mask in {
            "all": np.ones(len(group), dtype=bool),
            "KS-good": group.ks_good.to_numpy(bool),
        }.items():
            subset = group.loc[mask]
            selected_pairs = pair_group if population == "all" else pair_group[pair_group.both_good]
            component_count = graph_component_count(subset.unit_id.to_numpy(), selected_pairs)
            rows.append(
                {
                    "condition": condition,
                    "population": population,
                    "n_units": len(subset),
                    "median_rate_hz": float(subset.mean_rate_hz.median()),
                    "median_presence_ratio_10s": float(subset.presence_ratio_10s.median()),
                    "median_rate_cv_10s": float(subset.rate_cv_10s.median()),
                    "positive_dominant_fraction": float(subset.positive_dominant_template.mean()),
                    "median_max_template_similarity": float(
                        subset.max_nearby_template_similarity.median()
                    ),
                    "nearby_similar_pairs": int(len(selected_pairs)),
                    "nearby_similar_pairs_per_unit": (
                        len(selected_pairs) / max(len(subset), 1)
                    ),
                    "similarity_graph_components": component_count,
                    "redundant_units_in_similarity_graph": len(subset) - component_count,
                    "median_similar_pair_active_jaccard": float(
                        selected_pairs.active_bin_jaccard.median()
                    ),
                }
            )
    return pd.DataFrame(rows)


def plot(units: pd.DataFrame, pairs: pd.DataFrame, output: Path) -> None:
    colors = {"Current conditioning": "#dd8452", "Single Kilosort preprocessing": "#55a868"}
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for ax, metric, title in [
        (axes[0, 0], "mean_rate_hz", "Unit firing rate"),
        (axes[0, 1], "presence_ratio_10s", "10 s presence ratio"),
        (axes[1, 0], "rate_cv_10s", "10 s firing-rate CV"),
    ]:
        data = [units.loc[units.condition.eq(c) & units.ks_good, metric] for c in colors]
        bp = ax.boxplot(data, labels=["Current", "Single"], patch_artist=True, showfliers=False)
        for patch, color in zip(bp["boxes"], colors.values()):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)
        ax.set_title(f"KS-good {title.lower()}")
        ax.grid(axis="y", alpha=0.2)
    ax = axes[1, 1]
    for condition, color in colors.items():
        values = pairs.loc[pairs.condition.eq(condition), "active_bin_jaccard"]
        ax.hist(values, bins=np.linspace(0, 1, 16), histtype="step", linewidth=2, label=condition, color=color)
    ax.set_title("Nearby similar-template pair activity overlap")
    ax.set_xlabel("Active-bin Jaccard")
    ax.set_ylabel("Pairs")
    ax.legend(fontsize=8)
    fig.suptitle("Luke shared-window unit-structure audit", fontsize=14)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    unit_frames = []
    pair_frames = []
    for condition, sorter in CONDITIONS.items():
        units, pairs = unit_metrics(condition, sorter)
        unit_frames.append(units)
        pair_frames.append(pairs)
    units = pd.concat(unit_frames, ignore_index=True)
    pairs = pd.concat(pair_frames, ignore_index=True)
    summary = summarize(units, pairs)
    units.to_csv(OUT / "luke_preprocessing_unit_metrics.csv", index=False)
    pairs.to_csv(OUT / "luke_preprocessing_similar_template_pairs.csv", index=False)
    summary.to_csv(OUT / "luke_preprocessing_unit_structure_summary.csv", index=False)
    plot(units, pairs, OUT / "luke_preprocessing_unit_structure_audit.png")


if __name__ == "__main__":
    main()
