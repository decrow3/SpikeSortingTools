"""Cross-condition unit-family audit for Luke preprocessing candidates.

This complements the within-sort template-similarity audit.  It compares the
same 240 s voltage interval sorted with current conditioning versus a single
Kilosort high-pass/CAR pass.  Unit pairs are related by one-to-one spike-time
agreement, then a common graph rule collapses splits in either direction.

The result is a decision gate, not a claim of biological identity: agreement
is evaluated over one short window and task-locked coincidences can inflate
weak edges.  Threshold sensitivity and a Poisson coincidence expectation are
therefore retained in the outputs.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(
    "/mnt/NPX/Luke/20250804/dredge_pipeline_results_"
    "Luke0804_V2V1_g0_imec1/motion_candidate_replication/shared_template"
)
DEFAULT_OUTPUT = Path("testing/outputs/luke_preprocessing_family_continuity")
SORTERS = {
    "current": ROOT / "sorts/no_external_correction/sorter_output",
    "single_pass": ROOT / "sorts/single_ks_preprocessing/sorter_output",
}
DURATION_S = 240.0
TIME_TOLERANCE_MS = 0.5
DEPTH_TOLERANCE_UM = 100.0
PRIMARY_AGREEMENT = 0.20
SENSITIVITY_THRESHOLDS = (0.10, 0.20, 0.30, 0.50)
MIN_MATCHES = 20
MIN_OBSERVED_EXPECTED_RATIO = 3.0


@dataclass
class SortData:
    condition: str
    unit_ids: np.ndarray
    times_by_unit: dict[int, np.ndarray]
    spike_counts: np.ndarray
    depths_um: np.ndarray
    ks_good: np.ndarray
    presence_ratio_10s: np.ndarray
    rate_cv_10s: np.ndarray
    max_nearby_template_similarity: np.ndarray
    fs: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current-sorter", type=Path, default=SORTERS["current"])
    parser.add_argument("--single-sorter", type=Path, default=SORTERS["single_pass"])
    parser.add_argument("--duration-s", type=float, default=DURATION_S)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def count_one_to_one_matches(first: np.ndarray, second: np.ndarray, tolerance: int) -> int:
    """Greedy one-to-one coincidence count for sorted integer spike samples."""
    i = j = matches = 0
    while i < len(first) and j < len(second):
        delta = int(first[i]) - int(second[j])
        if abs(delta) <= tolerance:
            matches += 1
            i += 1
            j += 1
        elif delta < 0:
            i += 1
        else:
            j += 1
    return matches


def template_peak_depths(templates: np.ndarray, channel_positions: np.ndarray) -> np.ndarray:
    peak_channels = np.argmax(np.max(np.abs(templates), axis=1), axis=1)
    return channel_positions[peak_channels, 1].astype(float)


def load_sort(condition: str, sorter: Path, duration_s: float = DURATION_S) -> SortData:
    required = (
        "spike_times.npy",
        "spike_clusters.npy",
        "templates.npy",
        "similar_templates.npy",
        "channel_positions.npy",
        "cluster_KSLabel.tsv",
        "ops.npy",
    )
    missing = [name for name in required if not (sorter / name).exists()]
    if missing:
        raise FileNotFoundError(f"{condition} is missing sorter files: {missing}")

    ops = np.load(sorter / "ops.npy", allow_pickle=True).item()
    fs = float(ops["fs"])
    times = np.load(sorter / "spike_times.npy").reshape(-1).astype(np.int64)
    clusters = np.load(sorter / "spike_clusters.npy").reshape(-1).astype(int)
    templates = np.load(sorter / "templates.npy", mmap_mode="r")
    positions = np.load(sorter / "channel_positions.npy").astype(float)
    labels = pd.read_csv(sorter / "cluster_KSLabel.tsv", sep="\t").set_index(
        "cluster_id"
    )
    label_column = next(column for column in labels if column != "cluster_id")
    valid = (times >= 0) & (times < int(round(duration_s * fs)))
    times, clusters = times[valid], clusters[valid]
    unit_ids = np.arange(templates.shape[0], dtype=int)
    times_by_unit = {
        int(unit): np.sort(times[clusters == unit]) for unit in unit_ids
    }
    counts = np.asarray([len(times_by_unit[int(unit)]) for unit in unit_ids], dtype=int)
    if np.any(counts == 0):
        raise ValueError(f"{condition} contains templates with no valid spikes")
    ks_good = np.asarray(
        [
            str(labels[label_column].get(int(unit), "")).lower() == "good"
            for unit in unit_ids
        ],
        dtype=bool,
    )
    n_bins = int(np.ceil(duration_s / 10.0))
    bin_counts = np.zeros((len(unit_ids), n_bins), dtype=np.int64)
    for unit in unit_ids:
        unit_times = times_by_unit[int(unit)]
        bins = np.minimum((unit_times / fs / 10.0).astype(int), n_bins - 1)
        bin_counts[int(unit)] = np.bincount(bins, minlength=n_bins)
    bin_rates = bin_counts / 10.0
    mean_rates = bin_rates.mean(axis=1)
    rate_cv = bin_rates.std(axis=1) / np.maximum(mean_rates, np.finfo(float).eps)
    presence = np.mean(bin_counts > 0, axis=1)
    depths = template_peak_depths(templates, positions)
    similarity = np.load(sorter / "similar_templates.npy").astype(float)
    np.fill_diagonal(similarity, -np.inf)
    nearby = np.abs(depths[:, None] - depths[None, :]) <= DEPTH_TOLERANCE_UM
    max_nearby_similarity = np.where(nearby, similarity, -np.inf).max(axis=1)
    return SortData(
        condition=condition,
        unit_ids=unit_ids,
        times_by_unit=times_by_unit,
        spike_counts=counts,
        depths_um=depths,
        ks_good=ks_good,
        presence_ratio_10s=presence,
        rate_cv_10s=rate_cv,
        max_nearby_template_similarity=max_nearby_similarity,
        fs=fs,
    )


def pairwise_agreement(
    first: SortData, second: SortData, duration_s: float = DURATION_S
) -> pd.DataFrame:
    if not np.isclose(first.fs, second.fs, rtol=0, atol=1e-6):
        raise ValueError(f"Sampling-rate mismatch: {first.fs} versus {second.fs}")
    tolerance = int(round(TIME_TOLERANCE_MS * first.fs / 1000.0))
    rows = []
    for first_index, first_unit in enumerate(first.unit_ids):
        nearby = np.flatnonzero(
            np.abs(second.depths_um - first.depths_um[first_index])
            <= DEPTH_TOLERANCE_UM
        )
        first_times = first.times_by_unit[int(first_unit)]
        n_first = len(first_times)
        for second_index in nearby:
            second_unit = int(second.unit_ids[second_index])
            second_times = second.times_by_unit[second_unit]
            n_second = len(second_times)
            matches = count_one_to_one_matches(first_times, second_times, tolerance)
            if matches == 0:
                continue
            expected = (
                2.0
                * TIME_TOLERANCE_MS
                / 1000.0
                * n_first
                * n_second
                / duration_s
            )
            union = n_first + n_second - matches
            rows.append(
                {
                    "current_unit": int(first_unit),
                    "single_pass_unit": second_unit,
                    "current_ks_good": bool(first.ks_good[first_index]),
                    "single_pass_ks_good": bool(second.ks_good[second_index]),
                    "current_spikes": n_first,
                    "single_pass_spikes": n_second,
                    "current_depth_um": float(first.depths_um[first_index]),
                    "single_pass_depth_um": float(second.depths_um[second_index]),
                    "depth_difference_um": float(
                        abs(first.depths_um[first_index] - second.depths_um[second_index])
                    ),
                    "matched_spikes": matches,
                    "expected_chance_matches": float(expected),
                    "observed_expected_ratio": float(matches / max(expected, 1e-12)),
                    "current_recall": float(matches / n_first),
                    "single_pass_recall": float(matches / n_second),
                    "agreement": float(matches / union),
                }
            )
    return pd.DataFrame(rows)


def qualifying_edges(pairs: pd.DataFrame, threshold: float) -> pd.DataFrame:
    return pairs[
        (pairs.agreement >= threshold)
        & (pairs.matched_spikes >= MIN_MATCHES)
        & (pairs.observed_expected_ratio >= MIN_OBSERVED_EXPECTED_RATIO)
    ].copy()


class UnionFind:
    def __init__(self, nodes: list[tuple[str, int]]):
        self.parent = {node: node for node in nodes}

    def find(self, node: tuple[str, int]) -> tuple[str, int]:
        while self.parent[node] != node:
            self.parent[node] = self.parent[self.parent[node]]
            node = self.parent[node]
        return node

    def union(self, first: tuple[str, int], second: tuple[str, int]) -> None:
        first_root, second_root = self.find(first), self.find(second)
        if first_root != second_root:
            self.parent[second_root] = first_root


def family_assignments(
    first: SortData, second: SortData, pairs: pd.DataFrame, threshold: float
) -> pd.DataFrame:
    nodes = [(first.condition, int(unit)) for unit in first.unit_ids] + [
        (second.condition, int(unit)) for unit in second.unit_ids
    ]
    graph = UnionFind(nodes)
    for edge in qualifying_edges(pairs, threshold).itertuples(index=False):
        graph.union(
            (first.condition, int(edge.current_unit)),
            (second.condition, int(edge.single_pass_unit)),
        )
    roots = {node: graph.find(node) for node in nodes}
    ordered_roots = {root: index for index, root in enumerate(sorted(set(roots.values())))}
    rows = []
    for data in (first, second):
        for index, unit in enumerate(data.unit_ids):
            node = (data.condition, int(unit))
            rows.append(
                {
                    "condition": data.condition,
                    "unit_id": int(unit),
                    "ks_good": bool(data.ks_good[index]),
                    "spike_count": int(data.spike_counts[index]),
                    "depth_um": float(data.depths_um[index]),
                    "family_id": ordered_roots[roots[node]],
                }
            )
    result = pd.DataFrame(rows)
    sizes = result.groupby("family_id").agg(
        family_units=("unit_id", "size"),
        family_conditions=("condition", "nunique"),
    )
    return result.join(sizes, on="family_id")


def summarize_thresholds(
    first: SortData, second: SortData, pairs: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    for threshold in SENSITIVITY_THRESHOLDS:
        assigned = family_assignments(first, second, pairs, threshold)
        edges = qualifying_edges(pairs, threshold)
        for population, population_mask in {
            "all": np.ones(len(assigned), dtype=bool),
            "KS-good": assigned.ks_good.to_numpy(bool),
        }.items():
            selected = assigned[population_mask]
            for condition, group in selected.groupby("condition", sort=False):
                rows.append(
                    {
                        "agreement_threshold": threshold,
                        "population": population,
                        "condition": condition,
                        "n_units": len(group),
                        "effective_cross_condition_families": int(
                            group.family_id.nunique()
                        ),
                        "units_collapsed_by_family_rule": int(
                            len(group) - group.family_id.nunique()
                        ),
                        "units_in_cross_condition_families": int(
                            (group.family_conditions > 1).sum()
                        ),
                        "qualifying_cross_condition_edges": int(len(edges)),
                    }
                )
    return pd.DataFrame(rows)


def classify_single_pass_units(
    second: SortData, pairs: pd.DataFrame, threshold: float = PRIMARY_AGREEMENT
) -> pd.DataFrame:
    edges = qualifying_edges(pairs, threshold)
    rows = []
    for index, unit in enumerate(second.unit_ids):
        candidate = edges[edges.single_pass_unit == int(unit)].sort_values(
            "agreement", ascending=False
        )
        raw_candidate = pairs[pairs.single_pass_unit == int(unit)].sort_values(
            "agreement", ascending=False
        )
        top = raw_candidate.iloc[0] if len(raw_candidate) else None
        rows.append(
            {
                "single_pass_unit": int(unit),
                "ks_good": bool(second.ks_good[index]),
                "spike_count": int(second.spike_counts[index]),
                "depth_um": float(second.depths_um[index]),
                "presence_ratio_10s": float(second.presence_ratio_10s[index]),
                "rate_cv_10s": float(second.rate_cv_10s[index]),
                "max_nearby_template_similarity": float(
                    second.max_nearby_template_similarity[index]
                ),
                "classification": "related_family" if len(candidate) else "unmatched",
                "n_related_current_units": int(len(candidate)),
                "best_current_unit": int(top.current_unit) if top is not None else -1,
                "best_agreement": float(top.agreement) if top is not None else 0.0,
                "best_current_recall": float(top.current_recall) if top is not None else 0.0,
                "best_single_pass_recall": float(top.single_pass_recall)
                if top is not None
                else 0.0,
            }
        )
    return pd.DataFrame(rows)


def build_decision(
    sensitivity: pd.DataFrame,
    classifications: pd.DataFrame,
    duration_s: float = DURATION_S,
) -> dict:
    primary = sensitivity[
        sensitivity.agreement_threshold.eq(PRIMARY_AGREEMENT)
        & sensitivity.population.eq("KS-good")
    ].set_index("condition")
    current_families = int(primary.loc["current", "effective_cross_condition_families"])
    single_families = int(
        primary.loc["single_pass", "effective_cross_condition_families"]
    )
    single_good = classifications[classifications.ks_good]
    related_fraction = float(
        single_good.classification.eq("related_family").mean()
    )
    family_gain = single_families - current_families
    unmatched_good = single_good[single_good.classification.eq("unmatched")]
    conservative_candidates = unmatched_good[
        (unmatched_good.presence_ratio_10s >= 0.90)
        & (unmatched_good.max_nearby_template_similarity < 0.80)
    ]
    moderate_candidates = unmatched_good[
        (unmatched_good.presence_ratio_10s >= 0.75)
        & (unmatched_good.max_nearby_template_similarity < 0.90)
    ]
    threshold_gain = {}
    for threshold, group in sensitivity[sensitivity.population.eq("KS-good")].groupby(
        "agreement_threshold"
    ):
        indexed = group.set_index("condition")
        threshold_gain[str(float(threshold))] = int(
            indexed.loc["single_pass", "effective_cross_condition_families"]
            - indexed.loc["current", "effective_cross_condition_families"]
        )
    return {
        "decision": (
            "advance_single_pass_to_broader_validation_with_strict_candidate_tracking"
            if family_gain > 0
            else "do_not_advance_single_pass"
        ),
        "duration_s": float(duration_s),
        "primary_agreement_threshold": PRIMARY_AGREEMENT,
        "ks_good_effective_families": {
            "current": current_families,
            "single_pass": single_families,
            "difference": family_gain,
            "relative_change": family_gain / max(current_families, 1),
        },
        "single_pass_ks_good_related_to_current_fraction": related_fraction,
        "ks_good_family_gain_by_agreement_threshold": threshold_gain,
        "unmatched_single_pass_ks_good_units": int(len(unmatched_good)),
        "conservative_independent_candidate_count": int(len(conservative_candidates)),
        "conservative_candidate_rule": (
            "KS-good; no qualifying current-family edge; presence >= 0.90; "
            "maximum nearby within-sort template similarity < 0.80"
        ),
        "moderate_independent_candidate_count": int(len(moderate_candidates)),
        "moderate_candidate_rule": (
            "KS-good; no qualifying current-family edge; presence >= 0.75; "
            "maximum nearby within-sort template similarity < 0.90"
        ),
        "interpretation_guardrail": (
            "Advance only if the effective-family gain remains positive across "
            "reasonable agreement thresholds; raw unit count alone is insufficient."
        ),
        "scope_caveat": (
            f"This is a {duration_s:g} s cross-sort continuity audit, not identical "
            "full-session curation or biological ground truth. Weak edges can reflect "
            "task-locked coincidence; match-count and observed/expected guardrails "
            "are applied."
        ),
    }


def run(
    output_dir: Path,
    current_sorter: Path = SORTERS["current"],
    single_sorter: Path = SORTERS["single_pass"],
    duration_s: float = DURATION_S,
) -> dict:
    current = load_sort("current", current_sorter, duration_s)
    single = load_sort("single_pass", single_sorter, duration_s)
    pairs = pairwise_agreement(current, single, duration_s)
    sensitivity = summarize_thresholds(current, single, pairs)
    classifications = classify_single_pass_units(single, pairs)
    assignments = family_assignments(current, single, pairs, PRIMARY_AGREEMENT)
    decision = build_decision(sensitivity, classifications, duration_s)

    output_dir.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(output_dir / "cross_condition_unit_pairs.csv", index=False)
    sensitivity.to_csv(output_dir / "family_threshold_sensitivity.csv", index=False)
    classifications.to_csv(output_dir / "single_pass_unit_classification.csv", index=False)
    assignments.to_csv(output_dir / "primary_family_assignments.csv", index=False)
    (output_dir / "decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    return decision


def main() -> None:
    args = parse_args()
    decision = run(
        args.output_dir,
        current_sorter=args.current_sorter,
        single_sorter=args.single_sorter,
        duration_s=args.duration_s,
    )
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
