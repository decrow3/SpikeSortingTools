"""Reconcile cross-sort unit families on Luke's matched rapid-motion band."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from testing.luke_sorter_band_comparison import (
    DEFAULT_OUTPUT as BAND_OUTPUT,
    DEFAULT_RESCUE_ROOT,
    SpikeSet,
    _load_json,
    _validate_spikes,
    cross_unit_coincidence_fraction,
    load_dartsort_band,
    load_kiasort_band,
    load_ks4_band,
    unit_metrics,
)


DEFAULT_OUTPUT = Path("testing/outputs/luke_sorter_unit_families")


def directional_unit_support(
    source: SpikeSet,
    target: SpikeSet,
    time_radius: int,
    depth_radius_um: float,
    bin_samples: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Count each source event once per nearby target unit."""
    pair_counts: Counter[tuple[int, int]] = Counter()
    bin_counts: Counter[tuple[int, int, int]] = Counter()
    left = 0
    for index, (time, label, depth) in enumerate(
        zip(source.times, source.labels, source.depths_um)
    ):
        while left < target.times.size and target.times[left] < time - time_radius:
            left += 1
        matched_units: set[int] = set()
        right = left
        while right < target.times.size and target.times[right] <= time + time_radius:
            if abs(target.depths_um[right] - depth) <= depth_radius_um:
                matched_units.add(int(target.labels[right]))
            right += 1
        source_unit = int(label)
        time_bin = int(time // bin_samples)
        for target_unit in matched_units:
            pair_counts[(source_unit, target_unit)] += 1
            bin_counts[(source_unit, target_unit, time_bin)] += 1
    pairs = pd.DataFrame(
        [
            {"source_unit": key[0], "target_unit": key[1], "matched_source_events": value}
            for key, value in pair_counts.items()
        ]
    )
    bins = pd.DataFrame(
        [
            {
                "source_unit": key[0],
                "target_unit": key[1],
                "time_bin": key[2],
                "matched_source_events": value,
            }
            for key, value in bin_counts.items()
        ]
    )
    return pairs, bins


def unit_pair_edges(
    first: SpikeSet,
    second: SpikeSet,
    time_radius: int,
    depth_radius_um: float,
    bin_samples: int,
    minimum_unit_spikes: int = 20,
    minimum_pair_events: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    first_totals = Counter(map(int, first.labels))
    second_totals = Counter(map(int, second.labels))
    forward, forward_bins = directional_unit_support(
        first, second, time_radius, depth_radius_um, bin_samples
    )
    reverse, reverse_bins = directional_unit_support(
        second, first, time_radius, depth_radius_um, bin_samples
    )
    if forward.empty or reverse.empty:
        return pd.DataFrame(), pd.DataFrame()
    forward = forward.rename(
        columns={
            "source_unit": "first_unit",
            "target_unit": "second_unit",
            "matched_source_events": "matched_first_events",
        }
    )
    reverse = reverse.rename(
        columns={
            "source_unit": "second_unit",
            "target_unit": "first_unit",
            "matched_source_events": "matched_second_events",
        }
    )
    edges = forward.merge(reverse, on=["first_unit", "second_unit"], how="outer").fillna(0)
    for column in ("first_unit", "second_unit", "matched_first_events", "matched_second_events"):
        edges[column] = edges[column].astype(int)
    edges.insert(0, "first_sorter", first.name)
    edges.insert(1, "second_sorter", second.name)
    edges["first_spike_count"] = edges.first_unit.map(first_totals).astype(int)
    edges["second_spike_count"] = edges.second_unit.map(second_totals).astype(int)
    edges["first_coverage"] = edges.matched_first_events / edges.first_spike_count
    edges["second_coverage"] = edges.matched_second_events / edges.second_spike_count
    eligible = (edges.first_spike_count >= minimum_unit_spikes) & (
        edges.second_spike_count >= minimum_unit_spikes
    )
    enough_events = np.minimum(edges.matched_first_events, edges.matched_second_events) >= minimum_pair_events
    enough_coverage = (np.maximum(edges.first_coverage, edges.second_coverage) >= 0.5) | (
        np.minimum(edges.first_coverage, edges.second_coverage) >= 0.2
    )
    edges["qualified_family_edge"] = eligible & enough_events & enough_coverage
    forward_bins.insert(0, "source_sorter", first.name)
    forward_bins.insert(1, "target_sorter", second.name)
    reverse_bins.insert(0, "source_sorter", second.name)
    reverse_bins.insert(1, "target_sorter", first.name)
    return edges.sort_values(
        ["qualified_family_edge", "matched_first_events"], ascending=[False, False]
    ), pd.concat([forward_bins, reverse_bins], ignore_index=True)


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


def component_table(
    spike_sets: list[SpikeSet], edges: pd.DataFrame, prefix: str
) -> pd.DataFrame:
    nodes = [
        (spikes.name, int(unit))
        for spikes in spike_sets
        for unit, count in Counter(map(int, spikes.labels)).items()
        if count >= 20
    ]
    graph = UnionFind(nodes)
    for row in edges.loc[edges.qualified_family_edge].itertuples():
        graph.union((row.first_sorter, int(row.first_unit)), (row.second_sorter, int(row.second_unit)))
    components: defaultdict[tuple[str, int], list[tuple[str, int]]] = defaultdict(list)
    for node in nodes:
        components[graph.find(node)].append(node)
    rows = []
    sorter_names = [spikes.name for spikes in spike_sets]
    for index, members in enumerate(sorted(components.values(), key=lambda value: min(value))):
        row: dict[str, object] = {"family_id": f"{prefix}_{index:04d}"}
        for sorter in sorter_names:
            units = sorted(unit for name, unit in members if name == sorter)
            row[f"{sorter}_unit_count"] = len(units)
            row[f"{sorter}_units"] = ";".join(map(str, units))
        counts = [int(row[f"{name}_unit_count"]) for name in sorter_names]
        row["mapping_shape"] = "x".join(map(str, counts))
        row["total_units"] = sum(counts)
        rows.append(row)
    return pd.DataFrame(rows)


def requalify_edges(
    edges: pd.DataFrame,
    minimum_pair_events: int,
    maximum_direction_coverage: float,
    minimum_direction_coverage: float,
) -> pd.DataFrame:
    result = edges.copy()
    enough_events = np.minimum(result.matched_first_events, result.matched_second_events) >= minimum_pair_events
    enough_coverage = (
        np.maximum(result.first_coverage, result.second_coverage) >= maximum_direction_coverage
    ) | (np.minimum(result.first_coverage, result.second_coverage) >= minimum_direction_coverage)
    eligible = (result.first_spike_count >= 20) & (result.second_spike_count >= 20)
    result["qualified_family_edge"] = eligible & enough_events & enough_coverage
    return result


def sensitivity_summary(
    first: SpikeSet, second: SpikeSet, edges: pd.DataFrame
) -> pd.DataFrame:
    settings = {
        "permissive": (10, 0.4, 0.1),
        "baseline": (20, 0.5, 0.2),
        "stringent": (50, 0.7, 0.3),
        "reciprocal": (20, 1.1, 0.5),
    }
    rows = []
    for name, (events, maximum, minimum) in settings.items():
        qualified = requalify_edges(edges, events, maximum, minimum)
        families = component_table([first, second], qualified, name)
        shapes = families.mapping_shape.value_counts()
        rows.append(
            {
                "setting": name,
                "minimum_pair_events": events,
                "maximum_direction_coverage_threshold": maximum,
                "minimum_direction_coverage_threshold": minimum,
                "qualified_edges": int(qualified.qualified_family_edge.sum()),
                "one_to_one_families": int(shapes.get("1x1", 0)),
                "first_only_units": int(shapes.get("1x0", 0)),
                "second_only_units": int(shapes.get("0x1", 0)),
                "multi_unit_families": int(
                    sum(count for shape, count in shapes.items() if shape not in {"1x0", "0x1", "1x1"})
                ),
                "largest_family_units": int(families.total_units.max()),
            }
        )
    return pd.DataFrame(rows)


def isolated_unit_table(
    families: pd.DataFrame,
    first: SpikeSet,
    second: SpikeSet,
    fs: float,
    duration_s: float,
    time_radius: int,
    depth_radius_um: float,
) -> pd.DataFrame:
    unit_tables = {
        first.name: unit_metrics(first, fs, duration_s).set_index("unit_id"),
        second.name: unit_metrics(second, fs, duration_s).set_index("unit_id"),
    }
    rows = []
    for family in families.itertuples():
        for source, target in ((first, second), (second, first)):
            values = getattr(family, f"{source.name}_units")
            other_values = getattr(family, f"{target.name}_units")
            if not values or other_values:
                continue
            for unit_text in values.split(";"):
                unit = int(unit_text)
                mask = source.labels == unit
                subset = SpikeSet(
                    source.name,
                    source.times[mask],
                    source.labels[mask],
                    source.depths_um[mask],
                )
                assignments = best_target_labels(
                    subset,
                    target,
                    set(map(int, np.unique(target.labels))),
                    time_radius,
                    depth_radius_um,
                )
                metrics = unit_tables[source.name].loc[unit]
                rows.append(
                    {
                        "family_id": family.family_id,
                        "sorter": source.name,
                        "unit_id": unit,
                        "spike_count": int(subset.times.size),
                        "counterpart_event_fraction": float(np.mean(assignments >= 0)),
                        "refractory_fraction_1p5ms": float(metrics.refractory_fraction_1p5ms),
                        "presence_fraction_10s": float(metrics.presence_fraction_10s),
                        "median_depth_um": float(metrics.median_depth_um),
                    }
                )
    return pd.DataFrame(rows).sort_values(["sorter", "spike_count"], ascending=[True, False])


def best_target_labels(
    source: SpikeSet,
    target: SpikeSet,
    allowed_target_units: set[int],
    time_radius: int,
    depth_radius_um: float,
) -> np.ndarray:
    """Choose the nearest eligible target unit for each source event."""
    result = np.full(source.times.size, -1, dtype=np.int64)
    left = 0
    for index, (time, depth) in enumerate(zip(source.times, source.depths_um)):
        while left < target.times.size and target.times[left] < time - time_radius:
            left += 1
        candidates = []
        right = left
        while right < target.times.size and target.times[right] <= time + time_radius:
            label = int(target.labels[right])
            depth_delta = abs(target.depths_um[right] - depth)
            if label in allowed_target_units and depth_delta <= depth_radius_um:
                candidates.append((abs(int(target.times[right]) - int(time)), depth_delta, label))
            right += 1
        if candidates:
            result[index] = min(candidates)[2]
    return result


def candidate_metrics(
    families: pd.DataFrame,
    ks4: SpikeSet,
    kia: SpikeSet,
    time_radius: int,
    depth_radius_um: float,
    bin_samples: int,
) -> pd.DataFrame:
    rows = []
    for family in families.itertuples():
        ks_units = {int(value) for value in family.ks4_no_motion_units.split(";") if value}
        kia_units = {int(value) for value in family.kiasort_band_pilot_units.split(";") if value}
        if len(ks_units) >= 2 and len(kia_units) == 1:
            category, source, target, targets = (
                "several_ks4_to_one_kiasort",
                kia,
                ks4,
                ks_units,
            )
            source_units = kia_units
        elif len(ks_units) == 1 and len(kia_units) >= 2:
            category, source, target, targets = (
                "one_ks4_to_several_kiasort",
                ks4,
                kia,
                kia_units,
            )
            source_units = ks_units
        else:
            continue
        source_mask = np.isin(source.labels, list(source_units))
        source_subset = SpikeSet(
            source.name,
            source.times[source_mask],
            source.labels[source_mask],
            source.depths_um[source_mask],
        )
        assignments = best_target_labels(
            source_subset, target, targets, time_radius, depth_radius_um
        )
        matched = assignments >= 0
        target_counts = Counter(map(int, assignments[matched]))
        best_single = max(target_counts.values(), default=0) / max(source_subset.times.size, 1)
        dominant = []
        for time_bin in range(12):
            mask = matched & (source_subset.times // bin_samples == time_bin)
            if np.any(mask):
                dominant.append(Counter(map(int, assignments[mask])).most_common(1)[0][0])
        target_mask = np.isin(target.labels, list(targets))
        target_coincidence = cross_unit_coincidence_fraction(
            target.times[target_mask],
            target.labels[target_mask],
            target.depths_um[target_mask],
            max(1, time_radius // 2),
            75.0,
        )
        rows.append(
            {
                "family_id": family.family_id,
                "category": category,
                "ks4_units": family.ks4_no_motion_units,
                "kiasort_units": family.kiasort_band_pilot_units,
                "source_event_count": int(source_subset.times.size),
                "source_union_coverage": float(matched.mean()),
                "best_single_target_coverage": float(best_single),
                "union_coverage_gain": float(matched.mean() - best_single),
                "distinct_dominant_targets": len(set(dominant)),
                "dominant_target_switches": int(
                    sum(first != second for first, second in zip(dominant, dominant[1:]))
                ),
                "target_cross_unit_coincidence_fraction": target_coincidence,
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["union_coverage_gain", "source_union_coverage"], ascending=False
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rescue-root", type=Path, default=DEFAULT_RESCUE_ROOT)
    parser.add_argument("--band-output", type=Path, default=BAND_OUTPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--channel-start", type=int, default=82)
    parser.add_argument("--channel-count", type=int, default=32)
    parser.add_argument("--kiasort-output-name")
    args = parser.parse_args()

    window_dir = args.rescue_root / "sorter_bakeoff/windows/rapid_motion-8b4978262d"
    channel_end = args.channel_start + args.channel_count
    kia_name = args.kiasort_output_name or (
        f"kiasort_channels_{args.channel_start}_{channel_end}"
    )
    manifests = {
        name: _load_json(window_dir / name / "bakeoff_sort_manifest.json")
        for name in ("ks4_no_motion", "dartsort_native", kia_name)
    }
    if not all(value.get("complete") is True for value in manifests.values()):
        raise RuntimeError("A source sorter manifest is incomplete")
    if any(value.get("raw_voltage_warp") is not False for value in manifests.values()):
        raise RuntimeError("Unit families require unwarped-voltage outputs")
    windows = [value["window"] for value in manifests.values()]
    if len({value["request_digest"] for value in windows}) != 1:
        raise RuntimeError("Source sorter windows differ")
    fs = float(windows[0]["sampling_frequency_hz"])
    frame_count = int(windows[0]["frame_count"])
    first_depth_row, last_depth_row = args.channel_start // 2, (channel_end - 1) // 2
    spike_sets = [
        load_ks4_band(
            args.rescue_root,
            int(windows[0]["start_frame"]),
            int(windows[0]["end_frame"]),
            first_depth_row,
            last_depth_row,
        ),
        load_dartsort_band(window_dir, args.channel_start, channel_end),
        load_kiasort_band(window_dir, kia_name),
    ]
    for spikes in spike_sets:
        _validate_spikes(spikes, frame_count)

    time_radius = int(round(0.5e-3 * fs))
    bin_samples = int(round(10.0 * fs))
    pair_results = []
    bin_results = []
    for first_index, second_index in ((0, 1), (0, 2), (1, 2)):
        edges, bins = unit_pair_edges(
            spike_sets[first_index],
            spike_sets[second_index],
            time_radius,
            60.0,
            bin_samples,
        )
        pair_results.append(edges)
        bin_results.append(bins)
    all_edges = pd.concat(pair_results, ignore_index=True)
    all_bins = pd.concat(bin_results, ignore_index=True)
    three_sorter_families = component_table(spike_sets, all_edges, "tri")
    ks_kia_edges = pair_results[1]
    ks_kia_families = component_table([spike_sets[0], spike_sets[2]], ks_kia_edges, "ks_kia")
    candidates = candidate_metrics(
        ks_kia_families,
        spike_sets[0],
        spike_sets[2],
        time_radius,
        60.0,
        bin_samples,
    )
    sensitivity = sensitivity_summary(spike_sets[0], spike_sets[2], ks_kia_edges)
    isolated = isolated_unit_table(
        ks_kia_families,
        spike_sets[0],
        spike_sets[2],
        fs,
        frame_count / fs,
        time_radius,
        60.0,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_edges.to_csv(args.output_dir / "unit_pair_edges.csv", index=False)
    all_bins.to_csv(args.output_dir / "unit_pair_time_bins.csv", index=False)
    ks_kia_families.to_csv(args.output_dir / "ks4_kiasort_families.csv", index=False)
    three_sorter_families.to_csv(args.output_dir / "three_sorter_families.csv", index=False)
    candidates.to_csv(args.output_dir / "ks4_kiasort_candidates.csv", index=False)
    sensitivity.to_csv(args.output_dir / "ks4_kiasort_threshold_sensitivity.csv", index=False)
    isolated.to_csv(args.output_dir / "isolated_units.csv", index=False)
    category_counts = (
        candidates.category.value_counts().to_dict() if not candidates.empty else {}
    )
    metadata = {
        "status": "event_level_unit_family_reconciliation_complete",
        "window_request_digest": windows[0]["request_digest"],
        "channel_selection": [args.channel_start, channel_end],
        "event_tolerance_ms": 0.5,
        "event_depth_tolerance_um": 60.0,
        "minimum_unit_spikes": 20,
        "kiasort_output_name": kia_name,
        "minimum_pair_events": 20,
        "qualified_edge_rule": "max directional coverage >= 0.5 or min directional coverage >= 0.2",
        "candidate_category_counts": category_counts,
        "isolated_unit_counts": isolated.sorter.value_counts().to_dict(),
        "limitations": [
            "Event coincidence does not establish biological unit identity.",
            "KIASORT was sorted on the band; KS4 and DARTsort were filtered post hoc.",
            "Waveform and residual arbitration remain required before advancing a family.",
        ],
    }
    (args.output_dir / "family_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(ks_kia_families.groupby("mapping_shape").size().sort_values(ascending=False))
    if not candidates.empty:
        print("\nTop KS4/KIASORT reconciliation candidates:\n")
        print(candidates.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
