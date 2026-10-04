#!/usr/bin/env python3
"""Corrected unit-population known-answer fixture for the EN crop evaluator.

Version 1 incorrectly formed the common domain from individual event depths.
The frozen definition selects units by each arm's median saved y and retains all
events from selected units.  This fixture makes that distinction observable.
It does not validate an evaluator output schema or any scientific outcome.
"""

from __future__ import annotations

import json

import numpy as np


def localize_aligned_sort(
    times: np.ndarray,
    clusters: np.ndarray,
    depths_um: np.ndarray,
    *,
    start_frame: int,
    stop_frame: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    times = np.asarray(times, dtype=np.int64)
    clusters = np.asarray(clusters, dtype=np.int64)
    depths_um = np.asarray(depths_um, dtype=np.float64)
    if not (len(times) == len(clusters) == len(depths_um)):
        raise ValueError("times, clusters, and depths must have identical rows")
    if np.any(times < start_frame) or np.any(times >= stop_frame):
        raise ValueError("exported spike lies outside the frozen half-open crop")
    return times - np.int64(start_frame), clusters.copy(), depths_um.copy()


def median_unit_population_masks(
    clusters: np.ndarray,
    depths_um: np.ndarray,
    *,
    lower_um: float = 120.0,
    upper_um: float = 3780.0,
) -> tuple[dict[str, np.ndarray], dict[int, float]]:
    """Select units by median saved y and map membership to all their events."""
    clusters = np.asarray(clusters, dtype=np.int64)
    depths_um = np.asarray(depths_um, dtype=np.float64)
    if len(clusters) != len(depths_um):
        raise ValueError("clusters and depths must have identical rows")
    units = np.unique(clusters)
    median_y = {int(unit): float(np.median(depths_um[clusters == unit])) for unit in units}
    eligible_units = {
        unit for unit, median in median_y.items() if lower_um <= median <= upper_um
    }
    common = np.fromiter(
        (int(unit) in eligible_units for unit in clusters), dtype=bool, count=len(clusters)
    )
    return {
        "all_output": np.ones(len(clusters), dtype=bool),
        "common_unit_median_y120_3780": common,
    }, median_y


def adjacent_pair_ids(times: np.ndarray, clusters: np.ndarray) -> set[tuple[int, int]]:
    """Return original adjacent event-row identities within each unit."""
    times = np.asarray(times, dtype=np.int64)
    clusters = np.asarray(clusters, dtype=np.int64)
    row = np.arange(len(times), dtype=np.int64)
    order = np.lexsort((times, clusters))
    grouped_row = row[order]
    grouped_cluster = clusters[order]
    same_unit = grouped_cluster[1:] == grouped_cluster[:-1]
    return {
        (int(left), int(right))
        for left, right in zip(grouped_row[:-1][same_unit], grouped_row[1:][same_unit])
    }


def main() -> None:
    start, stop = 1_000, 2_000
    # Unit 7 has two event depths outside the common band, but median y=200, so
    # all four events belong to the common-domain unit population. Unit 8 has
    # median y=3800 and contributes no common-domain events. Unit 9 is included.
    times = np.array([1000, 1010, 1020, 1050, 1005, 1015, 1025, 1030, 1040])
    clusters = np.array([7, 7, 7, 7, 8, 8, 8, 9, 9])
    depths = np.array([100.0, 200.0, 3800.0, 200.0, 100.0, 3800.0, 3900.0, 120.0, 3780.0])
    local, aligned_clusters, aligned_depths = localize_aligned_sort(
        times, clusters, depths, start_frame=start, stop_frame=stop
    )
    populations, medians = median_unit_population_masks(aligned_clusters, aligned_depths)
    expected_common = np.array([True, True, True, True, False, False, False, True, True])
    if not np.array_equal(populations["common_unit_median_y120_3780"], expected_common):
        raise RuntimeError("common population was not assigned by unit median y")
    for unit in np.unique(clusters):
        unit_membership = populations["common_unit_median_y120_3780"][clusters == unit]
        if np.any(unit_membership != unit_membership[0]):
            raise RuntimeError("common-domain membership varies within a unit")

    original_pairs = adjacent_pair_ids(local, aligned_clusters)
    common_rows = np.flatnonzero(populations["common_unit_median_y120_3780"])
    selected_pairs_local = adjacent_pair_ids(local[common_rows], aligned_clusters[common_rows])
    selected_pairs_global = {(int(common_rows[a]), int(common_rows[b])) for a, b in selected_pairs_local}
    expected_selected_pairs = {
        pair for pair in original_pairs if expected_common[pair[0]] and expected_common[pair[1]]
    }
    if selected_pairs_global != expected_selected_pairs:
        raise RuntimeError("full-unit selection changed adjacency within retained units")

    # Demonstrate why the superseded event-depth selection is prohibited: for
    # unit 7 it drops rows 0 and 2, inventing adjacency between rows 1 and 3.
    flawed_event_mask = (depths >= 120.0) & (depths <= 3780.0)
    flawed_rows = np.flatnonzero(flawed_event_mask)
    flawed_local_pairs = adjacent_pair_ids(local[flawed_rows], clusters[flawed_rows])
    flawed_pairs = {(int(flawed_rows[a]), int(flawed_rows[b])) for a, b in flawed_local_pairs}
    invented_by_event_filter = flawed_pairs - original_pairs
    if (1, 3) not in invented_by_event_filter:
        raise RuntimeError("fixture failed to expose event-filter adjacency invention")

    required_contract_fields = [
        "observed_coincidence",
        "null_coincidence",
        "coincidence_excess",
        "eligible_unit_count",
        "valid_bootstrap_draw_count",
        "positive_interval_denominator",
        "short_interval_count_1_29",
        "exact_duplicate_count",
    ]
    result = {
        "schema": "en-common-support-crop-evaluation-fixture-v2",
        "status": "pass",
        "supersedes_flawed_fixture_sha256": "3b5c5213c2f702741d62646cc1fedb2e34aee5b5a8dbf10e4f2c67ad2bb367bd",
        "localized_without_row_filtering": True,
        "unit_median_y_um": {str(unit): value for unit, value in medians.items()},
        "all_output_events": int(populations["all_output"].sum()),
        "common_domain_units": [7, 9],
        "common_domain_events_retained_whole_unit": int(expected_common.sum()),
        "membership_constant_within_unit": True,
        "retained_unit_adjacency_unchanged": True,
        "prohibited_event_filter_invented_pair": [1, 3],
        "required_output_contract_fields_not_yet_verified": required_contract_fields,
        "actual_evaluator_output_schema_checked": False,
        "scientific_outcome_evaluated": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
