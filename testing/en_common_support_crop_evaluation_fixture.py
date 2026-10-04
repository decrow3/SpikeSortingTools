#!/usr/bin/env python3
"""Known-answer contract fixture for the frozen EN crop-screen evaluator.

This does not evaluate sorter output.  It isolates the failure-prone bookkeeping
that a later evaluator must preserve: aligned rows, one global-origin
subtraction, original state-segment adjacency, bootstrap-block crossings, and
explicit all-output/common-domain reporting.
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
    """Assert the frozen crop, then subtract its origin without filtering rows."""
    times = np.asarray(times, dtype=np.int64)
    clusters = np.asarray(clusters, dtype=np.int64)
    depths_um = np.asarray(depths_um, dtype=np.float64)
    if not (len(times) == len(clusters) == len(depths_um)):
        raise ValueError("times, clusters, and depths must have identical rows")
    if np.any(times < start_frame) or np.any(times >= stop_frame):
        raise ValueError("exported spike lies outside the frozen half-open crop")
    return times - np.int64(start_frame), clusters.copy(), depths_um.copy()


def adjacent_interval_masks(
    local_times: np.ndarray,
    clusters: np.ndarray,
    state_segment_id: np.ndarray,
    block_id: np.ndarray,
) -> dict[str, np.ndarray]:
    """Form original adjacent pairs; block boundaries do not invalidate pairs."""
    local_times = np.asarray(local_times, dtype=np.int64)
    clusters = np.asarray(clusters, dtype=np.int64)
    segment = np.asarray(state_segment_id, dtype=np.int64)
    block = np.asarray(block_id, dtype=np.int64)
    order = np.lexsort((local_times, clusters))
    grouped_time = local_times[order]
    grouped_cluster = clusters[order]
    grouped_segment = segment[order]
    grouped_block = block[order]
    delta = np.diff(grouped_time)
    safe = (
        (grouped_cluster[1:] == grouped_cluster[:-1])
        & (grouped_segment[1:] == grouped_segment[:-1])
    )
    return {
        "delta": delta,
        "safe": safe,
        "positive": safe & (delta > 0),
        "short_1_29": safe & (delta >= 1) & (delta < 30),
        "exact": safe & (delta == 0),
        "cross_block": safe & (grouped_block[1:] != grouped_block[:-1]),
        "assigned_block": grouped_block[1:],
    }


def population_masks(depths_um: np.ndarray) -> dict[str, np.ndarray]:
    """Keep all-output and frozen common-domain populations simultaneously."""
    depths_um = np.asarray(depths_um, dtype=np.float64)
    return {
        "all_output": np.ones(len(depths_um), dtype=bool),
        "common_y120_3780": (depths_um >= 120.0) & (depths_um <= 3780.0),
    }


def main() -> None:
    start, stop = 1_000, 2_000
    times = np.array([1_000, 1_010, 1_020, 1_050, 1_100, 1_110], dtype=np.int64)
    clusters = np.array([7, 7, 7, 7, 7, 7], dtype=np.int64)
    depths = np.array([100.0, 120.0, 200.0, 3_780.0, 3_800.0, 200.0])
    local, aligned_clusters, aligned_depths = localize_aligned_sort(
        times, clusters, depths, start_frame=start, stop_frame=stop
    )
    if not np.array_equal(local, [0, 10, 20, 50, 100, 110]):
        raise RuntimeError("global origin was not subtracted exactly once")
    if not np.array_equal(aligned_clusters, clusters) or not np.array_equal(aligned_depths, depths):
        raise RuntimeError("aligned sort rows changed during localization")

    # Pair 20->50 crosses a bootstrap block but remains valid. Pair 50->100 is
    # invalid because it crosses an original state-segment boundary.
    segment = np.array([0, 0, 0, 0, 1, 1], dtype=np.int64)
    block = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    pair = adjacent_interval_masks(local, clusters, segment, block)
    if int(pair["positive"].sum()) != 4 or int(pair["short_1_29"].sum()) != 3:
        raise RuntimeError("original positive/short adjacency counts changed")
    if int(pair["cross_block"].sum()) != 1:
        raise RuntimeError("valid cross-bootstrap-block pair was discarded")
    cross_index = int(np.flatnonzero(pair["cross_block"])[0])
    if int(pair["assigned_block"][cross_index]) != 1:
        raise RuntimeError("cross-block pair was not assigned to its second event")

    populations = population_masks(depths)
    if int(populations["all_output"].sum()) != 6 or int(populations["common_y120_3780"].sum()) != 4:
        raise RuntimeError("all-output/common-domain populations are conflated")

    alignment_negative = False
    bounds_negative = False
    try:
        localize_aligned_sort(times, clusters[:-1], depths, start_frame=start, stop_frame=stop)
    except ValueError:
        alignment_negative = True
    try:
        localize_aligned_sort(
            np.r_[times[:-1], stop], clusters, depths, start_frame=start, stop_frame=stop
        )
    except ValueError:
        bounds_negative = True
    if not alignment_negative or not bounds_negative:
        raise RuntimeError("alignment or half-open-bound negative control failed")

    required_report_fields = {
        "observed_coincidence",
        "null_coincidence",
        "coincidence_excess",
        "eligible_unit_count",
        "valid_bootstrap_draw_count",
        "positive_interval_denominator",
        "short_interval_count_1_29",
        "exact_duplicate_count",
    }
    result = {
        "schema": "en-common-support-crop-evaluation-fixture-v1",
        "status": "pass",
        "localized_without_row_filtering": True,
        "aligned_row_negative_control": alignment_negative,
        "half_open_bound_negative_control": bounds_negative,
        "positive_intervals": int(pair["positive"].sum()),
        "short_intervals_1_29": int(pair["short_1_29"].sum()),
        "cross_bootstrap_block_pairs_retained": int(pair["cross_block"].sum()),
        "cross_block_pair_assigned_to_second_event": True,
        "all_output_rows": int(populations["all_output"].sum()),
        "common_domain_rows": int(populations["common_y120_3780"].sum()),
        "required_report_fields": sorted(required_report_fields),
        "scientific_outcome_evaluated": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
