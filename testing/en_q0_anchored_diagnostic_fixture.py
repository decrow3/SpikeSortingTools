#!/usr/bin/env python3
"""Independent known-answer fixture for q0 mapping and segment-safe matching."""

from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from testing.en_q0_anchored_diagnostic import (
    assert_unit_event_partition,
    exclusive_same_segment_count,
    full_unit_state_tables,
    q0_anchor_counts,
    q0_graph,
)
from testing.en_tier1_panel import FieldCells


def exhaustive_count(a: tuple[int, ...], b: tuple[int, ...], tolerance: int) -> int:
    """Independent exhaustive maximum-cardinality matcher for one segment."""
    @lru_cache(None)
    def solve(i: int, used: int) -> int:
        if i == len(a):
            return 0
        best = solve(i + 1, used)
        for j, value in enumerate(b):
            if not (used >> j) & 1 and abs(a[i] - value) <= tolerance:
                best = max(best, 1 + solve(i + 1, used | (1 << j)))
        return best
    return solve(0, 0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rng = np.random.default_rng(20261001)
    exhaustive_cases = 0
    for _ in range(200):
        a = np.sort(rng.choice(30, size=rng.integers(0, 7), replace=False)).astype(np.int64)
        b = np.sort(rng.choice(30, size=rng.integers(0, 7), replace=False)).astype(np.int64)
        tolerance = int(rng.integers(0, 4))
        observed = exclusive_same_segment_count(a, np.zeros(len(a), int), b, np.zeros(len(b), int), tolerance)
        expected = exhaustive_count(tuple(map(int, a)), tuple(map(int, b)), tolerance)
        if observed != expected:
            raise RuntimeError("chronological matcher differs from exhaustive maximum cardinality")
        exhaustive_cases += 1

    # A close event across an original segment boundary must not match.
    boundary = exclusive_same_segment_count(
        np.array([100]), np.array([7]), np.array([101]), np.array([8]), tolerance=2)
    if boundary != 0:
        raise RuntimeError("cross-segment event was matched")
    block_boundary = exclusive_same_segment_count(
        np.array([29999]), np.array([14]), np.array([30001]), np.array([15]), tolerance=2)
    if block_boundary != 0:
        raise RuntimeError("cross-accounting-block event was matched")
    segment_graph = q0_graph(
        {"times": np.array([100]), "clusters": np.array([0])},
        {"times": np.array([101]), "clusters": np.array([1])},
        np.zeros(1, np.int16), np.zeros(1, np.int16), np.array([7]), np.array([8]),
        0, 2, 0.1, 0.5)
    if len(segment_graph):
        raise RuntimeError("q0 graph admitted a cross-segment match")
    cells = FieldCells(
        states_um=np.array([0.0, 40.0]), knot_edges_s=np.array([0.0, 1.0, 2.0]),
        knot_state_index=np.array([0, 1], dtype=np.int16),
        knot_segment_id=np.array([7, 8], dtype=np.int32),
        state_exposure_s=np.array([1.0, 1.0]), block_exposure_s=np.array([[1.0, 1.0]]),
        block_edges_s=np.array([0.0, 2.0]))
    adjacency = full_unit_state_tables(
        {"times": np.array([50, 60, 99, 101]), "clusters": np.zeros(4, dtype=int)}, cells, 100.0)
    if adjacency["denominator"].sum() != 2 or adjacency["short"].sum() != 1:
        raise RuntimeError("full-train adjacency crossed a state-segment boundary or filtered too early")
    block_cells = FieldCells(
        states_um=np.array([0.0]), knot_edges_s=np.array([0.0, 2.0]),
        knot_state_index=np.array([0], dtype=np.int16),
        knot_segment_id=np.array([7], dtype=np.int32), state_exposure_s=np.array([2.0]),
        block_exposure_s=np.array([[1.0], [1.0]]), block_edges_s=np.array([0.0, 1.0, 2.0]))
    block_adjacency = full_unit_state_tables(
        {"times": np.array([99, 101]), "clusters": np.zeros(2, dtype=int)}, block_cells, 100.0)
    if (block_adjacency["denominator"].sum() != 1 or block_adjacency["short"].sum() != 1
            or block_adjacency["unit_block_denominator"][0, 1, 0] != 1
            or block_adjacency["unit_block_short"][0, 1, 0] != 1):
        raise RuntimeError("cross-block adjacency was not retained in the second-event block")

    # Anchor construction must retain units with no q0 events, and both the REF
    # and candidate coverage bucket schemes must partition all events.
    coverage_events = {"times": np.array([10, 20, 110, 120, 130]),
                       "clusters": np.array([0, 0, 1, 2, 2])}
    coverage_table = full_unit_state_tables(coverage_events, block_cells, 100.0)
    coverage_state = coverage_table["state"]
    anchors = q0_anchor_counts(coverage_events["clusters"], coverage_state, 0, 2)
    if anchors.set_index("reference_unit").q0_events.to_dict() != {0: 2, 1: 1, 2: 2}:
        raise RuntimeError("all-unit anchor q0 accounting failed")
    zero_q0_state = np.array([0, 0, 1, 1, 1], dtype=np.int16)
    zero_q0_anchors = q0_anchor_counts(coverage_events["clusters"], zero_q0_state, 0, 2)
    if int(zero_q0_anchors.set_index("reference_unit").loc[2, "q0_events"]) != 0:
        raise RuntimeError("zero-q0-only unit was omitted or miscounted")
    assert_unit_event_partition(coverage_table,
        {"mapped_eligible_ref_anchor": np.array([0]),
         "unmatched_eligible_ref_anchor": np.array([2]),
         "q0_ineligible_ref": np.array([1])}, "fixture-ref")
    assert_unit_event_partition(coverage_table,
        {"mapped_candidate": np.array([0, 2]),
         "candidate_outside_mapped_units": np.array([1])}, "fixture-candidate")

    # Identity graph: both units are reciprocal unique primaries.
    ref = {"times": np.array([10, 20, 30, 100, 110, 120]),
           "clusters": np.array([0, 0, 0, 1, 1, 1])}
    state = np.zeros(6, dtype=np.int16)
    identity = q0_graph(ref, ref, state, state, np.zeros(6, int), np.zeros(6, int), 0, 0, 0.1, 0.5)
    primaries = identity[identity.primary_match]
    if len(primaries) != 2 or not np.array_equal(
            primaries[["reference_cluster", "candidate_cluster"]].to_numpy(), [[0, 0], [1, 1]]):
        raise RuntimeError("identity primary mapping failed")

    # Split/tie control: reference unit 0 has equal candidate edges, so neither
    # is a reciprocal unique primary; both secondary edges remain preserved.
    split_candidate = {"times": np.array([10, 20, 30, 40]), "clusters": np.array([2, 2, 3, 3])}
    split_ref = {"times": np.array([10, 20, 30, 40]), "clusters": np.zeros(4, dtype=int)}
    split_state = np.zeros(4, dtype=np.int16)
    split = q0_graph(split_ref, split_candidate, split_state, split_state,
                     np.zeros(4, int), np.zeros(4, int), 0, 0, 0.1, 0.5)
    if len(split) != 2 or split.primary_match.any():
        raise RuntimeError("split/tie edges were collapsed into a primary")

    # Dropped events reduce reference retention and fail the primary threshold.
    dropped_candidate = {"times": np.array([10]), "clusters": np.array([5])}
    dropped = q0_graph(split_ref, dropped_candidate, split_state, np.zeros(1, dtype=np.int16),
                       np.zeros(4, int), np.zeros(1, int), 0, 0, 0.1, 0.5)
    if len(dropped) != 1 or dropped.primary_match.any() or dropped.iloc[0].reference_retention != 0.25:
        raise RuntimeError("failed-retention edge was not preserved separately")
    empty_candidate = {"times": np.array([], dtype=np.int64), "clusters": np.array([], dtype=int)}
    empty = q0_graph(split_ref, empty_candidate, split_state, np.array([], dtype=np.int16),
                     np.zeros(4, int), np.array([], dtype=int), 0, 0, 0.1, 0.5)
    if len(empty) or list(empty.columns) != [
            "reference_cluster", "candidate_cluster", "matched_events", "reference_events",
            "candidate_events", "reference_retention", "candidate_retention", "f1", "jaccard",
            "primary_match"]:
        raise RuntimeError("empty graph did not preserve the diagnostic schema")

    result = {"schema": "en-q0-anchored-diagnostic-fixture-v1", "status": "pass",
        "independent_exhaustive_cases": exhaustive_cases, "identity_self_mapping": True,
        "split_tie_edges_preserved_without_primary": True,
        "dropped_event_failed_primary_preserved": True,
        "cross_segment_boundary_match_rejected": True,
        "cross_accounting_block_match_rejected": True,
        "q0_graph_cross_segment_match_rejected": True,
        "full_train_short_adjacency_boundary_control": True,
        "cross_block_short_adjacency_retained_in_second_event_block": True,
        "zero_q0_unit_retained_in_ineligible_bucket": True,
        "reference_and_candidate_buckets_partition_events": True,
        "empty_graph_schema_preserved": True,
    }
    if args.output:
        if args.output.exists():
            raise FileExistsError(args.output)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
