#!/usr/bin/env python3
"""Exploratory +40 um partner-switch diagnostic on frozen saved sort outputs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd

from testing.en_q0_anchored_diagnostic import (
    ARMS,
    COMPARATORS,
    atomic_json,
    clean,
    load_events,
    load_field,
    q0_graph,
    sha256,
    state_assignments,
    verify_frozen_inputs,
    verify_manifest_products,
)


SCHEMA = "en-q40-partner-switch-v1"


def best_edge(edges: pd.DataFrame, endpoint: str, unit: int) -> pd.Series | None:
    selected = edges[edges[endpoint] == unit]
    if selected.empty:
        return None
    other = "candidate_cluster" if endpoint == "reference_cluster" else "reference_cluster"
    return selected.sort_values(
        ["jaccard", "matched_events", other], ascending=[False, False, True], kind="stable"
    ).iloc[0]


def classify_pair(row: pd.Series, edges: pd.DataFrame, ref_events: dict[int, int],
                  cand_events: dict[int, int]) -> dict[str, Any]:
    ref, cand = int(row.reference_cluster), int(row.candidate_cluster)
    primary = edges[edges.primary_match]
    if primary.reference_cluster.duplicated().any() or primary.candidate_cluster.duplicated().any():
        raise RuntimeError("state primary graph is not one-to-one")
    ref_primary_rows = primary[primary.reference_cluster == ref]
    cand_primary_rows = primary[primary.candidate_cluster == cand]
    ref_primary = None if ref_primary_rows.empty else int(ref_primary_rows.iloc[0].candidate_cluster)
    cand_primary = None if cand_primary_rows.empty else int(cand_primary_rows.iloc[0].reference_cluster)
    old_rows = edges[(edges.reference_cluster == ref) & (edges.candidate_cluster == cand)]
    if len(old_rows) > 1:
        raise RuntimeError("duplicate old-pair edge")
    old = None if old_rows.empty else old_rows.iloc[0]
    ref_best, cand_best = best_edge(edges, "reference_cluster", ref), best_edge(edges, "candidate_cluster", cand)

    result = {
        "reference_unit": ref,
        "candidate_unit": cand,
        "reference_state_events": int(ref_events.get(ref, 0)),
        "candidate_state_events": int(cand_events.get(cand, 0)),
        "old_pair_has_qualifying_edge": old is not None,
        "old_pair_is_state_primary": bool(old.primary_match) if old is not None else False,
        "old_pair_matched_events": int(old.matched_events) if old is not None else 0,
        "old_pair_reference_retention": float(old.reference_retention) if old is not None else math.nan,
        "old_pair_candidate_retention": float(old.candidate_retention) if old is not None else math.nan,
        "old_pair_f1": float(old.f1) if old is not None else math.nan,
        "old_pair_jaccard": float(old.jaccard) if old is not None else math.nan,
        "reference_state_primary_candidate": ref_primary,
        "candidate_state_primary_reference": cand_primary,
        "reference_endpoint_has_state_primary": ref_primary is not None,
        "candidate_endpoint_has_state_primary": cand_primary is not None,
        "reference_endpoint_switched": ref_primary is not None and ref_primary != cand,
        "candidate_endpoint_switched": cand_primary is not None and cand_primary != ref,
        "both_endpoints_switched": (ref_primary is not None and ref_primary != cand and
                                    cand_primary is not None and cand_primary != ref),
        "reference_qualifying_degree": int((edges.reference_cluster == ref).sum()),
        "candidate_qualifying_degree": int((edges.candidate_cluster == cand).sum()),
    }
    for prefix, edge, other in (("reference_best", ref_best, "candidate_cluster"),
                                ("candidate_best", cand_best, "reference_cluster")):
        result[f"{prefix}_other_unit"] = None if edge is None else int(edge[other])
        result[f"{prefix}_matched_events"] = 0 if edge is None else int(edge.matched_events)
        result[f"{prefix}_jaccard"] = math.nan if edge is None else float(edge.jaccard)
        result[f"{prefix}_is_primary"] = False if edge is None else bool(edge.primary_match)
    return result


def summarize(details: pd.DataFrame, comparison: str, cohort: str) -> dict[str, Any]:
    flags = [
        "old_pair_has_qualifying_edge", "old_pair_is_state_primary",
        "reference_endpoint_has_state_primary", "candidate_endpoint_has_state_primary",
        "reference_endpoint_switched", "candidate_endpoint_switched", "both_endpoints_switched",
    ]
    result: dict[str, Any] = {"comparison": comparison, "cohort": cohort, "pairs": int(len(details))}
    for flag in flags:
        result[flag] = int(details[flag].sum())
    result["reference_zero_state_events"] = int((details.reference_state_events == 0).sum())
    result["candidate_zero_state_events"] = int((details.candidate_state_events == 0).sum())
    result["reference_state_events"] = int(details.reference_state_events.sum())
    result["candidate_state_events"] = int(details.candidate_state_events.sum())
    # Old q0 primaries are disjoint pairs, so this total does not double-count a
    # secondary edge. Secondary-edge matched counts are never summed.
    result["old_pair_matched_events"] = int(details.old_pair_matched_events.sum())
    return result


def verify_inputs(config: dict[str, Any]) -> dict[str, Any]:
    if sha256(Path(__file__)) != config["source_sha256"]:
        raise RuntimeError("frozen source differs")
    parent_path = Path(config["parent_config_path"])
    if sha256(parent_path) != config["parent_config_sha256"]:
        raise RuntimeError("parent config differs")
    parent = json.loads(parent_path.read_text())
    verify_frozen_inputs(parent)
    root = Path(config["mapping_root"])
    if sha256(root / "MAPPINGS_MANIFEST.json") != config["mapping_manifest_sha256"]:
        raise RuntimeError("mapping manifest differs")
    complete = json.loads((root / "MAPPINGS_COMPLETE.json").read_text())
    if complete.get("status") != "complete" or complete.get("manifest_sha256") != config["mapping_manifest_sha256"]:
        raise RuntimeError("mapping completion binding differs")
    verify_manifest_products(root, "MAPPINGS_MANIFEST.json")
    return parent


def run(config: dict[str, Any], output: Path) -> dict[str, Any]:
    started = time.perf_counter()
    parent = verify_inputs(config)
    if output.exists() or output.with_name(output.name + ".partial").exists():
        raise FileExistsError(output)
    partial = output.with_name(output.name + ".partial")
    partial.mkdir(parents=True)
    root = Path(config["mapping_root"])
    fs = float(parent["sampling_frequency_hz"])
    start, stop = parent["crop"]["start_frame"], parent["crop"]["stop_frame"]
    cells = load_field(parent)
    state_indices = np.flatnonzero(cells.states_um == float(config["state_um"]))
    if len(state_indices) != 1:
        raise RuntimeError("expected exactly one requested state")
    state_index = int(state_indices[0])
    n_blocks = len(cells.block_edges_s) - 1
    events = {arm: load_events(Path(parent["arms"][arm]["sorter_output"]), start, stop) for arm in ARMS}
    assigned = {arm: state_assignments(events[arm], cells, fs) for arm in ARMS}
    intersection = set(pd.read_csv(root / "Q0_FOUR_WAY_INTERSECTION.csv").reference_unit.astype(int))
    summaries = []
    for candidate in COMPARATORS:
        ref_state, ref_segment, ref_block = assigned["REF384"]
        cand_state, cand_segment, cand_block = assigned[candidate]
        ref_group = ref_segment.astype(np.int64) * n_blocks + ref_block.astype(np.int64)
        cand_group = cand_segment.astype(np.int64) * n_blocks + cand_block.astype(np.int64)
        edges = q0_graph(
            events["REF384"], events[candidate], ref_state, cand_state, ref_group, cand_group,
            state_index, int(parent["tolerance_frames"]), float(parent["minimum_overlap"]),
            float(parent["primary_retention"]),
        )
        edges.to_csv(partial / f"Q40_GRAPH_REF384_vs_{candidate}.csv", index=False)
        ref_units, ref_counts = np.unique(events["REF384"]["clusters"][ref_state == state_index], return_counts=True)
        cand_units, cand_counts = np.unique(events[candidate]["clusters"][cand_state == state_index], return_counts=True)
        ref_count = dict(zip(map(int, ref_units), map(int, ref_counts)))
        cand_count = dict(zip(map(int, cand_units), map(int, cand_counts)))
        mapping = pd.read_csv(root / f"Q0_MAPPING_REF384_to_{candidate}.csv")
        mapping = mapping[mapping.has_primary].copy()
        classified = pd.DataFrame([
            {**row._asdict(), **classify_pair(pd.Series(row._asdict()), edges, ref_count, cand_count)}
            for row in mapping.itertuples(index=False)
        ])
        classified["comparison"] = f"REF384_vs_{candidate}"
        classified["in_four_way"] = classified.reference_unit.isin(intersection)
        classified.to_csv(partial / f"Q40_Q0_PAIR_RELATIONS_REF384_vs_{candidate}.csv", index=False)
        summaries.append(summarize(classified, f"REF384_vs_{candidate}", "pairwise"))
        summaries.append(summarize(classified[classified.in_four_way], f"REF384_vs_{candidate}", "four_way"))
    summary = pd.DataFrame(summaries)
    summary.to_csv(partial / "Q40_PARTNER_SWITCH_SUMMARY.csv", index=False)
    receipt = {
        "schema": SCHEMA, "status": "complete", "state_um": float(config["state_um"]),
        "source_sha256": sha256(Path(__file__)),
        "parent_config_sha256": config["parent_config_sha256"],
        "mapping_manifest_sha256": config["mapping_manifest_sha256"],
        "matching_rule": "same original segment and accounting block; 15 frames; overlap 0.1; reciprocal unique Jaccard; bilateral retention >=0.5",
        "secondary_matched_counts_summed": False,
        "interpretation": "Exploratory state-specific attribution only; partner relations do not establish biological identity or loss.",
        "raw_voltage_bytes_read": 0, "sorts_launched": 0, "rf_accesses": 0,
        "wall_seconds": time.perf_counter() - started,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    receipt["row_counts"] = {
        p.name: int(len(pd.read_csv(p))) for p in sorted(partial.glob("*.csv"))
    }
    atomic_json(partial / "RESULT.json", receipt)
    products = {p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in sorted(partial.iterdir())}
    atomic_json(partial / "MANIFEST.json", {"schema": SCHEMA, "products": products})
    atomic_json(partial / "COMPLETE.json", {"schema": SCHEMA, "status": "complete",
                "manifest_sha256": sha256(partial / "MANIFEST.json")})
    os.replace(partial, output)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if config.get("schema") != SCHEMA:
        raise RuntimeError("unexpected config schema")
    print(json.dumps(clean(run(config, args.output)), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
