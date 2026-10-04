#!/usr/bin/env python3
"""Two-phase q0-anchored diagnostic for the completed EN crop sorts.

Phase ``mappings`` reads only q0 event membership/counts after assigning the
frozen field and writes the complete all-unit q0 edge graphs plus frozen anchor
mappings. Phase ``diagnostics`` requires that immutable mapping receipt before
computing any nonzero-state endpoint. No voltage, RF, holdout, bootstrap, or
pass/fail threshold is used.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd

from testing.en_common_support_crop_panel import crop_field_cells_from_global_midpoints
from testing.en_rounded_field import rounded_rigid_field
from testing.en_tier1_panel import correspondence, safe_spearman, sha256


ARMS = ("REF384", "B384", "REF368", "B368")
COMPARATORS = ("REF368", "B384", "B368")
SCHEMA = "en-q0-anchored-diagnostic-v1"
EDGE_COLUMNS = ["reference_cluster", "candidate_cluster", "matched_events", "reference_events",
                "candidate_events", "reference_retention", "candidate_retention", "f1", "jaccard",
                "primary_match"]


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(clean(payload), indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def config_digest(config: dict[str, Any]) -> str:
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def verify_frozen_inputs(config: dict[str, Any]) -> None:
    source = config["source"]
    checks = ((Path(__file__), source["sha256"]),
              (Path(source["tier1_helper_path"]), source["tier1_helper_sha256"]),
              (Path(source["crop_field_helper_path"]), source["crop_field_helper_sha256"]),
              (Path(source["rounded_field_helper_path"]), source["rounded_field_helper_sha256"]))
    for path, expected in checks:
        if sha256(path) != expected:
            raise RuntimeError(f"frozen source differs: {path}")
    for arm, spec in config["arms"].items():
        root = Path(spec["sorter_output"])
        files = ((root / "spike_times.npy", spec["spike_times_sha256"]),
                 (root / "spike_clusters.npy", spec["spike_clusters_sha256"]),
                 (root.parent / "RECEIPT.json", spec["receipt_sha256"]),
                 (Path(spec["population_membership_path"]), spec["population_membership_sha256"]),
                 (Path(spec["unit_table_path"]), spec["unit_table_sha256"]))
        for path, expected in files:
            if sha256(path) != expected:
                raise RuntimeError(f"{arm}: frozen input differs: {path}")


def verify_manifest_products(root: Path, manifest_name: str) -> None:
    manifest = json.loads((root / manifest_name).read_text())
    for name, expected in manifest["products"].items():
        path = root / name
        if not path.is_file() or path.stat().st_size != expected["bytes"] or sha256(path) != expected["sha256"]:
            raise RuntimeError(f"mapping product differs: {name}")


def load_events(root: Path, start: int, stop: int) -> dict[str, np.ndarray]:
    times = np.asarray(np.load(root / "spike_times.npy", mmap_mode="r"), dtype=np.int64).reshape(-1)
    clusters = np.asarray(np.load(root / "spike_clusters.npy", mmap_mode="r"), dtype=np.int64).reshape(-1)
    if len(times) != len(clusters) or (len(times) and np.any(np.diff(times) < 0)):
        raise RuntimeError(f"unaligned or unsorted events: {root}")
    if len(times) and (times[0] < start or times[-1] >= stop):
        raise RuntimeError(f"event outside frozen crop: {root}")
    return {"times": times - np.int64(start), "clusters": clusters}


def load_field(config: dict[str, Any]) -> Any:
    field = config["field"]
    path = Path(field["path"])
    if sha256(path) != field["sha256"]:
        raise RuntimeError("field hash differs")
    with np.load(path, allow_pickle=False) as saved:
        displacement = np.asarray(saved[field["displacement_key"]], dtype=float)
        if displacement.ndim == 2:
            displacement = np.median(displacement, axis=1)
        rounded, _ = rounded_rigid_field(displacement)
        return crop_field_cells_from_global_midpoints(
            np.asarray(saved[field["time_key"]], dtype=float), rounded,
            float(config["recording_duration_s"]),
            config["crop"]["start_frame"] / config["sampling_frequency_hz"],
            config["crop"]["stop_frame"] / config["sampling_frequency_hz"],
            float(config["accounting_block_seconds"]),
        )


def state_assignments(events: dict[str, np.ndarray], cells: Any, fs: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    state, segment = cells.assign(events["times"], fs)
    block = np.searchsorted(cells.block_edges_s, events["times"] / fs, side="right") - 1
    block = np.clip(block, 0, len(cells.block_edges_s) - 2).astype(np.int16)
    return state, segment, block


def exclusive_same_segment_pairs(
    a_times: np.ndarray, a_segments: np.ndarray, b_times: np.ndarray, b_segments: np.ndarray, tolerance: int
) -> tuple[np.ndarray, np.ndarray]:
    """Chronological one-to-one matching, with both events in the same original segment."""
    a_match, b_match = [], []
    common = np.intersect1d(np.unique(a_segments), np.unique(b_segments), assume_unique=False)
    for segment in common:
        ai = np.flatnonzero(a_segments == segment)
        bi = np.flatnonzero(b_segments == segment)
        a = np.asarray(a_times[ai], dtype=np.int64)
        b = np.asarray(b_times[bi], dtype=np.int64)
        i = j = 0
        while i < len(a) and j < len(b):
            if a[i] < b[j] - tolerance:
                i += 1
            elif b[j] < a[i] - tolerance:
                j += 1
            else:
                a_match.append(int(ai[i]))
                b_match.append(int(bi[j]))
                i += 1
                j += 1
    return np.asarray(a_match, dtype=np.int64), np.asarray(b_match, dtype=np.int64)


def exclusive_same_segment_count(
    a_times: np.ndarray, a_segments: np.ndarray, b_times: np.ndarray, b_segments: np.ndarray, tolerance: int
) -> int:
    return int(len(exclusive_same_segment_pairs(a_times, a_segments, b_times, b_segments, tolerance)[0]))


def full_unit_state_tables(events: dict[str, np.ndarray], cells: Any, fs: float) -> dict[str, Any]:
    """Construct adjacency on full unit trains before any state or cohort filtering."""
    state, segment, block = state_assignments(events, cells, fs)
    units, inverse, unit_counts = np.unique(events["clusters"], return_inverse=True, return_counts=True)
    n_states, n_blocks = len(cells.states_um), len(cells.block_edges_s) - 1
    counts = np.zeros((len(units), n_states), dtype=np.int64)
    np.add.at(counts, (inverse, state), 1)
    unit_block_counts = np.zeros((len(units), n_blocks, n_states), dtype=np.int64)
    np.add.at(unit_block_counts, (inverse, block, state), 1)
    block_counts = np.zeros((n_blocks, n_states), dtype=np.int64)
    np.add.at(block_counts, (block, state), 1)
    unit_order = np.argsort(inverse, kind="stable")
    unit_event_indices = np.split(unit_order, np.cumsum(unit_counts)[:-1])
    denominator = np.zeros((len(units), n_states), dtype=np.int64)
    short = np.zeros_like(denominator)
    unit_block_denominator = np.zeros((len(units), n_blocks, n_states), dtype=np.int64)
    unit_block_short = np.zeros_like(unit_block_denominator)
    block_denominator = np.zeros((n_blocks, n_states), dtype=np.int64)
    block_short = np.zeros_like(block_denominator)
    order = np.lexsort((events["times"], inverse))
    u, t, s, g, b = inverse[order], events["times"][order], state[order], segment[order], block[order]
    safe = (u[1:] == u[:-1]) & (g[1:] == g[:-1])
    delta = np.diff(t)
    positive = safe & (delta > 0)
    is_short = safe & (delta >= 1) & (delta < 30)
    np.add.at(denominator, (u[1:][positive], s[1:][positive]), 1)
    np.add.at(short, (u[1:][is_short], s[1:][is_short]), 1)
    np.add.at(unit_block_denominator, (u[1:][positive], b[1:][positive], s[1:][positive]), 1)
    np.add.at(unit_block_short, (u[1:][is_short], b[1:][is_short], s[1:][is_short]), 1)
    np.add.at(block_denominator, (b[1:][positive], s[1:][positive]), 1)
    np.add.at(block_short, (b[1:][is_short], s[1:][is_short]), 1)
    return {"units": units, "inverse": inverse, "unit_event_indices": unit_event_indices,
            "state": state, "segment": segment, "block": block,
            "counts": counts, "unit_block_counts": unit_block_counts, "block_counts": block_counts,
            "denominator": denominator, "short": short,
            "unit_block_denominator": unit_block_denominator, "unit_block_short": unit_block_short,
            "block_denominator": block_denominator, "block_short": block_short}


def q0_graph(reference: dict[str, np.ndarray], candidate: dict[str, np.ndarray],
             ref_state: np.ndarray, cand_state: np.ndarray,
             ref_segment: np.ndarray, cand_segment: np.ndarray, q0_index: int,
             tolerance: int, minimum_overlap: float, primary_retention: float) -> pd.DataFrame:
    # The full q0 graph is built with every unit before anchor eligibility is applied.
    ref_keep, cand_keep = ref_state == q0_index, cand_state == q0_index
    stride = max(int(reference["times"][-1]) if len(reference["times"]) else 0,
                 int(candidate["times"][-1]) if len(candidate["times"]) else 0) + 2 * tolerance + 1
    ref_lifted = reference["times"][ref_keep] + ref_segment[ref_keep].astype(np.int64) * stride
    cand_lifted = candidate["times"][cand_keep] + cand_segment[cand_keep].astype(np.int64) * stride
    edges = correspondence(
        {"times": ref_lifted, "clusters": reference["clusters"][ref_keep]},
        {"times": cand_lifted, "clusters": candidate["clusters"][cand_keep]},
        tolerance_frames=tolerance, minimum_overlap=minimum_overlap,
        primary_retention=primary_retention,
    )
    for column in EDGE_COLUMNS:
        if column not in edges:
            edges[column] = pd.Series(dtype=bool if column == "primary_match" else float)
    return edges[EDGE_COLUMNS]


def q0_anchor_counts(clusters: np.ndarray, state: np.ndarray, q0_index: int,
                     minimum_q0_events: int) -> pd.DataFrame:
    """Return every observed unit, including units with zero q0 events."""
    all_units = pd.DataFrame({"reference_unit": np.unique(clusters)})
    q0_units, q0_counts = np.unique(clusters[state == q0_index], return_counts=True)
    q0 = pd.DataFrame({"reference_unit": q0_units, "q0_events": q0_counts})
    result = all_units.merge(q0, on="reference_unit", how="left", validate="one_to_one")
    result["q0_events"] = result.q0_events.fillna(0).astype(np.int64)
    result["eligible_anchor"] = result.q0_events >= minimum_q0_events
    return result


def assert_unit_event_partition(table: dict[str, Any], buckets: dict[str, np.ndarray], label: str) -> None:
    """Require disjoint unit buckets to cover every unit and every block/state event."""
    parts = [np.asarray(ids, dtype=np.int64) for ids in buckets.values()]
    combined = np.concatenate(parts) if parts else np.array([], dtype=np.int64)
    if len(np.unique(combined)) != len(combined) or not np.array_equal(np.sort(combined), table["units"]):
        raise RuntimeError(f"{label}: unit buckets do not form a partition")
    accounted = np.zeros_like(table["block_counts"])
    for ids in parts:
        indices = selected_indices(table, ids) if len(ids) else np.array([], dtype=int)
        accounted += table["unit_block_counts"][indices].sum(axis=0)
    if not np.array_equal(accounted, table["block_counts"]):
        raise RuntimeError(f"{label}: bucket event accounting does not partition block/state totals")


def mapping_phase(config: dict[str, Any], output: Path) -> dict[str, Any]:
    started = time.perf_counter()
    verify_frozen_inputs(config)
    if output.exists() or output.with_name(output.name + ".partial").exists():
        raise FileExistsError(output)
    partial = output.with_name(output.name + ".partial")
    partial.mkdir(parents=True)
    fs = float(config["sampling_frequency_hz"])
    start, stop = config["crop"]["start_frame"], config["crop"]["stop_frame"]
    cells = load_field(config)
    q0 = np.flatnonzero(cells.states_um == 0.0)
    if len(q0) != 1:
        raise RuntimeError("expected exactly one q0 state")
    events = {arm: load_events(Path(config["arms"][arm]["sorter_output"]), start, stop) for arm in ARMS}
    assignments = {arm: state_assignments(events[arm], cells, fs) for arm in ARMS}
    states = {arm: assignments[arm][0] for arm in ARMS}
    anchor_table = q0_anchor_counts(events["REF384"]["clusters"], states["REF384"], int(q0[0]),
                                    int(config["minimum_q0_events"]))
    ref_membership = pd.read_csv(config["arms"]["REF384"]["population_membership_path"])
    ref_units = pd.read_csv(config["arms"]["REF384"]["unit_table_path"])
    anchor_table = anchor_table.merge(
        ref_membership[["unit_id", "population", "common_unit"]].rename(columns={"unit_id": "reference_unit"}),
        on="reference_unit", how="left", validate="one_to_one")
    anchor_table = anchor_table.merge(
        ref_units[["unit_id", "edge_unit"]].rename(columns={"unit_id": "reference_unit"}),
        on="reference_unit", how="left", validate="one_to_one")
    anchor_table.to_csv(partial / "REF384_Q0_ANCHORS.csv", index=False)
    eligible = set(anchor_table.loc[anchor_table.eligible_anchor, "reference_unit"].astype(int))
    pair_summaries, mapping_tables = {}, {}
    for candidate in COMPARATORS:
        edges = q0_graph(events["REF384"], events[candidate], states["REF384"], states[candidate],
                         assignments["REF384"][1], assignments[candidate][1], int(q0[0]),
                         int(config["tolerance_frames"]), float(config["minimum_overlap"]),
                         float(config["primary_retention"]))
        edges.to_csv(partial / f"Q0_GRAPH_REF384_vs_{candidate}.csv", index=False)
        primary = edges[edges.primary_match].copy() if len(edges) else edges.copy()
        primary = primary[primary.reference_cluster.isin(eligible)] if len(primary) else primary
        degrees = edges.groupby("reference_cluster").size().rename("secondary_degree") if len(edges) else pd.Series(dtype=int)
        rows = anchor_table[anchor_table.eligible_anchor].rename(columns={"reference_unit": "reference_cluster"}).copy()
        rows = rows.merge(primary, on="reference_cluster", how="left", validate="one_to_one")
        candidate_membership = pd.read_csv(config["arms"][candidate]["population_membership_path"])
        candidate_units = pd.read_csv(config["arms"][candidate]["unit_table_path"])
        rows = rows.merge(candidate_membership[["unit_id", "population", "common_unit"]].rename(columns={
            "unit_id": "candidate_cluster", "population": "candidate_population",
            "common_unit": "candidate_common_unit"}), on="candidate_cluster", how="left", validate="many_to_one")
        rows = rows.merge(candidate_units[["unit_id", "edge_unit"]].rename(columns={
            "unit_id": "candidate_cluster", "edge_unit": "candidate_edge_unit"}),
            on="candidate_cluster", how="left", validate="many_to_one")
        rows["secondary_degree"] = rows.reference_cluster.map(degrees).fillna(0).astype(int)
        rows["has_primary"] = rows.candidate_cluster.notna()
        rows.to_csv(partial / f"Q0_MAPPING_REF384_to_{candidate}.csv", index=False)
        mapping_tables[candidate] = rows
        matched_mass = int(rows.loc[rows.has_primary, "q0_events"].sum())
        unmatched_mass = int(rows.loc[~rows.has_primary, "q0_events"].sum())
        candidate_total_mass = int(np.count_nonzero(states[candidate] == q0[0]))
        candidate_matched_mass = int(rows.loc[rows.has_primary, "candidate_events"].sum())
        pair_summaries[candidate] = {
            "eligible_anchors": int(len(rows)), "matched_anchors": int(rows.has_primary.sum()),
            "unmatched_anchors": int((~rows.has_primary).sum()),
            "matched_reference_q0_event_mass": matched_mass,
            "unmatched_reference_q0_event_mass": unmatched_mass,
            "matched_candidate_q0_event_mass": candidate_matched_mass,
            "candidate_q0_event_mass_outside_anchor_primaries": candidate_total_mass - candidate_matched_mass,
            "anchors_with_secondary_degree_gt1": int((rows.secondary_degree > 1).sum()),
            "note": "secondary degree >1 is preserved separately and does not invalidate a valid primary",
        }
    intersection = set(eligible)
    for candidate, table in mapping_tables.items():
        intersection &= set(table.loc[table.has_primary, "reference_cluster"].astype(int))
    intersection_table = anchor_table[anchor_table.reference_unit.isin(intersection)].copy()
    intersection_table.to_csv(partial / "Q0_FOUR_WAY_INTERSECTION.csv", index=False)
    receipt = {
        "schema": SCHEMA, "phase": "mappings", "status": "complete",
        "config_digest": config_digest(config), "source_sha256": sha256(Path(__file__)),
        "q0_state_um": 0.0, "full_graph_before_anchor_filter": True,
        "eligible_anchor_count": int(len(eligible)), "four_way_intersection_count": int(len(intersection)),
        "pairwise": pair_summaries, "nonzero_outcomes_read": False,
        "wall_seconds": time.perf_counter() - started,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(partial / "MAPPINGS.json", receipt)
    products = {p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in sorted(partial.iterdir())}
    atomic_json(partial / "MAPPINGS_MANIFEST.json", {"schema": SCHEMA, "products": products})
    atomic_json(partial / "MAPPINGS_COMPLETE.json", {"schema": SCHEMA, "status": "complete",
                "manifest_sha256": sha256(partial / "MAPPINGS_MANIFEST.json")})
    os.replace(partial, output)
    return receipt


def selected_indices(table: dict[str, Any], unit_ids: np.ndarray) -> np.ndarray:
    lookup = {int(unit): index for index, unit in enumerate(table["units"])}
    return np.asarray([lookup[int(unit)] for unit in unit_ids], dtype=np.int64)


def pair_diagnostics(reference: dict[str, np.ndarray], candidate: dict[str, np.ndarray],
                     ref_table: dict[str, Any], cand_table: dict[str, Any], mapping: pd.DataFrame,
                     cells: Any, tolerance: int, cohort: str) -> list[dict[str, Any]]:
    rows = []
    for edge in mapping.itertuples():
        for state_index, state_um in enumerate(cells.states_um):
            ref_id, cand_id = int(edge.reference_cluster), int(edge.candidate_cluster)
            ri = int(np.flatnonzero(ref_table["units"] == ref_id)[0])
            ci = int(np.flatnonzero(cand_table["units"] == cand_id)[0])
            ref_unit_events = ref_table["unit_event_indices"][ri]
            cand_unit_events = cand_table["unit_event_indices"][ci]
            ref_keep = ref_unit_events[ref_table["state"][ref_unit_events] == state_index]
            cand_keep = cand_unit_events[cand_table["state"][cand_unit_events] == state_index]
            n_blocks = len(cells.block_edges_s) - 1
            ref_group = ref_table["segment"][ref_keep].astype(np.int64) * n_blocks + ref_table["block"][ref_keep]
            cand_group = cand_table["segment"][cand_keep].astype(np.int64) * n_blocks + cand_table["block"][cand_keep]
            matched_ref, _ = exclusive_same_segment_pairs(
                reference["times"][ref_keep], ref_group,
                candidate["times"][cand_keep], cand_group, tolerance)
            matched_blocks = ref_table["block"][ref_keep[matched_ref]]
            for block in range(len(cells.block_edges_s) - 1):
                ref_spikes = int(ref_table["unit_block_counts"][ri, block, state_index])
                cand_spikes = int(cand_table["unit_block_counts"][ci, block, state_index])
                ref_den = int(ref_table["unit_block_denominator"][ri, block, state_index])
                cand_den = int(cand_table["unit_block_denominator"][ci, block, state_index])
                ref_short = int(ref_table["unit_block_short"][ri, block, state_index])
                cand_short = int(cand_table["unit_block_short"][ci, block, state_index])
                matched = int(np.count_nonzero(matched_blocks == block))
                rows.append({
                    "cohort": cohort, "reference_unit": ref_id, "candidate_unit": cand_id,
                    "block": block, "state_um": float(state_um), "reference_spikes": ref_spikes,
                    "candidate_spikes": cand_spikes, "exclusive_same_segment_matched_events": matched,
                    "reference_retention": matched / ref_spikes if ref_spikes else math.nan,
                    "candidate_retention": matched / cand_spikes if cand_spikes else math.nan,
                    "f1": 2 * matched / (ref_spikes + cand_spikes) if ref_spikes + cand_spikes else math.nan,
                    "reference_short_numerator": ref_short, "reference_short_denominator": ref_den,
                    "candidate_short_numerator": cand_short, "candidate_short_denominator": cand_den,
                })
    return rows


def summarize_pair_rows(details: pd.DataFrame) -> pd.DataFrame:
    keys = ["comparison", "cohort", "block", "state_um"]
    sums = ["reference_spikes", "candidate_spikes", "exclusive_same_segment_matched_events",
            "reference_short_numerator", "reference_short_denominator",
            "candidate_short_numerator", "candidate_short_denominator"]
    summary = details.groupby(keys, as_index=False)[sums].sum()
    sizes = details.groupby(keys).size().rename("mapped_pairs").reset_index()
    summary = summary.merge(sizes, on=keys, validate="one_to_one")
    matched = summary.exclusive_same_segment_matched_events
    summary["reference_retention"] = matched / summary.reference_spikes.replace(0, np.nan)
    summary["candidate_retention"] = matched / summary.candidate_spikes.replace(0, np.nan)
    summary["f1"] = 2 * matched / (summary.reference_spikes + summary.candidate_spikes).replace(0, np.nan)
    return summary


def diagnostics_phase(config: dict[str, Any], output: Path) -> dict[str, Any]:
    started = time.perf_counter()
    verify_frozen_inputs(config)
    complete = json.loads((output / "MAPPINGS_COMPLETE.json").read_text())
    if complete.get("status") != "complete":
        raise RuntimeError("mapping phase is not complete")
    manifest_hash = sha256(output / "MAPPINGS_MANIFEST.json")
    if complete.get("manifest_sha256") != manifest_hash:
        raise RuntimeError("mapping manifest differs")
    verify_manifest_products(output, "MAPPINGS_MANIFEST.json")
    mapping_receipt = json.loads((output / "MAPPINGS.json").read_text())
    if mapping_receipt.get("config_digest") != config_digest(config):
        raise RuntimeError("mapping config differs")
    diagnostics = output / "diagnostics"
    partial = output / "diagnostics.partial"
    if diagnostics.exists() or partial.exists():
        raise FileExistsError(diagnostics)
    partial.mkdir()
    fs = float(config["sampling_frequency_hz"])
    start, stop = config["crop"]["start_frame"], config["crop"]["stop_frame"]
    cells = load_field(config)
    events = {arm: load_events(Path(config["arms"][arm]["sorter_output"]), start, stop) for arm in ARMS}
    tables = {arm: full_unit_state_tables(events[arm], cells, fs) for arm in ARMS}
    all_population = []
    for arm in ARMS:
        table = tables[arm]
        for block in range(len(cells.block_edges_s) - 1):
            for state_index, state_um in enumerate(cells.states_um):
                all_population.append({"arm": arm, "block": block, "state_um": float(state_um),
                    "exposure_s": float(cells.block_exposure_s[block, state_index]),
                    "spikes": int(table["block_counts"][block, state_index]),
                    "short_numerator": int(table["block_short"][block, state_index]),
                    "short_denominator": int(table["block_denominator"][block, state_index])})
    pd.DataFrame(all_population).to_csv(partial / "ALL_POPULATION_BY_BLOCK_STATE.csv", index=False)
    all_population_frame = pd.DataFrame(all_population)
    all_population_frame.groupby(["arm", "state_um"], as_index=False)[
        ["exposure_s", "spikes", "short_numerator", "short_denominator"]
    ].sum().to_csv(partial / "ALL_POPULATION_STATE_TOTAL.csv", index=False)
    intersection = set(pd.read_csv(output / "Q0_FOUR_WAY_INTERSECTION.csv").reference_unit.astype(int))
    pair_rows, rho_rows, coverage_rows = [], [], []
    for candidate in COMPARATORS:
        raw_mapping = pd.read_csv(output / f"Q0_MAPPING_REF384_to_{candidate}.csv")
        mapping = raw_mapping[raw_mapping.has_primary].copy()
        anchors = pd.read_csv(output / "REF384_Q0_ANCHORS.csv")
        ref_buckets = {
            "mapped_eligible_ref_anchor": mapping.reference_cluster.astype(int).to_numpy(),
            "unmatched_eligible_ref_anchor": raw_mapping.loc[~raw_mapping.has_primary, "reference_cluster"].astype(int).to_numpy(),
            "q0_ineligible_ref": anchors.loc[~anchors.eligible_anchor, "reference_unit"].astype(int).to_numpy(),
        }
        mapped_candidates = mapping.candidate_cluster.astype(int).to_numpy()
        candidate_buckets = {
            "mapped_candidate": mapped_candidates,
            "candidate_outside_mapped_units": np.setdiff1d(tables[candidate]["units"], mapped_candidates),
        }
        assert_unit_event_partition(tables["REF384"], ref_buckets, f"REF384_vs_{candidate}:REF384")
        assert_unit_event_partition(tables[candidate], candidate_buckets, f"REF384_vs_{candidate}:{candidate}")
        for arm, buckets in (("REF384", ref_buckets), (candidate, candidate_buckets)):
            for bucket_name, ids in buckets.items():
                indices = selected_indices(tables[arm], ids) if len(ids) else np.array([], dtype=int)
                for block in range(len(cells.block_edges_s) - 1):
                    for state_index, state_um in enumerate(cells.states_um):
                        coverage_rows.append({"comparison": f"REF384_vs_{candidate}", "arm": arm,
                            "bucket": bucket_name, "block": block, "state_um": float(state_um),
                            "units": int(len(ids)),
                            "spikes": int(tables[arm]["unit_block_counts"][indices, block, state_index].sum())})
        for cohort, selected in (("pairwise", mapping),
                                 ("four_way", mapping[mapping.reference_cluster.isin(intersection)])):
            pair_rows.extend({"comparison": f"REF384_vs_{candidate}", **row} for row in pair_diagnostics(
                events["REF384"], events[candidate], tables["REF384"], tables[candidate], selected,
                cells, int(config["tolerance_frames"]), cohort))
            for arm, id_column in (("REF384", "reference_cluster"), (candidate, "candidate_cluster")):
                ids = selected[id_column].astype(int).to_numpy()
                indices = selected_indices(tables[arm], ids) if len(ids) else np.array([], dtype=int)
                zero = int(np.flatnonzero(cells.states_um == 0.0)[0])
                for state_index, state_um in enumerate(cells.states_um):
                    x = np.log((tables[arm]["counts"][indices, zero] + 0.5) / cells.state_exposure_s[zero])
                    exposure = cells.state_exposure_s[state_index]
                    y = (np.log((tables[arm]["counts"][indices, state_index] + 0.5) / exposure)
                         if exposure > 0 else np.full(len(indices), np.nan))
                    rho_rows.append({"comparison": f"REF384_vs_{candidate}", "cohort": cohort,
                        "arm": arm, "state_um": float(state_um), "fixed_units": int(len(ids)),
                        "rho_vs_q0": safe_spearman(x, y)})
    pd.DataFrame(coverage_rows).to_csv(partial / "STATEWISE_COVERAGE_BUCKETS.csv", index=False)
    pair_columns = ["comparison", "cohort", "reference_unit", "candidate_unit", "block", "state_um",
                    "reference_spikes", "candidate_spikes", "exclusive_same_segment_matched_events",
                    "reference_retention", "candidate_retention", "f1", "reference_short_numerator",
                    "reference_short_denominator", "candidate_short_numerator", "candidate_short_denominator"]
    pair_frame = pd.DataFrame(pair_rows, columns=pair_columns)
    pair_frame.to_csv(partial / "FIXED_PAIR_STATE_DETAILS.csv", index=False)
    block_summary = summarize_pair_rows(pair_frame) if len(pair_frame) else pd.DataFrame(columns=[
        "comparison", "cohort", "block", "state_um", "reference_spikes", "candidate_spikes",
        "exclusive_same_segment_matched_events", "reference_short_numerator", "reference_short_denominator",
        "candidate_short_numerator", "candidate_short_denominator", "mapped_pairs",
        "reference_retention", "candidate_retention", "f1"])
    block_summary.to_csv(partial / "FIXED_COHORT_BLOCK_STATE_SUMMARY.csv", index=False)
    total_sums = ["reference_spikes", "candidate_spikes", "exclusive_same_segment_matched_events",
                  "reference_short_numerator", "reference_short_denominator",
                  "candidate_short_numerator", "candidate_short_denominator"]
    total = block_summary.groupby(["comparison", "cohort", "state_um"], as_index=False)[total_sums].sum()
    pairs = block_summary.groupby(["comparison", "cohort", "state_um"], as_index=False).mapped_pairs.max()
    total = total.merge(pairs, on=["comparison", "cohort", "state_um"], validate="one_to_one")
    matched = total.exclusive_same_segment_matched_events
    total["reference_retention"] = matched / total.reference_spikes.replace(0, np.nan)
    total["candidate_retention"] = matched / total.candidate_spikes.replace(0, np.nan)
    total["f1"] = 2 * matched / (total.reference_spikes + total.candidate_spikes).replace(0, np.nan)
    total.to_csv(partial / "FIXED_COHORT_STATE_TOTAL.csv", index=False)
    pd.DataFrame(rho_rows).to_csv(partial / "FIXED_COHORT_ARM_RHO.csv", index=False)
    receipt = {"schema": SCHEMA, "phase": "diagnostics", "status": "complete",
               "mapping_manifest_sha256": manifest_hash, "config_digest": config_digest(config),
               "source_sha256": sha256(Path(__file__)), "bootstrap_used": False,
               "interpretation": "REF is a comparator, not ground truth; disagreement is not biological loss.",
               "raw_voltage_bytes_read": 0, "sorts_launched": 0, "rf_accesses": 0,
               "row_counts": {"all_population_by_block_state": int(len(all_population_frame)),
                              "statewise_coverage_buckets": int(len(coverage_rows)),
                              "fixed_pair_state_details": int(len(pair_frame)),
                              "fixed_cohort_block_state_summary": int(len(block_summary)),
                              "fixed_cohort_state_total": int(len(total)),
                              "fixed_cohort_arm_rho": int(len(rho_rows))},
               "wall_seconds": time.perf_counter() - started,
               "completed_at": datetime.now(timezone.utc).isoformat()}
    atomic_json(partial / "RESULT.json", receipt)
    products = {p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in sorted(partial.iterdir())}
    atomic_json(partial / "MANIFEST.json", {"schema": SCHEMA, "products": products})
    atomic_json(partial / "COMPLETE.json", {"schema": SCHEMA, "status": "complete",
                "manifest_sha256": sha256(partial / "MANIFEST.json")})
    os.replace(partial, diagnostics)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("mappings", "diagnostics"), required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if config.get("schema") != SCHEMA:
        raise RuntimeError("unexpected config schema")
    result = mapping_phase(config, args.output) if args.phase == "mappings" else diagnostics_phase(config, args.output)
    print(json.dumps(clean(result), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
