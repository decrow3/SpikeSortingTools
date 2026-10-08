#!/usr/bin/env python3
"""Corrected saved-event correspondence with observation/null common support.

This is a saved-array measurement repair.  It never opens a recording.  REF
positions and states remain tied to original REF event times.  For comparison
shift ``s``, an A event is eligible only when both inclusive temporal query
windows, ``A +/- radius`` and ``A - s +/- radius``, stay in the same original
state segment.  Observed and null modes therefore use the identical A-event
denominator.  The REF denominator is the identical full finite-position REF
population in the original state for both modes.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import numpy as np
from numba import njit

RADIUS_SAMPLES = 15
SPATIAL_RADIUS_UM = 40.0
NULL_MS = (-100, -50, 50, 100)
CHUNK = 2_000_000


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(clean(value), indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict], fieldnames=None) -> None:
    if not rows and not fieldnames:
        raise RuntimeError(f"empty table requires explicit schema: {path}")
    names = list(rows[0]) if rows else list(fieldnames)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def rounded_shift_samples(milliseconds: int, fs: float) -> int:
    return int(np.rint(float(milliseconds) * float(fs) / 1000.0))


def coordinate_ref_y(saved_y, displacement_um):
    return np.asarray(saved_y) - np.asarray(displacement_um)


def verify_packet(root: Path, expected_manifest: str):
    manifest_path = root / "MANIFEST.json"
    complete_path = root / "COMPLETE.json"
    if sha(manifest_path) != expected_manifest:
        raise RuntimeError(f"manifest drift: {root}")
    manifest = json.loads(manifest_path.read_text())
    complete = json.loads(complete_path.read_text())
    if complete.get("status") != "complete" or complete.get("manifest_sha256") != expected_manifest:
        raise RuntimeError(f"seal drift: {root}")
    verified = {}
    for rel, meta in manifest["files"].items():
        path = root / rel
        if path.stat().st_size != meta["size_bytes"] or sha(path) != meta["sha256"]:
            raise RuntimeError(f"member drift: {path}")
        verified[str(path)] = meta["sha256"]
    config = json.loads((root / "effective_config.json").read_text())
    for name, item in config["frozen_inputs"].items():
        path = Path(item["path"])
        if sha(path) != item["sha256"]:
            raise RuntimeError(f"frozen input drift: {name}: {path}")
        verified[str(path)] = item["sha256"]
    return config, verified


def contexts(times, fs, cells):
    seconds = np.asarray(times, dtype=np.float64) / fs
    knot = np.searchsorted(cells.knot_edges_s, seconds, side="right") - 1
    if np.any(knot < 0) or np.any(knot >= len(cells.knot_state_index)):
        raise RuntimeError("event time is outside the frozen field support")
    return (np.asarray(cells.knot_state_index[knot], dtype=np.int64),
            np.asarray(cells.knot_segment_id[knot], dtype=np.int64))


@njit(cache=True)
def _event_segment_state(sample, fs, edges_s, knot_state, knot_segment):
    k = np.searchsorted(edges_s, sample / fs, side="right") - 1
    if k < 0 or k >= len(knot_state):
        return -1, -1
    return knot_segment[k], knot_state[k]


@njit(cache=True)
def _common_a_eligible(sample, expected_segment, expected_state, comparison_shift,
                       fs, edges_s, knot_state, knot_segment):
    centers = (sample, sample - comparison_shift)
    for center in centers:
        for endpoint in (center - RADIUS_SAMPLES, center + RADIUS_SAMPLES):
            segment, state = _event_segment_state(endpoint, fs, edges_s, knot_state, knot_segment)
            if segment != expected_segment or state != expected_state:
                return False
    return True


@njit(cache=True)
def _candidate_count(a_t, a_x, a_y, a_state, a_segment, r_t, r_pos, fs,
                     edges_s, knot_state, knot_segment, state_values,
                     match_shift, comparison_shift):
    degree = np.zeros(len(a_t), dtype=np.int64)
    total = 0
    radius2 = SPATIAL_RADIUS_UM * SPATIAL_RADIUS_UM
    for ai in range(len(a_t)):
        if (not np.isfinite(a_x[ai]) or not np.isfinite(a_y[ai]) or
                not _common_a_eligible(a_t[ai], a_segment[ai], a_state[ai],
                                       comparison_shift, fs, edges_s, knot_state, knot_segment)):
            degree[ai] = -1
            continue
        lo = np.searchsorted(r_t, a_t[ai] - match_shift - RADIUS_SAMPLES, side="left")
        hi = np.searchsorted(r_t, a_t[ai] - match_shift + RADIUS_SAMPLES, side="right")
        for ri in range(lo, hi):
            rseg, rstate = _event_segment_state(r_t[ri], fs, edges_s, knot_state, knot_segment)
            if rseg != a_segment[ai] or rstate != a_state[ai]:
                continue
            rx = r_pos[ri, 0]
            ry = r_pos[ri, 1] - state_values[rstate]
            if not np.isfinite(rx) or not np.isfinite(ry):
                continue
            dx = a_x[ai] - rx
            dy = a_y[ai] - ry
            if dx * dx + dy * dy <= radius2:
                degree[ai] += 1
                total += 1
    return degree, total


@njit(cache=True)
def _candidate_fill(a_t, a_x, a_y, a_state, a_segment, r_t, r_pos, fs,
                    edges_s, knot_state, knot_segment, state_values,
                    match_shift, comparison_shift, total):
    out_a = np.empty(total, dtype=np.int64)
    out_r = np.empty(total, dtype=np.int64)
    degree = np.zeros(len(a_t), dtype=np.int64)
    radius2 = SPATIAL_RADIUS_UM * SPATIAL_RADIUS_UM
    cursor = 0
    for ai in range(len(a_t)):
        if (not np.isfinite(a_x[ai]) or not np.isfinite(a_y[ai]) or
                not _common_a_eligible(a_t[ai], a_segment[ai], a_state[ai],
                                       comparison_shift, fs, edges_s, knot_state, knot_segment)):
            degree[ai] = -1
            continue
        lo = np.searchsorted(r_t, a_t[ai] - match_shift - RADIUS_SAMPLES, side="left")
        hi = np.searchsorted(r_t, a_t[ai] - match_shift + RADIUS_SAMPLES, side="right")
        for ri in range(lo, hi):
            rseg, rstate = _event_segment_state(r_t[ri], fs, edges_s, knot_state, knot_segment)
            if rseg != a_segment[ai] or rstate != a_state[ai]:
                continue
            rx = r_pos[ri, 0]
            ry = r_pos[ri, 1] - state_values[rstate]
            if not np.isfinite(rx) or not np.isfinite(ry):
                continue
            dx = a_x[ai] - rx
            dy = a_y[ai] - ry
            if dx * dx + dy * dy <= radius2:
                degree[ai] += 1
                out_a[cursor] = ai
                out_r[cursor] = ri
                cursor += 1
    if cursor != total:
        raise RuntimeError("candidate count/fill disagreement")
    return degree, out_a, out_r


class MatchResult(NamedTuple):
    degree: np.ndarray
    candidate_a: np.ndarray
    candidate_r: np.ndarray
    match_shift_samples: int
    comparison_shift_samples: int


def run_match(a_t, a_pos, a_state, a_segment, r_t, r_pos, fs, cells,
              *, match_shift_samples, comparison_shift_samples):
    edges = np.asarray(cells.knot_edges_s, np.float64)
    ks = np.asarray(cells.knot_state_index, np.int64)
    kg = np.asarray(cells.knot_segment_id, np.int64)
    sv = np.asarray(cells.states_um, np.float64)
    common = (
        np.asarray(a_t, np.int64), np.asarray(a_pos[:, 0], np.float64),
        np.asarray(a_pos[:, 1], np.float64), np.asarray(a_state, np.int64),
        np.asarray(a_segment, np.int64), r_t, r_pos, float(fs), edges, ks, kg, sv,
        int(match_shift_samples), int(comparison_shift_samples),
    )
    degree, total = _candidate_count(*common)
    degree2, ca, cr = _candidate_fill(*common, int(total))
    if not np.array_equal(degree, degree2):
        raise RuntimeError("candidate count/fill degree disagreement")
    return MatchResult(degree2, ca, cr, int(match_shift_samples), int(comparison_shift_samples))


def load_selected_events(root: Path, selected: list[int], fs: float, cells):
    times = np.load(root / "spike_times.npy", mmap_mode="r").reshape(-1)
    clusters = np.load(root / "spike_clusters.npy", mmap_mode="r").reshape(-1)
    positions = np.load(root / "spike_positions.npy", mmap_mode="r")
    if not (len(times) == len(clusters) == len(positions)):
        raise RuntimeError("Arm A per-event arrays differ in length")
    lookup = np.full(int(max(np.max(clusters), max(selected))) + 1, -1, dtype=np.int64)
    for j, unit in enumerate(selected):
        lookup[unit] = j
    buckets = {unit: [[], []] for unit in selected}
    for start in range(0, len(times), CHUNK):
        stop = min(start + CHUNK, len(times))
        cc = np.asarray(clusters[start:stop], dtype=np.int64)
        mapped = np.full(len(cc), -1, dtype=np.int64)
        valid = (cc >= 0) & (cc < len(lookup))
        mapped[valid] = lookup[cc[valid]]
        for j, unit in enumerate(selected):
            hit = np.flatnonzero(mapped == j)
            if len(hit):
                buckets[unit][0].append(np.asarray(times[start:stop][hit], dtype=np.int64))
                buckets[unit][1].append(np.asarray(positions[start:stop][hit, :2], dtype=np.float64))
    result = {}
    for unit, (tt, pp) in buckets.items():
        t = np.concatenate(tt) if tt else np.empty(0, dtype=np.int64)
        p = np.concatenate(pp) if pp else np.empty((0, 2), dtype=np.float64)
        state, segment = contexts(t, fs, cells)
        result[unit] = (t, p, state, segment)
    return result


def full_ref_denominators(r_t, r_pos, fs, cells):
    totals = np.zeros(len(cells.states_um), dtype=np.int64)
    for start in range(0, len(r_t), CHUNK):
        stop = min(start + CHUNK, len(r_t))
        tt = np.asarray(r_t[start:stop], dtype=np.int64)
        pp = np.asarray(r_pos[start:stop, :2], dtype=np.float64)
        state, _ = contexts(tt, fs, cells)
        finite = np.all(np.isfinite(pp), axis=1)
        totals += np.bincount(state[finite], minlength=len(totals))
    return totals


def summarize_mode(probe, unit, mode, shift_ms, match, data, ref_clusters, ref_denoms, states):
    _, _, a_state, _ = data
    degree, candidate_a, candidate_r = match.degree, match.candidate_a, match.candidate_r
    rows, partners = [], []
    for si, state_um in enumerate(states):
        in_state = a_state == si
        eligible_a = in_state & (degree >= 0)
        covered = eligible_a & (degree > 0)
        multi = eligible_a & (degree > 1)
        edge_mask = a_state[candidate_a] == si if len(candidate_a) else np.zeros(0, bool)
        rr = candidate_r[edge_mask]
        partner = np.bincount(np.asarray(ref_clusters[rr], dtype=np.int64)) if len(rr) else np.zeros(0, np.int64)
        hits = np.flatnonzero(partner)
        top = int(hits[np.argmax(partner[hits])]) if len(hits) else None
        top_edges = int(partner[top]) if top is not None else 0
        a_denom = int(np.count_nonzero(eligible_a))
        row = {
            "probe": probe, "A_unit_id": unit, "state_um": float(state_um), "mode": mode,
            "comparison_null_shift_ms": int(shift_ms),
            "match_shift_samples": int(match.match_shift_samples),
            "comparison_shift_samples": int(match.comparison_shift_samples),
            "temporal_radius_samples": RADIUS_SAMPLES,
            "A_events_total_state": int(np.count_nonzero(in_state)), "A_events_eligible_common": a_denom,
            "REF_events_eligible_full_original_state": int(ref_denoms[si]),
            "A_events_candidate_covered": int(np.count_nonzero(covered)),
            "A_candidate_coverage": float(np.count_nonzero(covered) / a_denom) if a_denom else math.nan,
            "A_events_multi_candidate": int(np.count_nonzero(multi)),
            "A_multi_candidate_fraction": float(np.count_nonzero(multi) / a_denom) if a_denom else math.nan,
            "candidate_edges": int(len(rr)), "REF_events_candidate_covered": int(len(np.unique(rr))),
            "REF_candidate_coverage": float(len(np.unique(rr)) / ref_denoms[si]) if ref_denoms[si] else math.nan,
            "distinct_REF_partners": int(len(hits)), "top_REF_unit_id": top,
            "top_partner_edges": top_edges,
            "top_partner_edge_fraction": float(top_edges / len(rr)) if len(rr) else math.nan,
            "unique_assignment": "omitted_all_neighbor_semantics",
        }
        rows.append(row)
        for ref_unit in hits:
            partners.append({
                "probe": probe, "A_unit_id": unit, "state_um": float(state_um), "mode": mode,
                "comparison_null_shift_ms": int(shift_ms), "REF_unit_id": int(ref_unit),
                "candidate_edges": int(partner[ref_unit]), "edge_fraction": float(partner[ref_unit] / len(rr)),
            })
    return rows, partners


def fixture_cells(duration_samples=250, fs=1000.0):
    return SimpleNamespace(
        knot_edges_s=np.array([0.0, duration_samples / fs]),
        knot_state_index=np.array([0], dtype=np.int64),
        knot_segment_id=np.array([0], dtype=np.int64),
        states_um=np.array([0.0]),
    )


def production_path_fixtures():
    fs = 1000.0
    cells = fixture_cells(250, fs)
    dense_t = np.arange(0, 250, dtype=np.int64)
    dense_p = np.zeros((len(dense_t), 2), dtype=np.float64)
    results = []
    for shift in (-100, 100):
        if shift > 0:
            a_t = np.arange(116, 235, 7, dtype=np.int64)
        else:
            a_t = np.arange(15, 134, 7, dtype=np.int64)
        a_p = np.zeros((len(a_t), 2), dtype=np.float64)
        state = np.zeros(len(a_t), dtype=np.int64)
        segment = np.zeros(len(a_t), dtype=np.int64)
        observed = run_match(a_t, a_p, state, segment, dense_t, dense_p, fs, cells,
                             match_shift_samples=0, comparison_shift_samples=shift)
        null = run_match(a_t, a_p, state, segment, dense_t, dense_p, fs, cells,
                         match_shift_samples=shift, comparison_shift_samples=shift)
        if not np.array_equal(observed.degree >= 0, null.degree >= 0):
            raise RuntimeError("dense observation/null A opportunity differs")
        if not np.all(observed.degree > 0) or not np.all(null.degree > 0):
            raise RuntimeError("dense stationary null lost coverage")
        results.append({"fixture": "dense_stationary", "shift_samples": shift,
                        "eligible_A": int(np.count_nonzero(observed.degree >= 0)),
                        "observed_coverage": float(np.mean(observed.degree > 0)),
                        "null_coverage": float(np.mean(null.degree > 0))})
    sparse_t = np.arange(20, 241, 20, dtype=np.int64)
    sparse_p = np.zeros((len(sparse_t), 2), dtype=np.float64)
    for shift in (-100, 100):
        a_t = sparse_t.copy()
        a_p = sparse_p.copy()
        state = np.zeros(len(a_t), dtype=np.int64)
        segment = np.zeros(len(a_t), dtype=np.int64)
        observed = run_match(a_t, a_p, state, segment, sparse_t, sparse_p, fs, cells,
                             match_shift_samples=0, comparison_shift_samples=shift)
        null = run_match(a_t, a_p, state, segment, sparse_t, sparse_p, fs, cells,
                         match_shift_samples=shift, comparison_shift_samples=shift)
        keep = observed.degree >= 0
        if not np.array_equal(keep, null.degree >= 0) or not np.array_equal(observed.degree[keep], null.degree[keep]):
            raise RuntimeError("sparse stationary observation/null opportunity differs")
        results.append({"fixture": "sparse_stationary", "shift_samples": shift,
                        "eligible_A": int(np.count_nonzero(keep)),
                        "observed_edges": int(observed.degree[keep].sum()),
                        "null_edges": int(null.degree[keep].sum())})
    short = fixture_cells(120, fs)
    a_t = np.array([30, 60, 90], dtype=np.int64)
    zeros = np.zeros((3, 2), dtype=np.float64)
    state = np.zeros(3, dtype=np.int64)
    segment = np.zeros(3, dtype=np.int64)
    for shift in (-100, 100):
        observed = run_match(a_t, zeros, state, segment, a_t, zeros, fs, short,
                             match_shift_samples=0, comparison_shift_samples=shift)
        null = run_match(a_t, zeros, state, segment, a_t, zeros, fs, short,
                         match_shift_samples=shift, comparison_shift_samples=shift)
        if np.any(observed.degree >= 0) or np.any(null.degree >= 0):
            raise RuntimeError("short segment should have no common support")
    collision_cells = fixture_cells(500, fs)
    a_t = np.array([250], dtype=np.int64)
    a_p = np.array([[0.0, 0.0]])
    r_t = np.array([249, 250, 251], dtype=np.int64)
    r_p = np.array([[0.0, 0.0], [40.0, 0.0], [40.0001, 0.0]])
    collision = run_match(a_t, a_p, np.array([0]), np.array([0]), r_t, r_p, fs, collision_cells,
                          match_shift_samples=0, comparison_shift_samples=0)
    if collision.degree.tolist() != [2]:
        raise RuntimeError("inclusive radius/collision fixture failed")
    rates = {str(rate): {str(ms): rounded_shift_samples(ms, rate) for ms in NULL_MS}
             for rate in (29999.835983263598, 29999.759166666667)}
    if any(abs(value) not in (1500, 3000) for rr in rates.values() for value in rr.values()):
        raise RuntimeError("actual-rate rounding fixture failed")
    return {"status": "PASS", "production_path": True, "stationary_controls": results,
            "short_segment_zero_common_support": True, "inclusive_collision_edges": 2,
            "actual_rate_shift_samples": rates,
            "opportunity_definition": "identical common A events plus identical full original-state REF denominator"}


def main():
    from testing.full_session_r1_evaluation import load_field_cells

    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--contract-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    if args.output.exists():
        raise FileExistsError(args.output)
    if sha(args.contract) != args.contract_sha256:
        raise RuntimeError("contract argument hash mismatch")
    contract = json.loads(args.contract.read_text())
    if contract["status"] != "FROZEN_BEFORE_CORRECTED_SAVED_OUTCOMES":
        raise RuntimeError("contract is not frozen")
    if args.output.resolve() != Path(contract["output"]).resolve():
        raise RuntimeError("output differs from frozen contract")
    if sha(args.selection) != contract["inputs"]["selection_sha256"]:
        raise RuntimeError("selection drift")
    if sha(Path(__file__).resolve()) != contract["implementation"]["source_sha256"]:
        raise RuntimeError("executed source drift")
    for name, item in contract["documents"].items():
        if item.get("runtime_verify", False) and sha(Path(item["path"])) != item["sha256"]:
            raise RuntimeError(f"document drift: {name}")
    for item in contract["implementation"]["dependencies"]:
        if sha(Path(item["path"])) != item["sha256"]:
            raise RuntimeError(f"dependency drift: {item['path']}")
    args.output.mkdir(parents=False)
    write_json(args.output / "INTERPRETATION_FREEZE.json", contract["interpretation"])
    write_json(args.output / "FIXTURE_RECEIPT.json", production_path_fixtures())
    write_json(args.output / "PROGRESS.json", {"status": "started", "started_at": datetime.now(timezone.utc).isoformat()})

    selection = json.loads(args.selection.read_text())
    all_rows, all_partners, probe_summaries = [], [], {}
    verified_inputs = {str(args.selection): contract["inputs"]["selection_sha256"], str(args.contract): sha(args.contract)}
    for probe in ("imec0", "imec1"):
        spec = selection["probes"][probe]
        config, verified = verify_packet(Path(spec["packet"]), spec["manifest_sha256"])
        verified_inputs.update(verified)
        cells, reference, field_audit = load_field_cells(config)
        fs = float(config["sampling_frequency_hz"])
        arms = {name: Path(value["curated_output"]) for name, value in config["arms"].items()}
        geometry = np.load(arms["A"] / "channel_positions.npy")
        expected_geometry = contract["inputs"]["geometry_sha256"]
        if sha(arms["A"] / "channel_positions.npy") != expected_geometry or sha(arms["REF"] / "channel_positions.npy") != expected_geometry:
            raise RuntimeError("geometry drift")
        selected = sorted({int(row["unit_id"]) for row in spec["units"] if row["arm"] == "A"})
        a_events = load_selected_events(arms["A"], selected, fs, cells)
        r_t = np.load(arms["REF"] / "spike_times.npy", mmap_mode="r").reshape(-1)
        r_c = np.load(arms["REF"] / "spike_clusters.npy", mmap_mode="r").reshape(-1)
        r_p = np.load(arms["REF"] / "spike_positions.npy", mmap_mode="r")
        if not (len(r_t) == len(r_c) == len(r_p)):
            raise RuntimeError("REF per-event arrays differ in length")
        ref_denoms = full_ref_denominators(r_t, r_p, fs, cells)
        shifts = {ms: rounded_shift_samples(ms, fs) for ms in NULL_MS}
        probe_summaries[probe] = {"sampling_frequency_hz": fs, "selected_A_units": selected,
                                  "field_reference_median_um": reference, "field_audit": field_audit,
                                  "null_shift_samples": shifts,
                                  "REF_finite_position_events_full_original_state": ref_denoms.tolist()}
        for unit in selected:
            data = a_events[unit]
            for ms, shift in shifts.items():
                observed = run_match(*data, r_t, r_p, fs, cells,
                                     match_shift_samples=0, comparison_shift_samples=shift)
                null = run_match(*data, r_t, r_p, fs, cells,
                                 match_shift_samples=shift, comparison_shift_samples=shift)
                if not np.array_equal(observed.degree >= 0, null.degree >= 0):
                    raise RuntimeError("observation/null A opportunity mismatch")
                for mode, result in (("observed_common_support", observed), ("null_common_support", null)):
                    rows, partners = summarize_mode(probe, unit, mode, ms, result, data, r_c, ref_denoms, cells.states_um)
                    all_rows.extend(rows); all_partners.extend(partners)
        write_json(args.output / "PROGRESS.json", {"status": "probe_complete", "probe": probe,
                                                    "rows": len(all_rows), "updated_at": datetime.now(timezone.utc).isoformat()})

    observed, null = {}, {}
    for row in all_rows:
        key = (row["probe"], row["A_unit_id"], row["state_um"], row["comparison_null_shift_ms"])
        (observed if row["mode"] == "observed_common_support" else null)[key] = row
    if set(observed) != set(null):
        raise RuntimeError("observation/null row keys differ")
    excess, opportunity = [], []
    for key in sorted(observed):
        o, n = observed[key], null[key]
        equal = (o["A_events_eligible_common"] == n["A_events_eligible_common"] and
                 o["REF_events_eligible_full_original_state"] == n["REF_events_eligible_full_original_state"])
        if not equal:
            raise RuntimeError("observation/null denominator mismatch")
        def delta(name):
            return o[name] - n[name] if math.isfinite(o[name]) and math.isfinite(n[name]) else math.nan
        excess.append({"probe": key[0], "A_unit_id": key[1], "state_um": key[2], "null_shift_ms": key[3],
                       "A_events_eligible_common": o["A_events_eligible_common"],
                       "REF_events_eligible_full_original_state": o["REF_events_eligible_full_original_state"],
                       "observed_A_candidate_coverage": o["A_candidate_coverage"],
                       "null_A_candidate_coverage": n["A_candidate_coverage"],
                       "coverage_excess": delta("A_candidate_coverage"),
                       "observed_REF_candidate_coverage": o["REF_candidate_coverage"],
                       "null_REF_candidate_coverage": n["REF_candidate_coverage"],
                       "REF_coverage_excess": delta("REF_candidate_coverage"),
                       "observed_candidate_edges": o["candidate_edges"], "null_candidate_edges": n["candidate_edges"],
                       "candidate_edge_excess": o["candidate_edges"] - n["candidate_edges"],
                       "opportunity_equal": True, "unique_assignment": "omitted_all_neighbor_semantics"})
        opportunity.append({"probe": key[0], "A_unit_id": key[1], "state_um": key[2], "null_shift_ms": key[3],
                            "A_observed_denominator": o["A_events_eligible_common"],
                            "A_null_denominator": n["A_events_eligible_common"],
                            "REF_observed_denominator": o["REF_events_eligible_full_original_state"],
                            "REF_null_denominator": n["REF_events_eligible_full_original_state"],
                            "A_denominator_equal": True, "REF_denominator_equal": True,
                            "support_definition": "common A query support with +/-15 allowance; full original-state REF"})
    grouped = defaultdict(list)
    for row in excess:
        grouped[(row["probe"], row["A_unit_id"], row["state_um"])].append(row)
    headline = {}
    for probe in ("imec0", "imec1"):
        evaluable = [rows for key, rows in grouped.items() if key[0] == probe and len(rows) == 4 and
                     all(math.isfinite(row["coverage_excess"]) for row in rows)]
        above = sum(all(row["coverage_excess"] > 0 and row["candidate_edge_excess"] > 0 for row in rows)
                    for rows in evaluable)
        headline[probe] = {"above_all_four_nulls": above, "evaluable_unit_states": len(evaluable)}
    write_csv(args.output / "PER_UNIT_STATE_MODE.csv", all_rows)
    write_csv(args.output / "PARTNER_EDGES.csv", all_partners, fieldnames=[
        "probe", "A_unit_id", "state_um", "mode", "comparison_null_shift_ms",
        "REF_unit_id", "candidate_edges", "edge_fraction"])
    write_csv(args.output / "NULL_EXCESS.csv", excess)
    write_csv(args.output / "OPPORTUNITY_AUDIT.csv", opportunity)
    write_json(args.output / "SUMMARY.json", {"schema": "corrected-correspondence-summary-v1",
                                               "headline": headline, "probes": probe_summaries,
                                               "interpretation": contract["interpretation"]})
    write_json(args.output / "PROVENANCE.json", {
        "executed_source": str(Path(__file__).resolve()), "executed_source_sha256": sha(Path(__file__).resolve()),
        "source_base_commit": contract["implementation"]["source_base_commit"],
        "selection": str(args.selection), "contract": str(args.contract), "verified_inputs": verified_inputs,
        "environment": {"python": sys.version, "executable": sys.executable, "numpy": np.__version__, "platform": platform.platform()},
        "recording_opened": False, "voltage_bytes_read": 0, "gpu_used": False,
        "unique_assignment": "omitted; all-neighbor statistic only",
        "elapsed_seconds_before_sealing": time.monotonic() - started,
    })
    members = {p.relative_to(args.output).as_posix(): {"sha256": sha(p), "size_bytes": p.stat().st_size}
               for p in sorted(args.output.rglob("*")) if p.is_file() and p.name not in {"MANIFEST.json", "COMPLETE.json"}}
    write_json(args.output / "MANIFEST.json", {"files": members})
    partial = args.output / "COMPLETE.json.partial"
    write_json(partial, {"status": "complete_corrected_common_support_correspondence",
                         "manifest_sha256": sha(args.output / "MANIFEST.json"),
                         "completed_at": datetime.now(timezone.utc).isoformat()})
    os.replace(partial, args.output / "COMPLETE.json")


if __name__ == "__main__":
    main()
