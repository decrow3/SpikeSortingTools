#!/usr/bin/env python3
"""Saved-output diagnosis of EN state-specific support and interval burden.

This reads only Kilosort outputs, motion fields, and frozen metadata.  It never
opens a voltage binary.  Event positions and templates are deliberately named
as proxies because neither is a direct state-specific waveform observation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import time

import numpy as np
from numba import njit

from npx_preprocessing.motion.lattice_remap_si import exact_coordinate_mapping
from testing.en_rounded_field import rounded_rigid_field


SUPPORT_NAMES = ("supported", "unsupported", "ambiguous_row_tie", "invalid_position")
PAIR_NAMES = ("both_supported", "mixed", "both_unsupported", "ambiguous")
SIDE_NAMES = ("none", "lower", "upper", "interior", "multiple", "ambiguous")
DISTANCE_NAMES = ("lt20", "20_to_lt60", "60_to_lt140", "ge140", "no_switch", "invalid")
TEMPORAL_NAMES = ("lt0p25s", "0p25_to_lt1s", "1_to_lt5s", "ge5s", "no_state_change")
FOOTPRINT_NAMES = ("none_missing", "partial_missing", "all_missing", "invalid")


def sha256(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def atomic_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(clean(value), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing empty table: {path}")
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def field_cells(spec: dict, duration_s: float) -> dict:
    path = Path(spec["field_path"])
    actual = sha256(path)
    if actual != spec["field_sha256"]:
        raise RuntimeError(f"field hash mismatch: {path}")
    with np.load(path, allow_pickle=False) as saved:
        times = np.asarray(saved[spec["time_key"]], dtype=np.float64)
        displacement = np.asarray(saved[spec["displacement_key"]], dtype=np.float64)
    if displacement.ndim == 2:
        if spec.get("rigid_projection") != "median across depth":
            raise ValueError("2-D field lacks frozen median projection")
        displacement = np.median(displacement, axis=1)
    rounded, reference = rounded_rigid_field(displacement)
    if times.ndim != 1 or rounded.shape != times.shape or len(times) < 2:
        raise ValueError("invalid field vectors")
    if not np.all(np.diff(times) > 0):
        raise ValueError("field time must be strictly increasing")
    edges = np.empty(len(times) + 1, dtype=np.float64)
    edges[1:-1] = (times[:-1] + times[1:]) / 2
    edges[0], edges[-1] = 0.0, duration_s
    edges = np.clip(edges, 0.0, duration_s)
    states = np.unique(rounded)
    state_index = np.searchsorted(states, rounded).astype(np.int16)
    segment = np.cumsum(np.r_[True, rounded[1:] != rounded[:-1]]).astype(np.int32) - 1
    change_boundaries = edges[1:-1][rounded[1:] != rounded[:-1]]
    exposure = np.bincount(state_index, weights=np.diff(edges), minlength=len(states))
    if not np.isclose(exposure.sum(), duration_s, rtol=0, atol=1e-8):
        raise RuntimeError("state exposure does not close")
    return {
        "path": str(path), "sha256": actual, "times": times, "edges": edges,
        "states": states, "knot_state": state_index, "knot_segment": segment,
        "change_boundaries_s": change_boundaries,
        "exposure_s": exposure, "reference_median_um": float(reference),
    }


def temporal_change_bins(seconds: np.ndarray, change_boundaries_s: np.ndarray) -> np.ndarray:
    if len(change_boundaries_s) == 0:
        return np.full(len(seconds), 4, dtype=np.uint8)
    insertion = np.searchsorted(change_boundaries_s, seconds, side="left")
    left = change_boundaries_s[np.clip(insertion - 1, 0, len(change_boundaries_s) - 1)]
    right = change_boundaries_s[np.clip(insertion, 0, len(change_boundaries_s) - 1)]
    distance = np.minimum(np.abs(seconds - left), np.abs(seconds - right))
    return np.select(
        [distance < 0.25, distance < 1.0, distance < 5.0], [0, 1, 2], default=3
    ).astype(np.uint8)


def support_tables(geometry: np.ndarray, states: np.ndarray) -> dict:
    rows_y = np.unique(geometry[:, 1].astype(np.float64))
    row_support = np.zeros((len(states), len(rows_y)), dtype=np.uint8)
    row_side = np.zeros_like(row_support)
    switch_boundaries: list[np.ndarray] = []
    details = []
    for si, state in enumerate(states):
        mapping = exact_coordinate_mapping(geometry, float(state), direction="y")
        channel_support = mapping >= 0
        for ri, y in enumerate(rows_y):
            values = np.unique(channel_support[np.isclose(geometry[:, 1], y, rtol=0, atol=1e-6)])
            if len(values) != 1:
                raise RuntimeError(f"mixed within-row support at state {state}, y={y}")
            row_support[si, ri] = int(values[0])
        changes = np.flatnonzero(row_support[si, 1:] != row_support[si, :-1])
        boundaries = (rows_y[changes] + rows_y[changes + 1]) / 2
        switch_boundaries.append(boundaries)
        unsupported = np.flatnonzero(row_support[si] == 0)
        for ri in unsupported:
            if ri == 0 or (ri < len(rows_y) - 1 and not row_support[si, :ri].any()):
                side = 1
            elif ri == len(rows_y) - 1 or not row_support[si, ri + 1:].any():
                side = 2
            else:
                side = 3
            row_side[si, ri] = side
        details.append({
            "state_um": float(state), "supported_rows": int(row_support[si].sum()),
            "unsupported_rows": int((row_support[si] == 0).sum()),
            "switch_boundaries_um": boundaries.tolist(),
            "unsupported_y_um": rows_y[unsupported].tolist(),
        })
    max_switches = max(1, max(map(len, switch_boundaries)))
    boundary_matrix = np.full((len(states), max_switches), np.nan, dtype=np.float64)
    boundary_count = np.zeros(len(states), dtype=np.int16)
    for si, boundaries in enumerate(switch_boundaries):
        boundary_matrix[si, :len(boundaries)] = boundaries
        boundary_count[si] = len(boundaries)
    return {
        "rows_y": rows_y, "row_support": row_support, "row_side": row_side,
        "boundary_matrix": boundary_matrix, "boundary_count": boundary_count,
        "details": details,
    }


def assign_positions(y: np.ndarray, state: np.ndarray, support: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = support["rows_y"]
    insertion = np.searchsorted(rows, y, side="left")
    right = np.clip(insertion, 0, len(rows) - 1)
    left = np.clip(insertion - 1, 0, len(rows) - 1)
    dl = np.abs(y - rows[left])
    dr = np.abs(y - rows[right])
    choose_right = dr < dl
    nearest = np.where(choose_right, right, left)
    tie = np.isfinite(y) & (left != right) & np.isclose(dl, dr, rtol=0, atol=1e-6)
    invalid = ~np.isfinite(y)
    status = np.where(support["row_support"][state, nearest] != 0, 0, 1).astype(np.uint8)
    status[tie] = 2
    status[invalid] = 3
    side = support["row_side"][state, nearest].astype(np.uint8)
    side[status == 0] = 0
    side[tie | invalid] = 5
    distance_bin = np.full(len(y), 4, dtype=np.uint8)
    for si in np.unique(state):
        mask = state == si
        count = int(support["boundary_count"][si])
        if count:
            boundaries = support["boundary_matrix"][si, :count]
            distance = np.min(np.abs(y[mask, None] - boundaries[None, :]), axis=1)
            distance_bin[mask] = np.select(
                [distance < 20, distance < 60, distance < 140], [0, 1, 2], default=3
            ).astype(np.uint8)
    distance_bin[invalid] = 5
    return status, side, distance_bin


@njit(cache=True)
def accumulate_pairs(
    times, units, states, segments, support_status, sides, distance_bins, temporal_bins,
    last_time, last_state, last_segment, last_support, last_side, last_distance, last_temporal,
    unit_denominator, unit_short, pair_denominator, pair_short, pair_exact,
):
    for i in range(len(times)):
        unit = units[i]
        if last_time[unit] >= 0 and last_segment[unit] == segments[i]:
            delta = times[i] - last_time[unit]
            a, b = last_support[unit], support_status[i]
            if a >= 2 or b >= 2:
                pair_class = 3
            elif a == 0 and b == 0:
                pair_class = 0
            elif a == 1 and b == 1:
                pair_class = 2
            else:
                pair_class = 1
            sa, sb = last_side[unit], sides[i]
            if sa >= 5 or sb >= 5:
                pair_side = 5
            elif sa == 0:
                pair_side = sb
            elif sb == 0:
                pair_side = sa
            elif sa == sb:
                pair_side = sa
            else:
                pair_side = 4
            da, db = last_distance[unit], distance_bins[i]
            pair_distance = da if da < db else db
            ta, tb = last_temporal[unit], temporal_bins[i]
            pair_temporal = ta if ta < tb else tb
            state = states[i]
            if delta == 0:
                pair_exact[state, pair_class, pair_side, pair_distance, pair_temporal] += 1
            elif delta > 0:
                unit_denominator[unit, state] += 1
                pair_denominator[state, pair_class, pair_side, pair_distance, pair_temporal] += 1
                if delta < 30:
                    unit_short[unit, state] += 1
                    pair_short[state, pair_class, pair_side, pair_distance, pair_temporal] += 1
        last_time[unit] = times[i]
        last_state[unit] = states[i]
        last_segment[unit] = segments[i]
        last_support[unit] = support_status[i]
        last_side[unit] = sides[i]
        last_distance[unit] = distance_bins[i]
        last_temporal[unit] = temporal_bins[i]


def load_labels(path: Path) -> dict[int, str]:
    with path.open(newline="") as stream:
        rows = csv.DictReader(stream, delimiter="\t")
        return {int(row["cluster_id"]): row.get("KSLabel", row.get("group", "unknown")) for row in rows}


def template_footprint(root: Path, geometry: np.ndarray, support: dict) -> tuple[np.ndarray, np.ndarray]:
    templates = np.load(root / "templates.npy", mmap_mode="r")
    if templates.shape[2] != len(geometry):
        raise RuntimeError("template channel axis differs from geometry")
    p2p = np.ptp(np.asarray(templates), axis=1).astype(np.float64)
    maximum = p2p.max(axis=1)
    active = p2p >= (0.20 * maximum[:, None])
    active[maximum <= 0] = False
    row_index = np.searchsorted(support["rows_y"], geometry[:, 1])
    state_channel_support = support["row_support"][:, row_index].astype(bool)
    classes = np.full((len(templates), len(state_channel_support)), 3, dtype=np.uint8)
    missing_energy = np.full(classes.shape, np.nan, dtype=np.float64)
    energy = p2p ** 2
    for si, channel_supported in enumerate(state_channel_support):
        missing_active = active & ~channel_supported[None, :]
        active_count = active.sum(axis=1)
        missing_count = missing_active.sum(axis=1)
        valid = active_count > 0
        classes[valid & (missing_count == 0), si] = 0
        classes[valid & (missing_count > 0) & (missing_count < active_count), si] = 1
        classes[valid & (missing_count == active_count), si] = 2
        total_energy = energy.sum(axis=1)
        missing_energy[:, si] = np.divide(
            energy[:, ~channel_supported].sum(axis=1), total_energy,
            out=np.full(len(templates), np.nan), where=total_energy > 0,
        )
    return classes, missing_energy


def analyze_sort(
    root: Path, cells: dict, support: dict, chunk_size: int, sampling_frequency_hz: float
) -> dict:
    times = np.load(root / "spike_times.npy", mmap_mode="r")
    clusters = np.load(root / "spike_clusters.npy", mmap_mode="r")
    positions = np.load(root / "spike_positions.npy", mmap_mode="r")
    geometry = np.load(root / "channel_positions.npy")
    if len(times) != len(clusters) or len(times) != len(positions):
        raise RuntimeError("spike arrays are misaligned")
    if times.dtype.kind not in "iu" or clusters.dtype.kind not in "iu" or positions.shape[1] != 2:
        raise RuntimeError("unexpected spike array schema")
    if np.any(np.diff(np.asarray(times)) < 0):
        raise RuntimeError("spike times are not globally nondecreasing")
    unit_ids = np.unique(clusters)
    if not np.array_equal(unit_ids, np.arange(len(unit_ids))):
        raise RuntimeError("cluster ids are not contiguous template row ids")
    n_units, n_states = len(unit_ids), len(cells["states"])
    labels = load_labels(root / "cluster_KSLabel.tsv")
    event_shape = (n_units, n_states, len(SUPPORT_NAMES), len(SIDE_NAMES), len(DISTANCE_NAMES))
    event_counts = np.zeros(event_shape, dtype=np.int64)
    unit_denominator = np.zeros((n_units, n_states), dtype=np.int64)
    unit_short = np.zeros_like(unit_denominator)
    pair_shape = (n_states, len(PAIR_NAMES), len(SIDE_NAMES), len(DISTANCE_NAMES), len(TEMPORAL_NAMES))
    pair_denominator = np.zeros(pair_shape, dtype=np.int64)
    pair_short = np.zeros(pair_shape, dtype=np.int64)
    pair_exact = np.zeros(pair_shape, dtype=np.int64)
    last_time = np.full(n_units, -1, dtype=np.int64)
    last_state = np.full(n_units, -1, dtype=np.int16)
    last_segment = np.full(n_units, -1, dtype=np.int32)
    last_support = np.full(n_units, 3, dtype=np.uint8)
    last_side = np.full(n_units, 5, dtype=np.uint8)
    last_distance = np.full(n_units, 5, dtype=np.uint8)
    last_temporal = np.full(n_units, 4, dtype=np.uint8)
    edges = cells["edges"]
    for start in range(0, len(times), chunk_size):
        stop = min(len(times), start + chunk_size)
        t = np.asarray(times[start:stop], dtype=np.int64)
        u = np.asarray(clusters[start:stop], dtype=np.int64)
        y = np.asarray(positions[start:stop, 1], dtype=np.float64)
        seconds = t / sampling_frequency_hz
        knot = np.searchsorted(edges, seconds, side="right") - 1
        knot = np.clip(knot, 0, len(cells["knot_state"]) - 1)
        state = cells["knot_state"][knot]
        segment = cells["knot_segment"][knot]
        status, side, distance_bin = assign_positions(y, state, support)
        temporal_bin = temporal_change_bins(seconds, cells["change_boundaries_s"])
        code = (((u * n_states + state) * len(SUPPORT_NAMES) + status) * len(SIDE_NAMES) + side) * len(DISTANCE_NAMES) + distance_bin
        event_counts += np.bincount(code, minlength=event_counts.size).reshape(event_shape)
        accumulate_pairs(
            t, u, state, segment, status, side, distance_bin, temporal_bin,
            last_time, last_state, last_segment, last_support, last_side, last_distance, last_temporal,
            unit_denominator, unit_short, pair_denominator, pair_short, pair_exact,
        )
    footprint_class, footprint_missing_energy = template_footprint(root, geometry, support)
    unit_state_events = event_counts.sum(axis=(2, 3, 4))
    q0 = int(np.flatnonzero(cells["states"] == 0)[0])
    total_by_unit = unit_state_events.sum(axis=1)
    q0_fraction = np.divide(unit_state_events[:, q0], total_by_unit, out=np.zeros(n_units), where=total_by_unit > 0)
    return {
        "root": str(root), "n_events": int(len(times)), "n_units": n_units,
        "labels": labels, "event_counts": event_counts,
        "unit_state_events": unit_state_events, "unit_denominator": unit_denominator,
        "unit_short": unit_short, "q0_fraction": q0_fraction,
        "pair_denominator": pair_denominator, "pair_short": pair_short,
        "pair_exact": pair_exact, "footprint_class": footprint_class,
        "footprint_missing_energy": footprint_missing_energy,
        "geometry_sha256": hashlib.sha256(np.asarray(geometry).tobytes()).hexdigest(),
    }


def fixture() -> dict:
    times = np.array([0, 10, 20, 50, 100, 110], dtype=np.int64)
    units = np.zeros(6, dtype=np.int64)
    states = np.zeros(6, dtype=np.int16)
    segments = np.array([0, 0, 0, 0, 1, 1], dtype=np.int32)
    status = np.array([0, 1, 0, 0, 0, 0], dtype=np.uint8)
    side = np.where(status == 1, 1, 0).astype(np.uint8)
    distance = np.zeros(6, dtype=np.uint8)
    ud = np.zeros((1, 1), np.int64); us = np.zeros_like(ud)
    shape = (1, len(PAIR_NAMES), len(SIDE_NAMES), len(DISTANCE_NAMES), len(TEMPORAL_NAMES))
    pd = np.zeros(shape, np.int64); ps = np.zeros(shape, np.int64); pe = np.zeros(shape, np.int64)
    accumulate_pairs(
        times, units, states, segments, status, side, distance, np.zeros(6, np.uint8),
        np.full(1, -1, np.int64), np.full(1, -1, np.int16), np.full(1, -1, np.int32),
        np.full(1, 3, np.uint8), np.full(1, 5, np.uint8), np.full(1, 5, np.uint8), np.full(1, 4, np.uint8),
        ud, us, pd, ps, pe,
    )
    result = {
        "original_denominator": int(pd.sum()), "original_short": int(ps.sum()),
        "both_supported_denominator": int(pd[0, 0].sum()), "both_supported_short": int(ps[0, 0].sum()),
        "mixed_denominator": int(pd[0, 1].sum()), "mixed_short": int(ps[0, 1].sum()),
    }
    expected = {
        "original_denominator": 4, "original_short": 3,
        "both_supported_denominator": 2, "both_supported_short": 1,
        "mixed_denominator": 2, "mixed_short": 2,
    }
    if result != expected:
        raise AssertionError((result, expected))
    filtered_times = times[status == 0]
    filtering_first_invents_0_20 = bool(np.any(np.diff(filtered_times[:3]) == 20))
    if not filtering_first_invents_0_20:
        raise AssertionError("fixture failed to demonstrate filter-first bridge")
    return {"observed": result, "expected": expected, "pass": True, "filter_first_invents_pair_0_20": True}


def boundary_fixture() -> dict:
    edges = np.array([0.0, 0.125, 0.375, 1.0])
    knot_state = np.array([0, 1, 0], dtype=np.int16)
    knot_segment = np.array([0, 1, 2], dtype=np.int32)
    samples_s = np.array([0.124999, 0.125, 0.374999, 0.375])
    knot = np.searchsorted(edges, samples_s, side="right") - 1
    states = knot_state[knot].tolist(); segments = knot_segment[knot].tolist()
    expected_states, expected_segments = [0, 1, 1, 0], [0, 1, 1, 2]
    if states != expected_states or segments != expected_segments:
        raise AssertionError((states, segments))
    return {"pass": True, "states": states, "segments": segments,
            "repeated_state_after_gap_has_new_segment": segments[-1] != segments[0]}


def aggregate_rows(field: str, arm: str, result: dict, states: np.ndarray) -> tuple[list[dict], list[dict], list[dict]]:
    pair_rows, unit_rows, footprint_rows = [], [], []
    for si, state in enumerate(states):
        for pc, pair_name in enumerate(PAIR_NAMES):
            for side, side_name in enumerate(SIDE_NAMES):
                for db, distance_name in enumerate(DISTANCE_NAMES):
                    for tb, temporal_name in enumerate(TEMPORAL_NAMES):
                        den = int(result["pair_denominator"][si, pc, side, db, tb])
                        short = int(result["pair_short"][si, pc, side, db, tb])
                        exact = int(result["pair_exact"][si, pc, side, db, tb])
                        if den or short or exact:
                            pair_rows.append({
                                "field": field, "arm": arm, "state_um": float(state),
                                "support_pair": pair_name, "missing_side": side_name,
                                "spatial_support_boundary_bin_um": distance_name,
                                "temporal_state_change_bin_s": temporal_name,
                                "positive_adjacent_intervals": den,
                                "short_1_29": short, "exact_duplicates": exact,
                                "short_fraction": short / den if den else "",
                            })
        for unit in range(result["n_units"]):
            unit_rows.append({
                "field": field, "arm": arm, "unit_id": unit,
                "ks_label": result["labels"].get(unit, "unknown"),
                "q0_fraction": float(result["q0_fraction"][unit]),
                "state_um": float(state), "events": int(result["unit_state_events"][unit, si]),
                "positive_adjacent_intervals": int(result["unit_denominator"][unit, si]),
                "short_1_29": int(result["unit_short"][unit, si]),
                "template_footprint_class": FOOTPRINT_NAMES[result["footprint_class"][unit, si]],
                "template_missing_energy_fraction": float(result["footprint_missing_energy"][unit, si]),
            })
        for fc, footprint_name in enumerate(FOOTPRINT_NAMES):
            units = result["footprint_class"][:, si] == fc
            footprint_rows.append({
                "field": field, "arm": arm, "state_um": float(state),
                "template_footprint_class": footprint_name, "units": int(units.sum()),
                "events": int(result["unit_state_events"][units, si].sum()),
                "positive_adjacent_intervals": int(result["unit_denominator"][units, si].sum()),
                "short_1_29": int(result["unit_short"][units, si].sum()),
            })
    return pair_rows, unit_rows, footprint_rows


def compact_summary(field: str, candidate: dict, reference: dict, states: np.ndarray) -> dict:
    nonzero = states != 0
    def pool(result, pair_classes=None):
        selector = result["pair_short"][nonzero]
        denominator = result["pair_denominator"][nonzero]
        if pair_classes is not None:
            selector = selector[:, pair_classes]
            denominator = denominator[:, pair_classes]
        return {"events": int(result["unit_state_events"][:, nonzero].sum()),
                "short": int(selector.sum()), "denominator": int(denominator.sum())}
    categories = {}
    for pc, name in enumerate(PAIR_NAMES):
        c_short = int(candidate["pair_short"][nonzero, pc].sum())
        r_short = int(reference["pair_short"][nonzero, pc].sum())
        c_den = int(candidate["pair_denominator"][nonzero, pc].sum())
        r_den = int(reference["pair_denominator"][nonzero, pc].sum())
        categories[name] = {
            "candidate_short": c_short, "reference_short": r_short,
            "short_excess": c_short - r_short,
            "candidate_denominator": c_den, "reference_denominator": r_den,
            "candidate_fraction": c_short / c_den if c_den else math.nan,
            "reference_fraction": r_short / r_den if r_den else math.nan,
        }
    event_categories = {}
    for sc, name in enumerate(SUPPORT_NAMES):
        c = int(candidate["event_counts"][:, nonzero, sc].sum())
        r = int(reference["event_counts"][:, nonzero, sc].sum())
        event_categories[name] = {"candidate": c, "reference": r, "excess": c - r}
    footprint = {}
    for fc, name in enumerate(FOOTPRINT_NAMES):
        cmask = candidate["footprint_class"][:, nonzero] == fc
        rmask = reference["footprint_class"][:, nonzero] == fc
        ce = int(candidate["unit_state_events"][:, nonzero][cmask].sum())
        re = int(reference["unit_state_events"][:, nonzero][rmask].sum())
        cs = int(candidate["unit_short"][:, nonzero][cmask].sum())
        rs = int(reference["unit_short"][:, nonzero][rmask].sum())
        footprint[name] = {"candidate_events": ce, "reference_events": re, "event_excess": ce - re,
                           "candidate_short": cs, "reference_short": rs, "short_excess": cs - rs}
    return {"field": field, "nonzero_total": {"candidate": pool(candidate), "reference": pool(reference)},
            "position_pair_categories": categories, "position_event_categories": event_categories,
            "template_footprint_categories": footprint}


def verify_original(result: dict, published: list[dict], states: np.ndarray) -> list[dict]:
    lookup = {float(row["state_um"]): row for row in published}
    rows = []
    for si, state in enumerate(states):
        observed = {
            "spikes": int(result["unit_state_events"][:, si].sum()),
            "denominator": int(result["pair_denominator"][si].sum()),
            "short": int(result["pair_short"][si].sum()),
            "exact": int(result["pair_exact"][si].sum()),
        }
        expected_row = lookup[float(state)]
        expected = {
            "spikes": int(expected_row["spikes"]),
            "denominator": int(expected_row["segment_safe_positive_intervals"]),
            "short": int(expected_row["short_interval_count_1_29"]),
            "exact": int(expected_row["exact_duplicate_count"]),
        }
        rows.append({"state_um": float(state), "observed": observed, "published": expected,
                     "exact_match": observed == expected})
    if not all(row["exact_match"] for row in rows):
        raise RuntimeError(f"independent totals do not reproduce: {rows}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/en_state_support_diagnosis.v2.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    args = parser.parse_args()
    started = time.perf_counter()
    selected_config = json.loads(args.config.read_text())
    if selected_config.get("schema_version") == "en-state-support-diagnosis-v2":
        base_path = Path(selected_config["base_protocol"]["path"])
        if sha256(base_path) != selected_config["base_protocol"]["sha256"]:
            raise RuntimeError("base protocol hash mismatch")
        config = json.loads(base_path.read_text())
        config["schema_version"] = selected_config["schema_version"]
        config["protocol_correction"] = selected_config["correction"]
        config["temporal_state_change_attribution"] = selected_config["temporal_state_change_attribution"]
    else:
        raise ValueError("v2 corrected protocol is required")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    atomic_json(args.output / "STATUS.json", {"stage": "fixtures"})
    fixtures = {"pair_filter_order": fixture(), "half_open_boundary": boundary_fixture()}
    tier1_path = Path(config["tier1_result"]["path"])
    if sha256(tier1_path) != config["tier1_result"]["sha256"]:
        raise RuntimeError("Tier-1 result hash mismatch")
    tier1 = json.loads(tier1_path.read_text())
    fs = float(config["scope"]["sampling_frequency_hz"])
    num_samples = int(config["scope"]["num_samples"])
    duration_s = num_samples / fs
    all_pair_rows, all_unit_rows, all_footprint_rows = [], [], []
    summaries, reproductions, inputs, supports = {}, {}, {}, {}
    for field, spec in config["comparisons"].items():
        atomic_json(args.output / "STATUS.json", {"stage": "field", "field": field})
        cells = field_cells(spec, duration_s)
        candidate_root = Path(spec["candidate_sorter_output"])
        reference_root = Path(spec["reference_sorter_output"])
        geometry = np.load(candidate_root / "channel_positions.npy")
        if not np.array_equal(geometry, np.load(reference_root / "channel_positions.npy")):
            raise RuntimeError(f"{field}: candidate/reference geometries differ")
        support = support_tables(geometry, cells["states"])
        supports[field] = support["details"]
        results = {}
        for arm, root in ((field, candidate_root), ("REF", reference_root)):
            atomic_json(args.output / "STATUS.json", {"stage": "analyze", "field": field, "arm": arm})
            result = analyze_sort(root, cells, support, args.chunk_size, fs)
            results[arm] = result
            published = tier1["state_profiles"][field][arm]
            reproductions[f"{field}_{arm}"] = verify_original(result, published, cells["states"])
            pair_rows, unit_rows, footprint_rows = aggregate_rows(field, arm, result, cells["states"])
            all_pair_rows.extend(pair_rows); all_unit_rows.extend(unit_rows); all_footprint_rows.extend(footprint_rows)
            inputs[f"{field}_{arm}"] = {
                "root": str(root), "events": result["n_events"], "units": result["n_units"],
                "geometry_sha256": result["geometry_sha256"],
                "spike_times_sha256": tier1["inputs"][arm]["files"]["spike_times.npy"]["sha256"],
                "spike_clusters_sha256": tier1["inputs"][arm]["files"]["spike_clusters.npy"]["sha256"],
                "spike_positions_sha256": tier1["inputs"][arm]["files"]["spike_positions.npy"]["sha256"],
            }
        summaries[field] = compact_summary(field, results[field], results["REF"], cells["states"])
    write_csv(args.output / "PAIR_SUPPORT_COUNTS.csv", all_pair_rows)
    write_csv(args.output / "UNIT_STATE_COUNTS.csv", all_unit_rows)
    write_csv(args.output / "TEMPLATE_FOOTPRINT_COUNTS.csv", all_footprint_rows)
    result = {
        "schema": "en-state-support-diagnosis-result-v1", "status": "complete",
        "implementation_source": {"path": str(Path(__file__).resolve()),
                                  "sha256": sha256(Path(__file__).resolve())},
        "config": {"path": str(args.config.resolve()), "sha256": sha256(args.config),
                   "base_path": str(base_path.resolve()), "base_sha256": sha256(base_path),
                   "correction": selected_config["correction"]},
        "fixtures": fixtures, "original_total_reproduction": reproductions,
        "inputs": inputs, "state_support": supports, "summaries": summaries,
        "resources": {"wall_seconds": time.perf_counter() - started,
                      "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      "raw_voltage_bytes_read": 0, "sorts_launched": 0,
                      "rf_accesses": 0, "holdout_accesses": 0},
        "limitations": [
            "Saved Kilosort positions are feature-mass-weighted coordinates, not observed waveform origins.",
            "Saved templates are refitted arm-specific representations and are not state-specific observed waveforms.",
            "Association with a support category does not prove duplicates, biological identity, purity, or causality.",
        ],
    }
    atomic_json(args.output / "RESULT.json", result)
    products = []
    for path in sorted(args.output.iterdir()):
        if path.name in {"COMPLETE.json", "STATUS.json"} or not path.is_file():
            continue
        products.append({"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)})
    atomic_json(args.output / "MANIFEST.json", {"schema": "en-state-support-diagnosis-manifest-v1", "products": products})
    atomic_json(args.output / "STATUS.json", {"stage": "complete", "status": "complete"})
    atomic_json(args.output / "COMPLETE.json", {
        "schema": "en-state-support-diagnosis-complete-v1", "status": "complete",
        "manifest_sha256": sha256(args.output / "MANIFEST.json"),
    })


if __name__ == "__main__":
    main()
