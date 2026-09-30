#!/usr/bin/env python3
"""Evaluate the frozen EN Tier-1 panel from saved raw Kilosort outputs.

This module keeps field-state assignment and uncertainty independent of sorter
or curation code.  State cells reproduce the half-open temporal cells used by
the EN voltage adapter.  Adjacent-spike statistics never cross a state-segment
boundary.  The command can audit input availability before all arms arrive;
scientific output is written only when every requested arm is complete.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import time
from typing import Any

import numpy as np
import pandas as pd
from numba import njit
from scipy.stats import spearmanr

from testing.en_rounded_field import rounded_rigid_field


SCHEMA = "en-tier1-panel-result-v1"
OKABE_ITO = {
    "REF": "#000000",
    "XR": "#D55E00",
    "A": "#0072B2",
    "B": "#009E73",
}
ARM_MARKERS = {"REF": "o", "XR": "s", "A": "^", "B": "D"}
RAW_FILES = (
    "spike_times.npy",
    "spike_clusters.npy",
    "spike_positions.npy",
    "cluster_KSLabel.tsv",
)


def sha256(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(json_clean(value), indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def json_clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_clean(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    if keep.sum() < 3 or np.unique(x[keep]).size < 2 or np.unique(y[keep]).size < 2:
        return math.nan
    result = spearmanr(x[keep], y[keep])
    return float(result.statistic if hasattr(result, "statistic") else result[0])


def percentile_ci(values: np.ndarray, confidence: float = 0.95) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return math.nan, math.nan
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(values, [tail, 1.0 - tail])
    return float(low), float(high)


@dataclass(frozen=True)
class FieldCells:
    states_um: np.ndarray
    knot_edges_s: np.ndarray
    knot_state_index: np.ndarray
    knot_segment_id: np.ndarray
    state_exposure_s: np.ndarray
    block_exposure_s: np.ndarray
    block_edges_s: np.ndarray

    def assign(self, samples: np.ndarray, sampling_frequency_hz: float) -> tuple[np.ndarray, np.ndarray]:
        seconds = np.asarray(samples, dtype=np.float64) / float(sampling_frequency_hz)
        knot = np.searchsorted(self.knot_edges_s, seconds, side="right") - 1
        knot = np.clip(knot, 0, len(self.knot_state_index) - 1)
        return self.knot_state_index[knot], self.knot_segment_id[knot]


def build_field_cells(
    temporal_bins_s: np.ndarray,
    rounded_displacement_um: np.ndarray,
    *,
    duration_s: float,
    block_seconds: float,
) -> FieldCells:
    """Build clipped SI half-open cells and exact block/state exposures."""
    temporal_bins_s = np.asarray(temporal_bins_s, dtype=np.float64)
    rounded = np.asarray(rounded_displacement_um, dtype=np.float64)
    if temporal_bins_s.ndim != 1 or rounded.shape != temporal_bins_s.shape or len(rounded) < 2:
        raise ValueError("field times and rounded displacement must be equal-length vectors")
    if not np.all(np.isfinite(temporal_bins_s)) or not np.all(np.diff(temporal_bins_s) > 0):
        raise ValueError("field times must be finite and strictly increasing")
    if not np.all(np.isfinite(rounded)):
        raise ValueError("rounded displacement must be finite")
    edges = np.empty(len(temporal_bins_s) + 1, dtype=np.float64)
    edges[1:-1] = (temporal_bins_s[:-1] + temporal_bins_s[1:]) / 2.0
    edges[0], edges[-1] = 0.0, float(duration_s)
    edges = np.clip(edges, 0.0, float(duration_s))
    if np.any(np.diff(edges) < 0):
        raise ValueError("clipped field cells are not ordered")
    states = np.unique(rounded)
    state_index = np.searchsorted(states, rounded).astype(np.int16)
    segment = np.cumsum(np.r_[True, rounded[1:] != rounded[:-1]]).astype(np.int32) - 1
    widths = np.diff(edges)
    exposure = np.bincount(state_index, weights=widths, minlength=len(states)).astype(float)
    if not np.isclose(exposure.sum(), duration_s, rtol=0, atol=1e-8):
        raise RuntimeError("field-state exposures do not close the recording")

    block_edges = np.r_[np.arange(0.0, duration_s, block_seconds), duration_s]
    block_edges = np.unique(block_edges)
    block_exposure = np.zeros((len(block_edges) - 1, len(states)), dtype=np.float64)
    for knot, state in enumerate(state_index):
        left, right = edges[knot : knot + 2]
        if right <= left:
            continue
        first = max(0, np.searchsorted(block_edges, left, side="right") - 1)
        last = min(len(block_edges) - 2, np.searchsorted(block_edges, right, side="left"))
        for block in range(first, last + 1):
            overlap = max(0.0, min(right, block_edges[block + 1]) - max(left, block_edges[block]))
            block_exposure[block, state] += overlap
    if not np.isclose(block_exposure.sum(), duration_s, rtol=0, atol=1e-8):
        raise RuntimeError("block/state exposures do not close the recording")
    return FieldCells(states, edges, state_index, segment, exposure, block_exposure, block_edges)


def common_block_multiplicities(n_blocks: int, draws: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, n_blocks, size=(draws, n_blocks))
    multiplicities = np.zeros((draws, n_blocks), dtype=np.int16)
    rows = np.repeat(np.arange(draws), n_blocks)
    np.add.at(multiplicities, (rows, sampled.ravel()), 1)
    return multiplicities


@njit(cache=True)
def _possible_pair_counts(at, ac, bt, bc, na, nb, tolerance):
    counts = np.zeros((na, nb), dtype=np.int64)
    left = 0
    for i in range(len(at)):
        while left < len(bt) and bt[left] < at[i] - tolerance:
            left += 1
        j = left
        while j < len(bt) and bt[j] <= at[i] + tolerance:
            counts[ac[i], bc[j]] += 1
            j += 1
    return counts


@njit(cache=True)
def _exclusive_count(a, b, tolerance):
    i = j = count = 0
    while i < len(a) and j < len(b):
        if a[i] < b[j] - tolerance:
            i += 1
        elif b[j] < a[i] - tolerance:
            j += 1
        else:
            count += 1
            i += 1
            j += 1
    return count


@njit(cache=True)
def _marked_cross_unit_events(times, clusters, depths, tolerance, depth_tolerance):
    marked = np.zeros(len(times), dtype=np.uint8)
    for left in range(len(times)):
        right = left + 1
        while right < len(times) and times[right] - times[left] <= tolerance:
            if clusters[right] != clusters[left] and abs(depths[right] - depths[left]) <= depth_tolerance:
                marked[left] = 1
                marked[right] = 1
            right += 1
    return marked


def correspondence(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    *,
    tolerance_frames: int,
    minimum_overlap: float,
    primary_retention: float,
) -> pd.DataFrame:
    """Full exclusive time-only edge graph with reciprocal unique primaries."""
    at, ac_raw = np.asarray(reference["times"]), np.asarray(reference["clusters"])
    bt, bc_raw = np.asarray(candidate["times"]), np.asarray(candidate["clusters"])
    aid, ac = np.unique(ac_raw, return_inverse=True)
    bid, bc = np.unique(bc_raw, return_inverse=True)
    an, bn = np.bincount(ac), np.bincount(bc)
    possible = _possible_pair_counts(at, ac, bt, bc, len(aid), len(bid), tolerance_frames)
    ats = np.split(at[np.argsort(ac, kind="stable")], np.cumsum(an)[:-1])
    bts = np.split(bt[np.argsort(bc, kind="stable")], np.cumsum(bn)[:-1])
    rows = []
    threshold = minimum_overlap * np.minimum(an[:, None], bn[None, :])
    for i, j in np.argwhere(possible >= threshold):
        matched = int(_exclusive_count(ats[i], bts[j], tolerance_frames))
        if matched < minimum_overlap * min(an[i], bn[j]):
            continue
        rows.append(
            {
                "reference_cluster": int(aid[i]),
                "candidate_cluster": int(bid[j]),
                "matched_events": matched,
                "reference_events": int(an[i]),
                "candidate_events": int(bn[j]),
                "reference_retention": matched / an[i],
                "candidate_retention": matched / bn[j],
                "f1": 2.0 * matched / (an[i] + bn[j]),
                "jaccard": matched / (an[i] + bn[j] - matched),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        frame["primary_match"] = pd.Series(dtype=bool)
        return frame
    reference_best = frame.groupby("reference_cluster").jaccard.transform("max")
    candidate_best = frame.groupby("candidate_cluster").jaccard.transform("max")
    rb, cb = frame.jaccard == reference_best, frame.jaccard == candidate_best
    reference_unique = rb.groupby(frame.reference_cluster).transform("sum") == 1
    candidate_unique = cb.groupby(frame.candidate_cluster).transform("sum") == 1
    frame["primary_match"] = (
        rb & cb & reference_unique & candidate_unique
        & frame.reference_retention.ge(primary_retention)
        & frame.candidate_retention.ge(primary_retention)
    )
    return frame


def correspondence_summary(
    edges: pd.DataFrame,
    reference_units: np.ndarray,
    candidate_units: np.ndarray,
    *,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    primary = edges[edges.primary_match] if len(edges) else edges
    reference_primary = set(primary.reference_cluster.astype(int)) if len(primary) else set()
    candidate_primary = set(primary.candidate_cluster.astype(int)) if len(primary) else set()
    reference_degree = edges.groupby("reference_cluster").size() if len(edges) else pd.Series(dtype=int)
    candidate_degree = edges.groupby("candidate_cluster").size() if len(edges) else pd.Series(dtype=int)
    ref_lost = np.asarray([int(unit) not in reference_primary for unit in reference_units], dtype=float)
    cand_new = np.asarray([int(unit) not in candidate_primary for unit in candidate_units], dtype=float)
    ref_ambiguous = np.asarray(
        [reference_degree.get(int(unit), 0) > 1 or (reference_degree.get(int(unit), 0) > 0 and int(unit) not in reference_primary)
         for unit in reference_units], dtype=float,
    )
    cand_ambiguous = np.asarray(
        [candidate_degree.get(int(unit), 0) > 1 or (candidate_degree.get(int(unit), 0) > 0 and int(unit) not in candidate_primary)
         for unit in candidate_units], dtype=float,
    )
    output: dict[str, Any] = {
        "edge_count": int(len(edges)),
        "primary_matches": int(len(primary)),
        "lost_reference_units": int(ref_lost.sum()),
        "new_candidate_units": int(cand_new.sum()),
        "ambiguous_reference_units": int(ref_ambiguous.sum()),
        "ambiguous_candidate_units": int(cand_ambiguous.sum()),
    }
    for offset, (name, values) in enumerate(
        (("lost_reference_fraction", ref_lost), ("new_candidate_fraction", cand_new),
         ("ambiguous_reference_fraction", ref_ambiguous),
         ("ambiguous_candidate_fraction", cand_ambiguous))
    ):
        point, low, high = _unit_bootstrap(values, draws=draws, seed=seed + offset, statistic="mean")
        output[name] = point
        output[name + "_ci95"] = [low, high]
    f1 = primary.f1.to_numpy(float) if len(primary) else np.array([], dtype=float)
    point, low, high = _unit_bootstrap(f1, draws=draws, seed=seed + 10, statistic="median")
    output["median_primary_f1"] = point
    output["median_primary_f1_ci95"] = [low, high]
    return output


def chance_aware_coincidence(
    sort: dict[str, Any],
    unit_table: pd.DataFrame,
    *,
    num_samples: int,
    sampling_frequency_hz: float,
    block_edges_s: np.ndarray,
    multiplicities: np.ndarray,
    tolerance_ms: float,
    depth_tolerance_um: float,
    processing_depth_um: tuple[float, float],
    seed: int,
) -> dict[str, Any]:
    """Observed and circular-shift marked-spike burdens with block CIs."""
    times = np.asarray(sort["times"], dtype=np.int64)
    raw_clusters = np.asarray(sort["clusters"], dtype=np.int64)
    depths = np.asarray(sort["depths"], dtype=np.float64)
    unit_ids, inverse = np.unique(raw_clusters, return_inverse=True)
    processing_by_unit = unit_table.set_index("unit_id").loc[unit_ids, "in_processing_domain"].to_numpy(bool)
    p0, p1 = processing_depth_um
    keep = processing_by_unit[inverse] & (depths >= p0) & (depths <= p1)
    times, clusters, depths = times[keep], inverse[keep], depths[keep]
    tolerance = int(round(tolerance_ms * 1e-3 * sampling_frequency_hz))
    observed_mark = _marked_cross_unit_events(times, clusters, depths, tolerance, depth_tolerance_um)
    observed_block = np.searchsorted(block_edges_s, times / sampling_frequency_hz, side="right") - 1
    observed_block = np.clip(observed_block, 0, len(block_edges_s) - 2)
    n_blocks = len(block_edges_s) - 1
    observed_counts = np.bincount(observed_block, weights=observed_mark, minlength=n_blocks)
    observed_total = np.bincount(observed_block, minlength=n_blocks)

    rng = np.random.default_rng(seed)
    offsets = rng.integers(tolerance + 1, num_samples, size=len(unit_ids), dtype=np.int64)
    shifted = (times + offsets[clusters]) % num_samples
    order = np.argsort(shifted, kind="stable")
    shifted_times, shifted_clusters, shifted_depths = shifted[order], clusters[order], depths[order]
    null_mark = _marked_cross_unit_events(
        shifted_times, shifted_clusters, shifted_depths, tolerance, depth_tolerance_um
    )
    null_block = np.searchsorted(block_edges_s, shifted_times / sampling_frequency_hz, side="right") - 1
    null_block = np.clip(null_block, 0, n_blocks - 1)
    null_counts = np.bincount(null_block, weights=null_mark, minlength=n_blocks)
    null_total = np.bincount(null_block, minlength=n_blocks)
    observed = float(observed_mark.mean()) if len(observed_mark) else math.nan
    null = float(null_mark.mean()) if len(null_mark) else math.nan
    draw_observed = np.divide(
        multiplicities @ observed_counts, multiplicities @ observed_total,
        out=np.full(len(multiplicities), np.nan), where=(multiplicities @ observed_total) > 0,
    )
    draw_null = np.divide(
        multiplicities @ null_counts, multiplicities @ null_total,
        out=np.full(len(multiplicities), np.nan), where=(multiplicities @ null_total) > 0,
    )
    draw_excess = draw_observed - draw_null
    return {
        "eligible_spikes": int(len(times)),
        "observed_marked_spike_fraction": observed,
        "observed_ci95": list(percentile_ci(draw_observed)),
        "shift_null_fraction": null,
        "shift_null_ci95": list(percentile_ci(draw_null)),
        "excess": observed - null,
        "excess_ci95": list(percentile_ci(draw_excess)),
    }


def load_labels(path: Path) -> dict[int, str]:
    table = pd.read_csv(path, sep="\t")
    if "cluster_id" not in table or len(table.columns) != 2:
        raise ValueError(f"invalid Kilosort label table: {path}")
    label_column = next(column for column in table if column != "cluster_id")
    if table.cluster_id.duplicated().any():
        raise ValueError(f"duplicate cluster_id in {path}")
    return dict(zip(table.cluster_id.astype(int), table[label_column].astype(str).str.lower()))


def load_sort(root: Path, *, num_samples: int) -> dict[str, Any]:
    root = root.resolve()
    missing = [name for name in RAW_FILES if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing raw sort files in {root}: {missing}")
    times = np.asarray(np.load(root / "spike_times.npy", mmap_mode="r"), dtype=np.int64).reshape(-1)
    clusters = np.asarray(np.load(root / "spike_clusters.npy", mmap_mode="r"), dtype=np.int64).reshape(-1)
    positions = np.asarray(np.load(root / "spike_positions.npy", mmap_mode="r"), dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] < 2:
        raise ValueError(f"invalid spike_positions.npy in {root}")
    if len(times) != len(clusters) or len(times) != len(positions):
        raise ValueError(f"raw sort arrays are not aligned in {root}")
    if len(times) and (times[0] < 0 or times[-1] >= num_samples or np.any(np.diff(times) < 0)):
        raise ValueError(f"spike times are outside the recording or not sorted in {root}")
    if not np.isfinite(positions[:, 1]).all():
        raise ValueError(f"nonfinite saved spike depths in {root}")
    return {
        "times": times,
        "clusters": clusters,
        "depths": positions[:, 1],
        "labels": load_labels(root / "cluster_KSLabel.tsv"),
        "root": str(root),
    }


def _unit_bootstrap(
    values: np.ndarray, *, draws: int, seed: int, statistic: str
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return math.nan, math.nan, math.nan
    function = {"median": np.median, "p10": lambda x, axis: np.quantile(x, 0.1, axis=axis),
                "p90": lambda x, axis: np.quantile(x, 0.9, axis=axis), "mean": np.mean}[statistic]
    point = float(function(values, axis=0))
    rng = np.random.default_rng(seed)
    sampled = values[rng.integers(0, len(values), size=(draws, len(values)))]
    boot = function(sampled, axis=1)
    low, high = percentile_ci(boot)
    return point, low, high


def arm_state_metrics(
    sort: dict[str, Any],
    cells: FieldCells,
    *,
    sampling_frequency_hz: float,
    num_samples: int,
    multiplicities: np.ndarray,
    minimum_reference_events: int,
    processing_depth_um: tuple[float, float],
    scoring_depth_um: tuple[float, float],
    seed: int,
) -> dict[str, Any]:
    """Compute point estimates and frozen CIs for one arm on one field axis."""
    times = np.asarray(sort["times"], dtype=np.int64)
    clusters = np.asarray(sort["clusters"], dtype=np.int64)
    depths = np.asarray(sort["depths"], dtype=np.float64)
    unit_ids, inverse, unit_counts = np.unique(clusters, return_inverse=True, return_counts=True)
    n_units, n_states = len(unit_ids), len(cells.states_um)
    state, segment = cells.assign(times, sampling_frequency_hz)
    block = np.searchsorted(cells.block_edges_s, times / sampling_frequency_hz, side="right") - 1
    block = np.clip(block, 0, len(cells.block_edges_s) - 2).astype(np.int16)
    n_blocks = len(cells.block_edges_s) - 1

    combined = (block.astype(np.int64) * n_units + inverse) * n_states + state
    block_unit_state = np.bincount(
        combined, minlength=n_blocks * n_units * n_states
    ).reshape(n_blocks, n_units, n_states)
    block_unit = block_unit_state.sum(axis=2)
    unit_state = block_unit_state.sum(axis=0)
    state_counts = unit_state.sum(axis=0)
    block_spikes = block_unit.sum(axis=1)

    order = np.argsort(inverse, kind="stable")
    grouped_unit = inverse[order]
    grouped_time = times[order]
    grouped_state = state[order]
    grouped_segment = segment[order]
    same_unit = grouped_unit[1:] == grouped_unit[:-1]
    same_segment = grouped_segment[1:] == grouped_segment[:-1]
    delta = np.diff(grouped_time)
    safe = same_unit & same_segment
    interval_state = grouped_state[1:]
    interval_block = block[order[1:]]
    interval_code = interval_block.astype(np.int64) * n_states + interval_state
    denominator = np.bincount(
        interval_code[safe & (delta > 0)], minlength=n_blocks * n_states
    ).reshape(n_blocks, n_states)
    short = np.bincount(
        interval_code[safe & (delta >= 1) & (delta < 30)], minlength=n_blocks * n_states
    ).reshape(n_blocks, n_states)
    exact = np.bincount(
        interval_code[safe & (delta == 0)], minlength=n_blocks * n_states
    ).reshape(n_blocks, n_states)
    lag8 = np.bincount(
        interval_code[safe & (delta == 8)], minlength=n_blocks * n_states
    ).reshape(n_blocks, n_states)
    lag30 = np.bincount(
        interval_code[safe & (delta == 30)], minlength=n_blocks * n_states
    ).reshape(n_blocks, n_states)

    starts = np.r_[0, np.cumsum(unit_counts)[:-1]]
    grouped_depth = depths[order]
    median_depth = np.asarray(
        [np.median(grouped_depth[start : start + count]) for start, count in zip(starts, unit_counts)],
        dtype=float,
    )
    p0, p1 = processing_depth_um
    s0, s1 = scoring_depth_um
    processing = (median_depth >= p0) & (median_depth <= p1)
    edge = processing & ((median_depth < s0) | (median_depth > s1))
    presence = np.mean(block_unit > 0, axis=0)

    block_duration = np.diff(cells.block_edges_s)
    bootstrap_duration = multiplicities @ block_duration
    bootstrap_spikes = multiplicities @ block_spikes
    spike_rate_draws = bootstrap_spikes / bootstrap_duration
    active_unit_draws = np.count_nonzero((multiplicities @ block_unit) > 0, axis=1)
    rate_low, rate_high = percentile_ci(spike_rate_draws)
    yield_low, yield_high = percentile_ci(active_unit_draws)

    point_rows, bootstrap_rows = [], []
    zero = int(np.flatnonzero(cells.states_um == 0.0)[0]) if np.any(cells.states_um == 0.0) else -1
    eligible = unit_state[:, zero] >= minimum_reference_events if zero >= 0 else np.zeros(n_units, bool)
    total_counts_draw = np.einsum("db,bus->dus", multiplicities, block_unit_state, optimize=True)
    exposure_draw = multiplicities @ cells.block_exposure_s
    for state_index, state_um in enumerate(cells.states_um):
        positive_denominator = int(denominator[:, state_index].sum())
        short_count = int(short[:, state_index].sum())
        point_short = short_count / positive_denominator if positive_denominator else math.nan
        draw_denominator = multiplicities @ denominator[:, state_index]
        draw_short = multiplicities @ short[:, state_index]
        draw_fraction = np.divide(
            draw_short, draw_denominator, out=np.full(len(draw_short), np.nan), where=draw_denominator > 0
        )
        short_low, short_high = percentile_ci(draw_fraction)
        point_rho = math.nan
        rho_draws = np.full(len(multiplicities), np.nan, dtype=float)
        if zero >= 0 and eligible.sum() >= 3 and cells.state_exposure_s[state_index] > 0:
            reference_rate = np.log(
                (unit_state[eligible, zero] + 0.5) / cells.state_exposure_s[zero]
            )
            state_rate = np.log(
                (unit_state[eligible, state_index] + 0.5) / cells.state_exposure_s[state_index]
            )
            point_rho = safe_spearman(reference_rate, state_rate)
            for draw in range(len(multiplicities)):
                if exposure_draw[draw, zero] <= 0 or exposure_draw[draw, state_index] <= 0:
                    continue
                x = np.log((total_counts_draw[draw, eligible, zero] + 0.5) / exposure_draw[draw, zero])
                y = np.log(
                    (total_counts_draw[draw, eligible, state_index] + 0.5)
                    / exposure_draw[draw, state_index]
                )
                rho_draws[draw] = safe_spearman(x, y)
        rho_low, rho_high = percentile_ci(rho_draws)
        point_rows.append(
            {
                "state_um": float(state_um),
                "exposure_s": float(cells.state_exposure_s[state_index]),
                "spikes": int(state_counts[state_index]),
                "segment_safe_positive_intervals": positive_denominator,
                "short_interval_count_1_29": short_count,
                "short_interval_fraction_1_29": point_short,
                "short_interval_ci95_low": short_low,
                "short_interval_ci95_high": short_high,
                "exact_duplicate_count": int(exact[:, state_index].sum()),
                "lag8_count": int(lag8[:, state_index].sum()),
                "lag30_count": int(lag30[:, state_index].sum()),
                "state_rate_eligible_units": int(eligible.sum()),
                "state_rate_rho_vs_q0": point_rho,
                "state_rate_rho_ci95_low": rho_low,
                "state_rate_rho_ci95_high": rho_high,
            }
        )
        bootstrap_rows.extend(
            {
                "draw": draw,
                "state_um": float(state_um),
                "short_interval_fraction_1_29": float(draw_fraction[draw]),
                "state_rate_rho_vs_q0": float(rho_draws[draw]),
            }
            for draw in range(len(multiplicities))
        )

    presence_summary = {}
    for index, statistic in enumerate(("median", "p10", "p90")):
        point, low, high = _unit_bootstrap(
            presence, draws=len(multiplicities), seed=seed + 100 + index, statistic=statistic
        )
        presence_summary[statistic] = {"value": point, "ci95_low": low, "ci95_high": high}
    edge_point, edge_low, edge_high = _unit_bootstrap(
        edge[processing].astype(float),
        draws=len(multiplicities), seed=seed + 200, statistic="mean",
    )
    labels = sort["labels"]
    all_safe_intervals = exact.sum(axis=1) + denominator.sum(axis=1)
    exact_by_block = exact.sum(axis=1)
    duplicate_draws = np.divide(
        multiplicities @ exact_by_block,
        multiplicities @ all_safe_intervals,
        out=np.full(len(multiplicities), np.nan),
        where=(multiplicities @ all_safe_intervals) > 0,
    )
    duplicate_fraction = (
        float(exact_by_block.sum() / all_safe_intervals.sum()) if all_safe_intervals.sum() else math.nan
    )
    duplicate_low, duplicate_high = percentile_ci(duplicate_draws)
    duplicate_unit_count = int(
        np.count_nonzero(np.bincount(grouped_unit[1:][same_unit & (delta == 0)], minlength=n_units))
    )
    unit_table = pd.DataFrame(
        {
            "unit_id": unit_ids,
            "spike_count": unit_counts,
            "ks_label": [labels.get(int(unit), "unknown") for unit in unit_ids],
            "presence_ratio": presence,
            "median_depth_um": median_depth,
            "in_processing_domain": processing,
            "edge_unit": edge,
            "reference_state_events": unit_state[:, zero] if zero >= 0 else 0,
            "state_rate_eligible": eligible,
        }
    )
    summary = {
        "raw_unit_count": int(n_units),
        "ks_good_unit_count": int(sum(labels.get(int(unit), "unknown") == "good" for unit in unit_ids)),
        "assigned_spike_count": int(len(times)),
        "assigned_spike_rate_hz": float(len(times) / (num_samples / sampling_frequency_hz)),
        "assigned_spike_rate_ci95": [rate_low, rate_high],
        "bootstrap_active_unit_yield_ci95": [yield_low, yield_high],
        "exact_duplicate_count": int(exact.sum()),
        "units_with_exact_duplicates": duplicate_unit_count,
        "units_with_exact_duplicates_fraction": duplicate_unit_count / n_units if n_units else math.nan,
        "exact_duplicate_fraction": duplicate_fraction,
        "exact_duplicate_fraction_ci95": [duplicate_low, duplicate_high],
        "presence": presence_summary,
        "processing_domain_units": int(processing.sum()),
        "edge_unit_fraction": edge_point,
        "edge_unit_fraction_ci95": [edge_low, edge_high],
    }
    return {
        "summary": summary,
        "state_metrics": pd.DataFrame(point_rows),
        "bootstrap": pd.DataFrame(bootstrap_rows),
        "units": unit_table,
    }


def load_field(config: dict[str, Any], duration_s: float, block_seconds: float) -> tuple[FieldCells, dict[str, Any]]:
    path = Path(config["path"])
    actual = sha256(path)
    if actual != config["sha256"]:
        raise RuntimeError(f"field hash differs: {path}")
    with np.load(path, allow_pickle=False) as saved:
        times = np.asarray(saved[config["time_key"]], dtype=np.float64)
        displacement = np.asarray(saved[config["displacement_key"]], dtype=np.float64)
    if displacement.ndim == 2:
        if config.get("rigid_projection") != "median across depth":
            raise ValueError("two-dimensional field lacks the frozen rigid projection")
        displacement = np.median(displacement, axis=1)
    rounded, reference = rounded_rigid_field(displacement)
    cells = build_field_cells(times, rounded, duration_s=duration_s, block_seconds=block_seconds)
    receipt = {
        "path": str(path.resolve()), "sha256": actual, "reference_median_um": reference,
        "rounded_states_um": cells.states_um.tolist(),
    }
    return cells, receipt


def audit_inputs(config: dict[str, Any]) -> dict[str, Any]:
    rows = {}
    for arm, spec in config["arms"].items():
        root = Path(spec["sorter_output"])
        missing = [name for name in RAW_FILES if not (root / name).is_file()]
        identity_ready = True
        identity_error = None
        identity_path_value = spec.get("sort_identity")
        if identity_path_value:
            identity_path = Path(identity_path_value)
            identity_ready = identity_path.is_file()
            if identity_ready:
                identity = json.loads(identity_path.read_text())
                expected = spec.get("expected_sort_identity")
                identity_ready = bool(expected and identity.get("identity_digest") == expected)
                if not identity_ready:
                    identity_error = "sort identity is missing or unexpected"
            else:
                identity_error = "sort identity receipt is missing"
        complete = True
        receipt = spec.get("complete_receipt")
        if receipt:
            receipt_path = Path(receipt)
            complete = receipt_path.is_file()
            if complete:
                complete = json.loads(receipt_path.read_text()).get("status") == "complete"
        rows[arm] = {
            "ready": not missing and complete and identity_ready,
            "sorter_output": str(root),
            "missing": missing,
            "complete_receipt_ready": complete,
            "sort_identity_ready": identity_ready,
            "sort_identity_error": identity_error,
        }
    return {"schema": SCHEMA, "arms": rows, "all_ready": all(row["ready"] for row in rows.values())}


def validate_arm_input(arm: str, spec: dict[str, Any]) -> dict[str, Any]:
    root = Path(spec["sorter_output"]).resolve()
    files = {}
    for name in RAW_FILES:
        path = root / name
        files[name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    identity_path_value = spec.get("sort_identity")
    identity_digest = None
    if identity_path_value:
        identity_path = Path(identity_path_value)
        identity = json.loads(identity_path.read_text())
        identity_digest = identity.get("identity_digest")
        if identity_digest != spec.get("expected_sort_identity"):
            raise RuntimeError(f"{arm}: unexpected sort identity")
        declared = identity.get("files", {})
        for name in ("spike_times.npy", "spike_clusters.npy", "cluster_KSLabel.tsv"):
            expected = declared.get(name)
            if not isinstance(expected, dict):
                raise RuntimeError(f"{arm}: sort identity does not bind {name}")
            if files[name]["bytes"] != expected.get("size_bytes") or files[name]["sha256"] != expected.get("sha256"):
                raise RuntimeError(f"{arm}: {name} differs from sort identity")
    sorter_manifest = root.parent / "rescue_sort_manifest.json"
    sorter_request_digest = None
    if sorter_manifest.is_file():
        manifest = json.loads(sorter_manifest.read_text())
        if manifest.get("complete") is not True:
            raise RuntimeError(f"{arm}: sorter manifest is incomplete")
        sorter_request_digest = manifest.get("request_digest")
        summary = manifest.get("summary", {})
        if summary.get("final_spike_count") != np.load(root / "spike_times.npy", mmap_mode="r").shape[0]:
            raise RuntimeError(f"{arm}: sorter manifest spike count differs")
        settings = summary.get("critical_saved_settings", {})
        required = {
            "do_correction": False,
            "effective_nblocks": 0,
            "do_CAR": True,
            "artifact_threshold": "Infinity",
        }
        if settings != required:
            raise RuntimeError(f"{arm}: critical saved Kilosort settings differ")
    complete_receipt_value = spec.get("complete_receipt")
    if complete_receipt_value:
        complete_path = Path(complete_receipt_value)
        complete = json.loads(complete_path.read_text())
        if complete.get("status") != "complete":
            raise RuntimeError(f"{arm}: completion receipt is not complete")
        if complete.get("schema") == "en-xr-tier1-raw-sort-handoff-v1":
            from testing.en_xr_tier1_handoff import verify as verify_xr_handoff

            verify_xr_handoff(complete_path.parent, spec.get("expected_sort_identity"))
        completed_sort = complete.get("sort")
        if sorter_request_digest and isinstance(completed_sort, dict):
            if completed_sort.get("request_digest") != sorter_request_digest:
                raise RuntimeError(f"{arm}: completion receipt names another sort request")
    return {
        "arm": arm,
        "sorter_output": str(root),
        "sort_identity": identity_digest,
        "sorter_request_digest": sorter_request_digest,
        "files": files,
    }


def write_table(path: Path, table: pd.DataFrame) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    table.to_csv(temporary, index=False)
    os.replace(temporary, path)


def write_figures(
    output: Path,
    *,
    arm_summaries: dict[str, dict[str, Any]],
    coincidence: dict[str, dict[str, Any]],
    correspondence_results: dict[str, dict[str, Any]],
    state_tables: dict[str, dict[str, pd.DataFrame]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arms = [arm for arm in ("REF", "XR", "A", "B") if arm in arm_summaries]
    x = np.arange(len(arms))
    fig, axes = plt.subplots(2, 3, figsize=(13.0, 7.5), constrained_layout=True)

    def bars(axis, values, title, ylabel, intervals=None):
        colors = [OKABE_ITO[arm] for arm in arms]
        errors = None
        if intervals is not None:
            errors = np.asarray(
                [[max(0.0, value - interval[0]) for value, interval in zip(values, intervals)],
                 [max(0.0, interval[1] - value) for value, interval in zip(values, intervals)]],
                dtype=float,
            )
        axis.bar(x, values, color=colors, alpha=0.78, yerr=errors, capsize=3)
        for position, arm, value in zip(x, arms, values):
            axis.plot(position, value, marker=ARM_MARKERS[arm], color="white",
                      markeredgecolor="black", markersize=6, linestyle="none")
        axis.set(xticks=x, xticklabels=arms, title=title, ylabel=ylabel)
        axis.grid(axis="y", alpha=0.22)

    bars(axes[0, 0], [arm_summaries[arm]["raw_unit_count"] for arm in arms],
         "Raw Kilosort yield", "Units")
    bars(axes[0, 1], [arm_summaries[arm]["assigned_spike_count"] / 1e6 for arm in arms],
         "Assigned spikes", "Millions")
    presence = [arm_summaries[arm]["presence"]["median"]["value"] for arm in arms]
    presence_ci = [[arm_summaries[arm]["presence"]["median"]["ci95_low"],
                    arm_summaries[arm]["presence"]["median"]["ci95_high"]] for arm in arms]
    bars(axes[0, 2], presence, "Median presence ratio", "Fraction", presence_ci)
    excess = [coincidence[arm]["excess"] for arm in arms]
    excess_ci = [coincidence[arm]["excess_ci95"] for arm in arms]
    bars(axes[1, 0], excess, "Chance-aware coincidence excess", "Marked-spike fraction", excess_ci)
    edge = [arm_summaries[arm]["edge_unit_fraction"] for arm in arms]
    edge_ci = [arm_summaries[arm]["edge_unit_fraction_ci95"] for arm in arms]
    bars(axes[1, 1], edge, "Edge-unit fraction", "Fraction", edge_ci)
    f1 = [1.0 if arm == "REF" else correspondence_results[arm]["median_primary_f1"] for arm in arms]
    f1_ci = [[1.0, 1.0] if arm == "REF" else correspondence_results[arm]["median_primary_f1_ci95"]
             for arm in arms]
    bars(axes[1, 2], f1, "Median time-only primary F1", "F1 vs REF", f1_ci)
    for extension in ("png", "pdf"):
        fig.savefig(output / f"TIER1_PANEL.{extension}", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(state_tables), figsize=(6.2 * len(state_tables), 4.5),
                             squeeze=False, constrained_layout=True)
    for axis, (field, tables) in zip(axes[0], state_tables.items()):
        for arm, table in tables.items():
            ordered = table.sort_values("state_um")
            y = ordered.state_rate_rho_vs_q0.to_numpy(float)
            low = ordered.state_rate_rho_ci95_low.to_numpy(float)
            high = ordered.state_rate_rho_ci95_high.to_numpy(float)
            error = np.maximum(0.0, np.vstack([y - low, high - y]))
            axis.errorbar(
                ordered.state_um, y, yerr=error, color=OKABE_ITO[arm], marker=ARM_MARKERS[arm],
                linestyle="-", capsize=3, label=arm,
            )
        axis.axhline(0, color="#777777", linewidth=0.8)
        axis.set(title=f"{field}-field state-rate continuity", xlabel="Rounded state (µm)",
                 ylabel="Spearman ρ vs q=0", ylim=(-1.05, 1.05))
        axis.grid(alpha=0.22)
        axis.legend(frameon=False)
    for extension in ("png", "pdf"):
        fig.savefig(output / f"STATE_RATE_RHO.{extension}", dpi=180)
    plt.close(fig)


def run(config_path: Path, output: Path) -> dict[str, Any]:
    started = time.perf_counter()
    config = json.loads(config_path.read_text())
    if config.get("schema_version") != "en-tier1-panel-v1":
        raise ValueError("unsupported EN Tier-1 contract")
    audit = audit_inputs(config)
    if not audit["all_ready"]:
        raise RuntimeError("not all frozen Tier-1 arm inputs are complete; run --audit-inputs")
    output = output.resolve()
    if output.exists():
        raise FileExistsError(output)
    partial = output.with_name(output.name + ".partial")
    if partial.exists():
        raise FileExistsError(f"preserve or inspect prior partial output: {partial}")
    partial.mkdir(parents=True)
    atomic_json(partial / "STATUS.json", {"stage": "validate_inputs", "updated_at": datetime.now(timezone.utc).isoformat()})

    input_receipts = {
        arm: validate_arm_input(arm, spec) for arm, spec in config["arms"].items()
    }
    scope = config["scope"]
    fs = float(scope["sampling_frequency_hz"])
    num_samples = int(scope["num_samples"])
    duration_s = num_samples / fs
    uncertainty = config["uncertainty"]
    block_seconds = float(uncertainty["time_block_seconds"])
    field_cells, field_receipts = {}, {}
    for field_name, field_spec in config["fields"].items():
        cells, receipt = load_field(field_spec, duration_s, block_seconds)
        field_cells[field_name], field_receipts[field_name] = cells, receipt
    block_edges = next(iter(field_cells.values())).block_edges_s
    if not all(np.array_equal(cells.block_edges_s, block_edges) for cells in field_cells.values()):
        raise RuntimeError("field block grids differ")
    draws = int(uncertainty["bootstrap_draws"])
    seed = int(uncertainty["seed"])
    multiplicities = common_block_multiplicities(len(block_edges) - 1, draws, seed)
    np.save(partial / "COMMON_BLOCK_MULTIPLICITIES.npy", multiplicities, allow_pickle=False)

    atomic_json(partial / "STATUS.json", {"stage": "load_sorts", "updated_at": datetime.now(timezone.utc).isoformat()})
    sorts = {
        arm: load_sort(Path(spec["sorter_output"]), num_samples=num_samples)
        for arm, spec in config["arms"].items()
    }
    processing = tuple(float(value) for value in config["metrics"]["edge"]["processing_depth_um"])
    scoring = tuple(float(value) for value in config["metrics"]["edge"]["scoring_depth_um"])
    minimum_reference = int(config["state_assignment"]["minimum_reference_state_events_per_unit"])
    arm_summaries: dict[str, dict[str, Any]] = {}
    state_results: dict[str, dict[str, Any]] = {}
    state_tables: dict[str, dict[str, pd.DataFrame]] = {}
    unit_tables: dict[str, pd.DataFrame] = {}
    for field_index, (field_name, field_spec) in enumerate(config["fields"].items()):
        state_results[field_name] = {}
        state_tables[field_name] = {}
        for arm_index, arm in enumerate(field_spec["comparison_arms"]):
            atomic_json(
                partial / "STATUS.json",
                {"stage": "arm_state_metrics", "field": field_name, "arm": arm,
                 "updated_at": datetime.now(timezone.utc).isoformat()},
            )
            result = arm_state_metrics(
                sorts[arm], field_cells[field_name], sampling_frequency_hz=fs,
                num_samples=num_samples, multiplicities=multiplicities,
                minimum_reference_events=minimum_reference,
                processing_depth_um=processing, scoring_depth_um=scoring,
                seed=seed + 1000 * field_index + arm_index,
            )
            state_results[field_name][arm] = result["state_metrics"].to_dict("records")
            state_tables[field_name][arm] = result["state_metrics"]
            write_table(partial / f"STATE_METRICS_{field_name}_{arm}.csv", result["state_metrics"])
            write_table(partial / f"STATE_BOOTSTRAP_{field_name}_{arm}.csv", result["bootstrap"])
            write_table(partial / f"UNIT_METRICS_{field_name}_{arm}.csv", result["units"])
            if arm not in arm_summaries:
                arm_summaries[arm] = result["summary"]
                unit_tables[arm] = result["units"]
            else:
                invariant_keys = (
                    "raw_unit_count", "ks_good_unit_count", "assigned_spike_count",
                    "exact_duplicate_count", "processing_domain_units",
                )
                if any(arm_summaries[arm][key] != result["summary"][key] for key in invariant_keys):
                    raise RuntimeError(f"{arm}: field-invariant Tier-1 metric changed")

    coincidence = {}
    coincidence_config = config["metrics"]["chance_aware_coincidence"]
    for arm_index, arm in enumerate(config["arms"]):
        atomic_json(
            partial / "STATUS.json",
            {"stage": "chance_aware_coincidence", "arm": arm,
             "updated_at": datetime.now(timezone.utc).isoformat()},
        )
        coincidence[arm] = chance_aware_coincidence(
            sorts[arm], unit_tables[arm], num_samples=num_samples,
            sampling_frequency_hz=fs, block_edges_s=block_edges,
            multiplicities=multiplicities,
            tolerance_ms=float(coincidence_config["tolerance_ms"]),
            depth_tolerance_um=float(coincidence_config["depth_tolerance_um"]),
            processing_depth_um=processing, seed=seed + 5000 + arm_index,
        )

    correspondence_results = {}
    correspondence_config = config["metrics"]["correspondence"]
    reference = sorts[correspondence_config["reference"]]
    reference_units = np.unique(reference["clusters"])
    tolerance_frames = int(round(float(correspondence_config["tolerance_ms"]) * 1e-3 * fs))
    for arm_index, arm in enumerate(name for name in config["arms"] if name != "REF"):
        atomic_json(
            partial / "STATUS.json",
            {"stage": "correspondence", "arm": arm,
             "updated_at": datetime.now(timezone.utc).isoformat()},
        )
        edges = correspondence(
            reference, sorts[arm], tolerance_frames=tolerance_frames,
            minimum_overlap=float(correspondence_config["minimum_overlap"]),
            primary_retention=float(correspondence_config["reciprocal_primary_minimum_retention"]),
        )
        write_table(partial / f"CORRESPONDENCE_EDGES_REF_{arm}.csv", edges)
        write_table(partial / f"PRIMARY_MATCHES_REF_{arm}.csv", edges[edges.primary_match] if len(edges) else edges)
        correspondence_results[arm] = correspondence_summary(
            edges, reference_units, np.unique(sorts[arm]["clusters"]),
            draws=draws, seed=seed + 7000 + arm_index,
        )

    write_figures(
        partial, arm_summaries=arm_summaries, coincidence=coincidence,
        correspondence_results=correspondence_results, state_tables=state_tables,
    )
    result = {
        "schema": SCHEMA,
        "status": "complete",
        "contract": {
            "path": str(config_path.resolve()), "sha256": sha256(config_path),
            "frozen_status": config["status"],
        },
        "inputs": input_receipts,
        "fields": field_receipts,
        "common_block_multiplicities_sha256": sha256(partial / "COMMON_BLOCK_MULTIPLICITIES.npy"),
        "arm_summaries": arm_summaries,
        "state_profiles": state_results,
        "chance_aware_coincidence": coincidence,
        "correspondence_to_ref": correspondence_results,
        "decision": {
            "status": "ready_for_profile_review",
            "automatic_rank": None,
            "tier2_started": False,
            "reason": "No composite score is computed; apply the frozen EN profile reading.",
        },
        "interpretation": (
            "State/rate rho and time-only correspondence are continuity proxies. "
            "Neither establishes biological identity or purity."
        ),
        "resources": {
            "wall_seconds": time.perf_counter() - started,
            "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "raw_voltage_bytes_read": 0,
            "sorts_launched": 0,
            "rf_accesses": 0,
        },
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(partial / "RESULT.json", result)
    atomic_json(partial / "STATUS.json", {"stage": "complete", "updated_at": datetime.now(timezone.utc).isoformat()})
    products = [
        {"path": str(path.relative_to(partial)), "bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(partial.rglob("*"))
        if path.is_file() and path.name not in {"MANIFEST.json", "COMPLETE.json"}
    ]
    atomic_json(partial / "MANIFEST.json", {"schema": SCHEMA, "products": products})
    atomic_json(
        partial / "COMPLETE.json",
        {"schema": SCHEMA, "status": "complete", "manifest_sha256": sha256(partial / "MANIFEST.json"),
         "written_last_utc": datetime.now(timezone.utc).isoformat()},
    )
    os.replace(partial, output)
    return json.loads((output / "RESULT.json").read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/en_tier1_panel.v1.json"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--audit-inputs", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if config.get("schema_version") != "en-tier1-panel-v1":
        raise ValueError("unsupported EN Tier-1 contract")
    audit = audit_inputs(config)
    if args.audit_inputs:
        print(json.dumps(audit, indent=2, sort_keys=True))
        return
    if not audit["all_ready"]:
        raise RuntimeError("not all frozen Tier-1 arm inputs are complete; run --audit-inputs")
    if args.output is None:
        raise ValueError("--output is required for evaluation")
    print(json.dumps(run(args.config, args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
