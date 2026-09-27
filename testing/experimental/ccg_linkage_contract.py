"""Synthetic CCG/null and merge-graph contract helpers for BN.

Offline diagnostic tooling only; not imported by DARTsort or production code.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _segments_array(segments) -> np.ndarray:
    value = np.asarray(segments, dtype=np.int64)
    if value.ndim != 2 or value.shape[1] != 2 or np.any(value[:, 0] >= value[:, 1]):
        raise ValueError("segments must be nonempty half-open [start, stop) rows")
    if value.shape[0] > 1 and np.any(value[1:, 0] < value[:-1, 1]):
        raise ValueError("segments must be sorted and nonoverlapping")
    return value


def all_pair_lags(
    reference_times,
    target_times,
    *,
    segments,
    max_abs_lag: int,
) -> np.ndarray:
    """Enumerate every oriented target-reference lag within the same segment.

    Segment and lag limits are inclusive at start/±max and exclusive at stop.
    No nearest-neighbor or one-to-one matching is performed.
    """

    if not isinstance(max_abs_lag, (int, np.integer)) or max_abs_lag < 0:
        raise ValueError("max_abs_lag must be a nonnegative integer")
    reference = np.asarray(reference_times, dtype=np.int64)
    target = np.asarray(target_times, dtype=np.int64)
    if reference.ndim != 1 or target.ndim != 1:
        raise ValueError("spike trains must be one-dimensional")
    lags = []
    for start, stop in _segments_array(segments):
        a = reference[(start <= reference) & (reference < stop)]
        b = target[(start <= target) & (target < stop)]
        if not a.size or not b.size:
            continue
        delta = b[:, None] - a[None, :]
        keep = np.abs(delta) <= max_abs_lag
        lags.append(delta[keep])
    if not lags:
        return np.empty(0, dtype=np.int64)
    return np.concatenate(lags).astype(np.int64, copy=False)


def histogram_half_open_last_closed(lags, edges) -> np.ndarray:
    """NumPy histogram semantics: [left,right), with the final right included."""

    lags = np.asarray(lags)
    edges = np.asarray(edges)
    if edges.ndim != 1 or edges.size < 2 or np.any(np.diff(edges) <= 0):
        raise ValueError("edges must be strictly increasing")
    return np.histogram(lags, bins=edges)[0]


@dataclass(frozen=True)
class ShiftNull:
    offsets: np.ndarray
    lags: np.ndarray
    counts: np.ndarray
    exposure_samples: np.ndarray
    common_support_samples: int


def shifted_segment_null(
    reference_times,
    target_times,
    *,
    segments,
    offsets,
    lags,
) -> ShiftNull:
    """Whole-train shifts with no wrap and one common support across offsets.

    Each segment is trimmed by the largest absolute offset on both sides. The
    reference train and every shifted target train are evaluated only inside
    that identical half-open support. Exposure at lag ``d`` is the sum over
    trimmed segments of ``max(0, segment_length - abs(d))``.
    """

    reference = np.asarray(reference_times, dtype=np.int64)
    target = np.asarray(target_times, dtype=np.int64)
    offsets = np.asarray(offsets, dtype=np.int64)
    lag_grid = np.asarray(lags, dtype=np.int64)
    if offsets.ndim != 1 or lag_grid.ndim != 1:
        raise ValueError("offsets and lags must be one-dimensional")
    segs = _segments_array(segments)
    margin = int(np.abs(offsets).max(initial=0))
    counts = np.zeros((offsets.size, lag_grid.size), dtype=np.int64)
    exposure = np.zeros_like(counts)
    common_total = 0
    for start, stop in segs:
        common_start, common_stop = start + margin, stop - margin
        length = int(max(0, common_stop - common_start))
        common_total += length
        if not length:
            continue
        a = reference[(common_start <= reference) & (reference < common_stop)]
        for oi, offset in enumerate(offsets):
            shifted = target + offset
            b = shifted[(common_start <= shifted) & (shifted < common_stop)]
            if a.size and b.size:
                delta = b[:, None] - a[None, :]
                for li, lag in enumerate(lag_grid):
                    counts[oi, li] += np.count_nonzero(delta == lag)
            exposure[oi] += np.maximum(0, length - np.abs(lag_grid))
    return ShiftNull(offsets, lag_grid, counts, exposure, common_total)


def deduplicate_within_labels(times, labels, scores, *, radius_samples: int) -> np.ndarray:
    """Return a keep mask using the reviewed score-priority censoring rule."""

    times = np.asarray(times, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=float)
    if times.ndim != 1 or times.shape != labels.shape or times.shape != scores.shape:
        raise ValueError("times, labels, and scores must be aligned vectors")
    if radius_samples < 0:
        raise ValueError("radius_samples must be nonnegative")
    discard = np.zeros(times.size, dtype=bool)
    for label in np.unique(labels[labels >= 0]):
        original = np.flatnonzero(labels == label)
        order = np.argsort(times[original], kind="stable")
        indices = original[order]
        t = times[indices]
        dt = np.diff(t).astype(float)
        local_discard = np.zeros(t.size, dtype=bool)
        i0 = 0
        while i0 < t.size - 1:
            if dt[i0] > radius_samples:
                i0 += 1
                continue
            i1 = i0 + 1
            while i1 < t.size - 1 and dt[i1] <= radius_samples:
                i1 += 1
            dt_slice = dt[i0:i1]
            for relative in np.argsort(scores[indices[i0 : i1 + 1]])[:-1]:
                ii = i0 + relative
                local_discard[ii] = True
                bridge = np.inf
                if 0 < ii < t.size - 1:
                    bridge = dt[ii] + dt[ii - 1]
                if ii < t.size - 1:
                    dt[ii] = bridge
                if ii > 0:
                    dt[ii - 1] = bridge
                if dt_slice.size and dt_slice.min() > radius_samples:
                    break
            i0 = i1
        discard[indices[local_discard]] = True
    return ~discard


def union_route_mask(qda_mask, force_mask, optional_mask=None) -> np.ndarray:
    qda = np.asarray(qda_mask, dtype=bool)
    force = np.asarray(force_mask, dtype=bool)
    if qda.shape != force.shape or qda.ndim != 2 or qda.shape[0] != qda.shape[1]:
        raise ValueError("route masks must be same-shape square matrices")
    final = np.logical_or(qda, force)
    if optional_mask is not None:
        optional = np.asarray(optional_mask, dtype=bool)
        if optional.shape != final.shape:
            raise ValueError("optional route mask shape mismatch")
        final |= optional
    final |= final.T
    np.fill_diagonal(final, True)
    return final


def single_link_components(mask) -> np.ndarray:
    """Connected components of a symmetric accepted-edge mask."""

    graph = np.asarray(mask, dtype=bool)
    if graph.ndim != 2 or graph.shape[0] != graph.shape[1]:
        raise ValueError("mask must be square")
    graph = np.logical_or(graph, graph.T)
    n = graph.shape[0]
    parent = np.arange(n)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in zip(*np.triu(graph, 1).nonzero()):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri
    roots = np.array([find(i) for i in range(n)])
    _, labels = np.unique(roots, return_inverse=True)
    return labels
