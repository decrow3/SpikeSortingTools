"""Small synthetic audit helpers for DARTsort merge-map identifiability.

This is offline test tooling. It mirrors the candidate aggregation operations
needed by the BM fixtures, but is not imported by DARTsort or a sorting path.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MapInference:
    resolved: dict[int, int]
    ambiguous: dict[int, tuple[int, ...]]
    absent: tuple[int, ...]
    contradictory: tuple[int, ...]
    evidence_rows: dict[int, tuple[int, ...]]


@dataclass(frozen=True)
class AggregatedScores:
    candidates: np.ndarray
    log_liks: np.ndarray
    responsibilities: np.ndarray
    labels: np.ndarray


def infer_map_by_row_intersection(
    original_candidates: np.ndarray,
    merged_candidates: np.ndarray,
    *,
    source_universe: np.ndarray | None = None,
) -> MapInference:
    """Return necessary per-ID targets implied by paired candidate rows.

    A source ID's target must occur in every merged row where that source ID
    occurred. A singleton intersection is identifiable under DARTsort's
    source-equivalent mapping contract. This is not a general graph solver and
    does not infer IDs absent from the observed rows.
    """

    original = np.asarray(original_candidates)
    merged = np.asarray(merged_candidates)
    if original.ndim != 2 or merged.ndim != 2 or original.shape[0] != merged.shape[0]:
        raise ValueError("candidate arrays must be 2D with identical row count")
    if not np.issubdtype(original.dtype, np.integer) or not np.issubdtype(merged.dtype, np.integer):
        raise ValueError("candidate IDs must be integer, with negative values reserved for padding")

    observed = np.unique(original[original >= 0])
    if source_universe is None:
        universe = observed
    else:
        universe = np.asarray(source_universe)
        if universe.ndim != 1 or not np.issubdtype(universe.dtype, np.integer):
            raise ValueError("source_universe must be a 1D integer array")
        if (universe < 0).any() or np.unique(universe).size != universe.size:
            raise ValueError("source_universe must contain unique nonnegative IDs")

    resolved = {}
    ambiguous = {}
    absent = []
    contradictory = []
    evidence_rows = {}
    for source_id_ in universe:
        source_id = int(source_id_)
        rows = np.flatnonzero((original == source_id).any(axis=1))
        evidence_rows[source_id] = tuple(map(int, rows))
        if not rows.size:
            absent.append(source_id)
            continue
        possible = None
        for row in rows:
            row_targets = set(map(int, merged[row][merged[row] >= 0]))
            possible = row_targets if possible is None else possible.intersection(row_targets)
        assert possible is not None
        if len(possible) == 1:
            resolved[source_id] = next(iter(possible))
        elif possible:
            ambiguous[source_id] = tuple(sorted(possible))
        else:
            contradictory.append(source_id)

    return MapInference(
        resolved=resolved,
        ambiguous=ambiguous,
        absent=tuple(absent),
        contradictory=tuple(contradictory),
        evidence_rows=evidence_rows,
    )


def source_equivalent_aggregate(
    candidates: np.ndarray,
    log_liks: np.ndarray,
    responsibilities: np.ndarray,
    source_to_merged: np.ndarray,
) -> AggregatedScores:
    """Mirror `combine_gmm_scores` candidate aggregation for merge fixtures.

    Candidate columns come first and one noise column comes last. Padding is a
    negative integer. The mapping must cover all nonnegative candidate IDs and
    have contiguous nonnegative target IDs. Stable descending likelihood order,
    logaddexp, responsibility summation, padding handling, and
    candidate-wins-noise-ties match the reviewed local source.
    """

    candidates = np.asarray(candidates)
    log_liks = np.asarray(log_liks)
    responsibilities = np.asarray(responsibilities)
    mapping = np.asarray(source_to_merged)
    if candidates.ndim != 2 or not np.issubdtype(candidates.dtype, np.integer):
        raise ValueError("candidates must be a 2D integer array")
    if (candidates < -1).any():
        raise ValueError("candidate padding sentinel must be -1")
    n_rows, n_candidates = candidates.shape
    expected = (n_rows, n_candidates + 1)
    if log_liks.shape != expected or responsibilities.shape != expected:
        raise ValueError("scores must have candidate columns plus one noise column")
    if mapping.ndim != 1 or not np.issubdtype(mapping.dtype, np.integer):
        raise ValueError("source_to_merged must be a 1D integer array")
    if (mapping < 0).any():
        raise ValueError("mapped IDs must be nonnegative")
    unique, counts = np.unique(mapping, return_counts=True)
    if not unique.size or not np.array_equal(unique, np.arange(unique[-1] + 1)):
        raise ValueError("mapped IDs must be contiguous")
    valid = candidates >= 0
    if valid.any() and candidates[valid].max() >= mapping.size:
        raise ValueError("mapping does not cover every observed source ID")
    if unique.size == mapping.size:
        raise ValueError("fixture requires an actual merge; source returns no-merge inputs unchanged")
    if not np.isfinite(log_liks[:, -1]).all() or not np.isfinite(responsibilities).all():
        raise ValueError("noise scores and responsibilities must be finite")
    if responsibilities.shape[1] > 2 and np.diff(responsibilities[:, :-1], axis=1).max() > 1e-3:
        raise ValueError("candidate responsibilities must be in source score order")
    if not np.greater_equal(np.isneginf(log_liks[:, :-1]), candidates < 0).all():
        raise ValueError("padding candidates require -inf likelihoods")

    mapped = np.full_like(candidates, -1)
    mapped[valid] = mapping[candidates[valid]]
    merged_liks = log_liks[:, :-1].copy()
    merged_resp = responsibilities[:, :-1].copy()

    for row in range(n_rows):
        for col in range(n_candidates - 1):
            target = mapped[row, col]
            if target < 0 or counts[target] <= 1:
                continue
            later = np.flatnonzero(mapped[row, col + 1 :] == target) + col + 1
            for other in later:
                mapped[row, other] = -1
                if np.isneginf(merged_liks[row, other]):
                    continue
                merged_resp[row, col] += merged_resp[row, other]
                merged_liks[row, col] = np.logaddexp(
                    merged_liks[row, col], merged_liks[row, other]
                )
                merged_resp[row, other] = 0.0
                merged_liks[row, other] = -np.inf

        for col in range(n_candidates):
            if 0 <= mapped[row, col] < counts.size or merged_resp[row, col] == 0:
                continue
            merged_resp[row, -1] += merged_resp[row, col]
            merged_resp[row, col] = 0.0

    order = np.argsort(-merged_liks, axis=1, kind="stable")
    mapped = np.take_along_axis(mapped, order, axis=1)
    merged_liks = np.take_along_axis(merged_liks, order, axis=1)
    merged_resp = np.take_along_axis(merged_resp, order, axis=1)
    merged_liks = np.concatenate((merged_liks, log_liks[:, -1:]), axis=1)
    merged_resp = np.concatenate((merged_resp, responsibilities[:, -1:]), axis=1)
    labels = np.where(merged_liks[:, 0] >= merged_liks[:, -1], mapped[:, 0], -1)
    return AggregatedScores(mapped, merged_liks, merged_resp, labels)
