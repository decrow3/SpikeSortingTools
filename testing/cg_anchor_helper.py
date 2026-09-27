"""Freeze and score a bounded same-state raw-snippet anchor pilot.

The helper operates only in one post-TMM stage-unit namespace. Selection is
blind to waveform outcomes: rest/bounds/collision masks are applied before the
100-event cap, units in multiunit force components in either construction arm
are excluded, and ranking uses event counts plus corrected-3000 support
coverage. Waveform scoring is rest reproducibility, not identity validation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class AnchorCandidate:
    unit_id: int
    depth_um: float
    depth_bin: int
    peak_registered_channel: int
    support_channels: tuple[int, ...]
    support_coverage: float
    construction_count: int
    n_half1_pool: int
    n_half2_pool: int
    blocks_half1: int
    blocks_half2: int
    eligible: bool
    reason: str


def array_sha256(value: np.ndarray) -> str:
    """Hash dtype, shape and C-order bytes for a frozen-array receipt."""
    value = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(value.dtype.str.encode())
    h.update(str(value.shape).encode())
    h.update(value.view(np.uint8))
    return h.hexdigest()


def force_connected_units(expanded_a: np.ndarray, expanded_b: np.ndarray) -> np.ndarray:
    """Units with an off-diagonal force relation in either native arm."""
    first, second = np.asarray(expanded_a, bool), np.asarray(expanded_b, bool)
    if first.ndim != 2 or first.shape != second.shape or first.shape[0] != first.shape[1]:
        raise ValueError("force masks must be same-shape square arrays")
    union = first | second
    union = union.copy()
    np.fill_diagonal(union, False)
    return union.any(axis=0) | union.any(axis=1)


def _physical_support(peak_xy, physical_geom, radius_um):
    distance = np.linalg.norm(np.asarray(physical_geom, float) - peak_xy, axis=1)
    return np.flatnonzero(distance <= radius_um)


def build_candidate_table(*, labels, times_samples, rest_mask, bounds_mask,
                          collision_mask, split_sample, sampling_frequency,
                          templates_3000, spike_counts_by_channel_3000,
                          registered_geom, physical_geom, construction_counts,
                          expanded_force_30000, expanded_force_3000,
                          radius_um=150.0, n_depth_bins=8, cap_per_half=100,
                          min_blocks_per_half=10, block_seconds=5.0):
    """Build the pre-outcome candidate table in the native stage namespace."""
    labels = np.asarray(labels)
    times = np.asarray(times_samples)
    masks = [np.asarray(x, bool) for x in (rest_mask, bounds_mask, collision_mask)]
    if any(x.shape != labels.shape for x in [times, *masks]):
        raise ValueError("event arrays and masks must have identical shapes")
    templates = np.asarray(templates_3000, float)
    counts_by_channel = np.asarray(spike_counts_by_channel_3000)
    rgeom, pgeom = np.asarray(registered_geom, float), np.asarray(physical_geom, float)
    counts = np.asarray(construction_counts)
    n_units = templates.shape[0]
    if templates.ndim != 3 or counts_by_channel.shape != (n_units, templates.shape[2]):
        raise ValueError("template/count shapes disagree")
    if rgeom.shape != (templates.shape[2], 2) or pgeom.ndim != 2 or pgeom.shape[1] != 2:
        raise ValueError("geometry must be [channel,2]")
    if counts.shape != (n_units,):
        raise ValueError("construction_counts must contain one count per unit")
    force_excluded = force_connected_units(expanded_force_30000, expanded_force_3000)
    if force_excluded.shape != (n_units,):
        raise ValueError("force masks and templates disagree on unit count")

    # Every eligibility filter precedes event capping.
    usable = masks[0] & masks[1] & ~masks[2] & (labels >= 0) & (labels < n_units)
    block_samples = int(round(float(block_seconds) * float(sampling_frequency)))
    if block_samples <= 0:
        raise ValueError("block duration must be positive")
    blocks = times // block_samples

    amp = np.ptp(templates, axis=1) * np.sqrt(np.maximum(counts_by_channel, 0))
    amp = np.nan_to_num(amp, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)
    peak = amp.argmax(axis=1)
    peak_xy = rgeom[peak]
    lo, hi = float(pgeom[:, 1].min()), float(pgeom[:, 1].max())
    span = max(hi - lo, np.finfo(float).eps)
    result = []
    for unit in range(n_units):
        support = _physical_support(peak_xy[unit], pgeom, radius_um)
        # Exact/nearest physical-to-registered mapping is frozen for coverage only.
        nearest = np.argmin(
            np.linalg.norm(pgeom[support, None, :] - rgeom[None, :, :], axis=2), axis=1
        ) if support.size else np.empty(0, dtype=int)
        covered = counts_by_channel[unit, nearest] > 0 if support.size else np.empty(0, bool)
        coverage = float(covered.mean()) if support.size else 0.0
        iu = np.flatnonzero(usable & (labels == unit))
        ih1, ih2 = iu[times[iu] < split_sample], iu[times[iu] >= split_sample]
        b1, b2 = np.unique(blocks[ih1]).size, np.unique(blocks[ih2]).size
        reasons = []
        if force_excluded[unit]: reasons.append("force_connected")
        if min(len(ih1), len(ih2)) < cap_per_half: reasons.append("insufficient_events")
        if min(b1, b2) < min_blocks_per_half: reasons.append("insufficient_blocks")
        if support.size < 4 or coverage <= 0: reasons.append("insufficient_support")
        depth_bin = min(n_depth_bins - 1, max(0, int((peak_xy[unit, 1] - lo) / span * n_depth_bins)))
        result.append(AnchorCandidate(
            unit_id=unit, depth_um=float(peak_xy[unit, 1]), depth_bin=depth_bin,
            peak_registered_channel=int(peak[unit]),
            support_channels=tuple(map(int, support)), support_coverage=coverage,
            construction_count=int(counts[unit]), n_half1_pool=len(ih1),
            n_half2_pool=len(ih2), blocks_half1=int(b1), blocks_half2=int(b2),
            eligible=not reasons, reason=";".join(reasons)))
    return result


def _rank_key(candidate: AnchorCandidate):
    return (-min(candidate.blocks_half1, candidate.blocks_half2),
            -min(candidate.n_half1_pool, candidate.n_half2_pool),
            -candidate.support_coverage, -candidate.construction_count,
            candidate.unit_id)


def select_depth_stratified(candidates: Iterable[AnchorCandidate], n_select=8):
    """Choose one per occupied depth bin, then deterministic depth-spread backfill."""
    eligible = [x for x in candidates if x.eligible]
    by_bin = {}
    for candidate in eligible:
        by_bin.setdefault(candidate.depth_bin, []).append(candidate)
    chosen = [sorted(by_bin[key], key=_rank_key)[0] for key in sorted(by_bin)]
    chosen = chosen[:n_select]
    remaining = [x for x in eligible if x not in chosen]
    while len(chosen) < n_select and remaining:
        if chosen:
            spread = {x.unit_id: min(abs(x.depth_um - y.depth_um) for y in chosen)
                      for x in remaining}
            remaining.sort(key=lambda x: (-spread[x.unit_id], *_rank_key(x)))
        else:
            remaining.sort(key=_rank_key)
        chosen.append(remaining.pop(0))
    return chosen


def _balanced_cap(indices, blocks, cap, seed):
    indices, blocks = np.asarray(indices), np.asarray(blocks)
    if len(indices) < cap:
        raise ValueError("missing_pool")
    rng = np.random.default_rng(seed)
    pools = {}
    for block in np.unique(blocks):
        values = indices[blocks == block].copy()
        rng.shuffle(values)
        pools[int(block)] = list(values)
    selected = []
    keys = sorted(pools)
    while len(selected) < cap:
        progressed = False
        for key in keys:
            if pools[key]:
                selected.append(pools[key].pop())
                progressed = True
                if len(selected) == cap:
                    break
        if not progressed:
            raise ValueError("missing_pool")
    return np.asarray(sorted(selected), dtype=indices.dtype)


def freeze_event_rows(*, selected_candidates, event_row_ids, labels, times_samples,
                      rest_mask, bounds_mask, collision_mask, split_sample,
                      sampling_frequency, cap_per_half=100, block_seconds=5.0,
                      seed=0):
    """Freeze exactly 100 prefiltered rest rows per half for each selected unit."""
    rows, labels, times = map(np.asarray, (event_row_ids, labels, times_samples))
    rest, bounds, collision = map(np.asarray, (rest_mask, bounds_mask, collision_mask))
    if any(x.shape != rows.shape for x in (labels, times, rest, bounds, collision)):
        raise ValueError("row/event arrays must align")
    usable = rest.astype(bool) & bounds.astype(bool) & ~collision.astype(bool)
    block_samples = int(round(float(block_seconds) * float(sampling_frequency)))
    blocks = times // block_samples
    frozen = {}
    for candidate in selected_candidates:
        unit = int(candidate.unit_id)
        iu = np.flatnonzero(usable & (labels == unit))
        halves = []
        for half, mask in enumerate((times[iu] < split_sample, times[iu] >= split_sample)):
            pool = iu[mask]
            if np.unique(blocks[pool]).size < 10:
                raise ValueError(f"unit_{unit}_half_{half + 1}_missing_pool")
            take = _balanced_cap(pool, blocks[pool], cap_per_half,
                                 seed + 1009 * unit + 97 * half)
            halves.append({"event_indices": take, "event_row_ids": rows[take],
                           "blocks": blocks[take]})
        frozen[unit] = {"half1": halves[0], "half2": halves[1]}
    return frozen


def _cosine(first, second):
    first, second = np.asarray(first).ravel(), np.asarray(second).ravel()
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    return float(first @ second / denominator) if denominator else np.nan


def half1_noise_std(snippets_half1, edge_samples=20):
    """Frozen robust channel noise from half-1 waveform edges only."""
    snippets = np.asarray(snippets_half1, float)
    if snippets.ndim != 3 or snippets.shape[1] < 2 * edge_samples:
        raise ValueError("snippets must be [event,time,channel] with enough edge samples")
    edge = np.concatenate((snippets[:, :edge_samples], snippets[:, -edge_samples:]), axis=1)
    edge = edge - np.median(edge, axis=1, keepdims=True)
    noise = 1.4826 * np.median(np.abs(edge - np.median(edge, axis=(0, 1))), axis=(0, 1))
    return noise


def _conditioned_template(snippets, support, noise):
    template = np.median(np.asarray(snippets, float), axis=0)[:, support]
    template -= template.mean(axis=0, keepdims=True)
    return template / noise[support][None]


def score_anchor_reproducibility(snippets_half1, snippets_half2, *, blocks_half1,
                                 blocks_half2, support_channels, n_bootstrap=1000,
                                 seed=0, ci_alpha=0.05):
    """Score fixed-support, zero-lag/unit-gain rest reproducibility."""
    first, second = np.asarray(snippets_half1, float), np.asarray(snippets_half2, float)
    b1, b2 = np.asarray(blocks_half1), np.asarray(blocks_half2)
    support = np.asarray(support_channels, int)
    result = {"status": "unresolved", "qualified": False,
              "requested_replicates": int(n_bootstrap), "accepted_replicates": 0}
    if first.ndim != 3 or second.ndim != 3 or first.shape[1:] != second.shape[1:]:
        result["reason"] = "shape_mismatch"; return result
    if len(first) != 100 or len(second) != 100 or len(b1) != 100 or len(b2) != 100:
        result["reason"] = "requires_exactly_100_events_per_half"; return result
    if min(np.unique(b1).size, np.unique(b2).size) < 10:
        result["reason"] = "insufficient_blocks"; return result
    if support.size < 4 or support.min(initial=0) < 0 or support.max(initial=-1) >= first.shape[2]:
        result["reason"] = "invalid_support"; return result
    noise = half1_noise_std(first)
    if not np.isfinite(noise[support]).all() or np.any(noise[support] <= 0):
        result["reason"] = "nonfinite_half1_noise"; return result
    if not np.isfinite(first[:, :, support]).all() or not np.isfinite(second[:, :, support]).all():
        result["reason"] = "missing_frozen_support"; return result
    point = _cosine(_conditioned_template(first, support, noise),
                    _conditioned_template(second, support, noise))
    rng = np.random.default_rng(seed)
    samples = []
    u1, u2 = np.unique(b1), np.unique(b2)
    for _ in range(n_bootstrap):
        d1, d2 = rng.choice(u1, len(u1), replace=True), rng.choice(u2, len(u2), replace=True)
        i1 = np.concatenate([np.flatnonzero(b1 == block) for block in d1])
        i2 = np.concatenate([np.flatnonzero(b2 == block) for block in d2])
        value = _cosine(_conditioned_template(first[i1], support, noise),
                        _conditioned_template(second[i2], support, noise))
        if not np.isfinite(value):
            result["reason"] = "nonfinite_bootstrap"; return result
        samples.append(value)
    values = np.asarray(samples)
    lo, hi = np.quantile(values, (ci_alpha / 2, 1 - ci_alpha / 2))
    result.update(status="qualified_anchor" if lo >= 0.90 else "unresolved",
                  qualified=bool(lo >= 0.90), reason="" if lo >= 0.90 else "reliability_ci_below_0.90",
                  point_reliability=float(point), ci_lower=float(lo), ci_upper=float(hi),
                  accepted_replicates=len(values), noise_std=noise,
                  signal_rms=float(np.sqrt(np.mean(_conditioned_template(first, support, noise) ** 2))),
                  lag_samples=0, gain=1.0, support_channels=support.copy())
    return result


def competitor_diagnostics(anchor_template, competitor_templates, *, support_channels,
                           noise_std):
    """Return ranked cosine diagnostics only; never use them as an identity gate."""
    anchor = np.asarray(anchor_template, float)
    competitors = np.asarray(competitor_templates, float)
    support, noise = np.asarray(support_channels, int), np.asarray(noise_std, float)
    a = anchor[:, support] - anchor[:, support].mean(axis=0, keepdims=True)
    a = a / noise[support][None]
    values = []
    for index, template in enumerate(competitors):
        b = template[:, support] - template[:, support].mean(axis=0, keepdims=True)
        values.append((index, _cosine(a, b / noise[support][None])))
    return sorted(values, key=lambda item: (-np.nan_to_num(item[1], nan=-np.inf), item[0]))


def candidate_records(candidates):
    return [asdict(candidate) for candidate in candidates]
