"""Motion-aware post-clustering utilities without AP-voltage resampling.

This module is deliberately independent of Kilosort's merge implementation.  It
provides the geometry, per-spike depth, field-support, state-template, and
state-aggregation primitives needed for a paired static/motion-aware replay.
It does not accept merges on its own.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


MERGE_REPLAY_SCHEMA = "medicine-aware-ks4-merge-replay-v1"


def _numpy(value, dtype=None) -> np.ndarray:
    """Convert NumPy/CPU Torch arraylikes without importing Torch."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    out = np.asarray(value)
    return out.astype(dtype, copy=False) if dtype is not None else out


def feature_channel_indices(
    detection_templates: np.ndarray, i_cc, i_u
) -> np.ndarray:
    """Return the physical channel for every spike-local ``tF`` column."""
    templates = np.asarray(detection_templates, dtype=np.int64).reshape(-1)
    i_cc = _numpy(i_cc, np.int64)
    i_u = _numpy(i_u, np.int64).reshape(-1)
    if i_cc.ndim != 2 or i_u.ndim != 1:
        raise ValueError("iCC must be 2-D and iU must be 1-D")
    if np.any(templates < 0) or np.any(templates >= i_u.size):
        raise ValueError("detection template outside iU")
    anchors = i_u[templates]
    if np.any(anchors < 0) or np.any(anchors >= i_cc.shape[1]):
        raise ValueError("template anchor outside iCC")
    return i_cc[:, anchors].T


def squared_pc_feature_depths(
    t_f: np.ndarray,
    detection_templates: np.ndarray,
    i_cc,
    i_u,
    channel_y_um: np.ndarray,
    *,
    chunk_spikes: int = 250_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce Kilosort's squared-PC-energy depth centroid.

    Returns ``(depth_um, eligible)``.  Zero/nonfinite-energy spikes are NaN and
    ineligible rather than clipped or filled.
    """
    t_f = np.asarray(t_f)
    templates = np.asarray(detection_templates, dtype=np.int64).reshape(-1)
    y = np.asarray(channel_y_um, dtype=np.float64).reshape(-1)
    if t_f.ndim != 3 or t_f.shape[0] != templates.size:
        raise ValueError("tF must be spike x local-channel x PC")
    if chunk_spikes < 1:
        raise ValueError("chunk_spikes must be positive")
    channels = feature_channel_indices(templates, i_cc, i_u)
    if channels.shape != t_f.shape[:2] or np.any(channels >= y.size):
        raise ValueError("feature-channel mapping disagrees with tF/geometry")
    depth = np.full(templates.size, np.nan, dtype=np.float64)
    eligible = np.zeros(templates.size, dtype=bool)
    for lo in range(0, templates.size, chunk_spikes):
        hi = min(templates.size, lo + chunk_spikes)
        values = np.asarray(t_f[lo:hi], dtype=np.float64)
        energy = np.sum(values * values, axis=2)
        total = np.sum(energy, axis=1)
        good = np.isfinite(total) & (total > 0) & np.all(np.isfinite(energy), axis=1)
        if np.any(good):
            z = np.sum(energy[good] * y[channels[lo:hi][good]], axis=1) / total[good]
            depth[lo:hi][good] = z
            eligible[lo:hi][good] = np.isfinite(z)
    return depth, eligible


def exact_same_column_pairs(
    channel_positions_um: np.ndarray,
    shift_um: float,
    *,
    channel_shanks: np.ndarray | None = None,
    atol: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Map source channels to real targets after an exact y translation.

    Returned arrays are ``(target, source)`` so a shifted source template is
    compared as ``target_template[target]`` versus ``source_template[source]``.
    X coordinate and shank must be identical.
    """
    pos = np.asarray(channel_positions_um, dtype=np.float64)
    if pos.ndim != 2 or pos.shape[1] < 2 or not np.all(np.isfinite(pos[:, :2])):
        raise ValueError("channel positions must be finite N x >=2")
    shanks = (
        np.zeros(pos.shape[0], dtype=np.int64)
        if channel_shanks is None
        else np.asarray(channel_shanks).reshape(-1)
    )
    if shanks.size != pos.shape[0] or not np.isfinite(shift_um):
        raise ValueError("invalid shanks or shift")
    scale = 1.0 / atol
    lookup = {
        (int(round(x * scale)), int(round(y * scale)), str(s)): i
        for i, ((x, y), s) in enumerate(zip(pos[:, :2], shanks))
    }
    if len(lookup) != pos.shape[0]:
        raise ValueError("duplicate channel coordinates within shank")
    source, target = [], []
    for i, ((x, y), s) in enumerate(zip(pos[:, :2], shanks)):
        key = (int(round(x * scale)), int(round((y + shift_um) * scale)), str(s))
        if key in lookup:
            source.append(i)
            target.append(lookup[key])
    return np.asarray(target, dtype=np.int64), np.asarray(source, dtype=np.int64)


def aligned_template_similarity(
    target_template: np.ndarray,
    source_template: np.ndarray,
    channel_positions_um: np.ndarray,
    shift_source_um: float,
    *,
    channel_shanks: np.ndarray | None = None,
    temporal_lags: Iterable[int] = (0,),
    min_channels: int = 1,
) -> dict[str, float | int]:
    """Maximum cosine over declared temporal lags after an exact y shift."""
    a = np.asarray(target_template, dtype=np.float64)
    b = np.asarray(source_template, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 2 or a.shape[0] != len(channel_positions_um):
        raise ValueError("templates must have equal channel x feature shape")
    target, source = exact_same_column_pairs(
        channel_positions_um, shift_source_um, channel_shanks=channel_shanks
    )
    if target.size < min_channels:
        return {"score": np.nan, "lag": 0, "overlap_channels": int(target.size)}
    aa, bb = a[target], b[source]
    best_score, best_lag = -np.inf, 0
    for lag in tuple(int(x) for x in temporal_lags):
        if abs(lag) >= a.shape[1]:
            continue
        if lag > 0:
            av, bv = aa[:, lag:], bb[:, :-lag]
        elif lag < 0:
            av, bv = aa[:, :lag], bb[:, -lag:]
        else:
            av, bv = aa, bb
        denom = np.linalg.norm(av) * np.linalg.norm(bv)
        score = float(np.sum(av * bv) / denom) if denom > 0 else np.nan
        if np.isfinite(score) and score > best_score:
            best_score, best_lag = score, lag
    if best_score == -np.inf:
        best_score = np.nan
    return {
        "score": float(best_score),
        "lag": int(best_lag),
        "overlap_channels": int(target.size),
    }


def medicine_basis_weights(
    peak_times_s: np.ndarray,
    peak_depths_um: np.ndarray,
    time_grid_s: np.ndarray,
    depth_grid_um: np.ndarray,
    *,
    time_kernel_width_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Audit peak support using MEDiCINe's triangular time/depth bases.

    This is a support diagnostic, not statistical confidence.  It returns raw
    nonzero counts, total weights, and Kish effective sample size per grid knot.
    """
    times = np.asarray(peak_times_s, dtype=np.float64).reshape(-1)
    depths = np.asarray(peak_depths_um, dtype=np.float64).reshape(-1)
    tg = np.asarray(time_grid_s, dtype=np.float64).reshape(-1)
    zg = np.asarray(depth_grid_um, dtype=np.float64).reshape(-1)
    if times.shape != depths.shape or tg.size < 1 or zg.size < 1:
        raise ValueError("invalid peak or grid arrays")
    if time_kernel_width_s <= 0 or not np.all(np.diff(tg) > 0) or not np.all(np.diff(zg) > 0):
        raise ValueError("invalid grid/kernel")
    if not np.all(np.isfinite(times)) or not np.all(np.isfinite(depths)):
        raise ValueError("nonfinite peaks")
    depth_span = zg[-1] - zg[0]
    if depth_span <= 0 and zg.size > 1:
        raise ValueError("invalid depth span")
    zn = (depths - zg[0]) / depth_span if zg.size > 1 else np.zeros_like(depths)
    levels = np.linspace(0.0, 1.0, zg.size)
    depth_width = (1.0 / max(1, zg.size - 1)) + 1e-3
    order = np.argsort(times, kind="stable")
    times = times[order]
    zn = zn[order]
    raw = np.zeros((tg.size, zg.size), dtype=np.int64)
    total = np.zeros_like(raw, dtype=np.float64)
    sumsq = np.zeros_like(raw, dtype=np.float64)
    half_width = 0.5 * time_kernel_width_s
    for ti, center in enumerate(tg):
        lo = np.searchsorted(times, center - half_width, side="right")
        hi = np.searchsorted(times, center + half_width, side="left")
        if hi <= lo:
            continue
        local_t = times[lo:hi]
        local_z = zn[lo:hi]
        wt = np.maximum(0.0, 1.0 - np.abs(local_t - center) / half_width)
        for zi, level in enumerate(levels):
            wz = np.maximum(0.0, depth_width - np.abs(local_z - level))
            w = wt * wz
            active = w > 0
            raw[ti, zi] = int(np.sum(active))
            total[ti, zi] = np.sum(w[active])
            sumsq[ti, zi] = np.sum(w[active] ** 2)
    effective = np.divide(total * total, sumsq, out=np.zeros_like(total), where=sumsq > 0)
    return raw, total, effective


def sample_eligible_field(
    time_grid_s: np.ndarray,
    depth_grid_um: np.ndarray,
    displacement_um: np.ndarray,
    eligible_grid: np.ndarray,
    query_time_s: np.ndarray,
    query_depth_um: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Bilinear field sampling requiring every active corner to be eligible."""
    t = np.asarray(time_grid_s, dtype=np.float64)
    z = np.asarray(depth_grid_um, dtype=np.float64)
    d = np.asarray(displacement_um, dtype=np.float64)
    e = np.asarray(eligible_grid, dtype=bool)
    qt = np.asarray(query_time_s, dtype=np.float64).reshape(-1)
    qz = np.asarray(query_depth_um, dtype=np.float64).reshape(-1)
    if d.shape != (t.size, z.size) or e.shape != d.shape or qt.shape != qz.shape:
        raise ValueError("field/query shape mismatch")
    if t.size < 2 or z.size < 2 or not np.all(np.diff(t) > 0) or not np.all(np.diff(z) > 0):
        raise ValueError("field grids must be increasing with >=2 knots")
    th = np.clip(np.searchsorted(t, qt, side="right"), 1, t.size - 1)
    zh = np.clip(np.searchsorted(z, qz, side="right"), 1, z.size - 1)
    tl, zl = th - 1, zh - 1
    tf = (qt - t[tl]) / (t[th] - t[tl])
    zf = (qz - z[zl]) / (z[zh] - z[zl])
    weights = ((1-tf)*(1-zf), tf*(1-zf), (1-tf)*zf, tf*zf)
    corners = ((tl, zl), (th, zl), (tl, zh), (th, zh))
    good = np.isfinite(qt) & np.isfinite(qz) & (qt >= t[0]) & (qt <= t[-1]) & (qz >= z[0]) & (qz <= z[-1])
    out = np.zeros(qt.size, dtype=np.float64)
    for w, ix in zip(weights, corners):
        active = w > np.finfo(float).eps
        good &= ~active | (e[ix] & np.isfinite(d[ix]))
        out += w * np.nan_to_num(d[ix], nan=0.0)
    out[~good] = np.nan
    return out, good


@dataclass(frozen=True)
class StatePair:
    state_a: int
    state_b: int
    score: float
    covered_spikes_a: int
    covered_spikes_b: int
    compatible: bool = True
    rival_veto: bool = False


@dataclass(frozen=True)
class StateTemplate:
    cluster: int
    state: int
    displacement_center_um: float
    spike_count: int
    template: np.ndarray


def build_motion_state_templates(
    t_f: np.ndarray,
    cluster_ids: np.ndarray,
    feature_channels: np.ndarray,
    displacement_um: np.ndarray,
    eligible: np.ndarray,
    *,
    n_channels: int,
    state_step_um: float = 40.0,
    min_spikes: int = 100,
) -> tuple[list[StateTemplate], dict[int, int]]:
    """Embed saved local PC features and average them by cluster/motion state.

    Missing physical channels remain zero for a spike, matching Kilosort's
    global ``Wall`` embedding.  The template mean divides by state spike count,
    not per-channel observation count.
    """
    t_f = np.asarray(t_f)
    clu = np.asarray(cluster_ids, dtype=np.int64).reshape(-1)
    channels = np.asarray(feature_channels, dtype=np.int64)
    displacement = np.asarray(displacement_um, dtype=np.float64).reshape(-1)
    good = np.asarray(eligible, dtype=bool).reshape(-1)
    n = t_f.shape[0]
    if t_f.ndim != 3 or clu.size != n or channels.shape != t_f.shape[:2] or displacement.size != n or good.size != n:
        raise ValueError("state-template inputs disagree")
    if n_channels < 1 or state_step_um <= 0 or min_spikes < 1:
        raise ValueError("invalid state-template settings")
    if np.any(channels < 0) or np.any(channels >= n_channels):
        raise ValueError("feature channel outside geometry")
    good &= np.isfinite(displacement) & np.all(np.isfinite(t_f), axis=(1, 2))
    states = np.zeros(n, dtype=np.int64)
    states[good] = np.rint(displacement[good] / state_step_um).astype(np.int64)
    keys = np.stack((clu[good], states[good]), axis=1)
    if not len(keys):
        return [], {int(c): int(np.sum((clu == c) & ~good)) for c in np.unique(clu)}
    unique, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    templates = []
    good_indices = np.flatnonzero(good)
    for group, ((cluster, state), count) in enumerate(zip(unique, counts)):
        if count < min_spikes:
            continue
        members = good_indices[inverse == group]
        accumulator = np.zeros((n_channels, t_f.shape[2]), dtype=np.float64)
        for index in members:
            np.add.at(accumulator, channels[index], t_f[index])
        accumulator /= int(count)
        templates.append(StateTemplate(
            cluster=int(cluster), state=int(state),
            displacement_center_um=float(state * state_step_um),
            spike_count=int(count), template=accumulator.astype(np.float32),
        ))
    ineligible = {int(c): int(np.sum((clu == c) & ~good)) for c in np.unique(clu)}
    return templates, ineligible


def aggregate_reciprocal_state_pairs(
    pairs: Iterable[StatePair],
    *,
    eligible_spikes_a: int,
    eligible_spikes_b: int,
    min_pairs: int,
    min_coverage: float,
    score_threshold: float,
    incompatibility_threshold: float,
) -> dict[str, object]:
    """Aggregate reciprocal-best states without maximizing across states."""
    pairs = list(pairs)
    if eligible_spikes_a <= 0 or eligible_spikes_b <= 0 or min_pairs < 1:
        raise ValueError("invalid aggregation denominators")
    best_b = {}
    best_a = {}
    for p in pairs:
        if not np.isfinite(p.score):
            continue
        if p.state_a not in best_b or p.score > best_b[p.state_a].score:
            best_b[p.state_a] = p
        if p.state_b not in best_a or p.score > best_a[p.state_b].score:
            best_a[p.state_b] = p
    reciprocal = [p for p in pairs if best_b.get(p.state_a) is p and best_a.get(p.state_b) is p]
    reciprocal.sort(key=lambda p: (p.state_a, p.state_b))
    scores = np.asarray([p.score for p in reciprocal], dtype=np.float64)
    covered_a = sum({p.state_a: p.covered_spikes_a for p in reciprocal}.values())
    covered_b = sum({p.state_b: p.covered_spikes_b for p in reciprocal}.values())
    coverage_a = min(1.0, covered_a / eligible_spikes_a)
    coverage_b = min(1.0, covered_b / eligible_spikes_b)
    veto = any((not p.compatible) or p.rival_veto or p.score < incompatibility_threshold for p in reciprocal)
    median = float(np.median(scores)) if scores.size else np.nan
    lower_quartile = float(np.quantile(scores, 0.25)) if scores.size else np.nan
    accepted = bool(
        len(reciprocal) >= min_pairs
        and min(coverage_a, coverage_b) >= min_coverage
        and np.isfinite(lower_quartile)
        and lower_quartile >= score_threshold
        and not veto
    )
    return {
        "reciprocal_pairs": [(p.state_a, p.state_b) for p in reciprocal],
        "pair_count": len(reciprocal),
        "median_score": median,
        "lower_quartile_score": lower_quartile,
        "coverage_a": float(coverage_a),
        "coverage_b": float(coverage_b),
        "veto": bool(veto),
        "accepted": accepted,
    }
