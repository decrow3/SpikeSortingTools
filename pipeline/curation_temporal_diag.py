"""
Temporal handoff diagnostics for amp_bic and posthoc merge candidates.

For each proposed pair (a, b), this module asks:

  Are these units temporally complementary fragments of the same cell,
  or unrelated units connected by weak amplitude / feature coincidence?

Six metrics are computed per pair:

  depth_diff_um                   spatial locality sanity check
  template_cosine                 waveform compatibility (Wall)
  time_overlap_frac               fraction of active bins where both fire
  handoff_score                   anti-correlation of time-binned FRs
  union_amp_smoothness_gain       does merging produce a smoother amplitude trajectory?
  coactivity_refractory_violation ISI violation rate restricted to overlap bins

True temporal split:    small depth, high cosine, strong handoff,
                        smooth union trajectory, low coactive ISI.
False positive:         large depth OR low cosine, no handoff,
                        no trajectory improvement, or high coactive ISI.

Usage
-----
from pipeline.curation_temporal_diag import build_temporal_diagnostics

df = build_temporal_diagnostics(
    merge_groups_by_method={'amp_bic': [...], 'posthoc': [...]},
    clu=clu_base, spike_times_s=st_s, amplitudes=amps,
    spike_z=spz, Wall=Wall,
    out_path=comp_dir / 'candidate_pair_temporal_diagnostics.csv',
)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from itertools import combinations
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _time_binned_rates(
    spike_times_s: np.ndarray, t_min: float, t_max: float, bin_s: float
) -> np.ndarray:
    edges = np.arange(t_min, t_max + bin_s, bin_s)
    counts, _ = np.histogram(spike_times_s, bins=edges)
    return counts.astype(float)


def _roughness(x: np.ndarray) -> float:
    """Mean absolute first difference — lower = smoother trajectory."""
    finite = x[np.isfinite(x)]
    if len(finite) < 2:
        return 0.0
    return float(np.mean(np.abs(np.diff(finite))))


def _amp_trajectory(
    spike_times_s: np.ndarray, amplitudes: np.ndarray,
    t_min: float, t_max: float, bin_s: float,
) -> np.ndarray:
    """Median amplitude per time bin; bins with no spikes → NaN."""
    edges = np.arange(t_min, t_max + bin_s, bin_s)
    n = len(edges) - 1
    out = np.full(n, np.nan)
    for i in range(n):
        m = (spike_times_s >= edges[i]) & (spike_times_s < edges[i + 1])
        if m.sum() > 0:
            out[i] = float(np.median(amplitudes[m]))
    return out


def _isi_violation_rate(spike_times_s: np.ndarray, refractory_ms: float = 1.5) -> float:
    if len(spike_times_s) < 2:
        return 0.0
    isis = np.diff(np.sort(spike_times_s))
    return float(np.mean(isis < refractory_ms * 1e-3))


# ---------------------------------------------------------------------------
# Per-pair metrics
# ---------------------------------------------------------------------------

def _handoff_score(counts_a: np.ndarray, counts_b: np.ndarray) -> float:
    """
    Anti-correlation of time-binned firing rates.
    Positive → A and B trade off (one active when the other is quiet).
    """
    if len(counts_a) < 4 or counts_a.std() == 0 or counts_b.std() == 0:
        return np.nan
    return float(np.corrcoef(counts_a, -counts_b)[0, 1])


def _time_overlap_frac(
    counts_a: np.ndarray, counts_b: np.ndarray, min_count: int = 1,
) -> float:
    """Fraction of bins (where either is active) in which both are active."""
    active_a = counts_a >= min_count
    active_b = counts_b >= min_count
    n_either = int((active_a | active_b).sum())
    if n_either == 0:
        return 0.0
    return float((active_a & active_b).sum() / n_either)


def _union_amp_smoothness_gain(
    times_a: np.ndarray, amps_a: np.ndarray,
    times_b: np.ndarray, amps_b: np.ndarray,
    t_min: float, t_max: float, bin_s: float,
) -> float:
    """
    roughness(A) + roughness(B) − roughness(A∪B).
    Positive → merging makes the amplitude trajectory smoother → supports same drifting unit.
    Negative → merging makes it rougher → units are distinct.
    """
    traj_a = _amp_trajectory(times_a, amps_a, t_min, t_max, bin_s)
    traj_b = _amp_trajectory(times_b, amps_b, t_min, t_max, bin_s)
    traj_u = _amp_trajectory(
        np.concatenate([times_a, times_b]),
        np.concatenate([amps_a,  amps_b]),
        t_min, t_max, bin_s,
    )
    rough_single = _roughness(traj_a) + _roughness(traj_b)
    rough_union  = _roughness(traj_u)
    return float(rough_single - rough_union)


def _coactivity_refractory_violation(
    times_a: np.ndarray, times_b: np.ndarray,
    counts_a: np.ndarray, counts_b: np.ndarray,
    t_min: float, t_max: float,
    bin_s: float, refractory_ms: float, min_count: int = 1,
) -> float:
    """
    ISI violation rate of the merged spike train, computed *only* in time bins
    where both units are simultaneously active.

    NaN if the two units never co-fire (no overlap bins) — in that case the
    refractory check is vacuous and should not be used as evidence either way.
    """
    active_a = counts_a >= min_count
    active_b = counts_b >= min_count
    overlap_bins = np.where(active_a & active_b)[0]
    if len(overlap_bins) == 0:
        return np.nan

    edges = np.arange(t_min, t_max + bin_s, bin_s)
    merged = []
    for i in overlap_bins:
        t0, t1 = float(edges[i]), float(edges[i + 1])
        merged.append(times_a[(times_a >= t0) & (times_a < t1)])
        merged.append(times_b[(times_b >= t0) & (times_b < t1)])

    merged_times = np.concatenate(merged) if merged else np.array([])
    return _isi_violation_rate(merged_times, refractory_ms)


# ---------------------------------------------------------------------------
# Public: per-pair diagnosis
# ---------------------------------------------------------------------------

def diagnose_pair(
    a: int,
    b: int,
    clu: np.ndarray,
    spike_times_s: np.ndarray,
    amplitudes: np.ndarray,
    spike_z: np.ndarray,
    Wall: np.ndarray,
    bin_s: float = 60.0,
    refractory_ms: float = 1.5,
) -> dict:
    """
    Compute temporal handoff diagnostics for a candidate merge pair (a, b).

    Parameters
    ----------
    a, b          : cluster IDs in clu
    clu           : spike cluster assignments (no-merge base)
    spike_times_s : spike times in seconds
    amplitudes    : spike amplitudes
    spike_z       : spike depths µm
    Wall          : (N_clusters, n_chan, n_tp) — indexed by cluster ID
    bin_s         : time bin width in seconds
    refractory_ms : ISI violation threshold
    """
    mask_a = clu == a
    mask_b = clu == b
    times_a = spike_times_s[mask_a]
    times_b = spike_times_s[mask_b]

    base = dict(unit_a=int(a), unit_b=int(b), n_a=int(mask_a.sum()), n_b=int(mask_b.sum()))

    if len(times_a) == 0 or len(times_b) == 0:
        return base

    amps_a = amplitudes[mask_a]
    amps_b = amplitudes[mask_b]
    z_a    = spike_z[mask_a]
    z_b    = spike_z[mask_b]

    t_min = float(min(times_a.min(), times_b.min()))
    t_max = float(max(times_a.max(), times_b.max()))

    counts_a = _time_binned_rates(times_a, t_min, t_max, bin_s)
    counts_b = _time_binned_rates(times_b, t_min, t_max, bin_s)

    # Template cosine via Wall (indexed directly by cluster ID)
    if int(a) < len(Wall) and int(b) < len(Wall):
        w_a = Wall[int(a)].ravel().astype(float)
        w_b = Wall[int(b)].ravel().astype(float)
        denom = np.linalg.norm(w_a) * np.linalg.norm(w_b)
        template_cosine = float(np.dot(w_a, w_b) / denom) if denom > 1e-10 else np.nan
    else:
        template_cosine = np.nan

    depth_a = float(np.median(z_a))
    depth_b = float(np.median(z_b))

    base.update(dict(
        depth_a_um    = depth_a,
        depth_b_um    = depth_b,
        depth_diff_um = abs(depth_a - depth_b),
        template_cosine = template_cosine,
        time_overlap_frac = _time_overlap_frac(counts_a, counts_b),
        handoff_score = _handoff_score(counts_a, counts_b),
        union_amp_smoothness_gain = _union_amp_smoothness_gain(
            times_a, amps_a, times_b, amps_b, t_min, t_max, bin_s),
        post_merge_isi = _isi_violation_rate(
            np.sort(np.concatenate([times_a, times_b])), refractory_ms),
        coactivity_refractory_violation = _coactivity_refractory_violation(
            times_a, times_b, counts_a, counts_b, t_min, t_max,
            bin_s, refractory_ms),
    ))
    return base


# ---------------------------------------------------------------------------
# Public: build full diagnostics table
# ---------------------------------------------------------------------------

def build_temporal_diagnostics(
    merge_groups_by_method: dict,
    clu: np.ndarray,
    spike_times_s: np.ndarray,
    amplitudes: np.ndarray,
    spike_z: np.ndarray,
    Wall: np.ndarray,
    bin_s: float = 60.0,
    refractory_ms: float = 1.5,
    max_pairs_per_group: Optional[int] = 100,
    out_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    For every candidate pair from amp_bic / posthoc merge groups, compute
    temporal handoff diagnostics and assign classification labels.

    Parameters
    ----------
    merge_groups_by_method : {method: list_of_groups}  — from strategy caches
    clu, spike_times_s, amplitudes, spike_z            — no-merge base arrays
    Wall                   : (N_clusters, n_chan, n_tp) — indexed by cluster ID
    bin_s                  : time bin width in seconds
    max_pairs_per_group    : cap pairs per group (None = unlimited); large
                             connected components can have O(n²) pairs
    out_path               : if given, write CSV here
    """
    rows = []
    for method, groups in merge_groups_by_method.items():
        seen_pairs: set = set()
        for group in groups:
            pair_list = list(combinations(group, 2))
            if max_pairs_per_group is not None and len(pair_list) > max_pairs_per_group:
                print(f'  [{method}] group {group[:3]}… has {len(pair_list)} pairs '
                      f'— sampling {max_pairs_per_group}')
                rng = np.random.default_rng(0)
                idx = rng.choice(len(pair_list), max_pairs_per_group, replace=False)
                pair_list = [pair_list[i] for i in idx]

            for a, b in pair_list:
                key = frozenset([int(a), int(b)])
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
                row = diagnose_pair(
                    a, b, clu, spike_times_s, amplitudes, spike_z, Wall,
                    bin_s=bin_s, refractory_ms=refractory_ms,
                )
                row['method'] = method
                rows.append(row)

    if not rows:
        return pd.DataFrame()

    col_order = [
        'method', 'unit_a', 'unit_b', 'n_a', 'n_b',
        'depth_a_um', 'depth_b_um', 'depth_diff_um', 'template_cosine',
        'time_overlap_frac', 'handoff_score', 'union_amp_smoothness_gain',
        'post_merge_isi', 'coactivity_refractory_violation',
    ]
    df = pd.DataFrame(rows)
    df = df[[c for c in col_order if c in df.columns]]

    # Classification labels — thresholds are a starting point; expect to tune
    df['looks_temporal_split'] = (
        (df['depth_diff_um'] < 75) &
        (df['template_cosine'] > 0.90) &
        (df['handoff_score'] > 0.30) &
        (df['union_amp_smoothness_gain'] > 0) &
        (df['coactivity_refractory_violation'].fillna(1.0) < 0.05)
    )
    df['reject_nonlocal'] = (
        (df['depth_diff_um'] > 150) | (df['template_cosine'] < 0.85)
    )
    df['reject_coactive_contam'] = (
        (df['time_overlap_frac'] > 0.20) &
        (df['coactivity_refractory_violation'].fillna(0.0) > 0.05)
    )

    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)

    return df
