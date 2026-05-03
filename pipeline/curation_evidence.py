"""
Curation evidence table: compute per-merge-group quality metrics and produce
a machine-readable curation_moves.csv with rule-based accept/reject decisions.

Usage
-----
from pipeline.curation_evidence import build_evidence_table, DEFAULT_MERGE_RULES

df = build_evidence_table(
    all_strategy_groups,   # dict: strategy → list of merge groups
    clu, st_samples, spike_times_s, amplitudes, spike_z,
    templates=templates,   # (n_units, n_tp, n_chan) from templates.npy
    templates_ind=templates_ind,
    out_path=comp_dir / 'curation_moves.csv',
)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from itertools import combinations
from pathlib import Path
from typing import Optional

from spikeinterface.postprocessing.correlograms import correlogram_for_one_segment
from .curation_postpatch import _fit_gmm_bic


# ---------------------------------------------------------------------------
# Contamination helpers
# ---------------------------------------------------------------------------

def isi_violation_rate(spike_times_s: np.ndarray, refractory_ms: float = 1.5) -> float:
    """Fraction of consecutive ISIs shorter than refractory_ms for a single unit."""
    if len(spike_times_s) < 2:
        return 0.0
    isis = np.diff(np.sort(spike_times_s))
    return float((isis < refractory_ms * 1e-3).sum() / len(isis))


def contamination_rate_per_unit(
    spike_times_s: np.ndarray,
    spike_clusters: np.ndarray,
    refractory_ms: float = 1.5,
) -> dict[int, float]:
    """Per-unit ISI violation rate (simple contamination proxy)."""
    return {
        int(uid): isi_violation_rate(spike_times_s[spike_clusters == uid], refractory_ms)
        for uid in np.unique(spike_clusters)
    }


# ---------------------------------------------------------------------------
# Per-group evidence metrics
# ---------------------------------------------------------------------------

def _template_cosine_matrix(
    group: list[int],
    templates: np.ndarray,
    templates_ind: Optional[np.ndarray],
) -> np.ndarray:
    """
    Pairwise cosine similarity between full templates for all members in group.
    templates: (n_units, n_tp, n_chan)
    Returns (n, n) symmetric matrix; diagonal = 1.
    """
    n = len(group)
    mat = np.ones((n, n), dtype=float)
    for i, ui in enumerate(group):
        for j, uj in enumerate(group):
            if i >= j:
                continue
            if ui >= templates.shape[0] or uj >= templates.shape[0]:
                mat[i, j] = mat[j, i] = np.nan
                continue
            wi = templates[ui].flatten()
            wj = templates[uj].flatten()
            sim = float(np.dot(wi, wj) / (np.linalg.norm(wi) * np.linalg.norm(wj) + 1e-10))
            mat[i, j] = mat[j, i] = sim
    return mat


def _post_merge_isi_rate(
    group: list[int],
    clu: np.ndarray,
    spike_times_s: np.ndarray,
    refractory_ms: float = 1.5,
) -> float:
    """ISI violation rate for the hypothetical merged unit (union of spike trains)."""
    mask = np.isin(clu, group)
    if mask.sum() < 2:
        return 0.0
    return isi_violation_rate(spike_times_s[mask], refractory_ms)


def _amp_bic_delta(
    group: list[int],
    clu: np.ndarray,
    amplitudes: np.ndarray,
) -> float:
    """
    BIC(1-GMM) − BIC(2-GMM) on combined amplitude distribution.
    Positive  → one component wins  (supports merge).
    Negative  → two components win  (opposes merge).
    """
    combined = amplitudes[np.isin(clu, group)]
    if len(combined) < 10:
        return np.nan
    try:
        return float(_fit_gmm_bic(combined, 1) - _fit_gmm_bic(combined, 2))
    except Exception:
        return np.nan


def _ccg_raw_score(
    group: list[int],
    clu: np.ndarray,
    st_samples: np.ndarray,
    fs: int = 30000,
    nlags_ms: float = 100.0,
    binsize_ms: float = 2.0,
) -> float:
    """
    Mean pairwise Pearson correlation of normalised ACG/CCG vectors across all
    pairs in the group.  Higher = spike trains more similar.
    Returns nan if any pair has < 100 spikes.
    """
    bin_samp = max(1, int(round(binsize_ms * fs / 1000)))
    win_samp = max(1, int(round(2 * nlags_ms * fs / 1000)))

    pair_scores = []
    for a, b in combinations(group, 2):
        ta = st_samples[clu == a].astype(np.int64)
        tb = st_samples[clu == b].astype(np.int64)
        if len(ta) < 100 or len(tb) < 100:
            pair_scores.append(np.nan)
            continue

        spike_t = np.concatenate([ta, tb])
        spike_u = np.concatenate([np.zeros(len(ta), np.int64), np.ones(len(tb), np.int64)])
        order   = np.argsort(spike_t, kind='mergesort')
        corr    = correlogram_for_one_segment(
            spike_times=spike_t[order],
            spike_unit_indices=spike_u[order],
            window_size=win_samp,
            bin_size=bin_samp,
        )

        center = corr.shape[2] // 2
        xc1 = corr[0, 0, :].astype(float); xc1[center] = 0
        xc2 = corr[1, 1, :].astype(float); xc2[center] = 0
        xc3 = 0.5 * (corr[0, 1, :].astype(float) + corr[1, 0, ::-1].astype(float))

        def _s5(x):
            return np.convolve(x, np.ones(5) / 5, mode='same')

        def _nfun(x):
            n = np.linalg.norm(x)
            return x / n if n > 0 else x

        xc1, xc2, xc3 = [_nfun(_s5(x)) for x in (xc1, xc2, xc3)]
        c = np.corrcoef(np.stack([xc1, xc2, xc3]))
        pair_scores.append(float(np.mean([c[0, 1], c[0, 2], c[1, 2]])))

    valid = [s for s in pair_scores if not np.isnan(s)]
    return float(np.mean(valid)) if valid else np.nan


def _cross_strategy_support(
    group: list[int],
    all_strategy_groups: dict[str, list[list[int]]],
    own_strategy: str,
) -> list[str]:
    """
    Names of other strategies that propose a group sharing ≥1 member pair
    with this group.
    """
    candidate_pairs = {frozenset([a, b]) for a, b in combinations(group, 2)}
    supporting = []
    for strat, groups in all_strategy_groups.items():
        if strat == own_strategy:
            continue
        for g in groups:
            alt_pairs = {frozenset([a, b]) for a, b in combinations(g, 2)}
            if candidate_pairs & alt_pairs:
                supporting.append(strat)
                break
    return supporting


# ---------------------------------------------------------------------------
# Evidence row assembly
# ---------------------------------------------------------------------------

DEFAULT_MERGE_RULES: dict = {
    'max_depth_spread_um':  150.0,  # reject if member depths span > this
    'min_cosine':            0.70,  # reject if worst pairwise template cosine < this
    'max_post_merge_isi':    0.05,  # reject if post-merge ISI violation rate > 5 %
    'min_strategies_agree':     0,  # cross-strategy agreement check (0 = disabled)
    'bic_delta_min':         -50.0, # reject if BIC strongly favours 2-GMM
}


def compute_merge_evidence(
    group: list[int],
    strategy_name: str,
    clu: np.ndarray,
    st_samples: np.ndarray,
    spike_times_s: np.ndarray,
    amplitudes: np.ndarray,
    spike_z: np.ndarray,
    templates: Optional[np.ndarray],
    templates_ind: Optional[np.ndarray],
    all_strategy_groups: dict[str, list[list[int]]],
    fs: int = 30000,
    refractory_ms: float = 1.5,
) -> dict:
    """Compute all evidence metrics for one proposed merge group."""
    group = [int(u) for u in group]

    unit_depths = {}
    unit_nspikes = {}
    for uid in group:
        mask = clu == uid
        unit_nspikes[uid] = int(mask.sum())
        unit_depths[uid]  = float(np.median(spike_z[mask])) if mask.any() else np.nan

    depths = [d for d in unit_depths.values() if np.isfinite(d)]
    depth_spread = float(max(depths) - min(depths)) if len(depths) >= 2 else 0.0

    # Template cosine
    min_cosine = mean_cosine = np.nan
    if templates is not None:
        cosine_mat = _template_cosine_matrix(group, templates, templates_ind)
        off_diag   = cosine_mat[~np.eye(len(group), dtype=bool)]
        min_cosine  = float(np.nanmin(off_diag))
        mean_cosine = float(np.nanmean(off_diag))

    bic_delta  = _amp_bic_delta(group, clu, amplitudes)
    ccg_score  = _ccg_raw_score(group, clu, st_samples, fs=fs)
    pre_isi    = max(isi_violation_rate(spike_times_s[clu == u], refractory_ms) for u in group)
    post_isi   = _post_merge_isi_rate(group, clu, spike_times_s, refractory_ms)
    supporting = _cross_strategy_support(group, all_strategy_groups, strategy_name)

    return dict(
        strategy              = strategy_name,
        members               = ' '.join(str(u) for u in sorted(group)),
        n_members             = len(group),
        n_spikes_members      = ' '.join(str(unit_nspikes[u]) for u in sorted(group)),
        depth_spread_um       = depth_spread,
        min_cosine            = min_cosine,
        mean_cosine           = mean_cosine,
        bic_delta             = bic_delta,
        ccg_score             = ccg_score,
        pre_merge_max_isi     = pre_isi,
        post_merge_isi_viol   = post_isi,
        cross_strategy_support= ','.join(supporting),
        n_strategies_agree    = len(supporting),
    )


def _apply_rules(row: dict, rules: dict) -> tuple[str, list[str]]:
    """Return (decision, reject_reasons) for one evidence row."""
    reasons = []

    if row['depth_spread_um'] > rules['max_depth_spread_um']:
        reasons.append(
            f"depth_spread={row['depth_spread_um']:.0f}>{rules['max_depth_spread_um']}"
        )
    if np.isfinite(row['min_cosine']) and row['min_cosine'] < rules['min_cosine']:
        reasons.append(f"min_cosine={row['min_cosine']:.2f}<{rules['min_cosine']}")
    if row['post_merge_isi_viol'] > rules['max_post_merge_isi']:
        reasons.append(
            f"post_isi={row['post_merge_isi_viol']:.3f}>{rules['max_post_merge_isi']}"
        )
    if row['n_strategies_agree'] < rules['min_strategies_agree']:
        reasons.append(
            f"n_agree={row['n_strategies_agree']}<{rules['min_strategies_agree']}"
        )
    if np.isfinite(row['bic_delta']) and row['bic_delta'] < rules['bic_delta_min']:
        reasons.append(f"bic_delta={row['bic_delta']:.0f}<{rules['bic_delta_min']}")

    return ('accept', []) if not reasons else ('reject', reasons)


# ---------------------------------------------------------------------------
# Public: build full table
# ---------------------------------------------------------------------------

def build_evidence_table(
    all_strategy_groups: dict[str, list[list[int]]],
    clu: np.ndarray,
    st_samples: np.ndarray,
    spike_times_s: np.ndarray,
    amplitudes: np.ndarray,
    spike_z: np.ndarray,
    templates: Optional[np.ndarray] = None,
    templates_ind: Optional[np.ndarray] = None,
    fs: int = 30000,
    refractory_ms: float = 1.5,
    rules: Optional[dict] = None,
    out_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Build a complete evidence table for every proposed merge group across all
    strategies and save to curation_moves.csv.

    Parameters
    ----------
    all_strategy_groups : dict mapping strategy name → list of merge groups
    clu                 : spike cluster array (post-dup-removal, pre-merge)
    st_samples          : spike times in samples  (same spike ordering as clu)
    spike_times_s       : spike times in seconds
    amplitudes          : spike amplitude array
    spike_z             : spike depth (µm) per spike
    templates           : (n_units, n_tp, n_chan) loaded from templates.npy
    templates_ind       : (n_units, n_chan) loaded from templates_ind.npy
    fs                  : sampling rate
    refractory_ms       : ISI violation threshold
    rules               : accept/reject thresholds (DEFAULT_MERGE_RULES if None)
    out_path            : if provided, CSV is written here

    Returns
    -------
    pd.DataFrame — one row per proposed merge group
    """
    if rules is None:
        rules = DEFAULT_MERGE_RULES

    rows = []
    for strat, groups in all_strategy_groups.items():
        for group in groups:
            ev = compute_merge_evidence(
                group, strat,
                clu, st_samples, spike_times_s,
                amplitudes, spike_z,
                templates, templates_ind,
                all_strategy_groups,
                fs=fs, refractory_ms=refractory_ms,
            )
            decision, reasons = _apply_rules(ev, rules)
            ev['decision']       = decision
            ev['reject_reasons'] = ';'.join(reasons)
            rows.append(ev)

    _cols = [
        'strategy', 'members', 'n_members', 'n_spikes_members',
        'depth_spread_um', 'min_cosine', 'mean_cosine', 'bic_delta',
        'ccg_score', 'pre_merge_max_isi', 'post_merge_isi_viol',
        'cross_strategy_support', 'n_strategies_agree',
        'decision', 'reject_reasons',
    ]
    df = pd.DataFrame(rows, columns=_cols) if rows else pd.DataFrame(columns=_cols)

    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)

    return df
