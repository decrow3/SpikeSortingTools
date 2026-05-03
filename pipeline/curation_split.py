"""
Split candidate scoring: rank units by evidence for potential splitting.

Usage
-----
from pipeline.curation_split import score_split_candidates

df = score_split_candidates(
    clu, tF, spike_times_s, amplitudes, spike_z,
    out_path=comp_dir / 'split_candidates.csv',
)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

from .curation_evidence import isi_violation_rate


# ---------------------------------------------------------------------------
# Bimodality / stationarity helpers
# ---------------------------------------------------------------------------

def _gmm_bic_delta(data: np.ndarray, n_init: int = 3) -> float:
    """
    BIC(1-GMM) − BIC(2-GMM).
    Positive → unimodal wins.
    Negative → bimodal wins (split candidate signal).
    """
    from sklearn.mixture import GaussianMixture
    data = np.asarray(data, dtype=float).reshape(-1, 1)
    if len(data) < 20:
        return np.nan
    try:
        bic1 = GaussianMixture(1, n_init=n_init, random_state=0).fit(data).bic(data)
        bic2 = GaussianMixture(2, n_init=n_init, random_state=0).fit(data).bic(data)
        return float(bic1 - bic2)
    except Exception:
        return np.nan


def _tF_bimodality_score(unit_tF: np.ndarray) -> float:
    """
    Project unit spike features onto first PC; return GMM BIC delta.
    Negative = bimodal in feature space → split candidate.
    """
    if unit_tF.shape[0] < 20:
        return np.nan
    flat = unit_tF.reshape(unit_tF.shape[0], -1).astype(float)
    flat -= flat.mean(axis=0)
    try:
        _, _, Vt = np.linalg.svd(flat, full_matrices=False)
        pc1 = flat @ Vt[0]
        return _gmm_bic_delta(pc1)
    except Exception:
        return np.nan


def _amplitude_bimodality(unit_amplitudes: np.ndarray) -> float:
    """GMM BIC delta on amplitude distribution. Negative = bimodal."""
    return _gmm_bic_delta(unit_amplitudes)


def _rate_stationarity(unit_st_times_s: np.ndarray, n_windows: int = 10) -> float:
    """
    Coefficient of variation of firing rate across n_windows equal time bins.
    High CV → unstable rate (state changes, drift, or merged distinct unit epochs).
    """
    if len(unit_st_times_s) < n_windows * 5:
        return np.nan
    t0, t1 = float(unit_st_times_s.min()), float(unit_st_times_s.max())
    if t1 <= t0:
        return np.nan
    edges  = np.linspace(t0, t1, n_windows + 1)
    dur    = (t1 - t0) / n_windows
    counts = np.array([
        ((unit_st_times_s >= edges[i]) & (unit_st_times_s < edges[i + 1])).sum() / dur
        for i in range(n_windows)
    ], dtype=float)
    mean_rate = counts.mean()
    return float(counts.std() / mean_rate) if mean_rate > 0 else np.nan


def _temporal_waveform_drift(
    unit_tF: np.ndarray,
    unit_st_times_s: np.ndarray,
) -> float:
    """
    Cosine distance between mean tF feature vectors for first and second half
    of the recording (ordered by spike time).
    0 = perfectly stable; approaches 2 = completely reversed.
    High drift ≠ split candidate on its own — pair with bimodality.
    """
    if unit_tF.shape[0] < 20:
        return np.nan
    order      = np.argsort(unit_st_times_s)
    tF_ordered = unit_tF[order].reshape(unit_tF.shape[0], -1).astype(float)
    mid        = len(tF_ordered) // 2
    c1 = tF_ordered[:mid].mean(axis=0)
    c2 = tF_ordered[mid:].mean(axis=0)
    denom = np.linalg.norm(c1) * np.linalg.norm(c2) + 1e-10
    return float(1.0 - np.dot(c1, c2) / denom)


# ---------------------------------------------------------------------------
# Public: score all units
# ---------------------------------------------------------------------------

def score_split_candidates(
    clu: np.ndarray,
    tF: Optional[np.ndarray],
    spike_times_s: np.ndarray,
    amplitudes: np.ndarray,
    spike_z: np.ndarray,
    fs: int = 30000,
    min_spikes: int = 200,
    refractory_ms: float = 1.5,
    out_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Score every unit for potential splitting and return a ranked DataFrame.

    Parameters
    ----------
    clu           : spike cluster IDs (n_spikes,)
    tF            : KS4 feature matrix (n_spikes, nC, n_pcs) or None
    spike_times_s : spike times in seconds
    amplitudes    : spike amplitudes
    spike_z       : spike depths µm
    fs            : sampling rate
    min_spikes    : units below this are flagged as too_few_spikes and not scored
    refractory_ms : ISI violation threshold
    out_path      : if given, write CSV here

    Returns
    -------
    pd.DataFrame ranked by split_priority descending
    """
    rows = []

    for uid in np.unique(clu):
        mask = clu == uid
        n    = int(mask.sum())

        unit_st   = spike_times_s[mask]
        unit_amps = amplitudes[mask]
        unit_z    = spike_z[mask]
        unit_tF   = tF[mask] if tF is not None else None

        row: dict = dict(
            unit_id  = int(uid),
            n_spikes = n,
            depth_um = float(np.median(unit_z)),
        )

        if n < min_spikes:
            row.update(
                isi_viol_rate=np.nan, amp_bimodality=np.nan,
                tF_bimodality=np.nan, rate_cv=np.nan,
                waveform_drift=np.nan, split_priority=np.nan,
                recommended_action='too_few_spikes',
            )
            rows.append(row)
            continue

        isi  = isi_violation_rate(unit_st, refractory_ms)
        abic = _amplitude_bimodality(unit_amps)
        tfbic= _tF_bimodality_score(unit_tF) if unit_tF is not None else np.nan
        rcv  = _rate_stationarity(unit_st)
        drift= _temporal_waveform_drift(unit_tF, unit_st) if unit_tF is not None else np.nan

        row['isi_viol_rate']  = isi
        row['amp_bimodality'] = abic
        row['tF_bimodality']  = tfbic
        row['rate_cv']        = rcv
        row['waveform_drift'] = drift

        # Composite priority: more-negative BIC delta (stronger bimodality) → higher
        # priority; ISI violations add a direct contamination penalty.
        # Normalise loosely so both terms contribute on a similar scale.
        amp_term = abic  if np.isfinite(abic)  else 0.0
        tf_term  = tfbic if np.isfinite(tfbic) else 0.0
        priority = -(0.5 * amp_term + 0.5 * tf_term) + 200.0 * isi
        row['split_priority'] = float(priority)

        is_bimodal     = (np.isfinite(tfbic) and tfbic < -50) or (np.isfinite(abic) and abic < -50)
        is_contaminated = isi > 0.05
        is_drifting    = np.isfinite(drift) and drift > 0.30

        if is_contaminated and is_bimodal:
            action = 'split_review'
        elif is_drifting and not is_bimodal:
            action = 'drift_review'
        elif isi > 0.20:
            action = 'mua'
        else:
            action = 'ok'

        row['recommended_action'] = action
        rows.append(row)

    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values('split_priority', ascending=False, na_position='last')

    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)

    return df
