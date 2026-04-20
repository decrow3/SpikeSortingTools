#%%
"""compare_patched_vs_dredge.py

Direct, 1-to-1 comparison of two SpikeInterface/Kilosort pipeline outputs:
- "dredge" (pre-existing results)
- "patched" (claim-mask patch / modified KS4)

This script intentionally does NOT assume a shallow sweep layout (no run_* folders).
It compares two pipeline roots like:
  /mnt/NPX/.../dredge_pipeline_results_<sess>_<stream>
  /mnt/NPX/.../patch_pipeline_results_<sess>_<stream>

Outputs (written to OUT_DIR):

When comparing a single stage, outputs are written directly to OUT_DIR.
When comparing multiple stages (e.g. COMPARE_STAGE=both), stage outputs are written to:
    OUT_DIR/pre/...
    OUT_DIR/post/...

Per-stage outputs:
- agreement_scores.csv
- best_matches_dredge_to_patched.csv
- best_matches_patched_to_dredge.csv
- summary.json
- fig_overview.(pdf|png)

Additional comparisons (when COMPARE_STAGE=both):
- OUT_DIR/dredge_pre_vs_post/*
- OUT_DIR/patched_pre_vs_post/*

Root overview (always written to OUT_DIR):
- fig_overview.(pdf|png)  # 4-way condition overview

Always written to OUT_DIR:
- condition_stats.csv
- condition_stats.json
- comparisons_summary.json

Environment overrides:
- DREDGE_PIPE, PATCH_PIPE: absolute paths to pipeline roots
- COMPARE_STAGE: postcuration, precuration, or both (default)
- COMPARE_OUTDIR: output directory
- COMPARE_DELTA_MS: match window in ms (default 0.4)
- COMPARE_MIN_AGREEMENT: threshold for "well-matched" (default 0.5)
- COMPARE_GOOD_ONLY: 1/true to restrict analysis to KSLabel=='good'

"""

from __future__ import annotations

from pathlib import Path
import json
import os
from typing import cast

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import spikeinterface.full as si
import spikeinterface.comparison as scmp
from spikeinterface.core import BaseSorting


# =============================================================================
# Optional split diagnostics (compare_shallow_sweeps-style)
# =============================================================================

SPLIT_DIAGNOSTICS = (os.environ.get("COMPARE_SPLIT_DIAGNOSTICS", "1").strip().lower() in {"1", "true", "yes", "y"})
REF_CONDITION = (os.environ.get("COMPARE_REF_CONDITION", "dredge_pre") or "dredge_pre").strip().lower()
SPLIT_BIN_S = float(os.environ.get("COMPARE_SPLIT_BIN_S", "10.0"))
SPLIT_MIN_SPIKES = int(os.environ.get("COMPARE_SPLIT_MIN_SPIKES", "300"))
SPLIT_MAX_PAGES_PDF = int(os.environ.get("COMPARE_SPLIT_MAX_PAGES_PDF", "60"))

# Heuristic flagging thresholds (ported from compare_shallow_sweeps.py)
SPLIT_SCORE_THRESH = float(os.environ.get("COMPARE_SPLIT_SCORE_THRESH", "0.55"))
SPLIT_CHILD2_MIN_FRAC = float(os.environ.get("COMPARE_SPLIT_CHILD2_MIN_FRAC", "0.10"))
SEGREGATION_THRESH = float(os.environ.get("COMPARE_SEGREGATION_THRESH", "0.55"))
ANTICORR_THRESH = float(os.environ.get("COMPARE_ANTICORR_THRESH", "-0.20"))
CONSERVATION_THRESH = float(os.environ.get("COMPARE_CONSERVATION_THRESH", "0.60"))

# Fine-timescale child-child CCG (duplicate-peel signature)
SPLIT_FINE_CCG = (os.environ.get("COMPARE_SPLIT_FINE_CCG", "1").strip().lower() in {"1", "true", "yes", "y"})
FINE_CCG_WINDOW_S = float(os.environ.get("COMPARE_FINE_CCG_WINDOW_S", "0.005"))
FINE_CCG_BIN_S = float(os.environ.get("COMPARE_FINE_CCG_BIN_S", "0.0002"))
FINE_CCG_NEAR_ZERO_S = float(os.environ.get("COMPARE_FINE_CCG_NEAR_ZERO_S", "0.0005"))
DUPLICATE_NEAR_ZERO_FRAC_THRESH = float(os.environ.get("COMPARE_DUP_NEAR_ZERO_FRAC_THRESH", "0.05"))
DUPLICATE_ZERO_PEAK_RATIO_THRESH = float(os.environ.get("COMPARE_DUP_ZERO_PEAK_RATIO_THRESH", "1.25"))

# “Well-detected” heuristics (ported from compare_shallow_sweeps.py)
WELL_MPCT_THRESH = float(os.environ.get("COMPARE_WELL_MPCT_THRESH", "20.0"))
WELL_PRESENCE_THRESH = float(os.environ.get("COMPARE_WELL_PRESENCE_THRESH", "0.50"))


# =============================================================================
# Plot style (keep simple + publication-friendly defaults)
# =============================================================================
plt.rcParams.update({
    "font.size": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# =============================================================================
# Configuration
# =============================================================================

# Defaults aimed at your 20260316 imec1 example; override via env vars.
_DEFAULT_DREDGE_PIPE = "/mnt/NPX/Luke/20260316/dredge_pipeline_results_Luke03162026_V2V1_RH_g0_imec1"
_DEFAULT_PATCH_PIPE = "/mnt/NPX/Luke/20260316/patched_pipeline_results_Luke03162026_V2V1_RH_g0_imec1"

DREDGE_PIPE = Path(os.environ.get("DREDGE_PIPE", _DEFAULT_DREDGE_PIPE)).expanduser()
PATCH_PIPE = Path(os.environ.get("PATCH_PIPE", _DEFAULT_PATCH_PIPE)).expanduser()

STAGE = (os.environ.get("COMPARE_STAGE", "both") or "both").strip().lower()
OUT_DIR = Path(
    os.environ.get(
        "COMPARE_OUTDIR",
        str(DREDGE_PIPE.parent / f"compare_patched_vs_dredge_{DREDGE_PIPE.name}_vs_{PATCH_PIPE.name}"),
    )
).expanduser()
OUT_DIR.mkdir(parents=True, exist_ok=True)

DELTA_MS = float(os.environ.get("COMPARE_DELTA_MS", "0.4"))
MIN_AGREEMENT = float(os.environ.get("COMPARE_MIN_AGREEMENT", "0.5"))
GOOD_ONLY = (os.environ.get("COMPARE_GOOD_ONLY", "0").strip().lower() in {"1", "true", "yes", "y"})

LABEL_LEFT = "dredge"
LABEL_RIGHT = "patched"


def _stages_to_run(stage_raw: str) -> list[str]:
    s = (stage_raw or "").strip().lower()
    if s in {"both", "all", "pre_and_post", "prepost"}:
        return ["precuration", "postcuration"]
    return [s]


def _sorting_folder(pipe_dir: Path, stage: str) -> Path:
    stage = (stage or "").strip().lower()
    if stage in {"postcuration", "post", "cur", "curation"}:
        return pipe_dir / "cur" / "cur_sorter_output"
    if stage in {"precuration", "pre", "ks4", "sorter"}:
        return pipe_dir / "kilosort4" / "sorter_output"
    raise ValueError(f"Unknown stage='{stage}'")


def _amp_trunc_qc_dir(pipe_dir: Path, stage: str) -> Path:
    """Return best-effort QC folder.

    Some pipeline runs may store QC under qc/ (post) only, others may also have qc_pre/.
    We pick the most likely folder for the stage, but fall back if it doesn't exist.
    """

    stage = (stage or "").strip().lower()
    post = pipe_dir / "qc" / "amp_truncation"
    pre = pipe_dir / "qc_pre" / "amp_truncation"

    if stage in {"postcuration", "post", "cur", "curation"}:
        return post if post.exists() else pre
    if stage in {"precuration", "pre", "ks4", "sorter"}:
        return pre if pre.exists() else post
    raise ValueError(f"Unknown stage='{stage}'")


def _load_sorting(sort_folder: Path, *, stage: str):
    if not sort_folder.exists():
        raise FileNotFoundError(f"Sorting folder not found: {sort_folder}")

    stage = (stage or "").strip().lower()
    if stage in {"precuration", "pre", "ks4", "sorter"}:
        return cast(BaseSorting, si.read_kilosort(sort_folder))
    return cast(BaseSorting, si.load(sort_folder))


def _maybe_filter_good(sorting):
    if not GOOD_ONLY:
        return sorting
    keys = set(sorting.get_property_keys())
    if "KSLabel" not in keys:
        print("[warn] COMPARE_GOOD_ONLY=1 but sorting has no KSLabel property; skipping filter")
        return sorting
    labels = np.asarray(sorting.get_property("KSLabel"))
    unit_ids = np.asarray(sorting.unit_ids)
    keep = unit_ids[labels == "good"]
    return sorting.select_units(list(keep))


def _load_unit_qc_stats(pipe_dir: Path, *, stage: str, fs: float = 30_000.0) -> pd.DataFrame:
    """Best-effort per-unit missing% and presence fraction.

    Uses:
      - <sorting_folder>/spike_times.npy + spike_clusters.npy
      - <qc_dir>/truncation_qc.npz + present_qc.npz
    """

    sorter_out = _sorting_folder(pipe_dir, stage)
    qc_dir = _amp_trunc_qc_dir(pipe_dir, stage)

    spike_times_path = sorter_out / "spike_times.npy"
    spike_clusters_path = sorter_out / "spike_clusters.npy"

    if not spike_times_path.exists() or not spike_clusters_path.exists():
        return pd.DataFrame(columns=["unit_id", "mean_mpct", "presence_frac", "n_spikes"])  # empty

    spike_times_s = np.load(spike_times_path).astype(float) / fs
    spike_clusters = np.load(spike_clusters_path).astype(int)
    if spike_times_s.size == 0 or spike_clusters.size == 0:
        return pd.DataFrame(columns=["unit_id", "mean_mpct", "presence_frac", "n_spikes"])

    rec_dur = float(spike_times_s.max())

    # Group spikes by unit efficiently while preserving within-unit time order.
    # Kilosort spike_times are globally time-sorted, and a stable sort by cluster
    # preserves that ordering within each cluster.
    order = np.argsort(spike_clusters, kind="stable")
    clusters_sorted = spike_clusters[order]
    times_sorted = spike_times_s[order]
    unit_ids, start_idx, counts = np.unique(clusters_sorted, return_index=True, return_counts=True)

    trunc = None
    pres = None
    if (qc_dir / "truncation_qc.npz").exists():
        trunc = np.load(qc_dir / "truncation_qc.npz")
    if (qc_dir / "present_qc.npz").exists():
        pres = np.load(qc_dir / "present_qc.npz")

    trunc_cid = trunc["cid"].astype(int) if trunc is not None else np.array([], int)
    trunc_mpcts = trunc["mpcts"] if trunc is not None else np.array([])

    pres_cid = pres["cid"].astype(int) if pres is not None else np.array([], int)
    pres_vblk = pres["valid_blocks"] if pres is not None else np.zeros((0, 2), int)

    trunc_mean_by_uid: dict[int, float] = {}
    if trunc is not None and trunc_cid.size:
        for uid in np.unique(trunc_cid):
            m = trunc_cid == uid
            trunc_mean_by_uid[int(uid)] = float(np.nanmean(trunc_mpcts[m])) if m.any() else np.nan

    pres_blocks_by_uid: dict[int, list[tuple[int, int]]] = {}
    if pres is not None and pres_cid.size:
        for uid, (i0, i1) in zip(pres_cid.tolist(), pres_vblk.tolist()):
            pres_blocks_by_uid.setdefault(int(uid), []).append((int(i0), int(i1)))

    rows = []
    for uid, s0, n_spikes in zip(unit_ids.tolist(), start_idx.tolist(), counts.tolist()):
        uid = int(uid)
        s0 = int(s0)
        n_spikes = int(n_spikes)

        u_times = times_sorted[s0 : s0 + n_spikes]

        mean_mpct = trunc_mean_by_uid.get(uid, np.nan)

        blocks = pres_blocks_by_uid.get(uid)
        if blocks and rec_dur > 0 and n_spikes > 0:
            # valid_blocks are in spike-index space within the unit
            pres_s = 0.0
            last_idx = n_spikes - 1
            for i0, i1 in blocks:
                i0c = min(max(int(i0), 0), last_idx)
                i1c = min(max(int(i1), 0), last_idx)
                pres_s += float(u_times[i1c] - u_times[i0c])
            presence_frac = min(pres_s / rec_dur, 1.0)
        else:
            presence_frac = np.nan

        rows.append(dict(unit_id=uid, mean_mpct=mean_mpct, presence_frac=presence_frac, n_spikes=n_spikes))

    return pd.DataFrame(rows)


def _spike_count_summary(sort_folder: Path) -> dict:
    """Fast spike-count summary from Kilosort-style arrays (if present)."""

    spike_clusters_path = sort_folder / "spike_clusters.npy"
    if not spike_clusters_path.exists():
        return {"total_spikes": np.nan, "median_spikes_per_unit": np.nan}

    spike_clusters = np.load(spike_clusters_path)
    if spike_clusters.size == 0:
        return {"total_spikes": 0, "median_spikes_per_unit": np.nan}

    _, counts = np.unique(spike_clusters.astype(int), return_counts=True)
    return {"total_spikes": int(spike_clusters.size), "median_spikes_per_unit": float(np.median(counts))}


def _spike_array_summary(sort_folder: Path, *, fs: float = 30_000.0) -> dict:
    """Best-effort scalar summaries from Kilosort-style arrays.

    Intended to be fast and robust for both precuration (KS4 sorter_output)
    and postcuration (SI folder) when they contain spike_times.npy/spike_clusters.npy.
    """

    st_path = sort_folder / "spike_times.npy"
    sc_path = sort_folder / "spike_clusters.npy"
    if not st_path.exists() or not sc_path.exists():
        return {
            "duration_s": np.nan,
            "total_spikes": np.nan,
            "spikes_per_unit_q25": np.nan,
            "spikes_per_unit_median": np.nan,
            "spikes_per_unit_q75": np.nan,
            "firing_rate_hz_q25": np.nan,
            "firing_rate_hz_median": np.nan,
            "firing_rate_hz_q75": np.nan,
            "frac_units_fr_lt_0p1hz": np.nan,
            "frac_units_fr_gt_10hz": np.nan,
            "spike_dominance_top10": np.nan,
        }

    st = np.load(st_path, mmap_mode="r")
    sc = np.load(sc_path, mmap_mode="r")
    if st.size == 0 or sc.size == 0:
        return {
            "duration_s": 0.0,
            "total_spikes": int(sc.size),
            "spikes_per_unit_q25": np.nan,
            "spikes_per_unit_median": np.nan,
            "spikes_per_unit_q75": np.nan,
            "firing_rate_hz_q25": np.nan,
            "firing_rate_hz_median": np.nan,
            "firing_rate_hz_q75": np.nan,
            "frac_units_fr_lt_0p1hz": np.nan,
            "frac_units_fr_gt_10hz": np.nan,
            "spike_dominance_top10": np.nan,
        }

    try:
        duration_s = float(np.max(st)) / float(fs)
    except Exception:
        duration_s = np.nan

    _, counts = np.unique(np.asarray(sc, dtype=np.int64), return_counts=True)
    counts = counts.astype(np.int64)
    total = int(sc.size)

    if counts.size:
        q25, q50, q75 = np.percentile(counts.astype(float), [25, 50, 75])
    else:
        q25 = q50 = q75 = np.nan

    if np.isfinite(duration_s) and duration_s > 0 and counts.size:
        fr = counts.astype(float) / float(duration_s)
        fr_q25, fr_q50, fr_q75 = np.percentile(fr, [25, 50, 75])
        frac_lt_0p1 = float(np.mean(fr < 0.1))
        frac_gt_10 = float(np.mean(fr > 10.0))
    else:
        fr_q25 = fr_q50 = fr_q75 = np.nan
        frac_lt_0p1 = np.nan
        frac_gt_10 = np.nan

    # How concentrated are spikes in the top-K units?
    if total > 0 and counts.size:
        k = int(min(10, counts.size))
        topk = np.partition(counts, -k)[-k:]
        dominance = float(int(topk.sum()) / total)
    else:
        dominance = np.nan

    return {
        "duration_s": float(duration_s) if np.isfinite(duration_s) else np.nan,
        "total_spikes": total,
        "spikes_per_unit_q25": float(q25) if np.isfinite(q25) else np.nan,
        "spikes_per_unit_median": float(q50) if np.isfinite(q50) else np.nan,
        "spikes_per_unit_q75": float(q75) if np.isfinite(q75) else np.nan,
        "firing_rate_hz_q25": float(fr_q25) if np.isfinite(fr_q25) else np.nan,
        "firing_rate_hz_median": float(fr_q50) if np.isfinite(fr_q50) else np.nan,
        "firing_rate_hz_q75": float(fr_q75) if np.isfinite(fr_q75) else np.nan,
        "frac_units_fr_lt_0p1hz": float(frac_lt_0p1) if np.isfinite(frac_lt_0p1) else np.nan,
        "frac_units_fr_gt_10hz": float(frac_gt_10) if np.isfinite(frac_gt_10) else np.nan,
        "spike_dominance_top10": float(dominance) if np.isfinite(dominance) else np.nan,
    }


def _condition_stats(
    *,
    label: str,
    stage: str,
    sorting,
    sort_folder: Path,
    qc: pd.DataFrame,
) -> dict:
    unit_ids = np.asarray(sorting.unit_ids)
    n_units = int(unit_ids.size)

    spike_summary = _spike_count_summary(sort_folder)
    spike_arr = _spike_array_summary(sort_folder)

    ks_good = np.nan
    ks_frac_good = np.nan
    if "KSLabel" in set(sorting.get_property_keys()):
        labels = np.asarray(sorting.get_property("KSLabel"))
        ks_good = int(np.sum(labels == "good"))
        ks_frac_good = float(ks_good / n_units) if n_units else np.nan

    n_well = np.nan
    frac_well = np.nan
    if len(qc):
        good_qc = qc["mean_mpct"].to_numpy(dtype=float)
        pres_qc = qc["presence_frac"].to_numpy(dtype=float)
        ok = np.isfinite(good_qc) & np.isfinite(pres_qc)
        if np.any(ok):
            well = (good_qc[ok] < WELL_MPCT_THRESH) & (pres_qc[ok] > WELL_PRESENCE_THRESH)
            n_well = int(np.sum(well))
            frac_well = float(np.mean(well))

    return {
        "condition": label,
        "stage": stage,
        "n_units": n_units,
        "n_units_good": ks_good,
        "frac_units_good": ks_frac_good,
        "total_spikes": spike_summary["total_spikes"],
        "median_spikes_per_unit": spike_summary["median_spikes_per_unit"],
        "duration_s": spike_arr["duration_s"],
        "spikes_per_unit_q25": spike_arr["spikes_per_unit_q25"],
        "spikes_per_unit_median": spike_arr["spikes_per_unit_median"],
        "spikes_per_unit_q75": spike_arr["spikes_per_unit_q75"],
        "firing_rate_hz_q25": spike_arr["firing_rate_hz_q25"],
        "firing_rate_hz_median": spike_arr["firing_rate_hz_median"],
        "firing_rate_hz_q75": spike_arr["firing_rate_hz_q75"],
        "frac_units_fr_lt_0p1hz": spike_arr["frac_units_fr_lt_0p1hz"],
        "frac_units_fr_gt_10hz": spike_arr["frac_units_fr_gt_10hz"],
        "spike_dominance_top10": spike_arr["spike_dominance_top10"],
        "median_missing_pct": float(np.nanmedian(qc["mean_mpct"])) if len(qc) else np.nan,
        "median_presence_frac": float(np.nanmedian(qc["presence_frac"])) if len(qc) else np.nan,
        "n_well_detected": n_well,
        "frac_well_detected": frac_well,
        "well_mpct_thresh": WELL_MPCT_THRESH,
        "well_presence_thresh": WELL_PRESENCE_THRESH,
    }


def _coincident_mask(times_a_s: np.ndarray, times_b_s: np.ndarray, *, tol_s: float) -> np.ndarray:
    """Boolean mask for spikes in times_a_s that have a coincident spike in times_b_s."""

    ta = np.asarray(times_a_s, float)
    tb = np.asarray(times_b_s, float)
    if ta.size == 0 or tb.size == 0:
        return np.zeros(ta.size, dtype=bool)

    o = np.argsort(ta)
    ta_sorted = ta[o]
    tb_sorted = np.sort(tb)

    idx = np.searchsorted(tb_sorted, ta_sorted)
    il = np.clip(idx - 1, 0, tb_sorted.size - 1)
    ir = np.clip(idx, 0, tb_sorted.size - 1)
    min_dist = np.minimum(np.abs(ta_sorted - tb_sorted[il]), np.abs(ta_sorted - tb_sorted[ir]))
    m_sorted = (min_dist <= float(tol_s))

    m = np.zeros(m_sorted.size, dtype=bool)
    m[o] = m_sorted
    return m


def _fine_ccg_for_pair(
    times_a_s: np.ndarray,
    times_b_s: np.ndarray,
    *,
    window_s: float = FINE_CCG_WINDOW_S,
    bin_s: float = FINE_CCG_BIN_S,
    near_zero_s: float = FINE_CCG_NEAR_ZERO_S,
) -> dict:
    """Fine-timescale cross-correlogram summary for a pair."""

    ta = np.sort(np.asarray(times_a_s, float))
    tb = np.sort(np.asarray(times_b_s, float))
    edges = np.arange(-window_s, window_s + bin_s, bin_s)
    centers = (edges[:-1] + edges[1:]) / 2

    if ta.size == 0 or tb.size == 0:
        return {
            "bin_centers_s": centers,
            "counts": np.zeros(centers.size, dtype=int),
            "total_pairs": 0,
            "near_zero_pairs": 0,
            "near_zero_frac": np.nan,
            "zero_peak_ratio": np.nan,
        }

    dts: list[np.ndarray] = []
    left = 0
    right = 0
    for t in ta:
        while left < tb.size and tb[left] < (t - window_s):
            left += 1
        if right < left:
            right = left
        while right < tb.size and tb[right] <= (t + window_s):
            right += 1
        if right > left:
            dts.append(tb[left:right] - t)

    if dts:
        all_dt = np.concatenate(dts)
        counts, _ = np.histogram(all_dt, bins=edges)
    else:
        counts = np.zeros(centers.size, dtype=int)

    total_pairs = int(counts.sum())
    near_zero_mask = np.abs(centers) <= near_zero_s
    near_zero_pairs = int(counts[near_zero_mask].sum()) if np.any(near_zero_mask) else 0
    near_zero_frac = (near_zero_pairs / total_pairs) if total_pairs else np.nan

    outer_mask = np.abs(centers) >= (window_s * 0.6)
    baseline = float(np.median(counts[outer_mask])) if np.any(outer_mask) else 0.0
    zero_peak = float(counts[near_zero_mask].max()) if np.any(near_zero_mask) else 0.0
    zero_peak_ratio = (zero_peak / max(baseline, 1.0)) if total_pairs else np.nan

    return {
        "bin_centers_s": centers,
        "counts": counts,
        "total_pairs": total_pairs,
        "near_zero_pairs": near_zero_pairs,
        "near_zero_frac": near_zero_frac,
        "zero_peak_ratio": zero_peak_ratio,
    }


def _safe_corrcoef(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if x.size < 2 or y.size < 2:
        return np.nan
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def _load_times_by_unit(sort_folder: Path, *, fs: float = 30_000.0) -> tuple[dict[int, np.ndarray], float]:
    """Load sorted spike times (seconds) per unit from Kilosort-style arrays."""

    st_path = sort_folder / "spike_times.npy"
    sc_path = sort_folder / "spike_clusters.npy"
    if not st_path.exists() or not sc_path.exists():
        return {}, 0.0

    spike_times_s = np.load(st_path).astype(float) / fs
    spike_clusters = np.load(sc_path).astype(int)
    if spike_times_s.size == 0:
        return {}, 0.0

    # group efficiently by unit with stable sort
    order = np.argsort(spike_clusters, kind="stable")
    clusters_sorted = spike_clusters[order]
    times_sorted = spike_times_s[order]
    uids, start_idx, counts = np.unique(clusters_sorted, return_index=True, return_counts=True)

    by_unit: dict[int, np.ndarray] = {}
    for uid, s0, n in zip(uids.tolist(), start_idx.tolist(), counts.tolist()):
        uid = int(uid)
        s0 = int(s0)
        n = int(n)
        by_unit[uid] = times_sorted[s0 : s0 + n]

    return by_unit, float(spike_times_s.max())


def _split_metrics(
    *,
    ref_counts: np.ndarray,
    c1_counts: np.ndarray,
    c2_counts: np.ndarray,
) -> dict:
    """Compute segregation/anticorr/conservation-style metrics on binned counts."""

    ref_counts = np.asarray(ref_counts, float)
    c1_counts = np.asarray(c1_counts, float)
    c2_counts = np.asarray(c2_counts, float)

    denom = c1_counts + c2_counts
    with np.errstate(divide="ignore", invalid="ignore"):
        segregation = np.nanmean(np.abs(c1_counts - c2_counts) / np.where(denom > 0, denom, np.nan))

    anticorr = _safe_corrcoef(c1_counts, c2_counts)

    with np.errstate(divide="ignore", invalid="ignore"):
        conservation = np.nanmean(denom / np.where(ref_counts > 0, ref_counts, np.nan))

    return {
        "segregation": float(segregation) if np.isfinite(segregation) else np.nan,
        "anticorr": float(anticorr) if np.isfinite(anticorr) else np.nan,
        "conservation": float(conservation) if np.isfinite(conservation) else np.nan,
    }


def _best_matches_from_agreement(agreement: pd.DataFrame, *, axis: int) -> pd.DataFrame:
    """Return best match per unit.

    axis=1: for each row unit -> best column unit
    axis=0: for each column unit -> best row unit
    """

    if agreement.empty:
        return pd.DataFrame(columns=["unit", "best_match", "agreement"])

    if axis == 1:
        best_idx = agreement.values.argmax(axis=1)
        best_val = agreement.values.max(axis=1)
        units = agreement.index.to_numpy()
        best_units = agreement.columns.to_numpy()[best_idx]
    elif axis == 0:
        best_idx = agreement.values.argmax(axis=0)
        best_val = agreement.values.max(axis=0)
        units = agreement.columns.to_numpy()
        best_units = agreement.index.to_numpy()[best_idx]
    else:
        raise ValueError("axis must be 0 or 1")

    return pd.DataFrame({"unit": units.astype(int), "best_match": best_units.astype(int), "agreement": best_val.astype(float)})


def _write_comparison_outputs(
    *,
    out_dir: Path,
    sorting1: BaseSorting,
    sorting2: BaseSorting,
    label1: str,
    label2: str,
    qc1: pd.DataFrame | None = None,
    qc2: pd.DataFrame | None = None,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    cmp = scmp.compare_two_sorters(
        sorting1=sorting1,
        sorting2=sorting2,
        sorting1_name=label1,
        sorting2_name=label2,
        delta_time=DELTA_MS / 1000.0,
    )

    agreement_scores = cmp.agreement_scores
    agreement_scores.to_csv(out_dir / "agreement_scores.csv")

    # Optional: match-event-count matrix (useful for diagnosing best-match agreement mechanics)
    try:
        mec = cmp.match_event_count
        mec.to_csv(out_dir / "match_event_count.csv")
    except Exception:
        pass

    best_1_to_2 = _best_matches_from_agreement(agreement_scores, axis=1)
    best_2_to_1 = _best_matches_from_agreement(agreement_scores, axis=0)
    best_1_to_2.to_csv(out_dir / f"best_matches_{label1}_to_{label2}.csv", index=False)
    best_2_to_1.to_csv(out_dir / f"best_matches_{label2}_to_{label1}.csv", index=False)

    frac_1_well_matched = float(np.mean(best_1_to_2["agreement"] >= MIN_AGREEMENT)) if len(best_1_to_2) else np.nan
    frac_2_well_matched = float(np.mean(best_2_to_1["agreement"] >= MIN_AGREEMENT)) if len(best_2_to_1) else np.nan

    summary = {
        "delta_ms": DELTA_MS,
        "min_agreement": MIN_AGREEMENT,
        "good_only": bool(GOOD_ONLY),
        "label1": label1,
        "label2": label2,
        "n_units_1": int(len(sorting1.unit_ids)),
        "n_units_2": int(len(sorting2.unit_ids)),
        "frac_well_matched_1_to_2": frac_1_well_matched,
        "frac_well_matched_2_to_1": frac_2_well_matched,
        "median_best_agreement_1_to_2": float(np.nanmedian(best_1_to_2["agreement"])) if len(best_1_to_2) else np.nan,
        "median_best_agreement_2_to_1": float(np.nanmedian(best_2_to_1["agreement"])) if len(best_2_to_1) else np.nan,
    }
    if qc1 is not None and len(qc1):
        summary["median_missing_pct_1"] = float(np.nanmedian(qc1["mean_mpct"]))
        summary["median_presence_frac_1"] = float(np.nanmedian(qc1["presence_frac"]))
    if qc2 is not None and len(qc2):
        summary["median_missing_pct_2"] = float(np.nanmedian(qc2["mean_mpct"]))
        summary["median_presence_frac_2"] = float(np.nanmedian(qc2["presence_frac"]))

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


# =============================================================================
# Main
# =============================================================================

print(f"Comparing pipelines ({STAGE}):")
print(f"  dredge : {DREDGE_PIPE}")
print(f"  patched: {PATCH_PIPE}")
print(f"  out    : {OUT_DIR}")

condition_stats_rows: list[dict] = []
comparisons_summary: dict[str, dict] = {}

conditions: dict[str, dict] = {}

stages = _stages_to_run(STAGE or "both")
if not stages or any(s not in {"precuration", "pre", "ks4", "sorter", "postcuration", "post", "cur", "curation"} for s in stages):
    raise ValueError(f"Unknown COMPARE_STAGE='{STAGE}'")

for stage in stages:
    stage_norm = "pre" if stage in {"precuration", "pre", "ks4", "sorter"} else "post"
    stage_out = OUT_DIR if len(stages) == 1 else (OUT_DIR / stage_norm)
    stage_out.mkdir(parents=True, exist_ok=True)

    left_folder = _sorting_folder(DREDGE_PIPE, stage)
    right_folder = _sorting_folder(PATCH_PIPE, stage)
    print(f"Loading sortings ({stage_norm}):\n  {LABEL_LEFT}: {left_folder}\n  {LABEL_RIGHT}: {right_folder}")

    sorting_left = cast(BaseSorting, _maybe_filter_good(_load_sorting(left_folder, stage=stage)))
    sorting_right = cast(BaseSorting, _maybe_filter_good(_load_sorting(right_folder, stage=stage)))

    print(f"Units ({stage_norm}): {LABEL_LEFT}={len(sorting_left.unit_ids)}  {LABEL_RIGHT}={len(sorting_right.unit_ids)}")

    qc_left = _load_unit_qc_stats(DREDGE_PIPE, stage=stage)
    qc_right = _load_unit_qc_stats(PATCH_PIPE, stage=stage)

    # keep the 4 conditions around for a 4-way overview + additional comparisons
    conditions[f"{LABEL_LEFT}_{stage_norm}"] = {
        "label": f"{LABEL_LEFT}_{stage_norm}",
        "stage": stage_norm,
        "sorting": sorting_left,
        "folder": left_folder,
        "qc": qc_left,
    }
    conditions[f"{LABEL_RIGHT}_{stage_norm}"] = {
        "label": f"{LABEL_RIGHT}_{stage_norm}",
        "stage": stage_norm,
        "sorting": sorting_right,
        "folder": right_folder,
        "qc": qc_right,
    }

    # core stage comparison: dredge vs patched
    stage_summary = _write_comparison_outputs(
        out_dir=stage_out,
        sorting1=sorting_left,
        sorting2=sorting_right,
        label1=LABEL_LEFT,
        label2=LABEL_RIGHT,
        qc1=qc_left,
        qc2=qc_right,
    )

    # condition stats (4 points total across both stages)
    condition_stats_rows.append(
        _condition_stats(
            label=f"{LABEL_LEFT}_{stage_norm}",
            stage=stage_norm,
            sorting=sorting_left,
            sort_folder=left_folder,
            qc=qc_left,
        )
    )
    condition_stats_rows.append(
        _condition_stats(
            label=f"{LABEL_RIGHT}_{stage_norm}",
            stage=stage_norm,
            sorting=sorting_right,
            sort_folder=right_folder,
            qc=qc_right,
        )
    )

    comparisons_summary[stage_norm] = {
        "stage": stage_norm,
        "dredge_pipe": str(DREDGE_PIPE),
        "patched_pipe": str(PATCH_PIPE),
        "n_units_dredge": int(len(sorting_left.unit_ids)),
        "n_units_patched": int(len(sorting_right.unit_ids)),
        "frac_well_matched_dredge_to_patched": stage_summary.get("frac_well_matched_1_to_2"),
        "frac_well_matched_patched_to_dredge": stage_summary.get("frac_well_matched_2_to_1"),
        "median_best_agreement_dredge_to_patched": stage_summary.get("median_best_agreement_1_to_2"),
        "median_best_agreement_patched_to_dredge": stage_summary.get("median_best_agreement_2_to_1"),
        "median_missing_pct_dredge": float(np.nanmedian(qc_left["mean_mpct"])) if len(qc_left) else np.nan,
        "median_missing_pct_patched": float(np.nanmedian(qc_right["mean_mpct"])) if len(qc_right) else np.nan,
        "median_presence_frac_dredge": float(np.nanmedian(qc_left["presence_frac"])) if len(qc_left) else np.nan,
        "median_presence_frac_patched": float(np.nanmedian(qc_right["presence_frac"])) if len(qc_right) else np.nan,
        "delta_ms": DELTA_MS,
        "min_agreement": MIN_AGREEMENT,
        "good_only": bool(GOOD_ONLY),
    }

    print("Wrote:")
    print(f"  {stage_out / 'agreement_scores.csv'}")
    print(f"  {stage_out / f'best_matches_{LABEL_LEFT}_to_{LABEL_RIGHT}.csv'}")
    print(f"  {stage_out / f'best_matches_{LABEL_RIGHT}_to_{LABEL_LEFT}.csv'}")
    print(f"  {stage_out / 'summary.json'}")

    # =============================================================================
    # Figure: overview (per stage)
    # =============================================================================

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2))
    axes_arr = np.asarray(axes).reshape(2, 2)
    (ax0, ax1), (ax2, ax3) = axes_arr

    # bar: unit counts
    ax0.bar(
        [0, 1],
        [len(sorting_left.unit_ids), len(sorting_right.unit_ids)],
        color=["#2B6CB0", "#C05621"],
        width=0.7,
    )
    ax0.set_xticks([0, 1])
    ax0.set_xticklabels([LABEL_LEFT, LABEL_RIGHT])
    ax0.set_ylabel("Units")
    ax0.set_title("Total units")

    # histogram: best-match agreement
    best_1_to_2 = pd.read_csv(stage_out / f"best_matches_{LABEL_LEFT}_to_{LABEL_RIGHT}.csv")
    best_2_to_1 = pd.read_csv(stage_out / f"best_matches_{LABEL_RIGHT}_to_{LABEL_LEFT}.csv")
    bins = np.linspace(0, 1, 41)
    ax1.hist(best_1_to_2["agreement"].to_numpy(), bins=bins, alpha=0.7, label=f"{LABEL_LEFT}→{LABEL_RIGHT}", color="#2B6CB0")
    ax1.hist(best_2_to_1["agreement"].to_numpy(), bins=bins, alpha=0.5, label=f"{LABEL_RIGHT}→{LABEL_LEFT}", color="#C05621")
    ax1.axvline(MIN_AGREEMENT, color="#444", lw=1.0, ls="--")
    ax1.set_xlabel("Best-match agreement")
    ax1.set_ylabel("Unit count")
    ax1.set_title(f"Best-match agreement (Δ={DELTA_MS:.2f} ms)")
    ax1.legend(fontsize=7)

    # CDF: missing % (if available)
    for label, qc, color in [
        (LABEL_LEFT, qc_left, "#2B6CB0"),
        (LABEL_RIGHT, qc_right, "#C05621"),
    ]:
        v = np.asarray(qc["mean_mpct"].dropna().values) if len(qc) else np.array([])
        if v.size:
            v = np.sort(v)
            cdf = np.arange(1, len(v) + 1) / len(v)
            ax2.plot(v, cdf, color=color, lw=1.5, label=label)
    ax2.set_xlim(0, 60)
    ax2.set_xlabel("Mean missing % per unit")
    ax2.set_ylabel("CDF")
    ax2.set_title("Amplitude truncation QC")
    ax2.legend(fontsize=7)

    # CDF: presence fraction (if available)
    for label, qc, color in [
        (LABEL_LEFT, qc_left, "#2B6CB0"),
        (LABEL_RIGHT, qc_right, "#C05621"),
    ]:
        v = np.asarray(qc["presence_frac"].dropna().values) if len(qc) else np.array([])
        if v.size:
            v = np.sort(v)
            cdf = np.arange(1, len(v) + 1) / len(v)
            ax3.plot(v, cdf, color=color, lw=1.5, label=label)
    ax3.set_xlim(0, 1)
    ax3.set_xlabel("Presence fraction per unit")
    ax3.set_ylabel("CDF")
    ax3.set_title("Presence QC")
    ax3.legend(fontsize=7)

    fig.suptitle(f"Direct comparison ({stage_norm}): patched vs dredge", y=1.02)
    fig.tight_layout()

    fig.savefig(stage_out / "fig_overview.pdf")
    fig.savefig(stage_out / "fig_overview.png")
    plt.close(fig)

    print(f"Saved overview figure → {stage_out / 'fig_overview.pdf'}")


# If we ran both stages, also compare pre vs post within each pipeline.
if {"dredge_pre", "dredge_post"}.issubset(set(conditions.keys())):
    s_pre = conditions["dredge_pre"]["sorting"]
    s_post = conditions["dredge_post"]["sorting"]
    qc_pre = conditions["dredge_pre"]["qc"]
    qc_post = conditions["dredge_post"]["qc"]
    comparisons_summary["dredge_pre_vs_post"] = _write_comparison_outputs(
        out_dir=OUT_DIR / "dredge_pre_vs_post",
        sorting1=s_pre,
        sorting2=s_post,
        label1="dredge_pre",
        label2="dredge_post",
        qc1=qc_pre,
        qc2=qc_post,
    )

if {"patched_pre", "patched_post"}.issubset(set(conditions.keys())):
    s_pre = conditions["patched_pre"]["sorting"]
    s_post = conditions["patched_post"]["sorting"]
    qc_pre = conditions["patched_pre"]["qc"]
    qc_post = conditions["patched_post"]["qc"]
    comparisons_summary["patched_pre_vs_post"] = _write_comparison_outputs(
        out_dir=OUT_DIR / "patched_pre_vs_post",
        sorting1=s_pre,
        sorting2=s_post,
        label1="patched_pre",
        label2="patched_post",
        qc1=qc_pre,
        qc2=qc_post,
    )


df_cond = pd.DataFrame(condition_stats_rows)
df_cond.to_csv(OUT_DIR / "condition_stats.csv", index=False)
(OUT_DIR / "condition_stats.json").write_text(json.dumps(condition_stats_rows, indent=2))


# Optional: treat a default condition like compare_shallow_sweeps and run extra comparisons
# against it (e.g. dredge_pre vs each other condition).
if SPLIT_DIAGNOSTICS and REF_CONDITION in conditions:
    ref = conditions[REF_CONDITION]
    for k, other in conditions.items():
        if k == REF_CONDITION:
            continue
        key = f"{REF_CONDITION}_vs_{k}"
        comparisons_summary[key] = _write_comparison_outputs(
            out_dir=OUT_DIR / key,
            sorting1=ref["sorting"],
            sorting2=other["sorting"],
            label1=REF_CONDITION,
            label2=k,
            qc1=ref["qc"],
            qc2=other["qc"],
        )


(OUT_DIR / "comparisons_summary.json").write_text(json.dumps(comparisons_summary, indent=2))


# Root figure: 4-way condition overview
if {"dredge_pre", "patched_pre", "dredge_post", "patched_post"}.issubset(set(conditions.keys())):
    order = ["dredge_pre", "patched_pre", "dredge_post", "patched_post"]
    colors = {
        "dredge_pre": "#2B6CB0",
        "dredge_post": "#2B6CB0",
        "patched_pre": "#C05621",
        "patched_post": "#C05621",
    }
    linestyles = {"pre": "-", "post": "--"}

    fig, axes = plt.subplots(2, 2, figsize=(7.6, 5.2))
    axes_arr = np.asarray(axes).reshape(2, 2)
    (ax0, ax1), (ax2, ax3) = axes_arr

    # bar: unit counts
    xs = np.arange(len(order))
    counts = [int(len(conditions[k]["sorting"].unit_ids)) for k in order]
    ax0.bar(xs, counts, color=[colors[k] for k in order], width=0.75)
    ax0.set_xticks(xs)
    ax0.set_xticklabels(order, rotation=25, ha="right")
    ax0.set_ylabel("Units")
    ax0.set_title("Total units (4-way)")

    # bar: good fraction (if available)
    fracs = []
    for k in order:
        s = conditions[k]["sorting"]
        if "KSLabel" in set(s.get_property_keys()):
            labels = np.asarray(s.get_property("KSLabel"))
            fracs.append(float(np.mean(labels == "good")) if labels.size else np.nan)
        else:
            fracs.append(np.nan)
    ax1.bar(xs, fracs, color=[colors[k] for k in order], width=0.75)
    ax1.set_xticks(xs)
    ax1.set_xticklabels(order, rotation=25, ha="right")
    ax1.set_ylim(0, 1)
    ax1.set_ylabel("Fraction")
    ax1.set_title("Fraction KSLabel=='good'")

    # CDF: missing %
    for k in order:
        qc = conditions[k]["qc"]
        v = np.asarray(qc["mean_mpct"].dropna().values) if len(qc) else np.array([])
        if v.size:
            v = np.sort(v)
            cdf = np.arange(1, len(v) + 1) / len(v)
            stage = conditions[k]["stage"]
            ax2.plot(v, cdf, color=colors[k], lw=1.5, ls=linestyles.get(stage, "-"), label=k)
    ax2.set_xlim(0, 60)
    ax2.set_xlabel("Mean missing % per unit")
    ax2.set_ylabel("CDF")
    ax2.set_title("Amplitude truncation QC")
    ax2.legend(fontsize=6)

    # CDF: presence fraction
    for k in order:
        qc = conditions[k]["qc"]
        v = np.asarray(qc["presence_frac"].dropna().values) if len(qc) else np.array([])
        if v.size:
            v = np.sort(v)
            cdf = np.arange(1, len(v) + 1) / len(v)
            stage = conditions[k]["stage"]
            ax3.plot(v, cdf, color=colors[k], lw=1.5, ls=linestyles.get(stage, "-"), label=k)
    ax3.set_xlim(0, 1)
    ax3.set_xlabel("Presence fraction per unit")
    ax3.set_ylabel("CDF")
    ax3.set_title("Presence QC")
    ax3.legend(fontsize=6)

    fig.suptitle("4-way overview: dredge/patch × pre/post", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_overview.pdf")
    fig.savefig(OUT_DIR / "fig_overview.png")
    plt.close(fig)


# Optional: split diagnostics PDF/CSV (default vs each other condition)
if SPLIT_DIAGNOSTICS and REF_CONDITION in conditions:
    from matplotlib.backends.backend_pdf import PdfPages

    ref_folder = Path(conditions[REF_CONDITION]["folder"])
    ref_by_unit, rec_dur_s = _load_times_by_unit(ref_folder)
    if rec_dur_s > 0:
        bins_s = np.arange(0, rec_dur_s + SPLIT_BIN_S, SPLIT_BIN_S)
    else:
        bins_s = np.array([0.0, 1.0])

    split_rows: list[dict] = []
    pdf_path = OUT_DIR / "fig_split_diagnostics.pdf"
    csv_path = OUT_DIR / "split_diagnostics.csv"

    pages_written = 0
    with PdfPages(pdf_path) as pdf:
        for other_key in [k for k in conditions.keys() if k != REF_CONDITION]:
            # Need an agreement matrix for ref vs other. Prefer one we just wrote.
            comp_dir = OUT_DIR / f"{REF_CONDITION}_vs_{other_key}"
            agree_path = comp_dir / "agreement_scores.csv"
            if not agree_path.exists():
                continue

            agreement = pd.read_csv(agree_path, index_col=0)
            other_folder = Path(conditions[other_key]["folder"])
            oth_by_unit, _ = _load_times_by_unit(other_folder)

            for ref_uid_key, row in agreement.iterrows():
                ref_uid = int(str(ref_uid_key))
                ref_times = ref_by_unit.get(ref_uid)
                if ref_times is None or ref_times.size < SPLIT_MIN_SPIKES:
                    continue

                # top-2 matches in the other sorting by agreement score
                vals = row.to_numpy(dtype=float)
                if vals.size < 2:
                    continue
                top2_idx = np.argsort(vals)[-2:][::-1]
                child_uids = agreement.columns.to_numpy()[top2_idx].astype(int)
                a1 = float(vals[top2_idx[0]])
                a2 = float(vals[top2_idx[1]])
                child1, child2 = int(child_uids[0]), int(child_uids[1])

                c1_times = oth_by_unit.get(child1)
                c2_times = oth_by_unit.get(child2)
                if c1_times is None or c2_times is None:
                    continue

                # Compute tradeoff/conservation on coincident ref spikes (more faithful than raw child counts)
                ref_counts, _ = np.histogram(ref_times, bins=bins_s)
                tol_s = float(DELTA_MS) / 1000.0
                m1 = _coincident_mask(ref_times, c1_times, tol_s=tol_s)
                m2 = _coincident_mask(ref_times, c2_times, tol_s=tol_s)
                c1_counts, _ = np.histogram(ref_times[m1], bins=bins_s)
                c2_counts, _ = np.histogram(ref_times[m2], bins=bins_s)
                union_counts, _ = np.histogram(ref_times[m1 | m2], bins=bins_s)

                m = _split_metrics(ref_counts=ref_counts, c1_counts=c1_counts, c2_counts=c2_counts)
                # Conservation should be about union coverage of ref spikes
                with np.errstate(divide="ignore", invalid="ignore"):
                    cons = np.nanmean(union_counts / np.where(ref_counts > 0, ref_counts, np.nan))
                m["conservation"] = float(cons) if np.isfinite(cons) else np.nan
                split_score = a1 + a2

                flagged = (
                    (split_score >= SPLIT_SCORE_THRESH)
                    and (a2 >= SPLIT_CHILD2_MIN_FRAC)
                    and (np.isfinite(m["segregation"]) and m["segregation"] >= SEGREGATION_THRESH)
                    and (np.isfinite(m["anticorr"]) and m["anticorr"] <= ANTICORR_THRESH)
                    and (np.isfinite(m["conservation"]) and m["conservation"] >= CONSERVATION_THRESH)
                )

                ccg = None
                if flagged and SPLIT_FINE_CCG:
                    ccg = _fine_ccg_for_pair(c1_times, c2_times)

                near_zero_frac = float(ccg["near_zero_frac"]) if ccg is not None else np.nan
                zero_peak_ratio = float(ccg["zero_peak_ratio"]) if ccg is not None else np.nan
                duplicate_fit_candidate = (
                    bool(flagged)
                    and np.isfinite(near_zero_frac)
                    and np.isfinite(zero_peak_ratio)
                    and (near_zero_frac >= DUPLICATE_NEAR_ZERO_FRAC_THRESH)
                    and (zero_peak_ratio >= DUPLICATE_ZERO_PEAK_RATIO_THRESH)
                )

                split_rows.append(
                    {
                        "ref_condition": REF_CONDITION,
                        "other_condition": other_key,
                        "ref_unit": ref_uid,
                        "child1": child1,
                        "child2": child2,
                        "agree1": a1,
                        "agree2": a2,
                        "split_score": split_score,
                        "segregation": m["segregation"],
                        "anticorr": m["anticorr"],
                        "conservation": m["conservation"],
                        "fine_ccg_near_zero_frac": near_zero_frac,
                        "fine_ccg_zero_peak_ratio": zero_peak_ratio,
                        "duplicate_fit_candidate": bool(duplicate_fit_candidate),
                        "flagged": bool(flagged),
                    }
                )

                if not flagged:
                    continue
                if pages_written >= SPLIT_MAX_PAGES_PDF:
                    continue

                t_mid = 0.5 * (bins_s[:-1] + bins_s[1:])
                denom = union_counts.astype(float)
                with np.errstate(divide="ignore", invalid="ignore"):
                    frac1 = c1_counts / np.where(denom > 0, denom, np.nan)

                fig, axes = plt.subplots(
                    3,
                    1,
                    figsize=(7.2, 5.2),
                    sharex=False,
                    gridspec_kw=dict(hspace=0.28, height_ratios=[1.2, 1.0, 0.9]),
                )
                axes_arr = np.asarray(axes).reshape(-1)
                axa, axb, axc = axes_arr.tolist()

                axa.plot(t_mid, ref_counts, lw=1.0, label=f"ref={ref_uid}", color="#444", alpha=0.7)
                axa.plot(t_mid, union_counts, lw=1.2, ls="--", label="coincident (c1∪c2)", color="#111")
                axa.plot(t_mid, c1_counts, lw=1.0, label=f"child1={child1} (a={a1:.2f})", color="#2B6CB0")
                axa.plot(t_mid, c2_counts, lw=1.0, label=f"child2={child2} (a={a2:.2f})", color="#C05621")
                axa.set_ylabel(f"Counts / {SPLIT_BIN_S:.0f}s")
                axa.legend(fontsize=7, ncol=2, loc="upper right")
                axa.set_title(
                    f"Split diag: {REF_CONDITION} → {other_key} | ref {ref_uid} top2 {child1},{child2} | "
                    f"seg={m['segregation']:.2f} corr={m['anticorr']:.2f} cons={m['conservation']:.2f}"
                )

                axb.plot(t_mid, frac1, lw=1.0, color="#444")
                axb.axhline(0.5, color="#aaa", lw=0.8, ls=":")
                axb.set_ylim(0, 1)
                axb.set_ylabel("child1 frac\n(coincident)")
                axb.set_xlabel("Time (s)")

                if SPLIT_FINE_CCG and ccg is not None:
                    axc.bar(ccg["bin_centers_s"] * 1e3, ccg["counts"], width=FINE_CCG_BIN_S * 1e3, color="#666", edgecolor="none")
                    axc.axvline(0, color="#aa0000", lw=0.8, ls=":")
                    axc.set_xlim(-FINE_CCG_WINDOW_S * 1e3, FINE_CCG_WINDOW_S * 1e3)
                    axc.set_xlabel("Child-child lag (ms)")
                    axc.set_ylabel("Pair count")
                    axc.set_title(
                        f"Fine CCG: near-zero frac={near_zero_frac:.3f}, peak/base={zero_peak_ratio:.2f}",
                        fontsize=8,
                    )
                else:
                    axc.axis("off")

                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)
                pages_written += 1

    split_df = pd.DataFrame(split_rows)
    split_df.to_csv(csv_path, index=False)

    print("Wrote:")
    print(f"  {pdf_path}")
    print(f"  {csv_path}")

print("Wrote:")
print(f"  {OUT_DIR / 'condition_stats.csv'}")
print(f"  {OUT_DIR / 'condition_stats.json'}")
print(f"  {OUT_DIR / 'comparisons_summary.json'}")
print(f"  {OUT_DIR / 'fig_overview.pdf'}")
