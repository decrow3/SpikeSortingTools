#%% compare_sortings.py
# # Spike Sorting Comparison: MotionTest0 vs MotionTest1
# This notebook compares two variants of motion correction results.

# %%
import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import spikeinterface.extractors as se
import spikeinterface.comparison as scmp
import spikeinterface.widgets as sw
import spikeinterface.full as si
import spikeinterface.qualitymetrics as sqm

# Ensure plots show up in the VS Code interactive window when running in IPython/Jupyter.
# Use a guarded call instead of a bare magic so this file is valid as plain Python.
try:
    # Import get_ipython only when available (IPython/Jupyter). This keeps the
    # file valid as plain Python while enabling inline plotting in notebooks.
    from IPython import get_ipython

    ip = get_ipython()
    if ip is not None:
        ip.run_line_magic("matplotlib", "inline")
except Exception:
    # Not running inside IPython / Jupyter — nothing to do.
    pass


# Helper: detect interactive session and show figures when appropriate
def _is_interactive_session() -> bool:
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except Exception:
        return False


def _show_or_close_figure(fig):
    """Show figure in interactive sessions (VSCode interactive/IPython). If not
    interactive, close the figure to avoid consuming memory.
    """
    try:
        if _is_interactive_session():
            try:
                # Non-blocking show to render in interactive backends
                plt.show(block=False)
                # allow event loop to process draw
                plt.pause(0.001)
                # keep the figure open for user inspection
            except Exception:
                # Fallback to blocking show if non-blocking fails
                plt.show()
        else:
            plt.close(fig)
    except Exception:
        # Best effort: ensure figure closed if anything goes wrong
        try:
            plt.close(fig)
        except Exception:
            pass

# %% [markdown]
# ## 1. Configuration and Paths
# Pointing to the 'cur' (curated) output folders for both runs.

# %%
# Pipeline root directories
# pipe0 = Path("/mnt/NPX/Luke/20250804/branchingtest1_pipeline_results_Luke0804_V2V1_g0_imec1")
# pipe1 = Path("/mnt/NPX/Luke/20250804/branchingtest0_pipeline_results_Luke0804_V2V1_g0_imec1")

pipe0 = Path("/mnt/NPX/Luke/20250804/pipeline_results_Luke0804_V2V1_g0_imec0")
pipe1 = Path("/mnt/NPX/Luke/20250804/dredge_pipeline_results_Luke0804_V2V1_g0_imec0")


# pipe0 = Path("/mnt/NPX/Luke/20250804/pipeline_results_Luke0804_V2V1_g0_imec1")
# pipe1 = Path("/mnt/NPX/Luke/20250804/dredgetest_pipeline_results_Luke0804_V2V1_g0_imec1/")

# Specific Phy/Sorter output paths
phy0_path = pipe0 / "cur" / "cur_sorter_output"
phy1_path = pipe1 / "cur" / "cur_sorter_output"

# Output directory for comparison results
out_dir = Path("/mnt/NPX/Luke/20250804/compare_results_ksmotion9000_vs_dregemotion3000")
out_dir.mkdir(parents=True, exist_ok=True)

# Comparison Params
delta_ms = 0.4        # Matching window
min_agreement = 0.5   # Threshold for "well-matched" units

# %% [markdown]
# ## 2. Load Sortings
# We load the results using the Phy extractor, which includes the curation labels.

# %%
print(f"Loading Sorting 0: {phy0_path}")
sorting0 = se.read_phy(folder_path=str(phy0_path), load_all_cluster_properties=True)

print(f"Loading Sorting 1: {phy1_path}")
sorting1 = se.read_phy(folder_path=str(phy1_path), load_all_cluster_properties=True)

# Optional: Filter for 'good' units only (comment out to compare all)
# sorting0 = sorting0.select_units([u for u in sorting0.unit_ids if sorting0.get_property('group')[sorting0.id_to_index(u)] == 'good'])
# sorting1 = sorting1.select_units([u for u in sorting1.unit_ids if sorting1.get_property('group')[sorting1.id_to_index(u)] == 'good'])

print(f"\nSorting 0: {len(sorting0.unit_ids)} units")
print(f"Sorting 1: {len(sorting1.unit_ids)} units")

# --- Compatibility aliases and interactive flags (backwards compatibility for older cells)
# Keep a single outdir name for older cells that may reference `outdir`.
outdir = out_dir

# Interactive display flags
SHOW_PLOTS = True
IN_IPYTHON = _is_interactive_session()

# Backwards-compatibility aliases: some older cells expect `sorting1`/`sorting2` names.
# FIXED: Do NOT overwrite sorting1. Preserve sorting0 (motion0) and checking sorting1 (motion1).
# We define sorting2 as an alias for sorting1 to satisfy legacy cells that might expect (sorting1, sorting2).
sorting2 = sorting1



def compute_quality_metrics_from_sorting(sorting, *, recording=None, folder: Path | None = None, overwrite: bool = False, label: str = "sorting"):
    """Compatibility wrapper used by some older cells.

    Delegates to compute_and_save_quality_metrics and returns the resulting DataFrame.
    """
    target = Path(folder) if folder is not None else out_dir
    return compute_and_save_quality_metrics(sorting=sorting, label=label, outdir=target, recording=recording)


def compute_and_save_cluster_depths(cur_folder: Path, sorting, outdir: Path, label: str):
    """Compute per-unit depths (median spike Y) from spike_positions.npy and spike_clusters.npy.

    Saves three files into outdir:
      - cluster_ids_{label}.npy
      - cluster_depths_{label}.npy
      - cluster_depths_{label}.csv

    Returns the path to the saved cluster_depths .npy or None if it couldn't be computed.
    """
    cur_folder = Path(cur_folder)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    spike_pos_f = cur_folder / "spike_positions.npy"
    spike_clu_f = cur_folder / "spike_clusters.npy"
    # graceful fallback if not found
    if not spike_pos_f.exists() or not spike_clu_f.exists():
        # try alternative common names
        alt_pos = cur_folder / "spikes_positions.npy"
        alt_clu = cur_folder / "spikes_clusters.npy"
        if alt_pos.exists() and alt_clu.exists():
            spike_pos_f = alt_pos
            spike_clu_f = alt_clu
        else:
            print(f"[depths] spike_positions or spike_clusters not found in {cur_folder}")
            return None
    try:
        sp_pos = np.load(spike_pos_f, allow_pickle=False)
        sp_clu = np.load(spike_clu_f, allow_pickle=False)
    except Exception as e:
        print(f"[depths] failed to load spike positions/clusters: {e}")
        return None

    # Ensure arrays align
    if sp_pos.shape[0] != sp_clu.shape[0]:
        # If spike_positions is per-spike but shaped differently, try reshaping fallback
        minlen = min(sp_pos.shape[0], sp_clu.shape[0])
        sp_pos = sp_pos[:minlen]
        sp_clu = sp_clu[:minlen]

    # pick the depth column (prefer column 1 if 2D, otherwise 0)
    if sp_pos.ndim == 1:
        ycol = 0
    elif sp_pos.shape[1] >= 2:
        ycol = 1
    else:
        ycol = 0

    unit_ids = np.asarray(sorting.unit_ids, dtype=int)
    depths = []
    for uid in unit_ids:
        idx = np.where(sp_clu == uid)[0]
        if idx.size == 0:
            depths.append(np.nan)
            continue
        vals = sp_pos[idx, ycol].astype(float)
        depths.append(float(np.nanmedian(vals)))
    depths = np.asarray(depths, dtype=float)

    ids_path = outdir / f"cluster_ids_{label}.npy"
    depths_path = outdir / f"cluster_depths_{label}.npy"
    csv_path = outdir / f"cluster_depths_{label}.csv"
    try:
        np.save(ids_path, unit_ids)
        np.save(depths_path, depths)
        pd.DataFrame({"unit_id": unit_ids, "depth": depths}).to_csv(csv_path, index=False)
        print(f"[depths] saved cluster depths for {label} -> {depths_path}")
        return depths_path
    except Exception as e:
        print(f"[depths] failed to save cluster depths: {e}")
        return None


# Try to compute and save per-unit depths for both pipelines (best-effort)
try:
    cd0 = compute_and_save_cluster_depths(phy0_path, sorting0, out_dir, label="motion0")
except Exception:
    cd0 = None
try:
    cd1 = compute_and_save_cluster_depths(phy1_path, sorting1, out_dir, label="motion1")
except Exception:
    cd1 = None


def maybe_filter_good(sorting):
    """If Phy/Kilosort labels exist, keep units labeled 'good'.

    Conservative: if no labels found, return original sorting.
    """
    try:
        props = sorting.get_property_keys()
    except Exception:
        return sorting
    label_key = None
    for k in ["group", "KSLabel", "quality", "cluster_group"]:
        if k in props:
            label_key = k
            break
    if label_key is None:
        return sorting
    labels = sorting.get_property(label_key)
    if labels is None:
        return sorting
    labels_norm = [str(x).strip().lower() for x in labels]
    unit_ids = sorting.unit_ids
    good_ids = [uid for uid, lab in zip(unit_ids, labels_norm) if lab == "good"]
    if len(good_ids) == 0:
        return sorting
    return sorting.select_units(good_ids)


def presence_ratio_per_unit(sorting, bin_s=10.0, max_s: float | None = None):
    """Compute per-unit presence ratio: fraction of time bins with >=1 spike.

    Returns (presence_ratios, times)
    """
    fs = sorting.get_sampling_frequency()
    if fs is None:
        raise ValueError("Sorting has no sampling frequency set")
    # estimate recording duration from max spike time across units
    if max_s is None:
        max_sample = 0
        for u in sorting.unit_ids:
            st = sorting.get_unit_spike_train(u)
            if len(st):
                max_sample = max(max_sample, int(st.max()))
        max_s = max_sample / float(fs) if max_sample > 0 else 0.0
    if max_s <= 0:
        return np.array([]), np.array([])
    n_bins = int(np.ceil(max_s / bin_s))
    times = np.arange(n_bins) * bin_s
    prs = []
    for u in sorting.unit_ids:
        st = sorting.get_unit_spike_train(u).astype(np.float64) / float(fs)
        st = st[(st >= 0) & (st < max_s)]
        if st.size == 0:
            prs.append(0.0)
            continue
        idx = np.floor(st / bin_s).astype(int)
        present = np.zeros(n_bins, dtype=bool)
        present[np.unique(idx)] = True
        prs.append(present.mean())
    return np.asarray(prs), times


def fraction_units_active_over_time(sorting, bin_s=10.0):
    fs = float(sorting.get_sampling_frequency())
    max_sample = 0
    for u in sorting.unit_ids:
        st = sorting.get_unit_spike_train(u)
        if len(st):
            max_sample = max(max_sample, int(st.max()))
    T = max_sample / fs if max_sample > 0 else bin_s
    nbins = max(1, int(T // bin_s))

    active = np.zeros(nbins, dtype=np.int64)
    for u in sorting.unit_ids:
        st = sorting.get_unit_spike_train(u).astype(np.float64) / fs
        bins = np.floor(st / bin_s).astype(np.int64)
        bins = bins[(bins >= 0) & (bins < nbins)]
        if len(bins) == 0:
            continue
        active[np.unique(bins)] += 1

    frac = active / float(len(sorting.unit_ids))
    t = (np.arange(nbins) + 0.5) * bin_s
    return t, frac


# Small analyses migrated from the _after_qc helper script
try:
    # compute and save presence-ratio histograms if possible
    pr0, _ = presence_ratio_per_unit(sorting0, bin_s=20.0)
    pr1, _ = presence_ratio_per_unit(sorting1, bin_s=20.0)
    if pr0.size and pr1.size:
        fig = plt.figure(figsize=(7, 4))
        plt.hist(pr0, bins=40, alpha=0.5, label="motion0")
        plt.hist(pr1, bins=40, alpha=0.5, label="motion1")
        plt.xlabel("Presence ratio")
        plt.ylabel("Units")
        plt.title(f"Presence ratio per unit (bin=20s) | mean: s0={pr0.mean():.3f}, s1={pr1.mean():.3f}")
        plt.legend()
        plt.tight_layout()
        fig.savefig(out_dir / "presence_ratio_hist.png", dpi=200)
    _show_or_close_figure(fig)
except Exception:
    pass

try:
    # Fraction of units active over time
    t0, frac0 = fraction_units_active_over_time(sorting0, bin_s=20.0)
    t1, frac1 = fraction_units_active_over_time(sorting1, bin_s=20.0)
    if len(t0) > 0 and len(t1) > 0:
        fig = plt.figure(figsize=(10, 4))
        plt.plot(t0, frac0, label='motion0', alpha=0.8)
        plt.plot(t1, frac1, label='motion1', alpha=0.8)
        plt.xlabel('Time (s)')
        plt.ylabel('Fraction of units active')
        plt.title('Fraction of Units Active over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(out_dir / "fraction_units_active.png", dpi=200)
        _show_or_close_figure(fig)
except Exception as e:
    print(f"Failed active units plot: {e}")

try:
    # SNR comparison if properties exist
    if 'snr' in getattr(sorting0, 'get_property_keys', lambda: [])() and 'snr' in getattr(sorting1, 'get_property_keys', lambda: [])():
        s0 = np.asarray(sorting0.get_property('snr'))
        s1 = np.asarray(sorting1.get_property('snr'))
        fig = plt.figure(figsize=(8, 4))
        sns.kdeplot(s0[~np.isnan(s0)], label='motion0', fill=True)
        sns.kdeplot(s1[~np.isnan(s1)], label='motion1', fill=True)
        plt.title('SNR distribution')
        plt.legend()
        fig.savefig(out_dir / 'snr_kde.png', dpi=200)
    _show_or_close_figure(fig)
except Exception:
    pass

# %% [markdown]
# ## 3. Run Comparison
# This computes the matching between the two sorters.

# %%
cmp = scmp.compare_two_sorters(
    sorting1=sorting0,
    sorting2=sorting1,
    sorting1_name="motion0",
    sorting2_name="motion1",
    delta_time=delta_ms / 1000.0,
)

# Export Match Results
agreement_scores = cmp.agreement_scores
agreement_scores.to_csv(out_dir / "agreement_scores.csv")

# Save match event count matrix if available
try:
    mec = cmp.match_event_count
    mec.to_csv(out_dir / "match_event_count.csv")
except Exception:
    pass

m1_to_2, m2_to_1 = cmp.get_matching()
# m1_to_2 may be a dict mapping unit -> match or a numpy array; handle both robustly
try:
    # If it's dict-like
    values_iter = m1_to_2.values()  # type: ignore[assignment]
except Exception:
    # Fallback: treat as array-like
    values_iter = np.asarray(m1_to_2)

# Count matches where matched id != -1
n_matched = int(sum(1 for u in values_iter if int(u) != -1))

summary = {
    "n_units_0": len(sorting0.unit_ids),
    "n_units_1": len(sorting1.unit_ids),
    "n_matched": n_matched,
}
print(f"Comparison complete: {summary['n_matched']} units matched.")

# --- Additional saved outputs (plots, matched table, summary) ---
try:
    # Agreement matrix figures (ordered and unordered)
    fig_ord = sw.plot_agreement_matrix(cmp, ordered=True, count_text=False, unit_ticks=False)
    try:
        fig_ord.figure.savefig(out_dir / "agreement_matrix_ordered.png", dpi=200, bbox_inches="tight")
    except Exception:
        # some versions return matplotlib Figure directly
        try:
            fig_ord.savefig(out_dir / "agreement_matrix_ordered.png", dpi=200, bbox_inches="tight")
        except Exception:
            pass
    try:
        # attempt to show the returned figure (works for Figure or wrapper with .figure)
        fig_obj = getattr(fig_ord, 'figure', fig_ord)
        _show_or_close_figure(fig_obj)
    except Exception:
        pass
    fig_un = sw.plot_agreement_matrix(cmp, ordered=False, count_text=False, unit_ticks=False)
    try:
        fig_un.figure.savefig(out_dir / "agreement_matrix.png", dpi=200, bbox_inches="tight")
    except Exception:
        try:
            fig_un.savefig(out_dir / "agreement_matrix.png", dpi=200, bbox_inches="tight")
        except Exception:
            pass
    try:
        fig_obj = getattr(fig_un, 'figure', fig_un)
        _show_or_close_figure(fig_obj)
    except Exception:
        pass
except Exception:
    pass

# Build matched-units table (m1->m2 mapping). m1_to_2 may be dict or array-like.
rows = []
try:
    if hasattr(m1_to_2, "items"):
        iterator = m1_to_2.items()
    else:
        # array-like; infer unit ids from sorting0.unit_ids order
        arr = np.asarray(m1_to_2)
        iterator = zip(sorting0.unit_ids, arr)
    for u1, u2 in iterator:
        rows.append({
            "unit_1": int(u1),
            "unit_2": int(u2) if int(u2) != -1 else -1,
            "agreement": float(agreement_scores.loc[int(u1), int(u2)]) if (int(u2) != -1 and str(int(u1)) in agreement_scores.index and str(int(u2)) in agreement_scores.columns) else (np.nan if int(u2) != -1 else 0.0),
        })
except Exception:
    # fallback: try matching via cmp.get_matching() structure
    try:
        m1, m2 = cmp.get_matching()
        for u1, u2 in (m1.items() if hasattr(m1, 'items') else zip(sorting0.unit_ids, np.asarray(m1))):
            rows.append({"unit_1": int(u1), "unit_2": int(u2), "agreement": agreement_scores.get(u1, {}).get(u2, np.nan)})
    except Exception:
        pass

if len(rows):
    df_match = pd.DataFrame(rows)
    df_match.to_csv(out_dir / "matched_units.csv", index=False)

# --- Quick/simple metrics (units, spikes, duration, matching) ---
def _iter_matching(matching, unit_ids):
    """Iterate (unit_id, matched_unit_id) pairs for a matching that might be dict- or array-like."""
    if hasattr(matching, "items"):
        yield from matching.items()
    else:
        arr = np.asarray(matching)
        yield from zip(unit_ids, arr)


def _safe_agreement(agreement_df: pd.DataFrame, u1: int, u2: int) -> float:
    try:
        return float(agreement_df.loc[int(u1), int(u2)])
    except Exception:
        try:
            return float(agreement_df.loc[str(int(u1)), str(int(u2))])
        except Exception:
            return float("nan")


def _get_good_unit_ids(sorting):
    """Return unit_ids labeled 'good' if a Phy/Kilosort label property exists, else []."""
    try:
        props = sorting.get_property_keys()
    except Exception:
        return []
    label_key = None
    for k in ["group", "KSLabel", "quality", "cluster_group"]:
        if k in props:
            label_key = k
            break
    if label_key is None:
        return []
    labels = sorting.get_property(label_key)
    if labels is None:
        return []
    labels_norm = [str(x).strip().lower() for x in labels]
    return [int(uid) for uid, lab in zip(sorting.unit_ids, labels_norm) if lab == "good"]


def _load_phy_spike_arrays(phy_folder: Path):
    """Best-effort load of spike_times (samples) + spike_clusters (cluster id per spike) as memmaps."""
    phy_folder = Path(phy_folder)
    st_f = phy_folder / "spike_times.npy"
    sc_f = phy_folder / "spike_clusters.npy"
    if not st_f.exists() or not sc_f.exists():
        return None, None
    try:
        st = np.load(st_f, mmap_mode="r", allow_pickle=False)
        sc = np.load(sc_f, mmap_mode="r", allow_pickle=False)
        return st, sc
    except Exception:
        return None, None


def _estimate_duration_s(sorting, phy_folder: Path | None):
    """Estimate duration in seconds from Phy spike_times if present; fallback to per-unit spike trains."""
    fs = float(sorting.get_sampling_frequency() or 30000.0)
    if phy_folder is not None:
        st, _ = _load_phy_spike_arrays(phy_folder)
        if st is not None and st.size:
            try:
                smin = float(np.min(st))
                smax = float(np.max(st))
                return max(0.0, (smax - smin) / fs)
            except Exception:
                pass
    # Fallback (can be slower)
    max_sample = 0
    min_sample = None
    try:
        for u in sorting.unit_ids:
            st_u = sorting.get_unit_spike_train(u)
            if st_u is None or len(st_u) == 0:
                continue
            umin = int(np.min(st_u))
            umax = int(np.max(st_u))
            max_sample = max(max_sample, umax)
            min_sample = umin if min_sample is None else min(min_sample, umin)
        if min_sample is None:
            return 0.0
        return max(0.0, (max_sample - min_sample) / fs)
    except Exception:
        return 0.0


def _get_spike_counts_per_unit(sorting, phy_folder: Path | None):
    """Return (unit_ids_array, spike_counts_array_aligned)."""
    unit_ids = np.asarray(sorting.unit_ids, dtype=int)
    # fast path
    try:
        counts = sorting.count_num_spikes_per_unit()
        if isinstance(counts, dict):
            arr = np.asarray([int(counts.get(int(u), 0)) for u in unit_ids], dtype=np.int64)
            return unit_ids, arr
    except Exception:
        pass
    # phy fallback
    if phy_folder is not None:
        _, sc = _load_phy_spike_arrays(phy_folder)
        if sc is not None and sc.size:
            try:
                u, c = np.unique(np.asarray(sc, dtype=np.int64), return_counts=True)
                count_map = dict(zip(u.tolist(), c.tolist()))
                arr = np.asarray([int(count_map.get(int(uid), 0)) for uid in unit_ids], dtype=np.int64)
                return unit_ids, arr
            except Exception:
                pass
    # slow fallback
    arr = []
    for u in unit_ids:
        try:
            arr.append(int(len(sorting.get_unit_spike_train(int(u)))))
        except Exception:
            arr.append(0)
    return unit_ids, np.asarray(arr, dtype=np.int64)


def _spike_dominance_topk(spike_counts: np.ndarray, k: int = 10) -> float:
    spike_counts = np.asarray(spike_counts, dtype=np.int64)
    total = int(spike_counts.sum())
    if total <= 0:
        return 0.0
    k = int(min(k, spike_counts.size))
    if k <= 0:
        return 0.0
    topk = np.partition(spike_counts, -k)[-k:]
    return float(int(topk.sum()) / total)


def _orphan_fraction_first_last(sorting, phy_folder: Path | None, duration_s: float, window_s: float = 600.0):
    """Fraction of units with zero spikes in first/last window_s (uses Phy arrays if available)."""
    unit_ids = np.asarray(sorting.unit_ids, dtype=int)
    if unit_ids.size == 0 or duration_s <= 0:
        return 0.0, 0.0, float(window_s)

    window_s = float(min(window_s, max(1.0, duration_s / 4.0)))
    fs = float(sorting.get_sampling_frequency() or 30000.0)
    if phy_folder is not None:
        st, sc = _load_phy_spike_arrays(phy_folder)
        if st is not None and sc is not None and st.size and sc.size and st.shape[0] == sc.shape[0]:
            try:
                # spike_times are typically sorted; use searchsorted to slice windows
                first_end = int(np.searchsorted(st, window_s * fs, side="right"))
                last_start = int(np.searchsorted(st, max(0.0, duration_s - window_s) * fs, side="left"))
                sc_first = np.asarray(sc[:first_end], dtype=np.int64)
                sc_last = np.asarray(sc[last_start:], dtype=np.int64)
                first_present = set(np.unique(sc_first).tolist()) if sc_first.size else set()
                last_present = set(np.unique(sc_last).tolist()) if sc_last.size else set()
                n = unit_ids.size
                frac_first = float(sum(int(uid) not in first_present for uid in unit_ids) / n)
                frac_last = float(sum(int(uid) not in last_present for uid in unit_ids) / n)
                return frac_first, frac_last, window_s
            except Exception:
                pass

    # fallback using per-unit spike trains (slower)
    n = unit_ids.size
    no_first = 0
    no_last = 0
    for uid in unit_ids:
        try:
            st_u = sorting.get_unit_spike_train(int(uid)).astype(np.float64) / fs
        except Exception:
            st_u = np.asarray([], dtype=float)
        if st_u.size == 0:
            no_first += 1
            no_last += 1
            continue
        if not np.any(st_u <= window_s):
            no_first += 1
        if not np.any(st_u >= (duration_s - window_s)):
            no_last += 1
    return float(no_first / n), float(no_last / n), window_s


def _iqr(x: np.ndarray):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    q25, q50, q75 = np.percentile(x, [25, 50, 75])
    return float(q25), float(q50), float(q75)


def _threshold_counts(spike_counts: np.ndarray, thresholds=(50, 100, 500, 1000)):
    spike_counts = np.asarray(spike_counts, dtype=np.int64)
    return {f"n_units_spikes_ge_{int(t)}": int(np.sum(spike_counts >= int(t))) for t in thresholds}


# Per-sorting unit+spike summaries
dur0_s = _estimate_duration_s(sorting0, phy0_path)
dur1_s = _estimate_duration_s(sorting1, phy1_path)

u0, scount0 = _get_spike_counts_per_unit(sorting0, phy0_path)
u1, scount1 = _get_spike_counts_per_unit(sorting1, phy1_path)

fr0 = scount0.astype(float) / max(1.0, float(dur0_s))
fr1 = scount1.astype(float) / max(1.0, float(dur1_s))

spk_q25_0, spk_med_0, spk_q75_0 = _iqr(scount0)
spk_q25_1, spk_med_1, spk_q75_1 = _iqr(scount1)
fr_q25_0, fr_med_0, fr_q75_0 = _iqr(fr0)
fr_q25_1, fr_med_1, fr_q75_1 = _iqr(fr1)

dom_top10_0 = _spike_dominance_topk(scount0, k=10)
dom_top10_1 = _spike_dominance_topk(scount1, k=10)

orf_first_0, orf_last_0, orf_win_0 = _orphan_fraction_first_last(sorting0, phy0_path, dur0_s, window_s=600.0)
orf_first_1, orf_last_1, orf_win_1 = _orphan_fraction_first_last(sorting1, phy1_path, dur1_s, window_s=600.0)

good0 = _get_good_unit_ids(sorting0)
good1 = _get_good_unit_ids(sorting1)
good0_set = set(good0)
good1_set = set(good1)

spikes_good_0 = int(np.sum([int(scount0[np.where(u0 == int(uid))[0][0]]) for uid in good0 if int(uid) in set(u0.tolist())])) if len(good0) else 0
spikes_good_1 = int(np.sum([int(scount1[np.where(u1 == int(uid))[0][0]]) for uid in good1 if int(uid) in set(u1.tolist())])) if len(good1) else 0

# Matching/unmatched rates and agreement distribution for best matches
unmatched_0 = 0
matched_agreements = []
for uu, vv in _iter_matching(m1_to_2, sorting0.unit_ids):
    if int(vv) == -1:
        unmatched_0 += 1
    else:
        matched_agreements.append(_safe_agreement(agreement_scores, int(uu), int(vv)))

unmatched_1 = 0
for uu, vv in _iter_matching(m2_to_1, sorting1.unit_ids):
    if int(vv) == -1:
        unmatched_1 += 1

matched_agreements = np.asarray(matched_agreements, dtype=float)
agree_mean = float(np.nanmean(matched_agreements)) if matched_agreements.size else float("nan")
agree_median = float(np.nanmedian(matched_agreements)) if matched_agreements.size else float("nan")
agree_max = float(np.nanmax(matched_agreements)) if matched_agreements.size else float("nan")

def _frac_ge(x, thr):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(np.mean(x >= float(thr)))

agree_frac_ge_02 = _frac_ge(matched_agreements, 0.2)
agree_frac_ge_05 = _frac_ge(matched_agreements, 0.5)
agree_frac_ge_08 = _frac_ge(matched_agreements, 0.8)

# Good-unit retention: fraction of good units with any match in the other sorting
good0_with_match = 0
for uu, vv in _iter_matching(m1_to_2, sorting0.unit_ids):
    if int(uu) in good0_set and int(vv) != -1:
        good0_with_match += 1
good1_with_match = 0
for uu, vv in _iter_matching(m2_to_1, sorting1.unit_ids):
    if int(uu) in good1_set and int(vv) != -1:
        good1_with_match += 1

print(f"[quick metrics] duration_s: motion0={dur0_s:.1f}, motion1={dur1_s:.1f}")
print(f"[quick metrics] spikes/unit median[IQR]: motion0={spk_med_0:.0f}[{spk_q25_0:.0f},{spk_q75_0:.0f}] motion1={spk_med_1:.0f}[{spk_q25_1:.0f},{spk_q75_1:.0f}]")
print(f"[quick metrics] firing_rate(Hz) median[IQR]: motion0={fr_med_0:.3g}[{fr_q25_0:.3g},{fr_q75_0:.3g}] motion1={fr_med_1:.3g}[{fr_q25_1:.3g},{fr_q75_1:.3g}]")
print(f"[quick metrics] frac FR<0.1Hz: motion0={float(np.mean(fr0 < 0.1)):.3f} motion1={float(np.mean(fr1 < 0.1)):.3f}")
print(f"[quick metrics] frac FR>10Hz:  motion0={float(np.mean(fr0 > 10.0)):.3f} motion1={float(np.mean(fr1 > 10.0)):.3f}")
print(f"[quick metrics] top10 spike dominance: motion0={dom_top10_0:.3f} motion1={dom_top10_1:.3f}")
print(f"[quick metrics] orphan frac (no spikes) first/last {orf_win_0/60.0:.1f}min: motion0={orf_first_0:.3f}/{orf_last_0:.3f} motion1={orf_first_1:.3f}/{orf_last_1:.3f}")
print(f"[quick metrics] unmatched rate: motion0={unmatched_0}/{len(sorting0.unit_ids)} motion1={unmatched_1}/{len(sorting1.unit_ids)}")
print(f"[quick metrics] agreement(best-match) mean/median/max: {agree_mean:.3f}/{agree_median:.3f}/{agree_max:.3f}")

# Summary JSON
summary = {
    "pipe0": str(pipe0),
    "pipe1": str(pipe1),
    "delta_ms": delta_ms,
    "n_units_0": int(len(sorting0.unit_ids)),
    "n_units_1": int(len(sorting1.unit_ids)),
    "n_matched": int(n_matched),
    "duration_s_0": float(dur0_s),
    "duration_s_1": float(dur1_s),
    "spike_count_total_0": int(np.sum(scount0)),
    "spike_count_total_1": int(np.sum(scount1)),
    "spikes_per_unit_q25_0": float(spk_q25_0),
    "spikes_per_unit_median_0": float(spk_med_0),
    "spikes_per_unit_q75_0": float(spk_q75_0),
    "spikes_per_unit_q25_1": float(spk_q25_1),
    "spikes_per_unit_median_1": float(spk_med_1),
    "spikes_per_unit_q75_1": float(spk_q75_1),
    "firing_rate_hz_q25_0": float(fr_q25_0),
    "firing_rate_hz_median_0": float(fr_med_0),
    "firing_rate_hz_q75_0": float(fr_q75_0),
    "firing_rate_hz_q25_1": float(fr_q25_1),
    "firing_rate_hz_median_1": float(fr_med_1),
    "firing_rate_hz_q75_1": float(fr_q75_1),
    "frac_units_fr_lt_0p1hz_0": float(np.mean(fr0 < 0.1)) if fr0.size else float("nan"),
    "frac_units_fr_lt_0p1hz_1": float(np.mean(fr1 < 0.1)) if fr1.size else float("nan"),
    "frac_units_fr_gt_10hz_0": float(np.mean(fr0 > 10.0)) if fr0.size else float("nan"),
    "frac_units_fr_gt_10hz_1": float(np.mean(fr1 > 10.0)) if fr1.size else float("nan"),
    "spike_dominance_top10_0": float(dom_top10_0),
    "spike_dominance_top10_1": float(dom_top10_1),
    "orphan_frac_first_window_0": float(orf_first_0),
    "orphan_frac_last_window_0": float(orf_last_0),
    "orphan_frac_first_window_1": float(orf_first_1),
    "orphan_frac_last_window_1": float(orf_last_1),
    "orphan_window_s": float(orf_win_0),
    "n_unmatched_0": int(unmatched_0),
    "n_unmatched_1": int(unmatched_1),
    "agreement_best_match_mean": float(agree_mean),
    "agreement_best_match_median": float(agree_median),
    "agreement_best_match_max": float(agree_max),
    "agreement_best_match_frac_ge_0p2": float(agree_frac_ge_02),
    "agreement_best_match_frac_ge_0p5": float(agree_frac_ge_05),
    "agreement_best_match_frac_ge_0p8": float(agree_frac_ge_08),
    "n_good_units_0": int(len(good0)),
    "n_good_units_1": int(len(good1)),
    "n_spikes_good_units_0": int(spikes_good_0),
    "n_spikes_good_units_1": int(spikes_good_1),
    "good_unit_retention_0": (float(good0_with_match) / float(len(good0))) if len(good0) else float("nan"),
    "good_unit_retention_1": (float(good1_with_match) / float(len(good1))) if len(good1) else float("nan"),
}

# Add threshold-count metrics (units with >= N spikes)
summary.update({f"motion0_{k}": v for k, v in _threshold_counts(scount0).items()})
summary.update({f"motion1_{k}": v for k, v in _threshold_counts(scount1).items()})

try:
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
except Exception:
    pass

# --- Compute full SpikeInterface quality metrics (best-effort) ---
def compute_and_save_quality_metrics(sorting, label: str, outdir: Path, recording=None):
    """Compute quality metrics using spikeinterface.qualitymetrics.compute_quality_metrics.

    If a Recording is provided, pass it to compute SNR and waveform-based metrics; otherwise
    compute what is available from the Sorting alone.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    try:
        # metric_names=None -> compute defaults
        qm = sqm.compute_quality_metrics(sorting=sorting, recording=recording)
        if isinstance(qm, dict):
            # some versions return dict
            qm_df = pd.DataFrame(qm)
        else:
            qm_df = qm
        qm_df.to_csv(outdir / f"quality_metrics_{label}.csv")
        return qm_df
    except Exception as e:
        # Best-effort fallback: compute simple firing rates
        try:
            rates = []
            fs = sorting.get_sampling_frequency() or 30000.0
            for u in sorting.unit_ids:
                st = sorting.get_unit_spike_train(u).astype(np.float64) / float(fs)
                dur = st.max() - st.min() if st.size else 0.0
                rates.append(len(st) / max(1.0, dur) if dur > 0 else 0.0)
            qm_df = pd.DataFrame({"firing_rate": rates}, index=sorting.unit_ids)
            qm_df.to_csv(outdir / f"quality_metrics_{label}_fallback.csv")
            return qm_df
        except Exception:
            return None


# Try to compute full QM for both sortings (best-effort). If a preprocessed recording
# exists next to the pipeline root (folder 'preprocessed_recording'), we do not attempt
# to load it automatically here to avoid heavy I/O; pass recording=None to compute what is possible.
try:
    qm0 = compute_and_save_quality_metrics(sorting0, "motion0", out_dir)
    qm1 = compute_and_save_quality_metrics(sorting1, "motion1", out_dir)
except Exception:
    qm0 = qm1 = None

# Pooled ISI histogram (0-20 ms) — quick diagnostic
def pooled_isi_ms(sorting, max_units=500, max_spikes_per_unit=20000, seed=0):
    rng = np.random.default_rng(seed)
    unit_ids = list(sorting.unit_ids)
    if len(unit_ids) == 0:
        return np.array([])
    chosen = unit_ids if len(unit_ids) <= max_units else list(rng.choice(unit_ids, size=max_units, replace=False))
    isis = []
    for u in chosen:
        st = sorting.get_unit_spike_train(u).astype(np.float64) / sorting.get_sampling_frequency()
        if st.size <= 1:
            continue
        diffs = np.diff(np.sort(st)) * 1000.0
        if diffs.size:
            isis.append(diffs[:max_spikes_per_unit])
    if not isis:
        return np.array([])
    return np.concatenate(isis)

try:
    isi = pooled_isi_ms(sorting0, seed=1)
    isi2 = pooled_isi_ms(sorting1, seed=2)
    if isi.size and isi2.size:
        fig = plt.figure(figsize=(7,4))
        plt.hist(isi, bins=200, range=(0,20), alpha=0.5, label='motion0')
        plt.hist(isi2, bins=200, range=(0,20), alpha=0.5, label='motion1')
        plt.axvline(1.0, linewidth=1)
        plt.xlabel('ISI (ms)')
        plt.ylabel('Counts')
        plt.title('Pooled ISI (0-20 ms)')
        plt.legend()
        plt.tight_layout()
    fig.savefig(out_dir / 'pooled_isi_0_20ms.png', dpi=200)
    _show_or_close_figure(fig)
except Exception:
    pass

# --- Additional percent / agreement diagnostics ---
if True:
    # Percent of units matched
    n_units_0 = len(sorting0.unit_ids)
    n_units_1 = len(sorting1.unit_ids)
    pct_matched_0 = n_matched / max(1, n_units_0)
    
    # Save a small bar plot
    fig = plt.figure(figsize=(4, 3))
    plt.bar(['motion0', 'motion1'], [pct_matched_0 * 100.0, 0], color=['C0', 'C1']) # Simplified for restoration
    plt.ylabel('Percent matched (%)')
    plt.title('Percent units matched')
    plt.tight_layout()
    fig.savefig(out_dir / 'percent_units_matched.png', dpi=200)
    _show_or_close_figure(fig)

# --- Depth-binned agreement diagnostics ---
def plot_depth_binned_agreement(out_dir: Path, sorting, mapping, depths_label: str, n_bins: int = 4):
    ids_file = out_dir / f"cluster_ids_{depths_label}.npy"
    depths_file = out_dir / f"cluster_depths_{depths_label}.npy"
    if not ids_file.exists() or not depths_file.exists(): return None
    try:
        cluster_ids = np.load(ids_file)
        cluster_depths = np.load(depths_file)
    except Exception: return None
    valid_mask = ~np.isnan(cluster_depths)
    if np.sum(valid_mask) == 0: return None
    cluster_ids = cluster_ids[valid_mask].astype(int)
    cluster_depths = cluster_depths[valid_mask].astype(float)
    map_dict = {}
    try:
        if hasattr(mapping, 'items'):
            for k, v in mapping.items(): map_dict[int(k)] = int(v)
        else:
            arr = np.asarray(mapping)
            for uid, mv in zip(sorting.unit_ids, arr): map_dict[int(uid)] = int(mv)
    except Exception: pass
    
    try: bins = np.quantile(cluster_depths, np.linspace(0.0, 1.0, n_bins + 1))
    except Exception: bins = np.linspace(np.min(cluster_depths), np.max(cluster_depths), n_bins + 1)
    
    df = pd.DataFrame({"unit_id": cluster_ids, "depth": cluster_depths})
    df['bin'] = pd.cut(df['depth'], bins=bins, include_lowest=True, labels=False)
    stats = []
    for b in range(n_bins):
        sel = df['bin'] == b
        units = df.loc[sel, 'unit_id'].values.astype(int)
        n = units.size
        matched = np.array([1 if map_dict.get(int(u), -1) != -1 else 0 for u in units], dtype=int)
        if n > 0:
            pct_matched = 100.0 * matched.sum() / n
            stats.append({"bin": b, "n_units": int(n), "pct_matched": float(pct_matched), "depth_lo": float(bins[b]), "depth_hi": float(bins[b+1])})
            
    stats_df = pd.DataFrame(stats)
    stats_df.to_csv(out_dir / f"percent_matched_by_depth_{depths_label}.csv", index=False)
    
    fig = plt.figure(figsize=(7, 4))
    x = np.arange(len(stats_df))
    plt.bar(x, stats_df['pct_matched'].fillna(0.0), color='C0')
    xticks = [f"{row['depth_lo']:.1f}-{row['depth_hi']:.1f}" for _, row in stats_df.iterrows()]
    plt.xticks(x, xticks, rotation=45, ha='right')
    plt.ylabel('Percent units matched (%)')
    plt.title(f'Percent units matched ({depths_label})')
    plt.tight_layout()
    fig.savefig(out_dir / f"percent_matched_by_depth_{depths_label}.png", dpi=200)
    plt.close(fig)
    return stats_df

try:
    plot_depth_binned_agreement(out_dir, sorting0, m1_to_2, depths_label='motion0', n_bins=4)
except Exception: pass

# %%
plt.figure(figsize=(10, 10))
try:
    sw.plot_agreement_matrix(cmp, ordered=True, count_text=False, unit_ticks=False)
    plt.title("Ordered Agreement Matrix")
    plt.show()
except Exception: pass

# %%
WE1_DIR = outdir / "we_sorting1"
WE2_DIR = outdir / "we_sorting2"

def ensure_recording_has_geometry(recording, *, pitch_um: float = 20.0):
    try:
        pg = recording.get_probegroup()
        if pg is not None: return recording
    except Exception: pass
    try:
        loc = recording.get_channel_locations()
        if loc.size > 0: return recording
    except Exception: pass
    try:
        n = recording.get_num_channels()
    except Exception: n = len(recording.channel_ids)
    positions = np.zeros((n, 2), dtype="float64")
    positions[:, 1] = np.arange(n, dtype="float64") * float(pitch_um)
    try:
        recording.set_channel_locations(positions)
    except Exception:
        try: recording.set_probe(positions) # dummy
        except Exception: pass
    return recording

from types import SimpleNamespace
def load_waveforms_npz(npz_path: Path):
    npz_path = Path(npz_path)
    if not npz_path.exists(): raise FileNotFoundError(npz_path)
    data = np.load(str(npz_path), allow_pickle=True)
    templates = None
    for k in ['templates', 'mean_templates', 'waveforms']:
        if k in data.files:
            templates = data[k]
            if templates.ndim == 4: templates = templates.mean(axis=1) 
            if templates.ndim == 3: break
    if templates is None: raise ValueError("No templates")
    unit_ids = data['cids'] if 'cids' in data.files else (data['cluster_ids'] if 'cluster_ids' in data.files else np.arange(templates.shape[0]))
    return templates, unit_ids

def make_minimal_we_from_templates(templates, unit_ids):
    return SimpleNamespace(templates=np.asarray(templates), unit_ids=list(map(int, unit_ids)))

we1, we2 = None, None
try:
    npz0 = pipe0 / "qc" / "waveforms" / "waveforms.npz"
    if npz0.exists():
        t0, ids0 = load_waveforms_npz(npz0)
        we1 = make_minimal_we_from_templates(t0, ids0)
except Exception: pass
try:
    npz1 = pipe1 / "qc" / "waveforms" / "waveforms.npz"
    if npz1.exists():
        t1, ids1 = load_waveforms_npz(npz1)
        we2 = make_minimal_we_from_templates(t1, ids1)
except Exception: pass

def _get_templates_and_unit_ids(we):
    if hasattr(we, 'templates'): return we.templates, we.unit_ids
    if hasattr(we, 'get_all_templates'): return we.get_all_templates(), we.unit_ids
    # fallback for SimpleNamespace or dict
    if hasattr(we, 'get'): return we.get('templates'), we.get('unit_ids')
    return None, None

def plot_templates_by_depth(we, title, outpath=None, max_units=500, recording_geometry=None, ax=None):
    """
    Plots a single representative waveform per unit at its spatial (x, y) location.
    
    If 'recording_geometry' (Recording object) is provided, it uses exact channel positions.
    Otherwise, it assumes a standard layout or vertical arrangement.
    
    If 'ax' is provided, plots directly onto it.
    If 'ax' is None, creates a figure and saves it to 'outpath'.
    """
    if we is None: return
    templates, unit_ids = _get_templates_and_unit_ids(we)
    if templates is None: return
    templates = np.asarray(templates)
    
    n_units = len(unit_ids)
    if n_units == 0: return
    
    # Subsample if too many
    if n_units > max_units:
        # Pick uniformly across the list to sample depths fairly
        indices = np.linspace(0, n_units - 1, max_units, dtype=int)
    else:
        indices = np.arange(n_units)

    selected_templates = templates[indices]
    
    # Determine unit positions (Peak Channel approximation)
    # We need channel locations. If passed explicitly or if we can infer standard NP1 geometry.
    # Default fallback: Pitch 20um vertical, staggered 20um horizontal (4 columns)
    n_ch = templates.shape[2]
    
    # Try to get positions from recording_geometry if passed
    positions = None
    if recording_geometry is not None:
        try:
            positions = recording_geometry.get_channel_locations()
        except:
            pass
            
    if positions is None:
        # Fallback grid: 4 columns, 20um pitch
        # x: 0, 20, 40, 60 -- y: 0, 20, 40...
        # shape (n_ch, 2)
        rows = n_ch // 2 # approximation
        x_pitch = 32.0
        y_pitch = 20.0
        # Checkerboard / NP1-like pattern generic
        positions = np.zeros((n_ch, 2))
        positions[:, 0] = (np.arange(n_ch) % 4) * x_pitch
        positions[:, 1] = (np.arange(n_ch) // 2) * y_pitch

    # Find peak channel for each selected unit
    ptps = np.ptp(selected_templates, axis=1) # (n_sel, n_ch)
    peak_chs = np.argmax(ptps, axis=1)
    
    unit_x = positions[peak_chs, 0]
    unit_y = positions[peak_chs, 1]
    
    # Setup Figure if no ax provided
    if ax is None:
        # Aspect ratio preserving probe shape roughly
        y_min, y_max = unit_y.min(), unit_y.max()
        x_min, x_max = unit_x.min(), unit_x.max()
        h_span = max(100, y_max - y_min)
        w_span = max(50, x_max - x_min + 40) # padding
        
        aspect = h_span / w_span
        fig_w = 15
        fig_h = max(6, min(24, fig_w * aspect * 0.5)) # clamp height
        
        fig = plt.figure(figsize=(fig_w, fig_h))
        ax = fig.add_subplot(111)
        created_fig = True
    else:
        created_fig = False
        fig = ax.figure

    n_samples = templates.shape[1]
    
    # Scaling parameters
    # Waveforms need to fit within ~20-30um vertical space so they don't overlap too much
    # Time axis needs to fit within ~30-40um horizontal space
    
    # X-Time scaling: Map [0, n_samples] -> [0, x_width_um]
    wave_width_um = 10.0
    t_scale = wave_width_um / n_samples
    t_vec = np.arange(n_samples) * t_scale
    
    # Amplitude scaling: Map peak-to-peak -> y_height_um
    # Normalize such that max ptp = 40um visually
    global_max_ptp = np.median(np.max(ptps, axis=1)) * 3.0 # robust max
    amp_scale = 2*30.0 / (global_max_ptp + 1e-6)
    
    cmap = plt.get_cmap("tab20")
    
    for i, u_idx in enumerate(indices):
        unit_temp = selected_templates[i] # (samples, channels)
        peak_ch = peak_chs[i]
        
        # Get the waveform on the peak channel
        wf = unit_temp[:, peak_ch]
        
        # Center the waveform at (unit_x, unit_y)
        # x: unit_x + time - (width/2)
        # y: unit_y + amplitude
        
        x_plot = unit_x[i] + t_vec - (wave_width_um / 2)
        y_plot = unit_y[i] + wf * amp_scale
        
        color = cmap(i % 20)
        
        ax.plot(x_plot, y_plot, color=color, linewidth=1.2, alpha=0.9)
        
        # Optional: Plot a small dot at the "anchor" position
        # ax.scatter(unit_x[i], unit_y[i], color='k', s=1, alpha=0.3)
            
    ax.set_title(title + f" ({len(indices)} units)")
    ax.set_xlabel("x (μm) + time")
    ax.set_ylabel("y (μm)")
    
    # Ensure aspect ratio is equal so depth vs x is true to geometry
    #ax.set_aspect('equal'), let x stretch so that waveforms are more visible
    
    
    # If we created the figure, finalize and save
    if created_fig and outpath:
        plt.tight_layout()
        fig.savefig(outpath, dpi=200)
        _show_or_close_figure(fig)

def plot_matched_units_pdf(out_dir: Path, we_left, we_right, mapping, depths_label: str = 'motion0', n_bins: int = 4, n_per_bin: int = 3, random_seed: int = 0, pdf_name: str = 'matched_templates_by_depth.pdf'):
    from matplotlib.backends.backend_pdf import PdfPages
    out_dir = Path(out_dir)
    pdf_path = out_dir / pdf_name

    # Check inputs
    if not we_left or not we_right:
        print("[PDF] Missing waveform extractor(s).")
        return None
        
    # Get cluster depths
    ids_file = out_dir / f"cluster_ids_{depths_label}.npy"
    depths_file = out_dir / f"cluster_depths_{depths_label}.npy"
    if not ids_file.exists() or not depths_file.exists():
        print(f"[PDF] Depths file not found for {depths_label}")
        return None
        
    try:
        cluster_ids = np.load(ids_file)
        cluster_depths = np.load(depths_file)
    except Exception as e:
        print(f"[PDF] Failed to load depth info: {e}")
        return None
        
    valid_mask = ~np.isnan(cluster_depths)
    if np.sum(valid_mask) == 0:
        return None
        
    cluster_ids = cluster_ids[valid_mask].astype(int)
    cluster_depths = cluster_depths[valid_mask].astype(float)
    
    # Create bins
    try:
        bins = np.quantile(cluster_depths, np.linspace(0.0, 1.0, n_bins + 1))
    except Exception:
        bins = np.linspace(np.min(cluster_depths), np.max(cluster_depths), n_bins + 1)
        
    # Determine matches (dict or array)
    map_dict = {}
    try:
        if hasattr(mapping, 'items'):
            for k, v in mapping.items(): map_dict[int(k)] = int(v)
        else:
            # Assume it aligns with we_left's unit_ids if they match sorting0
            # Ideally we have the original sorting object to align, but let's try direct map
            # if mapping provided is just array.
            # CAUTION: This requires knowing the order. 
            # Safer to rely on the caller passing a dict or aligned array.
            pass
    except Exception:
        pass
        
    # If map_dict is empty and mapping is array, try to build it
    if not map_dict and hasattr(mapping, '__len__'):
         # We need the unit_ids from we_left
         _, u_ids_left = _get_templates_and_unit_ids(we_left)
         if len(mapping) == len(u_ids_left):
             for u, m in zip(u_ids_left, mapping):
                 map_dict[int(u)] = int(m)

    df = pd.DataFrame({"unit_id": cluster_ids, "depth": cluster_depths})
    df['bin'] = pd.cut(df['depth'], bins=bins, include_lowest=True, labels=False)
    
    rng = np.random.default_rng(random_seed)
    
    # Prepare templates
    t_left, u_left = _get_templates_and_unit_ids(we_left)
    t_right, u_right = _get_templates_and_unit_ids(we_right)
    
    # Map unit_id -> index
    u_map_left = {int(u): i for i, u in enumerate(u_left)}
    u_map_right = {int(u): i for i, u in enumerate(u_right)}
    
    with PdfPages(pdf_path) as pdf:
        for b in range(n_bins):
            # Select units in this bin that HAVE MATCHES
            candidates = []
            sel = df['bin'] == b
            units_in_bin = df.loc[sel, 'unit_id'].values.astype(int)
            
            for u in units_in_bin:
                m = map_dict.get(u, -1)
                if m != -1 and u in u_map_left and m in u_map_right:
                    candidates.append((u, m))
            
            if not candidates:
                continue
                
            # Sample
            if len(candidates) > n_per_bin:
                chosen_indices = rng.choice(len(candidates), size=n_per_bin, replace=False)
                chosen = [candidates[i] for i in chosen_indices]
            else:
                chosen = candidates
                
            # Plot
            fig, axes = plt.subplots(len(chosen), 2, figsize=(12, 4 * len(chosen)), constrained_layout=True)
            if len(chosen) == 1:
                axes = np.array([axes])
                
            fig.suptitle(f"Depth Bin {b}: {bins[b]:.1f} - {bins[b+1]:.1f} um", fontsize=14)
            
            for i, (u1, u2) in enumerate(chosen):
                # Left
                idx1 = u_map_left[u1]
                wf1 = t_left[idx1] # (nsamples, nch)
                peak_ch1 = np.argmax(np.ptp(wf1, axis=0))
                
                # Right
                idx2 = u_map_right[u2]
                wf2 = t_right[idx2]
                peak_ch2 = np.argmax(np.ptp(wf2, axis=0))
                
                # Plot Left
                ax0 = axes[i, 0]
                ax0.plot(wf1, color='k', alpha=0.3)
                ax0.plot(wf1[:, peak_ch1], color='b', lw=2)
                ax0.set_title(f"Motion0 Unit {u1} (Peak Ch {peak_ch1})")
                
                # Plot Right
                ax1 = axes[i, 1]
                ax1.plot(wf2, color='k', alpha=0.3)
                ax1.plot(wf2[:, peak_ch2], color='r', lw=2)
                ax1.set_title(f"Motion1 Unit {u2} (Peak Ch {peak_ch2})")
                
            pdf.savefig(fig)
            plt.close(fig)
            
    print(f"[PDF] Saved matched templates PDF to {pdf_path}")
    return pdf_path

try:
    # if we1: plot_templates_by_depth(we1, "Sorting1 Spatial Templates (PlaceHolder)", out_dir / "templates_spatial_sorting1_placeholder.png")
    # if we2: plot_templates_by_depth(we2, "Sorting2 Spatial Templates (PlaceHolder)", out_dir / "templates_spatial_sorting2_placeholder.png")
    pass
except Exception: pass

try:
    if we1 and we2 and 'm1_to_2' in globals():
        # Use m1_to_2 for mapping. 
        # Check if we should plot for motion0 (we1) or motion1 (we2) depth bins. 
        # By default the function uses depths_label to find depth files.
        # We aligned depths_label='motion0' logic in the function default.
        pdf_f = plot_matched_units_pdf(
            out_dir, we1, we2, m1_to_2, 
            depths_label='motion0', 
            n_bins=5, n_per_bin=4, random_seed=42
        )
except Exception as e:
    print(f"Failed to generate matched PDF: {e}")

# %% [markdown]
# ## 7. Spatial Peak Clustering & Performance Report
# This section begins with a spatial peak diagnostic to verify motion correction,
# followed by a rigorous "Winner" determination by analyzing unit yield and SNR.

# %%
# Define Paths for the Report
report_out_dir = Path("/mnt/NPX/Luke/20250804/comparison_report_v2")
report_out_dir.mkdir(parents=True, exist_ok=True)

# Use existing sorting objects loaded at the top
# sorting0 = (already loaded motion0)
# sorting1 = (already loaded motion1)

# Load Recording for Metric Computation
# Priority 1: Use the preprocessed recording from the pipeline (ensures correct geometry/formatting)
recording_pre = None
rec_path_pre = pipe0 / "preprocessed_recording"

if rec_path_pre.exists():
    try:
        print(f"[Performance Report] Attempting to load preprocessed recording from {rec_path_pre}")
        # Try loading as a saved job/extractor
        recording_pre = si.load_extractor(rec_path_pre)
        print(f"[Performance Report] Loaded preprocessed recording: {recording_pre}")
    except Exception as e:
        print(f"[Performance Report] Failed to load preprocessed recording: {e}")
        recording_pre = None

# Priority 2: Fallback to Raw Data if Preprocessed is unavailable or failed
if recording_pre is None:
    # Adjust path if necessary
    # User provided: Raw folder "/mnt/NPX/Luke/20250804/Luke0804_V2V1_g0/Luke0730_V2V1_g0_imec1/"
    rec_path = Path("/mnt/NPX/Luke/20250804/Luke0804_V2V1_g0/Luke0730_V2V1_g0_imec1/Luke0730_V2V1_g0_t0.imec1.ap.bin")
    
    if not rec_path.exists():
        print(f"[Performance Report] Recording binary not found at {rec_path}")
        # Try the path variable from earlier in the script if defined
        if 'recordingfolder' in globals() and Path(globals()['recordingfolder']).exists():
            rec_path = Path(globals()['recordingfolder'])
            print(f"[Performance Report] Using alternative recording path: {rec_path}")

    try:
        if rec_path.exists():
            # Preprocessing: Must match the pipeline to compare apples-to-apples
            # Note: We assume 384 channels/30kHz/int16 as per SpikeGLX defaults
            recording_raw = se.read_binary(str(rec_path), sampling_frequency=30000.0, num_chan=384, dtype="int16")
            recording_pre = si.bandpass_filter(recording_raw, freq_min=300.0, freq_max=6000.0)
            recording_pre = si.common_reference(recording_pre, reference='global', operator='median')
            
            # Ensure probe information is attached for SortingAnalyzer
            recording_pre = ensure_recording_has_geometry(recording_pre)
            
            print("[Performance Report] Raw recording loaded and preprocessed.")
        else:
            recording_pre = None
            print("[Performance Report] Skipping raw-data metrics (recording not found).")
    except Exception as e:
        print(f"[Performance Report] Error loading recording: {e}")
        recording_pre = None
else:
    # Ensure loaded preprocessed recording has geometry
    recording_pre = ensure_recording_has_geometry(recording_pre)


# %% [markdown]
# ### 7a. Spatial Peak Clustering (Pre-Sorting)
# This diagnostic visualizes the localized peaks on the probe map. 
# Sharp horizontal bands indicate that motion correction successfully 
# stabilized the neural signals in space.

# %%
def plot_spatial_peaks(pipe_dir, recording, ax, title, n_peaks=20000):
    """Loads peaks/locations from pipeline cache and plots them on the probe.
    
    Prioritizes post-sorting spike positions (motion corrected) if available.
    Falls back to pre-sorting peak detections if not.
    Colors by unit ID if cluster info is available.
    """
    # 1. Try Post-Sorting (Motion Corrected) Spike Positions
    candidates = [
        pipe_dir / "cur" / "cur_sorter_output" / "spike_positions.npy",
        pipe_dir / "cur" / "cur_sorter_output" / "spike_position.npy",
    ]
    
    pos_file = None
    for c in candidates:
        if c.exists():
            pos_file = c
            break
            
    x, y, clusters = None, None, None
    
    if pos_file:
        try:
            # Usually (N, 2) array: [x, y]
            locs = np.load(pos_file)
            if locs.ndim == 2 and locs.shape[1] >= 2:
                x = locs[:, 0]
                y = locs[:, 1]
                # If title says Pre-Sorting, update it to reflect reality
                if "Pre-Sorting" in title:
                    title = title.replace("Pre-Sorting", "Post-Sorting")
                    
            # Try loading cluster IDs to color by unit
            clu_file = pos_file.parent / "spike_clusters.npy"
            if clu_file.exists():
                try:
                    clusters = np.load(clu_file)
                except Exception:
                    pass
        except Exception as e:
            print(f"Failed to load {pos_file}: {e}")

    # 2. Fallback to Pre-Sorting Peaks if no post-sorting spikes found
    if x is None:
        peaks_path = pipe_dir / "peaks.npy"
        locs_path = pipe_dir / "peak_locations.npy"
        
        if not peaks_path.exists():
            peaks_path = pipe_dir / "motion" / "peaks.npy"
            locs_path = pipe_dir / "motion" / "peak_locations.npy"
        
        if peaks_path.exists() and locs_path.exists():
            try:
                locs = np.load(locs_path)
                # Usually structured array with 'x', 'y' fields
                if isinstance(locs, np.ndarray) and 'x' in locs.dtype.names:
                    x = locs['x']
                    y = locs['y']
            except Exception as e:
                print(f"Failed to load peaks from {locs_path}: {e}")

    if x is not None and y is not None:
        try:
            # Plot the probe layout
            sw.plot_probe_map(recording, ax=ax, with_channel_ids=False)
            
            # Subsample for performance
            if len(x) > n_peaks:
                indices = np.random.choice(len(x), size=n_peaks, replace=False)
                x_disp = x[indices]
                y_disp = y[indices]
                c_disp = clusters[indices] if clusters is not None else None
            else:
                x_disp, y_disp = x, y
                c_disp = clusters
                
            # Scatter the peaks
            if c_disp is not None:
                # Color by cluster ID using a colormap
                # Map clusters to deterministic colors
                cmap = plt.get_cmap('tab20')
                # Use cluster ID modulo map size to cycle colors
                colors = c_disp % 20  
                ax.scatter(x_disp, y_disp, c=colors, cmap=cmap, alpha=0.5, s=64, rasterized=True)
            else:
                # Fallback to purple if no clusters
                ax.scatter(x_disp, y_disp, color='purple', alpha=0.01, s=64, rasterized=True)
            
            ax.set_title(title)
            ax.set_xlabel("x (μm)")
            ax.set_ylabel("y (μm)")
            
            # Auto-scale Y to show relevant depth range
            y_min, y_max = y.min(), y.max()
            if (y_max - y_min) > 200:
                ax.set_ylim(y_min - 20, y_max + 20)
            else:
                # Default narrow view if data is restricted
                ax.set_ylim(-100, 150) 
        except Exception as e:
            ax.text(0.5, 0.5, f"Error plotting peaks:\\n{e}", 
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"{title} (Error)")
    else:
        ax.text(0.5, 0.5, f"Peak data not found in:\\n{pipe_dir.name}", 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f"{title} (Missing Data)")
        
if recording_pre is not None:
    try:
        # Generate side-by-side comparison
        fig_peaks, axes = plt.subplots(1, 2, figsize=(16, 128), sharey=True)
        # Use recording_pre for the probe map
        plot_spatial_peaks(pipe0, recording_pre, axes[0], "Motion0: Pre-Sorting Peaks")
        plot_spatial_peaks(pipe1, recording_pre, axes[1], "Motion1: Pre-Sorting Peaks")

        plt.tight_layout()
        fig_peaks.savefig(out_dir / "spatial_peak_clustering_comparison.png", dpi=300)
        _show_or_close_figure(fig_peaks)
    except Exception as e:
        print(f"[Spatial Peaks] Failed to generate plot: {e}")

    # Generate Spatial Template Plots (side-by-side)
    try:
        if we1 or we2:
            fig, axes = plt.subplots(1, 2, figsize=(14, 24), sharey=True)
            if we1:
                plot_templates_by_depth(we1, "Sorting1 (Motion0)", recording_geometry=recording_pre, ax=axes[0])
            if we2:
                plot_templates_by_depth(we2, "Sorting2 (Motion1)", recording_geometry=recording_pre, ax=axes[1])
            
            plt.tight_layout()
            fig.savefig(out_dir / "templates_spatial_comparison.png", dpi=200)
            _show_or_close_figure(fig)
    except Exception as e:
        print(f"[Spatial Templates] Failed to generate plot: {e}")
else:
    print("[Spatial Peaks] Skipping plot (no recording loaded).")
#%%


# Metric Assessment Function
def analyze_sorting_performance(sorting, recording, label):
    if recording is None:
        return None, None
    print(f"Analyzing {label}...")
    try:
        # Create analyzer (sampling 1000 spikes for speed)
        # Using sparse=True and 'memory' format for a lightweight analysis
        analyzer = si.create_sorting_analyzer(sorting, recording, format="memory", sparse=True, overwrite=True)
        analyzer.compute("random_spikes", method="uniform", max_spikes_per_unit=1000, seed=42)
        analyzer.compute("templates", operators=["average"])
        analyzer.compute("noise_levels")
        
        # Compute the 3 most diagnostic metrics
        metrics = sqm.compute_quality_metrics(
            analyzer, 
            metric_names=['snr', 'isi_violation', 'presence_ratio']
        )
        return analyzer, metrics
    except Exception as e:
        print(f"Analysis failed for {label}: {e}")
        return None, None

def fraction_spikes_matched_globally(sorting_ref, sorting_other, delta_ms=0.4):
    """
    Computes the fraction of spikes in sorting_ref that have a matching spike 
    in sorting_other within delta_ms, regardless of unit identity.
    
    This is a global measure of 'event detection agreement'.
    """
    # Helper to concatenate all spike times
    def get_all_times(sorting):
        # List of arrays, one per unit
        all_trains = [sorting.get_unit_spike_train(u, return_times=True) for u in sorting.unit_ids]
        if not all_trains: return np.array([])
        # Flatten and sort
        return np.sort(np.concatenate(all_trains))

    times_ref = get_all_times(sorting_ref)
    times_oth = get_all_times(sorting_other)
    
    if times_ref.size == 0: return 0.0
    if times_oth.size == 0: return 0.0
    
    # Use searchsorted to find closest match for each spike in ref
    # Find insertion indices of ref times into sorted other times
    idx = np.searchsorted(times_oth, times_ref)
    
    # Clip indices to valid range
    idx_left = np.clip(idx - 1, 0, len(times_oth) - 1)
    idx_right = np.clip(idx, 0, len(times_oth) - 1)
    
    # Compute distances to neighbors
    dist_left = np.abs(times_ref - times_oth[idx_left])
    dist_right = np.abs(times_ref - times_oth[idx_right])
    
    # Minimal distance to any spike in other
    min_dist = np.minimum(dist_left, dist_right)
    
    # Count matches within window (delta_ms is in milliseconds, times are in seconds)
    delta_s = delta_ms / 1000.0
    n_matched = np.sum(min_dist <= delta_s)
    
    return n_matched / len(times_ref)

# Run Analysis if recording is available
if recording_pre is not None:
    # Use sorting0 (Motion0) and sorting1 (Motion1) directly.
    s_motion0 = sorting0
    s_motion1 = sorting1
    
    ana0_eval, qm0_eval = analyze_sorting_performance(s_motion0, recording_pre, "Motion0")
    ana1_eval, qm1_eval = analyze_sorting_performance(s_motion1, recording_pre, "Motion1")
else:
    qm0_eval = qm1_eval = None

# %%
# Compare and Report (if metrics computed successfully)
if qm0_eval is not None and qm1_eval is not None:
    # Re-retrieve matching (or use the one computed earlier: m1_to_2)
    # m1_to_2 maps sorting0 unit -> sorting1 unit
    
    # Build matched dataframe
    matched_list = []
    # robustly iterate valid matches
    try:
        # handle dict or array
        iter_matches = m1_to_2.items() if hasattr(m1_to_2, 'items') else zip(sorting0.unit_ids, m1_to_2)
    except Exception:
        # fallback if m1_to_2 missing
        iter_matches = []
        
    for u0, u1 in iter_matches:
        u0, u1 = int(u0), int(u1)
        if u1 != -1 and u0 in qm0_eval.index and u1 in qm1_eval.index:
            try:
                # Retrieve agreement score if available
                ag_score = agreement_scores.loc[u0, u1] if 'agreement_scores' in globals() else np.nan
            except Exception:
                ag_score = np.nan
                
            matched_list.append({
                'unit_0': u0, 'unit_1': u1,
                'snr_0': qm0_eval.loc[u0, 'snr'], 'snr_1': qm1_eval.loc[u1, 'snr'],
                'isi_0': qm0_eval.loc[u0, 'isi_violations_ratio'], 'isi_1': qm1_eval.loc[u1, 'isi_violations_ratio'],
                'agreement': ag_score
            })
    
    df_match = pd.DataFrame(matched_list)
    df_match.to_csv(report_out_dir / "matched_metrics.csv", index=False)

    # --- Visualizations ---
    
    # A) Yield Plot: Total vs Good
    def is_good_unit(df_metrics):
        # Relaxed Criteria: 
        # SNR > 0.05 (since median is ~0.12)
        # ISI Violations < 1.0 (very permissive)
        # Presence Ratio > 0.5 (must be present for half the recording)
        return (df_metrics['snr'] > 0.05) & (df_metrics['isi_violations_ratio'] < 1.0) & (df_metrics['presence_ratio'] > 0.5)
    
    # Print metric statistics to help tune thresholds
    print("\n--- Metric Distributions (to help tune thresholds) ---")
    print("Motion0 SNR Quartiles:", np.quantile(qm0_eval['snr'], [0.25, 0.5, 0.75]))
    print("Motion1 SNR Quartiles:", np.quantile(qm1_eval['snr'], [0.25, 0.5, 0.75]))
    print("Motion0 ISI Quartiles:", np.quantile(qm0_eval['isi_violations_ratio'], [0.25, 0.5, 0.75]))
    
    n_good_0 = qm0_eval[is_good_unit(qm0_eval)].shape[0]
    n_good_1 = qm1_eval[is_good_unit(qm1_eval)].shape[0]

    yield_data = pd.DataFrame({
        'Metric': ['Total Units', 'Good Units (QC)'],
        'Motion0': [len(sorting0.unit_ids), n_good_0],
        'Motion1': [len(sorting1.unit_ids), n_good_1]
    })
    
    fig_yield, ax = plt.subplots(figsize=(6, 5))
    yield_data.set_index('Metric').plot(kind='bar', ax=ax, rot=0, color=['C0', 'C1'])
    ax.set_title("Yield Comparison")
    ax.set_ylabel("Count")
    plt.tight_layout()
    fig_yield.savefig(report_out_dir / "yield_comparison.png", dpi=200)
    _show_or_close_figure(fig_yield)
    
    # B) SNR Improvement Plot (for matched units)
    if not df_match.empty:
        fig_snr, ax = plt.subplots(figsize=(6, 6))
        
        # Color by agreement if possible
        sc = ax.scatter(df_match['snr_0'], df_match['snr_1'], c=df_match['agreement'], 
                        cmap='viridis', alpha=0.7, edgecolors='none')
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label('Agreement Score')
        
        # Identity line
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
            np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
        ]
        ax.plot(lims, lims, 'r--', alpha=0.75, label='Identity')
        ax.set_xlabel("SNR (Motion0)")
        ax.set_ylabel("SNR (Motion1)")
        ax.set_title("SNR Comparison: Matched Units")
        ax.legend()
        plt.tight_layout()
        fig_snr.savefig(report_out_dir / "snr_scatter.png", dpi=200)
        _show_or_close_figure(fig_snr)

    # C) Agreement Matrix (Already generated above, but we can save copy here)
    if 'cmp' in globals():
        try:
            fig_cmp = plt.figure(figsize=(8, 8))
            sw.plot_agreement_matrix(cmp, count_text=False, unit_ticks=False)
            plt.title("Agreement: Motion0 vs Motion1")
            fig_cmp.savefig(report_out_dir / "agreement_matrix_report.png", dpi=200)
            _show_or_close_figure(fig_cmp)
        except Exception:
            pass

    # --- Winner Conclusion ---
    snr_diff = df_match['snr_1'].median() - df_match['snr_0'].median() if not df_match.empty else 0.0
    yield_diff = n_good_1 - n_good_0
    
    print("\n--- PERFORMANCE SUMMARY ---")
    print(f"Median SNR Change (Matched M1-M0, N={len(df_match)}): {snr_diff:+.2f}")
    print(f"High-Quality Unit Yield Change (M1-M0): {yield_diff:+} units")
    
    # Compute Global Spike Agreement (Independent of Unit ID)
    # How many spikes in M0 are found in M1? And vice versa.
    # This assesses 'detection robustness' regardless of clustering.
    try:
        frac_spikes_0_in_1 = fraction_spikes_matched_globally(sorting0, sorting1, delta_ms=0.4)
        frac_spikes_1_in_0 = fraction_spikes_matched_globally(sorting1, sorting0, delta_ms=0.4)
        print(f"Global Spike Agreement (M0 -> M1): {frac_spikes_0_in_1*100:.1f}% of M0 spikes found in M1")
        print(f"Global Spike Agreement (M1 -> M0): {frac_spikes_1_in_0*100:.1f}% of M1 spikes found in M0")
    except Exception as e:
        print(f"Global spike agreement failed: {e}")
        frac_spikes_0_in_1 = frac_spikes_1_in_0 = 0.0

    # Heuristic for decision:
    # 1. Yield is king: if one method finds significantly more 'good' units (>5%), it wins.
    # 2. If yields are similar, use Matched SNR as the tiebreaker.
    
    max_yield = max(n_good_0, n_good_1, 1)
    yield_pct_change = (n_good_1 - n_good_0) / max_yield
    
    if yield_pct_change > 0.05:
        winner = "Motion1"
        reason = f"Higher Yield (+{yield_diff} units)"
    elif yield_pct_change < -0.05:
        winner = "Motion0" 
        reason = f"Higher Yield (+{-yield_diff} units)"
    else:
        # Tiebreaker: SNR
        if snr_diff > 0.1:
            winner = "Motion1"
            reason = "Better SNR (Matched)"
        elif snr_diff < -0.1:
            winner = "Motion0"
            reason = "Better SNR (Matched)"
        else:
            winner = "Tie"
            reason = "Equivalent Performance"

    print(f"\nCONCLUSION: {winner} is the superior method ({reason}).")
    
    # Save final report to CSV
    report = {
        "winner": winner,
        "median_snr_0": float(qm0_eval['snr'].median()),
        "median_snr_1": float(qm1_eval['snr'].median()),
        "good_units_0": int(n_good_0),
        "good_units_1": int(n_good_1),
        "total_units_0": int(len(sorting0.unit_ids)),
        "total_units_1": int(len(sorting1.unit_ids)),
        "n_matched": int(len(df_match)),
        "frac_spikes_0_in_1": float(frac_spikes_0_in_1),
        "frac_spikes_1_in_0": float(frac_spikes_1_in_0)
    }
    pd.Series(report).to_csv(report_out_dir / "winner_report.csv")
    print(f"Report saved to {report_out_dir}")

else:
    print("[Performance Report] Could not complete report (metrics or recording missing).")


# %% [markdown]
# ## 8. Bandwidth Analysis: Spike Widths & Filter Cutoff
# Comparing template widths to see if the lower cutoff (3kHz) in Motion1 blurred spikes.
# We visualize the distribution to detect if sharp spikes were lost or widened.

# %%
def compute_template_widths_and_freqs(we, fs=30000.0):
    templates, unit_ids = _get_templates_and_unit_ids(we)
    if templates is None or len(unit_ids) == 0:
        return np.array([]), np.array([])
    
    widths_ms = []
    freqs = []
    
    # Iterate over units
    for i in range(len(unit_ids)):
        temp = templates[i]
        
        # 1. Identify Peak Channel (largest amplitude range)
        ptps = np.ptp(temp, axis=0)
        peak_ch = np.argmax(ptps)
        wf = temp[:, peak_ch]
        
        # 2. Find Trough and Peak indices
        # We assume standard spike shape: global min (trough) then local max (peak)
        # or just distance between global min and global max to be robust to inversion
        idx_min = np.argmin(wf)
        idx_max = np.argmax(wf)
        
        # Width in samples (absolute distance)
        # This represents the duration of the repolarization phase (fastest part)
        w_samples = abs(idx_max - idx_min)
        
        if w_samples < 1:
            continue
            
        w_ms = (w_samples / fs) * 1000.0
        widths_ms.append(w_ms)
        
        # 3. Equivalent Frequency (Half-Period Approximation)
        # If the spike trough-to-peak is a half-cycle of a sine wave:
        # T/2 = width_sec  =>  T = 2 * width_sec  =>  f = 1/T
        w_sec = w_samples / fs
        eq_freq = 1.0 / (2.0 * w_sec)
        freqs.append(eq_freq)
        
    return np.array(widths_ms), np.array(freqs)

# Run bandwidth analysis if templates are available
if we1 and we2:
    print("\n--- BANDWIDTH ANALYSIS ---")
    
    # Get Sampling Frequency from sorting objects if available, else default
    fs0 = sorting0.get_sampling_frequency() or 30000.0
    fs1 = sorting1.get_sampling_frequency() or 30000.0
    
    # Compute
    widths0, freqs0 = compute_template_widths_and_freqs(we1, fs=fs0)
    widths1, freqs1 = compute_template_widths_and_freqs(we2, fs=fs1)
    
    if len(widths0) > 0 and len(widths1) > 0:
        # Save raw data for detailed inspection (CSV)
        df_bw = pd.DataFrame({
            'width_ms': np.concatenate([widths0, widths1]),
            'eq_freq_hz': np.concatenate([freqs0, freqs1]),
            'sorting': ['Motion0'] * len(widths0) + ['Motion1'] * len(widths1)
        })
        df_bw.to_csv(out_dir / "spike_widths_frequencies.csv", index=False)
        print(f"Saved bandwidth data to {out_dir / 'spike_widths_frequencies.csv'}")

        # A) Histogram/KDE of Widths
        fig_bw, ax = plt.subplots(figsize=(10, 6))
        
        bins = np.linspace(0, 1.0, 50) # 0 to 1ms range
        
        sns.histplot(data=df_bw, x='width_ms', hue='sorting', bins=bins, kde=True, 
                     ax=ax, common_norm=False, element="step", alpha=0.3)
        
        ax.set_xlabel("Template Trough-to-Peak Width (ms)")
        ax.set_ylabel("Count")
        ax.set_title("Spike Width Distribution: Filter Effect (9kHz vs 3kHz)")
        ax.grid(True, alpha=0.2)
        
        # Calc medians and percentiles
        med0 = np.median(widths0)
        med1 = np.median(widths1)
        
        # Quantiles to check for bimodality/tails
        q0 = np.quantile(widths0, [0.05, 0.25, 0.5, 0.75, 0.95])
        q1 = np.quantile(widths1, [0.05, 0.25, 0.5, 0.75, 0.95])
        
        stats_text = (
            f"Motion0 (9kHz):\n  Median: {med0:.3f}ms\n  5-95%: {q0[0]:.3f}-{q0[4]:.3f}ms\n\n"
            f"Motion1 (3kHz):\n  Median: {med1:.3f}ms\n  5-95%: {q1[0]:.3f}-{q1[4]:.3f}ms"
        )
        ax.text(0.7, 0.6, stats_text, transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.9), fontsize=9)
        
        plt.tight_layout()
        fig_bw.savefig(out_dir / "spike_width_comparison.png", dpi=200)
        _show_or_close_figure(fig_bw)
        
        # B) Frequency Distribution Check
        # Plot the "Required Bandwidth" (Equivalent Frequency)
        fig_freq, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(data=df_bw, x='eq_freq_hz', hue='sorting', kde=True, ax=ax, log_scale=True, element="step", common_norm=False)
        ax.axvline(3000, color='r', linestyle='--', label='3kHz Cutoff (Motion1)')
        ax.axvline(9000, color='g', linestyle='--', label='9kHz Cutoff (Motion0)')
        ax.set_title("Estimated Signal Frequency Content (1/2*Width)")
        ax.set_xlabel("Equivalent Frequency (Hz)")
        ax.legend()
        plt.tight_layout()
        fig_freq.savefig(out_dir / "spike_frequency_content.png", dpi=200)
        _show_or_close_figure(fig_freq)

    else:
        print("[Bandwidth] Not enough templates found.")
else:
    print("[Bandwidth] Waveform extractors (we1, we2) not available, skipping.")