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
pipe0 = Path("/mnt/NPX/Luke/20250804/branchingtest1_pipeline_results_Luke0804_V2V1_g0_imec1")
pipe1 = Path("/mnt/NPX/Luke/20250804/branchingtest0_pipeline_results_Luke0804_V2V1_g0_imec1")

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

# Summary JSON
summary = {
    "pipe0": str(pipe0),
    "pipe1": str(pipe1),
    "delta_ms": delta_ms,
    "n_units_0": int(len(sorting0.unit_ids)),
    "n_units_1": int(len(sorting1.unit_ids)),
    "n_matched": int(n_matched),
}
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