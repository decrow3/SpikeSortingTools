#%%
# curation_comparison_sweep.py
#
# Curation framework for patched KS4 output. The central design principle is:
#
#   no_merge is the default post-patch baseline.
#   cosine and amp_bic act as independent proposal engines.
#   posthoc proposes additional merges but is gated by cross-strategy agreement.
#   The evidence table (curation_moves.csv) is the authoritative record of every
#   proposed move and the multi-criteria evidence for or against it.
#
# Why this design?
#   The KS4 claim-mask patch substantially reduces oversplitting, so aggressive
#   broad merging is no longer the right default.  Post-patch curation should be
#   conservative: only accept a merge when independent evidence sources agree.
#   Posthoc (feature-projection) tends to over-merge at wide depth windows because
#   spatially-proximal but anatomically-distinct units can have overlapping PCA
#   projections by chance.  Requiring at least one waveform-based method (cosine
#   or amp_bic) to also flag the pair prevents these spurious chain merges.
#
# Strategies
#   no_merge  — dup-spike removal + redundant unit removal only    (baseline)
#   cosine    — Wall template cosine similarity + CCG              (proposal engine)
#   amp_bic   — amplitude BIC (1- vs 2-Gaussian) + CCG            (proposal engine)
#   posthoc   — feature-projection + CCG, gated by cosine/amp_bic (gated proposer)
#
#   Execution order matters: cosine and amp_bic run first so their caches exist
#   when posthoc applies the cross-strategy gate.
#
# Outputs (all written to cur_comparison/)
#   curation_summary.csv     — per-strategy unit counts, efficiency, contamination
#   merge_counts.csv         — merge groups and absorbed units per strategy
#   curation_moves.csv       — evidence table: one row per proposed merge group,
#                              with depth spread, template cosine, BIC delta, CCG
#                              score, ISI violation rates, cross-strategy support,
#                              and rule-based accept/reject decision
#   split_candidates.csv     — all units ranked by split evidence (tF bimodality,
#                              amplitude bimodality, rate CV, waveform drift, ISI)
#   fig1_overview            — unit counts, efficiency, amplitude truncation CDFs
#   fig2_merge_activity      — merge groups, absorbed units, ref-unit recovery
#   fig3_pairwise_agreement  — coincident spike fraction heatmap between strategies
#   fig4_divergent_units.pdf — per-unit diagnostics for units matched inconsistently
#   fig5_distributions       — spike count and match quality boxplots
#   fig6_merge_diagnostics.pdf — pre/post waveforms + ACG/CCG for each merge group
#   fig7_evidence_table      — evidence metric scatter coloured by accept/reject
#   fig8_split_candidates    — split priority vs depth, bimodality, contamination
#
# Usage
#   # Point at a run_* directory containing kilosort4/sorter_output/:
#   CUR_PIPELINE_DIR=/path/to/run_default python testing/curation_comparison_sweep.py
#
#   # Or use sweep env vars to auto-select the most recent run:
#   CUR_SWEEP_DIR=/path/to/shallow_sweep_testing CUR_RUN_NAME=default python ...
#
#   # Force full recompute (required after code changes to curation logic):
#   CUR_RECALC=1 CUR_PIPELINE_DIR=... python ...
#
#   # Change the reference strategy for unit-matching analysis:
#   CUR_REF=cosine CUR_PIPELINE_DIR=... python ...

import os
import sys
from itertools import combinations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Make repo-local imports (pipeline/) work when running from testing/.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spikeinterface.extractors import read_kilosort
from spikeinterface.postprocessing.correlograms import correlogram_for_one_segment
from pipeline.sorting import KilosortResults
from pipeline.qc import truncation_qc, refractory_qc, contamination_rate_from_rvl
from pipeline.curation_postpatch import (
    run_cur, run_cur_cosine, run_cur_amp_bic, run_cur_no_merge,
)
from pipeline.curation_evidence import (
    build_evidence_table, contamination_rate_per_unit, DEFAULT_MERGE_RULES,
)
from pipeline.curation_split import score_split_candidates
from pipeline.curation_temporal_diag import build_temporal_diagnostics

# =============================================================================
# Style
# =============================================================================
plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 7, 'axes.labelsize': 8, 'axes.titlesize': 8,
    'axes.titlepad': 4, 'axes.labelpad': 3,
    'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'legend.fontsize': 7, 'legend.frameon': False,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': 0.7, 'lines.linewidth': 1.2,
    'figure.dpi': 150, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'pdf.fonttype': 42,
})

PALETTE = ['#285E61', '#2B6CB0', '#C05621', '#6B46C1']

# =============================================================================
# Configuration  — edit these
# =============================================================================

_DEFAULT_PIPELINE_DIR = "/mnt/NPX/Luke/20260316/dredge_pipeline_results_Luke03162026_V2V1_RH_g0_imec1"
PIPELINE_DIR = Path(os.environ.get('CUR_PIPELINE_DIR', _DEFAULT_PIPELINE_DIR).strip())

# Optional: point at a shallow sweep directory containing run_* subfolders.
# Example:
#   CUR_SWEEP_DIR=/path/to/shallow_sweep_testing CUR_RUN_NAME=default python testing/curation_comparison_sweep.py
_sweep_dir_env = os.environ.get('CUR_SWEEP_DIR', '').strip()
if _sweep_dir_env:
    sweep_dir = Path(_sweep_dir_env)
    run_name = os.environ.get('CUR_RUN_NAME', '').strip()

    if run_name:
        run_dir = sweep_dir / (run_name if run_name.startswith('run_') else f'run_{run_name}')
        PIPELINE_DIR = run_dir
    else:
        # Pick the most recently modified run_* that has a sorter_output.
        candidates = []
        for d in sweep_dir.iterdir():
            if not (d.is_dir() and d.name.startswith('run_')):
                continue
            if (d / 'kilosort4' / 'sorter_output').exists():
                candidates.append(d)
        if not candidates:
            raise FileNotFoundError(
                f"No run_* dirs with kilosort4/sorter_output under {sweep_dir}"
            )
        PIPELINE_DIR = max(candidates, key=lambda p: p.stat().st_mtime)

print(f"Using PIPELINE_DIR={PIPELINE_DIR}")

FS             = 30_000.0
RECALC         = os.environ.get('CUR_RECALC', '0').strip() == '1'
REF_STRATEGY   = os.environ.get('CUR_REF', 'no_merge').strip()
COINC_TOLERANCE = 0.5e-3    # s — spike-matching window
COINC_THRESH    = 0.30      # min fraction of ref spikes that must coincide to count as a match
MATCH_DEPTH_UM  = 200.0     # only compare units whose median depths are within this range
MPCT_THRESH     = 20.0      # amplitude truncation threshold for "well-detected"
PRESENCE_THRESH = 0.50
MERGE_DIAG_MAX_UNITS = 4
MERGE_DIAG_WINDOW_MS = 100
MERGE_DIAG_BIN_MS = 2

# Per-strategy parameters — tweak here to explore the parameter space
STRATEGY_KWARGS = {
    'no_merge': {},
    # Stricter than the original defaults to reduce catastrophic over-merging.
    'posthoc':  {'posthoc_score_thresh': 2, 'posthoc_ccg_thresh': 0.60,
                 'posthoc_min_spikes_seed': 750, 'posthoc_min_spikes_pair': 200},
    'cosine':   {'cosine_thresh': 0.90, 'ccg_thresh': 0.5,
                 'min_spikes_seed': 500, 'min_spikes_pair': 100},
    # More permissive stage-1 gate so candidates can reach the CCG check.
    'amp_bic':  {'bic_margin': 100.0, 'ccg_thresh': 0.45,
                 'min_spikes_seed': 100, 'min_spikes_pair': 100},
}

STRATEGY_FNS = {
    'no_merge': run_cur_no_merge,
    'cosine':   run_cur_cosine,   # run before posthoc so gate caches exist
    'amp_bic':  run_cur_amp_bic,
    'posthoc':  run_cur,          # uses cross_strategy_gate against cosine/amp_bic
}

STRATEGY_OUT_SUBDIRS = {
    'no_merge': 'cur_no_merge_output',
    'posthoc':  'cur_sorter_output',
    'cosine':   'cur_cosine_output',
    'amp_bic':  'cur_amp_bic_output',
}

STRATEGY_COLORS = {name: PALETTE[i] for i, name in enumerate(STRATEGY_FNS)}

# =============================================================================
# Paths
# =============================================================================

ks4_out      = PIPELINE_DIR / 'kilosort4' / 'sorter_output'
cur_cache    = PIPELINE_DIR / 'cur'
comp_dir     = PIPELINE_DIR / 'cur_comparison'
comp_dir.mkdir(parents=True, exist_ok=True)

assert ks4_out.exists(), f"KS4 output not found: {ks4_out}"

# =============================================================================
# Step 1: Load KS4 base data
# =============================================================================

print("Loading KS4 base results...")
ks4_results = KilosortResults(ks4_out)
ks4_sorter  = read_kilosort(ks4_out)

# =============================================================================
# Step 2: Run all curation strategies (cached — rerun with RECALC=1)
# =============================================================================

print("\n--- Running curation strategies ---")
curation_results = {}   # strategy name → KilosortResults
for name, fn in STRATEGY_FNS.items():
    print(f"\n[{name}]")
    try:
        res = fn(
            ks4_sorter, ks4_results, cur_cache,
            recalc=RECALC, fs=int(FS),
            **STRATEGY_KWARGS[name],
        )
        curation_results[name] = res
        print(f"  → {res.directory}")
    except Exception as e:
        print(f"  ERROR: {e}")

# =============================================================================
# Step 3: Run QC on each strategy output (cached)
# =============================================================================

print("\n--- Running QC ---")
qc_results    = {}   # strategy name → (trunc_qc, pres_qc) or None
refrac_results= {}   # strategy name → refractory_qc result or None
contam_results= {}   # strategy name → {unit_id: contamination_rate}
for name, res in curation_results.items():
    st_s = res.spike_times.astype(float) / FS
    sc   = res.spike_clusters

    trunc_dir  = PIPELINE_DIR / f'qc_{name}' / 'amp_truncation'
    refrac_dir = PIPELINE_DIR / f'qc_{name}' / 'refractory'
    trunc_dir.mkdir(parents=True, exist_ok=True)
    refrac_dir.mkdir(parents=True, exist_ok=True)

    try:
        trunc, pres = truncation_qc(
            st_s, sc, res.spike_amplitudes,
            cache_dir=trunc_dir, recalc=RECALC,
        )
        qc_results[name] = (trunc, pres)
        print(f"  [{name}] truncation QC done")
    except Exception as e:
        print(f"  [{name}] truncation QC failed: {e}")
        qc_results[name] = None

    try:
        ref_qc = refractory_qc(st_s, sc, cache_dir=refrac_dir, recalc=RECALC)
        refrac_results[name] = ref_qc
        # Also compute simple per-unit ISI violation rate
        contam_results[name] = contamination_rate_per_unit(st_s, sc)
        print(f"  [{name}] refractory QC done")
    except Exception as e:
        print(f"  [{name}] refractory QC failed: {e}")
        refrac_results[name] = None
        contam_results[name] = {}

# =============================================================================
# Step 4: Load comparison data (spike arrays + QC)
# =============================================================================

def load_strategy_data(name, res):
    """Collect everything needed for unit-level comparison."""
    qc = qc_results.get(name)
    trunc, pres = qc if qc is not None else (None, None)
    sc  = res.spike_clusters
    spz = res.spike_positions[:, 1]   # depth (µm) per spike
    unit_depth = {
        int(u): float(np.median(spz[sc == u]))
        for u in np.unique(sc)
    }
    return dict(
        spike_times    = res.spike_times,
        spike_clusters = sc,
        spike_amps     = res.spike_amplitudes,
        spike_z        = spz,
        unit_depth     = unit_depth,
        trunc          = trunc,
        pres           = pres,
    )


all_data = {}
for name, res in curation_results.items():
    all_data[name] = load_strategy_data(name, res)
    n_units = len(np.unique(res.spike_clusters))
    print(f"  [{name}]  {n_units} units")

strategy_names = list(all_data.keys())

# =============================================================================
# Unit-level statistics
# =============================================================================

def unit_stats(data, results_dir, contam_map=None, fs=FS):
    st_s  = data['spike_times'].astype(float) / fs
    sc    = data['spike_clusters']
    trunc = data['trunc']
    pres  = data['pres']
    rec_dur = float(st_s.max())

    trunc_cid   = trunc['cid'].astype(int) if trunc is not None else np.array([], int)
    trunc_mpcts = trunc['mpcts']           if trunc is not None else np.array([])
    pres_cid    = pres['cid'].astype(int)  if pres  is not None else np.array([], int)
    pres_vblk   = pres['valid_blocks']     if pres  is not None else np.zeros((0, 2), int)

    # Load KSLabel if available (post-curation sorter_output has cluster_KSLabel.tsv)
    lbl_file = Path(results_dir) / 'cluster_KSLabel.tsv'
    if lbl_file.exists():
        ldf      = pd.read_csv(lbl_file, sep='\t')
        ldf.columns = [c.strip() for c in ldf.columns]
        label_map = dict(zip(ldf.iloc[:, 0].astype(int), ldf.iloc[:, 1]))
    else:
        label_map = {}

    rows = []
    for uid in np.unique(sc):
        u_st = st_s[sc == uid]
        tm   = (trunc_cid == uid)
        mean_mpct = float(np.nanmean(trunc_mpcts[tm])) if tm.any() else np.nan

        pm = (pres_cid == uid)
        if pm.any():
            vb = pres_vblk[pm]
            pres_s = sum(
                u_st[min(int(i1), len(u_st)-1)] - u_st[min(int(i0), len(u_st)-1)]
                for i0, i1 in vb
            )
            presence_frac = min(pres_s / rec_dur, 1.0)
        else:
            presence_frac = 0.0

        contam = contam_map.get(int(uid), np.nan) if contam_map else np.nan
        rows.append(dict(
            unit_id=int(uid),
            label=label_map.get(int(uid), 'unknown'),
            n_spikes=int((sc == uid).sum()),
            firing_rate_hz=float((sc == uid).sum()) / rec_dur,
            mean_mpct=mean_mpct,
            presence_frac=presence_frac,
            contamination_rate=contam,
        ))
    return pd.DataFrame(rows)


print("\nComputing per-unit statistics...")
all_stats = {}
for name in strategy_names:
    all_stats[name] = unit_stats(
        all_data[name], curation_results[name].directory,
        contam_map=contam_results.get(name, {}),
    )


def run_summary(stats_df):
    n_units = len(stats_df)
    n_good  = int((stats_df['label'] == 'good').sum())
    well    = (stats_df['mean_mpct'] < MPCT_THRESH) & (stats_df['presence_frac'] > PRESENCE_THRESH)
    clean   = well & (stats_df['contamination_rate'].fillna(1.0) < 0.05)
    spk     = stats_df['n_spikes'].to_numpy(float)
    q25, q50, q75 = (np.nanpercentile(spk, [25, 50, 75]) if len(spk) else (np.nan,)*3)
    return dict(
        n_units=n_units, n_good=n_good, n_well=int(well.sum()),
        n_clean=int(clean.sum()),
        efficiency=round(well.sum()/n_units, 3) if n_units else 0,
        median_mpct=float(np.nanmedian(stats_df['mean_mpct'])),
        med_presence=float(np.nanmedian(stats_df['presence_frac'])),
        median_contamination=float(np.nanmedian(stats_df['contamination_rate'])),
        total_spikes=int(spk.sum()),
        spikes_q25=float(q25), spikes_median=float(q50), spikes_q75=float(q75),
    )

summary_rows = []
for name in strategy_names:
    s = run_summary(all_stats[name])
    summary_rows.append(dict(strategy=name, **s))
summary_df = pd.DataFrame(summary_rows).set_index('strategy')
summary_df.to_csv(comp_dir / 'curation_summary.csv')
print("\n--- Summary ---")
print(summary_df[['n_units', 'n_good', 'n_well', 'efficiency', 'median_mpct']].to_string())

# =============================================================================
# Coincident spike matching
# =============================================================================

def find_coincident_spikes(ta, tb, tol=COINC_TOLERANCE):
    if not len(ta) or not len(tb):
        return 0
    ta, tb  = np.sort(ta), np.sort(tb)
    idx     = np.searchsorted(tb, ta)
    il      = np.clip(idx-1, 0, len(tb)-1)
    ir      = np.clip(idx,   0, len(tb)-1)
    min_d   = np.minimum(np.abs(ta - tb[il]), np.abs(ta - tb[ir]))
    return int((min_d <= tol).sum())


def build_matches(ref_data, other_data, fs=FS, min_spikes=100, max_depth_um=MATCH_DEPTH_UM):
    """ref unit → (best_other_uid, coinc_frac) or (None, 0).

    Only considers other-strategy units whose median depth is within
    max_depth_um of the reference unit, avoiding spurious cross-depth matches.
    """
    ref_st    = ref_data['spike_times'].astype(float) / fs
    ref_clu   = ref_data['spike_clusters']
    ref_depth = ref_data.get('unit_depth', {})
    oth_st    = other_data['spike_times'].astype(float) / fs
    oth_clu   = other_data['spike_clusters']
    oth_depth = other_data.get('unit_depth', {})

    oth_by_unit = {u: np.sort(oth_st[oth_clu == u]) for u in np.unique(oth_clu)}

    matches = {}
    for ref_uid in np.unique(ref_clu):
        rt = np.sort(ref_st[ref_clu == ref_uid])
        if len(rt) < min_spikes:
            matches[ref_uid] = (None, 0.0)
            continue
        ref_z = ref_depth.get(int(ref_uid))
        best_uid, best_frac = None, 0.0
        for oth_uid, ot in oth_by_unit.items():
            if len(ot) < min_spikes:
                continue
            if ref_z is not None and max_depth_um is not None:
                oth_z = oth_depth.get(int(oth_uid))
                if oth_z is not None and abs(ref_z - oth_z) > max_depth_um:
                    continue
            frac = find_coincident_spikes(rt, ot) / len(rt)
            if frac > best_frac:
                best_frac, best_uid = frac, oth_uid
        matches[ref_uid] = (best_uid, best_frac) if best_frac >= COINC_THRESH else (None, 0.0)
    return matches


def _resolve_results_dir(directory):
    directory = Path(directory)
    sorter_output = directory / 'sorter_output'
    return sorter_output if sorter_output.exists() else directory


def _load_templates(directory):
    """Load templates.npy (n_units, n_tp, n_chan) and templates_ind.npy (n_units, n_chan)."""
    d = _resolve_results_dir(directory)
    t_file  = d / 'templates.npy'
    ti_file = d / 'templates_ind.npy'
    if not t_file.exists():
        return None, None
    templates = np.load(t_file)          # (n_units, n_tp, n_chan)
    templates_ind = np.load(ti_file) if ti_file.exists() else None
    return templates, templates_ind


def _peak_trace(templates, templates_ind, uid):
    """Return (trace, peak_local_chan_idx) for unit uid using full templates."""
    if templates is None or uid < 0 or uid >= templates.shape[0]:
        return None, None
    w = templates[uid]                   # (n_tp, n_chan)
    rms = np.sqrt(np.mean(w ** 2, axis=0))
    peak_local = int(np.argmax(rms))
    return w[:, peak_local], peak_local


def _corr_trace(st_a, st_b=None, fs=FS, window_ms=MERGE_DIAG_WINDOW_MS, bin_ms=MERGE_DIAG_BIN_MS):
    st_a = np.asarray(st_a, dtype=np.int64)
    if st_b is None:
        st_b = st_a
    else:
        st_b = np.asarray(st_b, dtype=np.int64)

    if len(st_a) < 2 or len(st_b) < 2:
        return None, None

    bin_samp = max(1, int(round(bin_ms * fs / 1000)))
    window_samp = max(1, int(round(window_ms * fs / 1000)))

    spike_times = np.concatenate([st_a, st_b])
    spike_unit_indices = np.concatenate([
        np.zeros(st_a.size, dtype=np.int64),
        np.ones(st_b.size, dtype=np.int64),
    ])
    order = np.argsort(spike_times, kind='mergesort')
    spike_times = spike_times[order]
    spike_unit_indices = spike_unit_indices[order]

    corr = correlogram_for_one_segment(
        spike_times=spike_times,
        spike_unit_indices=spike_unit_indices,
        window_size=window_samp,
        bin_size=bin_samp,
    )
    center = corr.shape[2] // 2
    lags_ms = (np.arange(corr.shape[2]) - center) * (bin_samp / fs * 1000.0)

    if np.array_equal(st_a, st_b):
        trace = corr[0, 0, :].astype(float)
        trace[center] = 0
    else:
        trace = 0.5 * (corr[0, 1, :].astype(float) + corr[1, 0, ::-1].astype(float))
    return lags_ms, trace


def _best_matching_unit(union_samples, strategy_data, fs=FS):
    union_s = np.sort(np.asarray(union_samples, dtype=float) / fs)
    sc = strategy_data['spike_clusters']
    st_s = strategy_data['spike_times'].astype(float) / fs
    best_uid, best_frac = None, 0.0
    for uid in np.unique(sc):
        u_st = np.sort(st_s[sc == uid])
        if len(u_st) == 0:
            continue
        frac = find_coincident_spikes(union_s, u_st) / max(len(union_s), 1)
        if frac > best_frac:
            best_uid, best_frac = int(uid), float(frac)
    return best_uid, best_frac


print(f"\nMatching all strategies to reference '{REF_STRATEGY}'...")
if REF_STRATEGY not in all_data:
    raise RuntimeError(f"Reference strategy '{REF_STRATEGY}' not in {list(all_data.keys())}")

match_table = {}   # other_strategy → {ref_uid: (matched_uid, frac)}
for name in strategy_names:
    if name == REF_STRATEGY:
        continue
    print(f"  matching {name}...")
    match_table[name] = build_matches(all_data[REF_STRATEGY], all_data[name])

# Fraction of ref units with a match in each other strategy
ref_uids = np.unique(all_data[REF_STRATEGY]['spike_clusters'])
n_ref    = len(ref_uids)
print(f"\n  Reference units: {n_ref}")
for name, m in match_table.items():
    n_matched = sum(1 for _, (uid, _) in m.items() if uid is not None)
    print(f"  {name}: {n_matched}/{n_ref} ref units matched ({100*n_matched/max(n_ref,1):.1f}%)")

# =============================================================================
# Load merge group counts from cached .npy files
# =============================================================================

CACHE_FILES = {
    'no_merge': 'cur_todo_no_merge.npy',
    'posthoc':  'cur_todo_phy.npy',
    'cosine':   'cur_todo_cosine.npy',
    'amp_bic':  'cur_todo_amp_bic.npy',
}

merge_counts = {}
merge_todos = {}
for name in strategy_names:
    fname = CACHE_FILES[name]
    p = cur_cache / fname
    if p.exists():
        todo = np.load(p, allow_pickle=True).item()
        merge_todos[name] = todo
        mg   = todo.get('merge_unit_groups', [])
        n_merged_units = sum(len(g) for g in mg)
        merge_counts[name] = dict(
            n_groups=len(mg),
            n_merged_units=n_merged_units,
            n_removed=len(todo.get('removed_units', [])),
            n_duped=len(todo.get('duped_spikes', [])),
        )
    else:
        merge_todos[name] = {}
        merge_counts[name] = dict(n_groups=0, n_merged_units=0, n_removed=0, n_duped=0)

merge_df = pd.DataFrame(merge_counts).T
merge_df.index.name = 'strategy'
merge_df.to_csv(comp_dir / 'merge_counts.csv')
print("\n--- Merge counts ---")
print(merge_df.to_string())


# =============================================================================
# FIGURE 1 — Strategy overview (bar charts + CDFs)
# =============================================================================

fig1 = plt.figure(figsize=(7.2, 5.0))
gs1  = gridspec.GridSpec(2, 4, figure=fig1, hspace=0.60, wspace=0.45)
ax_nu  = fig1.add_subplot(gs1[0, 0])
ax_ng  = fig1.add_subplot(gs1[0, 1])
ax_nw  = fig1.add_subplot(gs1[0, 2])
ax_ef  = fig1.add_subplot(gs1[0, 3])
ax_cm  = fig1.add_subplot(gs1[1, 0:2])
ax_sp  = fig1.add_subplot(gs1[1, 2:4])

x       = np.arange(len(strategy_names))
bcolors = [STRATEGY_COLORS[n] for n in strategy_names]

def _bar(ax, values, ylabel, title, ylim=None):
    bars = ax.bar(x, values, color=bcolors, width=0.6, edgecolor='none')
    ref_idx = strategy_names.index(REF_STRATEGY) if REF_STRATEGY in strategy_names else None
    if ref_idx is not None:
        bars[ref_idx].set_edgecolor('k')
        bars[ref_idx].set_linewidth(1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(strategy_names, rotation=35, ha='right', fontsize=6)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim:
        ax.set_ylim(*ylim)
    ax.yaxis.set_major_locator(MaxNLocator(4, integer=True))

_bar(ax_nu, [summary_df.loc[n, 'n_units']    for n in strategy_names], 'count',    'Total units')
_bar(ax_ng, [summary_df.loc[n, 'n_good']     for n in strategy_names], 'count',    'Good units')
_bar(ax_nw, [summary_df.loc[n, 'n_well']     for n in strategy_names], 'count',    'Well-detected')
_bar(ax_ef, [summary_df.loc[n, 'efficiency'] for n in strategy_names], 'fraction', 'Efficiency', ylim=(0,1))

# CDF of mean missing %
for name, stats in all_stats.items():
    v   = np.sort(stats['mean_mpct'].dropna().values)
    cdf = np.arange(1, len(v)+1) / len(v)
    ax_cm.plot(v, cdf, color=STRATEGY_COLORS[name], lw=1.5, label=name,
               alpha=0.9, zorder=3 if name == REF_STRATEGY else 1)
ax_cm.axvline(MPCT_THRESH, color='#555', lw=0.8, ls='--')
ax_cm.set_xlabel('Mean missing % per unit')
ax_cm.set_ylabel('CDF')
ax_cm.set_title('Amplitude truncation distribution')
ax_cm.legend(ncol=2, loc='lower right')
ax_cm.set_xlim(0, 60)

# CDF of spikes per unit (log)
for name, stats in all_stats.items():
    v   = np.sort(stats['n_spikes'].values.astype(float))
    cdf = np.arange(1, len(v)+1) / len(v)
    ax_sp.semilogx(v, cdf, color=STRATEGY_COLORS[name], lw=1.5, label=name,
                   alpha=0.9, zorder=3 if name == REF_STRATEGY else 1)
ax_sp.set_xlabel('Spikes per unit')
ax_sp.set_ylabel('CDF')
ax_sp.set_title('Spike count distribution per unit')
ax_sp.legend(ncol=2, loc='lower right')

fig1.suptitle(f'Curation strategy comparison — {PIPELINE_DIR.name}', fontsize=9, y=1.01)
fig1.savefig(comp_dir / 'fig1_overview.pdf')
fig1.savefig(comp_dir / 'fig1_overview.png')
plt.close(fig1)
print("\nSaved Fig 1")


# =============================================================================
# FIGURE 2 — Merge activity + unit matching
# =============================================================================

fig2, axes2 = plt.subplots(1, 3, figsize=(7.2, 3.2))
plt.subplots_adjust(wspace=0.45)

ax_mg  = axes2[0]
ax_rm  = axes2[1]
ax_mt  = axes2[2]

# Merge groups and merged units per strategy
mg_vals = [merge_counts[n]['n_groups']       for n in strategy_names]
rm_vals = [merge_counts[n]['n_merged_units'] for n in strategy_names]

ax_mg.bar(x, mg_vals, color=bcolors, width=0.6, edgecolor='none')
ax_mg.set_xticks(x)
ax_mg.set_xticklabels(strategy_names, rotation=35, ha='right', fontsize=6)
ax_mg.set_ylabel('Count')
ax_mg.set_title('Merge groups created')
ax_mg.yaxis.set_major_locator(MaxNLocator(4, integer=True))

ax_rm.bar(x, rm_vals, color=bcolors, width=0.6, edgecolor='none')
ax_rm.set_xticks(x)
ax_rm.set_xticklabels(strategy_names, rotation=35, ha='right', fontsize=6)
ax_rm.set_ylabel('Count')
ax_rm.set_title('Units absorbed in merges')
ax_rm.yaxis.set_major_locator(MaxNLocator(4, integer=True))

# Fraction of reference units matched in each strategy
match_fracs = []
for name in strategy_names:
    if name == REF_STRATEGY:
        match_fracs.append(1.0)   # ref matches itself perfectly
        continue
    m = match_table.get(name, {})
    if not m:
        match_fracs.append(np.nan)
        continue
    n_matched = sum(1 for _, (uid, _) in m.items() if uid is not None)
    match_fracs.append(n_matched / max(n_ref, 1))

bars = ax_mt.bar(x, match_fracs, color=bcolors, width=0.6, edgecolor='none')
ref_idx = strategy_names.index(REF_STRATEGY) if REF_STRATEGY in strategy_names else None
if ref_idx is not None:
    bars[ref_idx].set_edgecolor('k')
    bars[ref_idx].set_linewidth(1.0)
ax_mt.set_xticks(x)
ax_mt.set_xticklabels(strategy_names, rotation=35, ha='right', fontsize=6)
ax_mt.set_ylabel('Fraction')
ax_mt.set_title(f'Ref-unit recovery\n(vs {REF_STRATEGY})')
ax_mt.set_ylim(0, 1.1)
ax_mt.axhline(1.0, color='#aaa', lw=0.7, ls=':')

fig2.suptitle('Merge activity and unit recovery', fontsize=9, y=1.02)
fig2.savefig(comp_dir / 'fig2_merge_activity.pdf')
fig2.savefig(comp_dir / 'fig2_merge_activity.png')
plt.close(fig2)
print("Saved Fig 2")


# =============================================================================
# FIGURE 3 — Coincident fraction heatmap (pairwise between all strategies)
# =============================================================================

# Build an N_strategies × N_strategies matrix of median coincident fractions
# between matched ref units.  Diagonal = 1 by definition.
strat_pairs = [(a, b) for a in strategy_names for b in strategy_names if a != b]

# Compute pairwise match fractions (median coinc_frac for matched units)
pairwise_median_frac = pd.DataFrame(np.nan, index=strategy_names, columns=strategy_names)
pairwise_n_matched   = pd.DataFrame(0,      index=strategy_names, columns=strategy_names)

for name in strategy_names:
    pairwise_median_frac.loc[name, name] = 1.0
    pairwise_n_matched.loc[name, name]   = len(np.unique(all_data[name]['spike_clusters']))

for name, m in match_table.items():
    fracs = [frac for _, (uid, frac) in m.items() if uid is not None]
    n_m   = len(fracs)
    med   = float(np.median(fracs)) if fracs else np.nan
    pairwise_median_frac.loc[REF_STRATEGY, name] = med
    pairwise_median_frac.loc[name, REF_STRATEGY] = med   # symmetric approximation
    pairwise_n_matched.loc[REF_STRATEGY, name]   = n_m
    pairwise_n_matched.loc[name, REF_STRATEGY]   = n_m

fig3, (ax_frac, ax_nm) = plt.subplots(1, 2, figsize=(6.0, 2.8))
plt.subplots_adjust(wspace=0.4)

for ax, mat, title, fmt in [
    (ax_frac, pairwise_median_frac, 'Median coincident fraction\n(matched units)', '.2f'),
    (ax_nm,   pairwise_n_matched,   f'Units matched\n(vs {REF_STRATEGY})', 'd'),
]:
    arr = mat.to_numpy(float)
    im  = ax.imshow(arr, vmin=0, vmax=float(arr[np.isfinite(arr)].max()) if np.isfinite(arr).any() else 1,
                    cmap='Blues', aspect='auto')
    ax.set_xticks(np.arange(len(strategy_names)))
    ax.set_yticks(np.arange(len(strategy_names)))
    ax.set_xticklabels(strategy_names, rotation=40, ha='right', fontsize=6)
    ax.set_yticklabels(strategy_names, fontsize=6)
    for i in range(len(strategy_names)):
        for j in range(len(strategy_names)):
            v = arr[i, j]
            if not np.isfinite(v):
                txt = '—'
            elif fmt == 'd':
                txt = f'{int(v):d}'
            else:
                txt = f'{v:{fmt}}'
            ax.text(
                j, i, txt,
                ha='center', va='center', fontsize=6,
                color='white' if v > 0.6 * float(arr[np.isfinite(arr)].max()) else 'black'
            )
    ax.set_title(title, fontsize=8)
    fig3.colorbar(im, ax=ax, shrink=0.85)

fig3.suptitle('Pairwise strategy agreement', fontsize=9, y=1.03)
fig3.savefig(comp_dir / 'fig3_pairwise_agreement.pdf')
fig3.savefig(comp_dir / 'fig3_pairwise_agreement.png')
plt.close(fig3)
print("Saved Fig 3")


# =============================================================================
# FIGURE 4 — Per-unit PDF: units that differ across strategies
# =============================================================================
# Show units from the reference strategy that are matched in some strategies
# but NOT in others (i.e. the merge changed their detectability).

print("\nBuilding per-unit PDF for divergent units...")

# A "divergent" ref unit is one that is matched in at least one non-ref strategy
# but not all of them (suggesting curation affects it differently).
other_strategies = [n for n in strategy_names if n != REF_STRATEGY]

divergent_units = []
ref_st_s  = all_data[REF_STRATEGY]['spike_times'].astype(float) / FS
ref_clu   = all_data[REF_STRATEGY]['spike_clusters']
rec_dur_s = float(ref_st_s.max())

for ref_uid in np.unique(ref_clu):
    matched_in = []
    for name in other_strategies:
        m = match_table.get(name, {})
        uid, frac = m.get(ref_uid, (None, 0.0))
        if uid is not None:
            matched_in.append(name)
    # Keep if at least one strategy matches AND at least one doesn't
    if 0 < len(matched_in) < len(other_strategies):
        divergent_units.append(ref_uid)

# Sort by spike count descending
divergent_units.sort(
    key=lambda u: -int((ref_clu == u).sum())
)
print(f"  {len(divergent_units)} divergent units → writing per-unit PDF")

with PdfPages(comp_dir / 'fig4_divergent_units.pdf') as pdf:
    if not divergent_units:
        fig, ax = plt.subplots(figsize=(6.0, 2.2))
        ax.axis('off')
        ax.text(
            0.5, 0.5,
            'No divergent units for this run.\nAll non-reference strategies either matched or missed the same reference units.',
            ha='center', va='center', fontsize=8,
        )
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    for ref_uid in tqdm(divergent_units[:200], desc='Per-unit PDF'):
        # Which strategies match this unit?
        display_rows = [(REF_STRATEGY, ref_uid, 1.0)]
        for name in other_strategies:
            m = match_table.get(name, {})
            uid, frac = m.get(ref_uid, (None, 0.0))
            display_rows.append((name, uid, frac))

        n_rows = len(display_rows)
        fig, axes = plt.subplots(n_rows, 2, figsize=(7.2, max(3.5, n_rows * 2.0)),
                                 squeeze=False,
                                 gridspec_kw=dict(hspace=0.18, wspace=0.35))

        ref_n_spk = int((ref_clu == ref_uid).sum())
        fig.suptitle(
            f'Ref unit {ref_uid}  —  {ref_n_spk:,} spikes\n'
            f'matched in: {[n for n in other_strategies if match_table.get(n, {}).get(ref_uid, (None,))[0] is not None]}',
            fontsize=8, y=1.0
        )

        for row_i, (sname, uid, coinc_frac) in enumerate(display_rows):
            d       = all_data[sname]
            st_s    = d['spike_times'].astype(float) / FS
            sc      = d['spike_clusters']
            amps    = d['spike_amps']
            trunc   = d['trunc']
            col     = STRATEGY_COLORS[sname]
            ax0     = axes[row_i, 0]
            ax1     = axes[row_i, 1]

            if uid is None:
                ax0.text(0.5, 0.5, f'{sname}\n(no match)',
                         transform=ax0.transAxes, ha='center', va='center',
                         fontsize=7, color='#999')
                ax1.set_visible(False)
                continue

            u_st   = st_s[sc == uid]
            u_amps = amps[sc == uid] if amps is not None else None
            coinc_str = '' if sname == REF_STRATEGY else f'  coinc={coinc_frac*100:.0f}%'
            row_title = f'{sname}{coinc_str}  unit {uid}  n={len(u_st):,}'

            # Amplitude raster
            if u_amps is not None:
                ax0.hist2d(u_st / 60, u_amps,
                           bins=[min(150, max(30, len(u_st)//50)), 40],
                           cmap='Blues', rasterized=True)
            else:
                ax0.plot(u_st / 60, np.ones_like(u_st), '|',
                         color=col, ms=1, alpha=0.3, rasterized=True)
            ax0.set_xlim(0, rec_dur_s / 60)
            ax0.set_title(row_title, fontsize=7, pad=2, color=col)
            ax0.set_ylabel('Amplitude', fontsize=6)
            if row_i < n_rows - 1:
                ax0.tick_params(labelbottom=False)
            else:
                ax0.set_xlabel('Time (min)', fontsize=7)

            # Missing % trace
            if trunc is not None:
                tcid  = trunc['cid'].astype(int)
                tblk  = trunc['window_blocks']
                tmpct = trunc['mpcts']
                tm    = (tcid == uid)
                if tm.any():
                    u_st_for_t = st_s[sc == uid]
                    centres, mpcts = [], []
                    for (i0, i1), mp in zip(tblk[tm], tmpct[tm]):
                        i0c = min(int(i0), len(u_st_for_t)-1)
                        i1c = min(int(i1), len(u_st_for_t)-1)
                        centres.append((u_st_for_t[i0c] + u_st_for_t[i1c]) / 2 / 60)
                        mpcts.append(float(mp))
                    ax1.bar(centres, mpcts, color=col, alpha=0.7, edgecolor='none')
                    ax1.axhline(MPCT_THRESH, color='#555', lw=0.7, ls='--')
                    ax1.set_ylim(0, 55)
                    ax1.set_xlim(0, rec_dur_s / 60)
                else:
                    ax1.text(0.5, 0.5, 'no trunc QC', transform=ax1.transAxes,
                             ha='center', va='center', fontsize=6, color='#999')
            else:
                ax1.text(0.5, 0.5, 'no QC data', transform=ax1.transAxes,
                         ha='center', va='center', fontsize=6, color='#999')

            ax1.set_ylabel('Missing %', fontsize=6)
            if row_i < n_rows - 1:
                ax1.tick_params(labelbottom=False)
            else:
                ax1.set_xlabel('Time (min)', fontsize=7)

        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

print(f"Saved per-unit PDF → {comp_dir / 'fig4_divergent_units.pdf'}")


# =============================================================================
# FIGURE 5 — Spike count per unit: strategy comparison (violin / box)
# =============================================================================

fig5, axes5 = plt.subplots(1, 2, figsize=(7.2, 3.5))

# Left: spike count distribution per strategy (boxplot)
box_data = [all_stats[n]['n_spikes'].to_numpy(float) for n in strategy_names]
bp = axes5[0].boxplot(
    box_data, positions=np.arange(len(strategy_names)),
    patch_artist=True, showfliers=False, widths=0.55,
    medianprops=dict(color='#111', linewidth=1.2),
    boxprops=dict(linewidth=0.8),
    whiskerprops=dict(linewidth=0.8),
    capprops=dict(linewidth=0.8),
)
for patch, col in zip(bp['boxes'], bcolors):
    patch.set_facecolor(col); patch.set_alpha(0.65); patch.set_edgecolor('none')
axes5[0].set_yscale('log')
axes5[0].set_xticks(np.arange(len(strategy_names)))
axes5[0].set_xticklabels(strategy_names, rotation=35, ha='right', fontsize=6)
axes5[0].set_ylabel('Spikes per unit (log)')
axes5[0].set_title('Per-unit spike count')

# Right: matched unit coinc_frac distribution (boxplot; each non-ref strategy vs ref)
coinc_box_data, coinc_labels, coinc_cols = [], [], []
for name in other_strategies:
    m = match_table.get(name, {})
    fracs = [frac for _, (uid, frac) in m.items() if uid is not None]
    if fracs:
        coinc_box_data.append(fracs)
        coinc_labels.append(name)
        coinc_cols.append(STRATEGY_COLORS[name])

if coinc_box_data:
    bp2 = axes5[1].boxplot(
        coinc_box_data, positions=np.arange(len(coinc_box_data)),
        patch_artist=True, showfliers=False, widths=0.55,
        medianprops=dict(color='#111', linewidth=1.2),
        boxprops=dict(linewidth=0.8), whiskerprops=dict(linewidth=0.8),
        capprops=dict(linewidth=0.8),
    )
    for patch, col in zip(bp2['boxes'], coinc_cols):
        patch.set_facecolor(col); patch.set_alpha(0.65); patch.set_edgecolor('none')
    axes5[1].axhline(COINC_THRESH, color='#555', lw=0.8, ls='--', label=f'match thresh')
    axes5[1].set_xticks(np.arange(len(coinc_labels)))
    axes5[1].set_xticklabels(coinc_labels, rotation=35, ha='right', fontsize=6)
    axes5[1].set_ylim(0, 1.05)
    axes5[1].set_ylabel(f'Coincident fraction (vs {REF_STRATEGY})')
    axes5[1].set_title('Match quality for recovered units')
    axes5[1].legend(fontsize=6)

fig5.suptitle('Spike count and match quality distributions', fontsize=9, y=1.02)
fig5.savefig(comp_dir / 'fig5_distributions.pdf')
fig5.savefig(comp_dir / 'fig5_distributions.png')
plt.close(fig5)
print("Saved Fig 5")


# =============================================================================
# FIGURE 6 — Merge diagnostics: waveforms + ACG/CCG before/after merge
# =============================================================================

print("\nBuilding merge diagnostics...")
premerge_name = 'no_merge' if 'no_merge' in all_data else REF_STRATEGY
premerge_data = all_data[premerge_name]
strategy_templates = {
    name: _load_templates(curation_results[name].directory)
    for name in strategy_names
}
diag_rows = []

with PdfPages(comp_dir / 'fig6_merge_diagnostics.pdf') as pdf:
    any_diag_pages = False
    for name in strategy_names:
        groups = merge_todos.get(name, {}).get('merge_unit_groups', [])
        if not groups:
            continue

        strat_data = all_data[name]
        for group_index, group in enumerate(groups, start=1):
            any_diag_pages = True
            group = [int(u) for u in group]
            member_counts = {uid: int((premerge_data['spike_clusters'] == uid).sum()) for uid in group}
            ordered_members = sorted(group, key=lambda uid: member_counts.get(uid, 0), reverse=True)
            focus_members = ordered_members[:MERGE_DIAG_MAX_UNITS]

            union_samples = np.sort(np.concatenate([
                premerge_data['spike_times'][premerge_data['spike_clusters'] == uid]
                for uid in group
            ]))
            merged_uid, match_frac = _best_matching_unit(union_samples, strat_data)

            diag_rows.append(dict(
                strategy=name,
                group_index=group_index,
                n_members=len(group),
                members=' '.join(map(str, group)),
                matched_merged_unit='' if merged_uid is None else int(merged_uid),
                matched_fraction=float(match_frac),
                premerge_total_spikes=int(len(union_samples)),
                merged_spikes=int((strat_data['spike_clusters'] == merged_uid).sum()) if merged_uid is not None else 0,
            ))

            fig, axes = plt.subplots(2, 2, figsize=(8.0, 6.0))
            fig.suptitle(
                f'{name} merge group {group_index} — members={group} '\
                f'(showing top {len(focus_members)} by spike count)\n'
                f'matched merged unit={merged_uid}  coinc={match_frac*100:.1f}%',
                fontsize=8, y=1.02,
            )

            ax_w_pre, ax_w_post = axes[0]
            ax_acg, ax_ccg = axes[1]

            # Pre-merge waveforms
            pre_tmpl, pre_tmpl_ind = strategy_templates.get(premerge_name, (None, None))
            for uid in focus_members:
                trace, peak_local = _peak_trace(pre_tmpl, pre_tmpl_ind, uid)
                if trace is not None:
                    ax_w_pre.plot(trace, lw=1.0, label=f'u{uid} (local ch{peak_local})')
            ax_w_pre.set_title('Pre-merge member waveforms (peak channel)')
            ax_w_pre.set_xlabel('Sample')
            ax_w_pre.set_ylabel('Amplitude')
            if focus_members:
                ax_w_pre.legend(fontsize=6, ncol=2)

            # Post-merge waveform
            post_tmpl, post_tmpl_ind = strategy_templates.get(name, (None, None))
            merged_trace, merged_peak_local = _peak_trace(
                post_tmpl, post_tmpl_ind,
                int(merged_uid) if merged_uid is not None else -1,
            )
            if merged_trace is not None:
                ax_w_post.plot(merged_trace, color='#111', lw=1.4,
                               label=f'merged u{merged_uid} (local ch{merged_peak_local})')
            else:
                ax_w_post.text(0.5, 0.5, 'merged waveform unavailable',
                               transform=ax_w_post.transAxes, ha='center', va='center', fontsize=7)
            ax_w_post.set_title('Post-merge waveform (peak channel)')
            ax_w_post.set_xlabel('Sample')
            ax_w_post.set_ylabel('Amplitude')
            if merged_trace is not None:
                ax_w_post.legend(fontsize=6)

            # ACGs before + after
            for uid in focus_members:
                unit_samples = premerge_data['spike_times'][premerge_data['spike_clusters'] == uid]
                lags_ms, trace = _corr_trace(unit_samples, fs=FS)
                if trace is not None:
                    ax_acg.plot(lags_ms, trace, lw=1.0, alpha=0.8, label=f'u{uid}')
            lags_ms, merged_acg = (None, None)
            if merged_uid is not None:
                merged_samples = strat_data['spike_times'][strat_data['spike_clusters'] == merged_uid]
                lags_ms, merged_acg = _corr_trace(merged_samples, fs=FS)
                if merged_acg is not None:
                    ax_acg.plot(lags_ms, merged_acg, color='#111', lw=1.6, label=f'merged u{merged_uid}')
            ax_acg.set_title('Autocorrelograms before / after merge')
            ax_acg.set_xlabel('Lag (ms)')
            ax_acg.set_ylabel('Count')
            if focus_members or merged_acg is not None:
                ax_acg.legend(fontsize=6, ncol=2)

            # Pairwise CCGs for the displayed members + union-vs-merged if available.
            ccg_lines = 0
            for uid_a, uid_b in combinations(focus_members, 2):
                a_samples = premerge_data['spike_times'][premerge_data['spike_clusters'] == uid_a]
                b_samples = premerge_data['spike_times'][premerge_data['spike_clusters'] == uid_b]
                lags_ms, trace = _corr_trace(a_samples, b_samples, fs=FS)
                if trace is not None:
                    ax_ccg.plot(lags_ms, trace, lw=1.0, alpha=0.8, label=f'{uid_a}-{uid_b}')
                    ccg_lines += 1
            if merged_uid is not None:
                merged_samples = strat_data['spike_times'][strat_data['spike_clusters'] == merged_uid]
                lags_ms, trace = _corr_trace(union_samples, merged_samples, fs=FS)
                if trace is not None:
                    ax_ccg.plot(lags_ms, trace, color='#111', lw=1.6, ls='--',
                                label=f'union vs merged {merged_uid}')
                    ccg_lines += 1
            if ccg_lines == 0:
                ax_ccg.text(0.5, 0.5, 'no CCG traces available',
                            transform=ax_ccg.transAxes, ha='center', va='center', fontsize=7)
            ax_ccg.set_title('Pairwise CCGs within merge group')
            ax_ccg.set_xlabel('Lag (ms)')
            ax_ccg.set_ylabel('Count')
            if ccg_lines:
                ax_ccg.legend(fontsize=6, ncol=2)

            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

    if not any_diag_pages:
        fig, ax = plt.subplots(figsize=(6.0, 2.2))
        ax.axis('off')
        ax.text(0.5, 0.5, 'No merge groups were produced by any successful strategy.',
                ha='center', va='center', fontsize=8)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

diag_df = pd.DataFrame(diag_rows)
diag_df.to_csv(comp_dir / 'merge_diagnostics.csv', index=False)
print("Saved Fig 6")


# =============================================================================
# FIGURE 7 + TABLE — Evidence table for all proposed merge groups
# =============================================================================
# Reconstruct pre-merge spike arrays from the KS4 base output + duped_spikes.
# All strategy todos share the same duped_spikes; use any available one.

print("\nBuilding evidence table...")

_any_todo = next(
    (np.load(cur_cache / fn, allow_pickle=True).item()
     for fn in list(CACHE_FILES.values()) if (cur_cache / fn).exists()),
    None,
)

if _any_todo is not None:
    duped = np.asarray(_any_todo['duped_spikes'], dtype=int)
    _clu_base = np.delete(ks4_results.spike_clusters,  duped)
    _st_base  = np.delete(ks4_results.spike_times,     duped)
    _amp_base = np.delete(ks4_results.spike_amplitudes,duped)
    _z_base   = np.delete(ks4_results.spike_positions[:, 1], duped)
    _st_s_base= _st_base.astype(float) / FS

    # Load full templates from no_merge output (most faithful to KS4 output)
    _no_merge_dir = _resolve_results_dir(curation_results.get('no_merge', list(curation_results.values())[0]).directory)
    _tmpl_file = _no_merge_dir / 'templates.npy'
    _tmpl_ind_file = _no_merge_dir / 'templates_ind.npy'
    _tmpl     = np.load(_tmpl_file)     if _tmpl_file.exists()     else None
    _tmpl_ind = np.load(_tmpl_ind_file) if _tmpl_ind_file.exists() else None

    _all_groups = {
        name: merge_todos.get(name, {}).get('merge_unit_groups', [])
        for name in strategy_names
    }

    ev_df = build_evidence_table(
        _all_groups,
        _clu_base, _st_base, _st_s_base,
        _amp_base, _z_base,
        templates=_tmpl, templates_ind=_tmpl_ind,
        fs=int(FS),
        out_path=comp_dir / 'curation_moves.csv',
    )
    print(f"  {len(ev_df)} merge proposals evaluated → {comp_dir / 'curation_moves.csv'}")

    # Fig 7: evidence metric distributions coloured by decision
    if len(ev_df):
        _metric_cols = ['depth_spread_um', 'min_cosine', 'bic_delta',
                        'ccg_score', 'post_merge_isi_viol', 'n_strategies_agree']
        _metric_labels = ['Depth spread (µm)', 'Min template cosine', 'BIC delta (1-2 GMM)',
                          'CCG score', 'Post-merge ISI viol.', 'Strategies agreeing']
        _dec_colors = {'accept': '#285E61', 'reject': '#C05621'}

        fig7, axes7 = plt.subplots(2, 3, figsize=(8.5, 5.0))
        plt.subplots_adjust(hspace=0.55, wspace=0.40)
        axes7_flat = axes7.flatten()

        for ax, col, lbl in zip(axes7_flat, _metric_cols, _metric_labels):
            for dec, grp in ev_df.groupby('decision'):
                vals = grp[col].dropna().values
                if len(vals) == 0:
                    continue
                ax.scatter(
                    grp.index[:len(vals)], vals,
                    color=_dec_colors.get(dec, '#888'), s=20, alpha=0.7,
                    label=dec,
                )
            ax.set_title(lbl, fontsize=8)
            ax.set_xlabel('Group index', fontsize=7)
            ax.tick_params(labelsize=6)
            ax.legend(fontsize=6)

        for ax in axes7_flat[len(_metric_cols):]:
            ax.set_visible(False)

        fig7.suptitle('Evidence table — merge proposal metrics', fontsize=9, y=1.02)
        fig7.savefig(comp_dir / 'fig7_evidence_table.pdf', bbox_inches='tight')
        fig7.savefig(comp_dir / 'fig7_evidence_table.png', dpi=150, bbox_inches='tight')
        plt.close(fig7)
        print("Saved Fig 7")
else:
    print("  No strategy caches found — skipping evidence table")
    ev_df = None


# =============================================================================
# FIGURE 8 + TABLE — Split candidate scoring on the no_merge output
# =============================================================================

print("\nScoring split candidates...")

_ref_res = curation_results.get('no_merge') or list(curation_results.values())[0]
_ref_dir = _resolve_results_dir(_ref_res.directory)
_tF_file = _ref_dir / 'tF.npy'

_tF_loaded = np.load(_tF_file) if _tF_file.exists() else None

split_df = score_split_candidates(
    clu           = _ref_res.spike_clusters,
    tF            = _tF_loaded,
    spike_times_s = _ref_res.spike_times.astype(float) / FS,
    amplitudes    = _ref_res.spike_amplitudes,
    spike_z       = _ref_res.spike_positions[:, 1],
    fs            = int(FS),
    out_path      = comp_dir / 'split_candidates.csv',
)
print(f"  {len(split_df)} units scored → {comp_dir / 'split_candidates.csv'}")

_action_summary = split_df['recommended_action'].value_counts().to_dict()
print(f"  Actions: {_action_summary}")

# Fig 8: ranked split candidate scores
if len(split_df):
    _action_palette = {
        'split_review': '#C05621',
        'drift_review': '#C05621',
        'mua':          '#6B46C1',
        'ok':           '#285E61',
        'too_few_spikes': '#aaa',
    }

    fig8, axes8 = plt.subplots(1, 3, figsize=(9.0, 3.5))
    plt.subplots_adjust(wspace=0.40)

    scored = split_df[split_df['recommended_action'] != 'too_few_spikes'].copy()
    colors8 = [_action_palette.get(a, '#888') for a in scored['recommended_action']]

    # Panel 1: split priority score by unit depth
    axes8[0].scatter(scored['depth_um'], scored['split_priority'],
                     c=colors8, s=15, alpha=0.75)
    axes8[0].set_xlabel('Unit depth (µm)')
    axes8[0].set_ylabel('Split priority')
    axes8[0].set_title('Split priority vs depth')

    # Panel 2: tF bimodality vs amplitude bimodality
    axes8[1].scatter(scored['amp_bimodality'], scored['tF_bimodality'],
                     c=colors8, s=15, alpha=0.75)
    axes8[1].axhline(0, color='#aaa', lw=0.7, ls='--')
    axes8[1].axvline(0, color='#aaa', lw=0.7, ls='--')
    axes8[1].set_xlabel('Amp BIC delta (neg=bimodal)')
    axes8[1].set_ylabel('tF BIC delta (neg=bimodal)')
    axes8[1].set_title('Bimodality: amplitude vs features')

    # Panel 3: ISI violation rate distribution per action
    for action, col in _action_palette.items():
        sub = scored[scored['recommended_action'] == action]['isi_viol_rate'].dropna()
        if len(sub):
            axes8[2].scatter(
                [action] * len(sub), sub,
                color=col, s=12, alpha=0.5,
            )
    axes8[2].axhline(0.05, color='#555', lw=0.8, ls='--', label='5% threshold')
    axes8[2].set_ylabel('ISI violation rate')
    axes8[2].set_title('Contamination by action')
    axes8[2].tick_params(axis='x', labelsize=6, rotation=30)
    axes8[2].legend(fontsize=6)

    # Legend
    from matplotlib.patches import Patch
    handles = [Patch(color=c, label=a) for a, c in _action_palette.items() if a != 'too_few_spikes']
    fig8.legend(handles=handles, loc='upper center', ncol=len(handles),
                fontsize=7, bbox_to_anchor=(0.5, 1.04))
    fig8.suptitle(f'Split candidate scoring — {_ref_res.directory.name}', fontsize=9, y=1.08)
    fig8.savefig(comp_dir / 'fig8_split_candidates.pdf', bbox_inches='tight')
    fig8.savefig(comp_dir / 'fig8_split_candidates.png', dpi=150, bbox_inches='tight')
    plt.close(fig8)
    print("Saved Fig 8")


# =============================================================================
# FIGURE 9 + TABLE — Temporal handoff diagnostics for amp_bic / posthoc pairs
# =============================================================================
# For each candidate pair from amp_bic and posthoc, diagnose whether the two
# units are temporally complementary fragments of the same cell, or unrelated
# units connected by weak amplitude / feature coincidence.
#
# Uses the no-merge base arrays (_clu_base etc.) already computed for Fig 7.
# Skips if the Fig 7 data block was not reached (no strategy caches found).

print("\nBuilding temporal handoff diagnostics...")

_DIAG_METHODS = ['amp_bic', 'posthoc']
_diag_groups = {
    m: merge_todos.get(m, {}).get('merge_unit_groups', [])
    for m in _DIAG_METHODS
}
_n_diag_groups = sum(len(g) for g in _diag_groups.values())

if _any_todo is not None and _n_diag_groups > 0:
    _Wall = np.load(ks4_out / 'Wall.npy')

    temporal_df = build_temporal_diagnostics(
        merge_groups_by_method = _diag_groups,
        clu           = _clu_base,
        spike_times_s = _st_s_base,
        amplitudes    = _amp_base,
        spike_z       = _z_base,
        Wall          = _Wall,
        out_path      = comp_dir / 'candidate_pair_temporal_diagnostics.csv',
    )
    print(f"  {len(temporal_df)} pairs evaluated "
          f"→ {comp_dir / 'candidate_pair_temporal_diagnostics.csv'}")

    if len(temporal_df):
        _n_split = int(temporal_df['looks_temporal_split'].sum())
        _n_nonlocal = int(temporal_df['reject_nonlocal'].sum())
        _n_coact = int(temporal_df['reject_coactive_contam'].sum())
        print(f"  looks_temporal_split={_n_split}  "
              f"reject_nonlocal={_n_nonlocal}  reject_coactive_contam={_n_coact}")

    # Fig 9: four-panel temporal diagnostic summary
    if len(temporal_df) > 1:
        _method_colors = {'amp_bic': PALETTE[2], 'posthoc': PALETTE[3]}
        _label_markers = {True: 'o', False: 'x'}

        fig9, axes9 = plt.subplots(2, 2, figsize=(7.5, 6.0))
        plt.subplots_adjust(hspace=0.45, wspace=0.40)

        # Panel 1: depth_diff vs template_cosine, coloured by method
        ax = axes9[0, 0]
        for method, grp in temporal_df.groupby('method'):
            ax.scatter(
                grp['depth_diff_um'], grp['template_cosine'],
                color=_method_colors.get(method, '#888'),
                s=15, alpha=0.65, label=method,
            )
        ax.axvline(75,  color='#aaa', lw=0.8, ls='--', label='75 µm')
        ax.axvline(150, color='#888', lw=0.8, ls=':',  label='150 µm')
        ax.axhline(0.90, color='#aaa', lw=0.8, ls='--')
        ax.set_xlabel('Depth diff (µm)')
        ax.set_ylabel('Template cosine')
        ax.set_title('Locality & waveform compatibility')
        ax.legend(fontsize=6)

        # Panel 2: handoff_score vs time_overlap_frac, coloured by looks_temporal_split
        ax = axes9[0, 1]
        _split_colors = {True: '#285E61', False: '#C05621'}
        for lts, grp in temporal_df.groupby('looks_temporal_split'):
            ax.scatter(
                grp['time_overlap_frac'], grp['handoff_score'],
                color=_split_colors[lts],
                s=15, alpha=0.65,
                label='temporal split' if lts else 'other',
            )
        ax.axhline(0.30, color='#aaa', lw=0.8, ls='--', label='handoff=0.3')
        ax.axvline(0.20, color='#aaa', lw=0.8, ls=':')
        ax.set_xlabel('Time overlap fraction')
        ax.set_ylabel('Handoff score')
        ax.set_title('Temporal structure')
        ax.legend(fontsize=6)

        # Panel 3: union_amp_smoothness_gain distribution by method
        ax = axes9[1, 0]
        for method, grp in temporal_df.groupby('method'):
            vals = grp['union_amp_smoothness_gain'].dropna().values
            if len(vals):
                ax.hist(vals, bins=20, alpha=0.55,
                        color=_method_colors.get(method, '#888'), label=method)
        ax.axvline(0, color='#555', lw=0.8, ls='--', label='break-even')
        ax.set_xlabel('Amplitude smoothness gain')
        ax.set_ylabel('Count')
        ax.set_title('Does merging smooth the amplitude trajectory?')
        ax.legend(fontsize=6)

        # Panel 4: post_merge_isi vs coactivity_refractory_violation
        ax = axes9[1, 1]
        for method, grp in temporal_df.groupby('method'):
            ax.scatter(
                grp['post_merge_isi'],
                grp['coactivity_refractory_violation'],
                color=_method_colors.get(method, '#888'),
                s=15, alpha=0.65, label=method,
            )
        ax.axvline(0.05, color='#aaa', lw=0.8, ls='--', label='5% ISI')
        ax.axhline(0.05, color='#aaa', lw=0.8, ls=':')
        ax.set_xlabel('Post-merge ISI viol. (full recording)')
        ax.set_ylabel('Coactivity ISI viol. (overlap bins only)')
        ax.set_title('Refractory violations')
        ax.legend(fontsize=6)

        fig9.suptitle('Temporal handoff diagnostics — amp_bic & posthoc candidates',
                      fontsize=9, y=1.02)
        fig9.savefig(comp_dir / 'fig9_temporal_diagnostics.pdf', bbox_inches='tight')
        fig9.savefig(comp_dir / 'fig9_temporal_diagnostics.png', dpi=150, bbox_inches='tight')
        plt.close(fig9)
        print("Saved Fig 9")
else:
    temporal_df = None
    if _n_diag_groups == 0:
        print("  No amp_bic / posthoc merge groups — skipping temporal diagnostics")
    else:
        print("  No strategy caches found — skipping temporal diagnostics")


# =============================================================================
# Done
# =============================================================================

print(f"\nAll outputs written to: {comp_dir}")
print("  curation_summary.csv")
print("  merge_counts.csv")
print("  curation_moves.csv      ← evidence table (new)")
print("  split_candidates.csv    ← split candidate ranking (new)")
print("  fig1_overview.pdf/png")
print("  fig2_merge_activity.pdf/png")
print("  fig3_pairwise_agreement.pdf/png")
print("  fig4_divergent_units.pdf")
print("  fig5_distributions.pdf/png")
print("  fig6_merge_diagnostics.pdf")
print("  fig7_evidence_table.pdf/png  ← new")
print("  fig8_split_candidates.pdf/png ← new")
print("  candidate_pair_temporal_diagnostics.csv ← new")
print("  fig9_temporal_diagnostics.pdf/png       ← new")
print("  merge_diagnostics.csv")
