#%%
# curation_comparison_sweep.py
#
# Runs four curation strategies on one KS4 output directory and compares them.
# Useful for deciding which post-hoc curation works best on patched KS4 data.
#
# Usage (from this file as a script, or #%% cell-by-cell):
#   PIPELINE_DIR = "/path/to/pipeline_dir"   # contains kilosort4/sorter_output/
#
# Strategies compared:
#   no_merge  — dup-spike + redundant removal only      (baseline)
#   posthoc   — feature-projection + CCG merge          (merge_posthoc3 port)
#   cosine    — Wall cosine similarity + CCG merge
#   amp_bic   — amplitude BIC (1 vs 2-Gaussian) + CCG merge

import os
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

from spikeinterface.extractors import read_kilosort
from pipeline.sorting import KilosortResults
from pipeline.qc import truncation_qc
from pipeline.curation_postpatch import (
    run_cur, run_cur_cosine, run_cur_amp_bic, run_cur_no_merge,
)

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

FS             = 30_000.0
RECALC         = os.environ.get('CUR_RECALC', '0').strip() == '1'
REF_STRATEGY   = os.environ.get('CUR_REF', 'no_merge').strip()
COINC_TOLERANCE = 0.5e-3    # s — spike-matching window
COINC_THRESH    = 0.30      # min fraction of ref spikes that must coincide to count as a match
MPCT_THRESH     = 20.0      # amplitude truncation threshold for "well-detected"
PRESENCE_THRESH = 0.50

# Per-strategy parameters — tweak here to explore the parameter space
STRATEGY_KWARGS = {
    'no_merge': {},
    'posthoc':  {'posthoc_score_thresh': 3, 'posthoc_ccg_thresh': 0.5,
                 'posthoc_min_spikes_seed': 500, 'posthoc_min_spikes_pair': 100},
    'cosine':   {'cosine_thresh': 0.90, 'ccg_thresh': 0.5,
                 'min_spikes_seed': 500, 'min_spikes_pair': 100},
    'amp_bic':  {'bic_margin': 0.0, 'ccg_thresh': 0.5,
                 'min_spikes_seed': 100, 'min_spikes_pair': 100},
}

STRATEGY_FNS = {
    'no_merge': run_cur_no_merge,
    'posthoc':  run_cur,
    'cosine':   run_cur_cosine,
    'amp_bic':  run_cur_amp_bic,
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
qc_results = {}   # strategy name → (trunc_qc, pres_qc) or None
for name, res in curation_results.items():
    qc_dir = PIPELINE_DIR / f'qc_{name}' / 'amp_truncation'
    qc_dir.mkdir(parents=True, exist_ok=True)
    try:
        trunc, pres = truncation_qc(
            res.spike_times, res.spike_clusters, res.spike_amplitudes,
            cache_dir=qc_dir, recalc=RECALC,
        )
        qc_results[name] = (trunc, pres)
        print(f"  [{name}] QC done → {qc_dir}")
    except Exception as e:
        print(f"  [{name}] QC failed: {e}")
        qc_results[name] = None

# =============================================================================
# Step 4: Load comparison data (spike arrays + QC)
# =============================================================================

def load_strategy_data(name, res):
    """Collect everything needed for unit-level comparison."""
    qc = qc_results.get(name)
    trunc, pres = qc if qc is not None else (None, None)
    return dict(
        spike_times    = res.spike_times,
        spike_clusters = res.spike_clusters,
        spike_amps     = res.spike_amplitudes,
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

def unit_stats(data, fs=FS):
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
    lbl_file = curation_results[name].directory / 'cluster_KSLabel.tsv'
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

        rows.append(dict(
            unit_id=int(uid),
            label=label_map.get(int(uid), 'unknown'),
            n_spikes=int((sc == uid).sum()),
            firing_rate_hz=float((sc == uid).sum()) / rec_dur,
            mean_mpct=mean_mpct,
            presence_frac=presence_frac,
        ))
    return pd.DataFrame(rows)


print("\nComputing per-unit statistics...")
all_stats = {}
for name in strategy_names:
    all_stats[name] = unit_stats(all_data[name])


def run_summary(stats_df):
    n_units = len(stats_df)
    n_good  = int((stats_df['label'] == 'good').sum())
    well    = (stats_df['mean_mpct'] < MPCT_THRESH) & (stats_df['presence_frac'] > PRESENCE_THRESH)
    spk     = stats_df['n_spikes'].to_numpy(float)
    q25, q50, q75 = (np.nanpercentile(spk, [25, 50, 75]) if len(spk) else (np.nan,)*3)
    return dict(
        n_units=n_units, n_good=n_good, n_well=int(well.sum()),
        efficiency=round(well.sum()/n_units, 3) if n_units else 0,
        median_mpct=float(np.nanmedian(stats_df['mean_mpct'])),
        med_presence=float(np.nanmedian(stats_df['presence_frac'])),
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


def build_matches(ref_data, other_data, fs=FS, min_spikes=100):
    """ref unit → (best_other_uid, coinc_frac) or (None, 0)."""
    ref_st  = ref_data['spike_times'].astype(float) / fs
    ref_clu = ref_data['spike_clusters']
    oth_st  = other_data['spike_times'].astype(float) / fs
    oth_clu = other_data['spike_clusters']

    oth_by_unit = {u: np.sort(oth_st[oth_clu == u]) for u in np.unique(oth_clu)}

    matches = {}
    for ref_uid in np.unique(ref_clu):
        rt = np.sort(ref_st[ref_clu == ref_uid])
        if len(rt) < min_spikes:
            matches[ref_uid] = (None, 0.0)
            continue
        best_uid, best_frac = None, 0.0
        for oth_uid, ot in oth_by_unit.items():
            if len(ot) < min_spikes:
                continue
            frac = find_coincident_spikes(rt, ot) / len(rt)
            if frac > best_frac:
                best_frac, best_uid = frac, oth_uid
        matches[ref_uid] = (best_uid, best_frac) if best_frac >= COINC_THRESH else (None, 0.0)
    return matches


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
for name, fname in CACHE_FILES.items():
    p = cur_cache / fname
    if p.exists():
        todo = np.load(p, allow_pickle=True).item()
        mg   = todo.get('merge_unit_groups', [])
        n_merged_units = sum(len(g) for g in mg)
        merge_counts[name] = dict(
            n_groups=len(mg),
            n_merged_units=n_merged_units,
            n_removed=len(todo.get('removed_units', [])),
            n_duped=len(todo.get('duped_spikes', [])),
        )
    else:
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
            ax.text(j, i, f'{v:{fmt}}' if np.isfinite(v) else '—',
                    ha='center', va='center', fontsize=6,
                    color='white' if v > 0.6 * float(arr[np.isfinite(arr)].max()) else 'black')
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
# Done
# =============================================================================

print(f"\nAll outputs written to: {comp_dir}")
print("  curation_summary.csv")
print("  merge_counts.csv")
print("  fig1_overview.pdf/png")
print("  fig2_merge_activity.pdf/png")
print("  fig3_pairwise_agreement.pdf/png")
print("  fig4_divergent_units.pdf")
print("  fig5_distributions.pdf/png")
