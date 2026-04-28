#%%
from pathlib import Path
import json
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

# =============================================================================
# Style — nature-journal figure defaults
# =============================================================================
plt.rcParams.update({
    'font.family':          'sans-serif',
    'font.sans-serif':      ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size':            7,
    'axes.labelsize':       8,
    'axes.titlesize':       8,
    'axes.titlepad':        4,
    'axes.labelpad':        3,
    'xtick.labelsize':      7,
    'ytick.labelsize':      7,
    'legend.fontsize':      7,
    'legend.frameon':       False,
    'axes.spines.top':      False,
    'axes.spines.right':    False,
    'axes.linewidth':       0.7,
    'xtick.major.width':    0.7,
    'ytick.major.width':    0.7,
    'xtick.major.size':     3,
    'ytick.major.size':     3,
    'xtick.minor.size':     1.5,
    'ytick.minor.size':     1.5,
    'lines.linewidth':      1.2,
    'figure.dpi':           150,
    'savefig.dpi':          300,
    'savefig.bbox':         'tight',
    'pdf.fonttype':         42,
    'ps.fonttype':          42,
})

PALETTE = ['#2B6CB0', '#C05621', '#276749', '#6B46C1', '#97266D', '#285E61', '#744210']

# =============================================================================
# Configuration
# =============================================================================

_DEFAULT_SWEEP_DIR = "/mnt/NPX/Luke/20260316/dredge_pipeline_results_Luke03162026_V2V1_RH_g0_imec1/shallow_sweep"
#_DEFAULT_SWEEP_DIR = "/mnt/NPX/Luke/20260302/dredge_pipeline_results_Luke03022026_V2V1_RH_g0_imec1/shallow_sweep"
# Allow overriding default via env var COMPARE_SWEEP_DIR or CLI --sweep-dir
sweep_dir = Path(os.environ.get('COMPARE_SWEEP_DIR', _DEFAULT_SWEEP_DIR).strip())



FS               = 30_000.0   # Hz
REF_RUN          = os.environ.get('COMPARE_REF_RUN', 'default').strip() or 'default'
ANALYSIS_STAGE   = os.environ.get('COMPARE_STAGE', 'postcuration').strip().lower() or 'postcuration'
COINC_TOLERANCE  = 0.5e-3     # seconds — spike coincidence window for matching
COINC_THRESH     = 0.30       # min fraction of ref spikes that must coincide
MPCT_THRESH      = 20.0       # max mean missing% to be "well-detected"
PRESENCE_THRESH  = 0.50       # min presence fraction

STAGE_SUFFIX = '' if ANALYSIS_STAGE == 'postcuration' else f'_{ANALYSIS_STAGE}'
REF_SUFFIX = '' if REF_RUN == 'default' else f'_ref-{REF_RUN}'

# Parse light CLI overrides (use parse_known_args so importing won't break)
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--sweep-dir', dest='sweep_dir', default=None,
                    help='Path to shallow-sweep directory (overrides COMPARE_SWEEP_DIR env)')
parser.add_argument('--ref-run', dest='ref_run', default=None,
                    help='Reference run name to compare against (overrides COMPARE_REF_RUN env)')
parser.add_argument('--stage', dest='stage', default=None,
                    help='Analysis stage (overrides COMPARE_STAGE env)')
args, _ = parser.parse_known_args()
if args.sweep_dir:
    sweep_dir = Path(args.sweep_dir)
if args.ref_run:
    REF_RUN = args.ref_run.strip() or REF_RUN
if args.stage:
    ANALYSIS_STAGE = args.stage.strip().lower() or ANALYSIS_STAGE

# recompute suffixes in case ref/stage were overridden
STAGE_SUFFIX = '' if ANALYSIS_STAGE == 'postcuration' else f'_{ANALYSIS_STAGE}'
REF_SUFFIX = '' if REF_RUN == 'default' else f'_ref-{REF_RUN}'


def out_name(base_name):
    """Return an output filename with a reference-run suffix when needed."""
    p = Path(base_name)
    return f"{p.stem}{STAGE_SUFFIX}{REF_SUFFIX}{p.suffix}"


def get_stage_paths(run_dir):
    """Return sorter-output and QC paths for the configured analysis stage."""
    if ANALYSIS_STAGE in ('precuration', 'pre', 'ks4', 'sorter'):
        return run_dir / 'kilosort4' / 'sorter_output', run_dir / 'qc_pre' / 'amp_truncation'
    return run_dir / 'cur' / 'cur_sorter_output', run_dir / 'qc' / 'amp_truncation'

# =============================================================================
# Stimulus-locked split diagnostics (experimental)
# =============================================================================
# These analyses try to flag the pattern:
#   - a reference unit's spikes are explained by two units in another run
#   - the two "child" units trade off across time (anti-correlation / segregation)
#   - but the combined spikes are conserved (suggesting an algorithmic split)

SPLIT_BIN_S = 10.0            # seconds — time bin for tradeoff / conservation
SPLIT_TOPK  = 3               # consider top-K coincident matches per ref unit
SPLIT_MIN_SPIKES = 300        # skip low-spike units to reduce noise

# Heuristic thresholds for flagging candidates (tune as needed)
SPLIT_SCORE_THRESH   = 0.55   # (frac1 + frac2) must exceed this
SPLIT_CHILD2_MIN_FRAC = 0.10  # second child must explain at least this fraction
SEGREGATION_THRESH   = 0.55   # mean |c1-c2|/(c1+c2) across bins
ANTICORR_THRESH      = -0.20  # corr(c1, c2) across bins; more negative = stronger handoff
CONSERVATION_THRESH  = 0.60   # mean (c1+c2)/ref_count across bins with ref_count>0
MAX_SPLIT_PAGES_PDF  = 60     # cap pages in split diagnostics PDF
FINE_CCG_WINDOW_S    = 5e-3   # +/- window for pairwise fine-timescale CCG
FINE_CCG_BIN_S       = 0.2e-3 # seconds per fine CCG bin
FINE_CCG_NEAR_ZERO_S = 0.5e-3 # summarize near-zero-lag coincidence within +/- this window
DUPLICATE_NEAR_ZERO_FRAC_THRESH = 0.05   # heuristic: excess near-zero-lag fraction
DUPLICATE_ZERO_PEAK_RATIO_THRESH = 1.25  # heuristic: near-zero bin exceeds baseline

# Within-run duplicate screen (tests peeling duplication hypothesis directly)
WITHIN_RUN_NEARBY_DEPTH_UM  = 100.0   # max template-depth separation between candidate pairs
WITHIN_RUN_MUA_ONLY         = True    # skip good×good pairs — focus on mua-involved pairs
WITHIN_RUN_MIN_SPIKES       = 200     # min spikes per unit to include in screen
WITHIN_RUN_TOP_PAIRS_PDF    = 40      # max CCG trace panels per run in fig4
WITHIN_RUN_RUNS             = ['default', 'claim_tonly', 'claim_spatial', 'peel3', 'peel2', 'peel1']  # runs to compare

# Sorter params to read from each run's spikeinterface_params.json
TRACKED_SORTER_PARAMS = [
    'Th_universal',
    'Th_learned',
    'max_peels',
    'ccg_threshold',
    'nearest_chans',
    'max_channel_distance',
    'nearest_templates',
    'cross_peel_claim_ms',
    'cross_peel_claim_um',
]

# Parameter families to plot in Figure 2. Runs are selected dynamically:
# include runs that vary the target param while matching default on the others.
SWEEP_FAMILY_SPECS = [
    ('Detection threshold\n(Th_universal)', 'Th_universal'),
    ('Template threshold\n(Th_learned)', 'Th_learned'),
    ('Peeling passes\n(max_peels)', 'max_peels'),
    ('CCG merge threshold\n(ccg_threshold)', 'ccg_threshold'),
]

# =============================================================================
#%% Data loading helpers
# =============================================================================

def load_unit_depths_from_templates(sorter_out_dir):
    """Per-unit depth estimated from the peak-RMS channel of each template."""
    tmpl   = np.load(sorter_out_dir / 'templates.npy')           # (n_units, n_t, n_ch)
    ch_pos = np.load(sorter_out_dir / 'channel_positions.npy')   # (n_ch, 2)
    rms    = np.sqrt(np.mean(tmpl ** 2, axis=1))                  # (n_units, n_ch)
    peak_ch = np.argmax(rms, axis=1)                              # (n_units,)
    return ch_pos[peak_ch, 1]                                     # y-depth per unit


def within_run_duplicate_screen(run_dir, fs=FS):
    """
    Cross-CCG screen between nearby units within a single KS4 run.
    Uses raw KS4 sorter_output (pre-curation) — peeling artifacts are expected
    to exist before any post-hoc merging or labelling.

    For each pair of units separated by <= WITHIN_RUN_NEARBY_DEPTH_UM, computes
    a fine-timescale cross-CCG and returns metrics including near-zero-lag excess.
    A strong near-zero-lag peak between nearby mua units is the direct signature
    of duplicate-peel detections of the same event.

    Returns list of dicts (one per qualifying pair), or None if data missing.
    """
    sorter_out = run_dir / 'kilosort4' / 'sorter_output'
    if not sorter_out.exists():
        return None

    spike_times    = np.load(sorter_out / 'spike_times.npy').astype(float) / fs
    spike_clusters = np.load(sorter_out / 'spike_clusters.npy')
    labels_df      = pd.read_csv(sorter_out / 'cluster_KSLabel.tsv', sep='\t')
    labels_df.columns = [c.strip() for c in labels_df.columns]
    label_map      = dict(zip(labels_df.iloc[:, 0].astype(int), labels_df.iloc[:, 1]))

    unit_depths = load_unit_depths_from_templates(sorter_out)  # (n_units,)

    by_unit = {}
    for uid in np.unique(spike_clusters):
        t = np.sort(spike_times[spike_clusters == uid])
        if len(t) >= WITHIN_RUN_MIN_SPIKES:
            by_unit[int(uid)] = t

    uid_list = sorted(by_unit.keys())
    rows = []
    for i, ua in enumerate(uid_list):
        for ub in uid_list[i + 1:]:
            if ua >= len(unit_depths) or ub >= len(unit_depths):
                continue
            depth_diff = abs(float(unit_depths[ua]) - float(unit_depths[ub]))
            if depth_diff > WITHIN_RUN_NEARBY_DEPTH_UM:
                continue
            la = label_map.get(ua, 'unknown')
            lb = label_map.get(ub, 'unknown')
            if WITHIN_RUN_MUA_ONLY and (la == 'good' and lb == 'good'):
                continue

            ccg = fine_ccg_for_pair(by_unit[ua], by_unit[ub])
            rows.append(dict(
                unit_a=ua, unit_b=ub,
                label_a=la, label_b=lb,
                depth_a=float(unit_depths[ua]),
                depth_b=float(unit_depths[ub]),
                depth_diff=depth_diff,
                n_spikes_a=len(by_unit[ua]),
                n_spikes_b=len(by_unit[ub]),
                near_zero_frac=float(ccg['near_zero_frac'])
                               if np.isfinite(ccg['near_zero_frac']) else np.nan,
                zero_peak_ratio=float(ccg['zero_peak_ratio'])
                                if np.isfinite(ccg['zero_peak_ratio']) else np.nan,
                total_pairs=int(ccg['total_pairs']),
                near_zero_pairs=int(ccg['near_zero_pairs']),
                _counts=np.asarray(ccg['counts']).copy(),
                _bin_centers_s=np.asarray(ccg['bin_centers_s']).copy(),
            ))
    return rows


def load_run_data(run_dir):
    """Load all spike sorting outputs needed for comparison."""
    sorter_out, qc = get_stage_paths(run_dir)
    if not sorter_out.exists():
        return None

    spike_times    = np.load(sorter_out / 'spike_times.npy')      # samples
    spike_clusters = np.load(sorter_out / 'spike_clusters.npy')
    spike_amps     = np.load(sorter_out / 'amplitudes.npy') if (sorter_out / 'amplitudes.npy').exists() else None

    labels_df = pd.read_csv(sorter_out / 'cluster_KSLabel.tsv', sep='\t')
    labels_df.columns = [c.strip() for c in labels_df.columns]

    trunc = np.load(qc / 'truncation_qc.npz') if (qc / 'truncation_qc.npz').exists() else None
    pres  = np.load(qc / 'present_qc.npz')    if (qc / 'present_qc.npz').exists()    else None

    return dict(spike_times=spike_times, spike_clusters=spike_clusters,
                spike_amps=spike_amps, labels_df=labels_df, trunc=trunc, pres=pres)


def load_sorter_params(run_dir):
    """Load spikeinterface sorter params for this run (if present)."""
    p = run_dir / 'kilosort4' / 'spikeinterface_params.json'
    if not p.exists():
        return {}
    try:
        obj = json.loads(p.read_text())
    except Exception:
        return {}
    if isinstance(obj, dict) and isinstance(obj.get('sorter_params'), dict):
        return obj['sorter_params']
    return {}


def _float_eq(a, b, tol=1e-9):
    try:
        return abs(float(a) - float(b)) <= tol
    except Exception:
        return False


def select_family_runs(run_names, params_by_run, family_param, default_run=REF_RUN):
    """
    Pick runs suitable for a sweep-family plot of family_param.
    Criteria: run has family_param, and matches default for all other tracked params.
    Returns a list of run names sorted by family_param value.
    """
    default_params = params_by_run.get(default_run, {})
    if not default_params:
        return []

    selected = []
    for rn in run_names:
        rp = params_by_run.get(rn, {})
        if family_param not in rp:
            continue

        ok = True
        for k in TRACKED_SORTER_PARAMS:
            if k == family_param:
                continue
            if k not in default_params or k not in rp:
                continue

            dv = default_params[k]
            rv = rp[k]
            if isinstance(dv, (int, float)) and isinstance(rv, (int, float)):
                if not _float_eq(rv, dv):
                    ok = False
                    break
            else:
                if rv != dv:
                    ok = False
                    break

        if ok:
            selected.append(rn)

    selected = sorted(selected, key=lambda r: float(params_by_run.get(r, {}).get(family_param, np.nan)))
    # Ensure default is included (and in correct position by sorting)
    return selected


def unit_stats(data, fs=FS):
    """Per-unit metrics from loaded run data. Returns a DataFrame."""
    spike_times_s  = data['spike_times'].astype(float) / fs
    spike_clusters = data['spike_clusters']
    labels_df      = data['labels_df']
    trunc          = data['trunc']
    pres           = data['pres']

    rec_dur = float(spike_times_s.max())

    trunc_cid   = trunc['cid'].astype(int)   if trunc else np.array([], int)
    trunc_mpcts = trunc['mpcts']             if trunc else np.array([])
    pres_cid    = pres['cid'].astype(int)    if pres  else np.array([], int)
    pres_vblk   = pres['valid_blocks']       if pres  else np.zeros((0, 2), int)
    trunc_wblk  = trunc['window_blocks']     if trunc else np.zeros((0, 2), int)

    rows = []
    for uid in np.unique(spike_clusters):
        u_spikes_s = spike_times_s[spike_clusters == uid]

        tm = (trunc_cid == uid)
        mean_mpct  = float(np.nanmean(trunc_mpcts[tm])) if tm.any() else np.nan
        n_windows  = int(tm.sum())

        pm = (pres_cid == uid)
        if pm.any():
            vb = pres_vblk[pm]
            pres_s = sum(
                u_spikes_s[min(int(i1), len(u_spikes_s)-1)] - u_spikes_s[min(int(i0), len(u_spikes_s)-1)]
                for i0, i1 in vb
            )
            presence_frac = min(pres_s / rec_dur, 1.0)
        else:
            presence_frac = 0.0

        match = labels_df[labels_df.iloc[:, 0] == uid]
        label = match.iloc[0, 1] if len(match) > 0 else 'unknown'

        n_spikes = int((spike_clusters == uid).sum())
        fr_hz = (float(n_spikes) / rec_dur) if (rec_dur and np.isfinite(rec_dur) and rec_dur > 0) else np.nan

        rows.append(dict(unit_id=uid, mean_mpct=mean_mpct, n_windows=n_windows,
                 presence_frac=presence_frac,
                 n_spikes=n_spikes, firing_rate_hz=fr_hz, label=label))
    return pd.DataFrame(rows)


def run_summary(stats_df):
    """Scalar summary metrics from a per-unit DataFrame."""
    n_units = len(stats_df)
    n_good  = int((stats_df['label'] == 'good').sum())
    well    = (stats_df['mean_mpct'] < MPCT_THRESH) & (stats_df['presence_frac'] > PRESENCE_THRESH)

    n_spikes_by_unit = stats_df['n_spikes'].to_numpy(dtype=float) if len(stats_df) else np.array([])
    total_spikes = int(np.nansum(n_spikes_by_unit)) if n_spikes_by_unit.size else 0
    if n_spikes_by_unit.size:
        q25, q50, q75 = np.nanpercentile(n_spikes_by_unit, [25, 50, 75])
    else:
        q25 = q50 = q75 = np.nan

    fr_by_unit = stats_df['firing_rate_hz'].to_numpy(dtype=float) if ('firing_rate_hz' in stats_df.columns and len(stats_df)) else np.array([])
    fr_med = float(np.nanmedian(fr_by_unit)) if fr_by_unit.size else np.nan

    return dict(
        n_units=n_units,
        n_good=n_good,
        n_well=int(well.sum()),
        efficiency=round(well.sum() / n_units, 3) if n_units else 0,
        median_mpct=float(np.nanmedian(stats_df['mean_mpct'])),
        med_presence=float(np.nanmedian(stats_df['presence_frac'])),
        total_spikes=total_spikes,
        spikes_per_unit_q25=float(q25) if np.isfinite(q25) else np.nan,
        spikes_per_unit_median=float(q50) if np.isfinite(q50) else np.nan,
        spikes_per_unit_q75=float(q75) if np.isfinite(q75) else np.nan,
        firing_rate_hz_median=fr_med if np.isfinite(fr_med) else np.nan,
    )


# =============================================================================
#%% Coincident spike matching (from compare_sortings_ryan.py)
# =============================================================================

def find_coincident_spikes(times_a, times_b, tol=COINC_TOLERANCE):
    """Count spikes in times_a with a coincident spike in times_b within tol seconds."""
    if not len(times_a) or not len(times_b):
        return 0
    idx       = np.searchsorted(np.sort(times_b), np.sort(times_a))
    il        = np.clip(idx - 1, 0, len(times_b) - 1)
    ir        = np.clip(idx,     0, len(times_b) - 1)
    tb        = np.sort(times_b)
    min_dist  = np.minimum(np.abs(np.sort(times_a) - tb[il]),
                           np.abs(np.sort(times_a) - tb[ir]))
    return int((min_dist <= tol).sum())


def build_matches(ref_data, other_data, fs=FS, min_spikes=100):
    """
    Match units from other_data to ref_data by coincident spike fraction.
    Returns dict: ref_uid -> (best_other_uid, coinc_frac) or (None, 0).
    """
    ref_st  = ref_data['spike_times'].astype(float)   / fs
    ref_clu = ref_data['spike_clusters']
    oth_st  = other_data['spike_times'].astype(float) / fs
    oth_clu = other_data['spike_clusters']

    oth_uids = np.unique(oth_clu)
    # pre-slice other spike times by unit for speed
    oth_by_unit = {u: oth_st[oth_clu == u] for u in oth_uids}

    matches = {}
    for ref_uid in np.unique(ref_clu):
        rt = ref_st[ref_clu == ref_uid]
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
        if best_frac >= COINC_THRESH:
            matches[ref_uid] = (best_uid, best_frac)
        else:
            matches[ref_uid] = (None, 0.0)
    return matches


def build_topk_matches(ref_data, other_data, fs=FS, min_spikes=100, topk=3):
    """
    For each ref unit, return the top-k matches in other_data ranked by coincident fraction.
    Returns dict: ref_uid -> list[(oth_uid, coinc_frac)] (length <= topk)
    """
    ref_st  = ref_data['spike_times'].astype(float)   / fs
    ref_clu = ref_data['spike_clusters']
    oth_st  = other_data['spike_times'].astype(float) / fs
    oth_clu = other_data['spike_clusters']

    oth_uids = np.unique(oth_clu)
    oth_by_unit = {u: np.sort(oth_st[oth_clu == u]) for u in oth_uids}

    out = {}
    for ref_uid in np.unique(ref_clu):
        rt = np.sort(ref_st[ref_clu == ref_uid])
        if len(rt) < min_spikes:
            out[ref_uid] = []
            continue

        scores = []
        for oth_uid, ot in oth_by_unit.items():
            if len(ot) < min_spikes:
                continue
            frac = find_coincident_spikes(rt, ot) / len(rt)
            if frac > 0:
                scores.append((int(oth_uid), float(frac)))

        scores.sort(key=lambda x: x[1], reverse=True)
        out[ref_uid] = scores[:topk]
    return out


def coincident_mask(times_a, times_b, tol=COINC_TOLERANCE):
    """Boolean mask for spikes in times_a having a coincident spike in times_b."""
    if not len(times_a) or not len(times_b):
        return np.zeros(len(times_a), dtype=bool)
    ta = np.asarray(times_a, float)
    tb = np.asarray(times_b, float)
    o  = np.argsort(ta)
    ta = ta[o]
    tb = np.sort(tb)

    idx      = np.searchsorted(tb, ta)
    il       = np.clip(idx - 1, 0, len(tb) - 1)
    ir       = np.clip(idx,     0, len(tb) - 1)
    min_dist = np.minimum(np.abs(ta - tb[il]), np.abs(ta - tb[ir]))
    m_sorted = (min_dist <= tol)

    m = np.zeros(len(m_sorted), dtype=bool)
    m[o] = m_sorted
    return m


def _safe_corrcoef(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if len(x) < 2:
        return np.nan
    if np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def split_diagnostics_for_pair(ref_times_s, child1_times_s, child2_times_s, bins_s):
    """Compute time-binned tradeoff and conservation metrics for a candidate split pair."""
    rt = np.sort(np.asarray(ref_times_s, float))
    c1t = np.sort(np.asarray(child1_times_s, float))
    c2t = np.sort(np.asarray(child2_times_s, float))

    ref_counts, _ = np.histogram(rt, bins=bins_s)
    m1 = coincident_mask(rt, c1t)
    m2 = coincident_mask(rt, c2t)
    c1_counts, _ = np.histogram(rt[m1], bins=bins_s)
    c2_counts, _ = np.histogram(rt[m2], bins=bins_s)

    # Union coincidence for conservation: each ref spike counts at most once
    m_union = (m1 | m2)
    union_counts, _ = np.histogram(rt[m_union], bins=bins_s)

    s = c1_counts + c2_counts
    eps = 1e-9
    segregation = float(np.nanmean(np.abs(c1_counts - c2_counts) / (s + eps))) if len(s) else np.nan
    anticorr = _safe_corrcoef(c1_counts, c2_counts)

    nz = ref_counts > 0
    cons = (union_counts[nz] / np.maximum(ref_counts[nz], 1)) if nz.any() else np.array([])
    conservation_mean = float(np.nanmean(cons)) if len(cons) else np.nan
    conservation_std  = float(np.nanstd(cons))  if len(cons) else np.nan

    return dict(
        ref_counts=ref_counts,
        c1_counts=c1_counts,
        c2_counts=c2_counts,
        union_counts=union_counts,
        segregation=segregation,
        anticorr=anticorr,
        conservation_mean=conservation_mean,
        conservation_std=conservation_std,
    )


def fine_ccg_for_pair(times_a_s, times_b_s, window_s=FINE_CCG_WINDOW_S,
                      bin_s=FINE_CCG_BIN_S, near_zero_s=FINE_CCG_NEAR_ZERO_S):
    """Compute a fine-timescale cross-correlogram and summary metrics for a pair."""
    ta = np.sort(np.asarray(times_a_s, float))
    tb = np.sort(np.asarray(times_b_s, float))
    edges = np.arange(-window_s, window_s + bin_s, bin_s)
    centers = (edges[:-1] + edges[1:]) / 2

    if len(ta) == 0 or len(tb) == 0:
        return dict(
            bin_centers_s=centers,
            counts=np.zeros(len(centers), dtype=int),
            total_pairs=0,
            near_zero_pairs=0,
            near_zero_frac=np.nan,
            zero_peak_ratio=np.nan,
        )

    dts = []
    left = 0
    right = 0
    for t in ta:
        while left < len(tb) and tb[left] < (t - window_s):
            left += 1
        if right < left:
            right = left
        while right < len(tb) and tb[right] <= (t + window_s):
            right += 1
        if right > left:
            dts.append(tb[left:right] - t)

    if dts:
        all_dt = np.concatenate(dts)
        counts, _ = np.histogram(all_dt, bins=edges)
    else:
        counts = np.zeros(len(centers), dtype=int)

    total_pairs = int(counts.sum())
    near_zero_mask = np.abs(centers) <= near_zero_s
    near_zero_pairs = int(counts[near_zero_mask].sum()) if np.any(near_zero_mask) else 0
    near_zero_frac = (near_zero_pairs / total_pairs) if total_pairs else np.nan

    outer_mask = np.abs(centers) >= (window_s * 0.6)
    baseline = float(np.median(counts[outer_mask])) if np.any(outer_mask) else 0.0
    zero_peak = float(counts[near_zero_mask].max()) if np.any(near_zero_mask) else 0.0
    zero_peak_ratio = (zero_peak / max(baseline, 1.0)) if total_pairs else np.nan

    return dict(
        bin_centers_s=centers,
        counts=counts,
        total_pairs=total_pairs,
        near_zero_pairs=near_zero_pairs,
        near_zero_frac=near_zero_frac,
        zero_peak_ratio=zero_peak_ratio,
    )


# =============================================================================
#%% Load all run data
# =============================================================================

run_dirs = sorted(d for d in sweep_dir.iterdir() if d.is_dir() and d.name.startswith('run_'))
print(f"Found {len(run_dirs)} runs")

all_data  = {}   # run_name -> raw data dict
all_stats = {}   # run_name -> per-unit DataFrame
run_dir_map = {} # run_name -> Path
sorter_params_by_run = {}  # run_name -> sorter_params dict
for rd in run_dirs:
    name = rd.name.replace('run_', '')
    run_dir_map[name] = rd
    sorter_params_by_run[name] = load_sorter_params(rd)
    d    = load_run_data(rd)
    if d is not None:
        all_data[name]  = d
        all_stats[name] = unit_stats(d)
        print(f"  {name:20s} — {len(all_stats[name])} units")

if REF_RUN not in all_data:
    raise RuntimeError(f"Reference run '{REF_RUN}' not found. Check REF_RUN.")

run_names = list(all_data.keys())
colors    = {n: PALETTE[i % len(PALETTE)] for i, n in enumerate(run_names)}


# =============================================================================
#%% Build unit matches (all runs → reference)
# =============================================================================

print(f"\nMatching units to reference run '{REF_RUN}'...")
match_table = {}   # ref_uid -> {run_name: (matched_uid or None, coinc_frac)}
topk_table  = {}   # run_name -> {ref_uid: [(oth_uid, frac), ...]}
for name, data in all_data.items():
    if name == REF_RUN:
        continue
    print(f"  matching {name}...")
    m = build_matches(all_data[REF_RUN], data)
    topk_table[name] = build_topk_matches(all_data[REF_RUN], data, min_spikes=SPLIT_MIN_SPIKES, topk=SPLIT_TOPK)
    for ref_uid, (matched_uid, frac) in m.items():
        match_table.setdefault(ref_uid, {})[name] = (matched_uid, frac)


# =============================================================================
#%% Summary table
# =============================================================================

summ_rows = []
for name, stats in all_stats.items():
    s = run_summary(stats)
    rp = sorter_params_by_run.get(name, {})
    tracked = {k: rp.get(k, np.nan) for k in TRACKED_SORTER_PARAMS}
    summ_rows.append(dict(run=name, **s, **tracked))
summary_df = pd.DataFrame(summ_rows).set_index('run')
summary_df.to_csv(sweep_dir / out_name('sweep_summary.csv'))
print("\n--- Summary ---")
print(summary_df[['n_units','n_good','n_well','efficiency','median_mpct']].to_string())


# =============================================================================
#%% Claim-mask sweep summary (spike totals + per-unit spike distributions)
# =============================================================================

def _get_claim_params_for_run(run_name: str) -> tuple[float, float]:
    rp = sorter_params_by_run.get(run_name, {})
    ms = rp.get('cross_peel_claim_ms', np.nan)
    um = rp.get('cross_peel_claim_um', np.nan)
    try:
        ms = float(ms)
    except Exception:
        ms = np.nan
    try:
        um = float(um)
    except Exception:
        um = np.nan
    return ms, um


claim_rows = []
for rn in run_names:
    ms, um = _get_claim_params_for_run(rn)
    if not (np.isfinite(ms) or np.isfinite(um)):
        continue
    claim_rows.append({
        'run': rn,
        'cross_peel_claim_ms': ms,
        'cross_peel_claim_um': um,
        'n_units': float(summary_df.loc[rn, 'n_units']) if rn in summary_df.index else np.nan,
        'total_spikes': float(summary_df.loc[rn, 'total_spikes']) if rn in summary_df.index else np.nan,
        'spikes_per_unit_q25': float(summary_df.loc[rn, 'spikes_per_unit_q25']) if rn in summary_df.index else np.nan,
        'spikes_per_unit_median': float(summary_df.loc[rn, 'spikes_per_unit_median']) if rn in summary_df.index else np.nan,
        'spikes_per_unit_q75': float(summary_df.loc[rn, 'spikes_per_unit_q75']) if rn in summary_df.index else np.nan,
        'firing_rate_hz_median': float(summary_df.loc[rn, 'firing_rate_hz_median']) if rn in summary_df.index else np.nan,
    })

if claim_rows:
    claim_df = pd.DataFrame(claim_rows)
    claim_df = claim_df.sort_values(['cross_peel_claim_ms', 'cross_peel_claim_um', 'run'])
    claim_df.to_csv(sweep_dir / out_name('claim_sweep_spike_summary.csv'), index=False)
    print(f"Wrote claim sweep summary CSV → {sweep_dir / out_name('claim_sweep_spike_summary.csv')}")

    # Figure: total spikes (bars) + per-unit n_spikes (boxplots)
    fig = plt.figure(figsize=(7.2, 4.2))
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])

    x = np.arange(len(claim_df))
    labels = [f"{r['run']}\n{r['cross_peel_claim_ms']:.3g}ms,{r['cross_peel_claim_um']:.0f}µm" for _, r in claim_df.iterrows()]
    colors_claim = [colors.get(rn, '#666') for rn in claim_df['run'].tolist()]
    ax0.bar(x, claim_df['total_spikes'].to_numpy(dtype=float), color=colors_claim, width=0.75, edgecolor='none')
    ax0.set_xticks(x)
    ax0.set_xticklabels(labels, rotation=60, ha='right', fontsize=6)
    ax0.set_ylabel('Total spikes')
    ax0.set_title('Total detected spikes vs claim mask')

    # boxplot per-unit spike counts
    box_data = []
    for rn in claim_df['run'].tolist():
        s = all_stats.get(rn)
        if s is None or not len(s):
            box_data.append(np.array([]))
        else:
            box_data.append(s['n_spikes'].to_numpy(dtype=float))

    bp = ax1.boxplot(
        box_data,
        positions=x,
        widths=0.6,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color='#111', linewidth=1.0),
        boxprops=dict(linewidth=0.8),
        whiskerprops=dict(linewidth=0.8),
        capprops=dict(linewidth=0.8),
    )
    for patch, c in zip(bp['boxes'], colors_claim):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
        patch.set_edgecolor('none')

    ax1.set_yscale('log')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=60, ha='right', fontsize=6)
    ax1.set_ylabel('Spikes per unit (log)')
    ax1.set_title('Per-unit spike-count distribution')

    fig.tight_layout()
    fig.savefig(sweep_dir / out_name('fig_claim_sweep_spike_counts.pdf'))
    fig.savefig(sweep_dir / out_name('fig_claim_sweep_spike_counts.png'))
    plt.close(fig)
    print(f"Saved claim sweep spike-count figure → {sweep_dir / out_name('fig_claim_sweep_spike_counts.pdf')}")

    # Optional heatmaps if we have a grid of ms×um
    if claim_df['cross_peel_claim_ms'].nunique() > 1 or claim_df['cross_peel_claim_um'].nunique() > 1:
        try:
            # average in case multiple runs share the same param pair
            piv_total = claim_df.pivot_table(
                index='cross_peel_claim_um',
                columns='cross_peel_claim_ms',
                values='total_spikes',
                aggfunc='mean',
            ).sort_index(ascending=True)

            piv_med = claim_df.pivot_table(
                index='cross_peel_claim_um',
                columns='cross_peel_claim_ms',
                values='spikes_per_unit_median',
                aggfunc='mean',
            ).sort_index(ascending=True)

            def _heatmap(piv: pd.DataFrame, title: str, out_base: str, cmap: str = 'viridis'):
                fig, ax = plt.subplots(figsize=(5.4, 4.2))
                mat = piv.to_numpy(dtype=float)
                im = ax.imshow(mat, aspect='auto', origin='lower', cmap=cmap)
                ax.set_title(title)
                ax.set_xlabel('cross_peel_claim_ms')
                ax.set_ylabel('cross_peel_claim_um')
                ax.set_xticks(np.arange(piv.shape[1]))
                ax.set_xticklabels([f"{v:g}" for v in piv.columns.to_numpy(dtype=float)], rotation=45, ha='right')
                ax.set_yticks(np.arange(piv.shape[0]))
                ax.set_yticklabels([f"{v:g}" for v in piv.index.to_numpy(dtype=float)])
                fig.colorbar(im, ax=ax, shrink=0.85)
                fig.tight_layout()
                fig.savefig(sweep_dir / out_name(out_base + '.pdf'))
                fig.savefig(sweep_dir / out_name(out_base + '.png'))
                plt.close(fig)

            _heatmap(piv_total, 'Mean total spikes', 'fig_claim_heatmap_total_spikes')
            _heatmap(piv_med, 'Mean median spikes/unit', 'fig_claim_heatmap_median_spikes_per_unit')
            print(f"Saved claim heatmaps → {sweep_dir / out_name('fig_claim_heatmap_total_spikes.pdf')}")
        except Exception as e:
            print(f"[warn] could not generate claim heatmaps: {e}")



# =============================================================================
#%% Split diagnostics across runs (CSV + PDF)
# =============================================================================

print("\nComputing split diagnostics (top-2 match tradeoff + conservation)...")
ref_st_all = all_data[REF_RUN]['spike_times'].astype(float) / FS
ref_sc_all = all_data[REF_RUN]['spike_clusters']
rec_dur_s  = float(ref_st_all.max())
bins_s     = np.arange(0, rec_dur_s + SPLIT_BIN_S, SPLIT_BIN_S)

ref_by_unit = {u: np.sort(ref_st_all[ref_sc_all == u]) for u in np.unique(ref_sc_all)}

oth_by_unit_cache = {}  # run_name -> {unit_id: sorted spike times (s)}

split_rows = []
for run_name in [n for n in run_names if n != REF_RUN]:
    oth_st_all = all_data[run_name]['spike_times'].astype(float) / FS
    oth_sc_all = all_data[run_name]['spike_clusters']
    oth_by_unit = {u: np.sort(oth_st_all[oth_sc_all == u]) for u in np.unique(oth_sc_all)}
    oth_by_unit_cache[run_name] = oth_by_unit

    run_topk = topk_table.get(run_name, {})
    for ref_uid, matches in run_topk.items():
        if len(matches) < 2:
            continue
        (u1, f1), (u2, f2) = matches[0], matches[1]
        rt = ref_by_unit.get(ref_uid, np.array([]))
        if len(rt) < SPLIT_MIN_SPIKES:
            continue
        ot1 = oth_by_unit.get(u1, np.array([]))
        ot2 = oth_by_unit.get(u2, np.array([]))
        if len(ot1) == 0 or len(ot2) == 0:
            continue

        diag = split_diagnostics_for_pair(rt, ot1, ot2, bins_s)
        split_score = float(f1 + f2)

        # NaNs should fail: weak sampling should not silently pass criteria.
        anticorr_ok = np.isfinite(diag['anticorr']) and (diag['anticorr'] <= ANTICORR_THRESH)
        conservation_ok = np.isfinite(diag['conservation_mean']) and (diag['conservation_mean'] >= CONSERVATION_THRESH)
        flagged = (
            (split_score >= SPLIT_SCORE_THRESH) and
            (diag['segregation'] >= SEGREGATION_THRESH) and
            anticorr_ok and
            conservation_ok
        )

        split_rows.append(dict(
            run=run_name,
            ref_unit=int(ref_uid),
            child1_unit=int(u1), child1_frac=float(f1),
            child2_unit=int(u2), child2_frac=float(f2),
            split_score=split_score,
            segregation=float(diag['segregation']),
            anticorr=float(diag['anticorr']) if not np.isnan(diag['anticorr']) else np.nan,
            conservation_mean=float(diag['conservation_mean']),
            conservation_std=float(diag['conservation_std']),
            flagged=bool(flagged),
        ))

split_df = pd.DataFrame(split_rows)
if len(split_df):
    split_df['fine_ccg_total_pairs'] = np.nan
    split_df['fine_ccg_near_zero_pairs'] = np.nan
    split_df['fine_ccg_near_zero_frac'] = np.nan
    split_df['fine_ccg_zero_peak_ratio'] = np.nan
    split_df['duplicate_fit_candidate'] = False

    for idx, row in split_df[split_df['flagged']].iterrows():
        run_name = row['run']
        u1 = int(row['child1_unit'])
        u2 = int(row['child2_unit'])
        oth_by_unit = oth_by_unit_cache.get(run_name, {})
        ot1 = oth_by_unit.get(u1, np.array([]))
        ot2 = oth_by_unit.get(u2, np.array([]))
        ccg = fine_ccg_for_pair(ot1, ot2)
        split_df.at[idx, 'fine_ccg_total_pairs'] = int(ccg['total_pairs'])
        split_df.at[idx, 'fine_ccg_near_zero_pairs'] = int(ccg['near_zero_pairs'])
        split_df.at[idx, 'fine_ccg_near_zero_frac'] = float(ccg['near_zero_frac'])
        split_df.at[idx, 'fine_ccg_zero_peak_ratio'] = float(ccg['zero_peak_ratio'])

    # Tighten the generic split flag: require a real second child, not just a tiny residue.
    split_df['flagged'] = (
        split_df['flagged'].astype(bool) &
        (split_df['child2_frac'] >= SPLIT_CHILD2_MIN_FRAC)
    )

    split_df['duplicate_fit_candidate'] = (
        split_df['flagged'].astype(bool) &
        np.isfinite(split_df['fine_ccg_near_zero_frac']) &
        np.isfinite(split_df['fine_ccg_zero_peak_ratio']) &
        (split_df['fine_ccg_near_zero_frac'] >= DUPLICATE_NEAR_ZERO_FRAC_THRESH) &
        (split_df['fine_ccg_zero_peak_ratio'] >= DUPLICATE_ZERO_PEAK_RATIO_THRESH)
    )

    split_df = split_df.sort_values(['flagged', 'split_score', 'segregation'], ascending=[False, False, False])
    split_df.to_csv(sweep_dir / out_name('split_diagnostics.csv'), index=False)
    print(f"Wrote split diagnostics CSV → {sweep_dir / out_name('split_diagnostics.csv')}")
    print("Top flagged candidates:")
    print(split_df[split_df['flagged']].head(12).to_string(index=False))

    if split_df['duplicate_fit_candidate'].any():
        print("\nTop duplicate-fit candidates:")
        print(split_df[split_df['duplicate_fit_candidate']].head(12).to_string(index=False))

    flagged_counts = split_df[split_df['flagged']].groupby('run').size()
    summary_df['n_flagged_splits_vs_ref'] = [int(flagged_counts.get(r, 0)) for r in summary_df.index]
    duplicate_counts = split_df[split_df['duplicate_fit_candidate']].groupby('run').size()
    summary_df['n_duplicate_fit_candidates_vs_ref'] = [int(duplicate_counts.get(r, 0)) for r in summary_df.index]
    summary_df.to_csv(sweep_dir / out_name('sweep_summary.csv'))
    print("\nFlagged split candidates vs reference (by run):")
    print(summary_df[['n_flagged_splits_vs_ref']].to_string())
    print("\nDuplicate-fit candidates vs reference (by run):")
    print(summary_df[['n_duplicate_fit_candidates_vs_ref']].to_string())
else:
    print("No split diagnostics computed (no runs or no eligible units).")


if len(split_df) and split_df['flagged'].any():
    out_pdf = sweep_dir / out_name('fig_split_diagnostics.pdf')
    flagged_df = split_df[split_df['flagged']].head(MAX_SPLIT_PAGES_PDF)
    print(f"Writing split diagnostics PDF ({len(flagged_df)} pages) → {out_pdf}")

    with PdfPages(out_pdf) as pdf:
        for _, row in flagged_df.iterrows():
            run_name = row['run']
            ref_uid  = int(row['ref_unit'])
            u1       = int(row['child1_unit'])
            u2       = int(row['child2_unit'])

            rt = ref_by_unit.get(ref_uid, np.array([]))

            oth_by_unit = oth_by_unit_cache.get(run_name, {})
            if not oth_by_unit:
                oth_st_all = all_data[run_name]['spike_times'].astype(float) / FS
                oth_sc_all = all_data[run_name]['spike_clusters']
                oth_by_unit = {u: np.sort(oth_st_all[oth_sc_all == u]) for u in np.unique(oth_sc_all)}
                oth_by_unit_cache[run_name] = oth_by_unit

            ot1 = oth_by_unit.get(u1, np.array([]))
            ot2 = oth_by_unit.get(u2, np.array([]))
            if len(rt) == 0 or len(ot1) == 0 or len(ot2) == 0:
                continue

            diag = split_diagnostics_for_pair(rt, ot1, ot2, bins_s)
            ref_counts = diag['ref_counts']
            c1 = diag['c1_counts']
            c2 = diag['c2_counts']
            s  = diag['union_counts']

            t_mid_min = (bins_s[:-1] + bins_s[1:]) / 2 / 60
            frac1 = c1 / np.maximum(s, 1)

            ccg = fine_ccg_for_pair(ot1, ot2)

            fig, axes = plt.subplots(3, 1, figsize=(7.2, 5.1), sharex=False,
                                     squeeze=False,
                                     gridspec_kw=dict(hspace=0.28, height_ratios=[1.2, 1.0, 0.9]))
            ax0 = axes[0, 0]
            ax1 = axes[1, 0]
            ax2 = axes[2, 0]

            ax0.plot(t_mid_min, ref_counts, color='#333', lw=1.1, label='ref spikes/bin')
            ax0.plot(t_mid_min, s,         color='#111', lw=1.4, ls='--', label='coincident (child1∪child2)')
            ax0.plot(t_mid_min, c1,        color=PALETTE[0], lw=1.2, label=f'child1 {u1}')
            ax0.plot(t_mid_min, c2,        color=PALETTE[1], lw=1.2, label=f'child2 {u2}')
            ax0.set_ylabel(f"Counts / {SPLIT_BIN_S:.0f}s")
            ax0.legend(ncol=2, fontsize=6, loc='upper right')

            ax1.plot(t_mid_min, frac1, color='#444', lw=1.2)
            ax1.axhline(0.5, color='#aaa', lw=0.7, ls=':')
            ax1.set_ylim(0, 1)
            ax1.set_ylabel('child1 fraction')
            ax1.set_xlabel('Time (min)')

            ax2.bar(ccg['bin_centers_s'] * 1e3, ccg['counts'], width=FINE_CCG_BIN_S * 1e3,
                    color='#666', edgecolor='none')
            ax2.axvline(0, color='#aa0000', lw=0.8, ls=':')
            ax2.set_xlim(-FINE_CCG_WINDOW_S * 1e3, FINE_CCG_WINDOW_S * 1e3)
            ax2.set_xlabel('Child-child lag (ms)')
            ax2.set_ylabel('Pair count')
            ax2.set_title(
                f"Fine CCG: near-zero frac={ccg['near_zero_frac']:.3f}, peak/base={ccg['zero_peak_ratio']:.2f}",
                fontsize=7,
            )

            fig.suptitle(
                f"Split candidate: ref {ref_uid}  vs run '{run_name}'\n"
                f"top matches: {u1} ({row['child1_frac']*100:.0f}%), {u2} ({row['child2_frac']*100:.0f}%)  "
                f"score={row['split_score']:.2f}  seg={row['segregation']:.2f}  "
                f"anticorr={row['anticorr']:.2f}  cons={row['conservation_mean']:.2f}  "
                f"dupCand={bool(row.get('duplicate_fit_candidate', False))}",
                fontsize=8, y=1.02
            )

            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

    print(f"Saved split diagnostics PDF → {out_pdf}")
else:
    print("No flagged split candidates; skipping split diagnostics PDF.")


# =============================================================================
#%% FIGURE 1 — Run overview (bars + CDFs)
# =============================================================================

fig1 = plt.figure(figsize=(7.2, 4.5))
gs1  = gridspec.GridSpec(2, 4, figure=fig1, hspace=0.55, wspace=0.45)

ax_nu  = fig1.add_subplot(gs1[0, 0])
ax_ng  = fig1.add_subplot(gs1[0, 1])
ax_nw  = fig1.add_subplot(gs1[0, 2])
ax_ef  = fig1.add_subplot(gs1[0, 3])
ax_cm  = fig1.add_subplot(gs1[1, 0:2])
ax_cp  = fig1.add_subplot(gs1[1, 2:4])

x      = np.arange(len(run_names))
bcolors = [colors[n] for n in run_names]

def _bar(ax, values, ylabel, title, ylim=None):
    bars = ax.bar(x, [values[n] for n in run_names], color=bcolors,
                  width=0.6, edgecolor='none')
    # highlight reference
    ref_idx = run_names.index(REF_RUN) if REF_RUN in run_names else None
    if ref_idx is not None:
        bars[ref_idx].set_edgecolor('k')
        bars[ref_idx].set_linewidth(1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(run_names, rotation=50, ha='right', fontsize=6)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim:
        ax.set_ylim(*ylim)
    ax.yaxis.set_major_locator(MaxNLocator(4, integer=True))

_bar(ax_nu, summary_df['n_units'],   'count',    'All units')
_bar(ax_ng, summary_df['n_good'],    'count',    'Good units')
_bar(ax_nw, summary_df['n_well'],    'count',    'Well-detected')
_bar(ax_ef, summary_df['efficiency'],'fraction', 'Efficiency', ylim=(0, 1))

# CDF of mean missing %
for name, stats in all_stats.items():
    v   = np.sort(stats['mean_mpct'].dropna().values)
    cdf = np.arange(1, len(v) + 1) / len(v)
    ax_cm.plot(v, cdf, color=colors[name], lw=1.5, label=name,
               alpha=0.9, zorder=2 if name == REF_RUN else 1)
ax_cm.axvline(MPCT_THRESH, color='#555', lw=0.8, ls='--')
ax_cm.set_xlabel('Mean missing % per unit')
ax_cm.set_ylabel('CDF')
ax_cm.set_title('Amplitude truncation distribution')
ax_cm.legend(ncol=2, loc='lower right')
ax_cm.set_xlim(0, 60)

# CDF of presence fraction
for name, stats in all_stats.items():
    v   = np.sort(stats['presence_frac'].values)
    cdf = np.arange(1, len(v) + 1) / len(v)
    ax_cp.plot(v, cdf, color=colors[name], lw=1.5, label=name,
               alpha=0.9, zorder=2 if name == REF_RUN else 1)
ax_cp.axvline(PRESENCE_THRESH, color='#555', lw=0.8, ls='--')
ax_cp.set_xlabel('Presence fraction per unit')
ax_cp.set_ylabel('CDF')
ax_cp.set_title('Unit stability across recording')
ax_cp.set_xlim(0, 1)

fig1.savefig(sweep_dir / out_name('fig1_run_overview.pdf'))
fig1.savefig(sweep_dir / out_name('fig1_run_overview.png'))
print("Saved Fig 1")


# =============================================================================
#%% FIGURE 2 — Parameter sweep families
# =============================================================================

METRICS = [
    ('n_units',    'Total units',           False),
    ('n_good',     'Good-labeled units',    False),
    ('n_well',     'Well-detected units',   False),
    ('median_mpct','Median missing % (↓)',  True),
]

n_fam = len(SWEEP_FAMILY_SPECS)
n_met = len(METRICS)

fig2, axes2 = plt.subplots(n_met, n_fam, figsize=(2.4 * n_fam, 2.0 * n_met),
                            squeeze=False)
plt.subplots_adjust(hspace=0.55, wspace=0.45)

for col, (fam_label, param_key) in enumerate(SWEEP_FAMILY_SPECS):
    fam_runs  = [r for r in select_family_runs(run_names, sorter_params_by_run, param_key, default_run=REF_RUN)
                 if r in summary_df.index]
    x_vals    = [sorter_params_by_run.get(r, {}).get(param_key, np.nan) for r in fam_runs]

    for row, (met_key, met_label, lower_better) in enumerate(METRICS):
        ax = axes2[row, col]
        y_vals = [summary_df.loc[r, met_key] for r in fam_runs]

        # shaded background for better direction
        if lower_better:
            ax.set_facecolor('#fff9f0')

        ax.plot(x_vals, y_vals, 'o-', color=PALETTE[col], lw=1.5, ms=5,
                zorder=3, clip_on=False)

        # mark reference point
        if REF_RUN in fam_runs:
            xi = fam_runs.index(REF_RUN)
            ax.plot(x_vals[xi], y_vals[xi], 'o', color=PALETTE[col],
                ms=8, mec='k', mew=0.8, zorder=4, clip_on=False)
            ax.axvline(x_vals[xi], color='#aaa', lw=0.7, ls=':', zorder=1)

        ax.set_xticks(x_vals)
        ax.set_xticklabels([str(v) for v in x_vals], fontsize=6)
        ax.yaxis.set_major_locator(MaxNLocator(4, integer=(not lower_better)))

        if row == 0:
            ax.set_title(fam_label, fontsize=8)
        if col == 0:
            ax.set_ylabel(met_label, fontsize=7)
        if row == n_met - 1:
            ax.set_xlabel(param_key.replace('_', '\n'), fontsize=7)

fig2.suptitle(f'Quality metrics vs swept Kilosort4 parameters\n(filled circle = reference/{REF_RUN})',
              fontsize=8, y=1.01)
fig2.savefig(sweep_dir / out_name('fig2_param_sweep.pdf'))
fig2.savefig(sweep_dir / out_name('fig2_param_sweep.png'))
print("Saved Fig 2")


# =============================================================================
#%% FIGURE 3 — Per-unit PDF (one page per matched unit)
# =============================================================================

def _amp_raster(ax, u_times_s, u_amps, valid_blocks, pres_cid_arr, uid,
                rec_dur_s, title, color):
    """
    2D histogram of amplitude vs time for one unit, with presence blocks shaded.
    """
    if u_amps is not None and len(u_amps) == len(u_times_s):
        h = ax.hist2d(u_times_s / 60, u_amps,
                      bins=[min(200, max(50, len(u_times_s)//50)), 40],
                      cmap='Blues', rasterized=True)
    else:
        ax.plot(u_times_s / 60, np.ones_like(u_times_s), '|',
                color=color, ms=1, alpha=0.3, rasterized=True)

    # shade valid presence blocks
    pm = (pres_cid_arr == uid) if len(pres_cid_arr) else np.array([], bool)
    if pm.any():
        pass  # valid shading needs spike-indexed times — skip for brevity

    ax.set_xlim(0, rec_dur_s / 60)
    ax.set_ylabel('Amplitude (a.u.)', fontsize=6)
    ax.set_title(title, fontsize=7, pad=2, color=color)
    ax.tick_params(labelbottom=False)


def _mpct_trace(ax, trunc_data, spike_times_s, spike_clusters, uid,
                rec_dur_s, color, last_row=False):
    """
    Bar chart of missing % per 1k-spike window for one unit.
    """
    if trunc_data is None:
        ax.text(0.5, 0.5, 'no QC data', transform=ax.transAxes,
                ha='center', va='center', fontsize=6, color='#999')
        return

    tcid  = trunc_data['cid'].astype(int)
    tblk  = trunc_data['window_blocks']
    tmpct = trunc_data['mpcts']
    tm    = (tcid == uid)
    if not tm.any():
        ax.text(0.5, 0.5, 'not enough spikes', transform=ax.transAxes,
                ha='center', va='center', fontsize=6, color='#999')
        return

    u_st = spike_times_s[spike_clusters == uid]
    centres, mpcts = [], []
    for (i0, i1), mp in zip(tblk[tm], tmpct[tm]):
        i0c = min(int(i0), len(u_st) - 1)
        i1c = min(int(i1), len(u_st) - 1)
        centres.append((u_st[i0c] + u_st[i1c]) / 2 / 60)
        mpcts.append(float(mp))

    widths = np.diff(np.concatenate([[0], centres, [rec_dur_s / 60]]))
    w      = np.minimum(widths[:-1], widths[1:]) * 0.9
    ax.bar(centres, mpcts, width=w, color=color, alpha=0.7, edgecolor='none')
    ax.axhline(MPCT_THRESH, color='#555', lw=0.7, ls='--')
    ax.set_ylim(0, 55)
    ax.set_xlim(0, rec_dur_s / 60)
    ax.set_yticks([0, 20, 40])
    ax.set_ylabel('Missing %', fontsize=6)
    if last_row:
        ax.set_xlabel('Time (min)', fontsize=7)
    else:
        ax.tick_params(labelbottom=False)


# Decide which units to include in the PDF:
# must appear in ref run AND match at least one other run
ref_uid_order = sorted(match_table.keys())
pdf_units = [uid for uid in ref_uid_order
             if any(mu is not None for mu, _ in match_table[uid].values())]
print(f"\n{len(pdf_units)} reference units matched to ≥1 other run → writing per-unit PDF...")

# Sort by depth if stage-matched spike_positions are available, otherwise by spike count
ref_sorter_out, _ = get_stage_paths(sweep_dir / f'run_{REF_RUN}')
pos_file = ref_sorter_out / 'spike_positions.npy'
if pos_file.exists():
    sp = np.load(pos_file)
    sc = all_data[REF_RUN]['spike_clusters']
    if len(sp) == len(sc):
        unit_depths = {uid: float(np.median(sp[sc == uid, 1])) for uid in pdf_units}
        pdf_units = sorted(pdf_units, key=lambda u: -unit_depths.get(u, 0))
    else:
        unit_depths = {}
        ref_stats = all_stats[REF_RUN].set_index('unit_id')
        pdf_units = sorted(pdf_units,
                           key=lambda u: -ref_stats.loc[u, 'n_spikes'] if u in ref_stats.index else 0)
else:
    unit_depths = {}
    ref_stats = all_stats[REF_RUN].set_index('unit_id')
    pdf_units = sorted(pdf_units,
                       key=lambda u: -ref_stats.loc[u, 'n_spikes'] if u in ref_stats.index else 0)

non_ref_runs = [n for n in run_names if n != REF_RUN]

with PdfPages(sweep_dir / out_name('fig3_per_unit.pdf')) as pdf:
    for ref_uid in tqdm(pdf_units, desc='Per-unit PDF'):
        matches = match_table[ref_uid]  # {run_name: (uid_or_None, frac)}
        # rows: reference + each other run that has a match
        display_rows = [(REF_RUN, ref_uid, 1.0)]
        for rn in non_ref_runs:
            mu, frac = matches.get(rn, (None, 0.0))
            if mu is not None:
                display_rows.append((rn, mu, frac))

        n_rows     = len(display_rows)
        fig_h      = max(3.5, n_rows * 2.2)
        fig, axes  = plt.subplots(n_rows, 2,
                                  figsize=(7.2, fig_h),
                                  squeeze=False,
                                  gridspec_kw=dict(hspace=0.15, wspace=0.35))

        ref_s   = all_stats[REF_RUN].set_index('unit_id')
        n_sp    = int(ref_s.loc[ref_uid, 'n_spikes']) if ref_uid in ref_s.index else 0
        depth_s = f" @ {unit_depths[ref_uid]:.0f} µm" if ref_uid in unit_depths else ""
        ref_lbl = ref_s.loc[ref_uid, 'label'] if ref_uid in ref_s.index else '?'
        fig.suptitle(
            f"Unit {ref_uid}{depth_s}   [{ref_lbl}]   {n_sp:,} spikes (ref run)",
            fontsize=9, y=1.0
        )

        for row_i, (rname, uid, coinc_frac) in enumerate(display_rows):
            d       = all_data[rname]
            st_s    = d['spike_times'].astype(float) / FS
            sc      = d['spike_clusters']
            amps    = d['spike_amps']
            trunc   = d['trunc']
            pres    = d['pres']
            rec_dur = float(st_s.max())
            col     = colors[rname]

            u_st    = st_s[sc == uid]
            u_amps  = amps[sc == uid] if amps is not None else None
            pres_cid_arr = pres['cid'].astype(int) if pres is not None else np.array([], int)

            coinc_str = '' if rname == REF_RUN else f'  coinc={coinc_frac*100:.0f}%'
            row_title = f"{rname}{coinc_str}   unit {uid}   n={len(u_st):,}"
            is_last   = (row_i == n_rows - 1)

            _amp_raster(axes[row_i, 0], u_st, u_amps, None, pres_cid_arr,
                        uid, rec_dur, row_title, col)
            _mpct_trace(axes[row_i, 1], trunc, st_s, sc,
                        uid, rec_dur, col, last_row=is_last)

        axes[-1, 0].set_xlabel('Time (min)', fontsize=7)
        axes[-1, 0].tick_params(labelbottom=True)

        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

print(f"Saved per-unit PDF → {sweep_dir / out_name('fig3_per_unit.pdf')}")


# =============================================================================
#%% Figure 4 — Within-run duplicate screen (peeling hypothesis test)
# =============================================================================
# Tests the peeling duplication hypothesis directly: if extra peeling passes
# create near-zero-lag duplicate detections assigned to different templates,
# nearby mua-unit pairs in default (max_peels=100) should show a strong
# near-zero-lag cross-CCG peak that is absent in peel1 (max_peels=1).
# =============================================================================

print("\nRunning within-run duplicate screen...")
screen_results = {}
for rn in WITHIN_RUN_RUNS:
    rd = run_dir_map.get(rn)
    if rd is None:
        print(f"  {rn}: run directory not found, skipping")
        continue
    rows = within_run_duplicate_screen(rd)
    if rows is None:
        print(f"  {rn}: sorter_output not found, skipping")
        continue
    screen_results[rn] = rows
    nz_fracs = [r['near_zero_frac'] for r in rows if np.isfinite(r['near_zero_frac'])]
    n_flag = sum(1 for r in rows
                 if np.isfinite(r['near_zero_frac'])
                 and r['near_zero_frac'] >= DUPLICATE_NEAR_ZERO_FRAC_THRESH
                 and np.isfinite(r['zero_peak_ratio'])
                 and r['zero_peak_ratio'] >= DUPLICATE_ZERO_PEAK_RATIO_THRESH)
    print(f"  {rn}: {len(rows)} nearby pairs, "
          f"median near_zero_frac={np.nanmedian(nz_fracs):.3f}, "
          f"{n_flag} pairs exceed duplicate thresholds")

if len(screen_results) >= 1:
    out_pdf_4 = sweep_dir / out_name('fig4_within_run_ccg.pdf')
    print(f"Writing within-run CCG figure → {out_pdf_4}")

    with PdfPages(out_pdf_4) as pdf:
        # --- Page 1: Summary distributions ---
        n_panels = len(screen_results)
        fig, axes = plt.subplots(2, n_panels,
                                 figsize=(3.5 * n_panels, 5.0),
                                 squeeze=False)
        fig.suptitle('Within-run nearby-pair CCG summary\n'
                     '(pre-curation KS4 output; pairs within '
                     f'{WITHIN_RUN_NEARBY_DEPTH_UM:.0f} µm depth)',
                     fontsize=8)

        for col_i, (rn, rows) in enumerate(screen_results.items()):
            nzf   = np.array([r['near_zero_frac']  for r in rows], float)
            zpr   = np.array([r['zero_peak_ratio'] for r in rows], float)
            valid = np.isfinite(nzf) & np.isfinite(zpr)
            color = colors.get(rn, PALETTE[col_i % len(PALETTE)])

            ax0 = axes[0, col_i]
            bins_nzf = np.linspace(0, min(0.5, np.nanmax(nzf[valid]) + 0.01), 30) if valid.any() else 30
            ax0.hist(nzf[valid], bins=bins_nzf, color=color, alpha=0.8, edgecolor='none')
            ax0.axvline(DUPLICATE_NEAR_ZERO_FRAC_THRESH, color='red', lw=0.8, ls='--',
                        label=f'thresh {DUPLICATE_NEAR_ZERO_FRAC_THRESH}')
            ax0.set_xlabel('Near-zero-lag fraction', fontsize=7)
            ax0.set_ylabel('Pair count', fontsize=7)
            ax0.set_title(rn, fontsize=8)
            ax0.legend(fontsize=6)

            ax1 = axes[1, col_i]
            bins_zpr = np.linspace(0, min(10, np.nanmax(zpr[valid]) + 0.5), 30) if valid.any() else 30
            ax1.hist(zpr[valid], bins=bins_zpr, color=color, alpha=0.8, edgecolor='none')
            ax1.axvline(DUPLICATE_ZERO_PEAK_RATIO_THRESH, color='red', lw=0.8, ls='--',
                        label=f'thresh {DUPLICATE_ZERO_PEAK_RATIO_THRESH}')
            ax1.set_xlabel('Zero-lag peak / baseline ratio', fontsize=7)
            ax1.set_ylabel('Pair count', fontsize=7)
            ax1.legend(fontsize=6)

        plt.tight_layout(rect=(0, 0, 1, 0.92))
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # --- Page 2+: Scatter near_zero_frac vs zero_peak_ratio with pair labels ---
        for col_i, (rn, rows) in enumerate(screen_results.items()):
            nzf = np.array([r['near_zero_frac']  for r in rows], float)
            zpr = np.array([r['zero_peak_ratio'] for r in rows], float)
            la  = [r['label_a'] for r in rows]
            lb  = [r['label_b'] for r in rows]
            valid = np.isfinite(nzf) & np.isfinite(zpr)
            color = colors.get(rn, PALETTE[col_i % len(PALETTE)])

            fig, ax = plt.subplots(figsize=(5.5, 4.5))
            # colour by pair type
            for pair_type, marker, fc in [('mua×mua', 'o', color),
                                           ('good×mua', 's', '#aaa'),
                                           ('good×good', '^', '#ddd')]:
                if pair_type == 'mua×mua':
                    mask = valid & np.array([a != 'good' and b != 'good' for a, b in zip(la, lb)], dtype=bool)
                elif pair_type == 'good×mua':
                    mask = valid & np.array([(a == 'good') != (b == 'good') for a, b in zip(la, lb)], dtype=bool)
                else:
                    mask = valid & np.array([a == 'good' and b == 'good' for a, b in zip(la, lb)], dtype=bool)
                if mask.any():
                    ax.scatter(nzf[mask], zpr[mask], s=18, marker=marker,
                               color=fc, alpha=0.7, label=f'{pair_type} (n={mask.sum()})',
                               linewidths=0)

            ax.axvline(DUPLICATE_NEAR_ZERO_FRAC_THRESH, color='red', lw=0.8, ls='--')
            ax.axhline(DUPLICATE_ZERO_PEAK_RATIO_THRESH, color='red', lw=0.8, ls='--')
            ax.set_xlabel('Near-zero-lag fraction', fontsize=8)
            ax.set_ylabel('Zero-lag peak / baseline', fontsize=8)
            ax.set_title(f'{rn} — within-run nearby-pair CCG', fontsize=8)
            ax.legend(fontsize=7)
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

        # --- CCG trace pages for top pairs (highest near_zero_frac) per run ---
        for rn, rows in screen_results.items():
            valid_rows = [r for r in rows
                          if np.isfinite(r['near_zero_frac']) and r['total_pairs'] > 0]
            top_rows = sorted(valid_rows, key=lambda r: r['near_zero_frac'], reverse=True)
            top_rows = top_rows[:WITHIN_RUN_TOP_PAIRS_PDF]
            if not top_rows:
                continue

            cols_per_page = 4
            rows_per_page = 5
            pairs_per_page = cols_per_page * rows_per_page
            n_pages = int(np.ceil(len(top_rows) / pairs_per_page))

            for page_i in range(n_pages):
                batch = top_rows[page_i * pairs_per_page: (page_i + 1) * pairs_per_page]
                fig, axes_grid = plt.subplots(rows_per_page, cols_per_page,
                                              figsize=(10.0, 8.5),
                                              squeeze=False)
                fig.suptitle(
                    f'{rn} — top nearby-pair CCGs ranked by near-zero-lag fraction '
                    f'(page {page_i+1}/{n_pages})',
                    fontsize=8)
                color = colors.get(rn, PALETTE[0])

                for panel_i, row in enumerate(batch):
                    ax = axes_grid[panel_i // cols_per_page][panel_i % cols_per_page]
                    bc  = row['_bin_centers_s'] * 1e3  # ms
                    cnt = row['_counts']
                    ax.bar(bc, cnt, width=(bc[1] - bc[0]) if len(bc) > 1 else 0.2,
                           color=color, alpha=0.8, edgecolor='none')
                    ax.axvline(0, color='red', lw=0.6, ls='--')
                    nzf_v = row['near_zero_frac']
                    zpr_v = row['zero_peak_ratio']
                    ax.set_title(
                        f"u{row['unit_a']}({row['label_a']})×u{row['unit_b']}({row['label_b']})\n"
                        f"Δdepth={row['depth_diff']:.0f}µm  "
                        f"nzf={nzf_v:.3f}  zpr={zpr_v:.2f}",
                        fontsize=5.5)
                    ax.set_xlabel('Lag (ms)', fontsize=5)
                    ax.tick_params(labelsize=5)

                # blank unused panels
                for panel_i in range(len(batch), rows_per_page * cols_per_page):
                    axes_grid[panel_i // cols_per_page][panel_i % cols_per_page].set_visible(False)

                plt.tight_layout(rect=(0, 0, 1, 0.95))
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)

    # Save pair-level CSV for default and peel1
    for rn, rows in screen_results.items():
        csv_rows = [{k: v for k, v in r.items() if not k.startswith('_')} for r in rows]
        csv_df = pd.DataFrame(csv_rows)
        csv_path = sweep_dir / out_name(f'within_run_screen_{rn}.csv')
        csv_df.to_csv(csv_path, index=False)
        print(f"  Wrote {csv_path}")

print("\nDone. Outputs:")
for f in ['fig1_run_overview.pdf', 'fig2_param_sweep.pdf', 'fig3_per_unit.pdf',
          'fig4_within_run_ccg.pdf', 'sweep_summary.csv']:
    print(f"  {sweep_dir / out_name(f)}")
