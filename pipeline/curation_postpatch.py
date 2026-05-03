#%%
# Post-sorting curation: duplicate spike removal, cosine-template merge,
# redundant unit removal, and re-export to Phy/KS4 format.
#
# Recommended entry point
#   run_cur_final    — cosine template merge → cur_output/   ← USE THIS
#
# Available strategies (kept for comparison / research use)
#   run_cur_cosine   — Wall template cosine similarity + CCG merge
#   run_cur_no_merge — no merge; dup-spike removal + redundant unit removal only
#
# Retired strategies (empirically shown to produce false positives at probe scale;
# see curation_comparison_sweep.py for the full diagnostic evidence)
#   run_cur          — feature-projection + CCG merge (posthoc)
#   run_cur_amp_bic  — amplitude BIC (1- vs 2-Gaussian) + CCG merge

import shutil
import numpy as np
from pathlib import Path
from kilosort.run_kilosort import save_sorting
from kilosort.io import load_ops
from spikeinterface.postprocessing.correlograms import correlogram_for_one_segment
import copy
import time
import torch


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _to_numpy_cpu(x, dtype=None):
    """Convert torch tensors (including CUDA) or arraylikes into CPU NumPy."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    else:
        x = np.asarray(x)
    if dtype is not None:
        x = x.astype(dtype)
    return x

def _resolve_merge_groups(raw_merge_groups):
    """Resolve a list of merge groups into connected components."""
    groups = [list(g) for g in (raw_merge_groups or []) if g is not None and len(g) > 1]
    if not groups:
        return []

    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for g in groups:
        for u in g[1:]:
            union(g[0], u)

    comps = {}
    for g in groups:
        for u in g:
            comps.setdefault(find(u), set()).add(u)

    return [sorted(list(s)) for s in comps.values() if len(s) > 1]


def remove_duped_spikes(sorter, duped_spikes):
    cleaned_sorter = copy.deepcopy(sorter)
    duped_spikes = np.asarray(duped_spikes, dtype=int)
    len0 = len(cleaned_sorter.spikes)
    cleaned_sorter.spikes = np.delete(cleaned_sorter.spikes, duped_spikes, axis=0)
    print(f'{len(cleaned_sorter.spikes)} remaining of {len0} total spikes')
    return cleaned_sorter


# ---------------------------------------------------------------------------
# Feature-space merge helpers  (port of merge_posthoc3.m + merging_score.m)
# ---------------------------------------------------------------------------

def _get_feature_col(iCC_arr, peak_ch_from, target_chan):
    """
    Return the tF column index j such that iCC_arr[j, peak_ch_from] == target_chan.
    Returns None if target_chan is not a neighbour of peak_ch_from.
    iCC_arr[0, c] == c always (self-channel is nearest), so j=0 is the peak-channel projection.
    """
    hits = np.where(iCC_arr[:, peak_ch_from] == target_chan)[0]
    return int(hits[0]) if len(hits) else None


def _merging_score(fold, fnew, fracse=0.1):
    """
    Port of MATLAB merging_score (Pachitariu / Kilosort).
    Counts valley bins between 0 and the distribution mean; score < 3 → merge.

    fold = seed_self_proj − seed_cross_proj  (≥ 0)
    fnew = cand_cross_proj − cand_self_proj  (≤ 0)
    """
    if len(fold) < 10 or len(fnew) < 10:
        return np.inf

    l2   = float(np.max(fold))
    l1   = float(np.min(fnew))
    se   = (float(np.std(fold)) + float(np.std(fnew))) / 2
    se25 = fracse * se

    if se25 <= 0 or l2 <= 0 or l1 >= 0:
        return np.inf   # degenerate projection — don't merge without evidence

    b1 = np.arange(0, l2 + se25, se25)
    b2 = np.arange(0, -l1 + se25, se25)
    if len(b1) < 2 or len(b2) < 2:
        return np.inf   # histogram has no interior bins — don't merge

    hs1 = np.histogram(fold,  bins=b1)[0].astype(float)
    hs2 = np.histogram(-fnew, bins=b2)[0].astype(float)

    def _smooth3(h):
        return np.convolve(h, np.ones(3) / 3, mode='same') if len(h) >= 3 else h

    hs1 = _smooth3(hs1)
    hs2 = _smooth3(hs2)

    if hs1.size == 0 or hs2.size == 0:
        return 0

    mmax   = min(np.max(hs1), np.max(hs2))
    trough = mmax / 3.0

    m1 = max(0, min(int(np.ceil(np.mean(fold)  / se25)), len(hs1)))
    m2 = max(0, min(int(-np.ceil(np.mean(fnew) / se25)), len(hs2)))

    return int(np.sum(hs1[:m1] < trough) + np.sum(hs2[:m2] < trough))


def _ccg_similar(st_a, st_b, fs=30000, nlags_ms=100, binsize_ms=2, ccg_thresh=0.5):
    """
    CCG similarity check (merge_posthoc3.m port).
    Returns True if the mean pairwise Pearson correlation of unit-normed
    ACG(a), ACG(b), CCG(a,b) exceeds ccg_thresh.
    """
    if len(st_a) < 100 or len(st_b) < 100:
        return False

    st_a = np.asarray(st_a, dtype=np.int64)
    st_b = np.asarray(st_b, dtype=np.int64)

    bin_samp = max(1, int(round(binsize_ms * fs / 1000)))
    # Our legacy `nlags_ms` corresponds to a +/- window, so total window is 2*nlags_ms.
    window_samp = max(1, int(round(2 * nlags_ms * fs / 1000)))

    # Build a 2-unit combined spike train (sorted by time) for the correlogram helper.
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

    # corr shape: (2, 2, num_bins) with lags centered at `center`.
    center = corr.shape[2] // 2
    xc1 = corr[0, 0, :].astype(float)
    xc2 = corr[1, 1, :].astype(float)
    # Symmetrize cross-correlogram (direction can flip sign conventions).
    xc3 = 0.5 * (corr[0, 1, :].astype(float) + corr[1, 0, ::-1].astype(float))
    xc1[center] = 0
    xc2[center] = 0

    def _s5(x):
        return np.convolve(x, np.ones(5) / 5, mode='same')

    xc1, xc2, xc3 = _s5(xc1), _s5(xc2), _s5(xc3)

    def _nfun(x):
        n = np.linalg.norm(x)
        return x / n if n > 0 else x

    xc1, xc2, xc3 = _nfun(xc1), _nfun(xc2), _nfun(xc3)
    corr = np.corrcoef(np.stack([xc1, xc2, xc3]))
    return float(np.mean([corr[0, 1], corr[0, 2], corr[1, 2]])) > ccg_thresh


# ---------------------------------------------------------------------------
# Helpers for cosine / BIC strategies
# ---------------------------------------------------------------------------

def _cosine_sim(w1, w2):
    """Amplitude-invariant shape similarity between two waveform arrays."""
    a, b = w1.flatten(), w2.flatten()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def _wall_peak_channels(Wall):
    """
    Peak channel index per cluster from Wall (n_clusters, n_chan, n_tp).
    Wall is indexed by cluster ID (row i → cluster i).
    """
    rms = np.sqrt(np.mean(Wall ** 2, axis=2))   # (n_clusters, n_chan)
    return np.argmax(rms, axis=1)               # (n_clusters,)


def _fit_gmm_bic(data, n_components, n_init=3):
    """Fit a Gaussian mixture to 1-D data and return BIC."""
    from sklearn.mixture import GaussianMixture
    gm = GaussianMixture(n_components=n_components, n_init=n_init, random_state=0)
    gm.fit(data.reshape(-1, 1))
    return gm.bic(data.reshape(-1, 1))


# ---------------------------------------------------------------------------
# Merge strategies
# ---------------------------------------------------------------------------

def _posthoc_merge(
    tF, clu, st_times, spike_templates, iCC, iU,
    fs=30000, fracse=0.1, score_thresh=3, ccg_thresh=0.5,
    min_spikes_seed=500, min_spikes_pair=100,
    spike_z=None, max_depth_um=None,
):
    """
    Port of merge_posthoc3.m. Greedy two-stage merge operating on KS4 feature
    projections (tF).

    Stage 1 — merging_score < score_thresh (unimodal projection test)
    Stage 2 — CCG similarity > ccg_thresh

    tF              : (n_spikes, nC, n_pcs) float32
    clu             : (n_spikes,) int — post-KS cluster IDs
    spike_templates : (n_spikes,) int — original KS4 template IDs (pre-merge)
    iCC             : (nC, Nchan) int — nearest channel indices per channel
    iU              : (n_templates,) int — peak channel per template
    """
    iCC_arr = np.asarray(iCC, dtype=int)
    iU_arr  = np.asarray(iU,  dtype=int)

    unique_clus = np.unique(clu)

    clu_to_peak = {}
    for c in unique_clus:
        tmps = spike_templates[clu == c].astype(int)
        dom  = int(np.bincount(np.clip(tmps, 0, len(iU_arr) - 1)).argmax())
        clu_to_peak[c] = int(iU_arr[dom]) if dom < len(iU_arr) else -1

    spikes_of = {c: np.flatnonzero(clu == c) for c in unique_clus}
    nbins     = {c: len(v) for c, v in spikes_of.items()}
    times_of  = {c: st_times[idx].astype(np.int64, copy=False) for c, idx in spikes_of.items()}

    chan_neighbors = {}
    for c in unique_clus:
        pk = clu_to_peak[c]
        if pk < 0:
            chan_neighbors[c] = set()
            continue
        hood = set(iCC_arr[:, pk].tolist())
        chan_neighbors[c] = {
            c2 for c2 in unique_clus
            if c2 != c and clu_to_peak.get(c2, -1) in hood
        }

    if spike_z is not None and max_depth_um is not None:
        unit_depth = {c: float(np.median(spike_z[clu == c])) for c in unique_clus}
        chan_neighbors = {
            c: {c2 for c2 in hood if abs(unit_depth[c] - unit_depth[c2]) <= max_depth_um}
            for c, hood in chan_neighbors.items()
        }
        print(f'  [posthoc] depth gate {max_depth_um:.0f} µm applied')

    picked        = set()
    mega_clusters = []

    # Progress counters
    t0 = time.time()
    n_seed = 0
    n_pair_checked = 0
    n_ccg_checked = 0
    n_merged = 0

    for seed in sorted(unique_clus, key=lambda c: nbins[c], reverse=True):
        if seed in picked:
            continue
        if nbins[seed] < min_spikes_seed:
            break

        n_seed += 1
        remaining_eligible = sum(
            1 for c in unique_clus if c not in picked and nbins.get(c, 0) >= min_spikes_pair
        )
        print(
            f'    [posthoc] seed {n_seed}: unit={seed} spikes={nbins[seed]} '
            f'neighbors={len(chan_neighbors.get(seed, ()))} remaining={remaining_eligible}'
        )

        picked.add(seed)
        run_list  = [seed]
        pair_list = list(chan_neighbors.get(seed, ()))

        while pair_list:
            valid = [(p, nbins[p]) for p in pair_list
                     if p not in picked and nbins.get(p, 0) >= min_spikes_pair]
            if not valid:
                break
            ipair     = max(valid, key=lambda x: x[1])[0]
            pair_list = [p for p in pair_list if p != ipair]
            n_pair_checked += 1

            if n_pair_checked % 50 == 0:
                dt = time.time() - t0
                print(
                    f'    [posthoc] seeds={n_seed} checked={n_pair_checked} '
                    f'ccg={n_ccg_checked} merges={n_merged} elapsed={dt/60:.1f} min'
                )

            ipair_hood = chan_neighbors.get(ipair, set())
            mutual     = [r for r in run_list if r in ipair_hood]
            if not mutual:
                continue

            pk_ipair = clu_to_peak.get(ipair, -1)
            if pk_ipair < 0:
                continue
            new_idx = spikes_of[ipair]

            f1old_parts, f2old_parts = [], []
            for r in run_list:
                pk_r = clu_to_peak.get(r, -1)
                if pk_r < 0:
                    continue
                j = _get_feature_col(iCC_arr, pk_r, pk_ipair)
                if j is None:
                    continue
                idx_r = spikes_of[r]
                f1old_parts.append(tF[idx_r, 0, 0])
                f2old_parts.append(tF[idx_r, j, 0])

            if not f1old_parts:
                continue

            fold = np.concatenate(f1old_parts) - np.concatenate(f2old_parts)

            f1new_cols = []
            for r in mutual:
                pk_r = clu_to_peak.get(r, -1)
                if pk_r < 0:
                    continue
                j = _get_feature_col(iCC_arr, pk_ipair, pk_r)
                if j is None:
                    continue
                f1new_cols.append(tF[new_idx, j, 0])

            if not f1new_cols:
                continue

            f1new = (np.max(np.stack(f1new_cols, axis=1), axis=1)
                     if len(f1new_cols) > 1 else f1new_cols[0])
            fnew  = f1new - tF[new_idx, 0, 0]

            if _merging_score(fold, fnew, fracse) >= score_thresh:
                continue

            run_times = np.concatenate([times_of[r] for r in run_list])
            n_ccg_checked += 1
            if not _ccg_similar(run_times.astype(int), st_times[new_idx].astype(int),
                                 fs=fs, ccg_thresh=ccg_thresh):
                continue

            print(f'  Merging cluster {ipair} into run [{run_list[0]}]')
            run_list.append(ipair)
            picked.add(ipair)
            n_merged += 1

            if nbins[ipair] > 300:
                pair_list.extend(
                    p for p in chan_neighbors.get(ipair, ())
                    if p not in picked and p not in pair_list
                )

        mega_clusters.append(run_list)

    return [g for g in mega_clusters if len(g) > 1]


def _template_cosine_merge(
    Wall, clu, st_times, iCC,
    cosine_thresh=0.90, ccg_thresh=0.5,
    min_spikes_seed=500, min_spikes_pair=100, fs=30000,
    spike_z=None, max_depth_um=None,
):
    """
    Greedy two-stage merge based on Wall template cosine similarity + CCG.
    Does not require tF or spike_templates.

    Wall : (N_original_clusters, n_chan, n_tp) — indexed by cluster ID.
    clu  : (n_spikes,) — cluster IDs (may skip values after dup removal).

    Stage 1 — cosine_sim(Wall[seed], Wall[cand]) > cosine_thresh
    Stage 2 — CCG similarity > ccg_thresh
    """
    iCC_arr     = np.asarray(iCC, dtype=int)
    unique_clus = np.unique(clu)
    peak_chs    = _wall_peak_channels(Wall)   # Wall row i == cluster i → peak channel

    # Pre-normalise Wall templates once so each cosine check is a single dot product
    _wall_flat   = Wall.reshape(len(Wall), -1).astype(float)
    _wall_norms  = np.linalg.norm(_wall_flat, axis=1, keepdims=True)
    wall_normed  = _wall_flat / np.where(_wall_norms > 1e-10, _wall_norms, 1.0)

    spikes_of = {c: np.flatnonzero(clu == c) for c in unique_clus}
    nbins     = {c: len(v) for c, v in spikes_of.items()}
    times_of  = {c: st_times[idx].astype(np.int64, copy=False) for c, idx in spikes_of.items()}

    chan_neighbors = {}
    for c in unique_clus:
        pk   = peak_chs[c]
        hood = set(iCC_arr[:, pk].tolist())
        chan_neighbors[c] = {
            c2 for c2 in unique_clus
            if c2 != c and peak_chs[c2] in hood
        }

    if spike_z is not None and max_depth_um is not None:
        unit_depth = {c: float(np.median(spike_z[clu == c])) for c in unique_clus}
        chan_neighbors = {
            c: {c2 for c2 in hood if abs(unit_depth[c] - unit_depth[c2]) <= max_depth_um}
            for c, hood in chan_neighbors.items()
        }
        print(f'  [cosine] depth gate {max_depth_um:.0f} µm applied')

    picked        = set()
    mega_clusters = []

    t0 = time.time()
    n_seed = 0
    n_pair_checked = 0
    n_ccg_checked = 0
    n_merged = 0

    for seed in sorted(unique_clus, key=lambda c: nbins[c], reverse=True):
        if seed in picked:
            continue
        if nbins[seed] < min_spikes_seed:
            break

        n_seed += 1
        remaining_eligible = sum(
            1 for c in unique_clus if c not in picked and nbins.get(c, 0) >= min_spikes_pair
        )
        print(
            f'    [cosine] seed {n_seed}: unit={seed} spikes={nbins[seed]} '
            f'neighbors={len(chan_neighbors.get(seed, ()))} remaining={remaining_eligible}'
        )

        picked.add(seed)
        run_list  = [seed]
        seed_w    = wall_normed[seed]   # cached normed seed vector
        pair_list = list(chan_neighbors.get(seed, ()))

        while pair_list:
            valid = [(p, nbins[p]) for p in pair_list
                     if p not in picked and nbins.get(p, 0) >= min_spikes_pair]
            if not valid:
                break
            ipair     = max(valid, key=lambda x: x[1])[0]
            pair_list = [p for p in pair_list if p != ipair]
            n_pair_checked += 1

            if n_pair_checked % 50 == 0:
                dt = time.time() - t0
                print(
                    f'    [cosine] seeds={n_seed} checked={n_pair_checked} '
                    f'ccg={n_ccg_checked} merges={n_merged} elapsed={dt/60:.1f} min'
                )

            ipair_hood = chan_neighbors.get(ipair, set())
            if not any(r in ipair_hood for r in run_list):
                continue

            # Stage 1: cosine similarity — dot product on pre-normalised vectors
            sim = float(seed_w @ wall_normed[ipair])
            if sim < cosine_thresh:
                continue

            # Stage 2: CCG
            run_times = np.concatenate([times_of[r] for r in run_list])
            n_ccg_checked += 1
            if not _ccg_similar(run_times.astype(int),
                                 st_times[spikes_of[ipair]].astype(int),
                                 fs=fs, ccg_thresh=ccg_thresh):
                continue

            print(f'  [cosine] Merging {ipair} → [{run_list[0]}]  cos={sim:.3f}')
            run_list.append(ipair)
            picked.add(ipair)
            n_merged += 1

            if nbins[ipair] > 300:
                pair_list.extend(
                    p for p in chan_neighbors.get(ipair, ())
                    if p not in picked and p not in pair_list
                )

        mega_clusters.append(run_list)

    return [g for g in mega_clusters if len(g) > 1]


def _amplitude_bic_merge(
    amplitudes, clu, st_times, iCC, Wall,
    bic_margin=0.0, ccg_thresh=0.5,
    min_spikes_seed=100, min_spikes_pair=100, fs=30000,
    spike_z=None, max_depth_um=None,
):
    """
    Greedy two-stage merge based on amplitude BIC model selection + CCG.

    For each candidate pair, fit a 1-Gaussian and 2-Gaussian mixture to the
    combined amplitude distribution.  BIC(1-Gaussian) ≤ BIC(2-Gaussian) +
    bic_margin means one population → check CCG → merge.

    amplitudes : (n_spikes,) float — spike amplitudes (post dup-removal)
    Wall       : (N_original_clusters, n_chan, n_tp) — used for channel neighbourhoods
    """
    iCC_arr     = np.asarray(iCC, dtype=int)
    unique_clus = np.unique(clu)
    peak_chs    = _wall_peak_channels(Wall)

    spikes_of = {c: np.flatnonzero(clu == c) for c in unique_clus}
    nbins     = {c: len(v) for c, v in spikes_of.items()}

    chan_neighbors = {}
    for c in unique_clus:
        pk   = peak_chs[c]
        hood = set(iCC_arr[:, pk].tolist())
        chan_neighbors[c] = {
            c2 for c2 in unique_clus
            if c2 != c and peak_chs[c2] in hood
        }

    if spike_z is not None and max_depth_um is not None:
        unit_depth = {c: float(np.median(spike_z[clu == c])) for c in unique_clus}
        chan_neighbors = {
            c: {c2 for c2 in hood if abs(unit_depth[c] - unit_depth[c2]) <= max_depth_um}
            for c, hood in chan_neighbors.items()
        }
        print(f'  [amp_bic] depth gate {max_depth_um:.0f} µm applied')

    picked        = set()
    mega_clusters = []

    t0 = time.time()
    n_seed = 0
    n_pair_checked = 0
    n_ccg_checked = 0
    n_bic_fit = 0
    n_merged = 0

    for seed in sorted(unique_clus, key=lambda c: nbins[c], reverse=True):
        if seed in picked:
            continue
        if nbins[seed] < min_spikes_seed:
            break

        n_seed += 1
        remaining_eligible = sum(
            1 for c in unique_clus if c not in picked and nbins.get(c, 0) >= min_spikes_pair
        )
        print(
            f'    [amp_bic] seed {n_seed}: unit={seed} spikes={nbins[seed]} '
            f'neighbors={len(chan_neighbors.get(seed, ()))} remaining={remaining_eligible}'
        )

        picked.add(seed)
        run_list  = [seed]
        pair_list = list(chan_neighbors.get(seed, ()))

        while pair_list:
            valid = [(p, nbins[p]) for p in pair_list
                     if p not in picked and nbins.get(p, 0) >= min_spikes_pair]
            if not valid:
                break
            ipair     = max(valid, key=lambda x: x[1])[0]
            pair_list = [p for p in pair_list if p != ipair]
            n_pair_checked += 1

            if n_pair_checked % 50 == 0:
                dt = time.time() - t0
                print(
                    f'    [amp_bic] seeds={n_seed} checked={n_pair_checked} '
                    f'bic={n_bic_fit} ccg={n_ccg_checked} merges={n_merged} elapsed={dt/60:.1f} min'
                )

            ipair_hood = chan_neighbors.get(ipair, set())
            if not any(r in ipair_hood for r in run_list):
                continue

            # Stage 1: BIC test on combined amplitude distribution
            run_idx  = np.concatenate([spikes_of[r] for r in run_list])
            pair_idx = spikes_of[ipair]
            combined = np.concatenate([amplitudes[run_idx], amplitudes[pair_idx]])

            try:
                bic1 = _fit_gmm_bic(combined, 1)
                bic2 = _fit_gmm_bic(combined, 2)
                n_bic_fit += 1
            except Exception as e:
                print(f'  [amp_bic] GMM fit failed for seed={run_list[0]} pair={ipair}: {e}')
                continue

            if bic1 > bic2 + bic_margin:
                continue   # two-component model wins → different populations

            # Stage 2: CCG
            n_ccg_checked += 1
            if not _ccg_similar(st_times[run_idx].astype(int),
                                 st_times[pair_idx].astype(int),
                                 fs=fs, ccg_thresh=ccg_thresh):
                continue

            print(f'  [amp_bic] Merging {ipair} → [{run_list[0]}]  '
                  f'BIC1={bic1:.0f}  BIC2={bic2:.0f}')
            run_list.append(ipair)
            picked.add(ipair)
            n_merged += 1

            if nbins[ipair] > 300:
                pair_list.extend(
                    p for p in chan_neighbors.get(ipair, ())
                    if p not in picked and p not in pair_list
                )

        mega_clusters.append(run_list)

    return [g for g in mega_clusters if len(g) > 1]


# ---------------------------------------------------------------------------
# Redundant unit removal (pure numpy)
# ---------------------------------------------------------------------------

def _find_redundant_pairs(clu, st_times, delta_ms=0.4, fs=30000,
                           agreement_threshold=0.2, duplicate_threshold=0.8):
    """
    Find cluster pairs whose spikes coincide excessively — likely the same unit
    captured by two nearby templates.
    """
    delta_samp  = int(round(delta_ms * fs / 1000))
    unique_clus = np.unique(clu)

    # Pre-sort once per unit — avoids O(N²) redundant sorts inside the inner loop
    times_sorted = {c: np.sort(st_times[clu == c].astype(np.int64)) for c in unique_clus}

    pairs = []
    for i, c1 in enumerate(unique_clus):
        t1 = times_sorted[c1]
        for c2 in unique_clus[i + 1:]:
            t2  = times_sorted[c2]
            idx = np.searchsorted(t2, t1)
            lo  = np.clip(idx - 1, 0, len(t2) - 1)
            hi  = np.clip(idx,     0, len(t2) - 1)
            close = (np.abs(t2[lo] - t1) <= delta_samp) | \
                    (np.abs(t2[hi] - t1) <= delta_samp)
            n = int(np.sum(close))
            if n == 0:
                continue
            if (n / min(len(t1), len(t2)) > agreement_threshold and
                    n / max(len(t1), len(t2)) > duplicate_threshold):
                pairs.append((int(c1), int(c2)))

    return pairs


def _flag_redundant_units(clu, st_times, fs=30000):
    """Return list of cluster IDs to drop (smaller cluster of each redundant pair)."""
    red_pairs = _find_redundant_pairs(clu, st_times, fs=fs)
    nbins_map = {c: int(np.sum(clu == c)) for c in np.unique(clu)}
    seen, to_remove = set(), []
    for c1, c2 in red_pairs:
        if c1 in seen or c2 in seen:
            continue
        to_drop = c2 if nbins_map.get(c1, 0) >= nbins_map.get(c2, 0) else c1
        to_remove.append(to_drop)
        seen.add(to_drop)
    return to_remove


# ---------------------------------------------------------------------------
# Shared apply / save
# ---------------------------------------------------------------------------

def _apply_curation_and_save(
    ks4_sorter, ks4_results, oldphypath, out_dir,
    duped_spikes, merge_unit_groups, remove_unit_ids,
    split_depth_export=False, depth_overlap_um=75.0, depth_split_um=None,
):
    """
    Apply cached curation decisions to raw KS4 arrays and export via save_sorting.

    Order of operations:
      1. Remove duplicated spikes from st, clu, tF.
      2. Relabel merged clusters; build updated Wall (append merged, delete old).
      3. Renumber clusters to 0..M-1.
      4. Remove redundant units (spikes and Wall rows).
      5. Renumber again and export via save_sorting.
    """
    out_dir    = Path(out_dir)
    oldphypath = Path(oldphypath)

    ops0  = load_ops(oldphypath / 'ops.npy')
    # Export path: avoid GPU tensors being converted to NumPy inside Kilosort's
    # save_sorting (some versions don't .cpu() before .numpy()).
    # We only force CPU for *saving*, not for the upstream KS4 run.
    try:
        ops0 = dict(ops0)
        if str(ops0.get('device', '')).startswith('cuda'):
            ops0['device'] = 'cpu'
    except Exception:
        pass
    st0   = ks4_results.st
    clu0  = np.load(oldphypath / 'spike_clusters.npy')
    tF0   = np.load(oldphypath / 'tF.npy')
    Wall0 = np.load(oldphypath / 'Wall.npy')
    kept0 = np.load(oldphypath / 'kept_spikes.npy')
    # Indices of spikes kept by KS4 (1D). Avoid torch indexing here to prevent
    # accidental CUDA tensors being passed into NumPy.
    kept = np.flatnonzero(kept0)

    # Step 1: remove duplicated spikes
    st1      = np.delete(st0,  duped_spikes, axis=0)
    clu1     = np.delete(clu0, duped_spikes, axis=0)
    tF_kept = tF0[kept]
    tF1 = np.delete(tF_kept, duped_spikes, axis=0)
    spike_z0 = ks4_results.spike_positions[:, 1]
    spike_z1 = np.delete(spike_z0, duped_spikes, axis=0)

    # Unit IDs must be contiguous 0..N-1 for Wall row indexing to be valid
    max_id = np.max(ks4_sorter.unit_ids)
    if (max_id >= len(ks4_sorter.unit_ids) or
            not np.array_equal(np.sort(ks4_sorter.unit_ids),
                               np.arange(len(ks4_sorter.unit_ids)))):
        raise ValueError(
            f'Unit IDs not consecutive 0..N-1 '
            f'(max={max_id}, count={len(ks4_sorter.unit_ids)})'
        )

    n_groups = len(merge_unit_groups)
    print(f'Applying {n_groups} merge groups, {len(remove_unit_ids)} redundant removals')

    # Step 2: build updated Wall and relabel merged clusters
    # newids are above the current maximum so they sort after all original IDs
    newids  = int(np.max(clu1)) + 1 + np.arange(n_groups, dtype=np.int64)
    Wall1   = Wall0.copy()
    wall_del = []

    # Clusters that lost all spikes during dup-spike removal
    lost_ids = set(np.unique(clu0).tolist()) - set(np.unique(clu1).tolist())
    wall_del.extend(sorted(lost_ids))
    if lost_ids:
        print(f'  {len(lost_ids)} clusters lost all spikes in dup-removal')

    # Append one new Wall row per merge group (best = most spikes)
    def _sorter_row(uid):
        rows = np.argwhere(ks4_sorter.unit_ids == uid)
        if len(rows) == 0:
            raise ValueError(
                f'Merge group unit {uid} not found in ks4_sorter.unit_ids '
                f'(max={int(ks4_sorter.unit_ids.max())}). '
                'Merge groups must reference units present in the sorter.'
            )
        return int(rows[0][0])

    for ii in range(n_groups):
        group    = merge_unit_groups[ii]
        nspk     = [int(np.sum(clu1 == g)) for g in group]
        best     = group[int(np.argmax(nspk))]
        Wall1    = np.append(Wall1, Wall0[_sorter_row(best):_sorter_row(best) + 1], axis=0)

    # Relabel merged spikes and mark old Wall rows for deletion
    for ii in range(n_groups):
        for uid in merge_unit_groups[ii]:
            clu1[clu1 == uid] = newids[ii]
            wall_del.append(_sorter_row(uid))

    Wall1 = np.delete(Wall1, np.unique(wall_del).astype(int), axis=0)

    # Step 3: renumber clusters → 0..M-1
    unique_clus, clu_new = np.unique(clu1, return_inverse=True)
    assert len(unique_clus) == Wall1.shape[0], (
        f'Wall rows ({Wall1.shape[0]}) ≠ unique clusters ({len(unique_clus)})'
    )

    # Step 4: remove redundant units (actually applied here)
    merged_ids    = set(u for g in merge_unit_groups for u in g)
    to_remove_set = set(int(u) for u in remove_unit_ids) - merged_ids
    if to_remove_set:
        remove_rows = [i for i, uid in enumerate(unique_clus.tolist())
                       if uid in to_remove_set]
        if remove_rows:
            spike_keep  = ~np.isin(clu_new, remove_rows)
            st1         = st1[spike_keep]
            tF1         = tF1[spike_keep]
            spike_z1    = spike_z1[spike_keep]
            clu_tmp     = clu_new[spike_keep]
            Wall1       = np.delete(Wall1, remove_rows, axis=0)
            unique_clus, clu_new = np.unique(clu_tmp, return_inverse=True)
            print(f'  Removed {len(remove_rows)} redundant units')

    # Step 5: export
    ops_export = copy.deepcopy(ops0)
    # Duplicate spikes were already removed before relabeling; exporting with the
    # original KS setting would drop a second set of spikes after merges.
    ops_export['duplicate_spike_bins'] = 0

    def _validate_saved_sorting(save_dir, st_expected, clu_expected, wall_expected):
        raw_dir = save_dir / 'sorter_output' if (save_dir / 'sorter_output').exists() else save_dir
        saved_st = np.load(raw_dir / 'spike_times.npy')
        saved_clu = np.load(raw_dir / 'spike_clusters.npy')
        saved_wall = np.load(raw_dir / 'Wall.npy')

        from pipeline import KilosortResults
        loaded = KilosortResults(save_dir)
        loaded_st = loaded.spike_times
        loaded_clu = loaded.spike_clusters

        expected_counts = np.bincount(clu_expected.astype(np.int64), minlength=wall_expected.shape[0])
        saved_counts = np.bincount(saved_clu.astype(np.int64), minlength=saved_wall.shape[0])
        loaded_counts = np.bincount(loaded_clu.astype(np.int64), minlength=saved_wall.shape[0])

        issues = []
        if saved_st.shape[0] != st_expected.shape[0] or saved_clu.shape[0] != clu_expected.shape[0]:
            issues.append(
                f"save_sorting wrote {saved_clu.shape[0]} spikes from {clu_expected.shape[0]} expected"
            )
        if saved_wall.shape[0] != wall_expected.shape[0]:
            issues.append(
                f"save_sorting wrote {saved_wall.shape[0]} templates from {wall_expected.shape[0]} expected"
            )
        if loaded_st.shape[0] != st_expected.shape[0] or loaded_clu.shape[0] != clu_expected.shape[0]:
            issues.append(
                f"KilosortResults loaded {loaded_clu.shape[0]} spikes from {clu_expected.shape[0]} expected"
            )
        if not np.array_equal(expected_counts, saved_counts):
            issues.append('per-unit spike counts differ immediately after save_sorting')
        if not np.array_equal(expected_counts, loaded_counts):
            issues.append('per-unit spike counts differ after KilosortResults reload')

        if issues:
            np.savez(
                save_dir / 'export_validation_failure.npz',
                expected_spike_count=np.array([st_expected.shape[0]], dtype=np.int64),
                saved_spike_count=np.array([saved_st.shape[0]], dtype=np.int64),
                loaded_spike_count=np.array([loaded_st.shape[0]], dtype=np.int64),
                expected_unit_count=np.array([wall_expected.shape[0]], dtype=np.int64),
                saved_unit_count=np.array([saved_wall.shape[0]], dtype=np.int64),
                expected_counts=expected_counts,
                saved_counts=saved_counts,
                loaded_counts=loaded_counts,
            )
            raise RuntimeError(
                f"Export validation failed for {save_dir}: " + '; '.join(issues)
            )

        return loaded

    def _save_subset(save_dir, unit_rows):
        unit_rows  = np.asarray(unit_rows, dtype=np.int64)
        spike_mask = np.isin(clu_new, unit_rows)
        spk_idx    = np.flatnonzero(spike_mask)
        st_sub     = st1[spk_idx]
        tF_sub     = tF1[spk_idx]
        u_sub, clu_sub = np.unique(clu_new[spk_idx], return_inverse=True)
        Wall_sub   = Wall1[u_sub]
        save_dir.mkdir(parents=True, exist_ok=True)
        save_sorting(
            ops=ops_export, results_dir=save_dir,
            st=st_sub, clu=clu_sub.astype('int32'),
            tF=torch.as_tensor(tF_sub).detach().cpu(),
            Wall=torch.as_tensor(Wall_sub).detach().cpu(),
            imin=0, tic0=time.time(), save_extra_vars=True,
        )
        loaded = _validate_saved_sorting(save_dir, st_sub, clu_sub.astype('int32'), Wall_sub)
        np.savez(save_dir / 'depth_split_meta.npz',
                 global_unit_ids=u_sub, n_spikes=st_sub.shape[0])
        return loaded

    if split_depth_export:
        unit_rows  = np.arange(len(unique_clus))
        unit_depth = np.array([np.median(spike_z1[clu_new == r]) for r in unit_rows])
        if depth_split_um is None:
            depth_split_um = float(np.median(unit_depth))
        top_rows = unit_rows[unit_depth >= (depth_split_um - depth_overlap_um)]
        bot_rows = unit_rows[unit_depth <= (depth_split_um + depth_overlap_um)]
        top_results = _save_subset(out_dir / 'depth_top', top_rows)
        bot_results = _save_subset(out_dir / 'depth_bot', bot_rows)
        return {
            'top': top_results,
            'bot': bot_results,
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    save_sorting(
        ops=ops_export, results_dir=out_dir,
        st=st1, clu=clu_new.astype('int32'),
        tF=torch.as_tensor(tF1).detach().cpu(),
        Wall=torch.as_tensor(Wall1).detach().cpu(),
        imin=0, tic0=time.time(), save_extra_vars=True,
    )
    return _validate_saved_sorting(out_dir, st1, clu_new.astype('int32'), Wall1)


# ---------------------------------------------------------------------------
# Curation entry points
# ---------------------------------------------------------------------------

def run_cur(
    ks4_sorter,
    ks4_results,
    cache_dir,
    recalc=False,
    split_depth_export=False,
    depth_overlap_um=75.0,
    depth_split_um=None,
    posthoc_fracse=0.1,
    posthoc_score_thresh=3,
    posthoc_ccg_thresh=0.5,
    posthoc_min_spikes_seed=500,
    posthoc_min_spikes_pair=100,
    posthoc_max_depth_um=150.0,
    enable_redundant_removal=True,
    cross_strategy_gate=True,   # filter posthoc groups by cosine/amp_bic agreement
    fs=30000,
    _seg=None,      # retained for API compatibility
):
    """
    Feature-projection + CCG merge (port of merge_posthoc3.m).
    Cache: cur_todo_phy.npy  →  output: cur_sorter_output/
    """
    if isinstance(cache_dir, str):
        cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    npy_path   = cache_dir / 'cur_todo_phy.npy'
    oldphypath = cache_dir.parent / 'kilosort4/sorter_output/'
    out_dir    = cache_dir / 'cur_sorter_output'

    if npy_path.exists() and not recalc:
        todo              = np.load(npy_path, allow_pickle=True).item()
        merge_unit_groups = todo['merge_unit_groups']
        remove_unit_ids   = todo['removed_units']
        duped_spikes      = todo['duped_spikes']
    else:
        clu_raw = ks4_results.spike_clusters
        sp_z    = ks4_results.spike_positions[:, 1]
        sp_t    = ks4_results.spike_times

        duped_spikes = np.flatnonzero(
            (np.diff(sp_t) < 1) &
            (np.diff(sp_z) < 10) &
            (np.diff(clu_raw) != 0)
        )
        print(f'{100 * len(duped_spikes) / len(sp_t):.2f}%  are duped spikes')

        ops  = load_ops(oldphypath / 'ops.npy')
        iCC  = _to_numpy_cpu(ops['iCC'], dtype=int)
        iU   = _to_numpy_cpu(ops['iU'],  dtype=int)

        clu0  = np.load(oldphypath / 'spike_clusters.npy')
        tmps0 = ks4_results.spike_templates
        st0   = ks4_results.st
        tF0   = np.load(oldphypath / 'tF.npy')
        kept0 = np.load(oldphypath / 'kept_spikes.npy')
        kept  = np.flatnonzero(kept0)

        clu1   = np.delete(clu0,  duped_spikes, axis=0)
        tmps1  = np.delete(tmps0, duped_spikes, axis=0)
        st1    = np.delete(st0,   duped_spikes, axis=0)
        sp_z1  = np.delete(sp_z,  duped_spikes, axis=0)

        tF_kept = tF0[kept]
        tF1     = np.delete(tF_kept, duped_spikes, axis=0)
        st_times = st1[:, 0].astype(int) if st1.ndim > 1 else st1.astype(int)

        print('Running posthoc merge...')
        merge_unit_groups = _posthoc_merge(
            tF1, clu1, st_times, tmps1, iCC, iU,
            fs=fs,
            fracse=posthoc_fracse,
            score_thresh=posthoc_score_thresh,
            ccg_thresh=posthoc_ccg_thresh,
            min_spikes_seed=posthoc_min_spikes_seed,
            min_spikes_pair=posthoc_min_spikes_pair,
            spike_z=sp_z1,
            max_depth_um=posthoc_max_depth_um,
        )
        print(f'Found {len(merge_unit_groups)} merge groups (before gate)')

        if cross_strategy_gate and merge_unit_groups:
            cosine_path = cache_dir / 'cur_todo_cosine.npy'
            bic_path    = cache_dir / 'cur_todo_amp_bic.npy'
            alt_pairs: set = set()
            for p in (cosine_path, bic_path):
                if p.exists():
                    alt_groups = np.load(p, allow_pickle=True).item().get('merge_unit_groups', [])
                    for g in alt_groups:
                        for a, b in zip(g, g[1:]):
                            for x in g:
                                for y in g:
                                    if x != y:
                                        alt_pairs.add(frozenset([int(x), int(y)]))
            if alt_pairs:
                from itertools import combinations as _comb
                before = len(merge_unit_groups)
                merge_unit_groups = [
                    g for g in merge_unit_groups
                    if any(frozenset([int(a), int(b)]) in alt_pairs
                           for a, b in _comb(g, 2))
                ]
                print(f'Cross-strategy gate: {before} → {len(merge_unit_groups)} groups kept')
            else:
                print('Cross-strategy gate: no cosine/amp_bic caches found — skipping gate')

        remove_unit_ids = []
        if enable_redundant_removal:
            remove_unit_ids = _flag_redundant_units(clu1, st_times, fs=fs)
            print(f'Flagging {len(remove_unit_ids)} redundant units for removal')

        np.save(npy_path, {  # type: ignore[arg-type]
            'duped_spikes':      duped_spikes,
            'merge_unit_groups': merge_unit_groups,
            'removed_units':     remove_unit_ids,
        }, allow_pickle=True)

    return _apply_curation_and_save(
        ks4_sorter, ks4_results, oldphypath, out_dir,
        duped_spikes, merge_unit_groups, remove_unit_ids,
        split_depth_export=split_depth_export,
        depth_overlap_um=depth_overlap_um,
        depth_split_um=depth_split_um,
    )


def run_cur_cosine(
    ks4_sorter,
    ks4_results,
    cache_dir,
    recalc=False,
    split_depth_export=False,
    depth_overlap_um=75.0,
    depth_split_um=None,
    cosine_thresh=0.90,
    ccg_thresh=0.5,
    min_spikes_seed=500,
    min_spikes_pair=100,
    max_depth_um=200.0,
    enable_redundant_removal=True,
    fs=30000,
    ks4_out_path=None,
    _out_subdir='cur_cosine_output',
):
    """
    Wall template cosine similarity + CCG merge.
    Cache: cur_todo_cosine.npy  →  output: cur_cosine_output/

    ks4_out_path : path to kilosort4/sorter_output/ if it lives outside
                   cache_dir.parent/kilosort4/sorter_output/ (e.g. on a
                   different drive). Defaults to that conventional location.
    """
    if isinstance(cache_dir, str):
        cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    npy_path   = cache_dir / 'cur_todo_cosine.npy'
    oldphypath = Path(ks4_out_path) if ks4_out_path is not None \
                 else cache_dir.parent / 'kilosort4/sorter_output/'
    out_dir    = cache_dir / _out_subdir

    if npy_path.exists() and not recalc:
        todo              = np.load(npy_path, allow_pickle=True).item()
        merge_unit_groups = todo['merge_unit_groups']
        remove_unit_ids   = todo['removed_units']
        duped_spikes      = todo['duped_spikes']
    else:
        clu_raw = ks4_results.spike_clusters
        sp_z    = ks4_results.spike_positions[:, 1]
        sp_t    = ks4_results.spike_times

        duped_spikes = np.flatnonzero(
            (np.diff(sp_t) < 1) &
            (np.diff(sp_z) < 10) &
            (np.diff(clu_raw) != 0)
        )
        print(f'{100 * len(duped_spikes) / len(sp_t):.2f}%  are duped spikes')

        ops   = load_ops(oldphypath / 'ops.npy')
        iCC   = _to_numpy_cpu(ops['iCC'], dtype=int)
        clu0  = np.load(oldphypath / 'spike_clusters.npy')
        Wall0 = np.load(oldphypath / 'Wall.npy')
        st0   = ks4_results.st

        clu1     = np.delete(clu0, duped_spikes, axis=0)
        st1      = np.delete(st0,  duped_spikes, axis=0)
        sp_z1    = np.delete(sp_z, duped_spikes, axis=0)
        st_times = st1[:, 0].astype(int) if st1.ndim > 1 else st1.astype(int)

        print('Running cosine merge...')
        merge_unit_groups = _template_cosine_merge(
            Wall0, clu1, st_times, iCC,
            cosine_thresh=cosine_thresh,
            ccg_thresh=ccg_thresh,
            min_spikes_seed=min_spikes_seed,
            min_spikes_pair=min_spikes_pair,
            fs=fs,
            spike_z=sp_z1,
            max_depth_um=max_depth_um,
        )
        print(f'Found {len(merge_unit_groups)} merge groups')

        remove_unit_ids = []
        if enable_redundant_removal:
            remove_unit_ids = _flag_redundant_units(clu1, st_times, fs=fs)
            print(f'Flagging {len(remove_unit_ids)} redundant units for removal')

        np.save(npy_path, {  # type: ignore[arg-type]
            'duped_spikes':      duped_spikes,
            'merge_unit_groups': merge_unit_groups,
            'removed_units':     remove_unit_ids,
        }, allow_pickle=True)

    return _apply_curation_and_save(
        ks4_sorter, ks4_results, oldphypath, out_dir,
        duped_spikes, merge_unit_groups, remove_unit_ids,
        split_depth_export=split_depth_export,
        depth_overlap_um=depth_overlap_um,
        depth_split_um=depth_split_um,
    )


def run_cur_amp_bic(
    ks4_sorter,
    ks4_results,
    cache_dir,
    recalc=False,
    split_depth_export=False,
    depth_overlap_um=75.0,
    depth_split_um=None,
    bic_margin=0.25,
    ccg_thresh=0.5,
    min_spikes_seed=100,
    min_spikes_pair=100,
    max_depth_um=150.0,
    enable_redundant_removal=True,
    fs=30000,
):
    """
    Amplitude BIC (1- vs 2-Gaussian) + CCG merge.
    Cache: cur_todo_amp_bic.npy  →  output: cur_amp_bic_output/
    """
    if isinstance(cache_dir, str):
        cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    npy_path   = cache_dir / 'cur_todo_amp_bic.npy'
    oldphypath = cache_dir.parent / 'kilosort4/sorter_output/'
    out_dir    = cache_dir / 'cur_amp_bic_output'

    if npy_path.exists() and not recalc:
        todo              = np.load(npy_path, allow_pickle=True).item()
        merge_unit_groups = todo['merge_unit_groups']
        remove_unit_ids   = todo['removed_units']
        duped_spikes      = todo['duped_spikes']
    else:
        clu_raw = ks4_results.spike_clusters
        sp_z    = ks4_results.spike_positions[:, 1]
        sp_t    = ks4_results.spike_times

        duped_spikes = np.flatnonzero(
            (np.diff(sp_t) < 1) &
            (np.diff(sp_z) < 10) &
            (np.diff(clu_raw) != 0)
        )
        print(f'{100 * len(duped_spikes) / len(sp_t):.2f}%  are duped spikes')

        ops   = load_ops(oldphypath / 'ops.npy')
        iCC   = _to_numpy_cpu(ops['iCC'], dtype=int)
        clu0  = np.load(oldphypath / 'spike_clusters.npy')
        Wall0 = np.load(oldphypath / 'Wall.npy')
        st0   = ks4_results.st
        amps0 = ks4_results.spike_amplitudes   # (n_kept_spikes,)

        clu1     = np.delete(clu0,  duped_spikes, axis=0)
        st1      = np.delete(st0,   duped_spikes, axis=0)
        amps1    = np.delete(amps0, duped_spikes, axis=0)
        sp_z1    = np.delete(sp_z,  duped_spikes, axis=0)
        st_times = st1[:, 0].astype(int) if st1.ndim > 1 else st1.astype(int)

        print('Running amplitude BIC merge...')
        merge_unit_groups = _amplitude_bic_merge(
            amps1, clu1, st_times, iCC, Wall0,
            bic_margin=bic_margin,
            ccg_thresh=ccg_thresh,
            min_spikes_seed=min_spikes_seed,
            min_spikes_pair=min_spikes_pair,
            fs=fs,
            spike_z=sp_z1,
            max_depth_um=max_depth_um,
        )
        print(f'Found {len(merge_unit_groups)} merge groups')

        remove_unit_ids = []
        if enable_redundant_removal:
            remove_unit_ids = _flag_redundant_units(clu1, st_times, fs=fs)
            print(f'Flagging {len(remove_unit_ids)} redundant units for removal')

        np.save(npy_path, {  # type: ignore[arg-type]
            'duped_spikes':      duped_spikes,
            'merge_unit_groups': merge_unit_groups,
            'removed_units':     remove_unit_ids,
        }, allow_pickle=True)

    return _apply_curation_and_save(
        ks4_sorter, ks4_results, oldphypath, out_dir,
        duped_spikes, merge_unit_groups, remove_unit_ids,
        split_depth_export=split_depth_export,
        depth_overlap_um=depth_overlap_um,
        depth_split_um=depth_split_um,
    )


def run_cur_no_merge(
    ks4_sorter,
    ks4_results,
    cache_dir,
    recalc=False,
    split_depth_export=False,
    depth_overlap_um=75.0,
    depth_split_um=None,
    enable_redundant_removal=True,
    fs=30000,
):
    """
    No-merge control: duplicate spike removal + redundant unit removal only.
    Cache: cur_todo_no_merge.npy  →  output: cur_no_merge_output/
    """
    if isinstance(cache_dir, str):
        cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    npy_path   = cache_dir / 'cur_todo_no_merge.npy'
    oldphypath = cache_dir.parent / 'kilosort4/sorter_output/'
    out_dir    = cache_dir / 'cur_no_merge_output'

    if npy_path.exists() and not recalc:
        todo              = np.load(npy_path, allow_pickle=True).item()
        merge_unit_groups = todo['merge_unit_groups']
        remove_unit_ids   = todo['removed_units']
        duped_spikes      = todo['duped_spikes']
    else:
        clu_raw = ks4_results.spike_clusters
        sp_z    = ks4_results.spike_positions[:, 1]
        sp_t    = ks4_results.spike_times

        duped_spikes = np.flatnonzero(
            (np.diff(sp_t) < 1) &
            (np.diff(sp_z) < 10) &
            (np.diff(clu_raw) != 0)
        )
        print(f'{100 * len(duped_spikes) / len(sp_t):.2f}%  are duped spikes')

        clu0  = np.load(oldphypath / 'spike_clusters.npy')
        st0   = ks4_results.st
        clu1  = np.delete(clu0, duped_spikes, axis=0)
        st1   = np.delete(st0,  duped_spikes, axis=0)
        st_times = st1[:, 0].astype(int) if st1.ndim > 1 else st1.astype(int)

        merge_unit_groups = []

        remove_unit_ids = []
        if enable_redundant_removal:
            remove_unit_ids = _flag_redundant_units(clu1, st_times, fs=fs)
            print(f'Flagging {len(remove_unit_ids)} redundant units for removal')

        np.save(npy_path, {  # type: ignore[arg-type]
            'duped_spikes':      duped_spikes,
            'merge_unit_groups': merge_unit_groups,
            'removed_units':     remove_unit_ids,
        }, allow_pickle=True)

    return _apply_curation_and_save(
        ks4_sorter, ks4_results, oldphypath, out_dir,
        duped_spikes, merge_unit_groups, remove_unit_ids,
        split_depth_export=split_depth_export,
        depth_overlap_um=depth_overlap_um,
        depth_split_um=depth_split_um,
    )


def run_cur_final(
    ks4_sorter,
    ks4_results,
    cache_dir,
    recalc=False,
    split_depth_export=False,
    depth_overlap_um=75.0,
    depth_split_um=None,
    cosine_thresh=0.90,
    ccg_thresh=0.5,
    min_spikes_seed=500,
    min_spikes_pair=100,
    enable_redundant_removal=True,
    fs=30000,
    ks4_out_path=None,
):
    """
    Recommended final curation: cosine template merge → cur_output/

    Identical to run_cur_cosine but writes to cache_dir/cur_output/ so the
    result has a stable, strategy-agnostic path for downstream consumers.
    The cosine merge cache (cur_todo_cosine.npy) is shared with run_cur_cosine,
    so running both does not duplicate computation.

    ks4_out_path : override path to kilosort4/sorter_output/ — use when KS4
                   lives on a different drive from cache_dir (see pipeline scripts).
    """
    return run_cur_cosine(
        ks4_sorter, ks4_results, cache_dir,
        recalc=recalc,
        split_depth_export=split_depth_export,
        depth_overlap_um=depth_overlap_um,
        depth_split_um=depth_split_um,
        cosine_thresh=cosine_thresh,
        ccg_thresh=ccg_thresh,
        min_spikes_seed=min_spikes_seed,
        min_spikes_pair=min_spikes_pair,
        enable_redundant_removal=enable_redundant_removal,
        fs=fs,
        ks4_out_path=ks4_out_path,
        _out_subdir='cur_output',
    )


def load_cur(cache_dir):
    return np.load(cache_dir)
