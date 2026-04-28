#%%
# Post-sorting curation: duplicate spike removal, merge, redundant unit removal,
# and re-export to Phy/KS4 format.
#
# Four strategies:
#   run_cur          — feature-projection + CCG merge (port of merge_posthoc3.m)
#   run_cur_cosine   — Wall template cosine similarity + CCG merge
#   run_cur_amp_bic  — amplitude BIC (1- vs 2-Gaussian) + CCG merge
#   run_cur_no_merge — no merge; dup-spike removal + redundant unit removal only

import shutil
import numpy as np
from pathlib import Path
from kilosort.run_kilosort import save_sorting
from kilosort.io import load_ops
import copy
import time
import torch


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

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
        return 0

    b1 = np.arange(0, l2 + se25, se25)
    b2 = np.arange(0, -l1 + se25, se25)
    if len(b1) < 2 or len(b2) < 2:
        return 0

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

    bin_samp = max(1, int(round(binsize_ms * fs / 1000)))
    nlags    = max(1, int(round(nlags_ms   * fs / 1000 / bin_samp)))

    n_bins = int(max(np.max(st_a), np.max(st_b))) // bin_samp + 2
    sp1 = np.zeros(n_bins)
    sp2 = np.zeros(n_bins)
    np.add.at(sp1, np.clip(st_a // bin_samp, 0, n_bins - 1).astype(int), 1)
    np.add.at(sp2, np.clip(st_b // bin_samp, 0, n_bins - 1).astype(int), 1)

    mid = n_bins - 1
    xc1 = np.correlate(sp1, sp1, mode='full')[mid - nlags: mid + nlags + 1]
    xc2 = np.correlate(sp2, sp2, mode='full')[mid - nlags: mid + nlags + 1]
    xc3 = np.correlate(sp1, sp2, mode='full')[mid - nlags: mid + nlags + 1]
    xc1[nlags] = 0
    xc2[nlags] = 0

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

    chan_neighbors = {}
    for c in unique_clus:
        pk = clu_to_peak[c]
        if pk < 0:
            chan_neighbors[c] = []
            continue
        hood = set(iCC_arr[:, pk].tolist())
        chan_neighbors[c] = [
            c2 for c2 in unique_clus
            if c2 != c and clu_to_peak.get(c2, -1) in hood
        ]

    picked        = set()
    mega_clusters = []

    for seed in sorted(unique_clus, key=lambda c: nbins[c], reverse=True):
        if seed in picked:
            continue
        if nbins[seed] < min_spikes_seed:
            break

        picked.add(seed)
        run_list  = [seed]
        pair_list = list(chan_neighbors.get(seed, []))

        while pair_list:
            valid = [(p, nbins[p]) for p in pair_list
                     if p not in picked and nbins.get(p, 0) >= min_spikes_pair]
            if not valid:
                break
            ipair     = max(valid, key=lambda x: x[1])[0]
            pair_list = [p for p in pair_list if p != ipair]

            ipair_hood = set(chan_neighbors.get(ipair, []))
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

            run_times = st_times[np.concatenate([spikes_of[r] for r in run_list])]
            if not _ccg_similar(run_times.astype(int), st_times[new_idx].astype(int),
                                 fs=fs, ccg_thresh=ccg_thresh):
                continue

            print(f'  Merging cluster {ipair} into run [{run_list[0]}]')
            run_list.append(ipair)
            picked.add(ipair)

            if nbins[ipair] > 300:
                pair_list.extend(
                    p for p in chan_neighbors.get(ipair, [])
                    if p not in picked and p not in pair_list
                )

        mega_clusters.append(run_list)

    return [g for g in mega_clusters if len(g) > 1]


def _template_cosine_merge(
    Wall, clu, st_times, iCC,
    cosine_thresh=0.90, ccg_thresh=0.5,
    min_spikes_seed=500, min_spikes_pair=100, fs=30000,
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

    spikes_of = {c: np.flatnonzero(clu == c) for c in unique_clus}
    nbins     = {c: len(v) for c, v in spikes_of.items()}

    chan_neighbors = {}
    for c in unique_clus:
        pk   = peak_chs[c]
        hood = set(iCC_arr[:, pk].tolist())
        chan_neighbors[c] = [
            c2 for c2 in unique_clus
            if c2 != c and peak_chs[c2] in hood
        ]

    picked        = set()
    mega_clusters = []

    for seed in sorted(unique_clus, key=lambda c: nbins[c], reverse=True):
        if seed in picked:
            continue
        if nbins[seed] < min_spikes_seed:
            break

        picked.add(seed)
        run_list  = [seed]
        pair_list = list(chan_neighbors.get(seed, []))

        while pair_list:
            valid = [(p, nbins[p]) for p in pair_list
                     if p not in picked and nbins.get(p, 0) >= min_spikes_pair]
            if not valid:
                break
            ipair     = max(valid, key=lambda x: x[1])[0]
            pair_list = [p for p in pair_list if p != ipair]

            if not any(r in set(chan_neighbors.get(ipair, [])) for r in run_list):
                continue

            # Stage 1: cosine similarity between seed and candidate Wall templates
            sim = _cosine_sim(Wall[run_list[0]], Wall[ipair])
            if sim < cosine_thresh:
                continue

            # Stage 2: CCG
            run_times = st_times[np.concatenate([spikes_of[r] for r in run_list])]
            if not _ccg_similar(run_times.astype(int),
                                 st_times[spikes_of[ipair]].astype(int),
                                 fs=fs, ccg_thresh=ccg_thresh):
                continue

            print(f'  [cosine] Merging {ipair} → [{run_list[0]}]  cos={sim:.3f}')
            run_list.append(ipair)
            picked.add(ipair)

            if nbins[ipair] > 300:
                pair_list.extend(
                    p for p in chan_neighbors.get(ipair, [])
                    if p not in picked and p not in pair_list
                )

        mega_clusters.append(run_list)

    return [g for g in mega_clusters if len(g) > 1]


def _amplitude_bic_merge(
    amplitudes, clu, st_times, iCC, Wall,
    bic_margin=0.0, ccg_thresh=0.5,
    min_spikes_seed=100, min_spikes_pair=100, fs=30000,
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
        chan_neighbors[c] = [
            c2 for c2 in unique_clus
            if c2 != c and peak_chs[c2] in hood
        ]

    picked        = set()
    mega_clusters = []

    for seed in sorted(unique_clus, key=lambda c: nbins[c], reverse=True):
        if seed in picked:
            continue
        if nbins[seed] < min_spikes_seed:
            break

        picked.add(seed)
        run_list  = [seed]
        pair_list = list(chan_neighbors.get(seed, []))

        while pair_list:
            valid = [(p, nbins[p]) for p in pair_list
                     if p not in picked and nbins.get(p, 0) >= min_spikes_pair]
            if not valid:
                break
            ipair     = max(valid, key=lambda x: x[1])[0]
            pair_list = [p for p in pair_list if p != ipair]

            if not any(r in set(chan_neighbors.get(ipair, [])) for r in run_list):
                continue

            # Stage 1: BIC test on combined amplitude distribution
            run_idx  = np.concatenate([spikes_of[r] for r in run_list])
            pair_idx = spikes_of[ipair]
            combined = np.concatenate([amplitudes[run_idx], amplitudes[pair_idx]])

            try:
                bic1 = _fit_gmm_bic(combined, 1)
                bic2 = _fit_gmm_bic(combined, 2)
            except Exception:
                continue

            if bic1 > bic2 + bic_margin:
                continue   # two-component model wins → different populations

            # Stage 2: CCG
            if not _ccg_similar(st_times[run_idx].astype(int),
                                 st_times[pair_idx].astype(int),
                                 fs=fs, ccg_thresh=ccg_thresh):
                continue

            print(f'  [amp_bic] Merging {ipair} → [{run_list[0]}]  '
                  f'BIC1={bic1:.0f}  BIC2={bic2:.0f}')
            run_list.append(ipair)
            picked.add(ipair)

            if nbins[ipair] > 300:
                pair_list.extend(
                    p for p in chan_neighbors.get(ipair, [])
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
    pairs       = []

    for i, c1 in enumerate(unique_clus):
        t1 = np.sort(st_times[clu == c1].astype(np.int64))
        for c2 in unique_clus[i + 1:]:
            t2  = np.sort(st_times[clu == c2].astype(np.int64))
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
    st0   = ks4_results.st
    clu0  = np.load(oldphypath / 'spike_clusters.npy')
    tF0   = np.load(oldphypath / 'tF.npy')
    Wall0 = np.load(oldphypath / 'Wall.npy')
    kept0 = np.load(oldphypath / 'kept_spikes.npy')
    kept  = np.argwhere(kept0)

    # Step 1: remove duplicated spikes
    st1      = np.delete(st0,  duped_spikes, axis=0)
    clu1     = np.delete(clu0, duped_spikes, axis=0)
    tF_kept  = torch.from_numpy(tF0)[kept]
    tF1_     = torch.as_tensor(
        np.squeeze(np.delete(np.asarray(tF_kept), duped_spikes, axis=0))
    )
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
    for ii in range(n_groups):
        group    = merge_unit_groups[ii]
        nspk     = [int(np.sum(clu1 == g)) for g in group]
        best     = group[int(np.argmax(nspk))]
        best_row = int(np.argwhere(ks4_sorter.unit_ids == best)[0][0])
        Wall1    = np.append(Wall1, Wall0[best_row:best_row + 1], axis=0)

    # Relabel merged spikes and mark old Wall rows for deletion
    for ii in range(n_groups):
        for uid in merge_unit_groups[ii]:
            clu1[clu1 == uid] = newids[ii]
            wall_del.append(int(np.argwhere(ks4_sorter.unit_ids == uid)[0][0]))

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
            tF1_        = torch.as_tensor(np.asarray(tF1_)[spike_keep])
            spike_z1    = spike_z1[spike_keep]
            clu_tmp     = clu_new[spike_keep]
            Wall1       = np.delete(Wall1, remove_rows, axis=0)
            unique_clus, clu_new = np.unique(clu_tmp, return_inverse=True)
            print(f'  Removed {len(remove_rows)} redundant units')

    Wall1_ = torch.from_numpy(Wall1)

    # Step 5: export
    def _save_subset(save_dir, unit_rows):
        unit_rows  = np.asarray(unit_rows, dtype=np.int64)
        spike_mask = np.isin(clu_new, unit_rows)
        spk_idx    = np.flatnonzero(spike_mask)
        st_sub     = st1[spk_idx]
        tF_sub     = np.asarray(tF1_)[spk_idx]
        u_sub, clu_sub = np.unique(clu_new[spk_idx], return_inverse=True)
        Wall_sub   = Wall1[u_sub]
        save_dir.mkdir(parents=True, exist_ok=True)
        save_sorting(
            ops=ops0, results_dir=save_dir,
            st=st_sub, clu=clu_sub.astype('int32'),
            tF=torch.as_tensor(tF_sub), Wall=torch.as_tensor(Wall_sub),
            imin=0, tic0=time.time(), save_extra_vars=True,
        )
        np.savez(save_dir / 'depth_split_meta.npz',
                 global_unit_ids=u_sub, n_spikes=st_sub.shape[0])
        return save_dir

    if split_depth_export:
        unit_rows  = np.arange(len(unique_clus))
        unit_depth = np.array([np.median(spike_z1[clu_new == r]) for r in unit_rows])
        if depth_split_um is None:
            depth_split_um = float(np.median(unit_depth))
        top_rows = unit_rows[unit_depth >= (depth_split_um - depth_overlap_um)]
        bot_rows = unit_rows[unit_depth <= (depth_split_um + depth_overlap_um)]
        top_dir  = _save_subset(out_dir / 'depth_top', top_rows)
        bot_dir  = _save_subset(out_dir / 'depth_bot', bot_rows)
        from pipeline import KilosortResults
        return {
            'top': KilosortResults(top_dir) if top_dir is not None else None,
            'bot': KilosortResults(bot_dir) if bot_dir is not None else None,
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    save_sorting(
        ops=ops0, results_dir=out_dir,
        st=st1, clu=clu_new.astype('int32'),
        tF=tF1_, Wall=Wall1_,
        imin=0, tic0=time.time(), save_extra_vars=True,
    )
    from pipeline import KilosortResults
    return KilosortResults(out_dir)


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
    enable_redundant_removal=True,
    fs=30000,
    seg=None,       # retained for API compatibility
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
        iCC  = np.asarray(ops['iCC'], dtype=int)
        iU   = np.asarray(ops['iU'],  dtype=int)

        clu0  = np.load(oldphypath / 'spike_clusters.npy')
        tmps0 = ks4_results.spike_templates
        st0   = ks4_results.st
        tF0   = np.load(oldphypath / 'tF.npy')
        kept0 = np.load(oldphypath / 'kept_spikes.npy')
        kept  = np.argwhere(kept0)

        clu1  = np.delete(clu0,  duped_spikes, axis=0)
        tmps1 = np.delete(tmps0, duped_spikes, axis=0)
        st1   = np.delete(st0,   duped_spikes, axis=0)

        tF_kept = np.asarray(torch.from_numpy(tF0)[kept])
        tF1     = np.squeeze(np.delete(tF_kept, duped_spikes, axis=0))
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
        )
        print(f'Found {len(merge_unit_groups)} merge groups')

        remove_unit_ids = []
        if enable_redundant_removal:
            remove_unit_ids = _flag_redundant_units(clu1, st_times, fs=fs)
            print(f'Flagging {len(remove_unit_ids)} redundant units for removal')

        ks4_sorter_clean = remove_duped_spikes(ks4_sorter, duped_spikes)
        if out_dir.exists():
            shutil.rmtree(out_dir)
        ks4_sorter_clean.save_to_folder(out_dir)

        np.save(npy_path, {
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
    enable_redundant_removal=True,
    fs=30000,
):
    """
    Wall template cosine similarity + CCG merge.
    Cache: cur_todo_cosine.npy  →  output: cur_cosine_output/
    """
    if isinstance(cache_dir, str):
        cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    npy_path   = cache_dir / 'cur_todo_cosine.npy'
    oldphypath = cache_dir.parent / 'kilosort4/sorter_output/'
    out_dir    = cache_dir / 'cur_cosine_output'

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
        iCC   = np.asarray(ops['iCC'], dtype=int)
        clu0  = np.load(oldphypath / 'spike_clusters.npy')
        Wall0 = np.load(oldphypath / 'Wall.npy')
        st0   = ks4_results.st

        clu1     = np.delete(clu0, duped_spikes, axis=0)
        st1      = np.delete(st0,  duped_spikes, axis=0)
        st_times = st1[:, 0].astype(int) if st1.ndim > 1 else st1.astype(int)

        print('Running cosine merge...')
        merge_unit_groups = _template_cosine_merge(
            Wall0, clu1, st_times, iCC,
            cosine_thresh=cosine_thresh,
            ccg_thresh=ccg_thresh,
            min_spikes_seed=min_spikes_seed,
            min_spikes_pair=min_spikes_pair,
            fs=fs,
        )
        print(f'Found {len(merge_unit_groups)} merge groups')

        remove_unit_ids = []
        if enable_redundant_removal:
            remove_unit_ids = _flag_redundant_units(clu1, st_times, fs=fs)
            print(f'Flagging {len(remove_unit_ids)} redundant units for removal')

        np.save(npy_path, {
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
    bic_margin=0.0,
    ccg_thresh=0.5,
    min_spikes_seed=100,
    min_spikes_pair=100,
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
        iCC   = np.asarray(ops['iCC'], dtype=int)
        clu0  = np.load(oldphypath / 'spike_clusters.npy')
        Wall0 = np.load(oldphypath / 'Wall.npy')
        st0   = ks4_results.st
        amps0 = ks4_results.spike_amplitudes   # (n_kept_spikes,)

        clu1     = np.delete(clu0,  duped_spikes, axis=0)
        st1      = np.delete(st0,   duped_spikes, axis=0)
        amps1    = np.delete(amps0, duped_spikes, axis=0)
        st_times = st1[:, 0].astype(int) if st1.ndim > 1 else st1.astype(int)

        print('Running amplitude BIC merge...')
        merge_unit_groups = _amplitude_bic_merge(
            amps1, clu1, st_times, iCC, Wall0,
            bic_margin=bic_margin,
            ccg_thresh=ccg_thresh,
            min_spikes_seed=min_spikes_seed,
            min_spikes_pair=min_spikes_pair,
            fs=fs,
        )
        print(f'Found {len(merge_unit_groups)} merge groups')

        remove_unit_ids = []
        if enable_redundant_removal:
            remove_unit_ids = _flag_redundant_units(clu1, st_times, fs=fs)
            print(f'Flagging {len(remove_unit_ids)} redundant units for removal')

        np.save(npy_path, {
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

        np.save(npy_path, {
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


def load_cur(cache_dir):
    return np.load(cache_dir)
