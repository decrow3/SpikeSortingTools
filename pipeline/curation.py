#%%
#Curation with the SortingAnalyzer, to clean up the sorting results

import shutil
import gc
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
from tqdm import tqdm
from spikeinterface.curation import remove_duplicated_spikes
from spikeinterface.curation import remove_redundant_units
from spikeinterface.curation import curation_tools
from spikeinterface import create_sorting_analyzer
from spikeinterface.curation import compute_merge_unit_groups
from spikeinterface.curation import apply_curation
from spikeinterface.curation import find_redundant_units
from spikeinterface.core.template_tools import get_template_extremum_channel_peak_shift, get_template_amplitudes
from spikeinterface.postprocessing import align_sorting
from spikeinterface.exporters.to_phy import export_to_phy
from spikeinterface.extractors import read_phy
from spikeinterface.sorters import KilosortSorter
from kilosort.run_kilosort import save_sorting
from kilosort.io import load_ops, save_ops
from kilosort.postprocessing import remove_duplicates, compute_spike_positions, make_pc_features
from kilosort.clustering_qr import xy_templates
from kilosort import CCG
import time
import torch
import copy


def _rss_gb():
    '''Current process resident memory, in GB. Cheap, no dependencies.'''
    try:
        with open('/proc/self/status') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1]) / 1024 / 1024
    except Exception:
        pass
    return -1.0


def _memlog(tag):
    # flush=True is essential here: plain print() is block-buffered when
    # stdout is redirected to a file, so without an explicit flush these
    # checkpoints can be silently lost if the process is OOM-killed before
    # the buffer would otherwise have been written out.
    print(f'[MEM] {tag}: {_rss_gb():.2f} GB', flush=True)


def _make_pc_features_lowmem(ops, spike_templates, spike_clusters, tF, max_dd_elements=100_000_000):
    '''
    Memory-bounded reimplementation of kilosort.postprocessing.make_pc_features.

    The vendor implementation builds one dense (nspikes, nchan, nfeatures)
    tensor per cluster, where nchan is the number of unique channels spanned
    by the UNION of all templates belonging to that cluster (via
    `torch.unique(iC[:, ix])`). Before merging, clusters map ~1:1 to
    templates so nchan stays near `ops['nearest_chans']` (small). After
    curation.py's merges, a single cluster can span many original templates
    at different probe locations, pushing nchan up toward the full channel
    count -- and for a cluster with many spikes, that one dense allocation
    can reach tens of GB. This is what has been causing OOM kills on
    heavily-merged sessions even after fixing the save_to_phy redundant
    copy (see _save_to_phy_lowmem).

    This keeps the vendor's exact per-cluster algorithm (same dd
    construction, same mean/norm/argsort selection, same tF/feature_ind
    writes) for typical clusters. Only for a cluster whose dense tensor
    would exceed `max_dd_elements` does it fall back to processing that
    cluster's spikes in bounded-size chunks along the spike axis -- same
    math, same result, just computed piece by piece so peak memory for that
    cluster is bounded by max_dd_elements regardless of its total spike
    count. Verified to produce bit-identical output to the vendor function
    on real data (both the direct and chunked code paths).
    '''
    xy, iC = xy_templates(ops)
    n_templates = iC.shape[1]
    n_clusters = np.unique(spike_clusters).size
    n_chans = ops['nearest_chans']
    feature_ind = np.zeros((n_clusters, n_chans), dtype=np.uint32)

    PID = torch.from_numpy(spike_templates).long()

    for i in np.unique(spike_clusters):
        iunq = np.unique(spike_templates[spike_clusters == i]).astype(int)
        ix = torch.from_numpy(np.zeros(n_templates, bool))
        ix[iunq] = True

        igood = ix[PID].nonzero()[:, 0]
        if len(igood) == 0:
            continue
        pid = PID[igood]
        data = tF[igood]
        nspikes, nchanraw, nfeatures = data.shape
        ichan, imap = torch.unique(iC[:, ix], return_inverse=True)
        nchan = ichan.nelement()
        template_ids = ix.nonzero()[:, 0]

        if nspikes * nchan <= max_dd_elements:
            # Fast path: identical to the vendor's one-shot dense tensor.
            dd = torch.zeros((nspikes, nchan, nfeatures))
            for k, j in enumerate(template_ids):
                ij = torch.nonzero(pid == j)[:, 0]
                dd[ij.unsqueeze(-1), imap[:, k]] = data[ij]
            spike_mean = dd.mean(0)
            chan_norm = torch.linalg.norm(spike_mean, dim=1)
            _, ind = torch.sort(chan_norm, descending=True)
            tF[igood, :] = dd[:, ind[:n_chans], :]
            feature_ind[i, :] = ichan[ind[:n_chans]].cpu().numpy()
            del dd
        else:
            # Chunked path for outlier (huge nspikes*nchan) clusters:
            # same construction as the fast path, but bounded per-chunk.
            chunk_size = max(1, max_dd_elements // max(1, nchan))
            template_ij = [torch.nonzero(pid == j)[:, 0] for j in template_ids]

            total_sum = torch.zeros((nchan, nfeatures))
            for start in range(0, nspikes, chunk_size):
                end = min(start + chunk_size, nspikes)
                dd_chunk = torch.zeros((end - start, nchan, nfeatures))
                for k, ij in enumerate(template_ij):
                    m = (ij >= start) & (ij < end)
                    if not m.any():
                        continue
                    ij_local = ij[m] - start
                    dd_chunk[ij_local.unsqueeze(-1), imap[:, k]] = data[ij[m]]
                total_sum += dd_chunk.sum(0)
                del dd_chunk
            spike_mean = total_sum / nspikes
            chan_norm = torch.linalg.norm(spike_mean, dim=1)
            _, ind = torch.sort(chan_norm, descending=True)
            sel = ind[:n_chans]
            feature_ind[i, :] = ichan[sel].cpu().numpy()

            for start in range(0, nspikes, chunk_size):
                end = min(start + chunk_size, nspikes)
                dd_chunk = torch.zeros((end - start, nchan, nfeatures))
                for k, ij in enumerate(template_ij):
                    m = (ij >= start) & (ij < end)
                    if not m.any():
                        continue
                    ij_local = ij[m] - start
                    dd_chunk[ij_local.unsqueeze(-1), imap[:, k]] = data[ij[m]]
                tF[igood[start:end], :] = dd_chunk[:, sel, :]
                del dd_chunk

    tF = torch.permute(tF, (0, 2, 1))
    return tF, feature_ind


def remove_duped_spikes(sorter, duped_spikes):
    # I believe it may be this simple
    #cleaned_sorter=sorter #Does this actually make a copy, or just another pointer to the sorter object
    cleaned_sorter = copy.deepcopy(sorter)

    duped_spikes = np.asarray(duped_spikes, dtype=int)
    len0=len(cleaned_sorter.spikes)

    # Ensure we delete along the spike axis (not flatten)
    cleaned_sorter.spikes = np.delete(cleaned_sorter.spikes, duped_spikes, axis=0)

    print(len(cleaned_sorter.spikes), "remaining of ", len0, "total spikes")

    return cleaned_sorter


def _save_to_phy_lowmem(st, clu, tF_kept, kept_spikes, Wall, probe, ops, imin,
                         results_dir=None, data_dtype=None, save_extra_vars=False,
                         save_preprocessed_copy=False):
    '''
    Memory-optimized reimplementation of kilosort.io.save_to_phy.

    The vendor implementation internally does `tF = tF[kept_spikes]`, which
    creates a second full-size copy of the (very large) PC-feature tensor
    while the caller's copy is still alive -- roughly doubling peak memory
    for this step, which is what has been triggering repeated OOM kills on
    large sessions. This version instead requires the caller to have
    *already* sliced tF down to kept_spikes (and dropped its own reference
    to the unsliced tensor, e.g. via `del`+`gc.collect()`) before calling
    in, so only one full-size copy of tF is ever resident at once.

    `kept_spikes` must be exactly the mask returned by
    `kilosort.postprocessing.remove_duplicates(st[:,0].astype('int64') + imin,
    clu.astype('int32'), dt=ops['duplicate_spike_bins'])`, and `tF_kept`
    must equal `tF_full[kept_spikes]` for the corresponding full tF.

    All outputs match kilosort.io.save_to_phy's, EXCEPT: 'tF.npy' and
    'full_amp.npy' (only written when save_extra_vars=True) are skipped
    entirely, since correctly reproducing their full-length (pre-dedup)
    semantics would require materializing the full-size tF again -- exactly
    what this function avoids. Nothing in this pipeline reads either file
    back; they were vendor debugging/backup extras only.
    '''
    _memlog('_save_to_phy_lowmem: entry')
    if results_dir is None:
        results_dir = ops['data_dir'].joinpath('kilosort4')
    results_dir = Path(results_dir)
    results_dir.mkdir(exist_ok=True)

    # probe properties
    chan_map = probe['chanMap']
    channel_positions = np.stack((probe['xc'], probe['yc']), axis=-1)
    np.save((results_dir / 'channel_map.npy'), chan_map)
    np.save((results_dir / 'channel_positions.npy'), channel_positions)
    np.save((results_dir / 'channel_shanks.npy'), probe['kcoords'])

    # whitening matrix
    whitening_mat = ops['Wrot']
    np.save((results_dir / 'whitening_mat_dat.npy'), whitening_mat.cpu())
    whitening_mat_inv = torch.inverse(
        whitening_mat
        + 1e-5 * torch.eye(whitening_mat.shape[0]).to(whitening_mat.device)
        )
    np.save((results_dir / 'whitening_mat.npy'), whitening_mat.cpu())
    np.save((results_dir / 'whitening_mat_inv.npy'), whitening_mat_inv.cpu())

    # spike properties -- kept_spikes precomputed by caller, tF_kept already sliced
    spike_times_full = st[:, 0].astype('int64') + imin
    spike_templates_full = st[:, 1].astype('int32')
    spike_clusters_full = clu

    spike_times = spike_times_full[kept_spikes]
    spike_clusters = spike_clusters_full[kept_spikes]
    spike_templates = spike_templates_full[kept_spikes]
    st_kept = st[kept_spikes]  # small (n_kept,3); only used for the st[:,1] lookup below

    _memlog('_save_to_phy_lowmem: before compute_spike_positions')
    xs, ys = compute_spike_positions(st_kept, tF_kept, ops)
    spike_positions = np.vstack([xs, ys]).T
    _memlog('_save_to_phy_lowmem: before amplitude calc')
    # equivalent to (full-tF amplitudes)[kept_spikes], since this op is per-spike independent
    amp = ((tF_kept ** 2).sum(axis=(-2, -1)) ** 0.5).cpu().numpy()
    _memlog('_save_to_phy_lowmem: after amplitude calc')

    np.save((results_dir / 'spike_times.npy'), spike_times)
    np.save((results_dir / 'spike_templates.npy'), spike_clusters)
    np.save((results_dir / 'spike_clusters.npy'), spike_clusters)
    np.save((results_dir / 'spike_positions.npy'), spike_positions)
    np.save((results_dir / 'spike_detection_templates.npy'), spike_templates)
    np.save((results_dir / 'amplitudes.npy'), amp)
    np.save((results_dir / 'kept_spikes.npy'), kept_spikes)

    # template properties
    similar_templates = CCG.similarity(Wall, ops['wPCA'].contiguous(), nt=ops['nt'])
    template_amplitudes = ((Wall ** 2).sum(axis=(-2, -1)) ** 0.5).cpu().numpy()
    templates = (Wall.unsqueeze(-1).cpu() * ops['wPCA'].cpu()).sum(axis=-2).numpy()
    templates = templates.transpose(0, 2, 1)
    templates_ind = np.tile(np.arange(Wall.shape[1])[np.newaxis, :], (templates.shape[0], 1))
    np.save((results_dir / 'similar_templates.npy'), similar_templates)
    np.save((results_dir / 'templates.npy'), templates)
    np.save((results_dir / 'templates_ind.npy'), templates_ind)

    # pc features -- tF_kept is already the correct (kept-only) slice, no redundant copy here
    _memlog('_save_to_phy_lowmem: before make_pc_features')
    pc_features, pc_feature_ind = _make_pc_features_lowmem(
        ops, spike_templates, spike_clusters, tF_kept
        )
    _memlog('_save_to_phy_lowmem: after make_pc_features, before np.save(pc_features.npy)')
    np.save(results_dir / 'pc_features.npy', pc_features)
    _memlog('_save_to_phy_lowmem: after np.save(pc_features.npy)')
    np.save(results_dir / 'pc_feature_ind.npy', pc_feature_ind)

    # contamination ratio
    acg_threshold = ops['settings']['acg_threshold']
    ccg_threshold = ops['settings']['ccg_threshold']
    is_ref, est_contam_rate = CCG.refract(spike_clusters, spike_times / ops['fs'],
                                          acg_threshold=acg_threshold,
                                          ccg_threshold=ccg_threshold)

    # write properties to *.tsv
    stypes = ['ContamPct', 'Amplitude', 'KSLabel']
    ks_labels = [['mua', 'good'][int(r)] for r in is_ref]
    props = [est_contam_rate * 100, template_amplitudes, ks_labels]
    for stype, prop in zip(stypes, props):
        with open((results_dir / f'cluster_{stype}.tsv'), 'w') as f:
            f.write(f'cluster_id\t{stype}\n')
            for i, p in enumerate(prop):
                if stype != 'KSLabel':
                    f.write(f'{i}\t{p:.1f}\n')
                else:
                    f.write(f'{i}\t{p}\n')
        if stype == 'KSLabel':
            shutil.copyfile((results_dir / f'cluster_{stype}.tsv'),
                            (results_dir / f'cluster_group.tsv'))

    # params.py
    dtype = "'int16'" if data_dtype is None else f"'{data_dtype}'"
    params = {
        'n_channels_dat': ops['settings']['n_chan_bin'],
        'offset': 0,
        'sample_rate': ops['settings']['fs']
        }
    if save_preprocessed_copy:
        dat_path = results_dir / 'temp_wh.dat'
        params['dtype'] = "'int16'"
        params['hp_filtered'] = True
        params['dat_path'] = f"'{dat_path.resolve().as_posix()}'"
    else:
        dat_path = Path(ops['settings']['filename'])
        params['dtype'] = dtype
        params['hp_filtered'] = False
        params['dat_path'] = f"'{dat_path.resolve().as_posix()}'"

    with open((results_dir / 'params.py'), 'w') as f:
        for key in params.keys():
            f.write(f'{key} = {params[key]}\n')

    if save_extra_vars:
        # NOTE: tF.npy and full_amp.npy intentionally omitted -- see docstring
        np.save(results_dir / 'Wall.npy', Wall.cpu().numpy())
        np.save(results_dir / 'full_st.npy', st)
        np.save(results_dir / 'full_clu.npy', clu)

    phy_cache_path = Path(results_dir / '.phy')
    if phy_cache_path.is_dir():
        shutil.rmtree(phy_cache_path)

    _memlog('_save_to_phy_lowmem: before return')
    return results_dir, similar_templates, is_ref, est_contam_rate, kept_spikes


def save_sorting_lowmem(ops, results_dir, st, clu, tF_kept, kept_spikes, Wall, imin,
                         tic0=np.nan, save_extra_vars=False, save_preprocessed_copy=False,
                         probe=None):
    '''
    Drop-in, low-memory replacement for kilosort.run_kilosort.save_sorting.

    Unlike the vendor version, this expects the CALLER to have already
    computed `kept_spikes` (via `kilosort.postprocessing.remove_duplicates`)
    and sliced `tF` down to it -- and to have dropped its own reference to
    the unsliced tensor (`del`+`gc.collect()`) BEFORE calling in.

    This matters because Python keeps a caller's local variables alive for
    the entire duration of a nested call regardless of what the callee does
    internally -- so slicing/freeing tF *inside* this function would not
    actually free anything as long as the caller still holds its own
    reference to the unsliced tensor. The caller must do it before calling.
    See `_save_to_phy_lowmem` docstring for what else this changes.
    '''
    import logging
    logger = logging.getLogger('kilosort.run_kilosort')

    if probe is None:
        probe = ops['probe']

    logger.info(' ')
    logger.info('Saving to phy and computing refractory periods (low-memory export)')
    logger.info('-' * 40)

    results_dir, similar_templates, is_ref, est_contam_rate, kept_spikes = \
        _save_to_phy_lowmem(
            st, clu, tF_kept, kept_spikes, Wall, probe, ops, imin,
            results_dir=results_dir, data_dtype=ops['data_dtype'],
            save_extra_vars=save_extra_vars,
            save_preprocessed_copy=save_preprocessed_copy
            )
    logger.info(f'{int(is_ref.sum())} units found with good refractory periods')

    runtime = time.time() - tic0
    seconds = runtime % 60
    mins = runtime // 60
    hrs = mins // 60
    mins = mins % 60
    logger.info(f'Total runtime: {runtime:.2f}s = {int(hrs):02d}:'
                f'{int(mins):02d}:{round(seconds)} h:m:s')
    ops['runtime'] = runtime
    save_ops(ops, results_dir)
    logger.info(f'Sorting output saved in: {results_dir}.')

    return ops, similar_templates, is_ref, est_contam_rate, kept_spikes


def run_cur(
    seg,
    ks4_sorter,
    ks4_results,
    cache_dir,
    recalc=False,
    split_depth_export=False,
    depth_overlap_um=75.0,
    depth_split_um=None,
):
    '''
    Run the curation pipeline on the given sorted data.
    
    Parameters
    ----------
    seg: spikeinterface recording segment
        The recording segment which was sorted. Used to extract waveforms and other data.
    sorter: Kilosort sorter
        The sorter used to sort the data. 
    
    Returns
    -------
    cur_results: dict
        The results of the quality control pipeline
    '''

    if isinstance(cache_dir, str):
        cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    npy_path = cache_dir / 'cur_todo_phy.npy'
    #analyzer, cache_dir / 'clean_sorting_analyzer_phy'
    if npy_path.exists() and not recalc:
        curation_todo_wrapped = np.load(npy_path, allow_pickle=True)
        curation_todo=curation_todo_wrapped.item()
        merge_unit_groups=curation_todo['merge_unit_groups']
        remove_unit_ids=curation_todo['removed_units']
        duped_spikes=curation_todo['duped_spikes']

    else:    
        #Identify duplicated spikes
        clu=ks4_results.spike_clusters
        sp_z= ks4_results.spike_positions[:,1]
        sp_t=ks4_results.spike_times
    
        thr_z=150 #150 microns on zaxis
        thr_t=1 #0.000033*30000 # .033ms
        delta_sp_t=np.diff(sp_t)
        delta_sp_z=np.diff(sp_z)
        delta_clu=np.diff(clu)!=0 # not in same cluster

        #np.sum((delta_sp_t<thr_t)/len(sp_t))
        #duped_spikes=np.nonzero((delta_sp_t<thr_t)&(delta_sp_z<thr_z)&delta_clu)
        # np.nonzero returns a tuple; use flatnonzero to get a 1D index array
        duped_spikes = np.flatnonzero((delta_sp_t < thr_t) & (delta_sp_z < thr_z) & delta_clu)
        print(100*len(duped_spikes)/len(sp_t),"%  are duped spikes")

        #Search for spikes that might be duplicated across different units, that are unlikely to be actually different spikes, but may prevent merges
        #duped_spikes=curation_tools.find_duplicated_spikes(ks4_results.spike_times,(0.0001)*30000,"first") #.1ms
        ks4_sorter_clean=remove_duped_spikes(ks4_sorter, duped_spikes)

        analyzer = create_sorting_analyzer(sorting=ks4_sorter_clean, recording=seg)
        # # some extensions are required
        analyzer.compute(["random_spikes", "templates", "template_similarity", "correlograms"])
        analyzer.compute("unit_locations", method="monopolar_triangulation")

        #Merge units that are likely to be the same based on trade off in time
        #* | "temporal_splits": focused on finding temporal splits using presence distance.
        #  | It uses the following steps: "num_spikes", "remove_contaminated", "unit_locations",
        #  | "template_similarity", "presence_distance", "quality_score"
        #merge_unit_groups = compute_merge_unit_groups(analyzer,preset="temporal_splits", presence_distance=100)
        
        # 20-9-2025. Note I'm worried that might miss other types of merges, like similarity_correlograms
        #       * | "similarity_correlograms": mainly focused on template similarity and correlograms.
        #   | It uses the following steps: "num_spikes", "remove_contaminated", "unit_locations",
        #   | "template_similarity", "correlogram", "quality_score"


        #setting ccg threshould be 0.75, not 0.16, to account for long recordings where similar/same units trade off but have shared spikes
        #merge_unit_groups = compute_merge_unit_groups(analyzer,preset="similarity_correlograms", correlogram={"corr_diff_thresh": 0.75})

        #for ccg a higher value threshold means more merges, so 0.75 is more permissive than 0.16??
        # I may have this backwards, 1 is identical, 0 is uncorrelated so higher threshold means they have to be more similar, so fewer merges??
        # * "correlogram": the cross-correlograms of the two units are similar to each auto-corrleogram (`corr_diff_thresh`)
        # but merges happen if  (correlogram_diff < params["corr_diff_thresh"]) so higher threshold means more merges??
        # is it possibly 1-ccg that is used?? so that 0 is identical, 1 is uncorrelated
        # So low threshold means they have to be more similar, so fewer merges?? trying 0.25
        #merge_unit_groups = compute_merge_unit_groups(analyzer,preset="similarity_correlograms", correlogram={"corr_diff_thresh": 0.25})

        # NEW WAY 2026-02-24
        # Running multiple analyses that target different artifacts:

        # "similarity_correlograms": Catches units that coexist in time (e.g., amplitude splits, bursting) by verifying their cross-correlation looks like a single-unit auto-correlation.
        # "temporal_splits": Catches units that drift in/out of existence (e.g., drift loops) by verifying they are distinct in time but similar in shape/location.
        # Since these issues are largely orthogonal (one implies temporal overlap, the other implies temporal disjointness), you can run them sequentially or combine their results.
        
        # 1. Detect Temporal Splits
        # High threshold (100) ensures they are very clearly separated in time
        merges_temporal = compute_merge_unit_groups(
            analyzer, 
            preset="temporal_splits", 
            presence_distance={"presence_distance_thresh": 100}
        )
        print(f"Found {len(merges_temporal)} temporal merges.")

        # 2. Detect Similarity/CCG Splits
        merges_ccg = compute_merge_unit_groups(
            analyzer,
            preset="similarity_correlograms", 
            correlogram={"corr_diff_thresh": 0.25}
        )
        print(f"Found {len(merges_ccg)} CCG/Similarity merges.")

        # 3. Consolidate Merges
        # We need to combine these two lists. 
        # Since 'compute_merge_unit_groups' returns a list of lists (e.g. [[1, 2], [3, 4, 5]]),
        # we can just concatenate them and use a graph tool to resolve chaining.
        
        raw_merge_list = merges_temporal + merges_ccg
        
        # Use simple NetworkX or internal tool to resolve "A-B" and "B-C" -> "A-B-C"
        # Since 'resolve_merging_graph' is imported from 'curation_tools' (check imports), use it if available.
        # Otherwise, a simple connected components implementation works:
        
        import networkx as nx
        g = nx.Graph()
        # Add all unit IDs as nodes (optional, but good for completeness)
        # Add edges for every merge group
        for group in raw_merge_list:
            if len(group) > 1:
                # Add edges between the first element and all others (implies full connectivity in component)
                for u in group[1:]:
                    g.add_edge(group[0], u)
        
        # Extract the connected components (these are the final merged groups)
        final_merge_groups = [list(c) for c in nx.connected_components(g) if len(c) > 1]
        
        print(f"Final resolved merge groups: {len(final_merge_groups)}")
        
        merge_unit_groups = final_merge_groups

        #merges happen if  (correlogram_diff < params["corr_diff_thresh"]) so


        #default params:
        #         _default_step_params = {
        #     "num_spikes": {"min_spikes": 100},
        #     "snr": {"min_snr": 2},
        #     "remove_contaminated": {"contamination_thresh": 0.2, "refractory_period_ms": 1.0, "censored_period_ms": 0.3},
        #     "unit_locations": {"max_distance_um": 150},
        #     "correlogram": {
        #         "corr_diff_thresh": 0.16,
        #         "censor_correlograms_ms": 0.15,
        #         "sigma_smooth_ms": 0.6,
        #         "adaptative_window_thresh": 0.5,
        #     },
        #     "template_similarity": {"template_diff_thresh": 0.25},
        #     "presence_distance": {"presence_distance_thresh": 100},
        #     "knn": {"k_nn": 10},
        #     "cross_contamination": {
        #         "cc_thresh": 0.1,
        #         "p_value": 0.2,
        #         "refractory_period_ms": 1.0,
        #         "censored_period_ms": 0.3,
        #     },
        #     "quality_score": {"firing_contamination_balance": 1.5, "refractory_period_ms": 1.0, "censored_period_ms": 0.3},
        # }



        #redundant, bad units
        remove_unit_ids = []

        #copying from remove_redundant_units, but without applying the removal (yet)
        remove_strategy = "minimum_shift"
        peak_sign="neg"

        unit_peak_shifts = get_template_extremum_channel_peak_shift(analyzer)
        sorting_aligned = align_sorting(sorting=ks4_sorter_clean, unit_peak_shifts=unit_peak_shifts)
        redundant_unit_pairs= find_redundant_units(sorting=sorting_aligned, delta_time = 0.4, agreement_threshold=0.2, duplicate_threshold=0.8)
            #Just the main sorter data 'spikes.npy'

        #Always remove existing folder to avoid confusion, but this should be empty if recalc=False
        if (cache_dir / 'cur_sorter_output').exists():
            shutil.rmtree(cache_dir / 'cur_sorter_output')

        ks4_sorter_clean.save_to_folder(cache_dir / 'cur_sorter_output')

        if remove_strategy in ("minimum_shift", "highest_amplitude"):
            # this is the values at spike index !
            peak_values = get_template_amplitudes(analyzer, peak_sign=peak_sign, mode="at_index")
            peak_values = {unit_id: np.max(np.abs(values)) for unit_id, values in peak_values.items()}

        if remove_strategy == "minimum_shift":
            #assert align, "remove_strategy with minimum_shift needs align=True"
            for u1, u2 in redundant_unit_pairs:
                if np.abs(unit_peak_shifts[u1]) > np.abs(unit_peak_shifts[u2]):
                    remove_unit_ids.append(u1)
                elif np.abs(unit_peak_shifts[u1]) < np.abs(unit_peak_shifts[u2]):
                    remove_unit_ids.append(u2)
                else:
                    # equal shift use peak values
                    if np.abs(peak_values[u1]) < np.abs(peak_values[u2]):
                        remove_unit_ids.append(u1)
                    else:
                        remove_unit_ids.append(u2)
    


        curation_todo = {
            "duped_spikes": duped_spikes,
            "merge_unit_groups": merge_unit_groups,
            "removed_units":remove_unit_ids,
        }

        np.save(npy_path, curation_todo, allow_pickle=True)
        #ideally save to cluster_info.tsv and cluster_group.tsv
        #export_to_phy(analyzer, cache_dir / 'clean_sorting_analyzer_phy')


    # analyzer.compute(["waveforms", "templates"]) #phy needs waveforms to be computed
    # export_to_phy(analyzer, cache_dir / 'clean_sorting_analyzer_phy',copy_binary=False, compute_pc_features=False)
    
    # clear some memory before continuing
    if "analyzer" in locals():
        del analyzer
    if "seg" in locals():
        del seg

    # Prepare curation dictionary
    label_definitions={
        "quality": {
            "label_options": [
                "good",
                "noise",
                "mua",
                "artifact"
            ],
            "exclusive": "true"
        }
    }

    ks_labels = ks4_sorter.get_property('KSLabel')
    ks_ids=ks4_sorter.unit_ids

    #Remove overlapping units
    flat_list= [item for sublist in merge_unit_groups for item in sublist]
    setmerge=set(flat_list)
    setrem=set(remove_unit_ids)
    keeprem=list(setrem-setmerge)

    #Make dict of unit_ids and labels for curation_dict
    manual_labels_dict = {"unit_id": [], "quality": []}#define this as a dictionary outside of the loop
    unit_ids_list = []
    manual_labels_list=[]
    for i in range(len(ks_ids)):
        unit_ids_list.append((ks_ids[i]))
        manual_labels_list.append({"unit_id": (ks_ids[i]), "quality": [ks_labels[i]]})

    curation_dict = {
        "format_version": "1",
        "unit_ids": unit_ids_list,
        "label_definitions": label_definitions,
        "manual_labels": manual_labels_list, #curation_dict is trying to use lbl.get() but numpy.str object has no attribute get #need to add unit_ids to this, or change curation_dict behavior
        "merge_unit_groups": merge_unit_groups,
        "removed_units":keeprem,
        "merging_mode": "hard",
        "censor_ms": 0.25
    }
    
    # No great need to use this:
    # Clean_analyzer=apply_curation(analyzer, curation_dict=curation_dict), 
    # clean_analyzer.compute(["waveforms", "templates"]) #phy needs waveforms to be computed
    # export_to_phy(clean_analyzer, cache_dir / 'clean_sorting_analyzer_phy',copy_binary=False, compute_pc_features=False)

    # We can manually merge clusters in KS_results and save out to a .csv file for phy

    
    # JUST DUPED SPIKES AND MERGES SO FAR!
    # Need to pull from phy format, apply curations, and resave into phy format

    # 1) Pull all phy datafiles that have one axis n_spikes, need to remove duped spikes
    pipeline_dir=cache_dir.parent
    oldphypath = pipeline_dir / 'kilosort4/sorter_output/'
    newphypath = cache_dir / 'cur_sorter_output/'

    # ops0_wrapped=np.load(oldphypath / 'ops.npy',allow_pickle=True)
    # ops0=ops0_wrapped.item()
    
    _memlog('before loading ops/st0/clu0')
    ops0=load_ops(oldphypath / 'ops.npy')

    st0=ks4_results.st #np.load(oldphypath / 'spike_times.npy')
    clu0=np.load(oldphypath / 'spike_clusters.npy')
    _memlog('before loading tF0 (raw tF.npy)')
    tF0=np.load(oldphypath / 'tF.npy')
    _memlog('after loading tF0')
    Wall0=np.load(oldphypath / 'Wall.npy')
    kept0=np.load(oldphypath / 'kept_spikes.npy')
    kept=np.argwhere(kept0)

    ops1=ops0
    st1=np.delete(st0, duped_spikes, axis=0)
    clu1=np.delete(clu0, duped_spikes, axis=0)
    # tF00=tF0[kept]
    # tF1=np.delete(tF00, duped_spikes, axis=0)
    # tF11=np.squeeze(tF1)
    # import torch
    # tF1_=torch.from_numpy(tF11)

    import torch
    tF0=torch.from_numpy(tF0)
    tF00=tF0[kept]
    _memlog('after tF00 = tF0[kept] (both alive)')
    del tF0, kept, kept0
    gc.collect()
    _memlog('after del tF0 + gc.collect (only tF00 should remain)')
    tF1=np.delete(tF00, duped_spikes, axis=0)
    _memlog('after tF1 = delete(tF00, duped_spikes) (both alive)')
    del tF00
    gc.collect()
    _memlog('after del tF00 + gc.collect (only tF1 should remain)')
    tF1_=np.squeeze(tF1)
    # np.squeeze returns a VIEW (shares tF1's underlying buffer) whenever
    # possible, so tF1 itself must ALSO be deleted -- deleting only tF1_
    # later does not free anything as long as this name still holds a
    # reference to the same buffer. (Confirmed via memory instrumentation:
    # without this, del tF1_ further downstream had zero effect on RSS.)
    del tF1
    gc.collect()
    _memlog('after del tF1 + gc.collect (only tF1_ view should remain)')

    #tF1_=torch.from_numpy(tF11)



    n_groups=len(merge_unit_groups) # number of groups to merge 
    print('Need to merge', n_groups,' groups of clusters')
    newids=np.max(clu0)+range(n_groups)+1 #append new ids, This breaks KS

    Wall1=Wall0.copy() #was Wall1=Wall0, but this is just a pointer, so changes to Wall1 will change Wall0, which we don't want. We want to start with a copy of Wall0 and then remove entries from it, not change the entries in place
    nchan=np.size(Wall0,axis=1)
    ntp=np.size(Wall0,axis=2)
    best_unit_clu=[0]*(n_groups)
    for ii in range(n_groups):
        n_clu=len(merge_unit_groups[ii])
        nspikes=[0]*(n_clu)
        templates=[0]*(n_clu)
        #nspikes[jj]=np.sum(clu1==merge_unit_groups[ii][jj])
        for jj in range(n_clu):
            nspikes[jj]=np.sum(clu1==merge_unit_groups[ii][jj]) #count to decide which waveform to keep
            templates[jj]= np.unique(st1[np.argwhere(clu1==merge_unit_groups[ii][jj]),1])
        best_unit_clu[ii]=merge_unit_groups[ii][np.argmax(nspikes)]
        best_unit_idx=np.argwhere(ks4_sorter.unit_ids==best_unit_clu[ii])


        #Replace references to templates with best template, shouldn't need to do this
        #best_units_tmp=templates[np.argmax(nspikes)]
        #st1[np.argwhere(clu1==merge_unit_groups[ii][jj]),1]=best_units_tmp

        #Wall1[n_clu0+ii]=Wall0[best_unit_idx[0][0],:,:] #copy waveforms into the next slot
        appendthis=np.reshape(Wall0[best_unit_idx[0][0],:,:],newshape=[1,nchan,ntp])
        Wall1=np.append(Wall1,appendthis,0)
    
    n_clu0=len(set(clu0))
    n_clu1=len(set(clu1))
    if n_clu0==n_clu1:
        Wall_remove_idx=[]
    else:
        Wall_remove_idx=list(set(clu0)^set(clu1))
        print(' Need to remove', len(set(Wall_remove_idx)),' clusters')

    for ii in range(n_groups):
        n_clu=len(merge_unit_groups[ii])
        for jj in range(n_clu):
            clu1[np.argwhere(clu1==merge_unit_groups[ii][jj])]=newids[ii]

            #Remove entries in Wall, dim 0
            cluster_change_idx=np.argwhere(ks4_sorter.unit_ids==merge_unit_groups[ii][jj]) #referenced to original size of Wall
            Wall_remove_idx=np.append(Wall_remove_idx,cluster_change_idx)

    print('removing', len(set(Wall_remove_idx)),' clusters')
    #print type of Wall and Wall_remove_idx to make sure they are compatible
    print(type(Wall1), type(Wall_remove_idx))

    # ADDED SAFETY CHECK for Unit ID vs Index
    # Verify that unit IDs map 1:1 to indices [0..N-1]
    # If this fails, deleting by ID will corrupt the Wall matrix.
    max_id = np.max(ks4_sorter.unit_ids)
    if max_id >= len(ks4_sorter.unit_ids) or not np.array_equal(np.sort(ks4_sorter.unit_ids), np.arange(len(ks4_sorter.unit_ids))):
         raise ValueError(
             f"CRITICAL ERROR: Unit IDs are not consecutive 0..N-1 (Max ID: {max_id}, Count: {len(ks4_sorter.unit_ids)}). "
             "Direct deletion from Wall matrix using Unit IDs as indices will fail. "
             "Please remap IDs to row indices before proceeding."
         )

    #<class 'numpy.ndarray'> <class 'list'>
    # Wall1=np.delete(Wall1,Wall_remove_idx.astype(int),axis=0)
    Wall1 = np.delete(Wall1, np.array(Wall_remove_idx).astype(int), axis=0)
    Wall1_=torch.from_numpy(Wall1)


    #unfortunatley the internal KS save_to_phy needs clus to be a single continous matrix [0,nclus]
    #clu is referenced by tF? Wmat? st1[:,2]?, I think just clu?
    [unique_clus, clu_new]=np.unique(clu1,return_inverse=True)

    n_clu_new =len(unique_clus)
    n_clu_mat=Wall1[:,0,0].shape

    assert int(n_clu_new) == int(n_clu_mat[0])


    # #Testing format
    # tF0_=torch.from_numpy(np.squeeze(tF00))
    # Wall0_=torch.from_numpy(Wall0)
    # #Saving to Phy    
    # newphypath0 = cache_dir / 'cur_sorter_output0/'
    # save_sorting(ops=ops0,results_dir=newphypath0,st=st0,clu=clu0,tF=tF0_,Wall=Wall0_,imin=0)

    #spike_templates (n_spikes,) in range [0,559]
    #spike_clusters (n_spikes,) in range [0,550]

    #changes the dimensions of n_clusters but not n_templates for calculating matches,merges etc
    # iU is vector(n_templates,1) to channels on probe??


    # iU0_= ops0['iU']
    # iU= iU0_.cpu().numpy()
    # iU1=np.delete(iU,Wall_remove_idx.astype(int))
    # iU1_=torch.from_numpy(iU1)
    # ops1['iU']=iU1_

    #Need to pass the cluster labels from KS, so that they match the indices along cluster dimension
    #Wall1=do_merges(Wall0,ks_labels,merge_unit_groups,axis=0)
    #Wall1=remove_clus(Wall1,ks_labels,remove_unit_ids,axis=0)
    
    import time
    
    ## NEED TO SPLIT SOMETIMES
    spike_z0 = ks4_results.spike_positions[:, 1]
    spike_z1 = np.delete(spike_z0, duped_spikes, axis=0)  # aligns with st1/clu1

    import time
    import torch

    def _save_subset(out_dir, unit_ids_global):
        unit_ids_global = np.asarray(unit_ids_global, dtype=np.int64)
        if unit_ids_global.size == 0:
            return None

        spike_mask = np.isin(clu_new, unit_ids_global)
        spk_idx = np.flatnonzero(spike_mask)

        st_sub = st1[spk_idx]
        tF_sub = np.asarray(tF1_)[spk_idx]

        # remap cluster ids to contiguous [0..n_units_sub-1]
        u_sub, clu_sub = np.unique(clu_new[spk_idx], return_inverse=True)
        Wall_sub = Wall1[np.asarray(u_sub, dtype=np.int64), :, :]

        out_dir.mkdir(parents=True, exist_ok=True)
        save_sorting(
            ops=ops1,
            results_dir=out_dir,
            st=st_sub,
            clu=clu_sub.astype("int32"),
            tF=torch.as_tensor(tF_sub),
            Wall=torch.as_tensor(Wall_sub),
            imin=0,
            tic0=time.time(),
            save_extra_vars=True,  # keep your full_st/full vars behavior
        )

        np.savez(
            out_dir / "depth_split_meta.npz",
            global_unit_ids=u_sub,
            n_spikes=st_sub.shape[0],
        )
        return out_dir

    if split_depth_export:
        unit_ids = np.unique(clu_new)
        unit_depth = np.array([np.median(spike_z1[clu_new == u]) for u in unit_ids])

        if depth_split_um is None:
            depth_split_um = float(np.median(unit_depth))

        top_units = unit_ids[unit_depth >= (depth_split_um - depth_overlap_um)]
        bot_units = unit_ids[unit_depth <= (depth_split_um + depth_overlap_um)]

        top_dir = _save_subset(newphypath / "depth_top", top_units)
        bot_dir = _save_subset(newphypath / "depth_bot", bot_units)

        from pipeline import KilosortResults
        return {
            "top": KilosortResults(top_dir) if top_dir is not None else None,
            "bot": KilosortResults(bot_dir) if bot_dir is not None else None,
        }

    ##

    #Saving to Phy (original single export, no depth splits)
    # Low-memory path: compute kept_spikes and slice+free tF1_ HERE (in the
    # caller), before calling into save_sorting_lowmem. Freeing it inside
    # that function would not help -- this frame's tF1_ reference would
    # still be alive for the whole nested call regardless.
    _memlog('at start of low-mem export section (tF1_ alive)')
    clu_new_i32 = clu_new.astype('int32')
    spike_times_for_dedup = st1[:, 0].astype('int64')  # imin=0
    _, _, kept_spikes = remove_duplicates(
        spike_times_for_dedup, clu_new_i32, dt=int(ops1['duplicate_spike_bins'])
        )
    print(f'[MEM] kept_spikes: {kept_spikes.sum()} of {kept_spikes.size} '
          f'({100*kept_spikes.sum()/kept_spikes.size:.2f}%)', flush=True)
    tF_kept = tF1_[kept_spikes]
    _memlog('after tF_kept = tF1_[kept_spikes] (both alive)')
    del tF1_
    gc.collect()
    _memlog('after del tF1_ + gc.collect (only tF_kept should remain)')

    save_sorting_lowmem(ops=ops1, results_dir=newphypath, st=st1, clu=clu_new_i32,
                         tF_kept=tF_kept, kept_spikes=kept_spikes, Wall=Wall1_,
                         imin=0, tic0=time.time(), save_extra_vars=True)

    #but phy errors:
    # File "/home/huklab/anaconda3/envs/phy2/lib/python3.11/site-packages/phylib/io/model.py", line 786, in _load_features
    #    assert cols.shape == (self.n_templates, n_channels_loc)

    #SAME FOR TSV FILES FOR PASSING LABELS!! These use the pandas.core.frame.Dataframe
    #RETURN NEW KS_RESUTLS Object for QA

    from pipeline import KilosortResults
    ks4_results_clean = KilosortResults(newphypath) # Pull results from output directory into format expected by qc module


    return ks4_results_clean #ks4_results_clean


    # # All saved data files
    # filelist=os.listdir(oldphypath)
    # npyfile=[False]*len(filelist)
    # tsvfile=[False]*len(filelist)

    # for ii in range(len(filelist)):
    #     file=filelist[ii]
    #     #print(file)
    #     if bool(re.search(".npy",file)):
    #         npyfile[ii]=True
    #     if bool(re.search(".tsv",file)):
    #         tsvfile[ii]=True

    # npylist = [item for item, select in zip(filelist, npyfile) if select]
    # tsvlist = [item for item, select in zip(filelist, tsvfile) if select]

    # n_spikes0=len(sp_t)
    # n_clu0=len(ks4_results.cluster_labels)

    # for ii in range(len(npylist)):
    #     npydata=np.load(phypath / npylist[ii], allow_pickle=True)
    #     print(npylist[ii], npydata.shape)
    #     spfind=([dim==n_spikes0 for dim in npydata.shape])
    #     sp_dim=np.argwhere(spfind)
    #     if any(spfind):
    #         print("remove duped spikes in spike dimension first")
    #         print(sp_dim[0][0])

    #         np.delete(npydata, duped_spikes, axis=sp_dim[0][0])
    #         #Remaining dimensions will now match the 

    #     # 2) Apply curations by changing cluster ids of units
    #     clufind=([dim==n_clu0 for dim in npydata.shape])
    #     clu_dim=np.argwhere(clufind)
        
    #     if any(clufind):
    #         print("then merge clusters from remaining spikes")
    #         print(clu_dim[0][0])
    #         #Need to pass the cluster labels from KS, so that they match the indices along cluster dimension
    #         do_merges(npydata,cluster_dim_labels,merge_unit_groups,axis=clu_dim[0][0])
    #         remove_clus(npydata,cluster_dim_labels,remove_unit_ids,axis=clu_dim[0][0])

    #     # 3) Resave back into phy format.Should probably recompute waveforms/templates etc first    
    #     filesave=np.save(npydata, newphypath / npylist[ii], allow_pickle=True)

    





# def do_merges(data,clus,merge_unit_groups,axis):
#     #changes indices into position along cluster axis
#     merges_indices= find where clus==remove_ids

#     n_groups=len(merge_unit_groups) # number of groups to merge 
#     newids=np.max(clus)+range(n_groups)+1 #append new ids

#     for ii in range(n_groups):
#         n_clu=len(merge_unit_groups(ii))
#         for jj in range(n_clu):

#             find where clus==merge_units

#             nspikes[jj]=sum(clus0)

#             newid
    
#     #How does this work without deleting data
#     return merged_data


# def remove_clus(data,clus,remove_ids,axis):
#     remove_indices= find where clus==remove_ids #changes indices into position along cluster axis

#     removed_data=np.delete(data,remove_indices,axis=axis)
#     return removed_data

def load_cur(cache_dir):
    '''
    Load the quality control results from a given directory.
    
    Parameters
    ----------
    cache_dir: str or Path
        The directory to load the quality control results from.
    
    Returns
    -------
    cur_results: dict
        The quality control results
    '''
    cur_results=np.load(cache_dir)

    return cur_results


# #%% For Reference: kilosorts own merge function
# def merging_function(ops, Wall, clu, st, r_thresh=0.5, mode='ccg', device=torch.device('cuda')):
#     clu2 = clu.copy()
#     clu_unq, ns = np.unique(clu2, return_counts = True)

#     Ww = Wall.to(device)
#     NN = len(Ww)

#     isort = np.argsort(ns)[::-1]

#     is_merged = np.zeros(NN, 'bool')
#     is_good = np.zeros(NN,)

#     acg_threshold = ops['settings']['acg_threshold']
#     ccg_threshold = ops['settings']['ccg_threshold']
#     if mode == 'ccg':
#         is_ref, est_contam_rate = CCG.refract(clu, st/ops['fs'],
#                                               acg_threshold=acg_threshold,
#                                               ccg_threshold=ccg_threshold)

#     nt = ops['nt']
#     W = ops['wPCA'].contiguous()
#     WtW = conv1d(W.reshape(-1, 1,nt), W.reshape(-1, 1 ,nt), padding = nt) 
#     WtW = torch.flip(WtW, [2,])

#     t = 0
#     nmerge = 0
#     while t<NN:
#         #if t%100==0:
#             #print(t, nmerge)

#         kk = clu_unq[isort[t]]

#         if (mode == 'ccg') and is_ref[kk]==0:
#             t += 1
#             continue

#         if is_merged[kk]:            
#             t += 1
#             continue

#         mu = (Ww**2).sum((1,2), keepdims=True)**.5
#         Wnorm = Ww / (1e-6 + mu)

#         UtU = torch.einsum('lk, jlm -> jkm',  Wnorm[kk], Wnorm)
#         ctc = torch.einsum('jkm, kml -> jl', UtU, WtW)

#         cmax = ctc.max(1)[0]
#         cmax[kk] = 0

#         jsort = np.argsort(cmax.cpu().numpy())[::-1]

#         if mode == 'ccg':
#             st0 = st[clu2==kk] / ops['fs']
        
#         is_ccg  = 0
#         for j in range(NN):
#             jj = jsort[j]
#             if cmax[jj] < r_thresh:
#                 break
#             # compare with CCG
#             if mode == 'ccg':
#                 st1 = st[clu2==jj] / ops['fs']
#                 _, is_ccg, _ = CCG.check_CCG(st0, st1, acg_threshold=acg_threshold,
#                                              ccg_threshold=ccg_threshold)        
#             else:
#                 dmu = 2 * (mu[kk] - mu[jj]) / (mu[kk] + mu[jj])
#                 is_ccg = dmu.abs() < 0.2

#             if is_ccg:
#                 is_merged[jj] = 1
#                 Ww[kk] = ns[kk]/(ns[kk]+ns[jj]) * Ww[kk] + ns[jj]/(ns[kk]+ns[jj]) * Ww[jj]            
#                 Ww[jj] = 0

#                 ns[kk] += ns[jj]
#                 ns[jj] = 0
#                 clu2[clu2==jj] = kk            

#                 break

#         if is_ccg==0:            
#             t +=1    
#         else:                
#             nmerge+=1
    
#     imap = np.cumsum((~is_merged).astype('int32')) - 1
#     if imap.size > 0:
#         # Otherwise, everything has been merged into a single cluster
#         clu2 = imap[clu2]

#     Ww = Ww[~is_merged]

#     if mode == 'ccg':
#         is_ref = is_ref[~is_merged]
#     else:
#         is_ref = None

#     return Ww.cpu(), clu2, is_ref

