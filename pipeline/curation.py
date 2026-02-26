#%%
#Curation with the SortingAnalyzer, to clean up the sorting results

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
from kilosort.io import load_ops


def remove_duped_spikes(sorter, duped_spikes):
    # I believe it may be this simple
    cleaned_sorter=sorter #Does this actually make a copy, or just another pointer to the sorter object
    len0=len(cleaned_sorter.spikes)
    cleaned_sorter.spikes=np.delete(cleaned_sorter.spikes,duped_spikes)
    print(len(cleaned_sorter.spikes), "remaining of ", len0, "total spikes")

    return cleaned_sorter


def run_cur(seg, ks4_sorter, ks4_results, cache_dir, recalc=False):
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
        duped_spikes=np.nonzero((delta_sp_t<thr_t)&(delta_sp_z<thr_z)&delta_clu)
        print(100*len(duped_spikes[0])/len(sp_t),"%  are duped spikes")

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
    analyzer=[]
    seg=[]

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
    
    ops0=load_ops(oldphypath / 'ops.npy')

    st0=ks4_results.st #np.load(oldphypath / 'spike_times.npy')
    clu0=np.load(oldphypath / 'spike_clusters.npy')
    tF0=np.load(oldphypath / 'tF.npy')
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
    tF1=np.delete(tF00, duped_spikes, axis=0)
    tF1_=np.squeeze(tF1)
    
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
    
    #Saving to Phy    
    save_sorting(ops=ops1,results_dir=newphypath,st=st1,clu=clu_new.astype('int32'),tF=tF1_,Wall=Wall1_,imin=0,tic0=time.time(),save_extra_vars=True)

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

