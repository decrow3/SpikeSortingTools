#%%
#Curation with the SortingAnalyzer, to clean up the sorting results

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
from tqdm import tqdm
from spikeinterface import create_sorting_analyzer
from spikeinterface.curation import remove_duplicated_spikes
from spikeinterface.curation import remove_redundant_units
from spikeinterface import create_sorting_analyzer
from spikeinterface.curation import compute_merge_unit_groups
from spikeinterface.curation import auto_merge_units
from spikeinterface.curation import apply_curation
from spikeinterface.curation import find_redundant_units
from spikeinterface.core.template_tools import get_template_extremum_channel_peak_shift, get_template_amplitudes
from spikeinterface.postprocessing import align_sorting
from spikeinterface.exporters.to_phy import export_to_phy
from spikeinterface.extractors import read_phy
from spikeinterface.sorters import KilosortSorter
import os
from spikeinterface.core.sparsity import ChannelSparsity, estimate_sparsity


def automerge(analyzer):
    #Biggest issue is temporal shifts temporal_splits

    # some extensions are required
    # analyzer.compute(["random_spikes", "templates", "template_similarity", "correlograms"])
    # analyzer.compute("unit_locations", method="monopolar_triangulation")

    # presence_distance_thresh = [100]
    # presets = ["temporal_splits"] * len(presence_distance_thresh)
    # steps_params = [
    #     {"presence_distance": {"presence_distance_thresh": i}}
    #     for i in presence_distance_thresh
    # ]
    

    # # template_diff_thresh = [0.05, 0.15, 0.25]
    # # presets += ["x_contaminations"] * len(template_diff_thresh)
    # # steps_params += [
    # #     {"template_similarity": {"template_diff_thresh": i}}
    # #     for i in template_diff_thresh
    # # ]

    # compute_merge_args={
    #     "preset": presets,
    #     "steps_params": steps_params,
    #     "recursive": True
    # }
    compute_merge_args={
        "preset": "temporal_splits"
    }#    "recursive": True
    #} #     "presence_distance_thresh": [100],
    analyzer_merged = auto_merge_units(
        sorting_analyzer=analyzer,
        compute_merge_kwargs=compute_merge_args
    )

    # merge_unit_groups = get_potential_auto_merge(
    # analyzer=analyzer,
    # preset="similarity_correlograms",
    # resolve_graph=True
    # )

    # # here we apply the merges
    # analyzer_merged = analyzer.merge_units(merge_unit_groups=merge_unit_groups)
    return analyzer_merged

def get_default_job_kwargs():
    n_cpus = os.cpu_count()
    n_cpus = n_cpus if n_cpus is not None else 1
    n_jobs = max(1, n_cpus - 1) 
    job_kwargs = dict(n_jobs=n_jobs, 
                      chunk_duration='2s', 
                      progress_bar=True,)
    return job_kwargs

def run_cur(seg, ks4_sorter, cache_dir, recalc=False, job_kwargs={}):
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

    job_kwargs = dict(get_default_job_kwargs(), **job_kwargs)

    analyzer = create_sorting_analyzer(sorting=ks4_sorter, recording=seg, return_scaled=True, n_jobs=job_kwargs['n_jobs'], chunk_duration=job_kwargs['chunk_duration'], progress_bar=job_kwargs['progress_bar'])
    # # some extensions are required
    analyzer.compute(["random_spikes", "templates", "template_similarity", "correlograms"], n_jobs=job_kwargs['n_jobs'], chunk_duration=job_kwargs['chunk_duration'], progress_bar=job_kwargs['progress_bar'])
    analyzer.compute("unit_locations", method="monopolar_triangulation", n_jobs=job_kwargs['n_jobs'], chunk_duration=job_kwargs['chunk_duration'], progress_bar=job_kwargs['progress_bar'])

    if npy_path.exists() and not recalc:
        curation_todo_wrapped = np.load(npy_path, allow_pickle=True)
        curation_todo = curation_todo_wrapped.item()  # Extract the dictionary from the NumPy array
        merge_unit_groups = curation_todo['merge_unit_groups']
        remove_unit_ids = curation_todo['removed_units']
        #return curation_todo
    else:
        merge_unit_groups = compute_merge_unit_groups(analyzer,preset="temporal_splits", presence_distance=100, **job_kwargs)  #presence_distance_thresh=100
        

        #redundant, bad units
        remove_unit_ids = []

        #copying from remove_redundant_units, but without applying the removal (yet)
        remove_strategy = "minimum_shift"
        peak_sign="neg"

        unit_peak_shifts = get_template_extremum_channel_peak_shift(analyzer)
        sorting_aligned = align_sorting(sorting=ks4_sorter, unit_peak_shifts=unit_peak_shifts)
        redundant_unit_pairs= find_redundant_units(sorting=sorting_aligned, delta_time = 0.4, agreement_threshold=0.2, duplicate_threshold=0.8)

        if remove_strategy in ("minimum_shift", "highest_amplitude"):
            # this is the values at spike index !
            peak_values = get_template_amplitudes(analyzer, peak_sign=peak_sign, mode="at_index",return_scaled=True)
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
            "merge_unit_groups": merge_unit_groups,
            "removed_units":remove_unit_ids,
        }

        #Need to get to at least here    
        np.save(npy_path, curation_todo, allow_pickle=True)
        #ideally save to cluster_info.tsv and cluster_group.tsv
        #export_to_phy(analyzer, cache_dir / 'clean_sorting_analyzer_phy')


    # analyzer.compute(["waveforms", "templates"]) #phy needs waveforms to be computed
    # export_to_phy(analyzer, cache_dir / 'clean_sorting_analyzer_phy',copy_binary=False, compute_pc_features=False)
    


    # Prepare curation dictionary
    label_definitions={
        "quality": {
            "label_options": [
                "good",
                "noise",
                "mua",
                "artifact"
            ],
            "exclusive": True
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
    
    sparsity= ChannelSparsity.from_best_channels(analyzer, 75, peak_sign='neg')
    clean_analyzer=apply_curation(analyzer, sparsity=sparsity,curation_dict=curation_dict, merging_mode= "hard", n_jobs=job_kwargs['n_jobs'], chunk_duration=job_kwargs['chunk_duration'], progress_bar=job_kwargs['progress_bar'])

    clean_analyzer.compute(["waveforms", "templates"]) #export to phy needs waveforms to be recomputed???
    export_to_phy(clean_analyzer, cache_dir / 'clean_sorting_analyzer_phy',copy_binary=False, compute_pc_features=False, ChannelSparsity=sparsity, add_quality_metrics=True)

    return curation_todo

    # # merge units with similar templates and correlograms
    # analyzer.compute("waveforms")
    # analyzer_merged=automerge(analyzer)

    # #these shouldn't need to be recomputed, they should inherit the analyzer_merged
    # # analyzer_merged.compute(["random_spikes", "templates", "template_similarity", "correlograms"])
    # # analyzer_merged.compute("unit_locations", method="monopolar_triangulation")
    
    # # remove redundant units from SortingAnalyzer object
    # # note this returns a cleaned sorting
    # clean_sorting = remove_redundant_units(
    #     analyzer_merged,
    #     duplicate_threshold=0.9,
    #     remove_strategy="minimum_shift"
    # )
    # # in order to have a SortingAnalyer with only the non-redundant units one must
    # # select the designed units remembering to give format and folder if one wants
    # # a persistent SortingAnalyzer.
    # clean_sorting_analyzer = analyzer_merged.select_units(clean_sorting.unit_ids)

    # clean_sorting_analyzer.save_as(format="binary_folder", folder = cache_dir / 'clean_sorting_analyzer')
    # #apply_curation(ks4_sorter, curation_dict=)
    # export_to_phy(clean_sorting_analyzer, cache_dir / 'clean_sorting_analyzer_phy')
    # cur_sorting = read_phy(cache_dir / 'clean_sorting_analyzer') # Pull from output directory
    # return cur_sorting

    # cur_results = {}

    # spike_samples = clean_sorting_analyzer.spike_times
 

    #return clean_sorting_analyzer

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
    npy_path = cache_dir / 'cur_todo_phy.npy'
    curation_todo=np.load(npy_path, allow_pickle=True)

    return curation_todo


#%%
