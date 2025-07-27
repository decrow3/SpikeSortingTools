#%%
from pipeline import condition_signal, correct_motion, plot_motion_output, sort_ks4, save_binary_recording, run_qc, KilosortResults, load_qc, run_cur, load_cur
from spikeinterface.sorters import get_default_sorter_params
from pathlib import Path
import shutil
import gc
# I'm using a pinned version of spikeinterface, so if something doesn't work with the latest version, ask about it
import spikeinterface.full as si

#%% Change this code to load your data
data_dir=   r"/mnt/NPX/Luke/LukeV1-20240717/Luke0717_V1_g0/"

stream_id = "imec0.ap" #usually imec0 is first inserted probe (often V2/MT), imec1 is second probe (often V1)
seg = si.read_spikeglx(folder_path=data_dir, load_sync_channel=False, stream_id=stream_id)# experiment_names="experiment1")

#%% Run on a snippet to check params
# start_time = 0 #lots of motion around 10000s in, but time didn't start at 0?
# stop_time  = start_time + 100
# seg=seg.frame_slice(start_time * 30000, stop_time * 30000) #100 seconds snippet, if really low will need to change n_batches down from 50 to 5 in condition_signal ln137

#%%
# run pipelines
pipeline_dir = Path('/home/huklab/Documents/RyanSorting/SpikeSortingTools/pipeline_results_Luke0717_V1_g0_imec0_thresh1_tst')
pipeline_dir.mkdir(parents=True, exist_ok=True)

#%%
# condition signal runs 1) bad channel detection 2) . Can we also get a noise over time measure over all channels, may need to censor some completely
noise_thresh = 0.3 # higher for spikeGLX, around 0.3

# if uV_per_bit==2.34375: #spikeGLX, tip reference, 1.2mV 
#     uV_thresh=1200 #uV
uV_thresh = .5e3 #uV, 500uV, this is the default for spikeGLX for external reference, but can be changed to 350 or 400uV if you want to remove more saturation
seg_pre = condition_signal(seg, cache_dir=pipeline_dir / 'conditioning', noise_thresh=noise_thresh, uV_thresh=.5e3, recalc=False)

# #%% DEBUG: quick saving out of the preprocessed recording before motion correction
# save_binary_recording(seg_pre, pipeline_dir / 'preprocessed_recording_premotion', recalc=False)

# %% Test data curation step
# from spikeinterface.core import load_extractor
# #pipeline_dir = Path('/home/huklab/Documents/RyanSorting/SpikeSortingTools/pipeline_results')
# seg_saved = load_extractor(pipeline_dir / 'preprocessed_recording')
# ks4_sorter = load_extractor(pipeline_dir / 'kilosort4/sorter')
# ks4_results = KilosortResults(pipeline_dir / 'kilosort4/sorter_output') #load up the kilosort results

# #shutil.rmtree(pipeline_dir / 'cur')
# cur_results = run_cur(seg_saved, ks4_sorter, ks4_results, pipeline_dir / 'cur', recalc=False) # this should save out some merges

#%% Motion issue on SpikeGLX, this may have had more to do with the conditioning failing, kilosort4 is actually more robust??
seg_motion = correct_motion(seg_pre, cache_dir=pipeline_dir / 'motion', recalc=False, method='med')
plot_motion_output(seg_motion, cache_dir=pipeline_dir / 'motion')


#%% Kilosort4 parameters
# OpenEphys
sorter_params = get_default_sorter_params('kilosort4')
sorter_params['do_correction'] = False # Turns off drift correction
sorter_params['save_extra_vars'] = True # required for truncation qc
sorter_params['Th_universal'] = 9
sorter_params['Th_learned'] = 8
sorter_params['duplicate_spike_ms'] = 0.25 #ccgs shouldn't use less than 1ms anyway
sorter_params['ccg_threshold'] = 0.75 #increased from 0.25, to account for long recordings where similar/same units trade off but have shared spikes
sorter_params['nearest_chans'] = 20 #up from 10
sorter_params['nearest_templates'] = 200 #up from 100
sorter_params['max_channel_distance'] = 64 #up from 32
sorter_params['clear_cache'] = True # Necessary on some larger files to prevent CUDA out of memory errors
sorter_params = dict(sorter_params, **sorter_params)

#%% Clear seg, this shouldn't help since files are memory mapped. For memory problems try uhang and enable zswap, also set ulimit -v for oom messages
del seg
del seg_pre
#%% Run Pipeline
try:
    ks4_results = KilosortResults(pipeline_dir / 'kilosort4')
    if (pipeline_dir / 'qc').exists():
        shutil.rmtree(pipeline_dir / 'qc')
    qc_results = load_qc(pipeline_dir / 'qc')
    cur_results = load_cur(pipeline_dir / 'cur')
except Exception as e:
    print(f'Failed to load sorter or qc with error:\n{e}\nRunning the pipeline again')
    seg_saved = save_binary_recording(seg_motion, pipeline_dir / 'preprocessed_recording', recalc=False)
    del seg_motion
    gc.collect()
    # Run Kilosort4
    [ks4_results,ks4_sorter] = sort_ks4(seg_saved, pipeline_dir / 'kilosort4', sorter_params=sorter_params, recalc=False)
    cur_results = run_cur(seg_saved, ks4_sorter, ks4_results, pipeline_dir / 'cur', recalc=False) # this should save out some merges
    qc_results = run_qc(seg_saved, cur_results, pipeline_dir / 'qc', recalc=True)
    

# Remove the processed binary
# if (pipeline_dir / 'preprocessed_recording').exists():
#     print('Removing preprocessed recording')
#     shutil.rmtree(pipeline_dir / 'preprocessed_recording')

print(f'Finished processing')

#%%


