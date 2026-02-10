#%%
from pipeline import condition_signal, correct_motion, plot_motion_output, sort_ks4, save_binary_recording, run_qc, run_cur, KilosortResults, load_qc,  load_cur
from spikeinterface.sorters import get_default_sorter_params
from pathlib import Path
import shutil
import gc
# I'm using a pinned version of spikeinterface, so if something doesn't work with the latest version, ask about it
import spikeinterface.full as si

#Key sets: g20220412, g20211217, b20220701, b20220625
#g20211217 had missing timestamps.npy /mnt/NPX/Gru/20211217/2021-12-17_13-16-50/'
#%% Load a single file
# data_dir = Path('/mnt/NPX/Brie/b20220701/')#Path('/media/huklab/Data/NPX/Spikesorting/Combining/Gru_2022-0412_Probe1/') #Path('/home/ryanress/code/DataHorwitzLGN/data/raw/2024-12-10_Chihiro/2024-12-10_15-40-46')
# stream_name = "Record Node 101#Neuropix-PXI-100.0"
# seg = si.read_openephys(
#     data_dir,
#     load_sync_timestamps=False,
#     stream_name=stream_name,
#     experiment_names=["experiment1"], # specify experiment name to avoid loading unwanted data
# )


#%% Load and concatenate data
data_dir = Path('/mnt/NPX/Gru/20211217/') #Path('/media/huklab/Data/NPX/Spikesorting/Combining/Gru_2022-0412_Probe1/') #Path('/home/ryanress/code/DataHorwitzLGN/data/raw/2024-12-10_Chihiro/2024-12-10_15-40-46')
stream_name = "Record Node 101#Neuropix-PXI-100.0"
#This is for Brie's data, saved differently:
# probe_number=1 #1 or 2
# if probe_number==1:
#     stream_name = "Record Node 103#Neuropix-PXI-101.0" #101.0, 101.1, are ap and lfp on probe 1, 101.2, 101.3 are ap and lfp on probe 2
# else:
#     stream_name = "Record Node 103#Neuropix-PXI-101.2" #101.0, 101.1, are ap and lfp on probe 1, 101.2, 101.3 are ap and lfp on probe 2

subfolders=[f for f in Path(data_dir).iterdir() if f.is_dir()]

# Skip any folders that don't start with a year, 202x
subfolders = [f for f in subfolders if f.name.startswith("202")]
subfolders = sorted(subfolders) # Sort to ensure consistent order

# Try to load all experiments
print(f'Loading {subfolders[0]}')

from spikeinterface.extractors.neoextractors.openephys import OpenEphysBinaryRecordingExtractor

streams, ids = OpenEphysBinaryRecordingExtractor.get_streams(subfolders[0], experiment_names="experiment1")
print("Available streams:", streams)

seg_all = si.read_openephys(subfolders[0], load_sync_timestamps=False, stream_name=stream_name, experiment_names="experiment1")# experiment_names="experiment1")

if len(subfolders) > 1:
    for subfolder in subfolders[1:]:
        print(f'Loading {subfolder}')
        seg = si.read_openephys(subfolder, load_sync_timestamps=False, stream_name=stream_name, experiment_names="experiment1")# experiment_names="experiment1")
        seg_all=si.concatenate_recordings([seg_all, seg])#seg_all.add_recording_segment(seg)

#%% Todo, add in probe data manually by seg.set_probe
import probeinterface 
record_node = "Record Node 101"

#This is for Brie's data, saved differently
# if probe_number ==1:
#     stream_name = "Record Node 103#Neuropix-PXI-101.0" #Here its different to above,101.0 is probe 1 ap, 101.1 is probe 1 lfp, 101.2 is probe 2 ap, 101.3 is probe 2 lfp
# else:
#     stream_name = "Record Node 103#Neuropix-PXI-101.1" #Here its different to above,101.0 is probe 1 ap, 101.1 is probe 1 lfp, 101.2 is probe 2 ap, 101.3 is probe 2 lfp
exp_id = 1
settings_file= seg_all.neo_reader.folder_structure[record_node]["experiments"][exp_id]["settings_file"]
if Path(settings_file).is_file():
                probe = probeinterface.read_openephys(
                    settings_file=settings_file, stream_name=stream_name, raise_error=False
                )
seg_all=seg_all.set_probe(probe, in_place=False)
#%%
#%% Run on a snippet to check params
# start_time = 0 #lots of motion around 10000s in, but time didn't start at 0?
# stop_time  = start_time + 100
# seg=seg.frame_slice(start_time * 30000, stop_time * 30000) #100 seconds snippet, if really low will need to change n_batches down from 50 to 5 in condition_signal ln137

#%%
# run pipelines
# Set up output directory using name parts of the data directory
# Example: /home/huklab/Documents/SpikeSortingTools/pipeline_results
filename_parts = data_dir.parts[-2:]  # Get the last two parts of the path
output_dir_name = "_".join(filename_parts)  # Join them with an underscore
pipeline_dir = Path('/home/huklaban5/Documents/SpikeSortingTools/pipeline_results_' + output_dir_name)
pipeline_dir.mkdir(parents=True, exist_ok=True)

#%%
# condition signal runs 1) bad channel detection 2) . Can we also get a noise over time measure over all channels, may need to censor some completely
noise_thresh = 0.1 # higher for spikeGLX, around 0.3
seg_pre = condition_signal(seg_all, cache_dir=pipeline_dir / 'conditioning', noise_thresh=noise_thresh, recalc=False)

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
#seg_motion = correct_motion(seg_pre, cache_dir=pipeline_dir / 'motion', recalc=False, method='med')
#plot_motion_output(seg_motion, cache_dir=pipeline_dir / 'motion')
# skipping motion correction, just running it in kilosort
seg_motion= seg_pre

#%% Kilosort4 parameters
# OpenEphys
sorter_params = get_default_sorter_params('kilosort4')
sorter_params['do_correction'] = True # Turns off drift correction
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
del seg_all
del seg_pre

gc.collect()
#
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

#%% Saving out to matlab files
import numpy as np
import os

qc_outdir       = pipeline_dir / 'qc'
waveformsfile   ='waveforms/waveforms.npz'
refractoryfile  ='refractory/refractory_qc.npz'
truncation      ='amp_truncation/truncation_qc.npz'

presencefile    ='amp_truncation/present_qc.npz'


def load_qc_data(qc_outdir, filename):
    filepath = Path(qc_outdir) / filename
    if not filepath.exists():
        raise FileNotFoundError(f"File {filepath} does not exist.")
    
    try:
        data = np.load(filepath, allow_pickle=True)
        return data
    except Exception as e:
        raise RuntimeError(f"Failed to load {filepath}: {e}")

# Load the data
waveforms_data = load_qc_data(qc_outdir, waveformsfile)
refractory_data = load_qc_data(qc_outdir, refractoryfile)
truncation_data = load_qc_data(qc_outdir, truncation)
presence_data = load_qc_data(qc_outdir, presencefile)

#Saving out the data to matlab compatible mat files
import scipy.io as sio
def save_to_mat(data, filename):
    """Save numpy data to a .mat file."""
    try:
        sio.savemat(filename, data)
        print(f"Data saved to {filename}")
    except Exception as e:
        raise RuntimeError(f"Failed to save data to {filename}: {e}")
# Define output filenames
output_waveforms_file = os.path.join(qc_outdir, 'waveforms_data.mat')
output_refractory_file = os.path.join(qc_outdir, 'refractory_data.mat')
output_truncation_file = os.path.join(qc_outdir, 'truncation_data.mat')
output_presence_file = os.path.join(qc_outdir, 'presence_data.mat')
# Save the data to .mat files
save_to_mat(waveforms_data, output_waveforms_file)
save_to_mat(refractory_data, output_refractory_file)
save_to_mat(truncation_data, output_truncation_file)
save_to_mat(presence_data, output_presence_file)


import glob
npzFiles = glob.glob("pipeline_dir / 'cur' /cur_sorter_output/ops.npy")

for f in npzFiles:
    fm = os.path.splitext(f)[0]+'.mat'
    d = np.load(f,allow_pickle=True)
    xc=d.item()['xc']
    yc=d.item()['yc']
    matout={"xc":xc,"yc":yc}
    save_to_mat(fm, matout)
    print('generated ', fm, 'from', f)

# Print confirmation of saved files
print("All data has been saved successfully.")


