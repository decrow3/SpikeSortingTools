#%%
from pipeline import condition_signal, correct_motion, plot_motion_output, sort_ks4, save_binary_recording, run_qc, KilosortResults, load_qc, run_cur, load_cur
from spikeinterface.sorters import get_default_sorter_params
from pathlib import Path
import shutil
# I'm using a pinned version of spikeinterface, so if something doesn't work with the latest version, ask about it
import spikeinterface.full as si
#%%
# Test data curation step
from spikeinterface.core import load_extractor
pipeline_dir = Path('/media/huklab/Data/NPX/Ryansorting/Rocky/pipeline_results_Rocky20240826_V2MT_g0_imec1/')
seg_saved = load_extractor(pipeline_dir / 'preprocessed_recording')
ks4_sorter = load_extractor(pipeline_dir / 'kilosort4/sorter')
ks4_results = KilosortResults(pipeline_dir / 'kilosort4/sorter_output') #load up the kilosort results

# shutil.rmtree(pipeline_dir / 'cur')
#%%
# this should save out some merges, very slow, recalculates waveforms and templates etc
# Shouldn't it be able to pull some of this information out of the kilosort results?
cur_results = run_cur(seg_saved, ks4_sorter, pipeline_dir / 'cur11', recalc=False) 
#cur_todo = load_cur(pipeline_dir / 'cur')

#Fails after parrallel jobs, using binary file and scaling waveforms off,
# Test with scaling=True: cur7, ran out of RAM? Crashed
# Test with binary off, parallel on, scaling=True: cur8. Ran to line 260 in apply_curation_labels "AssertionError: Mismatch between existing property dtype b and provided values dtype U."
# cur9 debug, edited line 260 in curation_format. then crashed?
# cur10, trying again no debug. crashed. runnning again without vscode, up to compute waveforms 11:20am 4/3
#cur 11, trying to calculate sparsity first