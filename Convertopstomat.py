from scipy.io import savemat
import numpy as np
import glob
import os
from pathlib import Path
pipeline_dir = Path('/mnt/NPX/Luke/20250724/pipeline_results_Luke0724_V2V1_g0_imec1/') #Path to your pipeline directory
#pipeline_dir = Path('/media/huklab/Data/NPX/Ryansorting/Luke/pipeline_results_Luke0717_V1_g0_imec0_thresh1/')


# npzFiles = glob.glob("/home/huklab/Documents/RyanSorting/SpikeSortingTools/pipeline_results_Luke0804_V2V1_g0_imec0.ap/cur/cur_sorter_output/ops.npy")#
#npzFiles = glob.glob('/mnt/NPX/Rocky/20231209/Sorted/pipeline_results_Rocky20231209_V1V2_g0_imec1/cur/cur_sorter_output/ops.npy')
#/media/huklab/Data/NPX/Ryansorting/Luke/pipeline_results_Luke0804_V2V1_g0_imec0/cur/cur_sorter_output/ops.npy")
npzFiles = glob.glob(str(pipeline_dir / 'cur' / 'cur_sorter_output' / 'ops.npy'))

# curation failed
#npzFiles = glob.glob(str(pipeline_dir / 'kilosort4' / 'sorter_output' / 'ops.npy'))
for f in npzFiles:
    fm = os.path.splitext(f)[0]+'.mat'
    d = np.load(f,allow_pickle=True)
    xc=d.item()['xc']
    yc=d.item()['yc']
    matout={"xc":xc,"yc":yc}
    savemat(fm, matout)
    print('generated ', fm, 'from', f)


# ##
# #%% Saving out to matlab files
# import numpy as np
# import os

# qc_outdir       = pipeline_dir / 'qc'
# waveformsfile   ='waveforms/waveforms.npz'
# refractoryfile  ='refractory/refractory_qc.npz'
# truncation      ='amp_truncation/truncation_qc.npz'

# presencefile    ='amp_truncation/present_qc.npz'


# def load_qc_data(qc_outdir, filename):
#     filepath = Path(qc_outdir) / filename
#     if not filepath.exists():
#         raise FileNotFoundError(f"File {filepath} does not exist.")
    
#     try:
#         data = np.load(filepath, allow_pickle=True)
#         return data
#     except Exception as e:
#         raise RuntimeError(f"Failed to load {filepath}: {e}")

# # Load the data
# waveforms_data = load_qc_data(qc_outdir, waveformsfile)
# refractory_data = load_qc_data(qc_outdir, refractoryfile)
# truncation_data = load_qc_data(qc_outdir, truncation)
# presence_data = load_qc_data(qc_outdir, presencefile)

# #Saving out the data to matlab compatible mat files
# import scipy.io as sio
# def save_to_mat(data, filename):
#     """Save numpy data to a .mat file."""
#     try:
#         sio.savemat(filename, data)
#         print(f"Data saved to {filename}")
#     except Exception as e:
#         raise RuntimeError(f"Failed to save data to {filename}: {e}")
# # Define output filenames
# output_waveforms_file = os.path.join(qc_outdir, 'waveforms_data.mat')
# output_refractory_file = os.path.join(qc_outdir, 'refractory_data.mat')
# output_truncation_file = os.path.join(qc_outdir, 'truncation_data.mat')
# output_presence_file = os.path.join(qc_outdir, 'presence_data.mat')
# # Save the data to .mat files
# save_to_mat(waveforms_data, output_waveforms_file)
# save_to_mat(refractory_data, output_refractory_file)
# save_to_mat(truncation_data, output_truncation_file)
# save_to_mat(presence_data, output_presence_file)


# import glob
# npzFiles = glob.glob("pipeline_dir / 'cur' /cur_sorter_output/ops.npy")

# for f in npzFiles:
#     fm = os.path.splitext(f)[0]+'.mat'
#     d = np.load(f,allow_pickle=True)
#     xc=d.item()['xc']
#     yc=d.item()['yc']
#     matout={"xc":xc,"yc":yc}
#     save_to_mat(fm, matout)
#     print('generated ', fm, 'from', f)

# # Print confirmation of saved files
# print("All data has been saved successfully.")
