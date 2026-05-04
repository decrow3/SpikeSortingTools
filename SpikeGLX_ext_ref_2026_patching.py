#%%
from pipeline import condition_signal, correct_motion, plot_motion_output, sort_ks4, save_binary_recording, run_qc
from pipeline.curation_postpatch import run_cur_final
from spikeinterface.sorters import get_default_sorter_params
from pathlib import Path
import gc
# I'm using a pinned version of spikeinterface, so if something doesn't work with the latest version, ask about it
import spikeinterface.full as si

#%% Change this code to load your data
data_dir =   r"/mnt/NPX/Luke/20260313/Luke03132026_V2V1_RH_g0/"
out_dir  =  r"/media/huklaban5/Data/Patched/"
stream_id = "imec0.ap" #usually imec0 is first inserted probe (often V2/MT), imec1 is second probe (often V1)
seg = si.read_spikeglx(folder_path=data_dir, load_sync_channel=False, stream_id=stream_id)

#%% Run on a snippet to check params
# start_time = 0
# stop_time  = start_time + 100
# seg=seg.frame_slice(start_time * 30000, stop_time * 30000)

#%%
sess_name   = data_dir.split('/')[-2]
stream_name = stream_id.split('.')[0]
data_root   = '/'.join(data_dir.split('/')[0:5])
print(f'Using data root {data_root}, pipeline results will be saved to harddrive')

#%% Directory layout
# dredge_dir  — server, motion correction results (NEVER rerun)
# pipeline_dir — harddrive, KS4 / curation / QC outputs
dredge_dir   = Path(f'{data_root}/dredge_pipeline_results_{sess_name}_{stream_name}')
pipeline_dir = Path(out_dir) / f'patched_pipeline_results_{sess_name}_{stream_name}'

# Guard: motion correction must already exist on server
if not dredge_dir.exists():
    raise FileNotFoundError(
        f'Motion correction results not found: {dredge_dir}\n'
        'Run the pre-patch dredge pipeline first, or check the server is mounted.'
    )

pipeline_dir.mkdir(parents=True, exist_ok=True)
print(f'Dredge dir (server): {dredge_dir}')
print(f'Pipeline dir (harddrive): {pipeline_dir}')

#%% Signal conditioning — load from server cache, never rerun
noise_thresh = 0.3
uV_thresh    = .5e3  # 500 µV for external reference
seg_pre_motion_est, seg_pre_sorting = condition_signal(
    seg, cache_dir=dredge_dir / 'conditioning',
    noise_thresh=noise_thresh, uV_thresh=uV_thresh, recalc=False,
)

#%% Motion correction — load from server cache, never rerun
seg_motion = correct_motion(
    seg_pre_motion_est, rec_for_sorting=seg_pre_sorting,
    cache_dir=dredge_dir / 'motion', recalc=False, method='dredge',
)
plot_motion_output(seg_motion, cache_dir=dredge_dir / 'motion', save_dir=pipeline_dir / 'motion')

#%% Kilosort4 parameters
import json
sorter_params = get_default_sorter_params('kilosort4')
sorter_params['do_correction']        = False   # drift correction off (done by dredge)
sorter_params['save_extra_vars']      = True    # required for truncation qc
sorter_params['Th_universal']         = 9
sorter_params['Th_learned']           = 8
sorter_params['duplicate_spike_ms']   = 0.25
sorter_params['ccg_threshold']        = 0.75
sorter_params['nearest_chans']        = 20
sorter_params['nearest_templates']    = 200
sorter_params['max_channel_distance'] = 64
sorter_params['clear_cache']          = True
sorter_params['cross_peel_claim_ms']  = 0.25
sorter_params['cross_peel_claim_um']  = 75.0

# Override Th_universal / Th_learned from previous dredge run if available
_dredge_ks4_params = dredge_dir / 'kilosort4' / 'spikeinterface_params.json'
if _dredge_ks4_params.exists():
    with open(_dredge_ks4_params) as _f:
        _prev = json.load(_f).get('sorter_params', {})
    for _key in ('Th_universal', 'Th_learned'):
        if _key in _prev:
            sorter_params[_key] = _prev[_key]
            print(f'Loaded {_key}={_prev[_key]} from dredge params')
else:
    print(f'No dredge KS4 params found, using defaults: Th_universal={sorter_params["Th_universal"]}, Th_learned={sorter_params["Th_learned"]}')

#%% Free memory before heavy steps
del seg
del seg_pre_motion_est
del seg_pre_sorting

#%% Load preprocessed recording from server
seg_saved = save_binary_recording(seg_motion, dredge_dir / 'preprocessed_recording', recalc=False)
del seg_motion
gc.collect()

#%% KS4 fallback: harddrive first, then server old patched dir, else run fresh to harddrive
_ks4_hd     = pipeline_dir / 'kilosort4'
_ks4_server = Path(f'{data_root}/patched_pipeline_results_{sess_name}_{stream_name}') / 'kilosort4'

if _ks4_hd.exists():
    ks4_dir = _ks4_hd
    print(f'KS4: loading from harddrive: {ks4_dir}')
elif _ks4_server.exists():
    ks4_dir = _ks4_server
    print(f'KS4: loading from server: {ks4_dir}')
else:
    ks4_dir = _ks4_hd
    print(f'KS4: no existing results found, running fresh to harddrive: {ks4_dir}')

[ks4_results, ks4_sorter] = sort_ks4(seg_saved, ks4_dir, sorter_params=sorter_params, recalc=False)

#%% Curation and QC — always rerun to harddrive
cur_results = run_cur_final(
    ks4_sorter,
    ks4_results,
    pipeline_dir / 'cur',
    recalc=True,
    ks4_out_path=ks4_dir / 'sorter_output',
)
qc_results = run_qc(seg_saved, cur_results, pipeline_dir / 'qc', recalc=True)

print(f'Finished processing')

#%% Saving out to matlab files
import numpy as np
import os

qc_outdir      = pipeline_dir / 'qc'
waveformsfile  = 'waveforms/waveforms.npz'
refractoryfile = 'refractory/refractory_qc.npz'
truncation     = 'amp_truncation/truncation_qc.npz'
presencefile   = 'amp_truncation/present_qc.npz'


def load_qc_data(qc_outdir, filename):
    filepath = Path(qc_outdir) / filename
    if not filepath.exists():
        raise FileNotFoundError(f"File {filepath} does not exist.")
    try:
        return np.load(filepath, allow_pickle=True)
    except Exception as e:
        raise RuntimeError(f"Failed to load {filepath}: {e}")


waveforms_data  = load_qc_data(qc_outdir, waveformsfile)
refractory_data = load_qc_data(qc_outdir, refractoryfile)
truncation_data = load_qc_data(qc_outdir, truncation)
presence_data   = load_qc_data(qc_outdir, presencefile)


import scipy.io as sio

def save_to_mat(data, filename):
    try:
        sio.savemat(filename, data)
        print(f"Data saved to {filename}")
    except Exception as e:
        raise RuntimeError(f"Failed to save data to {filename}: {e}")


save_to_mat(waveforms_data,  os.path.join(qc_outdir, 'waveforms_data.mat'))
save_to_mat(refractory_data, os.path.join(qc_outdir, 'refractory_data.mat'))
save_to_mat(truncation_data, os.path.join(qc_outdir, 'truncation_data.mat'))
save_to_mat(presence_data,   os.path.join(qc_outdir, 'presence_data.mat'))


import glob
npzFiles = glob.glob(str(pipeline_dir / 'cur' / 'cur_output' / 'ops.npy'))
for f in npzFiles:
    fm = os.path.splitext(f)[0] + '.mat'
    d  = np.load(f, allow_pickle=True)
    save_to_mat({"xc": d.item()['xc'], "yc": d.item()['yc']}, fm)
    print('generated', fm, 'from', f)

print("All data has been saved successfully.")
