"""
Batch runner for SpikeGLX external-reference patching pipeline.

Edit SESSIONS below to add/remove datasets. Each entry is:
    (data_dir, stream_id)

The script logs successes and failures and continues on error so a single
bad session does not abort the batch.
"""

# ---------------------------------------------------------------------------
# Sessions to process — edit this list
# ---------------------------------------------------------------------------
SESSIONS = [
    # (data_dir,                                                  stream_id)
    # ── 2026 ──────────────────────────────────────────────────────────────
    # ("/mnt/NPX/Luke/20260316/Luke03162026_V2V1_RH_g0/",        "imec1.ap"),  # done
    # ("/mnt/NPX/Luke/20260316/Luke03162026_V2V1_RH_g0/",        "imec0.ap"),  # done
    # ("/mnt/NPX/Luke/20260313/Luke03132026_V2V1_RH_g0/",        "imec1.ap"),
    # ("/mnt/NPX/Luke/20260313/Luke03132026_V2V1_RH_g0/",        "imec0.ap"),
    # ("/mnt/NPX/Luke/20260311/Luke03112026_V2V1_RH_g0/",        "imec1.ap"),
    # ("/mnt/NPX/Luke/20260311/Luke03112026_V2V1_RH_g0/",        "imec0.ap"),
    # ("/mnt/NPX/Luke/20260309/Luke03092026_V1_RH_g0/",          "imec1.ap"),
    # ("/mnt/NPX/Luke/20260309/Luke03092026_V1_RH_g0/",          "imec0.ap"),
    # ("/mnt/NPX/Luke/20260308/Luke03082026_V1_RH_g0/",          "imec0.ap"),
    # ("/mnt/NPX/Luke/20260302/Luke03022026_V2V1_RH_g0/",        "imec1.ap"),  # done
    # ("/mnt/NPX/Luke/20260302/Luke03022026_V2V1_RH_g0/",        "imec0.ap"),  # done
    # ("/mnt/NPX/Luke/20260301/Luke03012026_V2V1_RH_g0/",        "imec1.ap"),  # done
    # ("/mnt/NPX/Luke/20260301/Luke03012026_V2V1_RH_g0/",        "imec0.ap"),  # done
    # ── 2025 ──────────────────────────────────────────────────────────────
    # ("/mnt/NPX/Luke/20251205/Luke12052025_V1_RH_g0/",          "imec1.ap"),
    # ("/mnt/NPX/Luke/20251205/Luke12052025_V1_RH_g0/",          "imec0.ap"),
    # ("/mnt/NPX/Luke/20251120/Luke1120_V1_RH_g0/",              "imec0.ap"),
    # ("/mnt/NPX/Luke/20251111/Luke1111_V1_RH_g0_g0/",           "imec0.ap"),
    # ("/mnt/NPX/Luke/20250805/Luke0805_V2V1_g0/",               "imec1.ap"),
    # ("/mnt/NPX/Luke/20250805/Luke0805_V2V1_g0/",               "imec0.ap"),
    # ("/mnt/NPX/Luke/20250804/Luke0804_V2V1_g0/",               "imec1.ap"),
    # ("/mnt/NPX/Luke/20250804/Luke0804_V2V1_g0/",               "imec0.ap"),
    # ("/mnt/NPX/Luke/20250730/Luke0730_V2V1_g0/",               "imec1.ap"),
    # ("/mnt/NPX/Luke/20250730/Luke0730_V2V1_g0/",               "imec0.ap"),
    # ("/mnt/NPX/Luke/20250724/Luke0724_V2V1_g0/",               "imec1.ap"),
    # ("/mnt/NPX/Luke/20250724/Luke0724_V2V1_g0/",               "imec0.ap"),
    # ("/mnt/NPX/Luke/20250717/Luke0717_V1_g0/",                 "imec0.ap"),
]

OUT_DIR = r"/media/huklaban5/Data/Patched/"

# ---------------------------------------------------------------------------
# Pipeline (do not edit below unless changing the processing steps)
# ---------------------------------------------------------------------------
import gc
import glob
import json
import os
import traceback
from pathlib import Path

import numpy as np
import scipy.io as sio
import spikeinterface.full as si
from spikeinterface.sorters import get_default_sorter_params

from pipeline import condition_signal, correct_motion, plot_motion_output, run_qc, save_binary_recording, sort_ks4
from pipeline.curation_postpatch import run_cur_final


def _run_session(data_dir, stream_id, out_dir):
    data_dir  = data_dir.rstrip('/')
    sess_name = data_dir.split('/')[-1]
    stream_name = stream_id.split('.')[0]
    data_root   = '/'.join(data_dir.split('/')[:-1])  # parent of session folder

    print(f'\n{"="*70}')
    print(f'SESSION: {sess_name}  stream: {stream_id}')
    print(f'{"="*70}')

    dredge_dir   = Path(data_root) / f'dredge_pipeline_results_{sess_name}_{stream_name}'
    pipeline_dir = Path(out_dir) / f'patched_pipeline_results_{sess_name}_{stream_name}'

    if not dredge_dir.exists():
        raise FileNotFoundError(
            f'Motion correction results not found: {dredge_dir}\n'
            'Run the pre-patch dredge pipeline first, or check the server is mounted.'
        )

    pipeline_dir.mkdir(parents=True, exist_ok=True)
    print(f'Dredge dir   : {dredge_dir}')
    print(f'Pipeline dir : {pipeline_dir}')

    # Load recording
    seg = si.read_spikeglx(folder_path=data_dir + '/', load_sync_channel=False, stream_id=stream_id)

    # Signal conditioning — load from server cache
    seg_pre_motion_est, seg_pre_sorting = condition_signal(
        seg, cache_dir=dredge_dir / 'conditioning',
        noise_thresh=0.3, uV_thresh=.5e3, recalc=False,
    )

    # Motion correction — load from server cache
    seg_motion = correct_motion(
        seg_pre_motion_est, rec_for_sorting=seg_pre_sorting,
        cache_dir=dredge_dir / 'motion', recalc=False, method='dredge',
    )
    plot_motion_output(seg_motion, cache_dir=dredge_dir / 'motion', save_dir=pipeline_dir / 'motion')

    # KS4 parameters
    sorter_params = get_default_sorter_params('kilosort4')
    sorter_params['do_correction']        = False
    sorter_params['save_extra_vars']      = True
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

    _dredge_ks4_params = dredge_dir / 'kilosort4' / 'spikeinterface_params.json'
    if _dredge_ks4_params.exists():
        with open(_dredge_ks4_params) as _f:
            _prev = json.load(_f).get('sorter_params', {})
        for _key in ('Th_universal', 'Th_learned'):
            if _key in _prev:
                sorter_params[_key] = _prev[_key]
                print(f'Loaded {_key}={_prev[_key]} from dredge params')
    else:
        print(f'No dredge KS4 params — using defaults Th_universal={sorter_params["Th_universal"]}, Th_learned={sorter_params["Th_learned"]}')

    del seg, seg_pre_motion_est, seg_pre_sorting

    # Load preprocessed recording from server
    seg_saved = save_binary_recording(seg_motion, dredge_dir / 'preprocessed_recording', recalc=False)
    del seg_motion
    gc.collect()

    # KS4 fallback: harddrive → server old patched dir → run fresh
    _ks4_hd     = pipeline_dir / 'kilosort4'
    _ks4_server = Path(data_root) / f'patched_pipeline_results_{sess_name}_{stream_name}' / 'kilosort4'

    if _ks4_hd.exists():
        ks4_dir = _ks4_hd
        print(f'KS4: loading from harddrive: {ks4_dir}')
    elif _ks4_server.exists():
        ks4_dir = _ks4_server
        print(f'KS4: loading from server: {ks4_dir}')
    else:
        ks4_dir = _ks4_hd
        print(f'KS4: no existing results — running fresh to harddrive: {ks4_dir}')

    ks4_results, ks4_sorter = sort_ks4(seg_saved, ks4_dir, sorter_params=sorter_params, recalc=False)

    # Curation and QC — always rerun
    cur_results = run_cur_final(
        ks4_sorter, ks4_results,
        pipeline_dir / 'cur',
        recalc=True,
        ks4_out_path=ks4_dir / 'sorter_output',
    )
    qc_results = run_qc(seg_saved, cur_results, pipeline_dir / 'qc', recalc=True)

    # Export to .mat
    qc_outdir = pipeline_dir / 'qc'
    _qc_files = {
        'waveforms':  'waveforms/waveforms.npz',
        'refractory': 'refractory/refractory_qc.npz',
        'truncation': 'amp_truncation/truncation_qc.npz',
        'presence':   'amp_truncation/present_qc.npz',
    }
    _mat_names = {
        'waveforms':  'waveforms_data.mat',
        'refractory': 'refractory_data.mat',
        'truncation': 'truncation_data.mat',
        'presence':   'presence_data.mat',
    }
    for key, relpath in _qc_files.items():
        filepath = qc_outdir / relpath
        if not filepath.exists():
            raise FileNotFoundError(f'QC output missing: {filepath}')
        data = np.load(filepath, allow_pickle=True)
        sio.savemat(str(qc_outdir / _mat_names[key]), data)
        print(f'Saved {_mat_names[key]}')

    ops_npy = glob.glob(str(pipeline_dir / 'cur' / 'cur_output' / 'ops.npy'))
    for f in ops_npy:
        fm = os.path.splitext(f)[0] + '.mat'
        d  = np.load(f, allow_pickle=True)
        sio.savemat(fm, {"xc": d.item()['xc'], "yc": d.item()['yc']})
        print(f'Generated {fm}')

    print(f'Done: {sess_name} {stream_id}')


# ---------------------------------------------------------------------------
# Batch loop
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    results = []

    for data_dir, stream_id in SESSIONS:
        label = f'{data_dir.rstrip("/").split("/")[-1]}  {stream_id}'
        try:
            _run_session(data_dir, stream_id, OUT_DIR)
            results.append(('OK', label))
        except Exception as e:
            print(f'\n[ERROR] {label}:\n{traceback.format_exc()}')
            results.append(('FAIL', label, str(e)))

    print(f'\n{"="*70}')
    print('BATCH SUMMARY')
    print(f'{"="*70}')
    for r in results:
        status = r[0]
        label  = r[1]
        msg    = f'  → {r[2]}' if status == 'FAIL' else ''
        print(f'  [{status}]  {label}{msg}')
    print(f'{"="*70}')

    n_ok   = sum(1 for r in results if r[0] == 'OK')
    n_fail = sum(1 for r in results if r[0] == 'FAIL')
    print(f'{n_ok} succeeded, {n_fail} failed')
