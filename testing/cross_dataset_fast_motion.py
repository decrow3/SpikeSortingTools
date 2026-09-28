"""Prespecified cross-session peak-only MEDiCINe pilot; run via systemd.

No clustering, matching, sorting, or voltage motion correction is performed.
Independent extraction and fitting stages retain failures and refuse overwrite.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/home/huklab/Documents/RyanSorting/SpikeSortingTools')
DS = Path('/home/huklab/Documents/DARTsort')
DSPY = DS / '.venv/bin/python'
MEDPY = DS / 'environments/medicine-estimator/.venv/bin/python'
ORIGINAL = Path('/home/huklab/Documents/DataRowleyV1V2/DataRowleyV1V2/outputs/ephys_quality_pilot/luke_20250804/run_config.json')
MED = dict(motion_bound=500., time_bin_size=.25, time_kernel_width=1.,
           num_depth_bins=4, amplitude_threshold_quantile=0., training_steps=10000,
           batch_size=4096, activity_network_hidden_features=[256, 256],
           learning_rate=.0005, initial_motion_noise=.1, motion_noise_steps=2000,
           epsilon=.001, plot_figures=False)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, default=str) + '\n')
    os.replace(tmp, path)


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def load_recording(spec):
    import numpy as np
    import spikeinterface.extractors as se
    from spikeinterface.core import read_binary
    if spec['dataset'] == 'Luke':
        return se.read_spikeglx('/media/huklab/Expansion/Luke0804_V2V1_g0', stream_id=spec['probe'] + '.ap')
    if spec['dataset'] == 'Bacon':
        return se.read_openephys('/mnt/MGS/Ephys/Raw/Bacon/20251016',
            stream_name='Record Node 101#Neuropix-PXI-100.Probe' + spec['probe'][-1] + '-AP')
    meta = json.loads(Path(spec['metadata']).read_text())
    r = read_binary(spec['raw_path'], sampling_frequency=meta['sample_rate'],
                    num_channels=64, dtype='int16', gain_to_uV=meta['uV_per_bit'], offset_to_uV=0.)
    r.set_channel_locations(np.asarray(meta['probe_geometry_um']))
    return r.select_channels(meta['shank_inds'][int(spec['probe'][-1]) - 1])


def prepare(out):
    import numpy as np
    out.mkdir(exist_ok=False, parents=True)
    original = json.loads(ORIGINAL.read_text())
    specs = []
    for r in original['raw_recordings']:
        dataset = r['dataset'].split()[0]
        names = ['shank1', 'shank2'] if dataset == 'Yates' else [r['probe']]
        for name in names:
            raw = r['path']
            if dataset == 'Luke':
                raw = str(Path('/media/huklab/Expansion/Luke0804_V2V1_g0') / Path(raw).parent.name / Path(raw).name)
            s = dict(dataset='Allen' if dataset == 'Yates' else dataset, probe=name,
                     session=r['dataset'], raw_path=raw,
                     metadata=str(Path(r['path']).parent / 'ephys_metadata.json'))
            rec = load_recording(s)
            stat = Path(raw).stat()
            s.update(sampling_frequency_hz=rec.get_sampling_frequency(), n_frames=rec.get_num_frames(),
                     raw_stat=[stat.st_size, stat.st_mtime_ns], channel_ids=rec.channel_ids.tolist(),
                     geometry_um=rec.get_channel_locations().tolist(), gains_uv=rec.get_channel_gains().tolist())
            assert np.isfinite(rec.get_channel_locations()).all()
            assert np.allclose(rec.get_channel_gains(), r['uv_per_bit'])
            assert abs(rec.get_sampling_frequency() - r['sample_rate_hz']) < 1e-6
            specs.append(s)
    windows = []
    # Interleave datasets, so partial results contain all preparations early.
    for frac in [.1, .5, .9]:
        for s in specs:
            fs = s['sampling_frequency_hz']
            start = round((s['n_frames'] / fs * frac - 60) * fs)
            stop = start + round(120 * fs)
            assert start > fs and stop < s['n_frames'] - fs
            windows.append(dict(id=f"{s['dataset'].lower()}_{s['probe']}_p{round(frac*100):02d}",
                dataset=s['dataset'], probe=s['probe'], fraction=frac,
                start_frame=start, stop_frame=stop, start_s=start/fs, stop_s=stop/fs))
    cfg = dict(schema='cross-dataset-fast-motion-v1', purpose='Compare magnitude, speed, depth dependence and noise across Bacon, Luke and Allen',
        records=specs, windows=windows, medicine=MED, seed=0,
        frontend='DARTsort ibllikecmr then initial_detection only; full coverage, per-window denoiser fit',
        noise='12 uniformly spaced 1-second samples/window; third-order 300-6000 Hz zero-phase; no reference, shank median, local median within 100um excluding self',
        restart='Only hash-sealed completed stages reusable. Interrupted stages refuse overwrite. No within-fit optimizer checkpoint.',
        original_config_sha256=sha(ORIGINAL), driver_sha256=sha(__file__),
        dartsort_source_sha256={str(p):sha(p) for p in sorted((DS/'src/dartsort').rglob('*.py'))},
        medicine_source_sha256={str(p):sha(p) for p in sorted((DS/'environments/medicine-estimator/.venv/lib/python3.12/site-packages/medicine').glob('*.py'))})
    save(out/'config.json', cfg)
    print(out/'config.json', flush=True)


def seal(stage):
    save(stage/'complete.json', {str(p.relative_to(stage)):sha(p) for p in sorted(stage.rglob('*'))
         if p.is_file() and p.name != 'complete.json'})


def complete(stage):
    if not stage.exists():
        return False
    receipt = stage/'complete.json'
    if not receipt.exists():
        raise RuntimeError(f'Unsealed stage: preserve and investigate {stage}')
    for name, digest in json.loads(receipt.read_text()).items():
        assert sha(stage/name) == digest, stage/name
    return True


def check_sources(cfg, family):
    for path, digest in cfg[family + '_source_sha256'].items():
        assert sha(path) == digest, f'Source changed: {path}'


def noise_rows(rec, window):
    import numpy as np
    from scipy.signal import butter, sosfiltfilt
    fs = rec.get_sampling_frequency()
    geo = rec.get_channel_locations()
    dist = np.linalg.norm(geo[:, None] - geo[None, :], axis=2)
    neighbors = [(dist[c] <= 100) & (dist[c] > 0) for c in range(len(geo))]
    assert all(x.any() for x in neighbors)
    sos = butter(3, [300, 6000], fs=fs, btype='bandpass', output='sos')
    pad = round(.05 * fs)
    n = round(fs)
    result = []
    requested = np.linspace(window['start_frame'], window['stop_frame'] - n, 12).round().astype(int)
    # Full-session deployment includes frame zero and the terminal AP frame.
    # Keep the filter pad inside the recording at those two boundaries. This
    # is a no-op for all prior interior stage-1/2 windows.
    starts = np.clip(requested, pad, rec.get_num_frames()-n-pad)
    for i, start in enumerate(starts):
        x = rec.get_traces(start_frame=int(start-pad), end_frame=int(start+n+pad), return_in_uV=True)
        x = sosfiltfilt(sos, x, axis=0)[pad:pad+n].astype('float32')
        shared = np.median(x, axis=1)
        for reference in ['none', 'shank_median', 'local_100um_exclude_self']:
            if reference == 'none':
                y = x
            elif reference == 'shank_median':
                y = x - shared[:, None]
            else:
                y = np.stack([x[:, c] - np.median(x[:, near], axis=1) for c, near in enumerate(neighbors)], axis=1)
            sig = np.median(abs(y - np.median(y, axis=0)), axis=0) / .6744897501960817
            for c, value in enumerate(sig):
                result.append(dict(window=window['id'], dataset=window['dataset'], probe=window['probe'],
                    sample=i, start_s=start/fs, channel=str(rec.channel_ids[c]), reference=reference,
                    sigma_uv=float(value), local_neighbor_count=int(neighbors[c].sum())))
    return result


def preprocess_comparison(rec, stage):
    """IBL-like CMR with coherence gates but no absolute HF-PSD exclusion.

    An absolute voltage-noise cutoff removed all of Bacon's signal channels.
    Keep noise as a measured dataset property; retain the same coherence-based
    dead/noisy/outside gates for every dataset. Preserve every decision.
    """
    import numpy as np
    import spikeinterface.full as si
    rec = si.highpass_filter(rec.astype(np.float32))
    if 'inter_sample_shift' in rec.get_property_keys():
        rec = si.phase_shift(rec)
    ids, labels = si.detect_bad_channels(rec, seed=0, psd_hf_threshold=float('inf'))
    save(stage/'channel_gate.json', dict(policy='coherence gates; absolute HF-PSD exclusion disabled for all datasets',
        channel_ids=rec.channel_ids.tolist(), labels=labels.tolist(), excluded_ids=ids.tolist(),
        num_random_chunks=100, seed=0))
    if len(ids) == rec.get_num_channels():
        raise RuntimeError('All channels rejected: inspect channel_gate.json before any materialization')
    rec = rec.remove_channels(ids)
    rec = si.common_reference(rec)
    noise = si.get_noise_levels(rec, return_in_uV=False,
        random_slices_kwargs=dict(seed=0, num_chunks_per_segment=100))
    assert np.isfinite(noise).all() and (noise > 0).all()
    rec = si.scale(rec, gain=1. / noise)
    return si.common_reference(rec).astype('float32')


def extract(out, cfg, w):
    import numpy as np
    import torch
    import h5py
    import dataclasses
    import shutil
    import importlib.metadata
    from dartsort.config import DARTsortUserConfig
    from dartsort.util.internal_config import to_internal_config
    from dartsort.util.preprocess_util import preprocess
    from dartsort.main import initial_detection
    stage = out/w['id']/'extraction'
    stage.mkdir(parents=True, exist_ok=False)
    check_sources(cfg, 'dartsort')
    torch.set_num_threads(4)
    torch.manual_seed(0)
    np.random.seed(0)
    spec = next(s for s in cfg['records'] if (s['dataset'], s['probe']) == (w['dataset'], w['probe']))
    rawstat = Path(spec['raw_path']).stat()
    assert [rawstat.st_size, rawstat.st_mtime_ns] == spec['raw_stat']
    started = time.monotonic()
    # Q/V can serialize the only phase which reads the USB source.  The lock is
    # opt-in so all earlier sealed extraction contexts retain their behavior.
    lock_path = os.environ.get('DARTSORT_RAW_PREPROCESS_LOCK')
    lock_file = None
    lock_wait_started = time.monotonic()
    if lock_path:
        import fcntl
        lock_file = open(lock_path, 'a+b')
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
    lock_wait_s = time.monotonic() - lock_wait_started
    try:
        raw_started = time.monotonic()
        rec = load_recording(spec)
        assert np.array_equal(rec.get_channel_locations(), spec['geometry_um'])
        write_csv(stage/'noise_channel_samples.csv', noise_rows(rec, w))
        raw_read_s = time.monotonic() - raw_started
        preprocess_started = time.monotonic()
        # One second of real context at each side prevents slice filtering edges entering the fit.
        context = round(rec.get_sampling_frequency())
        # Clip context at recording boundaries.  Ordinary pilot windows still get
        # one full second on each side; a full-session Q cache starts at AP frame
        # zero and therefore has no unavailable left context.
        slice_start = max(0, w['start_frame'] - context)
        slice_stop = min(rec.get_num_frames(), w['stop_frame'] + context)
        left_context = w['start_frame'] - slice_start
        slice_rec = rec.frame_slice(slice_start, slice_stop)
        c = to_internal_config(DARTsortUserConfig(preprocessing='ibllikecmr',
            preprocessing_dtype='float32', subsampling_presence=1.,
            subsampling_spikes_per_channel=5000, n_jobs_cpu=4, n_jobs_gpu=1,
            n_jobs_small=4, n_jobs_small_gpu=1, device='cuda:0'), rec.get_num_channels())
        save(stage/'resolved_dartsort.json', dataclasses.asdict(c))
        if cfg.get('channel_gate') == 'coherence_without_absolute_psd':
            pre = preprocess_comparison(slice_rec, stage)
        else:
            pre = preprocess(slice_rec, 'ibllikecmr', 'float32')
        assert pre.get_num_channels() > 0, 'All channels rejected by preprocessing'
        pre = pre.frame_slice(left_context, left_context+w['stop_frame']-w['start_frame'])
        save(stage/'channels.json', dict(channel_ids=pre.channel_ids.tolist(), geometry_um=pre.get_channel_locations().tolist()))
        cached_path = stage/'temporary_preprocessed'
        cached = pre.save(folder=cached_path, format='binary', n_jobs=4, chunk_duration='1s', progress_bar=False)
        preprocess_s = time.monotonic() - preprocess_started
    finally:
        if lock_file is not None:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            lock_file.close()
    detection = stage/'detection'
    detection.mkdir()
    detection_started = time.monotonic()
    gpu_lock_path = os.environ.get('DARTSORT_GPU_DETECTION_LOCK')
    gpu_lock_file = None
    gpu_slot_file = None
    gpu_global_lock_held = False
    gpu_schedule_mode = 'single_locked'
    gpu_lock_wait_started = time.monotonic()
    probe_index = None
    gpu_samples = []
    monitor_thread = None
    monitor_stop = None
    detection_compute_started = None
    if gpu_lock_path:
        import fcntl
        gpu_lock_file = open(gpu_lock_path, 'a+b')
        fcntl.flock(gpu_lock_file.fileno(), fcntl.LOCK_EX)
        gpu_global_lock_held = True
        mode_path_value = os.environ.get('DARTSORT_GPU_MODE_FILE')
        while mode_path_value:
            mode_path = Path(mode_path_value)
            mode = json.loads(mode_path.read_text()).get('mode', 'single_locked') if mode_path.exists() else 'single_locked'
            if mode == 'dual_trial':
                trial_counter = Path(os.environ['DARTSORT_GPU_TRIAL_COUNTER'])
                with open(str(trial_counter)+'.lock', 'a+b') as trial_lock:
                    fcntl.flock(trial_lock.fileno(), fcntl.LOCK_EX)
                    claimed = int(trial_counter.read_text()) if trial_counter.exists() else 0
                    if claimed < 2:
                        trial_counter.write_text(str(claimed+1))
                        gpu_schedule_mode = 'dual_trial'
                    fcntl.flock(trial_lock.fileno(), fcntl.LOCK_UN)
                if gpu_schedule_mode != 'dual_trial':
                    time.sleep(1)
                    continue
            elif mode == 'dual_keep':
                gpu_schedule_mode = 'dual_keep'
            else:
                gpu_schedule_mode = 'single_locked'
            if gpu_schedule_mode.startswith('dual_'):
                fcntl.flock(gpu_lock_file.fileno(), fcntl.LOCK_UN)
                gpu_global_lock_held = False
                slots = [open(gpu_lock_path+f'.slot{i}', 'a+b') for i in range(2)]
                while gpu_slot_file is None:
                    for candidate in slots:
                        try:
                            fcntl.flock(candidate.fileno(), fcntl.LOCK_EX|fcntl.LOCK_NB)
                            gpu_slot_file = candidate
                            break
                        except BlockingIOError:
                            pass
                    if gpu_slot_file is None: time.sleep(.25)
                for candidate in slots:
                    if candidate is not gpu_slot_file: candidate.close()
            break
    gpu_lock_wait_s = time.monotonic() - gpu_lock_wait_started
    try:
        probe_counter = os.environ.get('DARTSORT_GPU_PROBE_COUNTER')
        if probe_counter:
            counter_path = Path(probe_counter)
            counter_path.parent.mkdir(parents=True, exist_ok=True)
            with open(str(counter_path)+'.lock', 'a+b') as counter_lock:
                fcntl.flock(counter_lock.fileno(), fcntl.LOCK_EX)
                count = int(counter_path.read_text()) if counter_path.exists() else 0
                if count < 2:
                    probe_index = count + 1
                    counter_path.write_text(str(probe_index))
                fcntl.flock(counter_lock.fileno(), fcntl.LOCK_UN)
        if probe_index is not None:
            import threading
            monitor_stop = threading.Event()
            def sample_gpu():
                while not monitor_stop.is_set():
                    sample_time = time.time()
                    result = subprocess.run([
                        'nvidia-smi', '--query-gpu=utilization.gpu,memory.used',
                        '--format=csv,noheader,nounits'], capture_output=True, text=True)
                    if result.returncode == 0:
                        try:
                            util, memory = result.stdout.strip().splitlines()[0].split(',')
                            gpu_samples.append((sample_time, float(util), float(memory)))
                        except (ValueError, IndexError):
                            pass
                    monitor_stop.wait(1.0)
            monitor_thread = threading.Thread(target=sample_gpu, daemon=True)
            monitor_thread.start()
        gpu_stage_start_epoch = time.time()
        detection_compute_started = time.monotonic()
        initial_detection(detection, cached, c, show_progress=False, load_simple_features=False)
        gpu_stage_stop_epoch = time.time()
    finally:
        if monitor_stop is not None:
            monitor_stop.set()
        if monitor_thread is not None:
            monitor_thread.join(timeout=5)
        if gpu_slot_file is not None:
            fcntl.flock(gpu_slot_file.fileno(), fcntl.LOCK_UN)
            gpu_slot_file.close()
        if gpu_lock_file is not None and gpu_global_lock_held:
            fcntl.flock(gpu_lock_file.fileno(), fcntl.LOCK_UN)
        if gpu_lock_file is not None:
            gpu_lock_file.close()
    detection_s = time.monotonic() - detection_compute_started
    gpu_probe_summary = None
    if probe_index is not None:
        gpu_probe_summary = dict(index=probe_index, samples=len(gpu_samples),
            mean_utilization_percent=float(np.mean([x[1] for x in gpu_samples])) if gpu_samples else None,
            peak_memory_mib=float(np.max([x[2] for x in gpu_samples])) if gpu_samples else None)
        log_dir = os.environ.get('DARTSORT_GPU_LOG_DIR')
        if log_dir:
            log_path = Path(log_dir)
            log_path.mkdir(parents=True, exist_ok=True)
            temporary = log_path/f"{w['id']}.csv.tmp"
            with open(temporary, 'w') as f:
                f.write('time_epoch,utilization_percent,memory_mib\n')
                for sample_time, util, memory in gpu_samples:
                    f.write(f'{sample_time:.6f},{util:.3f},{memory:.3f}\n')
            os.replace(temporary, log_path/f"{w['id']}.csv")
    with h5py.File(detection/'subtraction.h5', 'r') as h:
        samples = h['times_samples'][:]
        depths = h['point_source_localizations'][:, 2]
        amps = abs(h['denoised_ptp_amplitudes'][:])
        fs = float(h['sampling_frequency'][()])
    valid = np.isfinite(depths) & np.isfinite(amps) & (amps > 0)
    assert valid.sum() > 100, 'Insufficient valid peaks'
    np.savez_compressed(stage/'population.npz', time_s=samples[valid]/fs,
                        depth_um=depths[valid], amplitude=amps[valid])
    save(stage/'audit.json', dict(seconds=time.monotonic()-started, input_rows=len(samples),
        valid_rows=int(valid.sum()), excluded_rows=int((~valid).sum()),
        time_range_s=[float(samples[valid].min()/fs), float(samples[valid].max()/fs)],
        amplitude_units='DARTsort denoised PTP in standardized voltage units',
        package_versions={k:importlib.metadata.version(k) for k in ['dartsort', 'spikeinterface', 'torch']},
        raw_stat=spec['raw_stat'], no_sort_or_voltage_motion_correction=True,
        phase_seconds=dict(lock_wait=lock_wait_s, raw_read=raw_read_s,
                           preprocess=preprocess_s, gpu_lock_wait=gpu_lock_wait_s,
                           detection_and_denoiser_fit=detection_s),
        gpu_schedule=dict(mode=gpu_schedule_mode,start_epoch=gpu_stage_start_epoch,
                          stop_epoch=gpu_stage_stop_epoch),
        gpu_probe=gpu_probe_summary))
    del cached
    # Only remove this job's disposable materialization after successful extraction.
    shutil.rmtree(cached_path)
    check_sources(cfg, 'dartsort')
    seal(stage)


def motion_metrics(times, field):
    import numpy as np
    assert field.shape == (len(times), 4) and np.isfinite(field).all()
    assert np.allclose(np.diff(times), .25, atol=1e-5)
    # Remove independent constant depth offsets; retain dynamic depth dependence.
    centered = field - np.median(field, axis=0)
    rigid = np.median(centered, axis=1)
    spread = np.percentile(centered, 95, axis=1)-np.percentile(centered, 5, axis=1)
    speed = abs(np.diff(centered, axis=0)) / np.diff(times)[:, None]
    local_exc = np.percentile(centered, 95, axis=0)-np.percentile(centered, 5, axis=0)
    return dict(rigid_excursion_p95_p5_um=float(np.ptp(np.percentile(rigid, [5,95]))),
        median_depth_excursion_p95_p5_um=float(np.median(local_exc)),
        max_depth_excursion_p95_p5_um=float(local_exc.max()),
        median_nonrigid_spread_um=float(np.median(spread)), p95_nonrigid_spread_um=float(np.percentile(spread,95)),
        p99_local_speed_um_s=float(np.percentile(speed,99)),
        p99_rigid_speed_um_s=float(np.percentile(abs(np.diff(rigid))/np.diff(times),99)),
        p99_local_1s_step_um=float(np.percentile(abs(centered[4:]-centered[:-4]),99)),
        max_depth_full_range_um=float(np.ptp(centered,axis=0).max()))


def fit(out, cfg, w):
    import numpy as np
    import torch
    import medicine
    import importlib.metadata
    stage = out/w['id']/'fit'
    stage.mkdir(exist_ok=False)
    check_sources(cfg, 'medicine')
    assert complete(out/w['id']/'extraction')
    p = np.load(out/w['id']/'extraction/population.npz')
    torch.set_num_threads(4)
    assert torch.cuda.is_available()
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    begin = time.monotonic()
    trainer = medicine.run_medicine(peak_times=p['time_s'], peak_depths=p['depth_um'],
        peak_amplitudes=p['amplitude'], output_dir=stage/'medicine', optimizer=torch.optim.Adam, **cfg['medicine'])
    t = np.load(stage/'medicine/time_bins.npy')
    d = np.load(stage/'medicine/depth_bins.npy')
    m = np.load(stage/'medicine/motion.npy')
    metrics = motion_metrics(t, m)
    np.savez_compressed(stage/'field.npz', time_s=t, session_time_s=t+w['start_s'], depth_um=d, displacement_um=m)
    np.save(stage/'loss.npy', np.asarray(trainer.losses))
    counts, _, _ = np.histogram2d(p['time_s'], p['depth_um'], bins=[np.arange(0,120.25,.25), np.linspace(p['depth_um'].min(), p['depth_um'].max(), 5)])
    np.save(stage/'peak_counts_250ms_4depth.npy', counts)
    save(stage/'summary.json', dict(window=w['id'], dataset=w['dataset'], probe=w['probe'], fraction=w['fraction'],
        start_s=w['start_s'], stop_s=w['stop_s'], seconds=time.monotonic()-begin,
        peaks=len(p['time_s']), empty_support_fraction=float(np.mean(counts==0)),
        median_peaks_per_time_depth_bin=float(np.median(counts)),
        depth_min_um=float(d.min()), depth_max_um=float(d.max()),
        time_min_s=float(t.min()), time_max_s=float(t.max()),
        medicine_version=importlib.metadata.version('medicine-neuro'), **metrics))
    check_sources(cfg, 'medicine')
    seal(stage)


def report(out, cfg):
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows, noise = [], []
    for w in cfg['windows']:
        stage = out/w['id']
        if not (stage/'fit/complete.json').exists():
            continue
        rows.append(json.loads((stage/'fit/summary.json').read_text()))
        n = pd.read_csv(stage/'extraction/noise_channel_samples.csv')
        ch = n.groupby(['dataset','probe','window','reference','channel']).sigma_uv.median().reset_index()
        noise.extend(ch.groupby(['dataset','probe','window','reference']).sigma_uv.agg(['median','min','max']).reset_index().to_dict('records'))
        p = np.load(stage/'extraction/population.npz')
        f = np.load(stage/'fit/field.npz')
        fig, axes = plt.subplots(3,1,figsize=(12,9),layout='constrained')
        ix = np.linspace(0,len(p['time_s'])-1,min(150000,len(p['time_s'])),dtype=int)
        axes[0].scatter(p['time_s'][ix],p['depth_um'][ix],s=.1,c='0.2',alpha=.25,rasterized=True)
        axes[0].set(ylabel='Localized depth (µm)',title=w['id']+' — raw localized peaks')
        for j, depth in enumerate(f['depth_um']):
            v = f['displacement_um'][:,j]
            axes[1].plot(f['time_s'],v-np.median(v),label=f'{depth:.0f} µm',lw=.9)
        axes[1].legend(ncol=4,fontsize=8)
        axes[1].set(ylabel='Displacement (µm)',title='Fast MEDiCINe; each depth centered by its temporal median')
        counts=np.load(stage/'fit/peak_counts_250ms_4depth.npy')
        im=axes[2].imshow(np.log1p(counts.T),origin='lower',aspect='auto',extent=[0,120,p['depth_um'].min(),p['depth_um'].max()])
        fig.colorbar(im,ax=axes[2],label='log(1 + peak count)')
        axes[2].set(xlabel='Seconds within window',ylabel='Depth (µm)',title='Measurement support: 250 ms × four equal depth intervals')
        fig.savefig(stage/'diagnostic.png',dpi=150)
        plt.close(fig)
    write_csv(out/'motion_summary.csv', rows)
    write_csv(out/'noise_summary.csv', noise)
    if rows:
        table=pd.DataFrame(rows)
        fig,axes=plt.subplots(1,3,figsize=(14,4.5),layout='constrained')
        groups=[(s['dataset'],s['probe']) for s in cfg['records']]
        for ax,(metric,label) in zip(axes,[('median_depth_excursion_p95_p5_um','Median depth excursion P95−P5 (µm)'),('p99_local_speed_um_s','P99 local estimated speed (µm/s)'),('p95_nonrigid_spread_um','P95 dynamic depth spread (µm)')]):
            for i,(dataset,probe) in enumerate(groups):
                sub=table[(table.dataset==dataset)&(table.probe==probe)]
                ax.scatter(i+(sub.fraction-.5)*.3,sub[metric],s=35)
            ax.set_xticks(range(len(groups)),[a+'\n'+b for a,b in groups],fontsize=8)
            ax.set_ylabel(label)
            ax.set_ylim(bottom=0)
            ax.grid(axis='y',alpha=.25)
        fig.suptitle(f'Fast MEDiCINe matched-window pilot: {len(rows)}/18 fits; each point is 120 s')
        fig.savefig(out/'comparison.png',dpi=160)
        plt.close(fig)
    save(out/'progress.json', dict(completed_fits=len(rows), planned_fits=len(cfg['windows']), updated_at=time.time()))


def run(out, cfg):
    for w in cfg['windows']:
        for phase, python in [('extract', DSPY), ('fit', MEDPY)]:
            stage=out/w['id']/('extraction' if phase=='extract' else 'fit')
            if complete(stage):
                continue
            print(phase.upper(),w['id'],flush=True)
            job=out/w['id']
            job.mkdir(exist_ok=True)
            cmd=[str(python),str(Path(__file__).resolve()),phase,'--output',str(out),'--window',w['id']]
            save(job/(phase+'_command.json'),cmd)
            with (job/(phase+'_stdout.log')).open('x') as stdout, (job/(phase+'_stderr.log')).open('x') as stderr:
                proc=subprocess.run(cmd,stdout=stdout,stderr=stderr)
            save(job/(phase+'_exit.json'),dict(returncode=proc.returncode,finished_at=time.time()))
            if proc.returncode:
                raise RuntimeError(f'{phase} failed for {w["id"]}; inspect preserved logs')
        subprocess.run([str(DSPY),str(Path(__file__).resolve()),'report','--output',str(out)],check=True)
    save(out/'summary.json',dict(status='complete',fits=len(cfg['windows']),finished_at=time.time()))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('phase',choices=['prepare','extract','fit','report','run'])
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--window')
    args=ap.parse_args()
    out=args.output.resolve()
    if args.phase=='prepare':
        prepare(out)
        return
    cfg=json.loads((out/'config.json').read_text())
    if args.phase in ['extract','fit']:
        w=next(w for w in cfg['windows'] if w['id']==args.window)
        globals()[args.phase](out,cfg,w)
    else:
        globals()[args.phase](out,cfg)


if __name__=='__main__':
    main()
