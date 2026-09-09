"""One full-session screened MEDiCINe estimate; never applies correction or sorts."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import time

import numpy as np

MED_ARGS = dict(motion_bound=500., time_bin_size=.25, time_kernel_width=1.,
                activity_network_hidden_features=(256, 256), num_depth_bins=4,
                amplitude_threshold_quantile=0., batch_size=4096, training_steps=10000,
                initial_motion_noise=.1, motion_noise_steps=2000, learning_rate=.0005,
                epsilon=.001, plot_figures=False)
SCREEN = dict(snr=6, center=.5, neighbor=.6)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)


def sealed(path):
    path = Path(path)
    if not path.exists():
        return False
    marker = path / 'complete.json'
    if not marker.exists():
        raise RuntimeError(f'Unsealed stage: {path}; preserve and investigate before relaunch')
    checks = json.loads(marker.read_text())
    if not checks:
        raise ValueError(f'Empty seal: {path}')
    for name, digest in checks.items():
        if sha(path / name) != digest:
            raise ValueError(f'Invalid sealed artifact: {path / name}')
    return True


def seal(path):
    save(path / 'complete.json', {str(p.relative_to(path)): sha(p)
                                for p in sorted(path.rglob('*'))
                                if p.is_file() and p.name != 'complete.json'})


def sample_chunks(total, fs):
    """Disjoint sample coverage; 10+20n seconds matches the historical cache grid."""
    if total <= 0 or not np.isfinite(fs) or fs <= 0:
        raise ValueError('Invalid recording dimensions')
    bounds = [0] + [round(t * fs) for t in np.arange(10., total / fs, 20.)] + [total]
    bounds = sorted(set(bounds))
    return list(zip(bounds[:-1], bounds[1:]))


def padded_read(raw, start, stop, total, channels, pad):
    """Read real context, reflect only beyond global endpoints, retain core bytes."""
    lo, hi = max(0, start - pad), min(total, stop + pad)
    stride = channels * 2
    with Path(raw).open('rb') as f:
        f.seek(lo * stride)
        buf = f.read((hi - lo) * stride)
    if len(buf) != (hi - lo) * stride:
        raise IOError('Short source read')
    core = memoryview(buf)[(start - lo) * stride:(stop - lo) * stride]
    x = np.frombuffer(buf, dtype='<i2').reshape(-1, channels).astype(np.float32)
    left, right = max(0, pad - start), max(0, stop + pad - total)
    if left or right:
        x = np.pad(x, ((left, right), (0, 0)), mode='reflect')
    if len(x) != stop - start + 2 * pad:
        raise ValueError('Incorrect padding')
    return x, core, [left, right]


def condition(x, n, pad, gain, fs, model):
    from scipy.signal import butter, sosfiltfilt
    x = sosfiltfilt(butter(3, [300, 6000], fs=fs, btype='bandpass', output='sos'),
                   x * gain, axis=0).astype('float32')
    reference = np.median(x, axis=1)
    lags = model['lag_samples']
    if np.max(np.abs(lags)) > pad:
        raise ValueError('Compensation support exceeds padding')
    clean = np.empty((n, x.shape[1]), dtype='float32')
    for lo in range(0, n, 10000):
        ev = np.arange(lo, min(n, lo + 10000))
        clean[ev] = x[pad + ev] - reference[pad + ev[:, None] + lags] @ model['coefficients']
    post = x[pad:pad + n] - reference[pad:pad + n, None]
    return clean, post


def extract(x, n, pad, manifest, model, noise):
    from spikeinterface.core import NumpyRecording
    from spikeinterface.sortingcomponents.peak_detection import detect_peaks
    from spikeinterface.sortingcomponents.peak_localization import localize_peaks
    from testing.luke_screen_sweep import features, mask
    fs = manifest['sampling_frequency_hz']
    geo = np.asarray(manifest['channel_locations_um'])
    clean, post = condition(x, n, pad, manifest['gain_uv_per_count'], fs, model)
    rec = NumpyRecording(clean, fs)
    rec.set_channel_locations(geo)
    p = detect_peaks(rec, method='locally_exclusive', peak_sign='neg', radius_um=50.,
                     detect_threshold=5., noise_levels=noise, n_jobs=1,
                     chunk_duration='1s', progress_bar=False)
    p = p[(p['sample_index'] >= 50) & (p['sample_index'] < n - 50)]
    # Apply the exact existing waveform features, bounded to 2048 events per batch.
    parts = [features(clean, post, p[i:i+2048], geo, model['residual_noise_uv'], fs)
             for i in range(0, len(p), 2048)]
    f = ({k: np.concatenate([part[k] for part in parts]) for k in parts[0]}
         if parts else features(clean, post, p, geo, model['residual_noise_uv'], fs))
    kept = np.flatnonzero(mask(f, **SCREEN))
    if len(kept):
        loc = localize_peaks(rec, p[kept], method='monopolar_triangulation', radius_um=75.,
                             n_jobs=4, mp_context='fork', chunk_duration='1s', progress_bar=False)
    else:
        loc = np.empty(0, dtype=[('x', 'f8'), ('y', 'f8'), ('z', 'f8'), ('alpha', 'f8')])
    if not np.isfinite(loc['y']).all():
        raise ValueError('Nonfinite localization')
    return p, f, kept, loc


def concatenate_chunks(stages, destination, total, fs):
    """Bound peak-array copying and preserve exact global sample identities."""
    counts = [len(np.load(p / 'peaks.npy', mmap_mode='r')) for p in stages]
    if not sum(counts):
        raise ValueError('No retained peaks in recording')
    first = stages[next(i for i, n in enumerate(counts) if n)]
    dtype = np.load(first / 'peaks.npy', mmap_mode='r').dtype
    ldtype = np.load(first / 'locations.npy', mmap_mode='r').dtype
    peaks = np.lib.format.open_memmap(destination / 'peaks.npy', mode='w+', dtype=dtype, shape=(sum(counts),))
    loc = np.lib.format.open_memmap(destination / 'locations.npy', mode='w+', dtype=ldtype, shape=(sum(counts),))
    offset = 0
    previous = -1
    for stage, number in zip(stages, counts):
        info = json.loads((stage / 'audit.json').read_text())
        p = np.load(stage / 'peaks.npy')
        y = np.load(stage / 'locations.npy')
        if len(p) != len(y):
            raise ValueError('Unpaired peaks/locations')
        if number:
            if np.any(p['sample_index'] < 0) or np.any(p['sample_index'] >= info['stop_sample'] - info['start_sample']):
                raise ValueError('Chunk-relative peak out of range')
            p['sample_index'] += info['start_sample']
            if p['sample_index'][0] < previous or np.any(np.diff(p['sample_index']) < 0):
                raise ValueError('Nonmonotonic global peak times')
            previous = p['sample_index'][-1]
            peaks[offset:offset+number] = p
            loc[offset:offset+number] = y
        offset += number
    peaks.flush(); loc.flush()
    save(destination / 'input.json', dict(start_s=0., stop_s=total/fs, sampling_frequency_hz=fs,
                                         num_samples=total, peaks=offset))


def fit(input_dir, output, settings):
    import inspect
    import medicine
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; no silent CPU/precision substitution')
    cfg = json.loads((input_dir / 'input.json').read_text())
    p = np.load(input_dir / 'peaks.npy', mmap_mode='r')
    loc = np.load(input_dir / 'locations.npy', mmap_mode='r')
    times = p['sample_index'] / cfg['sampling_frequency_hz'] + cfg['start_s']
    if len(p) != len(loc) or not len(p) or not np.isfinite(loc['y']).all():
        raise ValueError('Invalid fit input')
    if times.min() < cfg['start_s'] or times.max() >= cfg['stop_s']:
        raise ValueError('Incorrect fit time origin')
    torch.set_num_threads(4)
    np.random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    torch.cuda.reset_peak_memory_stats()
    impl = Path(inspect.getfile(medicine.run_medicine)).parent
    sources = {str(p): sha(p) for p in impl.glob('*.py')}
    begin = time.monotonic()
    trainer = medicine.run_medicine(peak_times=times, peak_depths=loc['y'],
                                   peak_amplitudes=abs(p['amplitude']), output_dir=output / 'native',
                                   optimizer=torch.optim.Adam, **settings)
    native = output / 'native'
    t, y, v = [np.load(native / name) for name in ['time_bins.npy', 'depth_bins.npy', 'motion.npy']]
    if v.shape != (len(t), settings['num_depth_bins']) or not np.isfinite(v).all():
        raise ValueError('Invalid fitted field')
    if not np.all(np.diff(t) > 0) or not np.all(np.diff(y) > 0):
        raise ValueError('Invalid field axes')
    # Native domain follows actual retained peaks; never manufacture terminal peaks.
    expected = int(np.ceil((times.max()-times.min()+2*settings['epsilon']) / settings['time_bin_size']))
    if len(t) != expected:
        raise ValueError('Unexpected time grid')
    np.savez_compressed(output / 'field.npz', time_s=t, depth_um=y, displacement_um=v)
    np.save(output / 'loss.npy', np.asarray(trainer.losses))
    if sources != {str(p): sha(p) for p in impl.glob('*.py')}:
        raise RuntimeError('MEDiCINe source changed during fit')
    save(output / 'fit_audit.json', dict(settings=settings, seed=0, optimizer='torch.optim.Adam',
         peaks=len(p), seconds=time.monotonic()-begin, max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
         cuda_max_allocated_bytes=torch.cuda.max_memory_allocated(), device=torch.cuda.get_device_name(),
         torch_version=torch.__version__, source_sha256=sources, field_support_s=[float(t[0]), float(t[-1])],
         native_depth_range_um=[float(y[0]), float(y[-1])], scientific_status='requires_review',
         checkpoint='No within-fit optimizer checkpoint; completed stages only'))


def child_fit(cfg, source, dest, label):
    if sealed(dest):
        return
    dest.mkdir()
    command = [cfg['medicine_python'], '-u', '-m', 'testing.luke_full_session_medicine',
               '--config', cfg['_config_path'], '--phase', 'fit', '--input', str(source), '--output', str(dest)]
    from testing.managed_job import run_managed_command
    save(dest / 'command.json', command)
    # A child stays in the enclosing independent systemd control group.
    code = run_managed_command(command, receipt_path=dest / 'receipt.json', cwd=Path(__file__).resolve().parents[1])
    if code:
        raise RuntimeError(f'{label} failed with exit {code}; preserve stage')
    seal(dest)


def run(cfg):
    out = Path(cfg['output'])
    out.mkdir(parents=True, exist_ok=True)
    start_time = time.monotonic()
    frozen = {k: v for k, v in cfg.items() if not k.startswith('_')}
    if (out / 'settings.json').exists():
        if json.loads((out / 'settings.json').read_text()) != frozen:
            raise ValueError('Configuration changed; use a new output version')
    else:
        save(out / 'settings.json', frozen)
    for path, expected in cfg['input_sha256'].items():
        if sha(path) != expected:
            raise ValueError(f'Input changed: {path}')
    manifest = json.loads(Path(cfg['recording_manifest']).read_text())
    total, channels, fs = manifest['num_samples'], manifest['num_channels'], manifest['sampling_frequency_hz']
    if channels != 384 or total != 314204894 or manifest['recording_content_sha256'] != cfg['recording_content_sha256']:
        raise ValueError('Wrong Luke recording')
    raw = Path(cfg['recording_binary'])
    before = [raw.stat().st_size, raw.stat().st_mtime_ns]
    if before[0] != total * channels * 2:
        raise ValueError('Wrong source length')
    save(out / 'source_start.json', dict(raw_stat=before, content_verification='pending streamed checksum'))

    def progress(stage, **kw):
        value = dict(stage=stage, updated_unix=time.time(), elapsed_s=time.monotonic()-start_time, **kw)
        save(out / 'progress.json', value)
        print(json.dumps(value), flush=True)

    progress('cached_100s_reproduction')
    child_fit(cfg, Path(cfg['control_input']), out / 'control_fit', 'reproduction')
    ref = np.load(cfg['control_field']); got = np.load(out / 'control_fit/field.npz')
    if not np.array_equal(ref['time_s'], got['time_s']) or not np.array_equal(ref['depth_um'], got['depth_um']):
        raise ValueError('Reproduction grids differ')
    difference = np.abs(ref['displacement_um'] - got['displacement_um'])
    save(out / 'reproduction.json', dict(max_difference_um=float(difference.max()), tolerance_um=.1))
    if difference.max() >= .1:
        raise ValueError('Cached field reproduction failed')

    model = np.load(cfg['model']); noise = np.load(cfg['noise'])
    pad = round(.05 * fs)
    cache = Path(cfg['cache'])
    old = json.loads((cache / 'settings.json').read_text())
    from testing.luke_screen_sweep import features
    import inspect
    old_sources = old['source_sha256']
    for path in [cfg['model'], cfg['noise'], cfg['recording_manifest']]:
        if old_sources[path] != sha(path):
            raise ValueError('Cache conditioning provenance mismatch')
    old_feature_sha = next(v for k, v in old_sources.items() if k.endswith('/testing/luke_screen_sweep.py'))
    if old_feature_sha != sha(inspect.getfile(features)) or old['raw_stat'] != before:
        raise ValueError('Cache source provenance mismatch')
    stages = []
    binary_hash = hashlib.sha256()
    chunks = sample_chunks(total, fs)
    for index, (start, stop) in enumerate(chunks):
        n = stop - start
        stage = out / f'chunk_{start:09d}'
        stages.append(stage)
        x, core, edge = padded_read(raw, start, stop, total, channels, pad)
        binary_hash.update(core)
        del core
        if not sealed(stage):
            stage.mkdir()
            seconds = int(round(start/fs))
            cached = cache / f'chunk_{seconds}'
            # Historical chunks used round(20*fs); reject one-sample mismatches.
            reusable = (round(seconds*fs) == start and n == round(20*fs)
                        and 930 <= seconds < 1230 and (seconds-930) % 20 == 0 and cached.exists())
            begin = time.monotonic()
            audit = dict(start_sample=start, stop_sample=stop, reflected_edge_samples=edge,
                         excluded_detector_margin_samples=50, reused_from=None)
            if reusable:
                sealed(cached)
                for name in ['peaks', 'locations']:
                    shutil.copy2(cached / f'5sigma_relaxed_{name}.npy', stage / f'{name}.npy')
                audit.update(reused_from=str(cached), cache_seal_sha256=sha(cached/'complete.json'),
                             retained=len(np.load(stage/'peaks.npy', mmap_mode='r')))
            else:
                p, f, kept, loc = extract(x, n, pad, manifest, model, noise)
                np.save(stage/'detected_peaks.npy', p)
                np.savez_compressed(stage/'features.npz', **f)
                np.save(stage/'retained_indices.npy', kept)
                np.save(stage/'peaks.npy', p[kept]); np.save(stage/'locations.npy', loc)
                audit.update(detected=len(p), retained=len(kept))
                del p, f, kept, loc
            audit.update(seconds=time.monotonic()-begin, max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            save(stage/'audit.json', audit)
            seal(stage)
        del x
        progress('extracting', completed_chunks=index+1, total_chunks=len(chunks), through_s=stop/fs)
    expected = next(p for p in manifest['recording_binary_files'] if p['name'] == raw.name)
    if binary_hash.hexdigest() != expected['sha256'] or before != [raw.stat().st_size, raw.stat().st_mtime_ns]:
        raise ValueError('Recording content mismatch; do not fit or publish')
    save(out/'source_verified.json', dict(file_sha256=binary_hash.hexdigest(), bytes=before[0], raw_stat=before,
                                         recording_content_sha256=cfg['recording_content_sha256']))
    source = out/'input'
    if not sealed(source):
        source.mkdir()
        concatenate_chunks(stages, source, total, fs)
        seal(source)
    progress('full_session_fit', peaks=json.loads((source/'input.json').read_text())['peaks'])
    child_fit(cfg, source, out/'fit', 'full-session fit')
    progress('report')
    report(out, manifest)
    for path, expected in cfg['input_sha256'].items():
        if sha(path) != expected:
            raise ValueError(f'Input changed during run: {path}')
    save(out/'summary.json', dict(status='complete', scientific_status='requires_review',
         interval_s=[0., total/fs], seconds=time.monotonic()-start_time,
         estimation_owner='huklaban1', fit_count_full_session=1,
         correction_applied=False, sorting_launched=False,
         next_step='Review field support and rigid/nonrigid application before either sort'))
    publish(out, cfg)
    progress('complete')


def publish(out, cfg):
    """Publish compact, hash-bound evidence once; never publish voltage or peak caches."""
    shared = Path(cfg['shared_output'])
    shared.mkdir(parents=True, exist_ok=True)
    destination = shared/'estimation_huklaban1_v1'
    partial = shared/'estimation_huklaban1_v1.partial'
    if destination.exists() or partial.exists() or (shared/'field_manifest.json').exists():
        raise RuntimeError('Shared destination already exists; preserve it and investigate')
    partial.mkdir()
    names = ['candidate_fields.npz', 'coverage.npz', 'field_review.json', 'full_session_fields.png',
             'full_session_fields.pdf', 'reproduction.json', 'source_verified.json', 'summary.json',
             'settings.json', 'fit/field.npz', 'fit/fit_audit.json', 'fit/loss.npy']
    for name in names:
        target = partial/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(out/name, target)
    package = dict(schema='luke-shared-medicine-field-v1', estimation_owner='huklaban1',
                   status='estimation_artifacts_complete', scientific_status='requires_review',
                   correction_ready=False, source_output=str(out),
                   field_package='estimation_huklaban1_v1',
                   recording_content_sha256=cfg['recording_content_sha256'],
                   files={name: sha(partial/name) for name in names},
                   final_job_exit_receipt='Check the independent service receipt; artifact completion is not exit status')
    save(partial/'manifest.json', package)
    os.rename(partial, destination)
    save(shared/'field_manifest.json', package)


def report(out, manifest):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    z = np.load(out/'fit/field.npz')
    t, y, v = z['time_s'], z['depth_um'], z['displacement_um']
    seed = (t >= 930) & (t < 940)
    if not seed.any():
        raise ValueError('Missing reference interval')
    # Preserve native values; seed-referenced arrays are separately named diagnostics.
    offset = np.median(v[seed], axis=0)
    centered = v - offset
    geo = np.asarray(manifest['channel_locations_um'])
    grid = np.linspace(max(y[0], geo[:,1].min()), min(y[-1], geo[:,1].max()), 101)
    if grid[0] >= grid[-1]:
        raise ValueError('No probe/field depth overlap')
    weights = np.interp(grid, y, np.arange(len(y)))
    lo = np.floor(weights).astype(int); hi = np.minimum(lo+1, len(y)-1); a = weights-lo
    rigid = np.mean(centered[:,lo]*(1-a) + centered[:,hi]*a, axis=1)
    np.savez_compressed(out/'candidate_fields.npz', time_s=t, depth_um=y,
                        nonrigid_displacement_um=centered, rigid_displacement_um=rigid,
                        seed_offset_um=offset, rigid_depth_grid_um=grid)
    p = np.load(out/'input/peaks.npy', mmap_mode='r'); loc = np.load(out/'input/locations.npy', mmap_mode='r')
    duration = manifest['num_samples']/manifest['sampling_frequency_hz']
    te = np.r_[np.arange(0, duration, 5.), duration]
    ye = np.arange(np.floor(geo[:,1].min()/100)*100, np.ceil(geo[:,1].max()/100)*100+101, 100)
    hist = np.zeros((len(te)-1, len(ye)-1))
    for start in range(0, len(p), 250000):
        sl = slice(start, start+250000)
        hist += np.histogram2d(p['sample_index'][sl]/manifest['sampling_frequency_hz'], loc['y'][sl], bins=(te, ye))[0]
    np.savez_compressed(out/'coverage.npz', counts=hist, time_edges_s=te, depth_edges_um=ye)
    gradient = np.diff(v, axis=1)/np.diff(y)[None,:]
    save(out/'field_review.json', dict(scientific_status='requires_review',
        field_support_s=[float(t[0]), float(t[-1])], recording_interval_s=[0., duration],
        peak_depth_range_um=[float(y[0]), float(y[-1])],
        rigid_depth_grid_is_model_domain_not_validated_cell_support=True,
        native_peak_to_peak_um=np.ptp(v, axis=0).tolist(),
        seed_referenced_max_abs_um=float(abs(centered).max()),
        max_abs_depth_gradient=float(abs(gradient).max()),
        observed_to_reference_nonpositive_grid_slopes=int(np.sum(1-gradient <= 0)),
        empty_5s_100um_cells=int(np.sum(hist == 0)),
        empty_5s_time_bins=int(np.sum(hist.sum(axis=1) == 0)),
        near_500um_peak_to_peak_columns=int(np.sum(np.ptp(v, axis=0) >= 475)),
        interpretation='Diagnostics only; rigid projection and supported application domain require review; no voltage authorization inferred'))
    fig, ax = plt.subplots(3, 1, figsize=(16, 10), sharex=True, constrained_layout=True)
    ax[0].pcolormesh(te, ye, np.log1p(hist.T), shading='auto', rasterized=True)
    ax[0].set_ylabel('Peak depth (µm)')
    for j, depth in enumerate(y):
        ax[1].plot(t, centered[:,j], label=f'{depth:.0f} µm', lw=.6)
    ax[1].plot(t, rigid, color='black', label='Rigid projection', lw=.8)
    ax[1].legend(ncol=5, fontsize=8); ax[1].set_ylabel('Displacement (µm)')
    ax[2].plot((te[1:]+te[:-1])/2, hist.sum(axis=1), lw=.7)
    ax[2].set(xlabel='Recording-relative time (s)', ylabel='Retained peaks / 5 s')
    fig.suptitle('Full-session screened MEDiCINe · 5σ/relaxed · 10,000 steps\nSeed 930–940 s; native field preserved; scientific review required')
    fig.savefig(out/'full_session_fields.png', dpi=150); fig.savefig(out/'full_session_fields.pdf')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--phase', choices=['run', 'fit'], default='run')
    ap.add_argument('--input', type=Path); ap.add_argument('--output', type=Path)
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text()); cfg['_config_path'] = str(args.config.resolve())
    if args.phase == 'fit':
        fit(args.input, args.output, cfg['medicine_settings'])
    else:
        run(cfg)


if __name__ == '__main__':
    main()
