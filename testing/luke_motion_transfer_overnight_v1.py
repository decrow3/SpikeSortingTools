"""Three bounded AP transfer epochs; checkpoint complete20s detection/localization chunks."""
import hashlib
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import time
import uuid

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt
from spikeinterface.core import NumpyRecording
from spikeinterface.sortingcomponents.peak_detection import detect_peaks
from spikeinterface.sortingcomponents.peak_localization import localize_peaks
from threadpoolctl import threadpool_limits
import torch

from testing.luke_epoch_corroboration import ROOT, BASE
from testing.luke_dredge_bounded import estimate_bounded

SRC = ROOT / 'testing/outputs'
OUT = SRC / 'luke_motion_transfer_overnight_v1'
EPOCHS = [('early', 940, 1040), ('middle', 6000, 6100), ('late', 9480, 9580)]
CACHE = SRC / 'luke_transfer_template_holdout_v1'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with tmp.open('w') as f:
        json.dump(value, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def validate_arrays(p, y, n, nch):
    assert len(p) == len(y) and len(p) > 0
    assert np.all((p['sample_index'] >= 50) & (p['sample_index'] < n - 50))
    assert np.all(p['segment_index'] == 0)
    assert np.all((p['channel_index'] >= 0) & (p['channel_index'] < nch))
    k = p['sample_index'].astype(np.int64) * nch + p['channel_index']
    assert np.all(np.diff(k) > 0), 'Unsorted or duplicate detection keys'
    assert np.isfinite(p['amplitude']).all()
    for field in y.dtype.names:
        assert np.isfinite(y[field]).all()


def committed_chunk(folder, expected):
    marker = folder / 'complete.json'
    if not marker.exists():
        return None
    receipt = json.loads(marker.read_text())
    assert receipt['input'] == expected, 'Checkpoint input/settings mismatch; investigate before restart'
    attempt = folder / receipt['attempt']
    for name, sha in receipt['sha256'].items():
        assert digest(attempt / name) == sha, 'Checkpoint artifact hash mismatch'
    p = np.load(attempt / 'peaks.npy')
    y = np.load(attempt / 'locations.npy')
    validate_arrays(p, y, expected['num_samples'], expected['num_channels'])
    return p, y


def chunk(start, settings_hash, raw, manifest, model, noise, geo):
    fs = manifest['sampling_frequency_hz']
    n, pad, first = round(20 * fs), round(.05 * fs), round(start * fs)
    with raw.open('rb') as f:
        f.seek((first - pad) * len(geo) * 2)
        buf = f.read((n + 2 * pad) * len(geo) * 2)
    assert len(buf) == (n + 2 * pad) * len(geo) * 2
    expected = dict(settings_sha256=settings_hash, raw_bytes_sha256=hashlib.sha256(buf).hexdigest(),
                    first_frame=first, num_samples=n, num_channels=len(geo), padding_samples=pad)
    folder = OUT / 'chunks' / f's{start}'
    folder.mkdir(parents=True, exist_ok=True)
    cached = committed_chunk(folder, expected)
    if cached is not None:
        print(start, 'validated checkpoint reused', flush=True)
        return cached, True
    attempt = folder / ('attempt_' + uuid.uuid4().hex)
    attempt.mkdir()
    atomic_json(attempt / 'input.json', expected)
    tick = time.monotonic()
    x = np.frombuffer(buf, dtype='<i2').reshape(-1, len(geo)).astype('float32') * manifest['gain_uv_per_count']
    del buf
    x = sosfiltfilt(butter(3, [300, 6000], fs=fs, btype='bandpass', output='sos'), x, axis=0).astype('float32')
    ref = np.median(x, axis=1)
    clean = np.empty((n, len(geo)), dtype='float32')
    for lo in range(0, n, 10000):
        e = np.arange(lo, min(n, lo + 10000))
        clean[e] = x[pad + e] - ref[pad + e[:, None] + model['lag_samples']] @ model['coefficients']
    del x, ref
    record = NumpyRecording(clean, fs)
    record.set_channel_locations(geo)
    p = detect_peaks(record, method='locally_exclusive', peak_sign='neg', radius_um=50.,
                     detect_threshold=3., noise_levels=noise, n_jobs=1,
                     chunk_duration='1s', progress_bar=False)
    p = p[(p['sample_index'] >= 50) & (p['sample_index'] < n - 50)]
    order = np.argsort(p['sample_index'].astype(np.int64) * len(geo) + p['channel_index'], kind='stable')
    p = p[order]
    print(start, 'localizing', len(p), flush=True)
    y = localize_peaks(record, p, method='monopolar_triangulation', radius_um=75.,
                       n_jobs=4, mp_context='fork', chunk_duration='1s', progress_bar=False)
    validate_arrays(p, y, n, len(geo))
    for name, array in [('peaks.npy', p), ('locations.npy', y)]:
        with (attempt / name).open('wb') as f:
            np.save(f, array)
            f.flush()
            os.fsync(f.fileno())
    receipt = dict(input=expected, attempt=attempt.name, peaks=len(p), elapsed_s=time.monotonic() - tick,
                   sha256={name: digest(attempt / name) for name in ['peaks.npy', 'locations.npy']})
    atomic_json(attempt / 'receipt.json', receipt)
    atomic_json(folder / 'complete.json', receipt)
    print(start, 'checkpoint committed', receipt['elapsed_s'], flush=True)
    return (p, y), False


def savefig(fig, folder, name):
    for ext in ['png', 'pdf']:
        fig.savefig(folder / f'{name}.{ext}', dpi=140)
    plt.close(fig)


def support_figure(folder, field):
    t, depth = field['time_s'], field['depth_um']
    d, c, u = (np.asarray(field[k]) for k in ['D','C','U'])
    assert d.shape == c.shape == u.shape == (len(depth), len(t), len(t))
    weights = np.where(np.isfinite(u) & (u > 0), u, 0.).copy()
    weights[:, np.arange(len(t)), np.arange(len(t))] = 0
    mass = weights.sum(axis=2)
    fraction = np.divide((weights * (abs(d) >= 80)).sum(axis=2), mass,
                         out=np.full_like(mass, np.nan), where=mass > 0)
    corr = np.divide((weights * c).sum(axis=2), mass, out=np.full_like(mass,np.nan), where=mass > 0)
    rows = [dict(depth_um=z, time_s=tt, outgoing_weight_mass=mass[b,i],
                 positive_edges=int((weights[b,i] > 0).sum()), bound_weight_fraction=fraction[b,i],
                 weighted_correlation=corr[b,i]) for b,z in enumerate(depth) for i,tt in enumerate(t)]
    pd.DataFrame(rows).to_csv(folder / 'pairwise_support.csv',index=False)
    rng = np.random.default_rng(20260908)
    triples = np.array([rng.choice(len(t),3,replace=False) for _ in range(2000)])
    cycles = []
    for b,z in enumerate(depth):
        i,j,k = triples.T
        keep = (weights[b,i,j] > 0) & (weights[b,j,k] > 0) & (weights[b,i,k] > 0)
        residual = abs(d[b,i[keep],j[keep]] + d[b,j[keep],k[keep]] - d[b,i[keep],k[keep]])
        cycles.append(dict(depth_um=z, sampled_triples=2000, supported_triples=int(keep.sum()),
            cycle_median_um=float(np.median(residual)) if len(residual) else np.nan,
            cycle_p95_um=float(np.quantile(residual,.95)) if len(residual) else np.nan))
    table = pd.DataFrame(cycles)
    table.to_csv(folder / 'sampled_cycle_consistency.csv',index=False)
    fig,axs = plt.subplots(1,3,figsize=(16,5),layout='constrained')
    for ax,data,title in zip(axs[:2],[np.log1p(mass),fraction],
                            ['log(1 + outgoing constraint weight)','Weight fraction at±80µm bound']):
        im=ax.pcolormesh(t,depth,data,shading='nearest',vmin=0,vmax=1 if data is fraction else None)
        fig.colorbar(im,ax=ax);ax.set(title=title,xlabel='Recording time (s)',ylabel='Depth (µm)')
    axs[2].plot(table.cycle_median_um,depth,label='Median')
    axs[2].plot(table.cycle_p95_um,depth,label='95th percentile')
    axs[2].set(title='Supported sampled triangle residual',xlabel='abs(Dij + Djk − Dik),µm',ylabel='Depth (µm)')
    axs[2].legend()
    fig.suptitle('Pairwise support and internal consistency are descriptive, not biological validation\nDiagonal excluded;2000 seeded triples per depth, requiring three positive-weight edges')
    savefig(fig,folder,'03_pairwise_support')


def figures(folder, label, start, end, p, y, field, fs, tracks, events, waves):
    t, depth, disp = field['time_s'], field['depth_um'], field['displacement_um']
    te, de = np.arange(start, end + .25, .25), np.arange(0, 3841, 10)
    count = np.histogram2d(start + p['sample_index'] / fs, y['y'], bins=[te, de])[0].T
    mass = np.histogram2d(start + p['sample_index'] / fs, y['y'], bins=[te, de], weights=abs(p['amplitude']))[0].T
    np.savez_compressed(folder / 'raster.npz', time_edges=te, depth_edges=de, count=count, amplitude_mass=mass)
    relative = disp - np.median(disp[(t >= start) & (t < start + 10)], axis=0)
    fig, axs = plt.subplots(3, 1, figsize=(13, 10), sharex=True, layout='constrained')
    for ax, data, title in zip(axs[:2], [count, mass], ['Peak count', 'Absolute amplitude sum']):
        z = np.log1p(data)
        im = ax.imshow(z, aspect='auto', origin='lower', extent=[start, end, 0, 3840],
                       cmap='magma', vmin=0, vmax=np.quantile(z, .995))
        ax.set(title=title, ylabel='Depth (µm)')
        fig.colorbar(im, ax=ax, label='log(1 + bin total)')
    lim = max(1., float(np.quantile(abs(relative), .99)))
    im = axs[2].pcolormesh(t, depth, relative.T, shading='nearest', cmap='RdBu_r', vmin=-lim, vmax=lim)
    fig.colorbar(im, ax=axs[2], label='Relative displacement (µm)')
    axs[2].set(xlabel='Recording time (s)', ylabel='Depth (µm)', title='DREDGE: each depth referenced to its first10s median')
    fig.suptitle(f'{label}: frozen compensated3σ AP input, no screening\nDescriptive bounded transfer; colors clipped at stated quantiles')
    savefig(fig, folder, '01_motion_raster')
    support_figure(folder, field)
    selected = tracks[(tracks.time_s >= start) & (tracks.time_s < end)]
    if selected.empty:
        fig, ax = plt.subplots(figsize=(12, 4), layout='constrained')
        ax.axis('off')
        ax.text(.5, .5, 'No cached waveform reference in this100s interval.\n'
                'Motion/raster coverage is descriptive only; biological validation is unavailable.\n'
                'No identity, stationary-motion, or waveform-preservation conclusion is made.',
                ha='center', va='center', transform=ax.transAxes)
        savefig(fig, folder, '02_cached_waveform_corroboration')
        pd.DataFrame(columns=['candidate','time_s','events','centroid_change_um',
                              'event_matched_dredge_um','status']).to_csv(folder / 'cached_observations.csv', index=False)
        return
    fig, axes = plt.subplots(selected.candidate.nunique(), 2, figsize=(14, 4 * selected.candidate.nunique()), squeeze=False, layout='constrained')
    observations = []
    for axs, (candidate, g) in zip(axes, selected.groupby('candidate')):
        g = g.sort_values('time_s')
        valid = (g.events >= 10) & np.isfinite(g.centroid_um)
        if not valid.any():
            for ax in axs:
                ax.axis('off')
                ax.text(.5, .5, f'{candidate}: no valid cached bins; reference gap', ha='center', transform=ax.transAxes)
            for row in g.itertuples():
                observations.append(dict(candidate=candidate, time_s=row.time_s, events=row.events,
                    centroid_change_um=np.nan, event_matched_dredge_um=np.nan, status='no_valid_cached_reference'))
            continue
        anchor = g[valid].iloc[0]
        trace = np.array([np.interp(anchor.depth_um, depth, row) for row in disp])
        own_events = events[events.candidate == candidate].frame.to_numpy() / fs
        baseline_events = own_events[(own_events >= anchor.time_s - 2.5) & (own_events < anchor.time_s + 2.5)]
        baseline = float(np.median(np.interp(baseline_events, t, trace)))
        axs[0].plot(t, trace - baseline, alpha=.5, label='Continuous local DREDGE')
        vals = []
        for row in g.itertuples():
            ev = own_events[(own_events >= row.time_s - 2.5) & (own_events < row.time_s + 2.5)]
            pred = float(np.median(np.interp(ev, t, trace)) - baseline) if len(ev) >= 10 else np.nan
            observed = float(row.centroid_um - anchor.centroid_um) if row.events >= 10 else np.nan
            vals.append(pred)
            observations.append(dict(candidate=candidate, time_s=row.time_s, events=len(ev),
                                     centroid_change_um=observed, event_matched_dredge_um=pred,
                                     status='provisional_fixed_support;identity_unqualified'))
        axs[0].plot(g.time_s, vals, 'o-', label='DREDGE at cached event times')
        axs[0].plot(g.time_s, np.where(valid, g.centroid_um - anchor.centroid_um, np.nan), 'ko-', label='Cached footprint; provisional')
        for row in g.itertuples():
            axs[0].annotate(f'n={row.events}', (row.time_s, 0), xytext=(0, 5), textcoords='offset points', fontsize=7)
        axs[0].set(title=f'{candidate} at{anchor.depth_um:.0f}µm: gaps outside cached holdout', xlabel='Recording time (s)', ylabel='Relative µm', xlim=(start, end))
        axs[0].legend(fontsize=7)
        for key in waves.files:
            if key.startswith(candidate + '_bin') and key.endswith('_waveform'):
                w = waves[key]
                axs[1].plot(np.arange(-30, 31) / fs * 1000, w[:, np.argmax(np.max(abs(w), axis=0))], label=key.split('_bin')[1].split('_')[0])
        axs[1].set(title='Cached original-voltage median waveforms\nStrongest channel per bin; not compensation-preservation evidence', xlabel='Time (ms)', ylabel='µV')
        axs[1].legend(title='Holdout bin', fontsize=7)
    fig.suptitle('Sparse cached corroboration only: fixed support and sparse competitors do not establish identity\nNo ground-truth error score; no reference outside the20s holdout or at shallow depths')
    savefig(fig, folder, '02_cached_waveform_corroboration')
    pd.DataFrame(observations).to_csv(folder / 'cached_observations.csv', index=False)


def run_locked():
    started = time.monotonic()
    raw = BASE / 'recording/traces_cached_seg0.raw'
    manifest_path = BASE / 'recording/rescue_recording_manifest.json'
    model_path = SRC / 'luke_common_event_screen_v1/shared_response_model.npz'
    noise_path = SRC / 'luke_peak_threshold_screen_v1/noise_uv.npy'
    baseline_path = SRC / 'luke_detection_threshold_sweep_v1/settings.json'
    source_paths = [manifest_path, model_path, noise_path, baseline_path, Path(__file__).resolve(),
                    ROOT / 'testing/luke_dredge_bounded.py', CACHE / 'tracks.csv',
                    CACHE / 'matched_events.csv', CACHE / 'waveforms_events.npz']
    hashes = {str(p): digest(p) for p in source_paths}
    manifest = json.loads(manifest_path.read_text())
    baseline = json.loads(baseline_path.read_text())
    assert hashes[str(noise_path)] == baseline['noise_sha256']
    assert hashes[str(model_path)] == baseline['model_sha256']
    fs = manifest['sampling_frequency_hz']
    geo = np.asarray(manifest['channel_locations_um'])
    stat = raw.stat()
    settings = dict(epochs=EPOCHS, estimator=baseline['estimator'], source_sha256=hashes,
        versions={name: importlib.metadata.version(name) for name in ['numpy','scipy','spikeinterface','torch','threadpoolctl']},
        raw=dict(path=str(raw), size=stat.st_size, mtime_ns=stat.st_mtime_ns, device=stat.st_dev, inode=stat.st_ino),
        sampling_frequency_hz=fs, detection=dict(method='locally_exclusive', peak_sign='neg', threshold=3, radius_um=50),
        localization=dict(method='monopolar_triangulation', radius_um=75, n_jobs=4), compute_threads=1,
        preprocessing='300–6000Hz third-order zero-phase Butterworth,50ms padding; frozen31tap shared-response compensation; original frozen noise',
        checkpoints='Each20s peak/localization pair commits atomically after validation, source/settings hash and output hashes. Restart rereads bounded raw bytes to verify input, skips only validated complete chunks, restarts interrupted chunks entirely; attempt evidence preserved. No within-chunk or within-DREDGE checkpoint.',
        reference='Existing early970–990/late9520–9540 fixed-support candidates only; sparse competitor identity and centroid sensitivity unqualified; no shallow reference and no reference outside these holdouts.',
        scope='Three100s epochs only; baseline3sigma only; no sweep, sort, production correction, or full-recording scan')
    serialized = json.dumps(settings, sort_keys=True)
    settings_hash = hashlib.sha256(serialized.encode()).hexdigest()
    OUT.mkdir(exist_ok=True)
    if (OUT / 'settings.json').exists():
        assert json.dumps(json.loads((OUT / 'settings.json').read_text()), sort_keys=True) == serialized, 'Settings changed: refuse resume'
    else:
        atomic_json(OUT / 'settings.json', settings)
    model, noise = np.load(model_path), np.load(noise_path)
    assert np.isfinite(noise).all() and (noise > 0).all()
    tracks = pd.read_csv(CACHE / 'tracks.csv')
    events = pd.read_csv(CACHE / 'matched_events.csv')
    waves = np.load(CACHE / 'waveforms_events.npz')
    torch.set_num_threads(1)
    summaries = []
    with threadpool_limits(limits=1):
        for label, start, end in EPOCHS:
            tick = time.monotonic()
            pp, yy, reused = [], [], 0
            for second in range(start, end, 20):
                (p, y), hit = chunk(second, settings_hash, raw, manifest, model, noise, geo)
                p = p.copy()
                p['sample_index'] += round(second * fs) - round(start * fs)
                pp.append(p); yy.append(y); reused += int(hit)
            p, y = np.concatenate(pp), np.concatenate(yy)
            epoch_folder = OUT / label
            epoch_folder.mkdir(exist_ok=True)
            folder = epoch_folder / ('attempt_' + uuid.uuid4().hex)
            folder.mkdir()
            np.save(folder / 'peaks.npy', p); np.save(folder / 'locations.npy', y)
            record = NumpyRecording(np.broadcast_to(np.zeros((1, len(geo)), dtype='float32'), (round((end-start)*fs), len(geo))), fs)
            record.set_channel_locations(geo)
            motion, extra = estimate_bounded(record, p, y, settings['estimator'])
            field = dict(time_s=motion.temporal_bins_s[0]+start, depth_um=motion.spatial_bins_um,
                         displacement_um=motion.displacement[0], D=extra['D'], C=extra['C'], U=extra['U'])
            np.savez_compressed(folder / 'motion.npz', **field)
            figures(folder, label, start, end, p, y, field, fs, tracks, events, waves)
            summaries.append(dict(epoch=label, peaks=len(p), reused_chunks=reused, elapsed_s=time.monotonic()-tick,
                                  artifact_directory=str(folder.relative_to(OUT)),
                                  strict_pairwise_bounds_pass=bool(abs(extra['D']).max() <= 80)))
            atomic_json(folder / 'summary.json', summaries[-1])
            atomic_json(epoch_folder / 'complete.json', summaries[-1])
    assert hashes == {str(p): digest(p) for p in source_paths}, 'Source changed during run'
    after = raw.stat()
    assert (stat.st_size,stat.st_mtime_ns,stat.st_ino,stat.st_dev) == (after.st_size,after.st_mtime_ns,after.st_ino,after.st_dev)
    atomic_json(OUT / 'summary.json', dict(status='complete', epochs=summaries, elapsed_s=time.monotonic()-started,
        source_hashes_unchanged=True, interpretation='Transfer diagnostic, not validated whole-session motion'))


def main():
    OUT.mkdir(exist_ok=True)
    with (OUT / '.run.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        run_locked()


if __name__ == '__main__':
    main()
