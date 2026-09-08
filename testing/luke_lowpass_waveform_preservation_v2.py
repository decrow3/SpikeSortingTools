"""Same-event waveform preservation audit; provisional, fixed-support lighthouse events."""
import hashlib
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt
from testing.luke_epoch_corroboration import ROOT, BASE

SRC = ROOT / 'testing/outputs'
OUT = SRC / 'luke_lowpass_waveform_preservation_v2'
STARTS = (4180, 4240)
CAP = 100


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare_waveforms(broad, lowpass, depths):
    """Arrays end in time, channel; supports event batches or a median template."""
    axes = (-2, -1)
    cosine = np.sum(broad * lowpass, axis=axes) / np.maximum(
        np.sqrt(np.sum(broad**2, axis=axes) * np.sum(lowpass**2, axis=axes)), 1e-20)
    amplitude_ratio = np.max(abs(lowpass), axis=axes) / np.maximum(np.max(abs(broad), axis=axes), 1e-20)
    def centroid(w):
        energy = np.sum(w**2, axis=-2)
        return np.sum(energy * depths, axis=-1) / np.maximum(np.sum(energy, axis=-1), 1e-20)
    bc, lc = centroid(broad), centroid(lowpass)
    return cosine, amplitude_ratio, bc, lc, lc - bc


def reconstruct(start, manifest, model):
    fs = manifest['sampling_frequency_hz']
    n, pad, first = round(20 * fs), round(.05 * fs), round(start * fs)
    with (BASE / 'recording/traces_cached_seg0.raw').open('rb') as handle:
        handle.seek((first - pad) * 768)
        buf = handle.read((n + 2 * pad) * 768)
    x = np.frombuffer(buf, dtype='<i2').reshape(-1, 384).astype('float32') * manifest['gain_uv_per_count']
    del buf
    x = sosfiltfilt(butter(3, [300, 6000], fs=fs, btype='bandpass', output='sos'), x, axis=0).astype('float32')
    ref = np.median(x, axis=1)
    compensated = np.zeros_like(x)
    for lo in range(15, len(x)-15, 10000):
        e = np.arange(lo, min(len(x)-15, lo+10000))
        compensated[e] = x[e] - ref[e[:, None] + model['lag_samples']] @ model['coefficients']
    del x, ref
    lowpass = sosfiltfilt(butter(3, 3000, fs=fs, btype='lowpass', output='sos'), compensated, axis=0).astype('float32')
    return compensated[pad:pad+n], lowpass[pad:pad+n], first


def main():
    begun = time.monotonic()
    OUT.mkdir(exist_ok=False)
    manifest_path = BASE / 'recording/rescue_recording_manifest.json'
    model_path = SRC / 'luke_common_event_screen_v1/shared_response_model.npz'
    events_path = SRC / 'luke_lighthouse_gentle_v1/gentle_events.csv'
    tracks_path = SRC / 'luke_lighthouse_gentle_v1/gentle_tracks.csv'
    template_path = SRC / 'luke_lighthouse_gentle_v1/templates.npz'
    manifest = json.loads(manifest_path.read_text())
    fs = manifest['sampling_frequency_hz']
    geometry = np.asarray(manifest['channel_locations_um'])
    events, tracks = pd.read_csv(events_path), pd.read_csv(tracks_path)
    original_templates = np.load(template_path)
    model = np.load(model_path)
    settings = dict(intervals_s=[[s, s+20] for s in STARTS], events_per_unit_bin_cap=CAP,
                    event_selection='Existing accepted frames unchanged; uniform chronological subsample per unit and original 10s bin.',
                    supports='Original fixed template channels and +/-120um from original detector depth.',
                    waveform_window_samples=[-30, 30], filter='Same compensated 300-6000Hz broadband versus added zero-phase order3 3000Hz lowpass as six-arm v2.',
                    metrics='Same-event multichannel cosine, maximum absolute amplitude ratio, squared-energy centroid, and centroid difference. Median-template metrics reduce independent background noise; no redetection/relocalization.',
                    caveats=['Conditional on existing identities and frames; fixed-support selection may miss moving cells.',
                             'Energy centroid is a descriptive waveform statistic, not monopolar localization or calibrated displacement.',
                             'Accepted-event rate is inherited selection support, not lowpass recovery or detection recall.',
                             'Filtering changes temporal waveforms by design; cosine loss alone is not evidence of damage.'],
                    resume='No automatic resume; existing output directory refused.',
                    sha256={str(p): digest(p) for p in [Path(__file__).resolve(), manifest_path, model_path, events_path, tracks_path, template_path]})
    (OUT/'settings.json').write_text(json.dumps(settings, indent=2))
    rows, event_rows, templates = [], [], {}
    offsets = np.arange(-30, 31)
    for start in STARTS:
        broad, lowpass, first = reconstruct(start, manifest, model)
        for unit, group in tracks[(tracks.time_s >= start) & (tracks.time_s < start+20)].groupby('unit_id'):
            original_channels = original_templates[f'unit_{unit}_channels']
            depth = float(group.depth_um.iloc[0])
            supports = {'original': original_channels, 'expanded120': np.flatnonzero(abs(geometry[:, 1] - depth) <= 120)}
            for _, binrow in group.iterrows():
                pool = events[(events.unit_id == unit) & (events.time_s == binrow.time_s)].sort_values('frame')
                pool = pool[(pool.frame-first >= 30) & (pool.frame-first < len(broad)-30)]
                total = len(pool)
                pool = pool.iloc[np.linspace(0, total-1, min(CAP, total), dtype=int)] if total else pool
                frames = pool.frame.to_numpy(dtype='int64')
                for support, channels in supports.items():
                    row = dict(unit_id=int(unit), depth_um=depth, time_s=float(binrow.time_s), support=support,
                               accepted_events=int(binrow.accepted_events), usable_events=total, sampled_events=len(pool),
                               inherited_accepted_rate_hz=float(binrow.accepted_events)/10)
                    if not len(pool):
                        rows.append(row)
                        continue
                    index = frames[:, None, None] - first + offsets[None, :, None]
                    wb, wl = broad[index, channels[None, None, :]], lowpass[index, channels[None, None, :]]
                    metrics = compare_waveforms(wb, wl, geometry[channels, 1])
                    mb, ml = np.median(wb, axis=0), np.median(wl, axis=0)
                    names = ['cosine', 'amplitude_ratio', 'broad_centroid_um', 'lowpass_centroid_um', 'centroid_shift_um']
                    for name, value, template_value in zip(names, metrics, compare_waveforms(mb, ml, geometry[channels, 1])):
                        row['event_median_'+name] = float(np.median(value))
                        row['event_p10_'+name] = float(np.quantile(value, .1))
                        row['event_p90_'+name] = float(np.quantile(value, .9))
                        row['template_'+name] = float(template_value)
                    for i, frame in enumerate(frames):
                        event_rows.append(dict(unit_id=int(unit), time_s=float(binrow.time_s), frame=int(frame), support=support,
                                               **{name:float(value[i]) for name,value in zip(names, metrics)}))
                    key = f'unit{unit}_t{int(binrow.time_s)}_{support}'
                    templates[key+'_broad_uv'], templates[key+'_lowpass_uv'], templates[key+'_channels'] = mb, ml, channels
                    rows.append(row)
        del broad, lowpass
        print(start, 'same-event waveform audit complete', flush=True)
    data = pd.DataFrame(rows)
    data.to_csv(OUT/'per_unit_bin_metrics.csv', index=False)
    pd.DataFrame(event_rows).to_csv(OUT/'sampled_event_metrics.csv', index=False)
    np.savez_compressed(OUT/'median_templates.npz', **templates)
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), layout='constrained')
    for support, color in [('original', 'tab:blue'), ('expanded120', 'tab:orange')]:
        d = data[data.support == support]
        xx = np.arange(len(d))
        for ax, col, label in zip(axes, ['template_cosine', 'template_amplitude_ratio', 'template_centroid_shift_um'],
                                 ['Median-template cosine', 'Lowpass / broad peak amplitude', 'Lowpass - broad centroid (um)']):
            ax.plot(xx, d[col], 'o-', label=support, color=color, ms=4)
            ax.set_ylabel(label)
        axes[-1].set_xticks(xx, [f"u{int(r.unit_id)}\n{int(r.time_s)}s" for _, r in d.iterrows()], rotation=90, fontsize=7)
    axes[1].axhline(1, color='gray', ls=':'); axes[2].axhline(0, color='gray', ls=':'); axes[0].legend()
    fig.suptitle('Same accepted lighthouse events: broadband versus added 3kHz lowpass\nFixed-selection corroboration; energy-centroid shifts are not calibrated motion error')
    for ext in ['png', 'pdf']: fig.savefig(OUT/f'01_preservation.{ext}', dpi=140)
    (OUT/'summary.json').write_text(json.dumps(dict(status='complete', seconds=time.monotonic()-begun,
        metrics_rows=len(data), sampled_event_rows=len(event_rows), source_intervals_s=settings['intervals_s'],
        interpretation='Descriptive waveform preservation only; does not validate identity, unobserved events, or motion.'), indent=2))


if __name__ == '__main__':
    main()
