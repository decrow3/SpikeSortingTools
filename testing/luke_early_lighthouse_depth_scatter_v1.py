"""Cached-only lighthouse scatter overlays on the original early depth/time view."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from testing.luke_epoch_corroboration import BASE

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'testing/outputs'
OUT = SRC / 'luke_early_lighthouse_depth_scatter_v1'

def main():
    OUT.mkdir(exist_ok=True)
    manifest = BASE / 'recording/rescue_recording_manifest.json'
    m = json.loads(manifest.read_text())
    fs = m['sampling_frequency_hz']
    geometry = np.asarray(m['channel_locations_um'])
    hist_path = SRC / 'luke_peak_population_review_v1/02_long_histograms_histograms.npz'
    window_path = SRC / 'luke_early_lighthouse_binning_v2/windows_with_motion.csv'
    hist = np.load(hist_path)
    windows = pd.read_csv(window_path)
    inputs = [manifest, hist_path, window_path]
    clouds, summaries, anchors = {}, {}, {}
    candidates = ['s960_c293_pos', 's960_c338_pos']
    for candidate in candidates:
        path = SRC / f'luke_early_lighthouse_events_v2/events_{candidate}.npz'
        inputs.append(path)
        z = np.load(path)
        t = z['frames'] / fs
        waves = z['waveforms'].astype(np.float64)
        depths = geometry[z['channels'], 1]
        energy = np.sum(waves ** 2, axis=1)
        y = energy @ depths / energy.sum(axis=1)
        reference = np.median(waves[(t >= 970) & (t < 975)], axis=0)
        ref_energy = np.sum(reference ** 2, axis=0)
        anchor = float(ref_energy @ depths / ref_energy.sum())
        anchors[candidate] = anchor
        clouds[candidate] = pd.DataFrame(dict(candidate=candidate, time_s=t, event_energy_centroid_um=y))
        q = windows[(windows.candidate == candidate) & (windows['mode'] == '100spikes') & (windows.status == 'supported')].copy()
        for source, target in [('centroid_relative_um', 'absolute_centroid_um'), ('low_um', 'absolute_low_um'), ('high_um', 'absolute_high_um')]:
            q[target] = q[source] + anchor
        summaries[candidate] = q
        assert np.isfinite(y).all() and np.all((t >= 930) & (t < 1030))
        assert (q.events == 100).all()
    pd.concat(clouds.values()).to_csv(OUT / 'event_centroids.csv', index=False)
    pd.concat(summaries.values()).to_csv(OUT / 'hundred_spike_centroids.csv', index=False)
    te, de = hist['time_edges'], hist['depth_edges']
    for metric, label, stem in [('counts', 'Peak counts', '01_counts_scatter'), ('amplitude_mass', 'Sum |peak amplitude|', '02_amplitude_mass_scatter')]:
        arrays = [np.log1p(hist[f'arm{i}_{metric}']) for i in range(2)]
        vmax = np.quantile(np.concatenate([a.ravel() for a in arrays]), .995)
        fig, axes = plt.subplots(3, 2, figsize=(15, 11), sharex=True, layout='constrained')
        for col, (arm, array) in enumerate(zip(['Original input', 'Compensated input'], arrays)):
            for row in range(3):
                ax = axes[row, col]
                im = ax.imshow(array, origin='lower', aspect='auto', extent=[te[0], te[-1], de[0], de[-1]], cmap='magma', vmin=0, vmax=vmax, rasterized=True)
                selected = candidates if row == 0 else [candidates[row-1]]
                for candidate in selected:
                    p, q = clouds[candidate], summaries[candidate]
                    ax.scatter(p.time_s, p.event_energy_centroid_um, s=3 if row == 0 else 7, c='#56DDE0', alpha=.25, linewidths=0, rasterized=True)
                    ax.errorbar(q.time_s, q.absolute_centroid_um, xerr=np.array([q.time_s-q.start_s, q.end_s-q.time_s]), fmt='D', ms=3.8, mfc='white', mec='#006F80', mew=.7, ecolor='white', elinewidth=.8, zorder=4)
                    ax.vlines(q.time_s, q.absolute_low_um, q.absolute_high_um, colors='white', lw=.8, zorder=4)
                if row == 0:
                    ax.set_ylim(de[0], de[-1])
                    for c in candidates:
                        ax.axhspan(anchors[c]-70, anchors[c]+70, facecolor='none', edgecolor='#56DDE0', lw=.7, ls='--')
                    title = f'{arm} · full probe'
                else:
                    c = selected[0]
                    ax.set_ylim(anchors[c]-70, anchors[c]+70)
                    title = f'{arm} · {c} · {len(clouds[c]):,} accepted spikes'
                for boundary in [960, 970]:
                    ax.axvline(boundary, color='white', ls=':', lw=.8, alpha=.7)
                ax.set(title=title, ylabel='Depth (µm)', xlim=(930, 1030))
                ax.ticklabel_format(axis='x', style='plain', useOffset=False)
        for ax in axes[-1]:
            ax.set_xlabel('Recording time (s)')
        fig.colorbar(im, ax=axes, shrink=.75, label=f'log(1 + bin total) · {label}')
        fig.suptitle(f'930–1,030 s · lighthouse scatter over {label.lower()}\n0.25 s × 10 µm bins · shared original/compensated scale · pooled 99.5th-percentile clipping', fontsize=14)
        handles = [Line2D([], [], marker='o', color='#56DDE0', lw=0, ms=4, label='Individual accepted-event energy centroid'), Line2D([], [], marker='D', color='gray', mfc='white', mec='#006F80', ms=5, label='100-spike median-waveform centroid; bars = time span / conditional 95% bootstrap')]
        axes[0, 0].legend(handles=handles, loc='lower left', fontsize=7, framealpha=.9)
        fig.supxlabel('Identical lighthouse observations on both backgrounds; waveforms use original referenced input. Dotted lines bound template training (960–970 s).\nCentroids are descriptive, not validated physical positions; individual-event and median-waveform estimators can have different biases.', fontsize=9)
        for ext in ['png', 'pdf']:
            fig.savefig(OUT / f'{stem}.{ext}', dpi=180)
        plt.close(fig)
    (OUT / 'manifest.json').write_text(json.dumps(dict(status='complete', interval_s=[930,1030], reference_centroids_um=anchors, source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}, note='Cached-only rendering; no new detections, motion estimates, or raw voltage reads. Original waveform centroids are repeated on both backgrounds. No identity qualification inferred.'), indent=2))
    print(OUT)

if __name__ == '__main__':
    main()
