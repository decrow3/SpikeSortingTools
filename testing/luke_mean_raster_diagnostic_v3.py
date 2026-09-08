"""Conditional amplitude-mean versus amplitude-sum AP raster diagnostic."""
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from spikeinterface.core import NumpyRecording
from spikeinterface.sortingcomponents.motion import estimate_motion
from spikeinterface.sortingcomponents.motion.dredge import (
    make_2d_motion_histogram, get_window_domains, normxcorr1d, dredge_ap,
)
from testing.luke_epoch_corroboration import ROOT, BASE

SRC = ROOT/'testing/outputs/luke_3sigma_lowpass_screen_v2'
OUT = ROOT/'testing/outputs/luke_mean_raster_diagnostic_v3'


def estimate(record, peaks, locations, config, average):
    cfg = dict(config, avg_in_bin=average, extra_outputs=True)
    _, old = estimate_motion(record, peaks, locations, **cfg)
    h, te, se = make_2d_motion_histogram(record, peaks, locations,
        weight_with_amplitude=True, avg_in_bin=average, direction=cfg['direction'],
        bin_s=cfg['bin_s'], bin_um=cfg['bin_um'], hist_margin_um=0.,
        spatial_bin_edges=None, depth_smooth_um=cfg['histogram_depth_smooth_um'],
        time_smooth_s=cfg['histogram_time_smooth_s'])
    assert cfg['bin_um'] == 1 and cfg['max_disp_um'] == 80
    raster = h.T
    windows = old['windows']
    D, C = np.empty_like(old['D']), np.empty_like(old['C'])
    for b, sl in enumerate(get_window_domains(windows)):
        lo, hi = max(0, sl.start-80), min(len(raster), sl.stop+80)
        pad = max(lo-(sl.start-80), sl.stop+80-hi)
        lags = -np.arange(-(pad+sl.start-lo), pad+hi-sl.stop+1)
        curve = normxcorr1d(torch.tensor(raster[sl].T, dtype=torch.float32),
            torch.tensor(raster[lo:hi].T, dtype=torch.float32),
            weights=torch.tensor(windows[b,sl], dtype=torch.float32),
            padding=pad, normalized=True, centered=True).numpy()
        assert np.array_equal(lags[curve.argmax(axis=2)].T, old['D'][b])
        assert np.allclose(curve.max(axis=2).T, old['C'][b], atol=1e-5)
        keep = abs(lags) <= 80
        v = curve[:,:,keep]
        D[b], C[b] = lags[keep][v.argmax(axis=2)].T, v.max(axis=2).T
    for key in ['method', 'verbose']:
        cfg.pop(key, None)
    motion, extra = dredge_ap(record, peaks, locations,
        precomputed_D_C_maxdisp=(D,C,80.), **cfg)
    assert np.max(abs(D)) <= 80 and np.isfinite(motion.displacement[0]).all()
    return motion, dict(D=D,C=C,U=extra['U'],raster=raster,
        time_edges_s=te,depth_edges_um=se,windows=windows)


def main():
    torch.set_num_threads(1)
    begun = time.monotonic()
    OUT.mkdir(exist_ok=False)
    (OUT/'fields').mkdir()
    source_settings = json.loads((SRC/'settings.json').read_text())
    cfg = source_settings['estimator']
    m = json.loads((BASE/'recording/rescue_recording_manifest.json').read_text())
    fs = m['sampling_frequency_hz']
    p = np.load(SRC/'broad_3sigma_peaks.npy')
    y = np.load(SRC/'broad_3sigma_locations.npy')
    record = NumpyRecording(np.broadcast_to(np.zeros((1,384),dtype='float32'),(round(100*fs),384)),fs)
    record.set_channel_locations(np.asarray(m['channel_locations_um']))
    settings = dict(interval_s=[4160,4260],estimator=cfg,
        comparison='Only avg_in_bin changes: sum versus mean absolute AP amplitude per occupied1um x1s bin, before unchanged smoothing. Empty bins follow installed SI behavior.',
        hypothesis='Depth-dependent population rate/composition changes can perturb spatial profiles. Uniform scaling already cancels in normalized correlation; mean occupancy is not guaranteed to help.',
        consistency='The same representation is used for initial DREDGE, strict correlation replay and solver-weight construction; actual rasters saved.',
        limits='Exploratory100s diagnostic, no voltage changes or new detections. Sparse occupied bins may receive disproportionate influence. No lighthouse-guided parameter search.',
        resume='No automatic or within-stage resume; existing output refused.',
        sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in
          [Path(__file__),SRC/'settings.json',SRC/'broad_3sigma_peaks.npy',SRC/'broad_3sigma_locations.npy',SRC/'fields/broad_3sigma.npz']})
    (OUT/'settings.json').write_text(json.dumps(settings,indent=2))
    rows=[]
    for name, average in [('amplitude_sum',False),('amplitude_mean',True)]:
        tick=time.monotonic()
        motion,extra=estimate(record,p,y,cfg,average)
        if not average:
            saved=np.load(SRC/'fields/broad_3sigma.npz')
            assert np.array_equal(extra['D'],saved['D'])
            assert np.allclose(extra['C'],saved['C'],atol=1e-5)
            assert np.allclose(extra['U'],saved['U'],atol=1e-5)
            assert np.allclose(motion.displacement[0],saved['displacement_um'],atol=1e-5)
        np.savez_compressed(OUT/'fields'/f'{name}.npz',time_s=motion.temporal_bins_s[0]+4160,
            depth_um=motion.spatial_bins_um,displacement_um=motion.displacement[0],
            D=extra['D'],C=extra['C'],U=extra['U'])
        np.savez_compressed(OUT/f'{name}_raster.npz',**{k:extra[k] for k in ['raster','time_edges_s','depth_edges_um','windows']})
        rows.append(dict(name=name,peaks=len(p),retained_fraction=1.,avg_in_bin=average,seconds=time.monotonic()-tick))
        print(name,'complete',flush=True)
    pd.DataFrame(rows).to_csv(OUT/'manifest.csv',index=False)
    from testing import luke_screen_sweep as scoring
    scoring.OUT=OUT
    scoring.analyze()
    import matplotlib.pyplot as plt
    fig=plt.gcf()
    fig.suptitle('Fixed AP events: amplitude sum versus mean per occupied bin\nOnly the actual raster aggregation changes; provisional lighthouse disagreement is not ground-truth error')
    for ext in ['png','pdf']:
        fig.savefig(OUT/f'01_summary.{ext}',dpi=140)
    (OUT/'summary.json').write_text(json.dumps(dict(status='complete',seconds=time.monotonic()-begun,
        baseline_replay_verified=True,selection='No automatic adoption from aggregate score'),indent=2))


if __name__=='__main__':
    main()
