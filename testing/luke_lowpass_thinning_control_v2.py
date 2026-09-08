"""Conditional cached-peak controls for a selected gentle-screen comparison."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from spikeinterface.core import NumpyRecording
from testing.luke_epoch_corroboration import ROOT, BASE
from testing.luke_dredge_bounded import estimate_bounded


def matched_mask(groups, target, seed, stratified):
    rng = np.random.default_rng(seed)
    keep = np.zeros(len(target), dtype=bool)
    if stratified:
        for group in np.unique(groups):
            ids = np.flatnonzero(groups == group)
            n = int(target[ids].sum())
            if n:
                keep[rng.choice(ids, n, replace=False)] = True
        assert np.array_equal(np.bincount(groups[keep]), np.bincount(groups[target]))
    else:
        keep[rng.choice(len(target), int(target.sum()), replace=False)] = True
    assert keep.sum() == target.sum()
    return keep


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', required=True, choices=['broad_3sigma', 'lowpass_fixed', 'lowpass_adjusted'])
    args = parser.parse_args()
    src = ROOT/'testing/outputs/luke_3sigma_lowpass_screen_v2'
    out = ROOT/'testing/outputs'/('luke_lowpass_thinning_'+args.parent+'_v2')
    out.mkdir(exist_ok=False)
    (out/'fields').mkdir()
    begun = time.monotonic()
    screened = 'broad_screen' if args.parent == 'broad_3sigma' else args.parent+'_screen'
    cfg = json.loads((src/'settings.json').read_text())['estimator']
    manifest = json.loads((BASE/'recording/rescue_recording_manifest.json').read_text())
    fs = manifest['sampling_frequency_hz']
    geo = np.asarray(manifest['channel_locations_um'])
    p = np.load(src/f'{args.parent}_peaks.npy')
    y = np.load(src/f'{args.parent}_locations.npy')
    sp = np.load(src/f'{screened}_peaks.npy')
    keys = p['sample_index']*384+p['channel_index']
    target = np.isin(keys, sp['sample_index']*384+sp['channel_index'])
    assert target.sum() == len(sp)
    assert np.array_equal(p[target], sp)
    groups = (p['sample_index']/fs).astype(int)*40+(geo[p['channel_index'],1]//100).astype(int)
    settings = dict(parent=args.parent, screen=screened, interval_s=[4160,4260], estimator=cfg,
                    controls='Two uniform exact-count samples, one exact 1s x 100um detector-depth sample; fixed seeds14,29.',
                    interpretation='Screen versus thinning of its own parent; no filtering or localization changes.',
                    resume='No automatic or within-stage resume; existing output refused.',
                    source_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in [Path(__file__),src/'settings.json',src/f'{args.parent}_peaks.npy',src/f'{args.parent}_locations.npy',src/f'{screened}_peaks.npy']})
    (out/'settings.json').write_text(json.dumps(settings,indent=2))
    rec=NumpyRecording(np.broadcast_to(np.zeros((1,384),dtype='float32'),(round(100*fs),384)),fs)
    rec.set_channel_locations(geo)
    rows=[]
    for name in [args.parent,screened]:
        z=np.load(src/'fields'/f'{name}.npz')
        np.savez_compressed(out/'fields'/f'{name}.npz',**{k:z[k] for k in z.files})
        count=len(p) if name==args.parent else len(sp)
        rows.append(dict(name=name,peaks=count,retained_fraction=count/len(p)))
    for label,seed,stratified in [('random14',14,False),('random29',29,False),('depth_time14',14,True)]:
        keep=matched_mask(groups,target,seed,stratified)
        np.save(out/f'{label}_mask.npy',keep)
        motion,extra=estimate_bounded(rec,p[keep],y[keep],cfg)
        np.savez_compressed(out/'fields'/f'{label}.npz',time_s=motion.temporal_bins_s[0]+4160,
                            depth_um=motion.spatial_bins_um,displacement_um=motion.displacement[0],
                            D=extra['D'],C=extra['C'],U=extra['U'])
        rows.append(dict(name=label,peaks=int(keep.sum()),retained_fraction=float(keep.mean())))
        print(label,'complete',flush=True)
    pd.DataFrame(rows).to_csv(out/'manifest.csv',index=False)
    from testing import luke_screen_sweep as scoring
    scoring.OUT=out
    scoring.analyze()
    (out/'summary.json').write_text(json.dumps(dict(status='complete',seconds=time.monotonic()-begun,
        selection='Controls are fixed before scores; no production adoption.'),indent=2))


if __name__=='__main__':
    main()
