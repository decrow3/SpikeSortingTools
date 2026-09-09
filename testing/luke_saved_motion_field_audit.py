"""Audit existing KS4 motion fields; CPU-only, never launches a sorter.

Sign is fixed from KS4 4.0.27 preprocessing: output(y) samples input(y-dshift).
SI samples input(y+motion), so physical displacement = -dshift. No fitted sign,
scale, lag, or temporal smoothing is used in the primary comparisons.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def describe(x):
    x = np.asarray(x, dtype=float)
    assert np.isfinite(x).all()
    q = np.quantile(x, [0, .05, .5, .95, 1])
    return dict(zip(['min', 'p05', 'median', 'p95', 'max'], map(float, q)))


def run(group, motion, out):
    out.mkdir(parents=True, exist_ok=False)
    hashes = {}
    def record(p):
        hashes[str(p)] = sha256(p)
        return p
    manifest = json.loads(record(group/'recording/rescue_recording_manifest.json').read_text())
    binary = json.loads(record(group/'recording/binary.json').read_text())
    origin = binary['kwargs']['t_starts'][0]
    fs = manifest['sampling_frequency_hz']
    duration = manifest['num_samples']/fs
    fields, summaries, rows, trajectories = {}, {}, [], {}
    for arm, blocks in [('motion_off',0), ('native_rigid',1), ('native_nonrigid',6)]:
        base = group/f'arms/rescue_12_9_{arm}'
        path = record(base/'sort/sorter_output/ops.npy')
        ops = np.load(path, allow_pickle=True).item()  # trusted local run artifact
        cur_path = record(base/'cur/cur_output/ops.npy')
        cur = np.load(cur_path, allow_pickle=True).item()
        sm = json.loads(record(base/'sort/rescue_sort_manifest.json').read_text())
        log = json.loads(record(base/'sort/spikeinterface_log.json').read_text())
        assert sm['complete'] and sm['recording_request_digest'] == manifest['request_digest']
        assert log['sorter_version'] == '4.0.27'
        assert ops['fs'] == fs and ops['nblocks'] == blocks
        assert ops['Th_universal'] == 12 and ops['Th_learned'] == 9
        assert ops['n_chan_bin'] == 100 and (ops['yc'].min(),ops['yc'].max()) == (1400,2380)
        assert ops['preprocessing'] and np.isfinite(ops['Wrot']).all()
        s = dict(effective_nblocks=blocks, settings_nblocks=ops['settings']['nblocks'],
                 version=log['sorter_version'], fs_hz=fs, channels=100,
                 ops_sha256=hashes[str(path)], curated_ops_sha256=hashes[str(cur_path)])
        summaries[arm] = s
        if blocks == 0:
            assert ops['dshift'] is None and cur['dshift'] is None
            s['motion_disabled_verified'] = True
            continue
        d = np.asarray(ops['dshift'],dtype=float)
        assert np.array_equal(d,cur['dshift']) and np.isfinite(d).all()
        assert d.shape == (int(ops['Nbatches']), 2*blocks-1)
        assert int(ops['Nbatches']) == int(np.ceil(manifest['num_samples']/ops['batch_size']))
        t = (np.arange(len(d))+.5)*ops['batch_size']/fs
        y = np.asarray(ops['yblk'])
        physical = -d
        np.savez_compressed(out/f'{arm}_field.npz',time_s=t,depth_um=y,
                            native_dshift_um=d,physical_displacement_um=physical,
                            channel_positions_um=np.c_[ops['xc'],ops['yc']])
        centered = physical-np.median(physical,axis=0)
        steps = abs(np.diff(d,axis=0))
        # Actual sampling positions, including extrapolation in nonrigid mode.
        from scipy.interpolate import interp1d
        channel_d = (np.broadcast_to(d,(len(d),100)) if blocks==1 else
                     interp1d(y,d,axis=1,fill_value='extrapolate')(ops['yc']))
        sample_y = ops['yc'][None,:]-channel_d
        interior = (ops['yc']>=1600)&(ops['yc']<=2180)
        s.update(shape=list(d.shape), batch_s=ops['batch_size']/fs,
                 depth_um=y.tolist(), physical_displacement_um=describe(physical),
                 centered_displacement_um=describe(centered), adjacent_absolute_step_um=describe(steps),
                 batches_any_centered_abs_gt50=int(np.any(abs(centered)>50,axis=1).sum()),
                 batches_any_centered_abs_gt100=int(np.any(abs(centered)>100,axis=1).sum()),
                 transitions_any_step_gt100=int(np.any(steps>100,axis=1).sum()),
                 interior_batch_channel_sampling_outside_strip_fraction=float(
                     ((sample_y[:,interior]<1400)|(sample_y[:,interior]>2380)).mean()),
                 batches_any_interior_sampling_outside_strip=int(
                     np.any((sample_y[:,interior]<1400)|(sample_y[:,interior]>2380),axis=1).sum()))
        fields[arm]=(t,y,physical)
        trajectories[arm]=np.median(centered,axis=1)
        pd.DataFrame(dict(time_s=t,median_centered_physical_um=trajectories[arm],
                          min_centered_physical_um=centered.min(axis=1),
                          max_centered_physical_um=centered.max(axis=1))).to_csv(out/f'{arm}_trajectory.csv',index=False)
    for estimator in ['dredge-motion','decentralized-motion','ks-motion']:
        tt=np.load(record(motion/estimator/'time_bins.npy')).reshape(-1)-origin
        yy=np.load(record(motion/estimator/'depth_bins.npy')).reshape(-1)
        mm=np.load(record(motion/estimator/'motion.npy'))
        assert mm.shape==(len(tt),len(yy)) and np.isfinite(mm).all()
        assert np.all(np.diff(tt)>0) and np.all(np.diff(yy)>0)
        assert 0<=tt[0]<=2 and tt[-1]>=duration-2
        interpolator=RegularGridInterpolator((tt,yy),mm,bounds_error=True)
        for arm,(t,y,a) in fields.items():
            valid=(t>=tt[0])&(t<=tt[-1])
            reference=interpolator(np.stack(np.meshgrid(t[valid],y,indexing='ij'),axis=-1))
            native=a[valid]-np.median(a[valid],axis=0)
            reference-=np.median(reference,axis=0)
            for j,depth in enumerate(y):
                x,v=native[:,j],reference[:,j]
                rows.append(dict(arm=arm,estimator=estimator,depth_um=float(depth),
                    batches=int(valid.sum()),pearson_r=float(np.corrcoef(x,v)[0,1]),
                    native_rms_um=float(np.sqrt(np.mean(x*x))),
                    reference_rms_um=float(np.sqrt(np.mean(v*v))),
                    difference_rms_um=float(np.sqrt(np.mean((v-x)**2))),
                    reference_p95_p05_um=float(np.diff(np.quantile(v,[.05,.95]))[0])))
            if arm=='native_rigid':
                trajectories[estimator]=(t[valid],reference[:,0])
    frame=pd.DataFrame(rows)
    frame.to_csv(out/'field_agreement.csv',index=False)
    a=trajectories['native_rigid'];b=trajectories['native_nonrigid']
    result=dict(schema_version='luke-saved-motion-field-audit-v1',
        source_group=str(group), source_motion=str(motion),clock_origin_s=origin,duration_s=duration,
        physical_sign_rule='physical_displacement_um = -native_dshift_um; fixed from application code',
        comparison='Linear time/depth interpolation, common support, per-depth median removal; no sign/scale/lag fitting.',
        rigid_nonrigid_median_correlation=float(np.corrcoef(a,b)[0,1]),arms=summaries,
        source_sha256=hashes, versions={k:importlib.metadata.version(k) for k in ['numpy','scipy','pandas','kilosort','spikeinterface']},
        limitations=['Independent fields are estimates from shared detections, not ground truth.',
                     'Different preprocessing/detection and spatial coverage confound estimator comparisons.',
                     'Agreement does not test raw-waveform recovery or validated neuron identities.',
                     'Full-probe native estimation may behave differently from this 100-contact strip.',
                     'Thresholds 50/100 um are descriptive screens, not preregistered promotion gates.'])
    # Plot native and same-depth independent fields without smoothing or clipping.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,1,figsize=(12,7),sharex=True,layout='constrained')
    t=fields['native_rigid'][0]
    axes[0].plot(t/60,a,color='#a33b32',lw=.6,label='Native rigid')
    axes[0].plot(t/60,b,color='#6e59a5',lw=.5,alpha=.65,label='Native nonrigid (median across blocks)')
    for e,color in zip(['dredge-motion','decentralized-motion','ks-motion'],['#007f73','#aa7600','#2265ac']):
        tt,v=trajectories[e]
        axes[0].plot(tt/60,v,lw=.8,color=color,label=e)
        axes[1].plot(tt/60,v,lw=.8,color=color,label=e)
    axes[0].set_title('Saved native motion: full recording, median-centered, physical sign')
    axes[1].set_title('Independent estimates at 1889 µm depth (separate vertical scale)')
    for ax in axes:
        ax.set_ylabel('Displacement (µm)'); ax.grid(alpha=.2);ax.legend(fontsize=8,loc='upper right')
    axes[1].set_xlabel('Minutes from selected recording start')
    fig.savefig(out/'motion_fields.png',dpi=170)
    plt.close(fig)
    (out/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'arms':summaries,'rigid_nonrigid_r':result['rigid_nonrigid_median_correlation']},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--group-root',type=Path,required=True)
    p.add_argument('--motion-root',type=Path,required=True)
    p.add_argument('--output-root',type=Path,required=True)
    args=p.parse_args()
    run(args.group_root,args.motion_root,args.output_root)
