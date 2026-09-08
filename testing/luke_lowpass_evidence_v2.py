"""Independent, cached-field evidence audit. No voltage reads or motion fitting.

Run: python -m testing.luke_lowpass_evidence_v2 [--input DIRECTORY]
D/C/U diagnostics describe selected constraints, not runner-up correlation peaks.
"""
from pathlib import Path
import argparse
import json
import hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'testing/outputs'
WINDOWS = {'overall': (4160, 4260), 'drop': (4180, 4200),
           'recovery': (4190, 4210), 'shallow_excursion': (4245, 4250)}

def stats(values):
    a = np.asarray(values); a = a[np.isfinite(a)]
    return dict(n=len(a), median=float(np.median(a)) if len(a) else None,
                p95=float(np.quantile(a, .95)) if len(a) else None)

def replay(source, out, fields, manifest_path):
    """Recompute two windows' exact bounded correlation curves from cached peaks."""
    import torch
    from spikeinterface.core import NumpyRecording
    from spikeinterface.sortingcomponents.motion.dredge import make_2d_motion_histogram, get_window_domains, normxcorr1d
    from spikeinterface.sortingcomponents.motion.motion_utils import get_spatial_windows
    torch.set_num_threads(1)
    m=json.loads(manifest_path.read_text()); fs=m['sampling_frequency_hz']
    geo=np.asarray(m['channel_locations_um']); cfg=json.loads((source/'settings.json').read_text())['estimator']
    rec=NumpyRecording(np.broadcast_to(np.zeros((1,len(geo)),dtype='float32'),(round(100*fs),len(geo))),fs)
    rec.set_channel_locations(geo); rows=[]; checks=[]
    for name,z in fields.items():
        if name=='flat_zero_control': continue
        peaks=np.load(source/f'{name}_peaks.npy',mmap_mode='r'); positions=np.load(source/f'{name}_locations.npy',mmap_mode='r')
        h,te,se=make_2d_motion_histogram(rec,peaks,positions,weight_with_amplitude=True,avg_in_bin=False,direction=cfg['direction'],bin_s=cfg['bin_s'],bin_um=cfg['bin_um'],hist_margin_um=0.,spatial_bin_edges=None,depth_smooth_um=cfg['histogram_depth_smooth_um'],time_smooth_s=cfg['histogram_time_smooth_s'])
        raster=h.T; centers=(se[1:]+se[:-1])/2
        windows, wc=get_spatial_windows(geo[:,1],centers,**{k:cfg[k] for k in ['rigid','win_shape','win_step_um','win_scale_um','win_margin_um']},zero_threshold=1e-5)
        assert np.allclose(wc,z['depth_um']); domains=get_window_domains(windows); t=z['time_s']
        for region,depth in [('shallow',410),('central',2210)]:
            b=int(np.argmin(abs(wc-depth))); sl=domains[b]; lo=max(0,sl.start-80); hi=min(len(raster),sl.stop+80)
            pad=max(lo-(sl.start-80),sl.stop+80-hi);lags=-np.arange(-(pad+sl.start-lo),pad+hi-sl.stop+1)
            curve=normxcorr1d(torch.tensor(raster[sl].T,dtype=torch.float32),torch.tensor(raster[lo:hi].T,dtype=torch.float32),weights=torch.tensor(windows[b,sl],dtype=torch.float32),padding=pad,normalized=True,centered=True).numpy()
            keep=abs(lags)<=80; curve=curve[:,:,keep]; lags=lags[keep]
            actual_d=lags[curve.argmax(axis=2)].T
            saved_indices=np.argmin(abs(lags[None,None,:]-z['D'][b].T[:,:,None]),axis=2)
            saved_c=np.take_along_axis(curve,saved_indices[:,:,None],axis=2)[:,:,0]
            deficit=curve.max(axis=2)-saved_c
            assert np.array_equal(actual_d,z['D'][b]), f'Winning lag mismatch; saved-lag deficit {np.max(deficit)}'
            assert np.allclose(curve.max(axis=2).T,z['C'][b],atol=1e-5)
            check=dict(name=name,region=region,different_winning_lags=int((actual_d!=z['D'][b]).sum()),max_saved_lag_correlation_deficit=float(deficit.max()),max_c_difference=float(abs(curve.max(axis=2).T-z['C'][b]).max()),pairs_beyond_horizon=int((abs(t[:,None]-t[None,:])>cfg['time_horizon_s']).sum()))
            print('Replay validation',json.dumps(check),flush=True)
            checks.append(check)
            best=curve.argmax(axis=2); bd=lags[best]; bc=curve.max(axis=2)
            alternative=np.where(abs(lags[None,None,:]-bd[:,:,None])>=15,curve,-np.inf)
            alt=alternative.argmax(axis=2); ac=alternative.max(axis=2)
            np.savez_compressed(out/f'correlation_{name}_{region}.npz',correlation=curve,lags_um=lags,time_s=t,depth_um=wc[b],best_lag_um=bd.T,alternative_lag_um=lags[alt].T,margin=(bc-ac).T)
            fig,axes=plt.subplots(1,4,figsize=(16,4),layout='constrained')
            replay_windows=list(WINDOWS.items())[1:]+[('central_cross_epoch',(4160,4240))]
            for ax,(window,(start,stop)) in zip(axes,replay_windows):
                ix=np.flatnonzero((t>=start)&(t<stop)); mask=np.zeros(bd.shape,bool); mask[np.ix_(ix,ix)]=True
                if window=='central_cross_epoch':
                    mask &= ((t>=4160)&(t<4220))[:,None] & ((t>=4225)&(t<4240))[None,:]
                mask &= (z['U'][b].T>0)&~np.eye(len(t),dtype=bool)
                yy,xx=np.where(mask)
                for j,i in zip(yy,xx):
                    rows.append(dict(name=name,region=region,depth_um=float(wc[b]),window=window,time_i_s=t[i],time_j_s=t[j],best_lag_um=int(bd[j,i]),alternative_lag_um=int(lags[alt[j,i]]),best_c=float(bc[j,i]),alternative_c=float(ac[j,i]),margin=float(bc[j,i]-ac[j,i]),weight=float(z['U'][b,i,j])))
                if len(xx):
                    # Representative largest selected shift; no lighthouse-score selection.
                    q=int(np.argmax(abs(bd[yy,xx])));j,i=yy[q],xx[q];order=np.argsort(lags)
                    ax.plot(lags[order],curve[j,i,order]);ax.axvline(bd[j,i],color='k',ls=':');ax.axvline(lags[alt[j,i]],color='r',ls=':')
                    ax.set(title=f'{window}: {t[i]:.1f}/{t[j]:.1f}s',xlabel='Lag (µm)',ylabel='Weighted correlation')
            fig.suptitle(f'{name}, {wc[b]:.0f} µm; bounded correlation replay verified')
            fig.savefig(out/f'alternatives_{name}_{region}.png',dpi=120);plt.close(fig)
    pd.DataFrame(rows).to_csv(out/'alignment_alternatives.csv',index=False)
    pd.DataFrame(checks).to_csv(out/'correlation_replay_checks.csv',index=False)


def audit(source, replay_curves=False):
    out = source / 'independent_evidence'; out.mkdir(exist_ok=True)
    tracks = pd.read_csv(SRC / 'luke_lighthouse_gentle_v1/gentle_tracks.csv')
    events = pd.read_csv(SRC / 'luke_lighthouse_gentle_v1/gentle_events.csv')
    # Original acquisition frame numbers are at 30 kHz; verify against saved manifest.
    manifest_path = Path('/mnt/NPX/Luke/20250804/rescue_pipeline_results_Luke0804_V2V1_g0_imec0/recording/rescue_recording_manifest.json')
    fs = json.loads(manifest_path.read_text())['sampling_frequency_hz']
    events['actual_s'] = events.frame / fs
    arms = list(pd.read_csv(source / 'manifest.csv').name)
    rows, excursions, constraints, provenance = [], [], [], {}
    fields = {}
    for name in arms:
        path = source / 'fields' / (name + '.npz')
        provenance[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path) as z: fields[name] = {k: z[k] for k in z.files}
    ref = fields[arms[0]]
    fields['flat_zero_control'] = dict(time_s=ref['time_s'], depth_um=ref['depth_um'],
                                      displacement_um=np.zeros_like(ref['displacement_um']))
    for name, field in fields.items():
        t, dep, motion = field['time_s'], field['depth_um'], field['displacement_um']
        assert motion.shape == (len(t), len(dep)) and np.isfinite(motion).all()
        for unit, g in tracks.groupby('unit_id'):
            g = g.sort_values('time_s'); depth = float(g.depth_um.iloc[0])
            v = np.array([np.interp(depth, dep, row) for row in motion])
            ev = events.loc[events.unit_id == unit, 'actual_s'].to_numpy()
            initial = ev[(ev >= 4160) & (ev < 4170)]
            q0 = g[g.time_s == 4165].iloc[0]
            baseline_ok = len(initial) >= 10 and q0.accepted_events >= 10
            base = np.median(np.interp(initial, t, v)) if baseline_ok else np.nan
            for q in g.itertuples():
                times = ev[(ev >= q.time_s-5) & (ev < q.time_s+5)]
                pred = np.median(np.interp(times, t, v))-base if len(times) else np.nan
                target = q.median_waveform_centroid_um-q0.median_waveform_centroid_um
                supported = baseline_ok and len(times) >= 10 and q.accepted_events >= 10
                rows.append(dict(name=name, unit_id=unit, depth_um=depth,
                    depth_band=f'{int(depth//960)*960}-{int(depth//960+1)*960}',
                    time_s=q.time_s, accepted_events=q.accepted_events, sampled_events=len(times),
                    supported=supported, predicted_um=pred, centroid_um=target,
                    abs_difference_um=abs(pred-target) if supported else np.nan))
            for window, (lo, hi) in WINDOWS.items():
                ids=(t >= lo)&(t < hi); z=v[ids]-base
                e=ev[(ev >= lo)&(ev < hi)]
                excursions.append(dict(name=name, unit_id=unit, depth_um=depth, window=window,
                    event_count=len(e), field_range_um=float(np.ptp(z)) if len(z) else np.nan,
                    field_max_abs_um=float(np.max(abs(z))) if len(z) else np.nan,
                    max_step_um=float(np.max(abs(np.diff(z)))) if len(z)>1 else np.nan,
                    interpretation='Field diagnostic; 10s lighthouse summaries cannot verify brief excursions'))
        if not all(k in field for k in ['D','C','U']): continue
        D,C,U=field['D'],field['C'],field['U']
        assert D.shape == C.shape == U.shape == (len(dep),len(t),len(t))
        assert np.isfinite(D).all() and np.max(abs(D)) <= 80
        for region, target_depth in [('shallow',410),('central',2210)]:
            b=int(np.argmin(abs(dep-target_depth)))
            fig, axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
            for ax, key, values in zip(axes,['D (µm)','C','U'],[D[b],C[b],U[b]]):
                im=ax.imshow(values,origin='lower',extent=[t[0],t[-1],t[0],t[-1]],aspect='auto'); fig.colorbar(im,ax=ax);ax.set(title=key,xlabel='Time (s)',ylabel='Time (s)')
            fig.suptitle(f'{name}: {region}, {dep[b]:.0f} µm; selected constraints')
            fig.savefig(out/f'constraints_{name}_{region}.png',dpi=120);plt.close(fig)
            for window,(lo,hi) in {**WINDOWS, 'central_cross_epoch':(4160,4240)}.items():
                ix=np.flatnonzero((t>=lo)&(t<hi)); d=D[b][np.ix_(ix,ix)]; c=C[b][np.ix_(ix,ix)]; u=U[b][np.ix_(ix,ix)]
                edge=~np.eye(len(ix),dtype=bool)
                if window=='central_cross_epoch':
                    edge &= ((t[ix]>=4225)&(t[ix]<4240))[:,None] & ((t[ix]>=4160)&(t[ix]<4220))[None,:]
                good=edge & (u>0)
                cyc=[]
                # Enumerate each temporal triple once; require all three solver edges.
                for i in range(len(ix) if window!='central_cross_epoch' else 0):
                    for j in range(i+1,len(ix)):
                        kk=np.arange(j+1,len(ix)); keep=(u[i,j]>0)&(u[j,kk]>0)&(u[i,kk]>0)
                        cyc.extend(abs(d[i,j]+d[j,kk[keep]]-d[i,kk[keep]]))
                w=np.maximum(u[good],0); dv=d[good]
                constraints.append(dict(name=name,region=region,depth_um=float(dep[b]),window=window,
                    active_edges=int(good.sum()),edge_fraction=float(good.sum()/max(1,edge.sum())),
                    median_c=float(np.median(c[good])) if good.any() else None,
                    selected_large_shift_weight_fraction=float(w[abs(dv)>=50].sum()/w.sum()) if w.sum() else None,
                    bound_hit_weight_fraction=float(w[abs(dv)>=79].sum()/w.sum()) if w.sum() else None,
                    antisymmetry_median_um=stats(abs(d+d.T)[good])['median'],
                    cycle_triples=len(cyc),cycle_median_um=stats(cyc)['median'],cycle_p95_um=stats(cyc)['p95']))
    d=pd.DataFrame(rows);d.to_csv(out/'event_matched_predictions.csv',index=False)
    valid=d[d.supported & (d.time_s>4165)]
    per=valid.groupby(['name','unit_id','depth_um','depth_band'],as_index=False).agg(overall_mae_um=('abs_difference_um','mean'),supported_bins=('time_s','size'))
    movement=[]
    for (name,unit),g in d.groupby(['name','unit_id']):
        g=g.set_index('time_s')
        for label,lo,hi in [('drop',4185,4195),('recovery',4195,4205)]:
            a,b=g.loc[lo],g.loc[hi];ok=bool(a.supported and b.supported)
            movement.append(dict(name=name,unit_id=unit,transition=label,supported=ok,
                predicted_change_um=b.predicted_um-a.predicted_um,centroid_change_um=b.centroid_um-a.centroid_um,
                abs_difference_um=abs((b.predicted_um-a.predicted_um)-(b.centroid_um-a.centroid_um)) if ok else np.nan))
    per.to_csv(out/'per_unit.csv',index=False)
    per.groupby(['name','depth_band'],as_index=False).agg(equal_cell_mae_um=('overall_mae_um','mean'),units=('unit_id','size')).to_csv(out/'per_depth.csv',index=False)
    valid.groupby(['name','time_s'],as_index=False).agg(equal_cell_mae_um=('abs_difference_um','mean'),units=('unit_id','size')).to_csv(out/'per_time.csv',index=False)
    pd.DataFrame(movement).to_csv(out/'transitions.csv',index=False)
    pd.DataFrame(excursions).to_csv(out/'excursions.csv',index=False)
    pd.DataFrame(constraints).to_csv(out/'constraint_consistency.csv',index=False)
    fig,axes=plt.subplots(3,3,figsize=(16,11),sharex=True,layout='constrained')
    for ax,(unit,g) in zip(axes.flat,tracks.groupby('unit_id')):
        for name in arms:
            q=d[(d.unit_id==unit)&(d.name==name)];ax.plot(q.time_s,q.predicted_um,label=name,alpha=.8)
        q=d[(d.unit_id==unit)&(d.name=='flat_zero_control')&d.supported];ax.plot(q.time_s,q.centroid_um,'ko',ms=4,label='Provisional lighthouse');ax.axvspan(4245,4250,color='gray',alpha=.15)
        ax.set(title=f'Unit {unit}: {g.depth_um.iloc[0]:.0f} µm',xlabel='Time (s)',ylabel='Relative displacement (µm)')
    axes.flat[0].legend(fontsize=6);fig.suptitle('All arms, accepted-event sampling; agreement is not ground-truth accuracy')
    fig.savefig(out/'all_arm_lighthouses.png',dpi=140);plt.close(fig)
    summary=dict(status='complete',source_sha256=provenance,arms=arms,
        limitations=['Existing fixed-template identities remain provisional.',
          '10s waveform summaries cannot establish correctness of 4245–4250s excursions.',
          'D/C/U retain winning alignments only; no runner-up margin can be inferred.',
          'Cycle consistency and low motion are diagnostics, not accuracy criteria.',
          'Interpolation clamps beyond field bin centers, as in original scoring.'],
        equal_cell_scores=per.groupby('name').overall_mae_um.mean().to_dict())
    if replay_curves:
        replay(source,out,fields,manifest_path)
        summary['exact_bounded_correlation_replay']='Verified exact winning D and C within1e-5 for both windows in every arm; replay uses one torch thread to match original; alternatives separated by at least15 µm'
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,default=SRC/'luke_3sigma_lowpass_screen_v2');p.add_argument('--replay-curves',action='store_true');args=p.parse_args();audit(args.input,args.replay_curves)
