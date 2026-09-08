"""Audit the completed full-probe registration field without running a sorter."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from testing.luke_full_probe_registration import sha,safe
from testing.managed_job import _atomic_json


def stats(values):
    a=np.asarray(values,dtype=float)
    assert np.isfinite(a).all()
    q=np.quantile(a,[0,.05,.5,.95,1])
    return dict(zip(['min','p05','median','p95','max'],map(float,q))) | {'rms':float(np.sqrt(np.mean(a*a))), 'p95_p05':float(q[3]-q[1])}


def profile_correlation(a,b):
    return float(np.corrcoef(a,b)[0,1]) if min(np.std(a),np.std(b))>0 else np.nan


def shift_profile_to_reference(profile, depth_um, target_depth_um, relative_native_dshift_um):
    """Native corrected output(y) samples observed input(y - dshift)."""
    return np.interp(target_depth_um-relative_native_dshift_um, depth_um, profile,
                     left=np.nan, right=np.nan)


def run(registration,strip,motion,out):
    out.mkdir(exist_ok=False)
    reg=json.loads((registration/'registration_summary.json').read_text())
    assert reg['status']=='registration_complete_audit_pending' and reg['full_sort_launched'] is False
    for name,digest in reg['file_sha256'].items():assert sha(registration/name)==digest
    plan=json.loads((registration/'plan.json').read_text())
    full=np.load(registration/'native_rigid_field.npz')
    old=np.load(strip)
    t=full['time_s'];d=full['native_dshift_um'].reshape(-1);physical=-d
    centered=physical-np.median(physical);positions=full['channel_positions_um']
    assert len(t)==5237 and positions.shape==(384,2)
    assert np.array_equal(full['physical_displacement_um'].reshape(-1),physical)
    assert np.all(np.diff(t)>0)
    steps=abs(np.diff(d));oldphysical=old['physical_displacement_um'].reshape(-1)
    assert np.allclose(t,old['time_s'],rtol=0,atol=1e-8)
    oldcentered=oldphysical-np.median(oldphysical)
    oldsteps=abs(np.diff(oldphysical))
    metrics=dict(raw_physical_um=stats(physical),centered_physical_um=stats(centered),
        absolute_adjacent_step_um=stats(steps),
        centered_batches_abs_gt50=int((abs(centered)>50).sum()),
        centered_batches_abs_gt100=int((abs(centered)>100).sum()),
        transitions_gt50=int((steps>50).sum()),transitions_gt100=int((steps>100).sum()),
        strip_centered_um=stats(oldcentered),strip_steps_um=stats(oldsteps),
        strip_transitions_gt100=int((oldsteps>100).sum()),
        full_strip_correlation=float(np.corrcoef(centered,oldcentered)[0,1]),
        batches=len(t),transitions=len(steps),native_reference_depth_um=float(full['depth_um'][0]))
    sampling=positions[:,1][None,:]-d[:,None]
    interior=(positions[:,1]>=200)&(positions[:,1]<=3620)
    for name,keep in [('all',np.ones(384,dtype=bool)),('interior_200um',interior)]:
        outside=(sampling[:,keep]<0)|(sampling[:,keep]>3820)
        metrics[name+'_sampling_outside_fraction']=float(outside.mean())
        metrics[name+'_batches_any_sampling_outside']=int(np.any(outside,axis=1).sum())
    rows=[];traces={};hashes={str(strip):sha(strip)}
    # Full probe rigid is evaluated both at its reference center and at the
    # old strip center. Depthwise sensitivity uses supported contact depths.
    for est in ['dredge-motion','decentralized-motion','ks-motion']:
        for name in ['time_bins.npy','depth_bins.npy','motion.npy']:
            p=motion/est/name;hashes[str(p)]=sha(p)
        tt=np.load(motion/est/'time_bins.npy')-plan['clock_origin_s']
        yy=np.load(motion/est/'depth_bins.npy');mm=np.load(motion/est/'motion.npy')
        assert mm.shape==(len(tt),len(yy)) and np.isfinite(mm).all()
        valid=(t>=tt[0])&(t<=tt[-1]);interp=RegularGridInterpolator((tt,yy),mm)
        depths=sorted(set([float(full['depth_um'][0]),1889.0]+list(np.arange(400,3500,200))))
        for y in depths:
            if y<yy.min() or y>yy.max():continue
            v=interp(np.c_[t[valid],np.full(valid.sum(),y)]);v-=np.median(v)
            a=physical[valid]-np.median(physical[valid])
            b=oldphysical[valid]-np.median(oldphysical[valid])
            rows.append(dict(estimator=est,depth_um=y,batches=int(valid.sum()),
                reference_rms_um=stats(v)['rms'],reference_p95_p05_um=stats(v)['p95_p05'],
                full_pearson=float(np.corrcoef(a,v)[0,1]),strip_pearson=float(np.corrcoef(b,v)[0,1]),
                full_difference_rms_um=float(np.sqrt(np.mean((a-v)**2))),
                strip_difference_rms_um=float(np.sqrt(np.mean((b-v)**2)))))
            if y==float(full['depth_um'][0]):traces[est]=(t[valid],v)
    pd.DataFrame(rows).to_csv(out/'independent_agreement.csv',index=False)
    pd.DataFrame(dict(time_s=t,raw_physical_um=physical,centered_physical_um=centered,
                      strip_centered_physical_um=oldcentered)).to_csv(out/'trajectories.csv',index=False)
    # Native detections permit a direct aggregate raster check, including all
    # batches and the subset of >50 um adjacent transitions separately.
    raster=np.load(registration/'native_detection_raster.npz')
    counts=raster['counts'];y=raster['depth_um'];inter=(y>=400)&(y<=3420)
    metrics['detections_per_batch']=stats(counts.sum(axis=1))
    metrics['zero_detection_batches']=int((counts.sum(axis=1)==0).sum())
    metrics['raster_detections']=int(counts.sum())
    metrics['saved_detections']=reg['detections']
    pd.DataFrame(dict(time_s=t,detections=counts.sum(axis=1))).to_csv(out/'detection_counts.csv',index=False)
    pairs=[]
    for h in range(1,len(t)):
        q=h-1;a=counts[h].astype(float);v=counts[q,inter].astype(float)
        shifted=shift_profile_to_reference(a,y,y[inter],d[h]-d[q])
        valid=np.isfinite(shifted)
        if valid.sum()<100 or a.sum()<100 or v.sum()<100:continue
        pairs.append(dict(high_batch=h,previous_batch=q,time_s=float(t[h]),
            absolute_relative_shift_um=float(abs(d[h]-d[q])),
            raw_profile_correlation=profile_correlation(a[inter][valid],v[valid]),
            shifted_profile_correlation=profile_correlation(shifted[valid],v[valid])))
    frame=pd.DataFrame(pairs).dropna();frame.to_csv(out/'native_raster_adjacent_pairs.csv',index=False)
    raster_metrics={}
    for name,g in [('all',frame),('step_gt50',frame[frame.absolute_relative_shift_um>50])]:
        raster_metrics[name]=dict(pairs=len(g),raw_median=float(g.raw_profile_correlation.median()) if len(g) else None,
            shifted_median=float(g.shifted_profile_correlation.median()) if len(g) else None,
            fraction_shifted_higher=float((g.shifted_profile_correlation>g.raw_profile_correlation).mean()) if len(g) else None)
    metrics['native_raster_profiles']=raster_metrics
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,1,figsize=(13,10),sharex=True,layout='constrained')
    axes[0].plot(t/60,oldcentered,color='#a33b32',lw=.55,label='100-contact strip')
    axes[0].plot(t/60,centered,color='#186c95',lw=.7,label='384-contact full probe')
    axes[0].set_title('Native rigid estimates: same recording, median-centered physical displacement')
    axes[0].set_ylabel('Displacement (µm)');axes[0].legend(loc='upper right')
    axes[1].plot(t/60,centered,color='#186c95',lw=.8,label='Full probe native rigid')
    for est,color in zip(traces,['#008568','#a57900','#8356a6']):
        tt,v=traces[est];axes[1].plot(tt/60,v,color=color,lw=.6,label=est)
    axes[1].set_title('Independent estimates at full-probe reference depth; separate vertical scale')
    axes[1].set_ylabel('Displacement (µm)');axes[1].legend(loc='upper right',fontsize=8)
    axes[2].imshow(np.log1p(counts.T),aspect='auto',origin='lower',extent=[t[0]/60,t[-1]/60,y[0],y[-1]],cmap='Greys')
    axes[2].set_title('Uncorrected native detection density, log(1 + count); no neuron identity inference')
    axes[2].set_ylabel('Depth (µm)');axes[2].set_xlabel('Minutes from recording start')
    fig.savefig(out/'full_probe_motion_audit.png',dpi=170);plt.close(fig)
    # Save detailed views of the five largest adjacent shifts without selecting
    # their sign or fitting the field to their raster.
    indices=np.argsort(steps)[-5:][::-1]+1
    fig,axes=plt.subplots(len(indices),2,figsize=(13,13),layout='constrained')
    for row,h in enumerate(indices):
        lo=max(0,h-15);hi=min(len(t),h+16);window=slice(lo,hi)
        axes[row,0].plot(t[window],centered[window],'.-',color='#186c95')
        axes[row,0].axvline(t[h],color='#a33b32',lw=.7)
        axes[row,0].set_title(f'Batch {h}, adjacent step {steps[h-1]:.1f} µm')
        axes[row,0].set_ylabel('Displacement (µm)')
        axes[row,1].imshow(np.log1p(counts[window].T),aspect='auto',origin='lower',
            extent=[t[lo],t[hi-1],y[0],y[-1]],cmap='Greys')
        axes[row,1].axvline(t[h],color='#a33b32',lw=.7);axes[row,1].set_ylabel('Depth (µm)')
    axes[-1,0].set_xlabel('Recording-relative seconds');axes[-1,1].set_xlabel('Recording-relative seconds')
    fig.savefig(out/'largest_step_raster_support.png',dpi=150);plt.close(fig)
    result=dict(status='computed_requires_scientific_review',metrics=metrics,
        registration_summary_sha256=sha(registration/'registration_summary.json'),source_sha256=hashes,
        methodology='Physical sign fixed at -dshift, batch centers and acquisition origin attested. Linear interpolation on common support; per-series median removal. No sign, lag, or scale optimization.',
        limitations=['Independent estimates share detection inputs and are not ground truth.',
            'Rigid full-probe field is a spatially global estimate; depthwise disagreement can reflect nonrigid motion.',
            'Native detection raster supports profile plausibility, not validated neuron recovery or raw waveform fidelity.',
            'Descriptive excursion thresholds are not a standalone promotion gate.'])
    _atomic_json(out/'summary.json',safe(result));print(json.dumps(safe(metrics),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['registration','strip','motion','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();run(a.registration,a.strip,a.motion,a.output)
