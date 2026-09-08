"""Calibrate median-waveform centroid from saved matches; no rescore or tracking."""
import json
import time
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from testing.luke_lowpass_waveform_preservation_v2 import reconstruct, digest, SRC, BASE
from testing.luke_shallow_fractional_calibration_v3 import centroid

OUT=SRC/'luke_shallow_template_centroid_audit_v4'
INJECT=SRC/'luke_shallow_fractional_calibration_v3'
TRANSITION=SRC/'luke_shallow_transition_audit_v3'
BOOTSTRAPS=200


def paired_template_displacement(shifted,zero,depths):
    return float(centroid(np.median(shifted,axis=0),depths)-centroid(np.median(zero,axis=0),depths))


def main():
    begun=time.monotonic();OUT.mkdir(exist_ok=False)
    manifestfile=BASE/'recording/rescue_recording_manifest.json'
    modelfile=SRC/'luke_common_event_screen_v1/shared_response_model.npz'
    injectionfile=INJECT/'injection_templates.npz';rowsfile=INJECT/'injected_events.csv'
    realfile=TRANSITION/'observed_templates.npz';galleryfile=TRANSITION/'waveform_gallery.csv'
    m=json.loads(manifestfile.read_text());geo=np.asarray(m['channel_locations_um'])
    z=np.load(injectionfile);channels=z['channels'];foot=abs(geo[channels,1]-620)<=120;depths=geo[channels[foot],1]
    rows=pd.read_csv(rowsfile)
    settings=dict(target=154,measurement='Centroid of median waveform, squared energy on fixed620±120um support; explicitly differs from median of individual-event centroids.',
        real_templates='Saved transition zero-shift medians at4240–4245 and4245–4250s, only these>=10event supported bins.',
        injection_selection='Freeze allv3 decisions/timing. Both shift and pairedzero must have acceptedidentity+uniqueshift AND winninggridshift0. >=10pairs. Report±20asoutsideprimaryfractionalrange evenif enoughpairs.',
        injection_comparison='Centroid of median aligned shifted waveform minus centroid of median alignedzero waveform on exactly same acceptedbackgroundpair subset.',
        bootstrap=dict(resamples=BOOTSTRAPS,unit='paired background waveform; shared resampleindices across shifted andzero',interpretation='Conditional injection-model sampling interval, not biological motion accuracy interval.'),
        matching='No detection, rescore, thresholdchange, DREDGE selection, or realtransitionvoltage reread.',
        limitations=['Linear same-column interpolation is approximate, not established physical displacement truth.',
                    'Quiet-background conditional calibration does not establish transition-time identity or calibrate all firing amplitudes.',
                    'Median-waveform measurement changes the statistic; do not transfer individual-event slope correction to it.',
                    'Late sparse/mixed bins and other depths remain unvalidated.'],
        sha256={str(p):digest(p) for p in [Path(__file__).resolve(),manifestfile,modelfile,injectionfile,rowsfile,realfile,galleryfile,
            Path(__file__).with_name('luke_lowpass_waveform_preservation_v2.py'),Path(__file__).with_name('luke_shallow_fractional_calibration_v3.py')]})
    (OUT/'settings.json').write_text(json.dumps(settings,indent=2))
    real=np.load(realfile);realchannels=real['channels'];realfoot=abs(geo[realchannels,1]-620)<=120
    gallery=pd.read_csv(galleryfile);realrows=[]
    for start in [4240,4245]:
        g=gallery[(gallery.start_s==start)&(gallery.shift_um==0)];assert len(g)==1 and g.iloc[0].events>=10
        c=float(centroid(real[f't{start}_shift0'][:,realfoot],geo[realchannels[realfoot],1]))
        realrows.append(dict(start_s=start,events=int(g.iloc[0].events),median_waveform_centroid_um=c))
    realdata=pd.DataFrame(realrows);realdata['relative_um']=realdata.median_waveform_centroid_um-realdata.median_waveform_centroid_um.iloc[0]
    realdata.to_csv(OUT/'real_median_template_measurements.csv',index=False)
    quiet,unused,first=reconstruct(4100,m,np.load(modelfile));del unused
    unique_frames=rows[['pair_id','background_frame']].drop_duplicates();assert unique_frames.pair_id.is_unique
    background={int(r.pair_id):quiet[int(r.background_frame)-first+np.arange(-33,34)[:,None],channels[None,:]].copy() for r in unique_frames.itertuples()};del quiet
    summaries=[];templates={};bootvalues={};rng=np.random.default_rng(20260908155)
    for (shift,amplitude),g in rows.groupby(['requested_shift_um','gain'],sort=True):
        zero=rows[(rows.requested_shift_um==0)&(rows.gain==amplitude)].set_index('pair_id')
        paired=g.merge(zero[['accepted_identity','unique_shift','best_shift_um','timing_lag']],left_on='pair_id',right_index=True,suffixes=('','_pairedzero'),validate='one_to_one')
        accepted=paired.accepted_identity&paired.unique_shift&paired.accepted_identity_pairedzero&paired.unique_shift_pairedzero
        zero_grid=(paired.best_shift_um==0)&(paired.best_shift_um_pairedzero==0)
        usable=paired[accepted&zero_grid]
        row=dict(requested_shift_um=int(shift),gain=amplitude,injections=len(g),paired_accepted=int(accepted.sum()),paired_zero_grid=len(usable),
                 paired_nonzero_grid=int((accepted&~zero_grid).sum()),primary_range=abs(shift)<=10,
                 status='supported' if len(usable)>=10 and abs(shift)<=10 else 'outside_primary_range' if abs(shift)>10 else 'insufficient_pairs')
        # ±20 remain descriptive outside the primary range, retaining their masks and counts.
        if len(usable)>=10:
            wa=[];wz=[]
            for r in usable.itertuples():
                a=background[int(r.pair_id)].copy();b=a.copy();a[3:64]+=amplitude*z[f'shift_{int(shift)}'];b[3:64]+=amplitude*z['shift_0']
                la=int(r.timing_lag);lb=int(r.timing_lag_pairedzero)
                wa.append(a[3+la:64+la][:,foot]);wz.append(b[3+lb:64+lb][:,foot])
            wa,wz=np.asarray(wa),np.asarray(wz)
            value=paired_template_displacement(wa,wz,depths)
            bs=[]
            for _ in range(BOOTSTRAPS):
                ix=rng.integers(0,len(wa),size=len(wa));bs.append(paired_template_displacement(wa[ix],wz[ix],depths))
            low,high=np.quantile(bs,[.025,.975]);row.update(recovered_template_displacement_um=value,conditional_bootstrap_low_um=float(low),conditional_bootstrap_high_um=float(high))
            key=f'shift{int(shift)}_gain{amplitude}';templates[key+'_shifted']=np.median(wa,axis=0);templates[key+'_zero']=np.median(wz,axis=0);bootvalues[key]=np.asarray(bs)
        summaries.append(row)
    d=pd.DataFrame(summaries);d.to_csv(OUT/'template_sensitivity_by_shift_gain.csv',index=False)
    np.savez_compressed(OUT/'paired_median_templates.npz',channels=channels[foot],**templates)
    np.savez_compressed(OUT/'conditional_bootstrap_displacements.npz',**bootvalues)
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    for amplitude,color in zip([.75,1.,1.25],['#2468a2','#cc7722','#555555']):
        g=d[(d.gain==amplitude)&d.primary_range]
        supported=g[g.status=='supported'];y=supported.recovered_template_displacement_um
        axes[0].plot(supported.requested_shift_um,y,'o-',color=color,label=f'gain{amplitude}')
        axes[0].vlines(supported.requested_shift_um,supported.conditional_bootstrap_low_um,supported.conditional_bootstrap_high_um,color=color,lw=1)
        axes[1].plot(g.requested_shift_um,g.paired_zero_grid,'o-',color=color,label=f'gain{amplitude}')
    axes[0].plot([-10,10],[-10,10],':',color='black',label='Requested interpolation shift')
    axes[0].set(xlabel='Requested injection shift (µm)',ylabel='Paired median-waveform centroid difference (µm)');axes[0].legend(fontsize=8)
    axes[1].set(xlabel='Requested injection shift (µm)',ylabel='Paired accepted zero-grid events (of40)',ylim=(0,42));axes[1].legend(fontsize=8)
    fig.suptitle('Frozen unit154 measurement calibration: centroid of median waveform\nApproximate interpolation model; paired-background95% bootstrap intervals are conditional, not physical accuracy')
    for ext in ['png','pdf']:fig.savefig(OUT/f'01_template_sensitivity.{ext}',dpi=150)
    result=dict(status='complete',seconds=time.monotonic()-begun,real_template_centroid_change_um=float(realdata.relative_um.iloc[-1]),
        interpretation='Measurement sensitivity within a fixed approximate injection model; no DREDGE comparison or physical motion error claim.')
    (OUT/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
