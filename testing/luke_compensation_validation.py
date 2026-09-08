"""Frozen-model validation on a separate, predeclared gentle-recovery interval."""
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt
from testing.luke_epoch_corroboration import ROOT, BASE
from testing.luke_multidepth_anchors_v2 import near
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = ROOT/'testing/outputs/luke_compensation_validation_v1'
SOURCE = ROOT/'testing/outputs'
START, STOP = 4240, 4260

def main():
    OUT.mkdir(exist_ok=False)
    model_path = SOURCE/'luke_common_event_screen_v1/shared_response_model.npz'
    noise_path = SOURCE/'luke_peak_threshold_screen_v1/noise_uv.npy'
    config = json.loads((SOURCE/'luke_compensated_dredge_trial_v1/settings.json').read_text())['estimator']
    settings = dict(interval_s=[START, STOP], selection='Separate later interval with existing multi-depth lighthouse coverage and upward recovery; chosen before computing compensated outputs.', model_fit_interval_s=[4180,4190], model_refit=False, model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(), noise_sha256=hashlib.sha256(noise_path.read_bytes()).hexdigest(), estimator=config, detection=dict(method='locally_exclusive',peak_sign='neg',radius_um=50,detect_threshold=5,noise='Frozen original per-channel MAD from development interval'), localization=dict(method='monopolar_triangulation',radius_um=75), resume='No within-stage checkpoint; interruption requires a new output version. No sort.', scope='Temporal transfer within the same recording and known gentle tracking interval; not an untouched recording or blinded interval selection.')
    (OUT/'settings.json').write_text(json.dumps(settings,indent=2))
    m=json.loads((BASE/'recording/rescue_recording_manifest.json').read_text())
    fs=m['sampling_frequency_hz']; loc=np.asarray(m['channel_locations_um'])
    first=round(START*fs); n=round((STOP-START)*fs); pad=round(.05*fs)
    with (BASE/'recording/traces_cached_seg0.raw').open('rb') as f:
        f.seek((first-pad)*768); buf=f.read((n+2*pad)*768)
    x=np.frombuffer(buf,dtype='<i2').reshape(-1,384).astype('float32')*m['gain_uv_per_count']
    x=sosfiltfilt(butter(3,[300,6000],fs=fs,btype='bandpass',output='sos'),x,axis=0).astype('float32')
    del buf
    # Predict on padded data to avoid zero-prediction boundary artifacts.
    model=np.load(model_path); coeff=model['coefficients']; lags=model['lag_samples']
    ref=np.median(x,axis=1); clean=np.empty((n,384),dtype='float32')
    for lo in range(0,n,10000):
        e=np.arange(lo,min(lo+10000,n)); clean[e]=x[pad+e]-ref[pad+e[:,None]+lags]@coeff
    post=x[pad:pad+n]-ref[pad:pad+n,None]; del x,ref
    noise=np.load(noise_path)
    light=pd.read_csv(SOURCE/'luke_lighthouse_gentle_v1/gentle_events.csv')
    light=light[(light.frame>=first+50)&(light.frame<first+n-50)]
    tracks=pd.read_csv(SOURCE/'luke_lighthouse_gentle_v1/gentle_tracks.csv')
    depths=tracks.groupby('unit_id').depth_um.first()
    templates=np.load(SOURCE/'luke_lighthouse_gentle_v1/templates.npz')
    rows=[]; waves={}; off=np.arange(-30,31)
    for cid,g in light.groupby('unit_id'):
        ch=templates[f'unit_{cid}_channels']
        for half in [0,1]:
            ev=g.frame.to_numpy()-first; ev=ev[(ev>=half*10*fs)&(ev<(half+1)*10*fs)]
            ev=ev[np.linspace(0,len(ev)-1,min(200,len(ev)),dtype=int)]
            assert len(ev)>=10, (cid,half,len(ev))
            a=np.median(post[ev[:,None,None]+off[None,:,None],ch[None,None,:]],axis=0)
            b=np.median(clean[ev[:,None,None]+off[None,:,None],ch[None,None,:]],axis=0)
            cosine=float(np.sum(a*b)/(np.linalg.norm(a)*np.linalg.norm(b)))
            ratio=float(np.max(abs(b))/np.max(abs(a)))
            rows.append(dict(unit_id=int(cid),half=half,events=len(ev),waveform_cosine=cosine,peak_amplitude_ratio=ratio,passes=cosine>=.9 and .8<=ratio<=1.2))
            waves[f'unit_{cid}_half{half}_original']=a; waves[f'unit_{cid}_half{half}_compensated']=b
    checks=pd.DataFrame(rows); checks.to_csv(OUT/'waveform_preservation.csv',index=False)
    np.savez_compressed(OUT/'waveform_checks.npz',**waves)
    print('Waveform preservation:',checks.to_json(orient='records'),flush=True)
    if not checks.passes.all():
        (OUT/'summary.json').write_text(json.dumps(dict(status='preservation_gate_failed',checks=rows),indent=2)); return
    from spikeinterface.core import NumpyRecording
    from spikeinterface.sortingcomponents.peak_detection import detect_peaks
    from spikeinterface.sortingcomponents.peak_localization import localize_peaks
    from spikeinterface.sortingcomponents.motion import estimate_motion
    fields={}; recovery=[]; counts={}; common=[]
    for name,voltage in [('Original',post),('Compensated',clean)]:
        record=NumpyRecording(voltage,fs); record.set_channel_locations(loc)
        peaks=detect_peaks(record,method='locally_exclusive',peak_sign='neg',radius_um=50.,detect_threshold=5.,noise_levels=noise,n_jobs=1,chunk_duration='1s',progress_bar=False)
        # Identical exclusion in both arms; removes filter/detection edge ambiguity.
        peaks=peaks[(peaks['sample_index']>=50)&(peaks['sample_index']<n-50)]
        np.save(OUT/f'{name.lower()}_peaks.npy',peaks); counts[name]=len(peaks)
        print(name,'detections',len(peaks),flush=True)
        pos=localize_peaks(record,peaks,method='monopolar_triangulation',radius_um=75.,n_jobs=4,mp_context='fork',chunk_duration='1s',progress_bar=False)
        assert np.isfinite(pos['y']).all(); np.save(OUT/f'{name.lower()}_locations.npy',pos)
        motion=estimate_motion(record,peaks,pos,**config)
        a=motion.displacement[0]; t=motion.temporal_bins_s[0]+START; dep=motion.spatial_bins_um
        assert np.isfinite(a).all(); fields[name]=(t,dep,a)
        np.savez_compressed(OUT/f'{name.lower()}_motion.npz',time_s=t,depth_um=dep,displacement_um=a)
        for cid,g in light.groupby('unit_id'):
            hits=peaks['sample_index'][abs(loc[peaks['channel_index'],1]-depths[cid])<=60]
            ev=g.frame.to_numpy()-first
            hit=near(ev,hits,.0008*fs)
            for frame,yes in zip(g.frame,hit): recovery.append(dict(unit_id=int(cid),frame=int(frame),arm=name,coincident=bool(yes)))
        print(name,'motion complete',flush=True)
    rec=pd.DataFrame(recovery); rec.to_csv(OUT/'event_recovery.csv',index=False)
    paired=rec.pivot(index=['unit_id','frame'],columns='arm',values='coincident')
    transition=dict(both=int((paired.Original&paired.Compensated).sum()),original_only=int((paired.Original&~paired.Compensated).sum()),compensated_only=int((~paired.Original&paired.Compensated).sum()),neither=int((~paired.Original&~paired.Compensated).sum()))
    comparisons=[]
    for name,(t,dep,a) in fields.items():
        for cid,depth in depths.items():
            g=tracks[tracks.unit_id==cid].set_index('time_s')
            v=np.array([np.interp(depth,dep,w) for w in a])
            step=float(np.median(v[t>=START+10])-np.median(v[t<START+10]))
            observed=float(g.loc[START+15,'median_waveform_centroid_um']-g.loc[START+5,'median_waveform_centroid_um'])
            comparisons.append(dict(arm=name,unit_id=int(cid),depth_um=float(depth),dredge_step_um=step,lighthouse_centroid_step_um=observed))
    comparison=pd.DataFrame(comparisons); comparison.to_csv(OUT/'lighthouse_comparison.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for name,(t,dep,a) in fields.items():
        delta=np.median(a[t>=START+10],axis=0)-np.median(a[t<START+10],axis=0)
        axes[0].plot(delta,dep,label=name,ls='--' if name=='Original' else '-')
    c=comparison[comparison.arm=='Original']
    axes[0].scatter(c.lighthouse_centroid_step_um,c.depth_um,c='k',marker='x',label='Lighthouse centroid')
    for r in c.itertuples(): axes[0].annotate(str(r.unit_id),(r.lighthouse_centroid_step_um,r.depth_um),xytext=(5,2),textcoords='offset points',fontsize=8)
    axes[0].axvline(0,c='gray',lw=.6); axes[0].set(xlabel='Second-half − first-half change (µm)',ylabel='Depth (µm)',ylim=(0,3820),title='Movement across probe depth'); axes[0].legend(fontsize=8)
    for name,(t,dep,a) in fields.items():
        v=np.array([np.interp(2380,dep,w) for w in a]); axes[1].plot(t,v-np.median(v[t<START+10]),label=name,ls='--' if name=='Original' else '-')
    axes[1].set(xlabel='Recording time (s)',ylabel='Relative displacement (µm)',title='DREDGE at 2380 µm'); axes[1].legend()
    fig.suptitle('Frozen compensation model: separate upward-recovery interval, 4240–4260 s\nSame DREDGE settings; lighthouse centroids are descriptive, not calibrated ground truth',fontsize=11)
    for suffix in ['png','pdf']: fig.savefig(OUT/f'01_motion_validation.{suffix}',dpi=160)
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for half in [0,1]:
        c=checks[checks.half==half]; axes[0].scatter(c.waveform_cosine,c.unit_id.astype(str),marker='o' if half==0 else 'x',label=f'Half {half+1}')
    axes[0].set(xlabel='Original / compensated median waveform cosine',ylabel='Lighthouse unit',xlim=(.9,1.002),title='Waveform preservation'); axes[0].legend()
    count=rec.groupby(['unit_id','arm']).coincident.sum().unstack(); yy=np.arange(len(count))
    axes[1].barh(yy-.18,count.Original,height=.36,label='Original'); axes[1].barh(yy+.18,count.Compensated,height=.36,label='Compensated'); axes[1].set_yticks(yy,count.index.astype(str)); axes[1].set(xlabel='Reference events with coincident detections',title='Detection recovery'); axes[1].legend()
    fig.suptitle('Neural checks in the separate 4240–4260 s interval\nSelected lighthouse cohort; coincidence is not detection precision',fontsize=11)
    for suffix in ['png','pdf']: fig.savefig(OUT/f'02_neural_preservation.{suffix}',dpi=160)
    result=dict(status='complete',peak_counts=counts,waveform_checks_pass=int(checks.passes.sum()),waveform_checks_total=len(checks),waveform_cosine_min=float(checks.waveform_cosine.min()),amplitude_ratio_range=[float(checks.peak_amplitude_ratio.min()),float(checks.peak_amplitude_ratio.max())],recovery=rec.groupby('arm').coincident.sum().to_dict(),event_transitions=transition,comparisons=comparisons,scope=settings['scope'])
    (OUT/'summary.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result),flush=True)

if __name__=='__main__': main()
