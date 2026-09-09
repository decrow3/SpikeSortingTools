"""Extend two frozen early provisional lighthouses; exact970–990 accepted-frame reuse."""
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import butter,sosfiltfilt
from spikeinterface.core import NumpyRecording
from spikeinterface.sortingcomponents.peak_detection import detect_peaks
from testing.luke_epoch_corroboration import ROOT,BASE
from testing.luke_multidepth_anchors_v2 import scores

SRC=ROOT/'testing/outputs'
OUT=SRC/'luke_early_lighthouse_events_v2'
PRE=SRC/'luke_transfer_template_holdout_v1'
TARGETS=['s960_c293_pos','s960_c338_pos']
STARTS=[970,930,950,990,1010]


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    begun=time.monotonic();OUT.mkdir(exist_ok=False)
    manifestfile=BASE/'recording/rescue_recording_manifest.json'
    noisefile=SRC/'luke_peak_threshold_screen_v1/noise_uv.npy'
    templatesfile=SRC/'luke_candidate_footprint_audit_v1/full_probe_waveforms.npz'
    candidatesfile=SRC/'luke_independent_transfer_review_v1/candidate_screen.csv'
    m=json.loads(manifestfile.read_text());fs=m['sampling_frequency_hz'];geo=np.asarray(m['channel_locations_um'])
    noise=np.load(noisefile);z=np.load(templatesfile);candidates=pd.read_csv(candidatesfile);templates={}
    for r in candidates.itertuples():
        channels=np.flatnonzero(abs(geo[:,1]-r.depth_um)<=60);v=z[r.candidate+'_h0'][15:76,channels]
        weights=v*v/(v*v+(2*noise[channels])**2);offset=int(abs(v[:,np.flatnonzero(channels==r.channel)[0]]).argmax())-30
        templates[r.candidate]=(channels,v,weights,offset,float(r.depth_um),int(r.start_s))
    competitors={target:[key for key,value in templates.items() if key!=target and value[5]==templates[target][5] and abs(value[4]-templates[target][4])<=120] for target in TARGETS}
    inputs=[Path(__file__).resolve(),ROOT/'testing/luke_transfer_template_holdout.py',ROOT/'testing/luke_multidepth_anchors_v2.py',manifestfile,noisefile,templatesfile,candidatesfile,PRE/'matched_events.csv',PRE/'tracks.csv',PRE/'waveforms_events.npz',PRE/'settings.json']
    settings=dict(interval_s=[930,1030],targets=TARGETS,template_training_interval_s=[960,970],training_overlap='960–970s is inside requested window; per-event flag saved; it is not independent validation.',
        match='Exactly frozen transfer_template_holdout: firsthalf61sample localtemplates, energy/noiseweightedcosine>=.9,gain.4–2.5,timing±3samples, best available sameepoch competitor margin>=.03, pertarget1msNMS.',
        detection='Otherfour20schunks:both-sign5sigma original frozennoise,50um locallyexclusive,n_jobs1; no localization.970–990 exact acceptedframes/scores reused.',
        preprocessing='Original300–6000Hz thirdorderforward/backward bandpass and globalmedianreference,50mspadding; NOT shared-response-compensated voltage.',
        competitors=competitors,waveform_format='s{start}_waveforms.npz: {candidate}_frames, _waveforms_uv[event,61,channel], _channels, _scores, _gains, _competitor_scores. Accepted CSV also stores training_overlap/cached_frame_reuse.',
        resume='Fresh output required; no automatic resume or withinstage checkpoint. Preserve partial evidence on failure.',
        limitations=['Two existing provisional identities only; no new identity qualification or threshold tuning.',
                    'Fixed support can reject a moved cell; sparse competitor bank can confuse identities.',
                    'Fine-time binning or100-spike windows cannot overcome absentidentity evidence; centroids are not calibrated displacement.',
                    'No DREDGE input, localization, sorting, or full-session scan.'],
        sources_sha256={str(p):sha(p) for p in inputs})
    (OUT/'settings.json').write_text(json.dumps(settings,indent=2))
    original=pd.read_csv(PRE/'matched_events.csv');oldtracks=pd.read_csv(PRE/'tracks.csv');savedold=np.load(PRE/'waveforms_events.npz')
    rows=[];regression=[];timings=[];raw=BASE/'recording/traces_cached_seg0.raw';before=raw.stat()
    for start in STARTS:
        t=time.monotonic();first=round(start*fs);n=round(20*fs);pad=round(.05*fs)
        with raw.open('rb') as f:f.seek((first-pad)*768);buf=f.read((n+2*pad)*768)
        assert len(buf)==(n+2*pad)*768
        x=np.frombuffer(buf,dtype='<i2').reshape(-1,384).astype('float32')*m['gain_uv_per_count'];del buf
        x=sosfiltfilt(butter(3,[300,6000],fs=fs,btype='bandpass',output='sos'),x,axis=0).astype('float32');x-=np.median(x,axis=1,keepdims=True);x=x[pad:pad+n].copy()
        if start!=970:
            record=NumpyRecording(x,fs);record.set_channel_locations(geo)
            p=detect_peaks(record,method='locally_exclusive',peak_sign='both',radius_um=50.,detect_threshold=5.,noise_levels=noise,n_jobs=1,chunk_duration='1s',progress_bar=False)
            p=p[(p['sample_index']>50)&(p['sample_index']<n-50)];np.save(OUT/f's{start}_peaks.npy',p);del record
        saved={}
        for target in TARGETS:
            channels,v,weights,offset,depth,training=templates[target]
            if start==970:
                g=original[(original.candidate==target)&(original.start_s==970)].sort_values('frame').copy()
                frames=g.frame.to_numpy(dtype='int64')-first
                assert np.array_equal(frames+first,savedold[target+'_holdout_frames'])
                score_values=g.score.to_numpy();gains=g.gain.to_numpy();rivals=g.competitor_score.to_numpy()
            else:
                pk=p[abs(geo[p['channel_index'],1]-depth)<=80];ev=pk['sample_index']-offset
                result=scores(x,ev,channels,v,weights);bestother=np.full(len(ev),-np.inf)
                for other in competitors[target]:
                    cc,vv,ww,oo,_,_=templates[other];r=scores(x,pk['sample_index']-oo,cc,vv,ww);valid=(r[:,1]>=.4)&(r[:,1]<=2.5);bestother=np.maximum(bestother,np.where(valid,r[:,0],-np.inf))
                accepted=(result[:,0]>=.9)&(result[:,1]>=.4)&(result[:,1]<=2.5)&(result[:,0]-bestother>=.03)
                selected=np.flatnonzero(accepted);keep=[]
                for j in selected[np.argsort(-result[selected,0])]:
                    if not keep or np.min(abs(result[keep,2]-result[j,2]))>.001*fs:keep.append(j)
                keep=np.array(sorted(keep,key=lambda j:result[j,2]),dtype=int)
                frames=result[keep,2].astype(int);score_values=result[keep,0];gains=result[keep,1];rivals=bestother[keep]
            waveforms=x[frames[:,None,None]+np.arange(-30,31)[None,:,None],channels[None,None,:]]
            saved.update({target+'_frames':frames+first,target+'_waveforms_uv':waveforms,target+'_channels':channels,target+'_scores':score_values,target+'_gains':gains,target+'_competitor_scores':rivals})
            for frame,score_value,gain,rival in zip(frames+first,score_values,gains,rivals):
                rows.append(dict(candidate=target,start_s=start,frame=int(frame),time_s=float(frame/fs),depth_um=depth,score=float(score_value),gain=float(gain),competitor_score=float(rival) if np.isfinite(rival) else np.nan,training_overlap=bool(960<=frame/fs<970),cached_frame_reuse=start==970))
            if start==970:
                for b in range(4):
                    use=(frames>=b*5*fs)&(frames<(b+1)*5*fs);old=oldtracks[(oldtracks.candidate==target)&(oldtracks.time_s==start+b*5+2.5)].iloc[0]
                    assert int(use.sum())==int(old.events)
                    centroid=np.nan;maxerror=np.nan
                    if use.sum()>=10:
                        wave=np.median(waveforms[use],axis=0);energy=np.sum(wave*wave,axis=0);centroid=float(energy@geo[channels,1]/energy.sum())
                        assert np.isclose(centroid,old.centroid_um,atol=1e-8,rtol=0)
                        maxerror=float(np.max(abs(wave-savedold[target+f'_bin{b}_waveform'])))
                        assert maxerror==0.,'Cached median waveform mismatch'
                    regression.append(dict(candidate=target,time_s=start+b*5+2.5,events=int(use.sum()),centroid_um=centroid,old_centroid_um=float(old.centroid_um),median_waveform_max_error_uv=maxerror))
            print(start,target,'accepted',len(frames),flush=True)
        np.savez_compressed(OUT/f's{start}_waveforms.npz',**saved)
        pd.DataFrame(rows).sort_values(['candidate','frame']).to_csv(OUT/'matched_events.csv',index=False)
        timings.append(dict(start_s=start,seconds=time.monotonic()-t));pd.DataFrame(timings).to_csv(OUT/'timings.csv',index=False)
        if regression:pd.DataFrame(regression).to_csv(OUT/'cached970_regression.csv',index=False)
        del x
    for target in TARGETS:
        aggregate_frames=[];aggregate_waves=[];aggregate_channels=None
        for start in sorted(STARTS):
            with np.load(OUT/f's{start}_waveforms.npz') as chunk:
                ch=chunk[target+'_channels']
                if aggregate_channels is None:aggregate_channels=ch.copy()
                else:assert np.array_equal(aggregate_channels,ch)
                aggregate_frames.append(chunk[target+'_frames']);aggregate_waves.append(chunk[target+'_waveforms_uv'])
        frames=np.concatenate(aggregate_frames);waves=np.concatenate(aggregate_waves);order=np.argsort(frames,kind='stable')
        np.savez_compressed(OUT/f'events_{target}.npz',frames=frames[order],waveforms=waves[order],channels=aggregate_channels)
    after=raw.stat();assert(before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
    assert settings['sources_sha256']=={str(p):sha(p) for p in inputs},'Source changed during run'
    result=dict(status='complete',seconds=time.monotonic()-begun,matched_events=len(rows),counts=pd.DataFrame(rows).groupby('candidate').size().to_dict(),cached970_exact_frames=True,cached970_median_waveforms_exact=True,scope='Existing two provisional identities only; fixedsupport/sparsecompetitor limitations unchanged; training960–970 flagged.')
    (OUT/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
