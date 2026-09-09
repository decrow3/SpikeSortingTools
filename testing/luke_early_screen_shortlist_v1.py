"""Five published screening arms on exact cached early100s compensated5sigma inputs."""
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import butter,sosfiltfilt
from spikeinterface.core import NumpyRecording
from testing.luke_epoch_corroboration import ROOT,BASE
from testing.luke_screen_sweep import features,mask
from testing.luke_dredge_bounded import estimate_bounded

SRC=ROOT/'testing/outputs'
PRE=SRC/'luke_long_context_validation_v1'
OUT=SRC/'luke_early_screen_shortlist_v1'
STARTS=[930,950,970,990,1010]
ARMS=[('compensated',{}),('center_only',{'only':['center']}),
      ('relaxed_combination',{'snr':6,'center':.5,'neighbor':.6}),
      ('without_snr',{'omit':['snr']}),('full_screen',{})]


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def chunk_paths(start):
    directory=SRC/'luke_distant_motion_validation_v1' if start==970 else PRE
    return directory/f's{start}_compensated_peaks.npy',directory/f's{start}_compensated_locations.npy'


def main():
    begun=time.monotonic();OUT.mkdir(exist_ok=False);(OUT/'fields').mkdir();(OUT/'masks').mkdir()
    manifestpath=BASE/'recording/rescue_recording_manifest.json';modelpath=SRC/'luke_common_event_screen_v1/shared_response_model.npz'
    m=json.loads(manifestpath.read_text());fs=m['sampling_frequency_hz'];geo=np.asarray(m['channel_locations_um'])
    model=np.load(modelpath);noise=model['residual_noise_uv'];cfg=json.loads((PRE/'settings.json').read_text())['estimator']
    sources=[Path(__file__).resolve(),ROOT/'testing/luke_screen_sweep.py',ROOT/'testing/luke_dredge_bounded.py',manifestpath,modelpath,PRE/'settings.json',PRE/'compensated_peaks.npy',PRE/'compensated_locations.npy',PRE/'compensated_dredge_motion.npz']
    for start in STARTS:sources.extend(chunk_paths(start))
    settings=dict(interval_s=[930,1030],arms=[dict(name=name,screen_kwargs=kw) for name,kw in ARMS],
        baseline='Exact cached compensated5sigma peaks/locations; all five fields recomputed with the same strict bounded helper, including baseline. No detection/localization.',
        gates=dict(full_screen='SNR>=8,center>=.65,neighbor>=.8,width.067–.8ms,broadfraction<.2,reject sharedexplained>=.5 AND cosine>=.8',center_only='Onlycenter>=.65',without_snr='All full_screen gates except SNR',relaxed_combination='Fullscreen withSNR>=6,center>=.5,neighbor>=.6'),
        feature_noise='Frozen shared-response-model residual_noise_uv, identical to published screen_sweep feature definition, not baseline detection noise.',
        preprocessing='Exact existing300–6000Hz thirdorder forward/backward bandpass +frozen shared-response compensation;50ms padding. Original medianreferenced waveform retained solely for shared-noise features.',
        estimator=cfg,strict_pairwise_bound_um=80,resume='No automatic resume or within-stage checkpoint; fresh output directory required. Persisted partial evidence preserved on failure.',
        scope='Screen-only transfer diagnostic; previously viewed earlyinterval; no motionaccuracy claim, sort, newdetections, or parameter search.',
        sources_sha256={str(p):sha(p) for p in sources})
    (OUT/'settings.json').write_text(json.dumps(settings,indent=2))
    # Verify complete cached inputs before reading voltage or calculating masks.
    parts=[];positions=[];basefirst=round(930*fs)
    for start in STARTS:
        pp,yy=chunk_paths(start);p=np.load(pp).copy();p['sample_index']+=round(start*fs)-basefirst;parts.append(p);positions.append(np.load(yy))
    allp=np.concatenate(parts);ally=np.concatenate(positions)
    assert np.array_equal(allp,np.load(PRE/'compensated_peaks.npy'))
    assert np.array_equal(ally,np.load(PRE/'compensated_locations.npy'))
    del parts,positions
    featureparts=[];timings=[];raw=BASE/'recording/traces_cached_seg0.raw';before=raw.stat()
    for start in STARTS:
        t=time.monotonic();n=round(20*fs);pad=round(.05*fs);first=round(start*fs)
        with raw.open('rb') as handle:handle.seek((first-pad)*768);buf=handle.read((n+2*pad)*768)
        assert len(buf)==(n+2*pad)*768
        x=np.frombuffer(buf,dtype='<i2').reshape(-1,384).astype('float32')*m['gain_uv_per_count'];del buf
        x=sosfiltfilt(butter(3,[300,6000],fs=fs,btype='bandpass',output='sos'),x,axis=0).astype('float32')
        ref=np.median(x,axis=1);clean=np.empty((n,384),np.float32)
        for lo in range(0,n,10000):
            e=np.arange(lo,min(n,lo+10000));clean[e]=x[pad+e]-ref[pad+e[:,None]+model['lag_samples']]@model['coefficients']
        post=x[pad:pad+n]-ref[pad:pad+n,None];del x,ref
        peaks=np.load(chunk_paths(start)[0])
        assert np.allclose(clean[peaks['sample_index'],peaks['channel_index']],peaks['amplitude'],atol=1e-4,rtol=1e-5),f'Baseline amplitude mismatch at{start}'
        f=features(clean,post,peaks,geo,noise,fs);featureparts.append(f)
        np.savez_compressed(OUT/f's{start}_features.npz',**f)
        for name,kwargs in ARMS:
            keep=np.ones(len(peaks),bool) if name=='compensated' else mask(f,**kwargs)
            np.save(OUT/'masks'/f's{start}_{name}.npy',keep)
        timings.append(dict(stage='features',start_s=start,seconds=time.monotonic()-t,peaks=len(peaks)));pd.DataFrame(timings).to_csv(OUT/'timings.csv',index=False)
        del clean,post;print(start,'features complete; baseline amplitudes verified',flush=True)
    after=raw.stat();assert(before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
    f={key:np.concatenate([part[key] for part in featureparts]) for key in featureparts[0]}
    np.savez_compressed(OUT/'features.npz',**f)
    record=NumpyRecording(np.broadcast_to(np.zeros((1,384),np.float32),(round(100*fs),384)),fs);record.set_channel_locations(geo)
    manifest=[]
    for name,kwargs in ARMS:
        t=time.monotonic();keep=np.ones(len(allp),bool) if name=='compensated' else mask(f,**kwargs)
        np.save(OUT/'masks'/f'{name}.npy',keep);np.save(OUT/f'{name}_peaks.npy',allp[keep]);np.save(OUT/f'{name}_locations.npy',ally[keep])
        path=OUT/'fields'/f'{name}.npz'
        motion,extra=estimate_bounded(record,allp[keep],ally[keep],cfg)
        np.savez_compressed(path,time_s=motion.temporal_bins_s[0]+930,depth_um=motion.spatial_bins_um,displacement_um=motion.displacement[0],D=extra['D'],C=extra['C'],U=extra['U'])
        manifest.append(dict(name=name,peaks=int(keep.sum()),retained_fraction=float(keep.mean()),mask_sha256=sha(OUT/'masks'/f'{name}.npy')))
        timings.append(dict(stage='motion',arm=name,seconds=time.monotonic()-t));pd.DataFrame(timings).to_csv(OUT/'timings.csv',index=False)
        pd.DataFrame(manifest).to_csv(OUT/'manifest.csv',index=False);print(name,'motion complete',int(keep.sum()),flush=True)
    assert settings['sources_sha256']=={str(p):sha(p) for p in sources},'Source changed during run'
    (OUT/'summary.json').write_text(json.dumps(dict(status='complete',seconds=time.monotonic()-begun,baseline_arrays_exact=True,baseline_field_recomputed_with_matched_strict_bounds=True,arms=manifest,scope=settings['scope']),indent=2))


if __name__=='__main__':main()
