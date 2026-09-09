"""Managed21-arm early100s sweep; frozen17 waveform candidates; no sort."""
import hashlib,json,os,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.signal import butter,sosfiltfilt
from spikeinterface.core import NumpyRecording
from spikeinterface.sortingcomponents.peak_detection import detect_peaks
from spikeinterface.sortingcomponents.peak_localization import localize_peaks
from testing.luke_epoch_corroboration import ROOT,BASE
from testing.luke_screen_sweep import features,mask
from testing.luke_early_screen_shortlist_v1 import chunk_paths
from testing.luke_dredge_bounded_range import estimate_bounded_range
SRC=ROOT/'testing/outputs';OUT=SRC/'luke_early_sigma_screen250_v1';SIGMAS=[3,4,5,6,8];SCREENS={'none':None,'center_only':{'only':['center']},'relaxed':{'snr':6,'center':.5,'neighbor':.6},'full':{}}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def atomic(p,v):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2));os.replace(t,p)
def valid(stage):
 seal=stage/'complete.json'
 if not seal.exists():
  if stage.exists():raise RuntimeError(f'Unsealed stage evidence at{stage}; investigate before restart')
  return False
 for fn,h in json.loads(seal.read_text()).items():assert sha(stage/fn)==h,f'Changed checkpoint{stage/fn}'
 return True
def seal(stage):atomic(stage/'complete.json',{p.name:sha(p) for p in sorted(stage.iterdir()) if p.is_file()})
def keys(p):return p['sample_index']*384+p['channel_index']
def main():
 torch.set_num_threads(1);OUT.mkdir(exist_ok=True);(OUT/'fields').mkdir(exist_ok=True);begin=time.monotonic();mp=BASE/'recording/rescue_recording_manifest.json';m=json.loads(mp.read_text());fs=m['sampling_frequency_hz'];geo=np.array(m['channel_locations_um']);raw=BASE/'recording/traces_cached_seg0.raw';stat=raw.stat();modelp=SRC/'luke_common_event_screen_v1/shared_response_model.npz';noisep=SRC/'luke_peak_threshold_screen_v1/noise_uv.npy';model=np.load(modelp);noise=np.load(noisep);cfg=json.loads((SRC/'luke_early_screen_shortlist_v1/settings.json').read_text())['estimator'];cfg['max_disp_um']=250.
 sources=[Path(__file__),ROOT/'testing/luke_dredge_bounded_range.py',ROOT/'testing/luke_early_sigma_screen250_report_v1.py',ROOT/'testing/luke_screen_sweep.py',mp,modelp,noisep,SRC/'luke_early_screen_shortlist_v1/settings.json',SRC/'luke_waveform_only_expansion_v1/overlay_events.csv',SRC/'luke_waveform_only_expansion_v1/candidate_audit.csv',SRC/'luke_population_depth_v2/templates.npz']
 for start in [930,950,970,990,1010]:sources.extend(chunk_paths(start))
 settings=dict(interval_s=[930,1030],sigmas=SIGMAS,screens=SCREENS,estimator=cfg,extra_control='5sigma_none_bound80 on exact same common-interior peaks as250',reference='Frozen17 candidates; strict,lower-score and ambiguous evidence remain separate; seed930–940s; heldout940–1030s. No candidate reselection or fit to any field.',noise='Frozen original per-channel MAD for detection; frozen residual_noise_uv for screening features; neither reestimated per arm.',detection='Fresh negative locally-exclusive50um radius; first/last50samples of each20s chunk excluded for all arms',localization='Union of all fresh threshold detections; reuse exact5sigma cached sample/channel locations; new union peaks localized once,monopolar triangulation75um',screening='Same frozen feature functions and masks as prior sweep; exact center/full/relaxed settings in screens',restart='Hash-validated completed chunk/field stages reused; no within-stage checkpoint. Unsealed stage refuses restart until interruption evidence reviewed.',early_control='First recompute250um on full cached5sigma input for cheap bound-only comparison to saved80um; keep separately from common-interior sweep.',source_sha256={str(p):sha(p) for p in sources},raw_stat=[stat.st_size,stat.st_mtime_ns])
 sp=OUT/'settings.json'
 if sp.exists():assert json.loads(sp.read_text())==settings,'Changed inputs/settings need a new output version'
 else:atomic(sp,settings)
 record=NumpyRecording(np.broadcast_to(np.zeros((1,384),np.float32),(round(100*fs),384)),fs);record.set_channel_locations(geo)
 quick=OUT/'cached_bound250'
 if not valid(quick):
  quick.mkdir();p=np.load(SRC/'luke_long_context_validation_v1/compensated_peaks.npy');y=np.load(SRC/'luke_long_context_validation_v1/compensated_locations.npy');motion,ex=estimate_bounded_range(record,p,y,cfg);np.savez_compressed(quick/'field.npz',time_s=motion.temporal_bins_s[0]+930,depth_um=motion.spatial_bins_um,displacement_um=motion.displacement[0],**ex);seal(quick);print('CACHED250 bound-only field complete',flush=True)
 from testing.luke_early_sigma_screen250_report_v1 import quick_report
 quick_report()
 # Same reference data remain frozen throughout extraction and screening.
 for start in [930,950,970,990,1010]:
  stage=OUT/f'chunk_{start}'
  if valid(stage):print(start,'reused verified chunk',flush=True);continue
  stage.mkdir();tick=time.monotonic();first=round(start*fs);n=round(20*fs);pad=round(.05*fs)
  with raw.open('rb') as f:f.seek((first-pad)*768);buf=f.read((n+2*pad)*768)
  assert len(buf)==(n+2*pad)*768
  x=np.frombuffer(buf,dtype='<i2').reshape(-1,384).astype('float32')*m['gain_uv_per_count'];del buf;x=sosfiltfilt(butter(3,[300,6000],fs=fs,btype='bandpass',output='sos'),x,axis=0).astype('float32');ref=np.median(x,axis=1);clean=np.empty((n,384),np.float32)
  for lo in range(0,n,10000):
   ev=np.arange(lo,min(n,lo+10000));clean[ev]=x[pad+ev]-ref[pad+ev[:,None]+model['lag_samples']]@model['coefficients']
  post=x[pad:pad+n]-ref[pad:pad+n,None];del x,ref;rec=NumpyRecording(clean,fs);rec.set_channel_locations(geo);det={}
  for sigma in SIGMAS:
   p=detect_peaks(rec,method='locally_exclusive',peak_sign='neg',radius_um=50.,detect_threshold=float(sigma),noise_levels=noise,n_jobs=1,chunk_duration='1s',progress_bar=False);p=p[(p['sample_index']>=50)&(p['sample_index']<n-50)];det[sigma]=p;np.save(stage/f'{sigma}sigma_peaks.npy',p);print(start,sigma,'sigma',len(p),'detections',flush=True)
  pp,yp=chunk_paths(start);old=np.load(pp);oldy=np.load(yp);keep=(old['sample_index']>=50)&(old['sample_index']<n-50);old=old[keep];oldy=oldy[keep];assert np.array_equal(keys(old),keys(det[5]));assert np.allclose(old['amplitude'],det[5]['amplitude'],rtol=1e-5,atol=1e-4)
  joined=np.concatenate(list(det.values()));uk,ix=np.unique(keys(joined),return_index=True);union=joined[ix];order=np.argsort(keys(old));ok=keys(old)[order];oy=oldy[order];pos=np.searchsorted(ok,uk);hit=pos<len(ok);hit[hit]=ok[pos[hit]]==uk[hit];loc=np.empty(len(union),dtype=oy.dtype);loc[hit]=oy[pos[hit]]
  if (~hit).any():
   print(start,'localizing',int((~hit).sum()),'new union events',flush=True);loc[~hit]=localize_peaks(rec,union[~hit],method='monopolar_triangulation',radius_um=75.,n_jobs=4,mp_context='fork',chunk_duration='1s',progress_bar=False)
  assert np.isfinite(loc['y']).all();np.save(stage/'union_peaks.npy',union);np.save(stage/'union_locations.npy',loc)
  print(start,'computing union screening features',flush=True);feat=features(clean,post,union,geo,model['residual_noise_uv'],fs);np.savez_compressed(stage/'union_features.npz',**feat)
  for sigma in SIGMAS:
   ix=np.searchsorted(uk,keys(det[sigma]));assert np.array_equal(uk[ix],keys(det[sigma]));np.save(stage/f'{sigma}sigma_union_indices.npy',ix)
  atomic(stage/'audit.json',dict(union_events=len(union),reused_locations=int(hit.sum()),new_locations=int((~hit).sum()),fresh5sigma_baseline_exact=True,seconds=time.monotonic()-tick));seal(stage);del rec,clean,post;print(start,'SEALED chunk',round(time.monotonic()-tick,1),'seconds',flush=True)
 # Complete stage outputs remain reusable without voltage reads.
 manifest=[]
 for sigma in SIGMAS:
  for screen,kwargs in SCREENS.items():
   for bound in ([80,250] if sigma==5 and screen=='none' else [250]):
    name=f'{sigma}sigma_{screen}_bound{bound}';stage=OUT/'fields'/name
    if valid(stage):manifest.append(json.loads((stage/'arm.json').read_text()));continue
    stage.mkdir();parts=[];positions=[];maskpaths=[];input_count=0
    for start in [930,950,970,990,1010]:
     source=OUT/f'chunk_{start}';union=np.load(source/'union_peaks.npy');loc=np.load(source/'union_locations.npy');ix=np.load(source/f'{sigma}sigma_union_indices.npy');input_count+=len(ix);f=np.load(source/'union_features.npz');keep=np.ones(len(ix),bool) if kwargs is None else mask({k:f[k][ix] for k in f.files},**kwargs);np.save(stage/f's{start}_keep.npy',keep);p=union[ix[keep]].copy();p['sample_index']+=round(start*fs)-round(930*fs);parts.append(p);positions.append(loc[ix[keep]])
    p=np.concatenate(parts);y=np.concatenate(positions);np.save(stage/'peaks.npy',p);np.save(stage/'locations.npy',y);conf=cfg.copy();conf['max_disp_um']=float(bound);print('ESTIMATING',name,len(p),'peaks',flush=True);motion,ex=estimate_bounded_range(record,p,y,conf);np.savez_compressed(stage/'field.npz',time_s=motion.temporal_bins_s[0]+930,depth_um=motion.spatial_bins_um,displacement_um=motion.displacement[0],**ex)
    te=np.arange(930,1030.001,.25);de=np.arange(0,3840.001,10);counts=np.histogram2d(930+p['sample_index']/fs,y['y'],bins=[te,de])[0].T;mass=np.histogram2d(930+p['sample_index']/fs,y['y'],bins=[te,de],weights=abs(p['amplitude']))[0].T;np.savez_compressed(stage/'coverage.npz',counts=counts,amplitude_mass=mass,time_edges=te,depth_edges=de)
    arm=dict(name=name,sigma=sigma,screen=screen,bound_um=bound,peaks=len(p),input_peaks=input_count,retained_fraction=len(p)/input_count,locations_outside_probe=int(((y['y']<geo[:,1].min())|(y['y']>geo[:,1].max())).sum()));atomic(stage/'arm.json',arm);seal(stage);manifest.append(arm);pd.DataFrame(manifest).to_csv(OUT/'manifest.csv',index=False);print('SEALED FIELD',name,flush=True)
 pd.DataFrame(manifest).to_csv(OUT/'manifest.csv',index=False);assert (raw.stat().st_size,raw.stat().st_mtime_ns)==(stat.st_size,stat.st_mtime_ns);assert settings['source_sha256']=={str(p):sha(p) for p in sources}
 from testing.luke_early_sigma_screen250_report_v1 import report
 report();atomic(OUT/'summary.json',dict(status='complete',arms=len(manifest),cached_bound_control=True,seconds=time.monotonic()-begin));print('COMPLETE21 arms and report',flush=True)
if __name__=='__main__':main()
