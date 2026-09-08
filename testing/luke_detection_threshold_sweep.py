"""Fresh compensated 3/4/5/6-sigma detection with fixed localization and DREDGE."""
import json,hashlib
import numpy as np
import pandas as pd
from scipy.signal import butter,sosfiltfilt
from spikeinterface.core import NumpyRecording
from spikeinterface.sortingcomponents.peak_detection import detect_peaks
from spikeinterface.sortingcomponents.peak_localization import localize_peaks
from testing.luke_epoch_corroboration import ROOT,BASE
from testing.luke_dredge_bounded import estimate_bounded
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
SRC=ROOT/'testing/outputs';LONG=SRC/'luke_long_lighthouse_motion_v1';OUT=SRC/'luke_detection_threshold_sweep_v1'
def keys(p):return p['sample_index']*384+p['channel_index']
def main():
 OUT.mkdir(exist_ok=False);(OUT/'fields').mkdir();m=json.loads((BASE/'recording/rescue_recording_manifest.json').read_text());fs=m['sampling_frequency_hz'];geo=np.asarray(m['channel_locations_um']);modelpath=SRC/'luke_common_event_screen_v1/shared_response_model.npz';model=np.load(modelpath);noisepath=SRC/'luke_peak_threshold_screen_v1/noise_uv.npy';noise=np.load(noisepath);cfg=json.loads((LONG/'settings.json').read_text())['estimator'];thresholds=[3,4,5,6]
 settings=dict(interval_s=[4160,4260],thresholds_sigma=thresholds,voltage='Same frozen shared-response compensation,300–6000Hz filter; no aggressive event screen',noise='Identical frozen ORIGINAL channel MAD vector used in preceding5sigma detection, not reestimated per threshold',noise_sha256=hashlib.sha256(noisepath.read_bytes()).hexdigest(),model_sha256=hashlib.sha256(modelpath.read_bytes()).hexdigest(),detection=dict(method='locally_exclusive',peak_sign='neg',radius_um=50),localization=dict(method='monopolar_triangulation',radius_um=75,n_jobs=8),localization_reuse='Union of fresh detections localized once; exact sample/channel matches reuse cached5sigma locations; missing union events localized with identical settings',edge_rule='All thresholds exclude first/last50samples of every20s chunk; fresh5sigma verifies baseline on same interior',estimator=cfg,strict_pairwise_bound_um=80,comparison='Same event-matched nine lighthouse references; equal-cell overall and drop/recovery scores; exploratory same-window comparison',resume='Completed chunk arrays and motion fields persist; no within-stage checkpoint or automatic restart; no sort')
 (OUT/'settings.json').write_text(json.dumps(settings,indent=2));allp={s:[] for s in thresholds};ally={s:[] for s in thresholds};reuse=[];examples={};basefirst=round(4160*fs)
 for start in [4160,4180,4200,4220,4240]:
  first=round(start*fs);n=round(20*fs);pad=round(.05*fs)
  with (BASE/'recording/traces_cached_seg0.raw').open('rb') as handle:handle.seek((first-pad)*768);buf=handle.read((n+2*pad)*768)
  x=np.frombuffer(buf,dtype='<i2').reshape(-1,384).astype('float32')*m['gain_uv_per_count'];del buf;x=sosfiltfilt(butter(3,[300,6000],fs=fs,btype='bandpass',output='sos'),x,axis=0).astype('float32');ref=np.median(x,axis=1);clean=np.empty((n,384),dtype='float32')
  for lo in range(0,n,10000):
   e=np.arange(lo,min(n,lo+10000));clean[e]=x[pad+e]-ref[pad+e[:,None]+model['lag_samples']]@model['coefficients']
  del x,ref
  record=NumpyRecording(clean,fs);record.set_channel_locations(geo);det={}
  for s in thresholds:
   p=detect_peaks(record,method='locally_exclusive',peak_sign='neg',radius_um=50.,detect_threshold=float(s),noise_levels=noise,n_jobs=1,chunk_duration='1s',progress_bar=False);p=p[(p['sample_index']>=50)&(p['sample_index']<n-50)];det[s]=p;np.save(OUT/f's{start}_{s}sigma_peaks.npy',p);print(start,s,'sigma detections',len(p),flush=True)
  old=np.load(LONG/f's{start}_compensated_peaks.npy');oldy=np.load(LONG/f's{start}_compensated_locations.npy');valid=(old['sample_index']>=50)&(old['sample_index']<n-50);old=old[valid];oldy=oldy[valid];assert np.array_equal(keys(old),keys(det[5]));assert np.allclose(old['amplitude'],det[5]['amplitude'],rtol=1e-5,atol=1e-4)
  joined=np.concatenate(list(det.values()));unionkey,idx=np.unique(keys(joined),return_index=True);union=joined[idx];oldkey=keys(old);order=np.argsort(oldkey);oldkey=oldkey[order];oldy=oldy[order];pos=np.searchsorted(oldkey,unionkey);hit=pos<len(oldkey);hit[hit]=oldkey[pos[hit]]==unionkey[hit];loc=np.empty(len(union),dtype=oldy.dtype);loc[hit]=oldy[pos[hit]]
  if (~hit).any():
   print(start,'localizing new union peaks',int((~hit).sum()),flush=True);newloc=localize_peaks(record,union[~hit],method='monopolar_triangulation',radius_um=75.,n_jobs=8,mp_context='fork',chunk_duration='1s',progress_bar=False);assert newloc.dtype==loc.dtype;loc[~hit]=newloc
  assert np.isfinite(loc['y']).all();reuse.append(dict(start_s=start,union_peaks=len(union),reused_locations=int(hit.sum()),new_locations=int((~hit).sum()),fresh5sigma_keys_exact=True))
  for s,p in det.items():
   q=np.searchsorted(unionkey,keys(p));assert np.array_equal(unionkey[q],keys(p));y=loc[q];np.save(OUT/f's{start}_{s}sigma_locations.npy',y);pp=p.copy();pp['sample_index']+=first-basefirst;allp[s].append(pp);ally[s].append(y)
  if start==4160:
   off=np.arange(-30,31)
   for low,high in [(3,4),(4,5)]:
    p=det[low];added=~np.isin(keys(p),keys(det[high]))
    for band in range(4):
     ids=np.flatnonzero(added&(geo[p['channel_index'],1]>=band*960)&(geo[p['channel_index'],1]<(band+1)*960));j=ids[len(ids)//2];ch=int(p['channel_index'][j]);channels=np.flatnonzero(abs(geo[:,1]-geo[ch,1])<=60);w=clean[p['sample_index'][j]+off[:,None],channels[None,:]];examples[(low,high,band)]=(w,channels,ch,float(start+p['sample_index'][j]/fs),float(p['amplitude'][j]/noise[ch]))
  del clean,record;print(start,'chunk complete',flush=True)
 pd.DataFrame(reuse).to_csv(OUT/'localization_reuse.csv',index=False)
 record=NumpyRecording(np.broadcast_to(np.zeros((1,384),dtype='float32'),(round(100*fs),384)),fs);record.set_channel_locations(geo);manifest=[]
 for s in thresholds:
  p=np.concatenate(allp[s]);y=np.concatenate(ally[s]);assert np.all(np.diff(p['sample_index'])>=0);np.save(OUT/f'{s}sigma_peaks.npy',p);np.save(OUT/f'{s}sigma_locations.npy',y);motion,extra=estimate_bounded(record,p,y,cfg);np.savez_compressed(OUT/'fields'/f'{s}sigma.npz',time_s=motion.temporal_bins_s[0]+4160,depth_um=motion.spatial_bins_um,displacement_um=motion.displacement[0],D=extra['D'],C=extra['C'],U=extra['U']);manifest.append(dict(name=f'{s}sigma',peaks=len(p),retained_fraction=len(p)/sum(map(len,allp[5]))));print(s,'sigma motion complete',flush=True)
 pd.DataFrame(manifest).to_csv(OUT/'manifest.csv',index=False)
 # Reuse exact same metric calculations; produce threshold-specific figures below.
 from testing import luke_screen_sweep as scoring
 scoring.OUT=OUT;scoring.analyze()
 figure=plt.gcf();figure.suptitle('Fresh detection threshold sweep on compensated voltage ·100s\nSame event-matched lighthouse scores; peak-count ratio is relative to5σ, not a screen retention fraction')
 for ext in ['png','pdf']:figure.savefig(OUT/f'01_sweep_summary.{ext}',dpi=140)
 d=pd.read_csv(OUT/'event_matched_predictions.csv');tracks=pd.read_csv(SRC/'luke_lighthouse_gentle_v1/gentle_tracks.csv');fig,axs=plt.subplots(3,3,figsize=(16,11),sharex=True,layout='constrained');colors={3:'#37906c',4:'#d49a00',5:'#2878b5',6:'#b13775'}
 for ax,(u,g) in zip(axs.ravel(),tracks.groupby('unit_id')):
  g=g.sort_values('time_s');base=float(g.iloc[0].median_waveform_centroid_um)
  for s in thresholds:
   v=d[(d.name==f'{s}sigma')&(d.unit_id==u)];ax.plot(v.time_s,v.predicted_um,'.-',label=f'{s}σ',color=colors[s])
  good=g.accepted_events>=10;ax.errorbar(g.time_s[good],g.median_waveform_centroid_um[good]-base,yerr=np.array([g.median_waveform_centroid_um[good]-g.bootstrap_low_um[good],g.bootstrap_high_um[good]-g.median_waveform_centroid_um[good]]),fmt='ko',ms=4,capsize=2,label='Lighthouse');ax.set(title=f'Unit{u} · {g.depth_um.iloc[0]:.0f}µm',xlabel='Recording time (s)',ylabel='Relative displacement (µm)');ax.axhline(0,color='gray',lw=.4)
 axs[0,0].legend(fontsize=8);fig.suptitle('Fresh detection threshold sweep on compensated voltage ·100s\nSame localization, noise vector and DREDGE settings; motion sampled at accepted lighthouse event times')
 for ext in ['png','pdf']:fig.savefig(OUT/f'02_lighthouse_overlay.{ext}',dpi=140)
 from testing import luke_peak_population_review as hist
 hist.OUT=OUT;hist.histogram([(f'{s}σ detections',np.concatenate(allp[s]),np.concatenate(ally[s])['y']) for s in thresholds],4160,4260,fs,'03_peak_histograms')
 for low,high in [(3,4),(4,5)]:
  fig,axs=plt.subplots(4,2,figsize=(12,10),layout='constrained')
  for band,ax in enumerate(axs):
   w,channels,ch,t,amp=examples[(low,high,band)];tt=np.arange(-30,31)/fs*1000;ci=np.flatnonzero(channels==ch)[0];ax[0].plot(tt,w[:,ci]);ax[0].set(title=f'Added at{low}σ, absent at{high}σ · ch{ch}\n{t:.4f}s · detected amplitude{amp:.2f}σ',xlabel='Time (ms)',ylabel='µV');lim=np.max(abs(w));im=ax[1].imshow(w.T,origin='lower',aspect='auto',extent=[tt[0],tt[-1],0,len(channels)],vmin=-lim,vmax=lim,cmap='RdBu_r');fig.colorbar(im,ax=ax[1],label='µV');ax[1].set(title='Compensated local waveform',xlabel='Time (ms)',ylabel='Local channel')
  fig.suptitle('Examples of additional detections across probe depth\nDeterministic middle-index example per depth quarter in4160–4180s; not an estimate of neural purity')
  for ext in ['png','pdf']:fig.savefig(OUT/f'04_added_{low}to{high}sigma.{ext}',dpi=130)
 (OUT/'summary.json').write_text(json.dumps(dict(status='complete',counts={s:sum(map(len,pp)) for s,pp in allp.items()},fresh5sigma_baseline_verified=True,scope='Motion diagnostic, no sorting or motion application'),indent=2))
if __name__=='__main__':main()
