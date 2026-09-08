"""3sigma motion-input screen × 3kHz low-pass, with noise-scale control."""
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
SRC=ROOT/'testing/outputs';PRE=SRC/'luke_detection_threshold_sweep_v1';OUT=SRC/'luke_3sigma_lowpass_screen_v1'
ARMS=['broad_3sigma','broad_screen','lowpass_fixed','lowpass_fixed_screen','lowpass_adjusted','lowpass_adjusted_screen']
def keys(p):return p['sample_index']*384+p['channel_index']
def center_screen(voltage,p):
 ratio=np.empty(len(p),dtype='float32');off=np.arange(-30,31)
 for lo in range(0,len(p),20000):
  sl=slice(lo,min(lo+20000,len(p)));v=voltage[p['sample_index'][sl,None]+off,p['channel_index'][sl,None]];ratio[sl]=np.sum(v[:,15:46]**2,axis=1)/(np.sum(v*v,axis=1)+1e-12)
 return ratio>=.65,ratio

def main():
 OUT.mkdir(exist_ok=False);(OUT/'fields').mkdir();m=json.loads((BASE/'recording/rescue_recording_manifest.json').read_text());fs=m['sampling_frequency_hz'];geo=np.asarray(m['channel_locations_um']);modelpath=SRC/'luke_common_event_screen_v1/shared_response_model.npz';model=np.load(modelpath);noisepath=SRC/'luke_peak_threshold_screen_v1/noise_uv.npy';noise=np.load(noisepath);cfg=json.loads((PRE/'settings.json').read_text())['estimator']
 settings=dict(interval_s=[4160,4260],arms=ARMS,screen='Detector-waveform energy within ±0.5ms / energy within ±1ms >=0.65; no additional amplitude or neighbor gate',lowpass='Third-order Butterworth3000Hz forward/backward AFTER existing broadband300–6000Hz filtering and frozen shared-response compensation; 50ms external padding; no change to model fit or sorting voltage',noise_control='fixed retains baseline noise vector; adjusted multiplies it by per-channel MAD(lowpass)/MAD(broad compensated) measured once on4180–4200s and then frozen',calibration_interval_s=[4180,4200],detection=dict(method='locally_exclusive',peak_sign='neg',radius_um=50,detect_threshold=3),localization=dict(method='monopolar_triangulation',radius_um=75,n_jobs=8),reuse='Broadband3sigma peaks/locations exactly reused; narrowband detections all freshly localized on narrowband voltage. Shared narrowband events localized once.',estimator=cfg,strict_bound_um=80,comparison='Same nine event-matched lighthouse tracks, overall and drop/recovery scores; previously inspected interval, no independent validation',resume='Chunk arrays/masks/fields persist; no automatic resume or within-stage checkpoint; no sort',model_sha256=hashlib.sha256(modelpath.read_bytes()).hexdigest(),baseline_noise_sha256=hashlib.sha256(noisepath.read_bytes()).hexdigest())
 (OUT/'settings.json').write_text(json.dumps(settings,indent=2));allp={a:[] for a in ARMS};ally={a:[] for a in ARMS};logs=[];examples={};basefirst=round(4160*fs)
 for start in [4180,4160,4200,4220,4240]:
  n=round(20*fs);pad=round(.05*fs);first=round(start*fs)
  with (BASE/'recording/traces_cached_seg0.raw').open('rb') as handle:handle.seek((first-pad)*768);buf=handle.read((n+2*pad)*768)
  x=np.frombuffer(buf,dtype='<i2').reshape(-1,384).astype('float32')*m['gain_uv_per_count'];del buf;x=sosfiltfilt(butter(3,[300,6000],fs=fs,btype='bandpass',output='sos'),x,axis=0).astype('float32');ref=np.median(x,axis=1);cp=np.zeros_like(x)
  for lo in range(15,len(x)-15,10000):
   e=np.arange(lo,min(len(x)-15,lo+10000));cp[e]=x[e]-ref[e[:,None]+model['lag_samples']]@model['coefficients']
  del x,ref
  broad=cp[pad:pad+n].copy();narrow=sosfiltfilt(butter(3,3000,fs=fs,btype='lowpass',output='sos'),cp,axis=0).astype('float32')[pad:pad+n].copy();del cp
  if start==4180:
   mad_b=np.median(abs(broad-np.median(broad,axis=0)),axis=0)/.67448975;mad_n=np.median(abs(narrow-np.median(narrow,axis=0)),axis=0)/.67448975;noise_ratio=mad_n/mad_b;adjusted_noise=noise*noise_ratio;assert np.isfinite(adjusted_noise).all() and (adjusted_noise>0).all();np.savez_compressed(OUT/'noise_calibration.npz',broad_mad_uv=mad_b,lowpass_mad_uv=mad_n,ratio=noise_ratio,baseline_noise_uv=noise,adjusted_noise_uv=adjusted_noise);print('Noise ratio lowpass/broad min/median/max',float(noise_ratio.min()),float(np.median(noise_ratio)),float(noise_ratio.max()),flush=True)
  bp=np.load(PRE/f's{start}_3sigma_peaks.npy');by=np.load(PRE/f's{start}_3sigma_locations.npy');assert np.allclose(broad[bp['sample_index'],bp['channel_index']],bp['amplitude'],atol=1e-4,rtol=1e-5);bk,br=center_screen(broad,bp);np.save(OUT/f's{start}_broad_keep.npy',bk);np.save(OUT/f's{start}_broad_center_fraction.npy',br)
  record=NumpyRecording(narrow,fs);record.set_channel_locations(geo);det={}
  for name,nv in [('lowpass_fixed',noise),('lowpass_adjusted',adjusted_noise)]:
   p=detect_peaks(record,method='locally_exclusive',peak_sign='neg',radius_um=50.,detect_threshold=3.,noise_levels=nv,n_jobs=1,chunk_duration='1s',progress_bar=False);p=p[(p['sample_index']>=50)&(p['sample_index']<n-50)];det[name]=p;np.save(OUT/f's{start}_{name}_peaks.npy',p);print(start,name,'detections',len(p),flush=True)
  joined=np.concatenate(list(det.values()));uk,ii=np.unique(keys(joined),return_index=True);union=joined[ii];print(start,'localizing narrowband union',len(union),flush=True);loc=localize_peaks(record,union,method='monopolar_triangulation',radius_um=75.,n_jobs=8,mp_context='fork',chunk_duration='1s',progress_bar=False);assert np.isfinite(loc['y']).all();chunk={'broad_3sigma':(bp,by),'broad_screen':(bp[bk],by[bk])}
  for name,p in det.items():
   jj=np.searchsorted(uk,keys(p));assert np.array_equal(uk[jj],keys(p));y=loc[jj];keep,ratios=center_screen(narrow,p);np.save(OUT/f's{start}_{name}_locations.npy',y);np.save(OUT/f's{start}_{name}_keep.npy',keep);np.save(OUT/f's{start}_{name}_center_fraction.npy',ratios);chunk[name]=(p,y);chunk[name+'_screen']=(p[keep],y[keep])
  for name,(p,y) in chunk.items():
   pp=p.copy();pp['sample_index']+=first-basefirst;allp[name].append(pp);ally[name].append(y);logs.append(dict(start_s=start,arm=name,peaks=len(p)))
  if start==4180:
   # Review first calibration chunk only, selected without motion scores.
   for label,vol,p,k in [('broad',broad,bp,bk),('lowpass_adjusted',narrow,det['lowpass_adjusted'],np.load(OUT/f's{start}_lowpass_adjusted_keep.npy'))]:
    for band in range(4):
     for retain in [False,True]:
      ids=np.flatnonzero((k==retain)&(geo[p['channel_index'],1]>=band*960)&(geo[p['channel_index'],1]<(band+1)*960))
      if len(ids):
       j=ids[len(ids)//2];ch=int(p['channel_index'][j]);channels=np.flatnonzero(abs(geo[:,1]-geo[ch,1])<=60);w=vol[p['sample_index'][j]+np.arange(-30,31)[:,None],channels[None,:]];examples[(label,band,retain)]=(w,channels,ch,float(start+p['sample_index'][j]/fs))
  del broad,narrow,record;print(start,'chunk complete',flush=True)
 pd.DataFrame(logs).to_csv(OUT/'chunk_counts.csv',index=False);record=NumpyRecording(np.broadcast_to(np.zeros((1,384),dtype='float32'),(round(100*fs),384)),fs);record.set_channel_locations(geo);manifest=[];totalbase=sum(map(len,allp['broad_3sigma']))
 for name in ARMS:
  p=np.concatenate(allp[name]);y=np.concatenate(ally[name]);order=np.argsort(keys(p),kind='stable');p=p[order];y=y[order];np.save(OUT/f'{name}_peaks.npy',p);np.save(OUT/f'{name}_locations.npy',y)
  if name=='broad_3sigma':
   oldp=np.load(PRE/'3sigma_peaks.npy');oldy=np.load(PRE/'3sigma_locations.npy');assert np.array_equal(keys(p),keys(oldp)) and np.array_equal(y,oldy);old=np.load(PRE/'fields/3sigma.npz');np.savez_compressed(OUT/'fields'/f'{name}.npz',**{key:old[key] for key in old.files})
  else:
   motion,extra=estimate_bounded(record,p,y,cfg);np.savez_compressed(OUT/'fields'/f'{name}.npz',time_s=motion.temporal_bins_s[0]+4160,depth_um=motion.spatial_bins_um,displacement_um=motion.displacement[0],D=extra['D'],C=extra['C'],U=extra['U'])
  manifest.append(dict(name=name,peaks=len(p),retained_fraction=len(p)/totalbase));print(name,'100s motion complete',flush=True)
 pd.DataFrame(manifest).to_csv(OUT/'manifest.csv',index=False)
 from testing import luke_screen_sweep as scoring
 scoring.OUT=OUT;scoring.analyze();fig=plt.gcf();fig.suptitle('3σ input screen and3kHz low-pass comparison ·100s\nFixed versus adjusted detection noise separates filtering from threshold-scale effects')
 for ext in ['png','pdf']:fig.savefig(OUT/f'01_summary.{ext}',dpi=140)
 d=pd.read_csv(OUT/'event_matched_predictions.csv');tracks=pd.read_csv(SRC/'luke_lighthouse_gentle_v1/gentle_tracks.csv');colors=['#2878b5','#b13775','#37906c','#d49a00'];mainarms=['broad_3sigma','broad_screen','lowpass_adjusted','lowpass_adjusted_screen'];fig,axs=plt.subplots(3,3,figsize=(16,11),sharex=True,layout='constrained')
 for ax,(u,g) in zip(axs.ravel(),tracks.groupby('unit_id')):
  g=g.sort_values('time_s');base=float(g.iloc[0].median_waveform_centroid_um)
  for name,color in zip(mainarms,colors):
   q=d[(d.name==name)&(d.unit_id==u)];ax.plot(q.time_s,q.predicted_um,'.-',color=color,label=name)
  good=g.accepted_events>=10;ax.errorbar(g.time_s[good],g.median_waveform_centroid_um[good]-base,yerr=np.array([g.median_waveform_centroid_um[good]-g.bootstrap_low_um[good],g.bootstrap_high_um[good]-g.median_waveform_centroid_um[good]]),fmt='ko',ms=4,capsize=2,label='Lighthouse');ax.set(title=f'Unit{u} · {g.depth_um.iloc[0]:.0f}µm',xlabel='Recording time (s)',ylabel='Relative displacement (µm)')
 axs[0,0].legend(fontsize=6);fig.suptitle('Screen and3kHz low-pass from3σ baseline · event-matched lighthouse verification\nNarrowband main arms use the frozen channelwise noise-ratio adjustment; fixed-threshold controls scored separately')
 for ext in ['png','pdf']:fig.savefig(OUT/f'02_lighthouse_overlay.{ext}',dpi=140)
 from testing import luke_peak_population_review as hist
 hist.OUT=OUT;hist.histogram([(name,np.load(OUT/f'{name}_peaks.npy'),np.load(OUT/f'{name}_locations.npy')['y']) for name in mainarms],4160,4260,fs,'03_peak_histograms')
 for label in ['broad','lowpass_adjusted']:
  fig,axs=plt.subplots(4,2,figsize=(12,10),layout='constrained')
  for band,ax in enumerate(axs):
   for a,retain in zip(ax,[False,True]):
    w,channels,ch,t=examples[(label,band,retain)];ci=np.flatnonzero(channels==ch)[0];a.plot(np.arange(-30,31)/fs*1000,w,color='gray',alpha=.2);a.plot(np.arange(-30,31)/fs*1000,w[:,ci],color='black');a.set(title=f'{"Retained" if retain else "Excluded"} · ch{ch} · {t:.4f}s',xlabel='Time (ms)',ylabel='µV')
  fig.suptitle(f'{label}: off-center-energy screen examples across probe depth\nMedian-index event per decision/depth quarter; gray neighboring channels, black detector; no neural purity claim')
  for ext in ['png','pdf']:fig.savefig(OUT/f'04_waveform_decisions_{label}.{ext}',dpi=140)
 (OUT/'summary.json').write_text(json.dumps(dict(status='complete',counts={r['name']:r['peaks'] for r in manifest},median_noise_ratio=float(np.median(noise_ratio)),baseline_reused_exactly=True,scope='Motion-only diagnostic; no sort or production signal change'),indent=2))
if __name__=='__main__':main()
