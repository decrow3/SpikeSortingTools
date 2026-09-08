"""Bounded motion-only waveform family inclusion; no motion estimator or sort."""
import json
import numpy as np
import pandas as pd
from scipy.signal import butter,sosfiltfilt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from testing.luke_epoch_corroboration import ROOT,BASE
OUT=ROOT/'testing/outputs/luke_motion_waveform_screen_v2'
SRC=ROOT/'testing/outputs'
def unit(v):return v/(np.linalg.norm(v,axis=-1,keepdims=True)+1e-12)
def main():
 OUT.mkdir(exist_ok=False)
 settings=dict(interval_s=[4180,4200],family_training_s=[4180,4190],evaluation_s=[4190,4200],shared_reject='Original referenced detector waveform: frozen common-model explained energy >=0.5 AND cosine>=0.8',event_gate='Detector max absolute waveform >=8 residual sigma; >=65% detector energy in ±0.5ms; second local channel max >=4 sigma and >=25% detector max',families='Per detector channel, local ±60um full 61sample waveforms; amplitude normalized; deterministic greedy cosine>=0.85; max12 seeds/channel; >=10 training members; odd/even local median cosine>=0.9; only first10s train; no lighthouse or motion input',retention='Event gates plus neighbor waveform cosine>=0.8, dominant-lobe half-height width0.067–0.8ms; exclude simultaneous broad residual excursions on >=20% of probe channels. Family match advisory only',limits=['Diagnostic thresholds, not optimized or validated neural classifier','Repeatable artifacts can pass; excluded events may be neural','Fixed channel families may lose moving events; retention must be reviewed through depth/time','Residual noise vector comes from model training interval','This interval was previously inspected; temporal split is not blind validation'],scope='Motion-only candidate mask. No source voltage or sort configuration changes.')
 (OUT/'settings.json').write_text(json.dumps(settings,indent=2))
 m=json.loads((BASE/'recording/rescue_recording_manifest.json').read_text());fs=m['sampling_frequency_hz'];geo=np.asarray(m['channel_locations_um']);n=round(20*fs);pad=round(.05*fs)
 model=np.load(SRC/'luke_common_event_screen_v1/shared_response_model.npz');noise=model['residual_noise_uv'];lags=model['lag_samples'];coef=model['coefficients']
 with (BASE/'recording/traces_cached_seg0.raw').open('rb') as f:f.seek((round(4180*fs)-pad)*768);buf=f.read((n+2*pad)*768)
 x=np.frombuffer(buf,dtype='<i2').reshape(-1,384).astype('float32')*m['gain_uv_per_count'];del buf
 x=sosfiltfilt(butter(3,[300,6000],fs=fs,btype='bandpass',output='sos'),x,axis=0).astype('float32');ref=np.median(x,axis=1);clean=np.empty((n,384),dtype='float32')
 for lo in range(0,n,10000):
  e=np.arange(lo,min(lo+10000,n));clean[e]=x[pad+e]-ref[pad+e[:,None]+lags]@coef
 post=x[pad:pad+n]-ref[pad:pad+n,None];del x,ref
 p=np.load(SRC/'luke_compensated_peak_trial_v1/peaks.npy');y=np.load(SRC/'luke_compensated_peak_trial_v1/locations.npy');off=np.arange(-30,31);mid=abs(off)<=15
 rows=[];families=[];templates={};example={};keep=np.zeros(len(p),bool)
 print('Voltage prepared; screening',len(p),'peaks',flush=True)
 for ch in range(384):
  ids=np.flatnonzero((p['channel_index']==ch)&(p['sample_index']>40)&(p['sample_index']<n-40))
  if not len(ids):continue
  ev=p['sample_index'][ids];channels=np.flatnonzero(abs(geo[:,1]-geo[ch,1])<=60);ci=int(np.flatnonzero(channels==ch)[0]);w=clean[ev[:,None,None]+off[None,:,None],channels[None,None,:]]
  v=w[:,:,ci];original=post[ev[:,None]+off,ch];pred=original-v
  explained=1-np.sum(v*v,axis=1)/(np.sum(original*original,axis=1)+1e-12);sharedcos=np.sum(original*pred,axis=1)/(np.linalg.norm(original,axis=1)*np.linalg.norm(pred,axis=1)+1e-12);shared=(explained>=.5)&(sharedcos>=.8)
  amp=np.max(abs(v),axis=1);snr=amp/noise[ch];concentration=np.sum(v[:,mid]**2,axis=1)/(np.sum(v*v,axis=1)+1e-12)
  other=np.max(abs(w),axis=1);other[:,ci]=0;support=((other>=.25*amp[:,None])&(other/noise[channels]>=4)).any(axis=1)
  gate=(snr>=8)&(concentration>=.65)&support&~shared
  features=unit(w.reshape(len(w),-1));train=np.flatnonzero(gate&(ev<10*fs));seed=[]
  # Earliest deterministic qualifying event seeds each shape; cap explicitly leaves rare families unresolved.
  for j in train:
   if not seed or np.max(features[seed]@features[j])<.85:
    if len(seed)<12:seed.append(j)
  good=[]
  if seed:
   corr=features[train]@features[seed].T;labels=np.argmax(corr,axis=1)
   for k,s in enumerate(seed):
    member=train[(labels==k)&(corr[:,k]>=.85)]
    if len(member)<10:continue
    a=np.median(w[member[::2]],axis=0);b=np.median(w[member[1::2]],axis=0);repeat=float(unit(a.ravel())@unit(b.ravel()))
    if repeat<.9:continue
    template=np.median(w[member],axis=0);good.append(unit(template.ravel()));key=f'ch{ch}_family{len(good)-1}';templates[key]=template;templates[f'{key}_channels']=channels
    families.append(dict(channel=ch,family=len(good)-1,training_events=len(member),odd_even_cosine=repeat))
  best=np.zeros(len(ids));label=np.full(len(ids),-1)
  if good:
   corr=features@np.asarray(good).T;label=np.argmax(corr,axis=1);best=np.max(corr,axis=1)
  # Evaluate coherent spatial support without requiring a fixed identity or absolute depth.
  neighborcos=np.einsum('ntc,nt->nc',w,v)/(np.linalg.norm(w,axis=1)*np.linalg.norm(v,axis=1)[:,None]+1e-12)
  supportmask=(other>=.25*amp[:,None])&(other/noise[channels]>=4)
  coherent=(supportmask&(neighborcos>=.8)).any(axis=1)
  dominant=np.argmax(abs(v),axis=1);width=np.zeros(len(ids))
  for j,k in enumerate(dominant):
   sign=np.sign(v[j,k]);lo=hi=int(k)
   while lo>0 and sign*v[j,lo-1]>=.5*amp[j]:lo-=1
   while hi<60 and sign*v[j,hi+1]>=.5*amp[j]:hi+=1
   width[j]=(hi-lo+1)/fs*1000
  # Broad residual activity is suspect for this intentionally selective motion input.
  broad=np.zeros(len(ids))
  for lo in range(0,len(ids),100):
   sl=slice(lo,min(lo+100,len(ids)));centers=ev[sl]+off[dominant[sl]]
   full=clean[centers[:,None]+np.arange(-3,4)]
   broad[sl]=np.mean(np.max(abs(full),axis=1)/noise>=4,axis=1)
  structure=coherent&(width>=.067)&(width<=.8)&(broad<.2)
  passed=gate&structure;keep[ids]=passed
  for j,idx in enumerate(ids):
   reason='retained' if passed[j] else 'shared_explained' if shared[j] else 'low_snr' if snr[j]<8 else 'offcenter_or_complex' if concentration[j]<.65 else 'weak_spatial_support' if not support[j] else 'broad_residual' if broad[j]>=.2 else 'implausible_width' if not (.067<=width[j]<=.8) else 'incoherent_neighbors'
   rows.append(dict(peak_index=int(idx),time_s=float(4180+ev[j]/fs),channel=ch,depth_um=float(y['y'][idx]),half=int(ev[j]>=10*fs),snr=float(snr[j]),central_energy_fraction=float(concentration[j]),shared_explained=float(explained[j]),shared_cosine=float(sharedcos[j]),neighbor_coherent=bool(coherent[j]),dominant_halfwidth_ms=float(width[j]),broad_residual_fraction=float(broad[j]),family_cosine=float(best[j]),family=int(label[j]),decision=reason,keep=bool(passed[j])))
   key=(reason,int(geo[ch,1]//960))
   if key not in example and ev[j]>=10*fs:example[key]=(v[j].copy(),w[j].copy(),channels.copy(),float(4180+ev[j]/fs),ch)
  if ch%50==0:print('Channel',ch,flush=True)
 d=pd.DataFrame(rows);d.to_csv(OUT/'peak_decisions.csv',index=False);pd.DataFrame(families).to_csv(OUT/'families.csv',index=False);np.savez_compressed(OUT/'templates.npz',**templates)
 np.save(OUT/'keep_mask.npy',keep);np.save(OUT/'retained_peaks.npy',p[keep]);np.save(OUT/'retained_locations.npy',y[keep]);np.save(OUT/'excluded_peaks.npy',p[~keep])
 counts=d.groupby(['half','decision']).size().unstack(fill_value=0);counts.to_csv(OUT/'decision_counts.csv')
 from testing import luke_peak_population_review as hist
 hist.OUT=OUT
 hist.histogram([('Compensated input',p,y['y']),('Strict waveform input',p[keep],y['y'][keep]),('Excluded from motion',p[~keep],y['y'][~keep])],4180,4200,fs,'01_histograms')
 # All categories shown across depth where available, without selecting on motion agreement.
 for page,band in enumerate(range(4)):
  selected=[(key,val) for key,val in example.items() if key[1]==band]
  fig,axs=plt.subplots(len(selected),2,figsize=(12,2.5*len(selected)),squeeze=False,layout='constrained')
  for ax,(key,(v,w,channels,t,ch)) in zip(axs,selected):
   ax[0].plot(off/fs*1000,v);ax[0].set(title=f'{key[0]} · ch{ch} · {t:.4f}s',xlabel='Time (ms)',ylabel='µV')
   lim=np.max(abs(w));im=ax[1].imshow(w.T,aspect='auto',origin='lower',extent=[off[0]/fs*1000,off[-1]/fs*1000,0,len(channels)],cmap='RdBu_r',vmin=-lim,vmax=lim);fig.colorbar(im,ax=ax[1],label='µV');ax[1].set(title='Compensated local waveform',xlabel='Time (ms)',ylabel='Local channel')
  fig.suptitle(f'Waveform inclusion/exclusion examples · depth quarter {band+1}\nFirst eligible second-half example per decision category; exclusion does not establish noise identity')
  for ext in ['png','pdf']:fig.savefig(OUT/f'02_decisions_depth{band+1}.{ext}',dpi=130)
 summary=dict(input_peaks=len(p),retained=int(keep.sum()),excluded=int((~keep).sum()),boundary_unclassified=int(len(p)-len(d)),families=len(families),by_half=counts.to_dict(),status='complete',production_change=False)
 (OUT/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
