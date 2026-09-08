"""Probe-time peak histograms and conservative versus aggressive artifact-mask review."""
import json
import numpy as np
import pandas as pd
from scipy.signal import butter,sosfiltfilt
from testing.luke_epoch_corroboration import ROOT,BASE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=ROOT/'testing/outputs/luke_peak_population_review_v1';SRC=ROOT/'testing/outputs'
def histogram(arms,start,stop,fs,name):
 te=np.arange(start,stop+.25,.25);de=np.arange(0,3841,10);hist=[]
 for label,p,y in arms:
  count=np.histogram2d(start+p['sample_index']/fs,y,bins=[te,de])[0].T;mass=np.histogram2d(start+p['sample_index']/fs,y,bins=[te,de],weights=abs(p['amplitude']))[0].T;hist.append((label,count,mass))
 limits=[np.quantile(np.concatenate([np.log1p(h[k]).ravel() for h in hist]),.995) for k in [1,2]];fig,axes=plt.subplots(len(arms),2,figsize=(13,3*len(arms)),sharex=True,sharey=True,layout='constrained')
 for ax,(label,count,mass) in zip(axes,hist):
  for a,v,limit,title in zip(ax,[count,mass],limits,['Peak counts','Sum |peak amplitude|']):
   im=a.imshow(np.log1p(v),origin='lower',aspect='auto',extent=[start,stop,0,3840],vmin=0,vmax=limit,cmap='magma');a.set(title=f'{label} · {title}',ylabel='Depth (µm)',xlabel='Recording time (s)');fig.colorbar(im,ax=a,label='log(1 + bin total)')
 fig.suptitle('Peak population across probe and time\n0.25s ×10µm bins; shared color scales across input arms; colors clipped at pooled99.5th percentile',fontsize=11)
 for ext in ['png','pdf']:fig.savefig(OUT/f'{name}.{ext}',dpi=150)
 np.savez_compressed(OUT/f'{name}_histograms.npz',time_edges=te,depth_edges=de,**{f'arm{i}_{k}':v for i,h in enumerate(hist) for k,v in [('counts',h[1]),('amplitude_mass',h[2]) ]})
def main():
 OUT.mkdir(exist_ok=False);m=json.loads((BASE/'recording/rescue_recording_manifest.json').read_text());fs=m['sampling_frequency_hz'];loc=np.asarray(m['channel_locations_um']);p=np.load(SRC/'luke_peak_threshold_screen_v1/peaks_5sigma.npy');y=np.load(SRC/'luke_peak_threshold_screen_v1/locations_5sigma.npy')['y'];d=pd.read_csv(SRC/'luke_common_event_screen_v1/peak_classification.csv');oldkeep=np.load(SRC/'luke_common_event_screen_v1/keep_mask.npy');suspect=(d.broad_coherence_fraction>=.8)&(d.energy_explained>=.8)&(d.predicted_waveform_cosine>=.95);reject=np.zeros(len(p),bool);reject[d.loc[suspect,'peak_index'].to_numpy()]=True;keep=oldkeep&~reject
 np.save(OUT/'aggressive_keep_mask.npy',keep);np.save(OUT/'aggressive_retained_peaks.npy',p[keep]);np.save(OUT/'aggressive_retained_locations.npy',np.load(SRC/'luke_peak_threshold_screen_v1/locations_5sigma.npy')[keep]);d['aggressive_reject']=reject[d.peak_index.to_numpy()];d.to_csv(OUT/'peak_decisions.csv',index=False)
 histogram([('Original',p,y),('Previous conservative screen',p[oldkeep],y[oldkeep]),('Aggressive shared-explained rejection',p[keep],y[keep])],4180,4200,fs,'01_screen_histograms')
 arms=[]
 for arm in ['original','compensated']:
  q=np.load(SRC/f'luke_long_context_validation_v1/{arm}_peaks.npy');yy=np.load(SRC/f'luke_long_context_validation_v1/{arm}_locations.npy')['y'];arms.append((arm,q,yy))
 histogram(arms,930,1030,fs,'02_long_histograms')
 # Individual examples selected deterministically for each documented decision category.
 groups=[('Previously rejected',d[d.reject_shared_artifact]),('Previously retained; now rejected',d[(~d.reject_shared_artifact)&d.aggressive_reject]),('Retained; not certified neural',d[(~d.aggressive_reject)&(d.energy_explained<.2)])];selected=[]
 for label,g in groups:
  for lo,hi in [(0,2500),(2500,3840)]:
   pool=g[(g.depth_um>=lo)&(g.depth_um<hi)&(g.time_s>4180.1)&(g.time_s<4199.9)]
   if len(pool):selected.append((label,pool.iloc[len(pool)//2]))
 first=round(4180*fs);off=np.arange(-30,31);model=np.load(SRC/'luke_common_event_screen_v1/shared_response_model.npz');fig,axes=plt.subplots(len(selected),2,figsize=(12,3*len(selected)),squeeze=False,layout='constrained');examples=[]
 for ax,(label,r) in zip(axes,selected):
  frame=int(p['sample_index'][int(r.peak_index)])+first;pad=round(.05*fs);half=100
  with (BASE/'recording/traces_cached_seg0.raw').open('rb') as f:f.seek((frame-half-pad)*768);buf=f.read((2*half+2*pad+1)*768)
  x=np.frombuffer(buf,dtype='<i2').reshape(-1,384).astype('float32')*m['gain_uv_per_count'];x=sosfiltfilt(butter(3,[300,6000],fs=fs,btype='bandpass',output='sos'),x,axis=0).astype('float32');ref=np.median(x,axis=1);e=half+pad+off;ch=int(r.channel);observed=x[e,ch]-ref[e];prediction=(ref[e[:,None]+model['lag_samples']]@model['coefficients'])[:,ch]-ref[e];residual=observed-prediction;tt=off/fs*1000
  ax[0].plot(tt,observed,label='Original referenced waveform');ax[0].plot(tt,prediction,'--',label='Predicted shared residual');ax[0].plot(tt,residual,label='Remainder');ax[0].set(title=f'{label}\n{r.time_s:.4f}s · ch{ch} · explained {r.energy_explained:.0%}',xlabel='Time (ms)',ylabel='µV');ax[0].legend(fontsize=7)
  local=np.flatnonzero(abs(loc[:,1]-loc[ch,1])<=80);wave=x[e[:,None],local]-ref[e,None];limit=np.max(abs(wave));im=ax[1].imshow(wave.T,origin='lower',aspect='auto',extent=[tt[0],tt[-1],0,len(local)],cmap='RdBu_r',vmin=-limit,vmax=limit);ax[1].set(title=f'Original local waveform · detector depth {loc[ch,1]:.0f}µm',xlabel='Time (ms)',ylabel='Local channel index');fig.colorbar(im,ax=ax[1],label='µV');examples.append(dict(category=label,peak_index=int(r.peak_index),time_s=float(r.time_s),channel=ch))
 fig.suptitle('Individual included/excluded peaks: no claim that retained equals neural\nAggressive rule removes shared-explained events even when a local remainder exists',fontsize=11)
 for ext in ['png','pdf']:fig.savefig(OUT/f'03_waveform_decisions.{ext}',dpi=150)
 pd.DataFrame(examples).to_csv(OUT/'waveform_examples.csv',index=False);result=dict(status='complete',original_peaks=len(p),previous_rejected=int((~oldkeep).sum()),aggressive_rejected=int((~keep).sum()),additional_rejected=int((oldkeep&~keep).sum()),remaining=int(keep.sum()),rule='Reject shared-coherent>=.8 AND detection-channel explained energy>=.8 AND predicted waveform cosine>=.95; remove previous local-residual rescue. This is one artifact rule, not a neural-only classifier.',scope='Motion-only diagnostic mask; source voltage and sorting unchanged');(OUT/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
