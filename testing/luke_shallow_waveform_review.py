"""Review high-contribution shallow peak cohorts without labeling them as noise."""
import json
import numpy as np
import pandas as pd
from scipy.signal import butter,sosfiltfilt
from testing.luke_epoch_corroboration import ROOT,BASE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=ROOT/'testing/outputs/luke_shallow_waveform_review_v1';SRC=ROOT/'testing/outputs/luke_compensation_validation_v1'
def main():
 OUT.mkdir(exist_ok=False);m=json.loads((BASE/'recording/rescue_recording_manifest.json').read_text());fs=m['sampling_frequency_hz'];loc=np.array(m['channel_locations_um']);p=np.load(SRC/'compensated_peaks.npy');y=np.load(SRC/'compensated_locations.npy')['y'];model=np.load(ROOT/'testing/outputs/luke_common_event_screen_v1/shared_response_model.npz');noise=np.load(ROOT/'testing/outputs/luke_peak_threshold_screen_v1/noise_uv.npy');off=np.arange(-30,31);rows=[];waves={};rank=[]
 for a in [6,14]:
  k=(p['sample_index']>=a*fs)&(p['sample_index']<(a+1)*fs)&(y>=0)&(y<900)
  for ch in np.unique(p['channel_index'][k]):
   mask=k&(p['channel_index']==ch);rank.append(dict(second=4240+a,channel=int(ch),events=int(mask.sum()),amplitude_mass=float(abs(p['amplitude'][mask]).sum())))
 ranking=pd.DataFrame(rank);ranking.to_csv(OUT/'channel_contributions.csv',index=False);channels=ranking.groupby('channel').amplitude_mass.sum().nlargest(8).index.to_numpy();fig,axes=plt.subplots(4,2,figsize=(12,11),layout='constrained')
 for a,color in [(6,'tab:blue'),(14,'tab:orange')]:
  first=round((4240+a)*fs);n=round(fs);pad=round(.05*fs)
  with (BASE/'recording/traces_cached_seg0.raw').open('rb') as f:f.seek((first-pad)*768);buf=f.read((n+2*pad)*768)
  x=np.frombuffer(buf,dtype='<i2').reshape(-1,384).astype('float32')*m['gain_uv_per_count'];x=sosfiltfilt(butter(3,[300,6000],fs=fs,btype='bandpass',output='sos'),x,axis=0).astype('float32');ref=np.median(x,axis=1);clean=np.empty((n,384),dtype='float32')
  for lo in range(0,n,10000):
   e=np.arange(lo,min(lo+10000,n));clean[e]=x[pad+e]-ref[pad+e[:,None]+model['lag_samples']]@model['coefficients']
  for ax,ch in zip(axes.flat,channels):
   k=(p['sample_index']>=a*fs)&(p['sample_index']<(a+1)*fs)&(y>=0)&(y<900)&(p['channel_index']==ch);ev=p['sample_index'][k]+round(4240*fs)-first;ev=ev[(ev>31)&(ev<n-31)];total=len(ev)
   if not total:continue
   ev=ev[np.linspace(0,total-1,min(100,total),dtype=int)];w=clean[ev[:,None]+off,ch];med=np.median(w,axis=0);cos=w@med/(np.linalg.norm(w,axis=1)*np.linalg.norm(med));local=np.flatnonzero(abs(loc[:,1]-loc[ch,1])<=80);sp=np.median(clean[ev[:,None,None]+off[None,:,None],local[None,None,:]],axis=0);en=(sp*sp).sum(axis=0);concentration=float(en[abs(loc[local,1]-loc[ch,1])<=20].sum()/en.sum())
   rows.append(dict(second=4240+a,channel=int(ch),depth_um=float(loc[ch,1]),events=total,sampled=len(ev),median_peak_uv=float(abs(med).max()),median_peak_original_noise_snr=float(abs(med).max()/noise[ch]),median_single_channel_cosine=float(np.median(cos)),fraction_cosine_ge08=float((cos>=.8).mean()),energy_fraction_within20_of80=concentration))
   waves[f'ch{ch}_s{4240+a}']=sp;waves[f'ch{ch}_local_channels']=local
   ax.plot(off/fs*1000,med,color=color,label=f'{4240+a} s, n={total}');ax.fill_between(off/fs*1000,*np.percentile(w,[25,75],axis=0),color=color,alpha=.15);ax.set(title=f'Channel {ch} / {loc[ch,1]:.0f} µm',xlabel='Time from detected peak (ms)',ylabel='Compensated voltage (µV)');ax.legend(fontsize=8)
 fig.suptitle('Largest shallow amplitude contributors: waveform review\nTop 8 detection-channel cohorts across 4246–4247 and 4254–4255 s; median and interquartile range',fontsize=11)
 for ext in ['png','pdf']:fig.savefig(OUT/f'01_shallow_waveforms.{ext}',dpi=150)
 pd.DataFrame(rows).to_csv(OUT/'waveform_review.csv',index=False);np.savez_compressed(OUT/'median_spatial_waveforms.npz',**waves);(OUT/'settings.json').write_text(json.dumps(dict(scope='Descriptive channel cohorts, not single-unit identity; no exclusions or tuning',selection='Top8 aggregate amplitude-mass detection channels for localized depths0–900um in specified seconds',model_refit=False),indent=2));print(pd.DataFrame(rows).to_string(index=False))
if __name__=='__main__':main()
