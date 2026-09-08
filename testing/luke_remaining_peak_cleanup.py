"""Review remaining compensated peak cohorts across depth before classifier changes."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import butter,sosfiltfilt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from testing.luke_epoch_corroboration import ROOT,BASE
OUT=ROOT/'testing/outputs/luke_remaining_peak_cleanup_v1'
def main():
 OUT.mkdir(exist_ok=False)
 src=ROOT/'testing/outputs/luke_long_context_validation_v1'
 p=np.load(src/'compensated_peaks.npy');y=np.load(src/'compensated_locations.npy')['y']
 m=json.loads((BASE/'recording/rescue_recording_manifest.json').read_text());fs=m['sampling_frequency_hz'];geo=np.asarray(m['channel_locations_um'])
 model=np.load(ROOT/'testing/outputs/luke_common_event_screen_v1/shared_response_model.npz');lags=model['lag_samples'];coef=model['coefficients'];noise=model['residual_noise_uv']
 rows=[]
 for ch in range(384):
  inds=np.flatnonzero(p['channel_index']==ch)
  if len(inds)<24:continue
  t=p['sample_index'][inds]/fs;occupancy=np.mean(np.histogram(t,np.arange(0,101,1))[0]>0)
  fraction=np.mean(abs(y[inds]-geo[ch,1])<5)
  rows.append(dict(channel=ch,depth_um=geo[ch,1],count=len(inds),occupancy=occupancy,within_5um_fraction=fraction,review_rank=occupancy*fraction))
 ranks=pd.DataFrame(rows);ranks.to_csv(OUT/'channel_population.csv',index=False)
 chosen=[]
 for lo,hi in [(0,960),(960,1920),(1920,2880),(2880,3840)]:
  chosen.extend(ranks[(ranks.depth_um>=lo)&(ranks.depth_um<hi)].sort_values(['review_rank','count'],ascending=False).head(3).channel.astype(int).tolist())
 settings=dict(interval_s=[930,1030],input='Existing compensated fresh negative 5-sigma peaks',selection='Top 3 channels within each probe quarter ranked by 1-second occupancy × fraction localized within 5um of detector depth; review priority only, not artifact classification',sample='24 evenly indexed events per selected detector channel; split first/last 12 for repeatability',preprocessing='Same frozen shared response compensation; 300–6000Hz, 50ms padding',scope='Read-only cohort audit; no estimator or sort')
 (OUT/'settings.json').write_text(json.dumps(settings,indent=2))
 sos=butter(3,[300,6000],fs=fs,btype='bandpass',output='sos');pad=round(.05*fs);off=np.arange(-30,31);results=[];saved={}
 fig,axs=plt.subplots(len(chosen),3,figsize=(15,2.7*len(chosen)),layout='constrained')
 with (BASE/'recording/traces_cached_seg0.raw').open('rb') as f:
  for ax,ch in zip(axs,chosen):
   ids=np.flatnonzero(p['channel_index']==ch);ids=ids[np.linspace(0,len(ids)-1,24).astype(int)];waves=[]
   for idx in ids:
    frame=round(930*fs)+int(p['sample_index'][idx]);half=60;f.seek((frame-pad-half)*768);buf=f.read((2*(pad+half)+1)*768)
    x=np.frombuffer(buf,dtype='<i2').reshape(-1,384).astype('float32')*m['gain_uv_per_count'];x=sosfiltfilt(sos,x,axis=0).astype('float32');ref=np.median(x,axis=1);e=pad+half+off
    waves.append(x[e]-ref[e[:,None]+lags]@coef)
   w=np.asarray(waves);med=np.median(w,axis=0);a=np.median(w[:12],axis=0);b=np.median(w[12:],axis=0);local=abs(geo[:,1]-geo[ch,1])<=60
   v=w[:,:,ch];cos=v@med[:,ch]/(np.linalg.norm(v,axis=1)*np.linalg.norm(med[:,ch])+1e-12);repeat=np.sum(a[:,local]*b[:,local])/(np.linalg.norm(a[:,local])*np.linalg.norm(b[:,local])+1e-12)
   r=ranks[ranks.channel==ch].iloc[0].to_dict();r.update(median_peak_snr=float(np.max(abs(med[:,ch]))/noise[ch]),event_shape_pass_fraction=float(np.mean(cos>=.8)),local_half_repeat_cosine=float(repeat));results.append(r)
   saved[f'ch{ch}_waveforms']=w;saved[f'ch{ch}_peak_indices']=ids
   tt=off/fs*1000
   ax[0].plot(tt,v.T,color='gray',alpha=.18,lw=.7);ax[0].plot(tt,med[:,ch],color='black',lw=2);ax[0].plot(tt,a[:,ch],label='Earlier half');ax[0].plot(tt,b[:,ch],label='Later half');ax[0].legend(fontsize=6);ax[0].set(title=f'ch{ch} · {geo[ch,1]:.0f}µm · {r["count"]:.0f} peaks\nshape ≥0.8: {r["event_shape_pass_fraction"]:.0%}; median {r["median_peak_snr"]:.1f}σ',xlabel='Time (ms)',ylabel='µV')
   lim=np.max(abs(med[:,local]));im=ax[1].imshow(med[:,local].T,origin='lower',aspect='auto',extent=[tt[0],tt[-1],0,local.sum()],cmap='RdBu_r',vmin=-lim,vmax=lim);fig.colorbar(im,ax=ax[1],label='µV');ax[1].set(title=f'Median local footprint; half cosine {repeat:.2f}',ylabel='Local channel',xlabel='Time (ms)')
   idsall=np.flatnonzero(p['channel_index']==ch);ax[2].hist2d(930+p['sample_index'][idsall]/fs,y[idsall],bins=[np.arange(930,1031),np.arange(geo[ch,1]-100,geo[ch,1]+101,2)],cmap='magma');ax[2].axhline(geo[ch,1],color='cyan',lw=.7);ax[2].set(title=f'Localized depths · within5µm {r["within_5um_fraction"]:.0%}\n1s occupancy {r["occupancy"]:.0%}',xlabel='Recording time (s)',ylabel='Depth (µm)')
   print(f'Reviewed ch{ch}',flush=True)
 fig.suptitle('Remaining motion-input cohorts after compensation: probe-wide review priorities, not certified artifacts\n24 individual events per cohort; mixed neurons can fail shape consistency; flat localization alone is insufficient',fontsize=12)
 for ext in ['png','pdf']:fig.savefig(OUT/f'01_remaining_cohorts.{ext}',dpi=120)
 pd.DataFrame(results).to_csv(OUT/'waveform_review.csv',index=False);np.savez_compressed(OUT/'review_waveforms.npz',**saved)
 print(pd.DataFrame(results).to_string(index=False),flush=True)
if __name__=='__main__':main()
