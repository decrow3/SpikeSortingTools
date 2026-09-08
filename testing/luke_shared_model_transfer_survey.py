"""Evenly spaced voltage transfer survey; descriptive, not neural-safety validation."""
import hashlib,json
import numpy as np
import pandas as pd
from scipy.signal import butter,sosfiltfilt
from testing.luke_epoch_corroboration import ROOT,BASE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=ROOT/'testing/outputs/luke_shared_model_transfer_survey_v1'
def main():
 OUT.mkdir(exist_ok=False);m=json.loads((BASE/'recording/rescue_recording_manifest.json').read_text());fs=m['sampling_frequency_hz'];raw=BASE/'recording/traces_cached_seg0.raw';duration=raw.stat().st_size/768/fs;starts=np.linspace(10,duration-12,12);model_path=ROOT/'testing/outputs/luke_common_event_screen_v1/shared_response_model.npz';model=np.load(model_path);coeff=model['coefficients'];lags=model['lag_samples'];rows=[];bychannel=[]
 (OUT/'settings.json').write_text(json.dumps(dict(starts_s=starts.tolist(),duration_per_sample_s=2,model_fit_interval_s=[4180,4190],model_refit=False,model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),scope='Evenly spaced transfer reconnaissance; conditional voltage summaries cannot establish neural preservation or motion accuracy'),indent=2))
 for start in starts:
  first=round(start*fs);n=round(2*fs);pad=round(.05*fs)
  with raw.open('rb') as f:f.seek((first-pad)*768);buf=f.read((n+2*pad)*768)
  x=np.frombuffer(buf,dtype='<i2').reshape(-1,384).astype('float32')*m['gain_uv_per_count'];x=sosfiltfilt(butter(3,[300,6000],fs=fs,btype='bandpass',output='sos'),x,axis=0).astype('float32');ref=np.median(x,axis=1);clean=np.empty((n,384),dtype='float32')
  for lo in range(0,n,10000):
   e=np.arange(lo,min(n,lo+10000));clean[e]=x[pad+e]-ref[pad+e[:,None]+lags]@coeff
  post=x[pad:pad+n]-ref[pad:pad+n,None];med=np.median(post,axis=0);mad=np.median(abs(post-med),axis=0)/.67448975;cmad=np.median(abs(clean-np.median(clean,axis=0)),axis=0)/.67448975;ratio=cmad/mad
  # High-common-voltage samples, selected solely from the original common trace.
  rr=ref[pad:pad+n];high=abs(rr)>=np.quantile(abs(rr),.99);e0=np.mean(post[high]**2,axis=0);e1=np.mean(clean[high]**2,axis=0);er=e1/np.maximum(e0,1e-12)
  rows.append(dict(start_s=float(start),common_abs_p99_uv=float(np.quantile(abs(rr),.99)),median_noise_ratio=float(np.median(ratio)),noise_ratio_p95=float(np.quantile(ratio,.95)),channels_noise_increase_gt20pct=int((ratio>1.2).sum()),high_common_energy_ratio_median=float(np.median(er)),high_common_energy_ratio_p95=float(np.quantile(er,.95))))
  for ch in range(384):bychannel.append(dict(start_s=float(start),channel=ch,depth_um=m['channel_locations_um'][ch][1],original_noise_uv=float(mad[ch]),compensated_noise_uv=float(cmad[ch]),noise_ratio=float(ratio[ch]),high_common_energy_ratio=float(er[ch])))
  print(json.dumps(rows[-1]),flush=True)
 d=pd.DataFrame(rows);d.to_csv(OUT/'summary.csv',index=False);c=pd.DataFrame(bychannel);c.to_csv(OUT/'channel_metrics.csv',index=False);fig,axes=plt.subplots(2,1,figsize=(12,7),layout='constrained')
 mat=c.pivot(index='channel',columns='start_s',values='noise_ratio').to_numpy();im=axes[0].imshow(mat,origin='lower',aspect='auto',vmin=.8,vmax=1.2,cmap='coolwarm');axes[0].set_xticks(np.arange(len(starts)),[f'{s/60:.1f}' for s in starts]);axes[0].set(xlabel='Recording time (min), 2 s samples',ylabel='Channel index',title='Compensated / original voltage MAD');fig.colorbar(im,ax=axes[0],label='Ratio (color clipped at 0.8–1.2)')
 axes[1].plot(starts/60,d.high_common_energy_ratio_median,'o-',label='Median across channels');axes[1].plot(starts/60,d.high_common_energy_ratio_p95,'x--',label='95th percentile across channels');axes[1].axhline(1,c='gray',lw=.7);axes[1].set(xlabel='Recording time (min)',ylabel='Compensated / original energy',title='Energy during largest 1% common-voltage samples');axes[1].legend()
 fig.suptitle('Frozen shared-response model: evenly spaced recording-wide reconnaissance\nVoltage summaries identify transfer concerns; they do not prove preservation of neural signals',fontsize=11)
 for ext in ['png','pdf']:fig.savefig(OUT/f'01_transfer_survey.{ext}',dpi=150)
 (OUT/'complete.json').write_text(json.dumps(dict(status='complete',samples=len(starts),duration_s=duration),indent=2))
if __name__=='__main__':main()
