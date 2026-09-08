"""Compare voltage footprints and localization for concentrated stationary sources."""
from testing.luke_epoch_corroboration import ROOT,BASE
from testing.luke_multidepth_anchors_v2 import scores
import numpy as np,pandas as pd,json
from scipy.signal import butter,sosfiltfilt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=ROOT/'testing/outputs/luke_stationary_voltage_audit_v1';PRE=ROOT/'testing/outputs/luke_peak_threshold_screen_v1'
def main():
 OUT.mkdir(exist_ok=False);rec=json.loads((BASE/'recording/rescue_recording_manifest.json').read_text());fs=rec['sampling_frequency_hz'];loc=np.asarray(rec['channel_locations_um']);first=round(4180*fs);n=round(20*fs);pad=round(.05*fs);raw=BASE/'recording/traces_cached_seg0.raw';stat=raw.stat()
 settings=dict(interval_s=[4180,4200],baseline_s=[4180,4190],repeat_s=[4190,4200],source_channels=[[238],[251,252]],candidate_radius_um=100,template_support_um=140,template_baseline_amplitude_quantile=.75,weighted_cosine_min=.9,gain_range=[.5,2],method='Build each baseline template from high-amplitude detections on source channels; then search all fresh5sigma events within ±100um using fixed-channel template, independent of localized depth. Compare voltage before/after and before/after global reference. No target depth gate.',limitations=['Frozen spatial template can favor stationary events; broad candidate window reduces channel-selection bias but does not remove template selection bias.','Stationary voltage does not establish non-neural origin; highly localized neuronal footprints may have weak displacement sensitivity.'])
 (OUT/'settings.json').write_text(json.dumps(settings,indent=2))
 with raw.open('rb') as f:f.seek((first-pad)*768);buf=f.read((n+2*pad)*768)
 assert len(buf)==(n+2*pad)*768
 x=np.frombuffer(buf,dtype='<i2').reshape(-1,384).astype('float32')*rec['gain_uv_per_count'];sos=butter(3,[300,6000],fs=fs,btype='bandpass',output='sos');filtered=sosfiltfilt(sos,x,axis=0).astype('float32');del x,buf;filtered=filtered[pad:pad+n].copy();x=filtered-np.median(filtered,axis=1,keepdims=True);noise=np.load(PRE/'noise_uv.npy');p=np.load(PRE/'peaks_5sigma.npy');pos=np.load(PRE/'locations_5sigma.npy');valid=(p['sample_index']>40)&(p['sample_index']<n-40);rng=np.random.default_rng(12);rows=[];eventrows=[];saved={};fig,axes=plt.subplots(2,3,figsize=(15,9));details=[]
 for axs,source in zip(axes,[[238],[251,252]]):
  tag='_'.join(map(str,source));dep=float(np.mean(loc[source,1]));channels=np.flatnonzero(abs(loc[:,1]-dep)<=140);base=valid&np.isin(p['channel_index'],source)&(p['sample_index']<10*fs);floor=float(np.quantile(abs(p['amplitude'][base]),.75));ii=np.flatnonzero(base&(abs(p['amplitude'])>=floor));ev=p['sample_index'][ii];template=np.median(x[ev[:,None,None]+np.arange(-30,31)[None,:,None],channels[None,None,:]],axis=0);weights=template**2/(template**2+(2*noise[channels])**2);cand=np.flatnonzero(valid&(abs(loc[p['channel_index'],1]-dep)<=100));sc=scores(x,p['sample_index'][cand],channels,template,weights);keep=(sc[:,0]>=.9)&(sc[:,1]>=.5)&(sc[:,1]<=2);accepted=[]
  for k in np.argsort(-sc[:,0]):
   if keep[k] and all(abs(sc[k,2]-sc[j,2])>.0008*fs for j in accepted):accepted.append(k)
  accepted=np.array(accepted,dtype=int);idx=cand[accepted];centers=sc[accepted,2].astype(int);order=np.argsort(centers);idx=idx[order];centers=centers[order];waves=[];prewaves=[];stats=[]
  for half in [0,1]:
   choose=np.flatnonzero((centers>=half*10*fs)&(centers<(half+1)*10*fs));e=centers[choose];take=e[np.linspace(0,len(e)-1,min(300,len(e)),dtype=int)];w=x[take[:,None,None]+np.arange(-30,31)[None,:,None],channels[None,None,:]];med=np.median(w,axis=0);pre=np.median(filtered[take[:,None,None]+np.arange(-30,31)[None,:,None],channels[None,None,:]],axis=0);waves.append(med);prewaves.append(pre)
   energy=np.sum(med*med,axis=0);center=float(energy@loc[channels,1]/energy.sum());pk=int(np.argmax(energy));fraction=float(energy[pk]/energy.sum());boot=[]
   for _ in range(100):
    v=np.median(w[rng.integers(0,len(w),len(w))],axis=0);en=np.sum(v*v,axis=0);boot.append(en@loc[channels,1]/en.sum())
   q=np.quantile(boot,[.025,.975]);stats.append(dict(source=tag,half=half,events=len(e),events_used=len(take),centroid_um=center,bootstrap_low_um=q[0],bootstrap_high_um=q[1],localized_median_um=float(np.median(pos['y'][idx[choose]])),peak_energy_channel=int(channels[pk]),peak_channel_energy_fraction=fraction));saved[f'{tag}_half{half}_median']=med;saved[f'{tag}_half{half}_individual']=w[:30];saved[f'{tag}_half{half}_preref']=pre
  saved[f'{tag}_channels']=channels;rows.extend(stats)
  for i,e in zip(idx,centers):eventrows.append(dict(source=tag,frame=int(first+e),detected_channel=int(p['channel_index'][i]),localized_depth_um=float(pos['y'][i]),saved_peak_uv=float(p['amplitude'][i])))
  peak=int(np.argmax(np.sum(waves[0]**2,axis=0)));tt=np.arange(-30,31)/fs*1000
  for half,color in [(0,'#167a9a'),(1,'#b65b24')]:
   axs[0].plot(tt,waves[half][:,peak],c=color,label=['Before','After'][half]);axs[0].plot(tt,prewaves[half][:,peak],c=color,ls=':',alpha=.8,label=['Before, no global reference','After, no global reference'][half]);axs[1].plot(np.sqrt(np.sum(waves[half]**2,axis=0)),loc[channels,1],'-o',ms=3,c=color,label=['Before','After'][half]);axs[2].plot(np.arange(len(channels)),np.ptp(waves[half],axis=0),'-o',ms=3,c=color,label=['Before','After'][half])
  co=float(np.sum(waves[0]*waves[1])/(np.linalg.norm(waves[0])*np.linalg.norm(waves[1])));delta=stats[1]['centroid_um']-stats[0]['centroid_um'];ldelta=stats[1]['localized_median_um']-stats[0]['localized_median_um'];details.append(dict(source=tag,baseline_amplitude_floor_uv=floor,waveform_cosine=co,centroid_step_um=delta,localized_step_um=ldelta,accepted_channel_counts={str(ch):int(np.sum(p['channel_index'][idx]==ch)) for ch in np.unique(p['channel_index'][idx])}))
  axs[0].set_title(f'Source ch{tag} | {stats[0]["events"]}/{stats[1]["events"]} events\nBefore/after waveform cosine {co:.4f}');axs[0].set(xlabel='Time from aligned detection (ms)',ylabel='Voltage (µV)');axs[0].legend(fontsize=7);axs[1].set_title(f'Voltage centroid step {delta:+.3f} µm\nLocalized-depth step {ldelta:+.3f} µm');axs[1].set(xlabel='Median-waveform energy amplitude (µV·√sample)',ylabel='Depth (µm)');axs[1].legend(fontsize=7);axs[2].set_title(f'Per-channel footprint\nPeak-channel energy: {stats[0]["peak_channel_energy_fraction"]:.0%} → {stats[1]["peak_channel_energy_fraction"]:.0%}');axs[2].set_xticks(np.arange(len(channels))[::3],[str(ch) for ch in channels[::3]],rotation=45);axs[2].set(xlabel='Channel index',ylabel='Peak-to-peak voltage (µV)')
 np.savez_compressed(OUT/'waveform_evidence.npz',**saved);pd.DataFrame(rows).to_csv(OUT/'voltage_comparison.csv',index=False);pd.DataFrame(eventrows).to_csv(OUT/'matched_events.csv',index=False);fig.suptitle('Do concentrated stationary sources stay fixed in voltage as well as localization?\n4180–4190 vs 4190–4200 s; candidate detections searched ±100 µm; waveform matching remains conditional',fontsize=12);fig.tight_layout(rect=[0,0,1,.92]);fig.savefig(OUT/'01_stationary_voltage.png',dpi=160);fig.savefig(OUT/'01_stationary_voltage.pdf');plt.close(fig)
 # Unselected source-channel cohort: all fresh detections, no amplitude or shape selection.
 rawrows=[]
 for source in [[238],[251,252]]:
  tag='_'.join(map(str,source));dep=np.mean(loc[source,1]);ch=np.flatnonzero(abs(loc[:,1]-dep)<=140)
  for half in [0,1]:
   ii=np.flatnonzero(valid&np.isin(p['channel_index'],source)&(p['sample_index']>=half*10*fs)&(p['sample_index']<(half+1)*10*fs));ev=p['sample_index'][ii];ev=ev[np.linspace(0,len(ev)-1,min(300,len(ev)),dtype=int)];w=np.median(x[ev[:,None,None]+np.arange(-30,31)[None,:,None],ch[None,None,:]],axis=0);en=np.sum(w*w,axis=0);rawrows.append(dict(source=tag,half=half,events=len(ii),centroid_um=float(en@loc[ch,1]/en.sum()),localized_median_um=float(np.median(pos['y'][ii]))))
 pd.DataFrame(rawrows).to_csv(OUT/'unselected_channel_cohorts.csv',index=False)
 assert (raw.stat().st_size,raw.stat().st_mtime_ns)==(stat.st_size,stat.st_mtime_ns)
 result=dict(status='complete',sources=details,limitations=settings['limitations']);(OUT/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
