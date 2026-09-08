"""Fresh detector threshold comparison and provisional morphology flags."""
from testing.luke_epoch_corroboration import ROOT,BASE
import numpy as np
import pandas as pd
import json
from scipy.signal import butter,sosfiltfilt
from spikeinterface.core import NumpyRecording
from spikeinterface.sortingcomponents.peak_detection import detect_peaks
from spikeinterface.sortingcomponents.peak_localization import localize_peaks
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=ROOT/'testing/outputs/luke_peak_threshold_screen_v1'
def main():
 OUT.mkdir(exist_ok=False);rec=json.loads((BASE/'recording/rescue_recording_manifest.json').read_text());fs=rec['sampling_frequency_hz'];loc=np.array(rec['channel_locations_um']);start=4180;duration=20;first=round(start*fs);n=round(duration*fs);pad=round(.05*fs)
 settings=dict(interval_s=[4180,4200],thresholds_sigma=[4,5,6,7],source=str(BASE/'recording/traces_cached_seg0.raw'),conditioning='Accepted phase/blank/interpolated voltage + 300–6000 Hz third-order forward-backward Butterworth + global median reference, actual µV gain',peak_sign='neg',radius_um=50,noise='Per-channel MAD/0.67448975 on same conditioned 20s, held identical across all thresholds',localization='monopolar_triangulation, radius75um, ±0.5ms',screen='Provisional flags: >=20 distant channels (>200um away) exceed 5 sigma within ±0.1ms; OR peak half-height duration <0.06ms. Not a neuronal classifier.',scope='First paired-epoch diagnostic; not full gentle interval or historical preprocessing replay; no sort or motion rerun')
 (OUT/'settings.json').write_text(json.dumps(settings,indent=2))
 with (BASE/'recording/traces_cached_seg0.raw').open('rb') as f:f.seek((first-pad)*768);buf=f.read((n+2*pad)*768)
 assert len(buf)==(n+2*pad)*768
 x=np.frombuffer(buf,dtype='<i2').reshape(-1,384).astype('float32')*rec['gain_uv_per_count'];sos=butter(3,[300,6000],fs=fs,btype='bandpass',output='sos');x=sosfiltfilt(sos,x,axis=0).astype('float32');x-=np.median(x,axis=1,keepdims=True);x=x[pad:pad+n].copy();del buf
 noise=np.median(abs(x-np.median(x,axis=0)),axis=0)/.67448975;np.save(OUT/'noise_uv.npy',noise);recording=NumpyRecording(x,fs);recording.set_channel_locations(loc);branches={}
 for threshold in [4,5,6,7]:
  p=detect_peaks(recording,method='locally_exclusive',peak_sign='neg',radius_um=50.,detect_threshold=float(threshold),noise_levels=noise,n_jobs=1,chunk_duration='1s',progress_bar=False);branches[threshold]=p;np.save(OUT/f'peaks_{threshold}sigma.npy',p);print(f'Fresh {threshold}sigma: {len(p)} peaks',flush=True)
 # Localization is independent of detection threshold. Localize union once, then map exact event/channel keys.
 allp=np.concatenate(list(branches.values()));keys=allp['sample_index']*384+allp['channel_index'];_,ix=np.unique(keys,return_index=True);union=allp[ix];order=np.argsort(union['sample_index'],kind='stable');union=union[order];uk=union['sample_index']*384+union['channel_index'];keyorder=np.argsort(uk)
 print(f'Localizing {len(union)} distinct peaks',flush=True)
 positions=localize_peaks(recording,union,method='monopolar_triangulation',radius_um=75.,n_jobs=4,mp_context='fork',chunk_duration='1s',progress_bar=False);np.save(OUT/'union_peaks.npy',union);np.save(OUT/'union_locations.npy',positions)
 metrics=[];examples={};offset=np.arange(-30,31)
 for begin in range(0,len(union),500):
  inds=np.arange(begin,min(begin+500,len(union)));valid=(union['sample_index'][inds]>=31)&(union['sample_index'][inds]<n-31);inds=inds[valid]
  if not len(inds):continue
  ev=union['sample_index'][inds];ch=union['channel_index'][inds];waves=x[ev[:,None]+offset];center=x[ev[:,None]+np.arange(-3,4)];peakamp=abs(x[ev,ch]);near=np.abs(loc[None,:,1]-loc[ch,1,None])<=60;far=np.abs(loc[None,:,1]-loc[ch,1,None])>200;sim=np.sum((np.max(abs(center),axis=1)>5*noise[None,:])&far,axis=1);energy=np.sum(waves*waves,axis=1);compact=np.sum(energy*near,axis=1)/np.sum(energy,axis=1)
  for j,idx in enumerate(inds):
   wf=waves[j,:,ch[j]];mask=abs(wf)>=peakamp[j]/2;left=right=30
   while left>0 and mask[left-1]:left-=1
   while right<60 and mask[right+1]:right+=1
   width=(right-left+1)/fs*1000;broad=sim[j]>=20;brief=width<.06;flag=bool(broad or brief);metrics.append(dict(union_index=int(idx),time_s=float(start+ev[j]/fs),channel=int(ch[j]),localized_depth_um=float(positions['y'][idx]),peak_uv=float(peakamp[j]),peak_noise_ratio=float(peakamp[j]/noise[ch[j]]),local_energy_fraction=float(compact[j]),distant_synchronous_channels=int(sim[j]),half_height_width_ms=width,broad_flag=bool(broad),brief_flag=bool(brief),screen_flag=flag))
   category='broad' if broad else 'brief' if brief else 'unflagged'
   # Bounded representative review: strongest sample per category and 1000um depth band.
   key=(category,int(np.clip(loc[ch[j],1]//1000,0,3)))
   if key not in examples or peakamp[j]>examples[key][0]:examples[key]=(float(peakamp[j]),waves[j].copy(),int(ch[j]),int(idx),width,int(sim[j]))
 d=pd.DataFrame(metrics).set_index('union_index');d.to_csv(OUT/'peak_morphology.csv');selected=pd.read_csv(ROOT/'testing/outputs/luke_lighthouse_top20_v1/top20_ranked.csv');ts=np.load(BASE/'cur/cur_output/spike_times.npy',mmap_mode='r').ravel();cl=np.load(BASE/'cur/cur_output/spike_clusters.npy',mmap_mode='r').ravel();mask=(ts>=first+40)&(ts<first+n-40);ref=ts[mask]-first;ids=cl[mask];summary=[];recovery=[]
 for threshold,p in branches.items():
  k=p['sample_index']*384+p['channel_index'];ii=keyorder[np.searchsorted(uk[keyorder],k)];assert np.array_equal(uk[ii],k);np.save(OUT/f'locations_{threshold}sigma.npy',positions[ii]);flag=d.reindex(ii).screen_flag.fillna(True).to_numpy(bool);np.save(OUT/f'provisional_keep_{threshold}sigma.npy',~flag)
  summary.append(dict(threshold_sigma=threshold,detected=len(p),flagged=int(flag.sum()),retained=int((~flag).sum()),fraction_flagged=float(flag.mean())))
  for row in selected.itertuples():
   truth=ref[ids==row.unit_id];local=np.abs(loc[p['channel_index'],1]-row.depth_um)<=60
   for screened,keep in [('all',local),('provisional_screen',local&~flag)]:
    hit=p['sample_index'][keep];depth=positions['y'][ii[keep]];got=[];matches=[]
    for event in truth:
     a,b=np.searchsorted(hit,[event-round(.0008*fs),event+round(.0008*fs)+1]);got.append(b>a)
     if b>a:
      j=a+np.argmin(abs(hit[a:b]-event));matches.append((event/fs+start,depth[j]))
    before=[v for t,v in matches if t<4190];after=[v for t,v in matches if t>=4190];recovery.append(dict(threshold_sigma=threshold,unit_id=int(row.unit_id),depth_um=row.depth_um,arm=screened,reference_events=len(truth),coincident_peak_fraction=float(np.mean(got)) if len(got) else np.nan,before_matches=len(before),after_matches=len(after),historical_style_localized_step_um=float(np.median(after)-np.median(before)) if min(len(before),len(after))>=10 else np.nan))
 s=pd.DataFrame(summary);s.to_csv(OUT/'threshold_summary.csv',index=False);r=pd.DataFrame(recovery);r.to_csv(OUT/'lighthouse_coincidence.csv',index=False)
 def save(fig,name):
  fig.savefig(OUT/(name+'.png'),dpi=160,bbox_inches='tight');fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
 fig,ax=plt.subplots(1,2,figsize=(12,6));ax[0].bar(s.threshold_sigma,s.retained,label='Unflagged');ax[0].bar(s.threshold_sigma,s.flagged,bottom=s.retained,label='Provisional morphology flag',color='#aaaaaa');ax[0].set(xlabel='Detection threshold (noise σ)',ylabel='Detected peaks',xticks=[4,5,6,7]);ax[0].legend(fontsize=8)
 for arm,style in [('all','-'),('provisional_screen','--')]:
  g=r[r.arm==arm].pivot(index='unit_id',columns='threshold_sigma',values='coincident_peak_fraction');ax[1].plot(g.columns,g.median(axis=0),style,marker='o',label=arm)
 ax[1].set(xlabel='Detection threshold (noise σ)',ylabel='Median per-candidate coincidence fraction',ylim=(0,1.03),xticks=[4,5,6,7]);ax[1].legend();fig.suptitle('Fresh threshold comparison: 4180–4200 s, fixed conditioning/noise/localization\nCoincidence within ±0.8 ms and ±60 µm is recovery evidence, not confirmed peak identity');fig.tight_layout();save(fig,'01_threshold_comparison')
 keys=sorted(examples);fig,axes=plt.subplots(len(keys),2,figsize=(11,2.6*len(keys)),squeeze=False)
 for axs,key in zip(axes,keys):
  amp,w,ch,idx,width,sim=examples[key];tt=offset/fs*1000;axs[0].plot(tt,w[:,ch],color='k');axs[0].fill_between(tt,-3*noise[ch],3*noise[ch],color='gray',alpha=.2);axs[0].set_title(f'{key[0]} · ch{ch} · {loc[ch,1]:.0f} µm · {amp:.0f} µV');axs[0].set(xlabel='Time (ms)',ylabel='µV');axs[1].plot(np.max(abs(w),axis=0),loc[:,1]);axs[1].set_title(f'Half-height {width:.3f} ms; distant synchronous channels {sim}');axs[1].set(xlabel='Peak absolute voltage (µV)',ylabel='Depth (µm)',ylim=(0,3820))
 fig.suptitle('Provisional morphology review — strongest example per category/depth band\nA flag is not proof of non-neural origin; unflagged is not proof of neural origin');fig.tight_layout(rect=[0,0,1,.97]);save(fig,'02_morphology_examples')
 result=dict(status='complete',summary=summary,unique_localizations=len(union),limitations=['This conditioning matches the lighthouse voltage branch, not a verified replay of historical detection.','Negative-only detection held fixed across thresholds.','Morphology flags require review; synchronous neural events can be flagged.','Coincidence is not one-to-one identity validation; localized steps are provisional.','No motion rerun is justified solely by removing more peaks.']);(OUT/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
