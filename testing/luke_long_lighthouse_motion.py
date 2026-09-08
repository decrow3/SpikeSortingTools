"""100s matched motion inputs and all nine existing lighthouse trajectories."""
import json
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
OUT=ROOT/'testing/outputs/luke_long_lighthouse_motion_v1';SRC=ROOT/'testing/outputs'
def screen(clean,post,p,geo,noise,fs):
 keep=np.zeros(len(p),bool);off=np.arange(-30,31);mid=abs(off)<=15
 for ch in range(384):
  ids=np.flatnonzero((p['channel_index']==ch)&(p['sample_index']>40)&(p['sample_index']<len(clean)-40))
  if not len(ids):continue
  ev=p['sample_index'][ids];channels=np.flatnonzero(abs(geo[:,1]-geo[ch,1])<=60);ci=np.flatnonzero(channels==ch)[0];w=clean[ev[:,None,None]+off[None,:,None],channels[None,None,:]];v=w[:,:,ci];original=post[ev[:,None]+off,ch];pred=original-v
  explained=1-np.sum(v*v,axis=1)/(np.sum(original**2,axis=1)+1e-12);sharedcos=np.sum(original*pred,axis=1)/(np.linalg.norm(original,axis=1)*np.linalg.norm(pred,axis=1)+1e-12)
  amp=np.max(abs(v),axis=1);concentration=np.sum(v[:,mid]**2,axis=1)/(np.sum(v*v,axis=1)+1e-12);other=np.max(abs(w),axis=1);other[:,ci]=0
  nc=np.einsum('ntc,nt->nc',w,v)/(np.linalg.norm(w,axis=1)*np.linalg.norm(v,axis=1)[:,None]+1e-12);coherent=((other>=.25*amp[:,None])&(other/noise[channels]>=4)&(nc>=.8)).any(axis=1)
  dominant=np.argmax(abs(v),axis=1);width=np.zeros(len(ids));broad=np.zeros(len(ids))
  for j,k in enumerate(dominant):
   sign=np.sign(v[j,k]);lo=hi=int(k)
   while lo>0 and sign*v[j,lo-1]>=.5*amp[j]:lo-=1
   while hi<60 and sign*v[j,hi+1]>=.5*amp[j]:hi+=1
   width[j]=(hi-lo+1)/fs*1000
  for lo in range(0,len(ids),100):
   sl=slice(lo,min(lo+100,len(ids)));centers=ev[sl]+off[dominant[sl]];full=clean[centers[:,None]+np.arange(-3,4)];broad[sl]=np.mean(np.max(abs(full),axis=1)/noise>=4,axis=1)
  keep[ids]=(amp/noise[ch]>=8)&(concentration>=.65)&coherent&(width>=.067)&(width<=.8)&(broad<.2)&~((explained>=.5)&(sharedcos>=.8))
 return keep

def main():
 OUT.mkdir(exist_ok=False);m=json.loads((BASE/'recording/rescue_recording_manifest.json').read_text());fs=m['sampling_frequency_hz'];geo=np.asarray(m['channel_locations_um']);model=np.load(SRC/'luke_common_event_screen_v1/shared_response_model.npz');noise=np.load(SRC/'luke_peak_threshold_screen_v1/noise_uv.npy');cfg=json.loads((SRC/'luke_compensated_dredge_trial_v1/settings.json').read_text())['estimator']
 settings=dict(interval_s=[4160,4260],chunks_s=[4160,4180,4200,4220,4240],estimator=cfg,strict_search_um=80,shared_model_refit=False,screen='Unchanged v2 event rules, advisory family fitting omitted; exact4180mask regression checked',reuse='4180 original/compensated peaks and locations and4240 original/compensated arrays reused exactly',lighthouses='Existing nine gentle tracks, 10s centroids with bootstrap intervals; independent of estimator and screen',limitations='Fixed-template centroids can underreport motion; sparse/missing bins shown.100s motion diagnostic, not amplitude completeness or full-session validation.',resume='Chunk arrays persist; no automatic resume or within-stage checkpoint. No sort.')
 (OUT/'settings.json').write_text(json.dumps(settings,indent=2));allp={a:[] for a in ['original','compensated','screened']};ally={a:[] for a in allp};firstbase=round(4160*fs)
 for start in settings['chunks_s']:
  first=round(start*fs);n=round(20*fs);pad=round(.05*fs)
  with (BASE/'recording/traces_cached_seg0.raw').open('rb') as f:f.seek((first-pad)*768);buf=f.read((n+2*pad)*768)
  x=np.frombuffer(buf,dtype='<i2').reshape(-1,384).astype('float32')*m['gain_uv_per_count'];del buf
  x=sosfiltfilt(butter(3,[300,6000],fs=fs,btype='bandpass',output='sos'),x,axis=0).astype('float32');ref=np.median(x,axis=1);clean=np.empty((n,384),dtype='float32')
  for lo in range(0,n,10000):
   e=np.arange(lo,min(n,lo+10000));clean[e]=x[pad+e]-ref[pad+e[:,None]+model['lag_samples']]@model['coefficients']
  post=x[pad:pad+n]-ref[pad:pad+n,None];del x,ref
  for arm in ['original','compensated']:
   if start==4180:
    prefix=SRC/('luke_peak_threshold_screen_v1' if arm=='original' else 'luke_compensated_peak_trial_v1');p=np.load(prefix/('peaks_5sigma.npy' if arm=='original' else 'peaks.npy'));y=np.load(prefix/('locations_5sigma.npy' if arm=='original' else 'locations.npy'))
   elif start==4240:
    prefix=SRC/'luke_compensation_validation_v1';p=np.load(prefix/f'{arm}_peaks.npy');y=np.load(prefix/f'{arm}_locations.npy')
   else:
    record=NumpyRecording(post if arm=='original' else clean,fs);record.set_channel_locations(geo);p=detect_peaks(record,method='locally_exclusive',peak_sign='neg',radius_um=50.,detect_threshold=5.,noise_levels=noise,n_jobs=1,chunk_duration='1s',progress_bar=False);p=p[(p['sample_index']>=50)&(p['sample_index']<n-50)]
    print(start,arm,'localizing',len(p),flush=True);y=localize_peaks(record,p,method='monopolar_triangulation',radius_um=75.,n_jobs=8,mp_context='fork',chunk_duration='1s',progress_bar=False)
   np.save(OUT/f's{start}_{arm}_peaks.npy',p);np.save(OUT/f's{start}_{arm}_locations.npy',y);pp=p.copy();pp['sample_index']+=first-firstbase;allp[arm].append(pp);ally[arm].append(y)
   if arm=='compensated':
    k=screen(clean,post,p,geo,model['residual_noise_uv'],fs)
    if start==4180:assert np.array_equal(k,np.load(SRC/'luke_motion_waveform_screen_v2/keep_mask.npy'))
    np.save(OUT/f's{start}_keep_mask.npy',k);allp['screened'].append(pp[k]);ally['screened'].append(y[k]);print(start,'screen retained',int(k.sum()),flush=True)
  del clean,post
 fields={}
 record=NumpyRecording(np.broadcast_to(np.zeros((1,384),dtype='float32'),(round(100*fs),384)),fs);record.set_channel_locations(geo)
 for arm in allp:
  p=np.concatenate(allp[arm]);y=np.concatenate(ally[arm]);assert np.all(np.diff(p['sample_index'])>=0);np.save(OUT/f'{arm}_peaks.npy',p);np.save(OUT/f'{arm}_locations.npy',y)
  motion,extra=estimate_bounded(record,p,y,cfg);t=motion.temporal_bins_s[0]+4160;dep=motion.spatial_bins_um;a=motion.displacement[0];np.savez_compressed(OUT/f'{arm}_motion.npz',time_s=t,depth_um=dep,displacement_um=a,**extra);fields[arm]=(t,dep,a);print(arm,'100s motion complete',flush=True)
 tracks=pd.read_csv(SRC/'luke_lighthouse_gentle_v1/gentle_tracks.csv');fig,axs=plt.subplots(3,3,figsize=(16,11),sharex=True,layout='constrained');rows=[];colors={'original':'gray','compensated':'#2878b5','screened':'#b13775'}
 for ax,(u,g) in zip(axs.ravel(),tracks.groupby('unit_id')):
  g=g.sort_values('time_s');depth=float(g.depth_um.iloc[0]);base=float(g.iloc[0].median_waveform_centroid_um)
  for arm,(t,dep,a) in fields.items():
   v=np.array([np.interp(depth,dep,w) for w in a]);v-=np.median(v[(t>=4160)&(t<4170)]);ax.plot(t,v,color=colors[arm],label=arm,alpha=.85)
   for _,q in g.iterrows():
    vbin=v[(t>=q.time_s-5)&(t<q.time_s+5)]
    rows.append(dict(arm=arm,unit_id=int(u),depth_um=depth,time_s=float(q.time_s),accepted_events=int(q.accepted_events),motion_bin_median_um=float(np.median(vbin)),centroid_change_um=float(q.median_waveform_centroid_um-base)))
  good=g.accepted_events>=10;ax.errorbar(g.time_s[good],g.median_waveform_centroid_um[good]-base,yerr=np.array([g.median_waveform_centroid_um[good]-g.bootstrap_low_um[good],g.bootstrap_high_um[good]-g.median_waveform_centroid_um[good]]),fmt='ko',ms=4,capsize=2,label='Lighthouse (10s)');ax.scatter(g.time_s[~good],g.median_waveform_centroid_um[~good]-base,marker='x',color='black',label='<10 events');ax.axhline(0,color='gray',lw=.4);ax.set(title=f'Unit{u} · {depth:.0f}µm',xlabel='Recording time (s)',ylabel='Relative displacement (µm)')
 axs[0,0].legend(fontsize=7);fig.suptitle('100-second motion estimates with nine provisional lighthouse tracks\nEach trace offset to its first10s only; no fitted gain or sign. Centroid bars are bootstrap uncertainty, not identity uncertainty.')
 for ext in ['png','pdf']:fig.savefig(OUT/f'01_lighthouse_overlay.{ext}',dpi=140)
 pd.DataFrame(rows).to_csv(OUT/'lighthouse_binned_comparison.csv',index=False)
 fig,axs=plt.subplots(3,1,figsize=(13,10),sharex=True,layout='constrained');lim=max(np.quantile(abs(a-np.median(a[t<4170],axis=0)),.99) for t,dep,a in fields.values())
 for ax,(arm,(t,dep,a)) in zip(axs,fields.items()):
  z=a-np.median(a[t<4170],axis=0);im=ax.imshow(z.T,origin='lower',aspect='auto',extent=[4160,4260,dep[0],dep[-1]],cmap='RdBu_r',vmin=-lim,vmax=lim);ax.set(title=arm,ylabel='Depth (µm)',xlabel='Recording time (s)');fig.colorbar(im,ax=ax,label='Displacement (µm)')
 fig.suptitle('Full-depth DREDGE fields · identical settings and common color scale\nOffset by first10s at each depth; no motion applied to voltage')
 for ext in ['png','pdf']:fig.savefig(OUT/f'02_depth_fields.{ext}',dpi=140)
 result=dict(status='complete',counts={arm:sum(map(len,pp)) for arm,pp in allp.items()},central_screen_exact_match=True,scope='100s motion and lighthouse comparison only');(OUT/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
