"""Bounded early transfer of frozen unit445 matcher, with cached DREDGE overlay."""
import json, time, hashlib
import numpy as np
import pandas as pd
from scipy.signal import butter,sosfiltfilt,find_peaks
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from testing.luke_epoch_corroboration import ROOT,BASE
from testing.luke_multidepth_anchors_v2 import scores
from testing.luke_early_lighthouse_binning_v2 import groups,centroid
SRC=ROOT/'testing/outputs'; OUT=SRC/'luke_early_unit445_v1'; OLD=SRC/'luke_lighthouse_gentle_v1'
def main():
 OUT.mkdir(exist_ok=False);begun=time.monotonic()
 meta=BASE/'recording/rescue_recording_manifest.json';m=json.loads(meta.read_text());fs=m['sampling_frequency_hz'];geo=np.asarray(m['channel_locations_um']);raw=BASE/'recording/traces_cached_seg0.raw';stat=raw.stat();sos=butter(3,[300,6000],fs=fs,btype='bandpass',output='sos')
 inputs=[meta,OLD/'templates.npz',OLD/'identity_validation.csv',OLD/'neighbor_template_similarity.csv',SRC/'luke_early_screen_shortlist_v1/fields/compensated.npz',__import__('pathlib').Path(__file__)]
 settings=dict(unit_id=445,nominal_depth_um=2260,interval_s=[930,1030],template_training_s=[4080,4090],threshold=.8850000000000001,method='Frozen gentle matcher: original 300–6000Hz/global median referenced voltage; ±20um detection; ±60um template support; timing±3 samples; gain .4–2.5; competitor margin .03; .8ms NMS. No DREDGE-guided selection.',resume='No automatic resume; saved completed chunk waveforms can be reused by an explicitly reviewed restart. Fresh output required.',limitations='Cross-epoch identity and displacement sensitivity unqualified; fixed support can preferentially accept small movements.',source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs})
 (OUT/'settings.json').write_text(json.dumps(settings,indent=2))
 def read(start,duration):
  first=round(start*fs);n=round(duration*fs);pad=round(.05*fs)
  with raw.open('rb') as f:f.seek((first-pad)*768);buf=f.read((n+2*pad)*768)
  assert len(buf)==(n+2*pad)*768
  x=np.frombuffer(buf,dtype='<i2').reshape(-1,384).astype('float32')*m['gain_uv_per_count'];x=sosfiltfilt(sos,x,axis=0).astype('float32');x-=np.median(x,axis=1,keepdims=True)
  return x[pad:pad+n],first
 def sigma(x):return np.median(abs(x-np.median(x,axis=0)),axis=0)/.67448975
 z=np.load(OLD/'templates.npz');v=z['unit_445_template'];channels=z['unit_445_channels'];depths=geo[channels,1]
 train,first=read(4080,10);noise=sigma(train);weights=v*v/(v*v+(2*noise[channels])**2+1e-20)
 ts=np.load(BASE/'cur/cur_output/spike_times.npy',mmap_mode='r').ravel();ids=np.load(BASE/'cur/cur_output/spike_clusters.npy',mmap_mode='r').ravel();ix=np.flatnonzero((ts>=first+40)&(ts<first+len(train)-40));tt=np.asarray(ts[ix])-first;cc=np.asarray(ids[ix]);templates={}
 neighbors=pd.read_csv(OLD/'neighbor_template_similarity.csv');neighbors=neighbors[(neighbors.unit_id==445)&(neighbors.weighted_template_cosine>=.7)]
 for cid in [445]+neighbors.neighbor_id.astype(int).tolist():
  ev=tt[cc==cid];ev=ev[np.linspace(0,len(ev)-1,min(len(ev),100),dtype=int)];templates[cid]=np.median(train[ev[:,None]+np.arange(-30,31)],axis=0)
 assert np.array_equal(templates[445][:,channels],v)
 peak=int(abs(templates[445]).max(axis=0).argmax());offset=int(abs(templates[445][:,peak]).argmax())-30;del train
 frames_all=[];waves_all=[];events=[];counts=[]
 for start in range(930,1030,10):
  x,first=read(start,10);noise=sigma(x);ev=[]
  for ch in np.flatnonzero(abs(geo[:,1]-2260)<=20):
   p=find_peaks(abs(x[:,ch]),height=max(30,3*noise[ch]),distance=round(.0008*fs))[0]-offset;ev.extend(p[(p>40)&(p<len(x)-40)])
  ev=np.unique(ev);sc=scores(x,ev,channels,v,weights);rival=np.full(len(ev),-1.)
  for cid,w in templates.items():
   if cid!=445:rival=np.maximum(rival,scores(x,ev,channels,w[:,channels],weights)[:,0])
  good=(sc[:,0]>=settings['threshold'])&(sc[:,1]>=.4)&(sc[:,1]<=2.5)&(sc[:,0]-rival>=.03);kept=[]
  for i in np.argsort(-sc[:,0]):
   if good[i] and all(abs(sc[i,2]-sc[j,2])>.0008*fs for j in kept):kept.append(i)
  keep=np.asarray(sorted(kept,key=lambda i:sc[i,2]),dtype=int);fr=sc[keep,2].astype(int);waves=x[fr[:,None,None]+np.arange(-30,31)[None,:,None],channels[None,None,:]]
  np.savez_compressed(OUT/f's{start}_waveforms.npz',frames=fr+first,waveforms=waves,channels=channels)
  np.savez_compressed(OUT/f's{start}_decisions.npz',scores=sc,rival=rival,accepted_indices=keep,first_frame=first)
  frames_all.extend(fr+first);waves_all.append(waves)
  events.extend(dict(frame=int(first+sc[i,2]),score=float(sc[i,0]),gain=float(sc[i,1]),rival_score=float(rival[i])) for i in keep)
  counts.append(dict(start_s=start,detected=len(ev),accepted=len(keep)));pd.DataFrame(counts).to_csv(OUT/'chunk_counts.csv',index=False)
  print(start,len(keep),'accepted',flush=True);del x
 frames=np.asarray(frames_all,dtype=np.int64);waves=np.concatenate(waves_all);t=frames/fs
 np.savez_compressed(OUT/'events_unit445.npz',frames=frames,waveforms=waves,channels=channels);pd.DataFrame(events).to_csv(OUT/'events.csv',index=False)
 energy=np.sum(waves.astype(float)**2,axis=1);y=energy@depths/energy.sum(axis=1);pd.DataFrame(dict(time_s=t,event_centroid_um=y)).to_csv(OUT/'event_centroids.csv',index=False)
 rows=[];rng=np.random.default_rng(445)
 for mode in ['adaptive','100spikes']:
  for a,b,ix in groups(t,mode):
   supported=len(ix)>=(100 if mode=='100spikes' else 10) and (b-a<=20) and (len(ix)<2 or np.diff(t[ix]).max()<=2)
   row=dict(mode=mode,start_s=a,end_s=b,time_s=float(np.median(t[ix])) if len(ix) else (a+b)/2,events=len(ix),supported=supported,centroid_um=np.nan,low_um=np.nan,high_um=np.nan)
   if supported:
    w=waves[ix];row['centroid_um']=centroid(w,depths);boot=[centroid(w[rng.integers(0,len(ix),len(ix))],depths) for _ in range(100)];row['low_um'],row['high_um']=np.quantile(boot,[.025,.975])
   rows.append(row)
 table=pd.DataFrame(rows);table.to_csv(OUT/'windows.csv',index=False)
 field=np.load(SRC/'luke_early_screen_shortlist_v1/fields/compensated.npz');ft=field['time_s'];motion=np.array([np.interp(2260,field['depth_um'],d) for d in field['displacement_um']])
 ref=(t>=970)&(t<975)
 if ref.sum()>=10:anchor=centroid(waves[ref],depths);offset=float(np.median(np.interp(t[ref],ft,motion)));reference='970–975 s accepted events'
 elif len(t)>=10:anchor=centroid(waves,depths);offset=float(np.median(np.interp(t,ft,motion)));reference='all accepted events in 930–1030 s (970–975 s lacks support)'
 else:anchor=2260.;offset=float(np.median(motion));reference='nominal 2260 µm + interval-median-centered DREDGE; insufficient lighthouse support'
 trajectory=anchor+motion-offset;pd.DataFrame(dict(time_s=ft,dredge_displacement_um=motion,display_depth_um=trajectory)).to_csv(OUT/'dredge_trace.csv',index=False)
 hist=np.load(SRC/'luke_peak_population_review_v1/02_long_histograms_histograms.npz');arrays=[np.log1p(hist[f'arm{i}_counts']) for i in range(2)];vmax=np.quantile(np.concatenate([a.ravel() for a in arrays]),.995)
 fig,axes=plt.subplots(2,2,figsize=(15,9),sharex=True,layout='constrained')
 for col,arr in enumerate(arrays):
  for row in range(2):
   ax=axes[row,col];im=ax.imshow(arr,origin='lower',extent=[930,1030,0,3840],aspect='auto',interpolation='nearest',cmap='magma',vmin=0,vmax=vmax)
   ax.scatter(t,y,s=7,c='#56DDE0',alpha=.35,linewidths=0,label='Accepted-event energy centroid')
   q=table[(table['mode']=='100spikes')&table.supported]
   if not len(q):q=table[(table['mode']=='adaptive')&table.supported];winlabel='Adaptive ≥10-spike median-waveform centroid'
   else:winlabel='100-spike median-waveform centroid'
   if len(q):
    ax.errorbar(q.time_s,q.centroid_um,xerr=np.array([q.time_s-q.start_s,q.end_s-q.time_s]),fmt='D',ms=4,mfc='white',mec='#006F80',ecolor='white',elinewidth=.8,label=winlabel,zorder=4);ax.vlines(q.time_s,q.low_um,q.high_um,colors='white',lw=.8)
   ax.plot(ft,trajectory,color='#00FF66',lw=1.8,ls='--',label='Compensated 5σ DREDGE at 2260 µm',zorder=5)
   ax.set(xlim=(930,1030),ylim=(1900,2400) if row==0 else (anchor-65,anchor+65),ylabel='Depth (µm)',title=('Original' if col==0 else 'Compensated')+' input · '+('regional view' if row==0 else 'unit 445 zoom'))
   ax.ticklabel_format(axis='x',style='plain',useOffset=False)
 axes[0,0].legend(fontsize=8,loc='lower left');fig.colorbar(im,ax=axes,label='log(1 + peak count)');axes[1,0].set_xlabel('Recording time (s)');axes[1,1].set_xlabel('Recording time (s)')
 fig.suptitle(f'Unit 445 template transfer to 930–1,030 s · {len(t):,} accepted matches\nFrozen matcher; early identity unqualified · 0.25 s × 10 µm background bins',fontsize=14)
 fig.supxlabel('DREDGE is vertically anchored to '+reference+'. No sign, gain, or lag fitting.\nCentroids are descriptive, with fixed-support selection bias; the dashed trace is inferred motion, not a measured cell trajectory.',fontsize=9)
 for ext in ['png','pdf']:fig.savefig(OUT/f'01_depth_time_dredge.{ext}',dpi=180)
 plt.close(fig)
 assert (raw.stat().st_size,raw.stat().st_mtime_ns)==(stat.st_size,stat.st_mtime_ns)
 (OUT/'summary.json').write_text(json.dumps(dict(status='complete',accepted_events=len(t),window_support=table.groupby('mode').supported.sum().astype(int).to_dict(),reference=reference,anchor_um=anchor,seconds=time.monotonic()-begun,identity='Early transfer unqualified; matches do not certify same neuron.'),indent=2))
if __name__=='__main__':main()
