"""Local identity controls followed by a bounded gentle-epoch footprint audit."""
from testing.luke_multidepth_anchors_v2 import scores,near
from testing.luke_epoch_corroboration import BASE,ROOT,SHARED,DATA
import json
import numpy as np
import pandas as pd
from scipy.signal import butter,sosfiltfilt,find_peaks
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=ROOT/'testing/outputs/luke_lighthouse_gentle_v1'
def main():
 OUT.mkdir(exist_ok=False)
 config=dict(train_s=[4080,4090],calibration_s=[4110,4130],validation_s=[4130,4150],target_s=[4160,4260],target_bin_s=10,channel_neighborhood_um=60,detection_neighborhood_um=20,gain_range=[.4,2.5],minimum_reference_events=15,minimum_recall=.5,maximum_unassigned_fraction=.1,competitor_margin=.03,score_floor=.8,method='Independent peaks on channels within 20um of template peak; fixed local spatial support; ±3 sample alignment. Nearby training templates compete under identical anchor weights. No spatial translation or motion-field-guided selection.',limitations=['Reference labels are provisional, not identity ground truth.','Fixed-support matching may preferentially retain stationary or smaller-displacement events.','Footprint centroid is descriptive; bootstrap uncertainty is conditional and not calibrated motion accuracy.'])
 (OUT/'settings.json').write_text(json.dumps(config,indent=2))
 rec=json.loads((BASE/'recording/rescue_recording_manifest.json').read_text());fs=rec['sampling_frequency_hz'];loc=np.asarray(rec['channel_locations_um']);sos=butter(3,[300,6000],fs=fs,btype='bandpass',output='sos');ts=np.load(BASE/'cur/cur_output/spike_times.npy',mmap_mode='r').ravel();cl=np.load(BASE/'cur/cur_output/spike_clusters.npy',mmap_mode='r').ravel();rng=np.random.default_rng(20260908);raw=BASE/'recording/traces_cached_seg0.raw';stat=raw.stat();bytes_read=0
 def read(start,duration):
  nonlocal bytes_read
  first=round(start*fs);n=round(duration*fs);pad=round(.05*fs)
  with raw.open('rb') as f:f.seek((first-pad)*768);buf=f.read((n+2*pad)*768)
  assert len(buf)==(n+2*pad)*768;bytes_read+=len(buf)
  x=np.frombuffer(buf,dtype='<i2').reshape(-1,384).astype('float32')*rec['gain_uv_per_count'];x=sosfiltfilt(sos,x,axis=0).astype('float32');x-=np.median(x,axis=1,keepdims=True);return x[pad:pad+n],first
 def refs(cid,first,n):
  mask=(ts>=first+40)&(ts<first+n-40);idx=np.flatnonzero(mask);return np.asarray(ts[idx][cl[idx]==cid])-first
 def sigma(x):return np.median(abs(x-np.median(x,axis=0)),axis=0)/.67448975
 train,first=read(4080,10);sig=sigma(train);idx=np.flatnonzero((ts>=first+40)&(ts<first+len(train)-40));tt=np.asarray(ts[idx])-first;ids=np.asarray(cl[idx]);templates={};depths={}
 for cid in np.unique(ids):
  ev=tt[ids==cid]
  if len(ev)<10:continue
  ev=ev[np.linspace(0,len(ev)-1,min(len(ev),100),dtype=int)];w=np.median(train[ev[:,None]+np.arange(-30,31)],axis=0);templates[int(cid)]=w;depths[int(cid)]=loc[np.max(abs(w),axis=0).argmax(),1]
 selected=pd.read_csv(ROOT/'testing/outputs/luke_lighthouse_top20_v1/top20_ranked.csv');candidates=[];pairs=[];saved={}
 for row in selected.itertuples():
  cid=int(row.unit_id);w=templates[cid];peak=int(np.max(abs(w),axis=0).argmax());depth=loc[peak,1];chs=np.flatnonzero(abs(loc[:,1]-depth)<=60);v=w[:,chs];weights=v*v/(v*v+(2*sig[chs])**2+1e-20);neighbors=[]
  for other,ww in templates.items():
   if other==cid or abs(depths[other]-depth)>120:continue
   vv=ww[:,chs];norm=np.sqrt(np.sum(weights*v*v)*np.sum(weights*vv*vv));co=float(np.sum(weights*v*vv)/(norm+1e-20));pairs.append(dict(unit_id=cid,neighbor_id=other,depth_um=float(depth),neighbor_depth_um=float(depths[other]),weighted_template_cosine=co))
   if co>=.7:neighbors.append((other,vv))
  candidates.append(dict(cid=cid,peak=peak,depth=float(depth),chs=chs,template=v,weights=weights,neighbors=neighbors,offset=int(np.argmax(abs(w[:,peak])))-30));saved[f'unit_{cid}_template']=v;saved[f'unit_{cid}_channels']=chs
 np.savez_compressed(OUT/'templates.npz',**saved);pd.DataFrame(pairs).to_csv(OUT/'neighbor_template_similarity.csv',index=False);del train
 def detect(x,c,noise):
  peaks=[]
  for ch in np.flatnonzero(abs(loc[:,1]-c['depth'])<=20):
   p=find_peaks(abs(x[:,ch]),height=max(30,3*noise[ch]),distance=round(.0008*fs))[0]-c['offset'];peaks.extend(p[(p>40)&(p<len(x)-40)])
  ev=np.unique(peaks);sc=scores(x,ev,c['chs'],c['template'],c['weights']);rival=np.full(len(ev),-1.)
  for other,v in c['neighbors']:
   rs=scores(x,ev,c['chs'],v,c['weights']);rival=np.maximum(rival,rs[:,0])
  eligible=(sc[:,1]>=.4)&(sc[:,1]<=2.5)&(sc[:,0]-rival>=.03)
  # NMS within each identity, not across distant cells that can legitimately fire together.
  kept=[]
  for i in np.argsort(-sc[:,0]):
   if eligible[i] and all(abs(sc[i,2]-sc[j,2])>.0008*fs for j in kept):kept.append(i)
  return sc, np.asarray(kept,dtype=int),rival
 cal,cf=read(4110,20);cs=sigma(cal);diagnostics=[];cal_events=[]
 for c in candidates:
  truth=refs(c['cid'],cf,len(cal));sc,ix,rv=detect(cal,c,cs);threshold=None
  for th in np.arange(.8,.996,.005):
   hit=sc[ix[sc[ix,0]>=th],2];recall=float(near(truth,np.sort(hit),.0005*fs).mean()) if len(truth) else 0;extra=float((~near(hit,truth,.0005*fs)).mean()) if len(hit) else 1
   if len(truth)>=15 and len(hit)>=10 and recall>=.5 and extra<=.1:threshold=float(th);break
  c['threshold']=threshold
  diagnostics.append(dict(unit_id=c['cid'],depth_um=c['depth'],neighbor_templates=len(c['neighbors']),calibration_reference_events=len(truth),threshold=threshold,calibration_pass=threshold is not None))
  cal_events.extend(dict(unit_id=c['cid'],frame=int(cf+sc[i,2]),score=float(sc[i,0]),gain=float(sc[i,1]),rival_score=float(rv[i]),near_reference=bool(near(np.array([sc[i,2]]),truth,.0005*fs)[0])) for i in ix)
 pd.DataFrame(cal_events).to_csv(OUT/'calibration_detections.csv',index=False);del cal
 print('Calibration complete; thresholds frozen',flush=True)
 val,vf=read(4130,20);vs=sigma(val);validated=[];vevents=[]
 for c,row in zip(candidates,diagnostics):
  truth=refs(c['cid'],vf,len(val));sc,ix,rv=detect(val,c,vs);th=c['threshold'];keep=ix[sc[ix,0]>=th] if th is not None else np.array([],dtype=int);hit=np.sort(sc[keep,2]);recall=float(near(truth,hit,.0005*fs).mean()) if len(truth) else 0;extra=float((~near(hit,truth,.0005*fs)).mean()) if len(hit) else 1;passed=th is not None and len(truth)>=15 and len(hit)>=10 and recall>=.5 and extra<=.1
  row.update(validation_reference_events=len(truth),validation_accepted=len(hit),validation_recall=recall,validation_unassigned_fraction=extra,validation_pass=passed)
  if passed:validated.append(c)
  vevents.extend(dict(unit_id=c['cid'],frame=int(vf+sc[i,2]),score=float(sc[i,0]),gain=float(sc[i,1]),rival_score=float(rv[i]),near_reference=bool(near(np.array([sc[i,2]]),truth,.0005*fs)[0])) for i in keep)
 pd.DataFrame(vevents).to_csv(OUT/'validation_detections.csv',index=False);d=pd.DataFrame(diagnostics);d.to_csv(OUT/'identity_validation.csv',index=False);del val
 print(json.dumps({'validation_pass_ids':[c['cid'] for c in validated]}),flush=True)
 def save(fig,name):
  fig.savefig(OUT/(name+'.png'),dpi=160,bbox_inches='tight');fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(11,7),sharey=True);yy=np.arange(len(d));labels=[f'{r.unit_id} ({r.depth_um:.0f} µm)' for r in d.itertuples()];colors=np.where(d.validation_pass,'#167a9a','#aaaaaa');axes[0].barh(yy,d.validation_recall,color=colors);axes[1].barh(yy,d.validation_unassigned_fraction,color=colors);axes[0].axvline(.5,c='k',ls=':');axes[1].axvline(.1,c='k',ls=':');axes[0].set_yticks(yy,labels);axes[0].invert_yaxis();axes[0].set_xlabel('Recall of provisional reference events');axes[1].set_xlabel('Accepted events without reference match');axes[0].set_xlim(0,1);axes[1].set_xlim(0,1);fig.suptitle(f'Local identity controls: {len(validated)}/20 pass separate quiet validation\n4130–4150 s; thresholds frozen from 4110–4130 s; blue = pass both gates');fig.tight_layout();save(fig,'01_identity_controls')
 # Keep multiple depths, but do not over-represent strongly similar co-located templates.
 anchors=[]
 for c in sorted(validated,key=lambda c:c['depth']):
  if all(abs(c['depth']-a['depth'])>=100 for a in anchors):anchors.append(c)
 rows=[];eventrows=[]
 for start in range(4160,4260,10) if anchors else []:
  x,first=read(start,10);noise=sigma(x)
  for c in anchors:
   sc,ix,rv=detect(x,c,noise);keep=ix[sc[ix,0]>=c['threshold']];events=np.sort(sc[keep,2].astype(int));n=len(events);cent=low=high=np.nan
   if n>=10:
    waves=x[events[:,None,None]+np.arange(-30,31)[None,:,None],c['chs'][None,None,:]]
    def center(w):
     energy=np.sum(w*w,axis=0);return float(np.sum(energy*loc[c['chs'],1])/energy.sum())
    cent=center(np.median(waves,axis=0));boot=np.array([center(np.median(waves[rng.integers(0,n,n)],axis=0)) for _ in range(100)]);low,high=np.quantile(boot,[.025,.975])
   rows.append(dict(unit_id=c['cid'],depth_um=c['depth'],time_s=start+5,accepted_events=n,median_waveform_centroid_um=cent,bootstrap_low_um=low,bootstrap_high_um=high))
   eventrows.extend(dict(unit_id=c['cid'],time_s=start+5,frame=int(first+sc[i,2]),score=float(sc[i,0]),gain=float(sc[i,1])) for i in keep)
  print(f'Completed gentle bin {start}–{start+10}',flush=True)
 tracks=pd.DataFrame(rows);tracks.to_csv(OUT/'gentle_tracks.csv',index=False);pd.DataFrame(eventrows).to_csv(OUT/'gentle_events.csv',index=False)
 fields=[];native=np.load(SHARED/'native_rigid_field.npz');nt=native['time_s'];nv=native['physical_displacement_um'].ravel();origin=3057.677050340359
 for name in ['dredge-motion','decentralized-motion','ks-motion']:
  p=DATA/'dredge_pipeline_results_Luke0804_V2V1_g0_imec0/motion'/name;tt=np.load(p/'time_bins.npy').ravel()-origin;dd=np.load(p/'depth_bins.npy').ravel();mm=np.load(p/'motion.npy');mask=(tt>=4160)&(tt<4260);fields.append((name,tt[mask],dd,mm[mask]))
 if anchors:
  fig,axes=plt.subplots(len(anchors),1,figsize=(12,2.6*len(anchors)),sharex=True,squeeze=False);comparisons=[]
  for ax,c in zip(axes[:,0],anchors):
   g=tracks[tracks.unit_id==c['cid']];base=g.iloc[0].median_waveform_centroid_um;ax.plot(g.time_s,g.median_waveform_centroid_um-base,'o-',c='k',label='Waveform centroid');ax.fill_between(g.time_s,g.bootstrap_low_um-base,g.bootstrap_high_um-base,color='k',alpha=.15,label='Conditional bootstrap interval')
   mask=(nt>=4160)&(nt<4260);a=nv[mask];ax.plot(nt[mask],a-np.median(a[nt[mask]<4170]),c='gray',label='Native global rigid',lw=.8)
   for name,tt,dd,mm in fields:
    v=np.array([np.interp(c['depth'],dd,row) for row in mm]);v-=np.median(v[tt<4170]);ax.plot(tt,v,label=name,lw=.8)
    predicted=np.interp(g.time_s,tt,v);actual=g.median_waveform_centroid_um.to_numpy()-base;ok=np.isfinite(actual);comparisons.append(dict(unit_id=c['cid'],method=name,valid_bins=int(ok.sum()),centroid_range_um=float(np.nanmax(actual)-np.nanmin(actual)) if ok.any() else np.nan,estimate_range_um=float(np.ptp(v)),median_absolute_difference_um=float(np.median(abs(actual[ok]-predicted[ok]))) if ok.any() else np.nan))
   ax.set_title(f"Unit {c['cid']} | {c['depth']:.0f} µm | events/bin {g.accepted_events.min()}–{g.accepted_events.max()}");ax.set_ylabel('Relative position (µm)');ax.axhline(0,c='gray',lw=.4)
  axes[0,0].legend(fontsize=7,ncol=3);axes[-1,0].set_xlabel('Seconds from recording frame zero');fig.suptitle('Gentle-epoch comparison — centroid changes are descriptive, not calibrated displacement\nFirst 10 s baseline; no sign/lag fitting; gaps remain unresolved; local matching may favor stationary events',fontsize=11);fig.tight_layout(rect=[0,0,1,.95]);save(fig,'02_gentle_tracks');pd.DataFrame(comparisons).to_csv(OUT/'descriptive_comparisons.csv',index=False)
 assert (raw.stat().st_size,raw.stat().st_mtime_ns)==(stat.st_size,stat.st_mtime_ns)
 result=dict(status='complete',validated_ids=[c['cid'] for c in validated],tracked_ids=[c['cid'] for c in anchors],bytes_read=bytes_read,interpretation='Local-control consistency only. No calibrated motion accuracy or biological efficacy conclusion.',limitations=config['limitations']);(OUT/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
