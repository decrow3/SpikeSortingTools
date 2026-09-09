"""Exploratory depth search for frozen unit445; DREDGE never enters selection."""
import json,time,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import butter,sosfiltfilt,find_peaks
from testing.luke_epoch_corroboration import ROOT,BASE
from testing.luke_multidepth_anchors_v2 import scores
SRC=ROOT/'testing/outputs'; OUT=SRC/'luke_early_unit445_depth_v2'; SHIFTS=np.arange(-120,121,40)
def main():
 OUT.mkdir(exist_ok=False);begun=time.monotonic();meta=BASE/'recording/rescue_recording_manifest.json';m=json.loads(meta.read_text());fs=m['sampling_frequency_hz'];geo=np.asarray(m['channel_locations_um']);mapping={tuple(g):i for i,g in enumerate(geo)};raw=BASE/'recording/traces_cached_seg0.raw';before=raw.stat();sos=butter(3,[300,6000],fs=fs,btype='bandpass',output='sos')
 settings=dict(interval_s=[930,1030],template_s=[4080,4090],unit=445,shift_um=SHIFTS.tolist(),threshold=.885,gain=[.4,2.5],identity_margin=.03,shift_margin=.03,method='Original 300–6000 Hz/global median reference. Exact geometry translations; target context weights shared by every rival and rival shift. No DREDGE selection or continuity prior.',qualification='Exploratory search expansion; historical threshold not recalibrated for search multiplicity. Coarse shift is a hypothesis, not calibrated physical displacement.',restart='Fresh directory required. Chunk decisions and accepted waveforms persist; incomplete chunk must restart; no automatic resume.',script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
 (OUT/'settings.json').write_text(json.dumps(settings,indent=2))
 def read(start,duration):
  first=round(start*fs);n=round(duration*fs);pad=round(.05*fs)
  with raw.open('rb') as f:f.seek((first-pad)*768);buf=f.read((n+2*pad)*768)
  assert len(buf)==(n+2*pad)*768
  x=np.frombuffer(buf,dtype='<i2').reshape(-1,384).astype('float32')*m['gain_uv_per_count'];x=sosfiltfilt(sos,x,axis=0).astype('float32');x-=np.median(x,axis=1,keepdims=True);return x[pad:pad+n],first
 def sigma(x):return np.median(abs(x-np.median(x,axis=0)),axis=0)/.67448975
 z=np.load(SRC/'luke_lighthouse_gentle_v1/templates.npz');v=z['unit_445_template'];channels=z['unit_445_channels'];mapped=np.array([[mapping[(geo[c,0],geo[c,1]+s)] for c in channels] for s in SHIFTS]);assert np.all(geo[mapped,0]==geo[channels,0]);assert np.all(geo[mapped,1]-geo[channels,1]==SHIFTS[:,None])
 # Exact translated synthetic waveforms must select their planted signed shift.
 train,first=read(4080,10);noise=sigma(train);weights=v*v/(v*v+(2*noise[channels])**2+1e-20)
 checks=[]
 for j,s in enumerate(SHIFTS):
  synthetic=np.zeros((101,384),dtype=np.float32);synthetic[20:81,mapped[j]]=v
  ss=np.array([scores(synthetic,[50],c,v,weights)[0,0] for c in mapped]);assert SHIFTS[ss.argmax()]==s
  checks.append(dict(planted_um=int(s),recovered_um=int(SHIFTS[ss.argmax()]),score=float(ss.max())))
 (OUT/'geometry_validation.json').write_text(json.dumps(checks,indent=2))
 ts=np.load(BASE/'cur/cur_output/spike_times.npy',mmap_mode='r').ravel();ids=np.load(BASE/'cur/cur_output/spike_clusters.npy',mmap_mode='r').ravel();ix=np.flatnonzero((ts>=first+40)&(ts<first+len(train)-40));tt=np.asarray(ts[ix])-first;cc=np.asarray(ids[ix]);templates={};inventory=[]
 for cid in np.unique(cc):
  ev=tt[cc==cid];ev=ev[np.linspace(0,len(ev)-1,min(len(ev),100),dtype=int)];w=np.median(train[ev[:,None]+np.arange(-30,31)],axis=0);peak=int(abs(w).max(axis=0).argmax());depth=geo[peak,1];include=1960<=depth<=2560
  inventory.append(dict(unit_id=int(cid),training_events=int((cc==cid).sum()),peak_depth_um=depth,included=include,reason='search region plus support plus shift' if include else 'peak outside bounded competitor inventory'))
  if include:templates[int(cid)]=w
 pd.DataFrame(inventory).to_csv(OUT/'competitor_inventory.csv',index=False);assert np.array_equal(templates[445][:,channels],v);peak=int(abs(templates[445]).max(axis=0).argmax());offset=int(abs(templates[445][:,peak]).argmax())-30
 # Rival waveform on each target context, all shift alternatives; frozen target weights.
 contexts=[]
 for dest in mapped:
  bank=[]
  for cid,w in templates.items():
   if cid==445:continue
   for shift in SHIFTS:
    source=np.array([mapping.get((geo[c,0],geo[c,1]-shift),-1) for c in dest]);assert np.all(source>=0)
    rival=w[:,source]
    if np.sum(weights*rival*rival)>1e-12:bank.append((cid,int(shift),rival))
  contexts.append(bank)
 del train
 accepted=[];counts=[];wavechunks=[];framechunks=[];shiftchunks=[]
 for start in range(930,1030,10):
  x,first=read(start,10);noise=sigma(x);ev=[]
  for ch in np.flatnonzero(abs(geo[:,1]-2260)<=140):
   p=find_peaks(abs(x[:,ch]),height=max(30,3*noise[ch]),distance=round(.0008*fs))[0]-offset;ev.extend(p[(p>40)&(p<len(x)-40)])
  ev=np.unique(ev);sc=np.array([scores(x,ev,c,v,weights) for c in mapped]);eligible=(sc[:,:,1]>=.4)&(sc[:,:,1]<=2.5);ss=np.where(eligible,sc[:,:,0],-1);winner=ss.argmax(axis=0);best=ss[winner,np.arange(len(ev))];second=np.sort(ss,axis=0)[-2];rscore=np.full(len(ev),-1.);rid=np.full(len(ev),-1);rshift=np.zeros(len(ev),int)
  for j in range(len(SHIFTS)):
   ii=np.flatnonzero((winner==j)&(best>=.885))
   if not len(ii):continue
   for cid,shift,w in contexts[j]:
    sr=scores(x,ev[ii],mapped[j],w,weights);val=np.where((sr[:,1]>=.4)&(sr[:,1]<=2.5),sr[:,0],-1);better=val>rscore[ii];rscore[ii[better]]=val[better];rid[ii[better]]=cid;rshift[ii[better]]=shift
  status=np.full(len(ev),'below_threshold',dtype='<U24');passed=best>=.885;status[passed]='ambiguous_shift';unique=passed&(best-second>=.03);status[unique]='identity_ambiguous';good=unique&(best-rscore>=.03);status[good]='accepted';kept=[]
  for i in np.argsort(-best):
   if not good[i]:continue
   frame=sc[winner[i],i,2]
   if any(abs(frame-sc[winner[k],k,2])<=.0008*fs for k in kept):status[i]='duplicate';continue
   kept.append(i)
  keep=np.asarray(sorted(kept,key=lambda i:sc[winner[i],i,2]),int);fr=sc[winner[keep],keep,2].astype(int);sh=SHIFTS[winner[keep]];ch=mapped[winner[keep]];waves=x[fr[:,None,None]+np.arange(-30,31)[None,:,None],ch[:,None,:]];energy=np.sum(waves.astype(float)**2,axis=1);cent=(energy*geo[ch,1]).sum(axis=1)/energy.sum(axis=1)
  np.savez_compressed(OUT/f's{start}_decisions.npz',event_frames=ev+first,scores=sc,shift_grid_um=SHIFTS,winner=winner,second_score=second,rival_score=rscore,rival_id=rid,rival_shift_um=rshift,status=status,accepted_indices=keep)
  np.savez_compressed(OUT/f's{start}_waveforms.npz',frames=fr+first,waveforms=waves,channels=ch,shift_um=sh)
  for k,i in enumerate(keep):accepted.append(dict(frame=int(fr[k]+first),time_s=(fr[k]+first)/fs,shift_um=int(sh[k]),boundary=abs(sh[k])==120,score=best[i],shift_margin=best[i]-second[i],rival_id=rid[i],rival_shift_um=rshift[i],identity_margin=best[i]-rscore[i],centroid_um=cent[k]))
  framechunks.append(fr+first);wavechunks.append(waves);shiftchunks.append(sh);counts.append(dict(start_s=start,detected=len(ev),**{str(k):int(n) for k,n in zip(*np.unique(status,return_counts=True))}));pd.DataFrame(counts).fillna(0).to_csv(OUT/'chunk_counts.csv',index=False);print(start,counts[-1],flush=True)
  del x
 pd.DataFrame(accepted).to_csv(OUT/'events.csv',index=False);np.savez_compressed(OUT/'events_unit445.npz',frames=np.concatenate(framechunks),waveforms=np.concatenate(wavechunks),shift_um=np.concatenate(shiftchunks),base_channels=channels)
 assert (raw.stat().st_size,raw.stat().st_mtime_ns)==(before.st_size,before.st_mtime_ns)
 summary=dict(status='complete',accepted_events=len(accepted),competitor_templates=len(templates)-1,shift_counts=pd.Series([r['shift_um'] for r in accepted]).value_counts().to_dict(),seconds=time.monotonic()-begun,limitations=settings['qualification']);(OUT/'summary.json').write_text(json.dumps(summary,indent=2));print(summary,flush=True)
 from testing.luke_early_unit445_depth_report_v2 import main as report
 report()
if __name__=='__main__':main()
