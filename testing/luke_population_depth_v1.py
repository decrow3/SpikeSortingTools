"""Exploratory whole-probe lighthouse cohort; labels seed templates, never target matches.
Completed template/chunk stages are hash-validated and reused. Interrupted chunk restarts.
"""
import hashlib,json,os,time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import butter,sosfiltfilt,find_peaks
from testing.luke_epoch_corroboration import ROOT,BASE
from testing.luke_multidepth_anchors_v2 import scores
OUT=ROOT/'testing/outputs/luke_population_depth_v1'
SHIFTS=np.arange(-120,121,40)
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atomic_json(p,obj):
 q=p.with_suffix('.tmp');q.write_text(json.dumps(obj,indent=2));os.replace(q,p)
def seal(stage,files):atomic_json(OUT/(stage+'.complete.json'),{str(p.relative_to(OUT)):digest(p) for p in files})
def valid(stage):
 p=OUT/(stage+'.complete.json')
 if not p.exists():return False
 for name,h in json.loads(p.read_text()).items():
  if digest(OUT/name)!=h:raise RuntimeError('Checkpoint hash mismatch: '+name)
 return True
def main():
 OUT.mkdir(exist_ok=True);begin=time.monotonic();m=json.loads((BASE/'recording/rescue_recording_manifest.json').read_text());fs=m['sampling_frequency_hz'];geo=np.array(m['channel_locations_um']);mapping={tuple(g):i for i,g in enumerate(geo)};raw=BASE/'recording/traces_cached_seg0.raw';st=raw.stat();sos=butter(3,[300,6000],fs=fs,btype='bandpass',output='sos')
 config=dict(interval_s=[930,1030],training_segments_s=[[930,940]],inventory_check_segment_s=[980,990],targets=30,depth_strata=10,shifts_um=SHIFTS.tolist(),score_threshold=.86,gain=[.35,3.0],identity_margin=.025,shift_margin=.025,detector='both signs on translated template peak channels; max(30uV,3 noise sigma), 0.8ms spacing',preprocessing='Original 300–6000Hz third-order zero-phase Butterworth and global median reference',selection='Spatial stratification, split-half template repeatability and SNR; sorted labels used for template proposals only. No DREDGE input.',source_script_sha256=digest(__file__),raw_stat=[st.st_size,st.st_mtime_ns],restart='Completed template/chunk stages validated by SHA256; interrupted chunk restarts. Not within-stage checkpointing.',limitations='Provisional identities; coarse geometry translations and expanded-search thresholds are not independently calibrated. All confidence classes retained. No continuous trajectory through missing support.')
 p=OUT/'settings.json'
 if p.exists():assert json.loads(p.read_text())==config,'Changed configuration needs a new output version'
 else:atomic_json(p,config)
 def read(start):
  first=round(start*fs);n=round(10*fs);pad=round(.05*fs)
  with raw.open('rb') as f:f.seek((first-pad)*768);buf=f.read((n+2*pad)*768)
  assert len(buf)==(n+2*pad)*768
  x=np.frombuffer(buf,dtype='<i2').reshape(-1,384).astype('float32')*m['gain_uv_per_count'];x=sosfiltfilt(sos,x,axis=0).astype('float32');x-=np.median(x,axis=1,keepdims=True);return x[pad:pad+n],first
 def sigma(x):return np.median(abs(x[::10]-np.median(x[::10],axis=0)),axis=0)/.67448975
 if not valid('templates'):
  ts=np.load(BASE/'cur/cur_output/spike_times.npy',mmap_mode='r').ravel();ids=np.load(BASE/'cur/cur_output/spike_clusters.npy',mmap_mode='r').ravel();proposals={};records=[]
  for start in [930,980]:
   x,first=read(start);noise=sigma(x);ix=np.flatnonzero((ts>=first+40)&(ts<first+len(x)-40));tt=np.asarray(ts[ix])-first;cc=np.asarray(ids[ix])
   for cid in np.unique(cc):
    ev=tt[cc==cid]
    if len(ev)<8:continue
    ev=ev[np.linspace(0,len(ev)-1,min(len(ev),120),dtype=int)];waves=x[ev[:,None]+np.arange(-30,31)];w=np.median(waves,axis=0);pk=int(abs(w).max(axis=0).argmax());ch=np.flatnonzero(abs(geo[:,1]-geo[pk,1])<=60);a=np.median(waves[::2][:,:,ch],axis=0);b=np.median(waves[1::2][:,:,ch],axis=0);rep=float(np.sum(a*b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-20));snr=float(abs(w[:,pk]).max()/noise[pk]);quality=rep*np.log1p(snr)*min(1,len(ev)/25)
    r=dict(unit_id=int(cid),training_start_s=start,training_events=len(ev),depth_um=float(geo[pk,1]),peak_channel=pk,repeatability=rep,snr=snr,quality=quality)
    records.append(r)
    if start==930:proposals[cid]=(r,w,noise)
   del x
  # Eligible proposal pool stays broad; grades are not identity certification.
  pool=[r for r,w,n in proposals.values() if r['snr']>=3 and r['repeatability']>=.65]
  chosen=[];families={};edges=np.linspace(geo[:,1].min(),geo[:,1].max()+1,11)
  def duplicate(cid):
   r,w,n=proposals[cid]
   for old in chosen:
    ro,wo,no=proposals[old]
    if abs(r['depth_um']-ro['depth_um'])>80:continue
    ch=np.flatnonzero(abs(geo[:,1]-r['depth_um'])<=100);aa=w[:,ch];bb=wo[:,ch];sim=np.sum(aa*bb)/(np.linalg.norm(aa)*np.linalg.norm(bb)+1e-20)
    if sim>.93:return old
   return None
  for rank in range(3):
   for lo,hi in zip(edges[:-1],edges[1:]):
    for r in sorted([r for r in pool if lo<=r['depth_um']<hi and r['unit_id'] not in chosen],key=lambda r:-r['quality']):
     cid=r['unit_id'];dup=duplicate(cid)
     if dup is not None:families[cid]=dup;continue
     chosen.append(cid);families[cid]=cid;break
  arr={};rows=[]
  for cid,(r,w,n) in proposals.items():
   ch=np.flatnonzero(abs(geo[:,1]-r['depth_um'])<=60);v=w[:,ch];en=(v*v).sum(axis=0);cent=float(en@geo[ch,1]/en.sum());rows.append(dict(**r,selected=cid in chosen,family_id=families.get(cid,cid),template_centroid_um=cent,grade='strong_seed' if r['snr']>=5 and r['repeatability']>=.85 else 'provisional_seed'));arr[f'unit_{cid}_full']=w;arr[f'unit_{cid}_template']=v;arr[f'unit_{cid}_channels']=ch;arr[f'unit_{cid}_noise']=n
  pd.DataFrame(rows).to_csv(OUT/'candidates.csv',index=False);pd.DataFrame(records).to_csv(OUT/'seed_inventory.csv',index=False);np.savez_compressed(OUT/'templates.npz',**arr);seal('templates',[OUT/'candidates.csv',OUT/'seed_inventory.csv',OUT/'templates.npz']);print('templates complete',len(chosen),flush=True)
 inv=pd.read_csv(OUT/'candidates.csv');sel=inv[inv.selected].sort_values('depth_um');bank=np.load(OUT/'templates.npz');models=[]
 for r in sel.itertuples():
  cid=r.unit_id;v=bank[f'unit_{cid}_template'];ch=bank[f'unit_{cid}_channels'];noise=bank[f'unit_{cid}_noise'];weights=v*v/(v*v+(2*noise[ch])**2+1e-20);maps=[];sh=[];pks=[]
  for s in SHIFTS:
   dest=np.array([mapping.get((geo[c,0],geo[c,1]+s),-1) for c in ch])
   if (dest<0).any():continue
   maps.append(dest);sh.append(s);pks.append(mapping[(geo[r.peak_channel,0],geo[r.peak_channel,1]+s)])
  offset=int(abs(v[:,np.flatnonzero(ch==r.peak_channel)[0]]).argmax())-30
  # Score only potentially similar rival shapes; preserve inventory and threshold.
  rivals=[]
  for dest in maps:
   rr=[]
   for q in inv.itertuples():
    if q.unit_id==cid or abs(q.depth_um-r.depth_um)>400 or q.repeatability<.65:continue
    w=bank[f'unit_{q.unit_id}_full']
    for s in SHIFTS:
     src=np.array([mapping.get((geo[c,0],geo[c,1]-s),-1) for c in dest]);rv=np.zeros_like(v);ok=src>=0;rv[:,ok]=w[:,src[ok]];sim=float(np.sum(weights*v*rv)/np.sqrt(np.sum(weights*v*v)*np.sum(weights*rv*rv)+1e-20))
     if sim>.65:rr.append((q.unit_id,int(s),rv))
   rivals.append(rr)
  models.append(dict(cid=cid,v=v,ch=ch,weights=weights,maps=maps,sh=np.array(sh),pks=pks,offset=offset,rivals=rivals))
 atomic_json(OUT/'rival_bank_summary.json',{str(c['cid']):[len(r) for r in c['rivals']] for c in models})
 for start in range(930,1030,10):
  stage=f's{start}'
  if valid(stage):print(stage,'reused validated checkpoint',flush=True);continue
  x,first=read(start);noise=sigma(x);detect={};rows=[];arrays={};decision_arrays={}
  for c in models:
   cid=c['cid'];ev=[]
   for pk in c['pks']:
    if pk not in detect:detect[pk]=find_peaks(abs(x[:,pk]),height=max(30,3*noise[pk]),distance=round(.0008*fs))[0]
    ev.extend(detect[pk]-c['offset'])
   ev=np.unique(ev);ev=ev[(ev>40)&(ev<len(x)-40)];sc=np.array([scores(x,ev,ch,c['v'],c['weights']) for ch in c['maps']]);eligible=(sc[:,:,1]>=.35)&(sc[:,:,1]<=3);ss=np.where(eligible,sc[:,:,0],-1);win=ss.argmax(axis=0);best=ss[win,np.arange(len(ev))];second=np.sort(ss,axis=0)[-2] if len(ss)>1 else np.full(len(ev),-1);rs=np.full(len(ev),-1.);rid=np.full(len(ev),-1)
   for j,rr in enumerate(c['rivals']):
    ii=np.flatnonzero((win==j)&(best>=.86))
    if not len(ii):continue
    for rcid,rshift,rv in rr:
     sr=scores(x,ev[ii],c['maps'][j],rv,c['weights']);val=np.where((sr[:,1]>=.35)&(sr[:,1]<=3),sr[:,0],-1);better=val>rs[ii];rs[ii[better]]=val[better];rid[ii[better]]=rcid
   status=np.full(len(ev),'unmatched',dtype='<U24');ii=np.flatnonzero(best>=.86);status[ii]='accepted';status[ii[best[ii]-second[ii]<.025]]='depth_ambiguous';status[ii[best[ii]-rs[ii]<.025]]='identity_ambiguous';boundary=abs(c['sh'][win])==120;status[ii[boundary[ii]& (status[ii]=='accepted')]]='boundary'
   # Duplicate detections collapse per identity, not across independent units.
   keep=[];occupied=np.zeros(len(x),bool)
   for i in np.argsort(-best):
    if best[i]<.86:break
    fr=int(sc[win[i],i,2]);a=max(0,fr-round(.0008*fs));b=min(len(x),fr+round(.0008*fs)+1)
    if occupied[a:b].any():status[i]='duplicate';continue
    occupied[fr]=True;keep.append(i)
   keep=np.array(sorted(keep,key=lambda i:sc[win[i],i,2]),int);fr=sc[win[keep],keep,2].astype(int);chs=np.array(c['maps'])[win[keep]];waves=x[fr[:,None,None]+np.arange(-30,31)[None,:,None],chs[:,None,:]];energy=(waves.astype(float)**2).sum(axis=1);cent=(energy*geo[chs,1]).sum(axis=1)/energy.sum(axis=1)
   for k,i in enumerate(keep):rows.append(dict(unit_id=cid,frame=int(fr[k]+first),time_s=(fr[k]+first)/fs,shift_um=int(c['sh'][win[i]]),centroid_um=cent[k],score=best[i],gain=sc[win[i],i,1],identity_margin=best[i]-rs[i],shift_margin=best[i]-second[i],rival_id=int(rid[i]),status=status[i],training_overlap=930<=start<940))
   for name,val in dict(frames=fr+first,waveforms=waves,channels=chs,shift_um=c['sh'][win[keep]],status=status[keep]).items():arrays[f'unit_{cid}_{name}']=val
   for name,val in dict(frames=ev+first,scores=sc,shifts=c['sh'],status=status,rival_score=rs,rival_id=rid).items():decision_arrays[f'unit_{cid}_{name}']=val
   print(stage,cid,'events',len(ev),'matches',len(keep),'accepted',int(np.sum(status[keep]=='accepted')),flush=True)
  pd.DataFrame(rows).to_csv(OUT/(stage+'_events.csv'),index=False);np.savez_compressed(OUT/(stage+'_waveforms.npz'),**arrays);np.savez_compressed(OUT/(stage+'_decisions.npz'),**decision_arrays);seal(stage,[OUT/(stage+s) for s in ['_events.csv','_waveforms.npz','_decisions.npz']]);del x
 events=pd.concat([pd.read_csv(OUT/f's{s}_events.csv') for s in range(930,1030,10)],ignore_index=True);events.to_csv(OUT/'events.csv',index=False)
 for c in models:
  arr={}
  for key in ['frames','waveforms','channels','shift_um','status']:
   arr[key]=np.concatenate([np.load(OUT/f's{s}_waveforms.npz')[f"unit_{c['cid']}_{key}"] for s in range(930,1030,10)])
  np.savez_compressed(OUT/f"events_unit_{c['cid']}.npz",**arr)
 assert (raw.stat().st_size,raw.stat().st_mtime_ns)==(st.st_size,st.st_mtime_ns)
 atomic_json(OUT/'summary.json',dict(status='complete',selected_units=len(models),events=len(events),status_counts=events.status.value_counts().to_dict(),seconds_this_launch=time.monotonic()-begin));print('COMPLETE',flush=True)
if __name__=='__main__':main()
