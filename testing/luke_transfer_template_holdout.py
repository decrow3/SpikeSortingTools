"""Fresh held-out detection with frozen candidate and competitor templates."""
import json
import numpy as np
import pandas as pd
from scipy.signal import butter,sosfiltfilt
from testing.luke_epoch_corroboration import ROOT,BASE
from testing.luke_multidepth_anchors_v2 import scores
from spikeinterface.core import NumpyRecording
from spikeinterface.sortingcomponents.peak_detection import detect_peaks
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=ROOT/'testing/outputs/luke_transfer_template_holdout_v1';SOURCE=ROOT/'testing/outputs/luke_candidate_footprint_audit_v1';COHORTS=ROOT/'testing/outputs/luke_independent_transfer_review_v1'
TARGETS={970:['s960_c293_pos','s960_c338_pos'],9520:['s9510_c290_pos','s9510_c330_pos']}
def main():
 OUT.mkdir(exist_ok=False);m=json.loads((BASE/'recording/rescue_recording_manifest.json').read_text());fs=m['sampling_frequency_hz'];loc=np.asarray(m['channel_locations_um']);noise=np.load(ROOT/'testing/outputs/luke_peak_threshold_screen_v1/noise_uv.npy');z=np.load(SOURCE/'full_probe_waveforms.npz');table=pd.read_csv(COHORTS/'candidate_screen.csv');allrows=[];tracks=[];saved={};templates={}
 for r in table.itertuples():
  ch=np.flatnonzero(abs(loc[:,1]-r.depth_um)<=60);v=z[r.candidate+'_h0'][15:76,ch];weights=v*v/(v*v+(2*noise[ch])**2);offset=int(abs(v[:,np.flatnonzero(ch==r.channel)[0]]).argmax())-30;templates[r.candidate]=(ch,v,weights,offset,r.depth_um,r.start_s)
 (OUT/'settings.json').write_text(json.dumps(dict(targets=TARGETS,interval_duration_s=20,detection='Both signs,5sigma fixed development noise,50um locally exclusive',match='Frozen first-half61sample templates; energy/noise weighted cosine>=0.9,gain0.4–2.5; timing±3samples',competition='All available original-voltage channel/sign cohort templates within120um; winner margin0.03',nms_ms=1,scope='Separate temporal holdout with sparse competitor library; no exhaustive identity/precision claim or DREDGE tuning'),indent=2))
 for start,targets in TARGETS.items():
  first=round(start*fs);n=round(20*fs);pad=round(.05*fs)
  with (BASE/'recording/traces_cached_seg0.raw').open('rb') as f:f.seek((first-pad)*768);buf=f.read((n+2*pad)*768)
  x=np.frombuffer(buf,dtype='<i2').reshape(-1,384).astype('float32')*m['gain_uv_per_count'];x=sosfiltfilt(butter(3,[300,6000],fs=fs,btype='bandpass',output='sos'),x,axis=0).astype('float32');x-=np.median(x,axis=1,keepdims=True);x=x[pad:pad+n].copy();del buf
  rec=NumpyRecording(x,fs);rec.set_channel_locations(loc);p=detect_peaks(rec,method='locally_exclusive',peak_sign='both',radius_um=50.,detect_threshold=5.,noise_levels=noise,n_jobs=1,chunk_duration='1s',progress_bar=False);p=p[(p['sample_index']>50)&(p['sample_index']<n-50)];np.save(OUT/f's{start}_peaks.npy',p)
  for target in targets:
   ch,v,w,offset,depth,training=templates[target];pk=p[abs(loc[p['channel_index'],1]-depth)<=80];ev=pk['sample_index']-offset;res=scores(x,ev,ch,v,w);competitors=[k for k,val in templates.items() if k!=target and val[5]==training and abs(val[4]-depth)<=120];bestother=np.full(len(ev),-np.inf)
   for comp in competitors:
    cc,vv,ww,oo,_,_=templates[comp];rr=scores(x,pk['sample_index']-oo,cc,vv,ww);valid=(rr[:,1]>=.4)&(rr[:,1]<=2.5);bestother=np.maximum(bestother,np.where(valid,rr[:,0],-np.inf))
   accepted=(res[:,0]>=.9)&(res[:,1]>=.4)&(res[:,1]<=2.5)&(res[:,0]-bestother>=.03);selected=np.flatnonzero(accepted);keep=[]
   for j in selected[np.argsort(-res[selected,0])]:
    if not keep or np.min(abs(res[keep,2]-res[j,2]))>.001*fs:keep.append(j)
   keep=np.array(sorted(keep,key=lambda j:res[j,2]),dtype=int);frames=res[keep,2].astype(int);saved[target+'_holdout_frames']=frames+first
   for j in keep:allrows.append(dict(candidate=target,start_s=start,frame=int(first+res[j,2]),score=float(res[j,0]),gain=float(res[j,1]),competitor_score=float(bestother[j]) if np.isfinite(bestother[j]) else None))
   for b in range(4):
    e=frames[(frames>=b*5*fs)&(frames<(b+1)*5*fs)];centroid=np.nan;cos=np.nan
    if len(e)>=10:
     wave=np.median(x[e[:,None,None]+np.arange(-30,31)[None,:,None],ch[None,None,:]],axis=0);energy=(wave*wave).sum(axis=0);centroid=float(energy@loc[ch,1]/energy.sum());cos=float(np.sum(wave*v)/(np.linalg.norm(wave)*np.linalg.norm(v)));saved[target+f'_bin{b}_waveform']=wave
    tracks.append(dict(candidate=target,time_s=start+b*5+2.5,depth_um=depth,events=len(e),centroid_um=centroid,template_cosine=cos))
   print(target,'accepted',len(frames),'competitors',competitors,flush=True)
  del x,rec
 pd.DataFrame(allrows).to_csv(OUT/'matched_events.csv',index=False);t=pd.DataFrame(tracks);t.to_csv(OUT/'tracks.csv',index=False);np.savez_compressed(OUT/'waveforms_events.npz',**saved);fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained')
 for ax,(candidate,g) in zip(axes.flat,t.groupby('candidate',sort=False)):
  good=g.centroid_um.notna();ref=g.loc[good,'centroid_um'].iloc[0] if good.any() else 0;ax.plot(g.time_s,g.centroid_um-ref,'o-');
  for r in g.itertuples():ax.annotate(f'n={r.events}',(r.time_s,r.centroid_um-ref if np.isfinite(r.centroid_um) else 0),xytext=(0,8),textcoords='offset points',fontsize=8)
  ax.set(title=candidate,xlabel='Recording time (s)',ylabel='Centroid relative to first valid bin (µm)',xlim=(g.time_s.min()-1,g.time_s.max()+1))
 fig.suptitle('Frozen candidate templates in separate temporal holdouts\n5s bins require10 accepted events; sparse competitor coverage, centroids not calibrated displacement',fontsize=11)
 for ext in ['png','pdf']:fig.savefig(OUT/f'01_holdout_tracks.{ext}',dpi=150)
 summary=dict(status='complete',matched_events=len(allrows),valid_bins=int(t.centroid_um.notna().sum()),total_bins=len(t),counts=pd.DataFrame(allrows).groupby('candidate').size().to_dict() if allrows else {});(OUT/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
