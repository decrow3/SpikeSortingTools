"""Translated-template holdout matching with frozen spatial/identity competition."""
import json
import numpy as np
import pandas as pd
from scipy.signal import butter,sosfiltfilt
from testing.luke_epoch_corroboration import ROOT,BASE
from testing.luke_multidepth_anchors_v2 import scores
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=ROOT/'testing/outputs/luke_translated_holdout_review_v1';SRC=ROOT/'testing/outputs';TARGETS={970:['s960_c293_pos','s960_c338_pos'],9520:['s9510_c290_pos','s9510_c330_pos']}
def main():
 OUT.mkdir(exist_ok=False);m=json.loads((BASE/'recording/rescue_recording_manifest.json').read_text());fs=m['sampling_frequency_hz'];loc=np.asarray(m['channel_locations_um']);lookup={tuple(v):i for i,v in enumerate(loc)};z=np.load(SRC/'luke_candidate_footprint_audit_v1/full_probe_waveforms.npz');table=pd.read_csv(SRC/'luke_independent_transfer_review_v1/candidate_screen.csv').set_index('candidate');noise=np.load(SRC/'luke_peak_threshold_screen_v1/noise_uv.npy');bank=[];rows=[];bins=[]
 for name,r in table.iterrows():
  source=np.flatnonzero(abs(loc[:,1]-r.depth_um)<=60);v=z[name+'_h0'][15:76,source]
  for shift in [-80,-40,0,40,80]:
   destinations=[lookup.get((loc[c,0],loc[c,1]+shift)) for c in source]
   if any(c is None for c in destinations):continue
   ch=np.asarray(destinations);w=v*v/(v*v+(2*noise[ch])**2);bank.append((name,int(r.start_s),shift,ch,v,w,int(r.sign)))
 (OUT/'settings.json').write_text(json.dumps(dict(shifts_um=[-80,-40,0,40,80],match='Cosine>=.9 gain.4–2.5; winner margin.03 over all other templates AND shifts',events='Reuse independently detected both-sign5sigma peaks; compare hypotheses with same detection sign as training cohort',qualification='Late3300um remains failed synthetic identity control; real matches are diagnostic only',scope='Coarse shift-aware corroboration, not continuous motion estimation or calibrated identity proof'),indent=2))
 for start,targets in TARGETS.items():
  first=round(start*fs);n=round(20*fs);pad=round(.05*fs)
  with (BASE/'recording/traces_cached_seg0.raw').open('rb') as f:f.seek((first-pad)*768);buf=f.read((n+2*pad)*768)
  x=np.frombuffer(buf,dtype='<i2').reshape(-1,384).astype('float32')*m['gain_uv_per_count'];x=sosfiltfilt(butter(3,[300,6000],fs=fs,btype='bandpass',output='sos'),x,axis=0).astype('float32');x-=np.median(x,axis=1,keepdims=True);x=x[pad:pad+n].copy();del buf;p=np.load(SRC/f'luke_transfer_template_holdout_v1/s{start}_peaks.npy');ep=start-10;hyp=[b for b in bank if b[1]==ep]
  for target in targets:
   depth=table.loc[target,'depth_um'];pp=p[(abs(loc[p['channel_index'],1]-depth)<=160)&(p['amplitude']>0)];ev=pp['sample_index'];hypotheses=[b for b in hyp if b[6]>0];results=[]
   for name,_,shift,ch,v,w,_ in hypotheses:
    s=scores(x,ev,ch,v,w);s[(s[:,1]<.4)|(s[:,1]>2.5),0]=-np.inf;results.append(s)
   result=np.asarray(results);rank=np.argsort(result[:,:,0],axis=0);winner=rank[-1];best=result[winner,np.arange(len(ev)),0];second=result[rank[-2],np.arange(len(ev)),0];chosen=[j for j in range(len(ev)) if hypotheses[winner[j]][0]==target and best[j]>=.9 and best[j]-second[j]>=.03];keep=[]
   for j in sorted(chosen,key=lambda j:-best[j]):
    if not keep or np.min(abs(result[winner[keep],keep,2]-result[winner[j],j,2]))>.001*fs:keep.append(j)
   for j in keep:
    shift=hypotheses[winner[j]][2];frame=int(result[winner[j],j,2]);rows.append(dict(candidate=target,start_s=start,time_s=start+frame/fs,frame=first+frame,shift_um=shift,score=float(best[j]),margin=float(best[j]-second[j])))
   for b in range(4):
    rr=[r for r in rows if r['candidate']==target and start+b*5<=r['time_s']<start+(b+1)*5];counts={s:sum(r['shift_um']==s for r in rr) for s in [-80,-40,0,40,80]};bins.append(dict(candidate=target,time_s=start+b*5+2.5,events=len(rr),**{f'shift_{s}':v for s,v in counts.items()}))
   print(target,'accepted',len(keep),flush=True)
  del x
 pd.DataFrame(rows).to_csv(OUT/'matched_events.csv',index=False);d=pd.DataFrame(bins);d.to_csv(OUT/'shift_counts.csv',index=False);fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained')
 for ax,(cid,g) in zip(axes.flat,d.groupby('candidate',sort=False)):
  bottom=np.zeros(len(g))
  for shift in [-80,-40,0,40,80]:v=g[f'shift_{shift}'].to_numpy();ax.bar(g.time_s,v,bottom=bottom,width=3.5,label=f'{shift:+d} µm');bottom+=v
  ax.set(title=cid+(' · identity control failed' if cid=='s9510_c330_pos' else ''),xlabel='Recording time (s)',ylabel='Accepted matches per5s bin');ax.legend(fontsize=7)
 fig.suptitle('Spatially flexible matches with fixed identity and shift margins\nCoarse40um grid; sparse competitor coverage; counts do not establish motion truth',fontsize=11)
 for ext in ['png','pdf']:fig.savefig(OUT/f'01_shift_counts.{ext}',dpi=150)
 (OUT/'summary.json').write_text(json.dumps(dict(status='complete',events=len(rows)),indent=2));print(d.to_string(index=False),flush=True)
if __name__=='__main__':main()
