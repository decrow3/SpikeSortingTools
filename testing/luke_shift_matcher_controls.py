"""Exact-geometry translated-template recovery and cross-template confusion controls."""
import json
import numpy as np
import pandas as pd
from testing.luke_epoch_corroboration import ROOT,BASE
from testing.luke_multidepth_anchors_v2 import scores
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=ROOT/'testing/outputs/luke_shift_matcher_controls_v1';SHIFTS=[-80,-40,0,40,80]
def main():
 OUT.mkdir(exist_ok=False);m=json.loads((BASE/'recording/rescue_recording_manifest.json').read_text());loc=np.asarray(m['channel_locations_um']);lookup={tuple(v):i for i,v in enumerate(loc)};z=np.load(ROOT/'testing/outputs/luke_candidate_footprint_audit_v1/full_probe_waveforms.npz');table=pd.read_csv(ROOT/'testing/outputs/luke_independent_transfer_review_v1/candidate_screen.csv');noise=np.load(ROOT/'testing/outputs/luke_peak_threshold_screen_v1/noise_uv.npy');targets=['s960_c293_pos','s960_c338_pos','s9510_c290_pos','s9510_c330_pos'];bank=[];fulls={};metadata=table.set_index('candidate')
 for r in table.itertuples():
  source=np.flatnonzero(abs(loc[:,1]-r.depth_um)<=60);v=z[r.candidate+'_h0'][15:76,source];full=np.zeros((61,384),dtype='float32');full[:,source]=v;fulls[r.candidate]=full
  for shift in SHIFTS:
   destinations=[lookup.get((loc[c,0],loc[c,1]+shift)) for c in source]
   if any(c is None for c in destinations):continue
   ch=np.asarray(destinations);weights=v*v/(v*v+(2*noise[ch])**2);bank.append((r.candidate,r.start_s,shift,ch,v,weights))
 rows=[];confusion=[]
 for target in targets:
  epoch=int(metadata.loc[target,'start_s']);depth=metadata.loc[target,'depth_um'];source=np.flatnonzero(abs(loc[:,1]-depth)<=60)
  for injected in SHIFTS:
   x=np.zeros((200,384),dtype='float32')
   for ch in source:x[70:131,lookup[(loc[ch,0],loc[ch,1]+injected)]]=fulls[target][:,ch]
   candidates=[]
   for name,ep,shift,ch,v,w in bank:
    if ep!=epoch:continue
    s=scores(x,np.array([100]),ch,v,w)[0]
    if .4<=s[1]<=2.5:candidates.append((float(s[0]),name,shift,float(s[1])))
   own=sorted([s for s in candidates if s[1]==target],reverse=True);other=sorted([s for s in candidates if s[1]!=target],reverse=True);best=own[0];comp=other[0] if other else (-1,'none',0,0);margin=best[0]-comp[0];rows.append(dict(target=target,injected_shift_um=injected,recovered_shift_um=best[2],score=best[0],competitor=comp[1],competitor_shift_um=comp[2],competitor_score=comp[0],margin=margin,passes=best[2]==injected and best[0]>=.9 and margin>=.03))
 # Place each same-epoch competitor at the target depth to test shape specificity independent of position.
 for target in targets:
  tr=metadata.loc[target];targetrow=[b for b in bank if b[0]==target and b[2]==0][0];_,_,_,tch,tv,tw=targetrow
  for r in table[table.start_s==tr.start_s].itertuples():
   if r.candidate==target:continue
   shift=float(tr.depth_um-r.depth_um)
   if shift%40:continue
   x=np.zeros((200,384),dtype='float32');source=np.flatnonzero(abs(loc[:,1]-r.depth_um)<=60)
   for ch in source:
    dest=lookup.get((loc[ch,0],loc[ch,1]+shift))
    if dest is not None:x[70:131,dest]=fulls[r.candidate][:,ch]
   s=scores(x,np.array([100]),tch,tv,tw)[0];confusion.append(dict(target=target,competitor=r.candidate,synthetic_competitor_shift_um=shift,score=float(s[0]),gain=float(s[1]),passes_target_threshold=bool(s[0]>=.9 and .4<=s[1]<=2.5)))
 d=pd.DataFrame(rows);d.to_csv(OUT/'recovery.csv',index=False);c=pd.DataFrame(confusion);c.to_csv(OUT/'shape_confusion.csv',index=False);fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
 for name,g in d.groupby('target',sort=False):axes[0].plot(g.injected_shift_um,g.recovered_shift_um,'o-',label=name)
 axes[0].plot([-80,80],[-80,80],'k:',label='Correct shift');axes[0].set(xlabel='Injected shift (µm)',ylabel='Recovered shift (µm)',title='Noiseless recovery');axes[0].legend(fontsize=7)
 for i,r in enumerate(c.itertuples()):axes[1].scatter(r.score,i,c='tab:red' if r.passes_target_threshold else 'gray')
 axes[1].axvline(.9,c='k',ls=':');axes[1].set(xlabel='Target score for a different waveform at its depth',ylabel='Synthetic competitor case',title='Shape specificity stress test')
 fig.suptitle('Spatially flexible matcher controls before real-data use\nExact geometry translations; noiseless; sparse competitors cannot establish full specificity',fontsize=11)
 for ext in ['png','pdf']:fig.savefig(OUT/f'01_controls.{ext}',dpi=150)
 result=dict(status='complete',correct_shift_cases=int((d.injected_shift_um==d.recovered_shift_um).sum()),total=len(d),recovery_and_margin_pass=int(d.passes.sum()),shape_confusion_cases=int(c.passes_target_threshold.sum()),shape_tests=len(c),scope='Noiseless synthetic controls only; not permission to treat real matches as calibrated motion');(OUT/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result));print(c[c.passes_target_threshold].to_string(index=False));print(d[~d.passes].to_string(index=False))
if __name__=='__main__':main()
