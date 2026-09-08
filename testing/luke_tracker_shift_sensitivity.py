"""Noiseless exact-geometry shift sensitivity of the fixed holdout matcher."""
import json
import numpy as np
import pandas as pd
from testing.luke_epoch_corroboration import ROOT,BASE
from testing.luke_multidepth_anchors_v2 import scores
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=ROOT/'testing/outputs/luke_tracker_shift_sensitivity_v1'
def main():
 OUT.mkdir(exist_ok=False);m=json.loads((BASE/'recording/rescue_recording_manifest.json').read_text());loc=np.asarray(m['channel_locations_um']);geometry={tuple(xy):i for i,xy in enumerate(loc)};z=np.load(ROOT/'testing/outputs/luke_candidate_footprint_audit_v1/full_probe_waveforms.npz');table=pd.read_csv(ROOT/'testing/outputs/luke_independent_transfer_review_v1/candidate_screen.csv').set_index('candidate');noise=np.load(ROOT/'testing/outputs/luke_peak_threshold_screen_v1/noise_uv.npy');names=['s960_c293_pos','s960_c338_pos','s9510_c290_pos','s9510_c330_pos'];rows=[]
 for name in names:
  r=table.loc[name];ch=np.flatnonzero(abs(loc[:,1]-r.depth_um)<=60);full=np.zeros((91,384),dtype='float32');full[:,ch]=z[name+'_h0'][:,ch];template=full[15:76,ch];weights=template**2/(template**2+(2*noise[ch])**2)
  for shift in [-80,-40,0,40,80]:
   shifted=np.zeros_like(full)
   for src in ch:
    dest=geometry.get((loc[src,0],loc[src,1]+shift))
    if dest is not None:shifted[:,dest]=full[:,src]
   x=np.zeros((200,384),dtype='float32');x[55:146]=shifted;s=scores(x,np.array([100]),ch,template,weights)[0];passed=bool(s[0]>=.9 and .4<=s[1]<=2.5);rows.append(dict(candidate=name,injected_shift_um=shift,score=float(s[0]),gain=float(s[1]),passes_fixed_matcher=passed))
 d=pd.DataFrame(rows);d.to_csv(OUT/'sensitivity.csv',index=False);fig,ax=plt.subplots(figsize=(9,5),layout='constrained')
 for name,g in d.groupby('candidate',sort=False):ax.plot(g.injected_shift_um,g.score,'o-',label=name)
 ax.axhline(.9,c='k',ls=':',label='Fixed acceptance threshold');ax.set(xlabel='Injected shift along exact same-x contact geometry (µm)',ylabel='Frozen matcher cosine',ylim=(-.1,1.05),title='Fixed-template matcher sensitivity to known waveform shifts');ax.legend(fontsize=8);fig.suptitle('Noiseless translated local templates; diagnostic of matcher acceptance, not a biological motion simulation',fontsize=10)
 for ext in ['png','pdf']:fig.savefig(OUT/f'01_shift_sensitivity.{ext}',dpi=150)
 (OUT/'settings.json').write_text(json.dumps(dict(shifts_um=[-80,-40,0,40,80],translation='Exact existing same-x contact geometry; local training waveform±60um only',background='Zero; isolates matching geometry',limitations='No firing model or noise; does not validate cell identity, displacements, or alternative tracking'),indent=2));print(d.to_string(index=False))
if __name__=='__main__':main()
