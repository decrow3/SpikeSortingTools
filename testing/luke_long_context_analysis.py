"""Compare central identical inputs under20s and100s registration context."""
import json
import numpy as np
import pandas as pd
from testing.luke_epoch_corroboration import ROOT,BASE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=ROOT/'testing/outputs/luke_long_context_validation_v1';SRC=ROOT/'testing/outputs'
def main():
 assert (OUT/'complete.json').exists(),'Existing job has not completed'
 m=json.loads((BASE/'recording/rescue_recording_manifest.json').read_text());fs=m['sampling_frequency_hz'];offset=round(970*fs)-round(930*fs);verified={};rows=[];tracks=pd.read_csv(SRC/'luke_transfer_template_holdout_v1/tracks.csv');tracks=tracks[tracks.time_s<1000]
 for arm in ['original','compensated']:
  p=np.load(OUT/f'{arm}_peaks.npy');y=np.load(OUT/f'{arm}_locations.npy');k=(p['sample_index']>=offset)&(p['sample_index']<offset+round(20*fs));central=p[k].copy();central['sample_index']-=offset;old=np.load(SRC/f'luke_distant_motion_validation_v1/s970_{arm}_peaks.npy');oldy=np.load(SRC/f'luke_distant_motion_validation_v1/s970_{arm}_locations.npy');assert np.array_equal(central,old) and np.array_equal(y[k],oldy);verified[arm]=len(old)
  for method in ['dredge','iterative']:
   for context in [20,100]:
    path=OUT/f'{arm}_{method}_motion.npz' if context==100 else SRC/('luke_distant_motion_validation_v1' if method=='dredge' else 'luke_alternative_registration_v1')/f's970_{arm}_motion.npz';z=np.load(path)
    for cid,g in tracks.groupby('candidate',sort=False):
     v=np.array([np.interp(g.depth_um.iloc[0],z['depth_um'],w) for w in z['displacement_um']])
     for r in g.itertuples():
      k=(z['time_s']>=r.time_s-2.5)&(z['time_s']<r.time_s+2.5);rows.append(dict(candidate=cid,arm=arm,method=method,context_s=context,time_s=r.time_s,centroid_um=r.centroid_um,displacement_um=float(np.median(v[k]))))
 d=pd.DataFrame(rows);d.to_csv(OUT/'central_comparison.csv',index=False);summary=[];fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained')
 for row,method in enumerate(['dredge','iterative']):
  for col,(cid,wave) in enumerate(tracks.groupby('candidate',sort=False)):
   ax=axes[row,col];rt=wave[wave.centroid_um.notna()].time_s.min();ref=wave[wave.time_s==rt].centroid_um.iloc[0];g=d[(d.method==method)&(d.candidate==cid)]
   for (arm,context),h in g.groupby(['arm','context_s']):
    v=h.displacement_um-h[h.time_s==rt].displacement_um.iloc[0];summary.append(dict(candidate=cid,method=method,arm=arm,context_s=int(context),central_range_um=float(v.max()-v.min())))
    if arm=='compensated':ax.plot(h.time_s,v,label=f'{context}s context',ls='--' if context==20 else '-')
   ax.plot(wave.time_s,wave.centroid_um-ref,'kx-',label='Waveform centroid');ax.set(title=f'{method} · {cid}',xlabel='Recording time (s)',ylabel='Change from first valid waveform bin (µm)');ax.legend(fontsize=8)
 fig.suptitle('Does longer context resolve compensated-input excursions?\nCentral peaks/locations verified identical; settings frozen; provisional waveform observations',fontsize=11)
 for ext in ['png','pdf']:fig.savefig(OUT/f'01_context_comparison.{ext}',dpi=150)
 (OUT/'analysis_summary.json').write_text(json.dumps(dict(status='complete',exact_central_input_counts=verified,ranges=summary),indent=2));print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
