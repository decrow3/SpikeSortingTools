"""Registration-input comparison after fresh detection, no estimator rerun."""
from pathlib import Path
import numpy as np,pandas as pd
from scipy.ndimage import gaussian_filter1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parents[1]/'testing/outputs/luke_peak_threshold_screen_v1';fs=29999.835983263598;edges=np.arange(0,3841);depth=edges[:-1]+.5;shift=np.arange(-20,21);rows=[];fig,axes=plt.subplots(1,2,figsize=(11,4.5))
for th in [4,5,6,7]:
 a=np.load(P/f'peaks_{th}sigma.npy');y=np.load(P/f'locations_{th}sigma.npy')['y'];t=a['sample_index']/fs;am=abs(a['amplitude']);keep=np.load(P/f'provisional_keep_{th}sigma.npy')
 for screened,mask in [('all',np.ones(len(a),bool)),('flagged_removed',keep)]:
  profiles=[]
  for start in [0,10]:
   k=mask&(t>=start)&(t<start+10);profiles.append(gaussian_filter1d(np.histogram(y[k],edges,weights=am[k])[0],1.))
  for ax,(lo,hi) in zip(axes,[(1960,2560),(2260,2860)]):
   mid=(depth>=lo)&(depth<hi);v=profiles[0][mid];corr=[np.corrcoef(v,np.interp(depth[mid]+s,depth,profiles[1]))[0,1] for s in shift];best=int(shift[np.argmax(corr)]);rows.append(dict(threshold=th,arm=screened,depth_lo=lo,depth_hi=hi,best_shift_um=best,best_correlation=max(corr),zero_correlation=corr[20]))
   if screened=='all':ax.plot(shift,corr,label=f'{th}σ')
for ax,(lo,hi) in zip(axes,[(1960,2560),(2260,2860)]):ax.set(xlabel='After − before shift (µm)',ylabel='Profile correlation',title=f'Depth {lo}–{hi} µm');ax.axvline(0,c='gray',lw=.5);ax.legend()
fig.suptitle('Fresh localized peak profiles: do they support the lighthouse drop?\n4180–4190 vs 4190–4200 s; summed-amplitude diagnostic, not an estimator replay');fig.tight_layout(rect=[0,0,1,.87]);fig.savefig(P/'03_fresh_alignment_objective.png',dpi=160);fig.savefig(P/'03_fresh_alignment_objective.pdf');pd.DataFrame(rows).to_csv(P/'fresh_alignment_objective.csv',index=False);print(pd.DataFrame(rows).to_string(index=False))
