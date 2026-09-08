"""Single-pair diagnostic of peak raster alignment, not an estimator replay."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'testing/outputs/luke_existing_motion_failure_audit_v1';P=Path('/mnt/NPX/Luke/20250804/dredge_pipeline_results_Luke0804_V2V1_g0_imec0/motion');fs=29999.835983263598
p=np.load(P/'peaks.npy',mmap_mode='r');l=np.load(P/'peak_locations.npy',mmap_mode='r');i,j=np.searchsorted(p['sample_index'],np.array([4180,4200])*fs);a=p[i:j];y=l['y'][i:j];t=a['sample_index']/fs;am=abs(a['amplitude']);edges=np.arange(0,3841,1);depth=edges[:-1]+.5;shift=np.arange(-20,21);rows=[];fig,axes=plt.subplots(2,3,figsize=(13,8))
for col,(lo,hi) in enumerate([(320,920),(1960,2560),(2260,2860)]):
 for name,exclude in [('All peaks',[]),('Omit ch252',[252]),('Omit ch238/251/252',[238,251,252])]:
  keep=~np.isin(a['channel_index'],exclude);profiles=[]
  for start in [4180,4190]:
   k=keep&(t>=start)&(t<start+10);h=np.histogram(y[k],edges,weights=am[k])[0];profiles.append(gaussian_filter1d(h,1.))
  mid=(depth>=lo)&(depth<hi);aa=profiles[0][mid];vals=[]
  for delta in shift:
   bb=np.interp(depth[mid]+delta,depth,profiles[1]);den=np.linalg.norm(aa-aa.mean())*np.linalg.norm(bb-bb.mean());vals.append(float(np.dot(aa-aa.mean(),bb-bb.mean())/den) if den else np.nan)
  best=int(shift[np.nanargmax(vals)]);rows.append(dict(depth_lo_um=lo,depth_hi_um=hi,condition=name,best_shift_um=best,best_correlation=float(np.max(vals)),zero_shift_correlation=float(vals[20])));axes[0,col].plot(shift,vals,label=name)
  if not exclude:
   axes[1,col].plot(depth[mid],aa,label='4180–4190');axes[1,col].plot(depth[mid],profiles[1][mid],label='4190–4200')
 axes[0,col].set_title(f'Depth {lo}–{hi} µm');axes[0,col].set_xlabel('Candidate after − before shift (µm)');axes[0,col].set_ylabel('Profile correlation');axes[0,col].axvline(0,c='gray',lw=.5);axes[1,col].set_xlabel('Depth (µm)');axes[1,col].set_ylabel('Smoothed sum |saved amplitude|');axes[0,col].legend(fontsize=7);axes[1,col].legend(fontsize=7)
fig.suptitle('Does removing concentrated detection channels change the preferred alignment?\nSingle 10 s pair; 1 µm bins / smoothing; summed-amplitude profiles; not a DREDGE or decentralized replay',fontsize=11);fig.tight_layout(rect=[0,0,1,.91]);fig.savefig(OUT/'02_pairwise_objective.png',dpi=160);fig.savefig(OUT/'02_pairwise_objective.pdf');pd.DataFrame(rows).to_csv(OUT/'pairwise_objective.csv',index=False);print(pd.DataFrame(rows).to_string(index=False))
