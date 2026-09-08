"""Fixed-scale decomposition of exact DREDGE profile matches; no estimator adoption."""
import json
import numpy as np
import pandas as pd
import torch
from scipy.ndimage import gaussian_filter1d
from testing.luke_epoch_corroboration import ROOT
from spikeinterface.sortingcomponents.motion.dredge import get_window_domains,normxcorr1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=ROOT/'testing/outputs/luke_profile_scale_diagnostic_v1';SRC=ROOT/'testing/outputs/luke_dredge_match_profiles_v1';AUD=ROOT/'testing/outputs/luke_dredge_pairwise_audit_v1'
def main():
 OUT.mkdir(exist_ok=False);settings=dict(sigma_um=40,reason='Fixed two-contact-row spatial scale (20um pitch), selected before computing outputs; descriptive broad/fine decomposition, no sweep.',lag_bound_um=80,scope='Same compensated peaks; no lighthouse data used; not a fitted motion method');(OUT/'settings.json').write_text(json.dumps(settings,indent=2));rows=[];fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained');z=np.load(AUD/'compensated_constraints.npz');domains=get_window_domains(z['windows'])
 for row,depth in enumerate([410,2410]):
  data=np.load(SRC/f'compensated_{depth}_curves.npz');raster=data['raster'];b=np.argmin(abs(z['window_centers']-depth));sl=domains[b];lo=max(0,sl.start-80);hi=min(len(raster),sl.stop+80);pad=max(lo-(sl.start-80),sl.stop+80-hi);lags=-np.arange(-(pad+sl.start-lo),pad+hi-sl.stop+1);keep=abs(lags)<=80
  broad=gaussian_filter1d(raster,40,axis=0);fine=raster-broad
  i,j=(6,14) if depth==410 else (3,15)
  for name,r in [('Original profile',raster),('Broad component',broad),('Fine remainder',fine)]:
   curve=normxcorr1d(torch.tensor(r[sl].T,dtype=torch.float32),torch.tensor(r[lo:hi].T,dtype=torch.float32),weights=torch.tensor(z['windows'][b,sl],dtype=torch.float32),padding=pad,normalized=True,centered=True).numpy()[:,:,keep]
   D=lags[keep][curve.argmax(axis=2)].T;C=curve.max(axis=2).T;cross=D[:10,10:];closure=D[:,:,None]+D[None,:,:]-D[:,None,:];v=curve[j,i];order=np.argsort(lags[keep]);axes[row,0].plot(lags[keep][order],v[order],label=name)
   rows.append(dict(depth_um=depth,representation=name,representative_lag_um=int(lags[keep][v.argmax()]),representative_corr=float(v.max()),large_cross_half_fraction=float((abs(cross)>20).mean()),cross_half_median_corr=float(np.median(C[:10,10:])),p90_cycle_error_um=float(np.percentile(abs(closure),90))))
   np.savez_compressed(OUT/f'{depth}_{name.split()[0].lower()}_constraints.npz',D=D,C=C)
  axes[row,0].set(title=f'{depth} µm · representative pair',xlabel='Internal pairwise lag (µm)',ylabel='Weighted correlation');axes[row,0].legend(fontsize=8)
  for t in [i,j]:axes[row,1].plot(data['depth_um'],fine[:,t],label=f'{4240.5+t} s')
  axes[row,1].set(xlim=(max(0,depth-450),min(3820,depth+450)),title='Fine remainder retains local peaks',xlabel='Depth (µm)',ylabel='Amplitude mass minus broad component');axes[row,1].legend()
 fig.suptitle('Fixed 40 µm spatial decomposition, compensated 4240–4260 s inputs\nDiagnostic only: signed fine remainder changes the correlation representation',fontsize=11)
 for ext in ['png','pdf']:fig.savefig(OUT/f'01_scale_diagnostic.{ext}',dpi=150)
 pd.DataFrame(rows).to_csv(OUT/'summary.csv',index=False);print(pd.DataFrame(rows).to_string(index=False))
if __name__=='__main__':main()
