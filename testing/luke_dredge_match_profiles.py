"""Inspect exact correlation curves and effective lag bounds; no parameter tuning."""
import json
import numpy as np
import pandas as pd
import torch
from testing.luke_epoch_corroboration import ROOT,BASE
from spikeinterface.core import NumpyRecording
from spikeinterface.sortingcomponents.motion.dredge import make_2d_motion_histogram,get_window_domains,normxcorr1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=ROOT/'testing/outputs/luke_dredge_match_profiles_v1'
SRC=ROOT/'testing/outputs/luke_compensation_validation_v1'
AUD=ROOT/'testing/outputs/luke_dredge_pairwise_audit_v1'
def main():
 OUT.mkdir(exist_ok=False);m=json.loads((BASE/'recording/rescue_recording_manifest.json').read_text());fs=m['sampling_frequency_hz'];rec=NumpyRecording(np.broadcast_to(np.zeros((1,384),dtype='float32'),(round(20*fs),384)),fs);rec.set_channel_locations(np.array(m['channel_locations_um']));bounds=[];rows=[]
 for arm in ['original','compensated']:
  z=np.load(AUD/f'{arm}_constraints.npz');peaks=np.load(SRC/f'{arm}_peaks.npy');pos=np.load(SRC/f'{arm}_locations.npy')
  h,te,se=make_2d_motion_histogram(rec,peaks,pos,weight_with_amplitude=True,avg_in_bin=False,direction='y',bin_s=1.,bin_um=1.,hist_margin_um=0.,spatial_bin_edges=None,depth_smooth_um=1.,time_smooth_s=1.)
  raster=h.T;windows=z['windows'];domains=get_window_domains(windows);centers=(se[1:]+se[:-1])/2
  fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
  for row,depth in enumerate([410,2410]):
   b=np.argmin(abs(z['window_centers']-depth));sl=domains[b];lo=max(0,sl.start-80);hi=min(len(raster),sl.stop+80);padding=max(lo-(sl.start-80),sl.stop+80-hi);left=padding+sl.start-lo;right=padding+hi-sl.stop;lags=-np.arange(-left,right+1)
   bounds.append(dict(arm=arm,depth_um=float(z['window_centers'][b]),lag_min=int(lags.min()),lag_max=int(lags.max()),padding=padding))
   curve=normxcorr1d(torch.tensor(raster[sl].T,dtype=torch.float32),torch.tensor(raster[lo:hi].T,dtype=torch.float32),weights=torch.tensor(windows[b,sl],dtype=torch.float32),padding=padding,normalized=True,centered=True).numpy()
   # Torch correlation array axes are target time, template time, lag.
   d=lags[curve.argmax(axis=2)].T;c=curve.max(axis=2).T
   assert np.allclose(d,z['D'][b]);assert np.allclose(c,z['C'][b],atol=1e-5)
   cross=abs(d[:10,10:]);i,j=np.unravel_index(np.argmax(cross),cross.shape);j+=10
   vals=curve[j,i];best=vals.argmax();near=abs(lags)<=10
   rows.append(dict(arm=arm,depth_um=float(z['window_centers'][b]),time_i_s=4240.5+i,time_j_s=4240.5+j,best_displacement_um=int(lags[best]),best_correlation=float(vals[best]),best_within10_correlation=float(vals[near].max()),best_within10_displacement_um=int(lags[near][vals[near].argmax()])))
   ax=axes[row,0];order=np.argsort(lags);ax.plot(lags[order],vals[order]);ax.axvline(lags[best],color='k',ls=':');ax.set(title=f'{depth} µm: {4240.5+i} vs {4240.5+j} s',xlabel='Pairwise lag (µm; internal D sign)',ylabel='Exact weighted correlation')
   ax=axes[row,1]
   for t in [i,j]:ax.plot(centers,raster[:,t],label=f'{4240.5+t} s',alpha=.8)
   ax.set(xlim=(max(0,depth-450),min(3820,depth+450)),xlabel='Localized depth (µm)',ylabel='Smoothed amplitude mass',title='Profiles before registration');ax.legend()
   np.savez_compressed(OUT/f'{arm}_{depth}_curves.npz',correlation=curve,lags=lags,raster=raster,depth_um=centers)
  fig.suptitle(f'{arm.title()}: largest cross-half match at each inspected depth\nExact curves reproduce saved DREDGE constraints; selection does not use lighthouse agreement',fontsize=11)
  for ext in ['png','pdf']:fig.savefig(OUT/f'{arm}_profiles.{ext}',dpi=150)
 pd.DataFrame(rows).to_csv(OUT/'representative_matches.csv',index=False);pd.DataFrame(bounds).to_csv(OUT/'effective_bounds.csv',index=False);print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
