"""Controlled argmax-domain correction using existing cached peak inputs."""
import json
import numpy as np
import pandas as pd
import torch
from testing.luke_epoch_corroboration import ROOT,BASE
from spikeinterface.core import NumpyRecording
from spikeinterface.sortingcomponents.motion.dredge import make_2d_motion_histogram,get_window_domains,normxcorr1d,dredge_ap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=ROOT/'testing/outputs/luke_dredge_strict_bounds_v1';SRC=ROOT/'testing/outputs/luke_compensation_validation_v1';AUD=ROOT/'testing/outputs/luke_dredge_pairwise_audit_v1'
def main():
 OUT.mkdir(exist_ok=False);m=json.loads((BASE/'recording/rescue_recording_manifest.json').read_text());fs=m['sampling_frequency_hz'];rec=NumpyRecording(np.broadcast_to(np.zeros((1,384),dtype='float32'),(round(20*fs),384)),fs);rec.set_channel_locations(np.array(m['channel_locations_um']));cfg=json.loads((SRC/'settings.json').read_text())['estimator'].copy()
 for key in ['method','verbose']:cfg.pop(key,None)
 cfg['extra_outputs']=True;(OUT/'settings.json').write_text(json.dumps(dict(estimator=cfg,change='Argmax restricted to abs(lag)<=80; no new motion threshold or peak changes'),indent=2));rows=[];fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
 for ax,arm in zip(axes,['original','compensated']):
  z=np.load(AUD/f'{arm}_constraints.npz');peaks=np.load(SRC/f'{arm}_peaks.npy');pos=np.load(SRC/f'{arm}_locations.npy')
  h,te,se=make_2d_motion_histogram(rec,peaks,pos,weight_with_amplitude=True,avg_in_bin=False,direction='y',bin_s=1.,bin_um=1.,hist_margin_um=0.,spatial_bin_edges=None,depth_smooth_um=1.,time_smooth_s=1.)
  raster=h.T;windows=z['windows'];domains=get_window_domains(windows);D=np.empty_like(z['D']);C=np.empty_like(z['C'])
  for b,sl in enumerate(domains):
   lo=max(0,sl.start-80);hi=min(len(raster),sl.stop+80);padding=max(lo-(sl.start-80),sl.stop+80-hi);lags=-np.arange(-(padding+sl.start-lo),padding+hi-sl.stop+1)
   curve=normxcorr1d(torch.tensor(raster[sl].T,dtype=torch.float32),torch.tensor(raster[lo:hi].T,dtype=torch.float32),weights=torch.tensor(windows[b,sl],dtype=torch.float32),padding=padding,normalized=True,centered=True).numpy()
   assert np.allclose(lags[curve.argmax(axis=2)].T,z['D'][b]);assert np.allclose(curve.max(axis=2).T,z['C'][b],atol=1e-5)
   keep=abs(lags)<=80;restricted=curve[:,:,keep];D[b]=lags[keep][restricted.argmax(axis=2)].T;C[b]=restricted.max(axis=2).T
  assert abs(D).max()<=80
  motion,extra=dredge_ap(rec,peaks,pos,precomputed_D_C_maxdisp=(D,C,80.),**cfg)
  new=motion.displacement[0];old=np.load(SRC/f'{arm}_motion.npz')['displacement_um'];dep=motion.spatial_bins_um
  assert np.isfinite(new).all();np.savez_compressed(OUT/f'{arm}_motion.npz',displacement_um=new,depth_um=dep,time_s=motion.temporal_bins_s[0]+4240,D=D,C=C)
  for a,label in [(old,'Existing search'),(new,'Strict ±80 µm')]:ax.plot(np.median(a[10:],axis=0)-np.median(a[:10],axis=0),dep,label=label,ls='--' if label=='Existing search' else '-')
  ax.set(title=arm.title(),xlabel='Second-half − first-half displacement (µm)',ylabel='Depth (µm)',ylim=(0,3820));ax.legend();ax.axvline(0,c='gray',lw=.5)
  for depth in [220,620,1740,2380]:
   oldv=np.array([np.interp(depth,dep,v) for v in old]);newv=np.array([np.interp(depth,dep,v) for v in new]);rows.append(dict(arm=arm,depth_um=depth,original_step_um=float(np.median(oldv[10:])-np.median(oldv[:10])),strict_step_um=float(np.median(newv[10:])-np.median(newv[:10]))))
  print(arm,'changed constraints',int((D!=z['D']).sum()),'of',D.size,flush=True)
 fig.suptitle('Enforcing the requested search bounds: 4240–4260 s\nSame peaks and solver settings; only out-of-range lag candidates excluded',fontsize=11)
 for ext in ['png','pdf']:fig.savefig(OUT/f'01_strict_bounds.{ext}',dpi=150)
 pd.DataFrame(rows).to_csv(OUT/'comparison.csv',index=False);(OUT/'summary.json').write_text(json.dumps(dict(status='complete',comparisons=rows),indent=2));print(json.dumps(rows),flush=True)
if __name__=='__main__':main()
