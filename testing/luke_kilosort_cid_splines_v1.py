"""Cheap fixed-penalty cubic smoothing splines of cached Kilosort spike depths."""
from pathlib import Path
import json,time
import numpy as np
import pandas as pd
from scipy.interpolate import make_smoothing_spline
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];SRC=ROOT/'testing/outputs/luke_kilosort_cid_depth_scatter_v1';OUT=ROOT/'testing/outputs/luke_kilosort_cid_splines_v1'
BIN=.25;LAMBDA=.0001;GAP=1.;GRID=.05

def main():
 OUT.mkdir(exist_ok=True);begin=time.monotonic();d=pd.read_csv(SRC/'spikes_930_1030.csv');c=pd.read_csv(SRC/'cid_color_and_counts.csv');d['bin']=np.floor((d.time_s-930)/BIN).astype(int)
 b=d.groupby(['cid','bin']).agg(time_s=('time_s','median'),depth_um=('depth_um','median'),spikes=('time_s','size')).reset_index();b.to_csv(OUT/'median_depth_knots.csv',index=False)
 rows=[];segments=[];fit_start=time.monotonic()
 for cid,q in b.groupby('cid',sort=False):
  q=q.sort_values('time_s');t=q.time_s.to_numpy();y=q.depth_um.to_numpy()
  for part in np.split(np.arange(len(q)),np.flatnonzero(np.diff(t)>GAP)+1):
   status='fitted' if len(part)>=5 else 'too_few_knots';seg=len(segments)
   segments.append(dict(segment=seg,cid=cid,start_s=t[part[0]],end_s=t[part[-1]],knots=len(part),status=status))
   if status!='fitted':continue
   x=t[part]-930;v=y[part];spline=make_smoothing_spline(x,v,lam=LAMBDA)
   xx=np.unique(np.r_[x[0],np.arange(x[0],x[-1],GRID),x[-1]]);yy=spline(xx)
   assert np.isfinite(yy).all()
   rows.extend(dict(cid=cid,segment=seg,time_s=float(a+930),depth_um=float(z)) for a,z in zip(xx,yy))
 fit_seconds=time.monotonic()-fit_start;r=pd.DataFrame(rows);r.to_csv(OUT/'spline_traces.csv',index=False);pd.DataFrame(segments).to_csv(OUT/'fit_segments.csv',index=False);colors=c.set_index('cid').color.to_dict()
 for stem,nrows,height in [('01_whole_probe',1,7),('02_depth_details',6,16)]:
  fig,axes=plt.subplots(nrows,1,figsize=(15,height),sharex=True,layout='constrained',squeeze=False)
  for ri,ax in enumerate(axes[:,0]):
   lo,hi=(0,3840) if nrows==1 else (ri*640,(ri+1)*640)
   for (cid,seg),q in r.groupby(['cid','segment'],sort=False):
    if q.depth_um.max()<lo or q.depth_um.min()>hi:continue
    ax.plot(q.time_s,q.depth_um,color=colors[cid],lw=.55 if nrows==1 else .8,alpha=.85,rasterized=True)
   ax.set(xlim=(930,1030),ylim=(lo,hi),ylabel='Kilosort depth (µm)');ax.ticklabel_format(axis='x',style='plain',useOffset=False)
   if nrows>1:ax.set_title(f'{lo}–{hi} µm')
  axes[-1,0].set_xlabel('Recording time (s)');fig.suptitle(f'Per-unit Kilosort depth traces · 930–1,030 s\n{r.cid.nunique()} / {len(c)} active CIDs fitted · same CID colors as spike scatter',fontsize=14)
  fig.supxlabel('Fixed cubic smoothing spline on0.25s median depths; λ=0.0001; no tuning. Gaps >1s split fits; ≥5 occupied bins per segment.\nNo extrapolation. Colors repeat across the probe. These are descriptive fits to sorter-derived positions, not independent motion estimates.',fontsize=9)
  for ext in ['png','pdf']:fig.savefig(OUT/f'{stem}.{ext}',dpi=180)
  plt.close(fig)
 summary=dict(status='complete',active_cids=len(c),fitted_cids=int(r.cid.nunique()),fitted_segments=sum(x['status']=='fitted' for x in segments),knots=len(b),bin_s=BIN,lambda_fixed=LAMBDA,split_gap_s=GAP,minimum_knots=5,evaluation_grid_s=GRID,fit_seconds=fit_seconds,total_seconds=time.monotonic()-begin,source=str(SRC),method='Unweighted median-depth knots; fixed-penalty cubic smoothing spline, no hyperparameter search, bootstrap or new spike processing. Grid spacing is drawing density, not measurement resolution.')
 (OUT/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
