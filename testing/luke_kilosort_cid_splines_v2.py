"""Cheap whole-unit smoothing splines, retaining and marking internal gap bridges."""
from pathlib import Path
import json,time,hashlib
import numpy as np
import pandas as pd
from scipy.interpolate import make_smoothing_spline, PchipInterpolator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
ROOT=Path(__file__).resolve().parents[1];SRC=ROOT/'testing/outputs/luke_kilosort_cid_splines_v1';COLORS=ROOT/'testing/outputs/luke_kilosort_cid_depth_scatter_v1/cid_color_and_counts.csv';OUT=ROOT/'testing/outputs/luke_kilosort_cid_splines_v2'
LAMBDA=.0001;GAP=1.;GRID=.05

def main():
 OUT.mkdir(exist_ok=True);begin=time.monotonic();b=pd.read_csv(SRC/'median_depth_knots.csv');c=pd.read_csv(COLORS);colors=c.set_index('cid').color.to_dict();rows=[];fits=[];bridges=[];lines=[];fit_start=time.monotonic()
 for cid,q in b.groupby('cid',sort=False):
  q=q.sort_values('time_s');t=q.time_s.to_numpy();y=q.depth_um.to_numpy()
  fits.append(dict(cid=cid,knots=len(q),status='fitted' if len(q)>=5 else 'too_few_knots',start_s=t[0],end_s=t[-1]))
  if len(q)<5:continue
  x=t-930;spline=make_smoothing_spline(x,y,lam=LAMBDA)
  xx=np.unique(np.r_[x,np.arange(x[0],x[-1],GRID)]);yy=spline(xx)
  # Bridge with a local shape-preserving cubic, avoiding unconstrained long-gap overshoot.
  bridge_curve=PchipInterpolator(x,spline(x),extrapolate=False)
  for j in np.flatnonzero(np.diff(x)>GAP):
   inside=(xx>x[j])&(xx<x[j+1]);yy[inside]=bridge_curve(xx[inside])
   assert np.all(yy[inside]>=min(spline(x[j]),spline(x[j+1]))-1e-8)
   assert np.all(yy[inside]<=max(spline(x[j]),spline(x[j+1]))+1e-8)
  assert np.isfinite(yy).all()
  mids=(xx[:-1]+xx[1:])/2;ii=np.clip(np.searchsorted(x,mids,side='right')-1,0,len(x)-2);gap=(x[ii+1]-x[ii])>GAP
  for j in np.flatnonzero(np.diff(x)>GAP):bridges.append(dict(cid=cid,start_s=t[j],end_s=t[j+1],gap_s=t[j+1]-t[j],left_spikes=int(q.spikes.iloc[j]),right_spikes=int(q.spikes.iloc[j+1])))
  knot_gap=np.r_[gap,False]
  rows.extend(dict(cid=cid,time_s=float(a+930),depth_um=float(z),following_segment_bridges_gap=bool(g)) for a,z,g in zip(xx,yy,knot_gap))
  coords=np.stack([xx+930,yy],axis=1);segments=np.stack([coords[:-1],coords[1:]],axis=1);lines.append((cid,segments,gap))
 fit_seconds=time.monotonic()-fit_start;r=pd.DataFrame(rows);r.to_csv(OUT/'spline_traces.csv',index=False);pd.DataFrame(fits).to_csv(OUT/'fit_units.csv',index=False);pd.DataFrame(bridges).to_csv(OUT/'bridged_intervals.csv',index=False)
 for stem,nrows,height in [('01_whole_probe',1,7),('02_depth_details',6,16)]:
  fig,axes=plt.subplots(nrows,1,figsize=(15,height),sharex=True,layout='constrained',squeeze=False)
  for ri,ax in enumerate(axes[:,0]):
   lo,hi=(0,3840) if nrows==1 else (ri*640,(ri+1)*640)
   for cid,segs,gap in lines:
    visible=(segs[:,:,1].max(axis=1)>=lo)&(segs[:,:,1].min(axis=1)<=hi)
    for bridge,style,alpha in [(False,'solid',.9),(True,'solid',.9)]:
     selected=visible&(gap==bridge)
     if selected.any():
      indices=np.flatnonzero(selected);runs=np.split(indices,np.flatnonzero(np.diff(indices)>1)+1)
      paths=[np.vstack([segs[run,0],segs[run[-1],1][None,:]]) for run in runs]
      ax.add_collection(LineCollection(paths,colors=colors[cid],linewidths=.65 if nrows==1 else .9,linestyles=style,alpha=alpha,rasterized=True))
   ax.set(xlim=(930,1030),ylim=(lo,hi),ylabel='Kilosort depth (µm)');ax.ticklabel_format(axis='x',style='plain',useOffset=False)
   if nrows>1:ax.set_title(f'{lo}–{hi} µm')
  # All portions are solid, as requested; support metadata remains in CSV.
  axes[-1,0].set_xlabel('Recording time (s)');fig.suptitle(f'Per-unit Kilosort depth traces · transitions bridged · 930–1,030 s\n{r.cid.nunique()} / {len(c)} active CIDs fitted · same CID colors',fontsize=14)
  fig.supxlabel('Fixed smoothing on0.25s median depths (λ=0.0001); shape-preserving cubic bridges across gaps; no tuning.\nSolid traces continue across internal gaps; same CID colors as the original scatter.',fontsize=9)
  for ext in ['png','pdf']:fig.savefig(OUT/f'{stem}.{ext}',dpi=180)
  plt.close(fig)
 summary=dict(status='complete',active_cids=len(c),fitted_cids=int(r.cid.nunique()),bridged_intervals=len(bridges),maximum_bridged_gap_s=float(max(a['gap_s'] for a in bridges)),fit_seconds=fit_seconds,total_seconds=time.monotonic()-begin,bridge_method='PCHIP between smoothed knots for gaps>1s; bridge bounded by smoothed endpoint depths',lambda_fixed=LAMBDA,median_bin_s=.25,minimum_knots=5,bridge_gap_threshold_s=GAP,display='All traces solid, including bridges, per user request',points_outside_probe=int(((r.depth_um<0)|(r.depth_um>3840)).sum()),no_extrapolation=True,source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [SRC/'median_depth_knots.csv',COLORS,Path(__file__)]})
 (OUT/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
