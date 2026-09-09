"""Cheap ranking and review of movement in cached Kilosort positions."""
from pathlib import Path
import json,io
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from testing.luke_epoch_corroboration import ROOT,BASE
OUT=ROOT/'testing/outputs/luke_kilosort_largest_movers_v1';SRC=ROOT/'testing/outputs'
def main():
 OUT.mkdir(exist_ok=True);knots=pd.read_csv(SRC/'luke_kilosort_cid_splines_v1/median_depth_knots.csv');events=pd.read_csv(SRC/'luke_kilosort_cid_depth_scatter_v1/spikes_930_1030.csv');units=pd.read_csv(SRC/'luke_kilosort_cid_depth_scatter_v1/cid_color_and_counts.csv');traces=pd.read_csv(SRC/'luke_kilosort_cid_splines_v2/spline_traces.csv');rows=[]
 for cid,q in knots.groupby('cid'):
  rows.append(dict(cid=cid,occupied_bins=len(q),spikes=int(q.spikes.sum()),median_depth_um=q.depth_um.median(),depth_p05_um=q.depth_um.quantile(.05),depth_p95_um=q.depth_um.quantile(.95),robust_span_um=q.depth_um.quantile(.95)-q.depth_um.quantile(.05),full_knot_range_um=q.depth_um.max()-q.depth_um.min(),eligible=len(q)>=20 and q.spikes.sum()>=100))
 ranking=pd.DataFrame(rows).sort_values('robust_span_um',ascending=False);ranking.to_csv(OUT/'all_unit_movement_ranking.csv',index=False);top=ranking[ranking.eligible].head(12).copy();top.to_csv(OUT/'top12.csv',index=False);colors=units.set_index('cid').color.to_dict()
 # Check saved localization bounds and template identities on this exact export slice.
 path=BASE/'cur/cur_output';torch.storage._load_from_bytes=lambda b:torch.load(io.BytesIO(b),map_location='cpu',weights_only=False)
 ops=np.load(path/'ops.npy',allow_pickle=True).item();settings=ops.get('settings',{});ic=np.asarray(ops['iCC']);mask=np.asarray(ops['iCC_mask']);iu=np.asarray(ops['iU']);yc=np.asarray(ops['yc']);td=np.load(path/'spike_detection_templates.npy',mmap_mode='r');dt=np.asarray(td[events.spike_index.to_numpy()]);anchor=iu[dt];depth_lo=np.array([yc[ic[:,i][mask[:,i]]].min() for i in range(len(yc))]);depth_hi=np.array([yc[ic[:,i][mask[:,i]]].max() for i in range(len(yc))]);outside=(events.depth_um.to_numpy()<depth_lo[anchor]-1e-3)|(events.depth_um.to_numpy()>depth_hi[anchor]+1e-3)
 bounds=[]
 for cid,g in events.groupby('cid'):
  idx=g.index.to_numpy();bounds.append(dict(cid=cid,detection_templates=len(np.unique(dt[idx])),anchor_depth_min_um=float(yc[anchor[idx]].min()),anchor_depth_max_um=float(yc[anchor[idx]].max()),spikes_outside_template_channel_bounds=int(outside[idx].sum())))
 pd.DataFrame(bounds).to_csv(OUT/'unit_detection_template_bounds.csv',index=False)
 report=dict(kilosort_version='4.0.27 per recording manifest',saved_settings={k:settings.get(k,ops.get(k)) for k in ['nearest_chans','position_limit','nblocks','binning_depth','batch_size','drift_smoothing','sig_interp','Th_learned','Th_universal']},saved_dshift_is_none=ops.get('dshift') is None,exported_spikes_checked=len(events),outside_template_channel_bounds=int(outside.sum()),ranking='P95−P05 of occupied0.25s median-depth knots, not interpolated samples; displayed units require≥20knots and≥100spikes.',sort_identity=json.loads((BASE/'rescue_sort_identity.json').read_text())['sort_summary']['critical_saved_settings'])
 (OUT/'settings_and_bounds.json').write_text(json.dumps(report,indent=2))
 fig,axes=plt.subplots(4,3,figsize=(16,12),sharex=True,layout='constrained')
 for ax,u in zip(axes.flat,top.itertuples()):
  e=events[events.cid==u.cid];k=knots[knots.cid==u.cid];s=traces[traces.cid==u.cid];ax.scatter(e.time_s,e.depth_um,s=3,c='#888888',alpha=.25,rasterized=True);ax.plot(s.time_s,s.depth_um,color=colors[u.cid],lw=1.3);ax.scatter(k.time_s,k.depth_um,s=5,c='#222222',alpha=.5,rasterized=True);ax.set(title=f'CID {u.cid} · 90% span {u.robust_span_um:.1f} µm\n{u.spikes:,} spikes · {u.occupied_bins} occupied bins',xlim=(930,1030));ax.ticklabel_format(axis='x',style='plain',useOffset=False);ax.grid(alpha=.12)
 for ax in axes[:,0]:ax.set_ylabel('Kilosort depth (µm)')
 for ax in axes[-1]:ax.set_xlabel('Recording time (s)')
 fig.suptitle('Twelve largest observed depth spans · per-unit Kilosort positions\nGray: original spikes · black:0.25s medians · color: existing continuous spline',fontsize=14)
 fig.supxlabel('Ranked on observed median-depth knots, not gap interpolation. ≥20 occupied bins and≥100spikes. Each panel has its own depth scale.\nLarge depth span is a candidate movement signal; cluster mixing and localization support can also contribute.',fontsize=9)
 for ext in ['png','pdf']:fig.savefig(OUT/f'01_top12_traces.{ext}',dpi=170)
 plt.close(fig)
 fig,ax=plt.subplots(figsize=(15,7),layout='constrained')
 for u in top.itertuples():
  q=traces[traces.cid==u.cid];ax.plot(q.time_s,q.depth_um,color=colors[u.cid],lw=1.0,label=f'{u.cid}: {u.robust_span_um:.0f} µm')
 ax.set(xlim=(930,1030),ylim=(0,3840),xlabel='Recording time (s)',ylabel='Kilosort depth (µm)',title='Largest observed movers · same depth/time coordinates and CID colors');ax.legend(ncol=4,fontsize=8);ax.ticklabel_format(axis='x',style='plain',useOffset=False)
 for ext in ['png','pdf']:fig.savefig(OUT/f'02_top12_whole_probe.{ext}',dpi=170)
 plt.close(fig);print(top[['cid','robust_span_um','full_knot_range_um','spikes']].to_string(index=False));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
