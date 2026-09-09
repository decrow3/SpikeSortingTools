"""Direct scatter of existing Kilosort-exported per-spike positions and CIDs."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from testing.luke_epoch_corroboration import ROOT,BASE
OUT=ROOT/'testing/outputs/luke_kilosort_cid_depth_scatter_v1'
def main():
 OUT.mkdir(exist_ok=True);source=BASE/'cur/cur_output';meta=BASE/'recording/rescue_recording_manifest.json';fs=json.loads(meta.read_text())['sampling_frequency_hz'];paths=[source/(n+'.npy') for n in ['spike_times','spike_clusters','spike_positions']];times,clusters,pos=[np.load(p,mmap_mode='r') for p in paths];assert len(times)==len(clusters)==len(pos)
 # Use the ordered export to select a bounded slice, then verify interval membership.
 a,b=np.searchsorted(times,[930*fs,1030*fs]);frames=np.asarray(times[a:b]);cid=np.asarray(clusters[a:b]);xy=np.asarray(pos[a:b]);t=frames/fs;y=xy[:,1];assert np.all((t>=930)&(t<1030)) and np.all(np.diff(frames)>=0)
 finite=np.isfinite(xy).all(axis=1);df=pd.DataFrame(dict(spike_index=np.arange(a,b),frame=frames,time_s=t,cid=cid,x_um=xy[:,0],depth_um=y));df.to_csv(OUT/'spikes_930_1030.csv',index=False)
 units=df.groupby('cid').agg(spikes=('time_s','size'),median_depth_um=('depth_um','median'),first_spike_s=('time_s','min'),last_spike_s=('time_s','max')).reset_index().sort_values('median_depth_um');palette=plt.get_cmap('tab20').colors;colors={int(c):to_hex(palette[i%20]) for i,c in enumerate(units.cid)};units['color']=units.cid.map(colors)
 labels=source/'cluster_KSLabel.tsv'
 if labels.exists():units=units.merge(pd.read_csv(labels,sep='\t'),left_on='cid',right_on='cluster_id',how='left')
 units.to_csv(OUT/'cid_color_and_counts.csv',index=False);np.savez_compressed(OUT/'spikes_930_1030.npz',spike_index=np.arange(a,b),frames=frames,cid=cid,xy_um=xy)
 # Interleave draw order deterministically so large/high-CID units do not always cover others.
 order=np.random.default_rng(20260908).permutation(np.flatnonzero(finite));cols=np.array([colors[int(c)] for c in cid])
 for stem,nrows,height in [('01_whole_probe',1,7),('02_depth_details',6,16)]:
  fig,axes=plt.subplots(nrows,1,figsize=(15,height),sharex=True,layout='constrained',squeeze=False)
  for r,ax in enumerate(axes[:,0]):
   lo,hi=(0,3840) if nrows==1 else (640*r,640*(r+1));ix=order[(y[order]>=lo)&(y[order]<(hi if hi<3840 else hi+1))]
   ax.scatter(t[ix],y[ix],c=cols[ix],s=.6 if nrows==1 else 1.8,alpha=.65,linewidths=0,rasterized=True)
   ax.set(xlim=(930,1030),ylim=(lo,hi),ylabel='Kilosort spike depth (µm)');ax.ticklabel_format(axis='x',style='plain',useOffset=False)
   if nrows>1:ax.set_title(f'{lo}–{hi} µm · {len(ix):,} spikes')
  axes[-1,0].set_xlabel('Recording time (s)');fig.suptitle(f'Kilosort spikes colored by cluster ID · 930–1,030 s\n{len(df):,} spikes · {len(units)} active CIDs · all labels included',fontsize=14)
  fig.supxlabel('Existing spike_positions.npy[:,1] and spike_clusters.npy; no new localization, screening, binning, or motion estimation.\nColors are categorical and repeat across the probe; exact CIDs and color mapping are saved in CSV. These are sorter-derived positions.',fontsize=9)
  for ext in ['png','pdf']:fig.savefig(OUT/f'{stem}.{ext}',dpi=180)
  plt.close(fig)
 summary=dict(status='complete',interval_s=[930,1030],sampling_frequency_hz=fs,spikes=len(df),active_cids=len(units),finite_positions=int(finite.sum()),export_slice=[int(a),int(b)],depth_source='spike_positions.npy[:,1], Kilosort per-spike exported coordinates (PC-feature weighted); not fixed per-unit depth or independent raw-voltage localization.',source_stat={str(p):dict(size=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in paths},slice_sha256=hashlib.sha256((OUT/'spikes_930_1030.npz').read_bytes()).hexdigest(),labels_included='All; no good/MUA/noise filtering',color_policy='tab20 repeated by increasing median cluster depth; fixed color per CID across both figures',script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
 (OUT/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
