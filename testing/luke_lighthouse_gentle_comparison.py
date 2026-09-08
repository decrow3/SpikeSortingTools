"""Compare fields at the same 10s resolution; retain conditional coverage."""
from testing.luke_epoch_corroboration import DATA,BASE,ROOT,SHARED
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=ROOT/'testing/outputs/luke_lighthouse_gentle_v1'
def main():
 t=pd.read_csv(OUT/'gentle_tracks.csv');events=pd.read_csv(OUT/'gentle_events.csv');fields=[];z=np.load(SHARED/'native_rigid_field.npz');fields.append(('Native rigid',z['time_s'],None,z['physical_displacement_um'].ravel()))
 for name in ['dredge-motion','decentralized-motion','ks-motion']:
  p=DATA/'dredge_pipeline_results_Luke0804_V2V1_g0_imec0/motion'/name;fields.append((name,np.load(p/'time_bins.npy').ravel()-3057.677050340359,np.load(p/'depth_bins.npy').ravel(),np.load(p/'motion.npy')))
 rows=[]
 for cid,g in t.groupby('unit_id'):
  depth=g.depth_um.iloc[0]
  for name,tt,dd,mm in fields:
   for r in g.itertuples():
    mask=(tt>=r.time_s-5)&(tt<r.time_s+5);v=mm[mask] if dd is None else np.array([np.interp(depth,dd,w) for w in mm[mask]])
    rows.append(dict(unit_id=cid,time_s=r.time_s,method=name,bin_median_um=float(np.median(v))))
 f=pd.DataFrame(rows);f.to_csv(OUT/'fields_10s_medians.csv',index=False)
 fig,axes=plt.subplots(3,2,figsize=(13,10),sharex=True)
 for ax,cid in zip(axes.ravel(),[154,317,445,463,510,587]):
  g=t[t.unit_id==cid];base=g.iloc[0].median_waveform_centroid_um;ax.plot(g.time_s,g.median_waveform_centroid_um-base,'ko-',label='Waveform centroid');ax.fill_between(g.time_s,g.bootstrap_low_um-base,g.bootstrap_high_um-base,color='k',alpha=.15)
  for name,_,_,_ in fields:
   h=f[(f.unit_id==cid)&(f.method==name)];ax.plot(h.time_s,h.bin_median_um-h.bin_median_um.iloc[0],label=name,lw=1)
  ax.set_title(f'Unit {cid} · {g.depth_um.iloc[0]:.0f} µm · {(g.accepted_events>=10).sum()}/10 bins');ax.set_ylabel('Relative position (µm)');ax.axhline(0,c='gray',lw=.4)
 axes[0,0].legend(fontsize=7,ncol=2);axes[-1,0].set_xlabel('Recording time (s)');axes[-1,1].set_xlabel('Recording time (s)');fig.suptitle('Gentle-epoch corroboration at matched 10 s resolution\nEach series referenced to its first bin; panel scales differ; centroid shifts are not calibrated motion\nBlack shading: conditional bootstrap interval; missing tracks remain gaps',fontsize=12);fig.tight_layout(rect=[0,0,1,.91]);fig.savefig(OUT/'04_matched_resolution.png',dpi=160);fig.savefig(OUT/'04_matched_resolution.pdf');plt.close(fig)
 # Reference availability alongside independent accepted-event counts; target labels never select tracks.
 ts=np.load(BASE/'cur/cur_output/spike_times.npy',mmap_mode='r').ravel();cl=np.load(BASE/'cur/cur_output/spike_clusters.npy',mmap_mode='r').ravel();fs=29999.835983263598;coverage=[]
 for start in range(4160,4260,10):
  ii=np.flatnonzero((ts>=round(start*fs))&(ts<round((start+10)*fs)));ids,counts=np.unique(cl[ii],return_counts=True);counts=dict(zip(ids,counts))
  for r in t[t.time_s==start+5].itertuples():coverage.append(dict(unit_id=r.unit_id,time_s=r.time_s,reference_events=int(counts.get(r.unit_id,0)),accepted_events=r.accepted_events))
 pd.DataFrame(coverage).to_csv(OUT/'reference_count_context.csv',index=False)
 print(pd.DataFrame(coverage).query('accepted_events<10').to_string(index=False))
if __name__=='__main__':main()
