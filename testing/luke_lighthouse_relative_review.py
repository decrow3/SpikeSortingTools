"""Descriptive relative geometry of provisional gentle-period lighthouse tracks."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'testing/outputs/luke_lighthouse_relative_review_v1'
out.mkdir(exist_ok=True)
d=pd.read_csv(ROOT/'testing/outputs/luke_lighthouse_gentle_v1/gentle_tracks.csv')
z=d.pivot(index='time_s',columns='unit_id',values='median_waveform_centroid_um')
n=d.pivot(index='time_s',columns='unit_id',values='accepted_events')
depth=d.groupby('unit_id').depth_um.first().sort_values()
z=z[depth.index];n=n[depth.index]
delta=z-z.iloc[0]
pairs=pd.DataFrame({f'{a}–{b}':(z[b]-z[a])-(z[b].iloc[0]-z[a].iloc[0]) for a,b in zip(depth.index[:-1],depth.index[1:])})
fig,axs=plt.subplots(3,1,figsize=(12,12),layout='constrained',sharex=True)
for u in depth.index:
 line,=axs[0].plot(z.index,delta[u],'.-',label=f'{u}: {depth[u]:.0f}µm')
 sparse=n[u]<10
 axs[0].scatter(z.index[sparse],delta[u][sparse],marker='x',s=65,color=line.get_color())
axs[0].set(ylabel='Centroid change (µm)',title='Provisional cell displacement relative to first bin; × marks <10 accepted events')
axs[0].legend(ncol=3,fontsize=8)
for p in pairs:axs[1].plot(pairs.index,pairs[p],'.-',label=p)
axs[1].set(ylabel='Change in separation (µm)',title='Depth-adjacent cell pairs: constant relative position would give a horizontal line')
axs[1].legend(ncol=4,fontsize=8)
im=axs[2].imshow(n.T,aspect='auto',origin='lower',extent=[z.index.min()-5,z.index.max()+5,-.5,len(depth)-.5],cmap='viridis')
axs[2].set(yticks=np.arange(len(depth)),yticklabels=[str(u) for u in depth.index],ylabel='Unit',xlabel='Recording time (s)',title='Accepted event counts: interpretation must account for selection and missing support')
fig.colorbar(im,ax=axs[2],label='Events / 10s')
fig.suptitle('Relative lighthouse geometry · 4160–4260s\nExisting fixed-template centroids are provisional and can miss moving events; no DREDGE-based selection')
for ext in ['png','pdf']:fig.savefig(out/f'01_relative_geometry.{ext}',dpi=140)
delta.to_csv(out/'centroid_changes.csv');pairs.to_csv(out/'adjacent_separation_changes.csv')
step=(z.loc[4195]-z.loc[4185]).rename('4185_to_4195_centroid_change_um')
step.to_csv(out/'motion_step.csv')
print(step.to_string())
print('Pair separation change ranges (um):')
print((pairs.max()-pairs.min()).to_string())
print('Minimum absolute adjacent separation:', min((z[b]-z[a]).min() for a,b in zip(depth.index[:-1],depth.index[1:])))
