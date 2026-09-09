"""Full-range displacement comparison, without clipping raster zoom excursions."""
from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
ROOT=Path(__file__).resolve().parents[1];SRC=ROOT/'testing/outputs';OUT=SRC/'luke_early_medicine_peak_comparison_v8';OLD=SRC/'luke_early_medicine_peak_comparison_v1'
fig,axs=plt.subplots(3,2,figsize=(14,9),sharex=True,sharey='row',layout='constrained')
for col,arm in enumerate(['original','compensated']):
 z=np.load(OUT/f'{arm}_dredge_motion.npz')
 methods=[('DREDGE · 0.5 s Gaussian',z['time_s'],z['depth_um'],z['displacement_um'],'#0072B2',3)]
 for name,src,c in [('MEDiCINe · 1 s kernel',SRC/'luke_early_medicine_peak_comparison_v6','#D55E00')]:
  methods.append((name,np.load(src/arm/'time_bins.npy'),np.load(src/arm/'depth_bins.npy'),np.load(src/arm/'motion.npy'),c,1.6))
 old=np.load(SRC/'luke_early_medicine_peak_comparison_v6'/f'{arm}_dredge_motion.npz')
 methods.append(('DREDGE · 1 s Gaussian',old['time_s'],old['depth_um'],old['displacement_um'],'#777777',1.5))
 for row,depth in enumerate([2250,2919,3381]):
  ax=axs[row,col]
  for name,t,y,d,c,lw in methods:
   tr=RegularGridInterpolator((t,y),d)(np.column_stack([t,np.full(len(t),depth)]));tr-=np.median(tr[(t>=970)&(t<975)])
   ax.plot(t,tr,color=c,lw=lw,label=name)
  ax.set(title=f'{arm} · {depth} µm',ylabel='Displacement (µm)',xlim=(930,1030));ax.grid(alpha=.2)
  if row==0:ax.legend(fontsize=8)
 axs[-1,col].set_xlabel('Recording time (s)')
fig.suptitle('DREDGE versus MEDiCINe · 0.25 s time bins · DREDGE smoothing comparison · same peaks · full displacement range\nAll traces referenced to their median at 970–975 s; shared vertical scale within each row')
for ext in ['png','pdf']:fig.savefig(OUT/f'02_full_displacement_comparison.{ext}',dpi=170)
plt.close(fig)
