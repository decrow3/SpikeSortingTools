"""Full-range displacement comparison, without clipping raster zoom excursions."""
from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
ROOT=Path(__file__).resolve().parents[1];SRC=ROOT/'testing/outputs';OUT=SRC/'luke_early_medicine_peak_comparison_v2';OLD=SRC/'luke_early_medicine_peak_comparison_v1'
fig,axs=plt.subplots(3,2,figsize=(14,9),sharex=True,sharey='row',layout='constrained')
for col,arm in enumerate(['original','compensated']):
 z=np.load(OUT/f'{arm}_dredge_motion.npz')
 methods=[('DREDGE fresh',z['time_s'],z['depth_um'],z['displacement_um'],'#0072B2',3)]
 for name,src,c in [('MEDiCINe 1 s',OUT,'#D55E00'),('MEDiCINe 50 s',OLD,'#777777')]:
  methods.append((name,np.load(src/arm/'time_bins.npy'),np.load(src/arm/'depth_bins.npy'),np.load(src/arm/'motion.npy'),c,1.6))
 for row,depth in enumerate([2250,2919,3381]):
  ax=axs[row,col]
  for name,t,y,d,c,lw in methods:
   tr=RegularGridInterpolator((t,y),d)(np.column_stack([t,np.full(len(t),depth)]));tr-=np.median(tr[(t>=970)&(t<975)])
   ax.plot(t,tr,color=c,lw=lw,label=name)
  ax.set(title=f'{arm} · {depth} µm',ylabel='Displacement (µm)',xlim=(930,1030));ax.grid(alpha=.2)
  if row==0:ax.legend(fontsize=8)
 axs[-1,col].set_xlabel('Recording time (s)')
fig.suptitle('Exact same peak populations · full displacement range\nAll traces referenced to their median at 970–975 s; shared vertical scale within each row')
for ext in ['png','pdf']:fig.savefig(OUT/f'02_full_displacement_comparison.{ext}',dpi=170)
plt.close(fig)
(OUT/'README.md').write_text('''# Early peak motion comparison v2

Fresh DREDGE and MEDiCINe fits on the exact original/compensated 930–1030 s detector peaks behind the earlier raster. Histogram identity is asserted. No voltage reads, new detections, sorting, or localization.

DREDGE settings unchanged: 1 s bins, 1 s histogram temporal smoothing, 200 µm spatial window step / 300 µm scale, strict ±80 µm pairwise search. Fresh fits differ from cached fields by at most 0.008 µm.

MEDiCINe changes only time_kernel_width from 50 s to 1 s versus v1: 1 s time bins, two depth bins, 10,000 training steps, seed 0, all peaks retained and absolute amplitudes supplied. Fits ran on CUDA. These methods still have different spatial representations and bounds.

01 preserves the old raster zooms and thickens DREDGE to 3 pt. Some MEDiCINe excursions exceed these zooms; 02 displays their full extent and includes the previous 50 s fit. All offsets use the 970–975 s median independently. At 2250 µm compensated DREDGE spans 83.24 µm, MEDiCINe 1 s spans 111.55 µm (50 s: 10.52 µm). Larger excursions alone do not establish more accurate motion.

Independent systemd service luke-early-medicine-peaks-v2; commands/logs/stage exit receipts in sibling luke_early_medicine_peak_comparison_v2_job. Completed stages persist; interruption inside training requires restarting that arm. No within-fit checkpoint claimed.
''')
