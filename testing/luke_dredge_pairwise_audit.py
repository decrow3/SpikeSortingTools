"""Save unmodified DREDGE constraints for the temporal validation inputs."""
import json
import numpy as np
import pandas as pd
from testing.luke_epoch_corroboration import ROOT,BASE
from spikeinterface.core import NumpyRecording
from spikeinterface.sortingcomponents.motion import estimate_motion
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=ROOT/'testing/outputs/luke_dredge_pairwise_audit_v1'
SRC=ROOT/'testing/outputs/luke_compensation_validation_v1'
def main():
 OUT.mkdir(exist_ok=False)
 m=json.loads((BASE/'recording/rescue_recording_manifest.json').read_text());fs=m['sampling_frequency_hz']
 rec=NumpyRecording(np.broadcast_to(np.zeros((1,384),dtype='float32'),(round(20*fs),384)),fs);rec.set_channel_locations(np.array(m['channel_locations_um']))
 cfg=json.loads((SRC/'settings.json').read_text())['estimator'];cfg['extra_outputs']=True
 (OUT/'settings.json').write_text(json.dumps(cfg,indent=2)); rows=[]
 for arm in ['original','compensated']:
  motion,extra=estimate_motion(rec,np.load(SRC/f'{arm}_peaks.npy'),np.load(SRC/f'{arm}_locations.npy'),**cfg)
  old=np.load(SRC/f'{arm}_motion.npz')['displacement_um'];assert np.allclose(motion.displacement[0],old,atol=1e-5)
  arrays={k:v for k,v in extra.items() if isinstance(v,np.ndarray)};np.savez_compressed(OUT/f'{arm}_constraints.npz',**arrays)
  print(arm,{k:np.shape(v) for k,v in arrays.items()},flush=True)
  D=extra['D'];C=extra['C'];U=extra['U'];dep=extra['window_centers'];fig,ax=plt.subplots(2,3,figsize=(13,8),layout='constrained')
  for row,depth in enumerate([400,2400]):
   b=np.argmin(abs(dep-depth))
   for a,key,val in zip(ax[row],['Pairwise displacement (µm)','Correlation','Solver weight'],[D[b],C[b],U[b]]):
    im=a.imshow(val,origin='lower',extent=[4240,4260,4240,4260],aspect='auto',cmap='coolwarm' if key.startswith('Pair') else 'viridis');fig.colorbar(im,ax=a);a.set(title=f'{dep[b]:.0f} µm · {key}',xlabel='Second time (s)',ylabel='First time (s)')
  fig.suptitle(f'{arm.title()} input: unchanged DREDGE pairwise evidence')
  for ext in ['png','pdf']:fig.savefig(OUT/f'{arm}_constraints.{ext}',dpi=150)
  for b,depth in enumerate(dep):
   cross=D[b,:10,10:];cc=C[b,:10,10:];uu=U[b,:10,10:]
   rows.append(dict(arm=arm,depth_um=float(depth),cross_D_median=float(np.median(cross)),cross_C_median=float(np.median(cc)),cross_D_abs_gt20_fraction=float(np.mean(abs(cross)>20)),cross_weight_large_fraction=float(uu[abs(cross)>20].sum()/uu.sum()) if uu.sum() else None,antisymmetry_median_um=float(np.median(abs(D[b]+D[b].T)))))
 pd.DataFrame(rows).to_csv(OUT/'constraint_summary.csv',index=False)
 (OUT/'summary.json').write_text(json.dumps(dict(status='complete',replay_matches_saved_fields=True),indent=2))
if __name__=='__main__':main()
