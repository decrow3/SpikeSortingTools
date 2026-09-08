"""Established iterative-template comparator on identical cached peak inputs."""
import inspect,json
import numpy as np
import pandas as pd
from testing.luke_epoch_corroboration import ROOT,BASE
from spikeinterface.core import NumpyRecording
from spikeinterface.sortingcomponents.motion import estimate_motion
from spikeinterface.sortingcomponents.motion.iterative_template import IterativeTemplateRegistration
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=ROOT/'testing/outputs/luke_alternative_registration_v1';SRC=ROOT/'testing/outputs/luke_distant_motion_validation_v1'
def main():
 OUT.mkdir(exist_ok=False);m=json.loads((BASE/'recording/rescue_recording_manifest.json').read_text());fs=m['sampling_frequency_hz'];rec=NumpyRecording(np.broadcast_to(np.zeros((1,384),dtype='float32'),(round(20*fs),384)),fs);rec.set_channel_locations(np.asarray(m['channel_locations_um']));cfg=dict(direction='y',rigid=False,win_shape='gaussian',win_step_um=200.,win_scale_um=300.,win_margin_um=50.,method='iterative_template',progress_bar=False,verbose=False);defaults={k:v.default for k,v in inspect.signature(IterativeTemplateRegistration.run).parameters.items() if v.default is not inspect.Parameter.empty};(OUT/'settings.json').write_text(json.dumps(dict(explicit=cfg,method_defaults=defaults,scope='Same cached peaks and geometry; method defaults include10um spatial/2s time bins and different search domains; method comparison, not a solver-only ablation'),indent=2));rows=[]
 comparison=pd.read_csv(SRC/'comparison.csv')
 for start in [970,9520]:
  for arm in ['original','compensated']:
   p=np.load(SRC/f's{start}_{arm}_peaks.npy');loc=np.load(SRC/f's{start}_{arm}_locations.npy');motion=estimate_motion(rec,p,loc,**cfg);a=motion.displacement[0];t=motion.temporal_bins_s[0]+start;dep=motion.spatial_bins_um;assert np.isfinite(a).all();np.savez_compressed(OUT/f's{start}_{arm}_motion.npz',time_s=t,depth_um=dep,displacement_um=a)
   for cid,g in comparison[(comparison.start_s==start)&(comparison.arm==arm)].groupby('candidate'):
    v=np.array([np.interp(g.depth_um.iloc[0],dep,w) for w in a])
    for r in g.itertuples():
     k=(t>=r.time_s-2.5)&(t<r.time_s+2.5);rows.append(dict(candidate=cid,arm=arm,start_s=start,time_s=r.time_s,centroid_um=r.centroid_um,dredge_um=r.dredge_um,iterative_um=float(np.median(v[k]))))
   print(start,arm,'complete',flush=True)
 d=pd.DataFrame(rows);d.to_csv(OUT/'comparison.csv',index=False);fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained');summary=[]
 for ax,(cid,g) in zip(axes.flat,d.groupby('candidate',sort=False)):
  valid=g[g.centroid_um.notna()];rt=valid.time_s.min();ref=valid[valid.time_s==rt].centroid_um.iloc[0]
  for arm,h in g.groupby('arm',sort=False):
   v=h.iterative_um-h[h.time_s==rt].iterative_um.iloc[0];ax.plot(h.time_s,v,label=f'Iterative {arm}',ls='--' if arm=='original' else '-');summary.append(dict(candidate=cid,arm=arm,iterative_range_um=float(h.iterative_um.max()-h.iterative_um.min()),dredge_range_um=float(h.dredge_um.max()-h.dredge_um.min())))
  h=g[g.arm=='compensated'];ax.plot(h.time_s,h.dredge_um-h[h.time_s==rt].dredge_um.iloc[0],':',label='DREDGE compensated',color='gray');ax.plot(h.time_s,h.centroid_um-ref,'kx-',label='Waveform centroid');ax.set(title=cid,xlabel='Recording time (s)',ylabel='Change from first valid waveform bin (µm)');ax.legend(fontsize=7)
 fig.suptitle('Established iterative-template comparison: same detected peaks\nMethod defaults differ; independent waveform observations remain provisional',fontsize=11)
 for ext in ['png','pdf']:fig.savefig(OUT/f'01_method_comparison.{ext}',dpi=150)
 (OUT/'summary.json').write_text(json.dumps(dict(status='complete',ranges=summary),indent=2));print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
