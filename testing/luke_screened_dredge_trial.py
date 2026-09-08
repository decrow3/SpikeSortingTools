"""Matched bounded DREDGE replay: compensation, amplitude, and waveform screen."""
import json
import hashlib
import numpy as np
import pandas as pd
from spikeinterface.core import NumpyRecording
from testing.luke_epoch_corroboration import ROOT,BASE
from testing.luke_dredge_bounded import estimate_bounded
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=ROOT/'testing/outputs/luke_screened_dredge_trial_v1';SRC=ROOT/'testing/outputs'
def main():
 OUT.mkdir(exist_ok=False)
 m=json.loads((BASE/'recording/rescue_recording_manifest.json').read_text());fs=m['sampling_frequency_hz'];r=NumpyRecording(np.broadcast_to(np.zeros((1,384),dtype='float32'),(round(20*fs),384)),fs);r.set_channel_locations(np.asarray(m['channel_locations_um']))
 cfg=json.loads((SRC/'luke_compensated_dredge_trial_v1/settings.json').read_text())['estimator']
 p=np.load(SRC/'luke_compensated_peak_trial_v1/peaks.npy');y=np.load(SRC/'luke_compensated_peak_trial_v1/locations.npy');d=pd.read_csv(SRC/'luke_motion_waveform_screen_v2/peak_decisions.csv');amp=np.zeros(len(p),bool);amp[d.peak_index]=d.snr>=8;k=np.load(SRC/'luke_motion_waveform_screen_v2/keep_mask.npy')
 arms=[('Original',np.load(SRC/'luke_peak_threshold_screen_v1/peaks_5sigma.npy'),np.load(SRC/'luke_peak_threshold_screen_v1/locations_5sigma.npy')),('Compensated',p,y),('Amplitude only',p[amp],y[amp]),('Waveform screen',p[k],y[k])]
 settings=dict(interval_s=[4180,4200],estimator=cfg,strict_bounds_um=80,arm_counts={name:len(p) for name,p,y in arms},scope='20s input comparison, no production motion application or sort',metadata_recording='Broadcast zeros for clock and geometry only; all observations from cached peaks and positions',limits='Previously inspected interval; lighthouse centroids are provisional, not calibrated physical ground truth',screen_mask_sha256=hashlib.sha256((SRC/'luke_motion_waveform_screen_v2/keep_mask.npy').read_bytes()).hexdigest())
 (OUT/'settings.json').write_text(json.dumps(settings,indent=2));fields={};rows=[];constraints=[]
 tracks=pd.read_csv(SRC/'luke_lighthouse_gentle_v1/gentle_tracks.csv')
 for name,p,y in arms:
  motion,extra=estimate_bounded(r,p,y,cfg);a=motion.displacement[0];t=motion.temporal_bins_s[0]+4180;dep=motion.spatial_bins_um;fields[name]=(t,dep,a)
  slug=name.lower().replace(' ','_');np.savez_compressed(OUT/f'{slug}_motion.npz',time_s=t,depth_um=dep,displacement_um=a);np.savez_compressed(OUT/f'{slug}_constraints.npz',**extra)
  constraints.append(dict(arm=name,changed_constraints=int(extra['changed_constraints']),max_pairwise_displacement_um=float(abs(extra['D']).max())))
  for u,g in tracks.groupby('unit_id'):
   g=g.set_index('time_s');depth=float(g.depth_um.iloc[0]);v=np.array([np.interp(depth,dep,w) for w in a]);observed=float(g.loc[4195,'median_waveform_centroid_um']-g.loc[4185,'median_waveform_centroid_um'])
   rows.append(dict(arm=name,unit_id=int(u),depth_um=depth,dredge_step_um=float(np.median(v[t>=4190])-np.median(v[t<4190])),lighthouse_centroid_step_um=observed))
  print(name,'complete',flush=True)
 c=pd.DataFrame(rows);c.to_csv(OUT/'lighthouse_comparison.csv',index=False)
 fig,axs=plt.subplots(1,2,figsize=(13,6),layout='constrained');colors=dict(zip(fields,['gray','#2878b5','#d49a00','#b13775']))
 for name,(t,dep,a) in fields.items():
  delta=np.median(a[t>=4190],axis=0)-np.median(a[t<4190],axis=0);axs[0].plot(delta,dep,label=name,color=colors[name]);v=np.array([np.interp(2380,dep,w) for w in a]);axs[1].plot(t,v-np.median(v[t<4190]),label=name,color=colors[name])
 for _,q in c[c.arm=='Waveform screen'].iterrows():
  axs[0].scatter(q.lighthouse_centroid_step_um,q.depth_um,color='black',marker='x');axs[0].annotate(str(int(q.unit_id)),(q.lighthouse_centroid_step_um,q.depth_um),fontsize=7)
 axs[0].axvline(0,color='gray',lw=.5);axs[0].set(xlabel='Second-half − first-half displacement (µm)',ylabel='Depth (µm)',title='Motion step across depth',ylim=(0,3820));axs[0].legend(fontsize=8)
 axs[1].set(xlabel='Recording time (s)',ylabel='Displacement relative to first half (µm)',title='Example trajectory at 2380µm');axs[1].legend(fontsize=8)
 fig.suptitle('DREDGE after input cleanup · identical settings, strictly bounded ±80µm search\nBlack crosses: provisional lighthouse centroid steps; no motion agreement used to select peaks')
 for ext in ['png','pdf']:fig.savefig(OUT/f'01_motion_comparison.{ext}',dpi=150)
 result=dict(status='complete',constraints=constraints,comparisons=rows,scope=settings['scope']);(OUT/'summary.json').write_text(json.dumps(result,indent=2));print(c.to_string(index=False),flush=True)
if __name__=='__main__':main()
