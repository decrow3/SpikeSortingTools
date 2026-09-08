"""Controlled DREDGE replay on original versus compensated cached peak inputs."""
from testing.luke_epoch_corroboration import ROOT,BASE
import numpy as np,pandas as pd,json
from spikeinterface.core import NumpyRecording
from spikeinterface.sortingcomponents.motion import estimate_motion
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=ROOT/'testing/outputs/luke_compensated_dredge_trial_v1'
def main():
 OUT.mkdir(exist_ok=False);m=json.loads((BASE/'recording/rescue_recording_manifest.json').read_text());fs=m['sampling_frequency_hz'];n=round(20*fs);record=NumpyRecording(np.broadcast_to(np.zeros((1,384),dtype='float32'),(n,384)),fs);record.set_channel_locations(np.asarray(m['channel_locations_um']));config=dict(direction='y',rigid=False,win_shape='gaussian',win_step_um=200.,win_scale_um=300.,win_margin_um=50.,method='dredge_ap',extra_outputs=False,progress_bar=False,verbose=False,bin_um=1.,bin_s=1.,histogram_time_smooth_s=1.,histogram_depth_smooth_um=1.,time_horizon_s=60.,max_disp_um=80.,mincorr=.1,device='cpu');(OUT/'settings.json').write_text(json.dumps(dict(estimator=config,interval_s=[4180,4200],metadata_recording='Broadcast zero array supplies only clock/geometry. Cached detected peaks and localizations supply all estimation observations.',limitation='20s bounded comparison; settings identical across arms but not recovered full-recording historical settings.'),indent=2));series={}
 for name,prefix,pfile,lfile in [('Original','luke_peak_threshold_screen_v1','peaks_5sigma.npy','locations_5sigma.npy'),('Compensated','luke_compensated_peak_trial_v1','peaks.npy','locations.npy')]:
  p=ROOT/'testing/outputs'/prefix;peaks=np.load(p/pfile);locations=np.load(p/lfile);motion=estimate_motion(record,peaks,locations,**config);a=motion.displacement[0];t=motion.temporal_bins_s[0]+4180;dep=motion.spatial_bins_um;np.savez_compressed(OUT/f'{name.lower()}_motion.npz',displacement_um=a,time_s=t,depth_um=dep);series[name]=(t,dep,a);print(f'{name} DREDGE complete {a.shape}',flush=True)
 tracks=pd.read_csv(ROOT/'testing/outputs/luke_lighthouse_gentle_v1/gentle_tracks.csv');rows=[];fig,axes=plt.subplots(1,2,figsize=(12,5))
 for name,(t,dep,a) in series.items():
  delta=np.median(a[t>=4190],axis=0)-np.median(a[t<4190],axis=0);axes[0].plot(delta,dep,label=name)
  for cid,depth in [(317,1740),(445,2260),(463,2380),(510,2740)]:
   v=np.array([np.interp(depth,dep,w) for w in a]);step=float(np.median(v[t>=4190])-np.median(v[t<4190]));g=tracks[tracks.unit_id==cid].set_index('time_s');observed=float(g.loc[4195,'median_waveform_centroid_um']-g.loc[4185,'median_waveform_centroid_um']);rows.append(dict(arm=name,unit_id=cid,depth_um=depth,dredge_step_um=step,lighthouse_centroid_step_um=observed))
 for cid in [317,445,463,510]:
  g=tracks[tracks.unit_id==cid].set_index('time_s');step=g.loc[4195,'median_waveform_centroid_um']-g.loc[4185,'median_waveform_centroid_um'];axes[0].scatter(step,g.depth_um.iloc[0],c='k',marker='x');axes[0].annotate(str(cid),(step,g.depth_um.iloc[0]),fontsize=8)
 axes[0].set(xlabel='Second-half − first-half displacement (µm)',ylabel='Depth (µm)',title='DREDGE response across probe depth',ylim=(0,3820));axes[0].axvline(0,c='gray',lw=.5);axes[0].legend()
 for name,(t,dep,a) in series.items():
  v=np.array([np.interp(2380,dep,w) for w in a]);axes[1].plot(t,v-np.median(v[t<4190]),label=name)
 axes[1].set(xlabel='Recording time (s)',ylabel='Relative displacement (µm)',title='Example: estimated movement at 2380 µm');axes[1].legend();fig.suptitle('Same DREDGE settings; only shared-response compensation and redetection differ\nBlack crosses: descriptive lighthouse-centroid changes, not calibrated displacement ground truth',fontsize=11);fig.tight_layout(rect=[0,0,1,.89]);fig.savefig(OUT/'01_dredge_comparison.png',dpi=160);fig.savefig(OUT/'01_dredge_comparison.pdf');pd.DataFrame(rows).to_csv(OUT/'lighthouse_comparison.csv',index=False);(OUT/'summary.json').write_text(json.dumps(dict(status='complete',comparisons=rows,scope='Bounded input experiment, no production motion application or sort'),indent=2));print(json.dumps(rows),flush=True)
if __name__=='__main__':main()
