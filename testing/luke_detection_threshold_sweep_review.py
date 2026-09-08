"""Audit threshold detections and export threshold-specific summaries."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from testing.luke_detection_threshold_sweep import OUT,LONG,SRC,keys
from testing.luke_epoch_corroboration import BASE

def main():
 m=json.loads((BASE/'recording/rescue_recording_manifest.json').read_text());fs=m['sampling_frequency_hz'];noise=np.load(SRC/'luke_peak_threshold_screen_v1/noise_uv.npy');checks=[];fields={}
 for s in [3,4,5,6]:
  p=np.load(OUT/f'{s}sigma_peaks.npy');y=np.load(OUT/f'{s}sigma_locations.npy');a=np.load(OUT/'fields'/f'{s}sigma.npz');assert len(p)==len(y);assert np.all(np.diff(p['sample_index'])>=0);assert np.all(p['amplitude']/noise[p['channel_index']]<=-s+1e-4);assert np.isfinite(y['y']).all();assert np.isfinite(a['displacement_um']).all() and abs(a['D']).max()<=80
  fields[s]=a;checks.append(dict(threshold_sigma=s,peaks=len(p),fraction_localized_outside_probe=float(np.mean((y['y']<0)|(y['y']>3820))),max_abs_field_um=float(abs(a['displacement_um']).max()),strict_pairwise_bound_um=float(abs(a['D']).max())))
 reused=[]
 for start in [4160,4180,4200,4220,4240]:
  old=np.load(LONG/f's{start}_compensated_peaks.npy');oldy=np.load(LONG/f's{start}_compensated_locations.npy');valid=(old['sample_index']>=50)&(old['sample_index']<round(20*fs)-50);p=np.load(OUT/f's{start}_5sigma_peaks.npy');y=np.load(OUT/f's{start}_5sigma_locations.npy');assert np.array_equal(keys(old[valid]),keys(p));assert np.array_equal(oldy[valid],y);reused.append(dict(start_s=start,interior_exact=True,old_boundary_peaks_removed=int((~valid).sum())))
 scores=pd.read_csv(OUT/'scores.csv').set_index('name');colors=['#37906c','#d49a00','#2878b5','#b13775'];names=[f'{s}sigma' for s in [3,4,5,6]];fig,axs=plt.subplots(1,3,figsize=(13,4),layout='constrained')
 for ax,col,title in zip(axs,['overall_mae_um','movement_mae_um','peaks'],['Overall lighthouse difference','Drop + recovery difference','Detected peaks']):
  ax.bar(['3σ','4σ','5σ','6σ'],scores.loc[names,col],color=colors);ax.set(title=title,ylabel='Mean absolute difference (µm)' if col!='peaks' else 'Count')
 fig.suptitle('Fresh detection thresholds on compensated voltage ·4160–4260s\nIdentical noise vector, localization and bounded DREDGE; descriptive agreement with provisional lighthouses')
 for ext in ['png','pdf']:fig.savefig(OUT/f'01_threshold_summary.{ext}',dpi=150)
 tracks=pd.read_csv(SRC/'luke_lighthouse_gentle_v1/gentle_tracks.csv');fig,axs=plt.subplots(3,3,figsize=(16,11),sharex=True,layout='constrained')
 for ax,(u,g) in zip(axs.ravel(),tracks.groupby('unit_id')):
  g=g.sort_values('time_s');depth=float(g.depth_um.iloc[0]);base=float(g.iloc[0].median_waveform_centroid_um)
  for s,color in zip([3,4,5,6],colors):
   a=fields[s];t=a['time_s'];v=np.array([np.interp(depth,a['depth_um'],w) for w in a['displacement_um']]);v-=np.median(v[t<4170]);ax.plot(t,v,label=f'{s}σ',color=color,alpha=.85)
  good=g.accepted_events>=10;ax.errorbar(g.time_s[good],g.median_waveform_centroid_um[good]-base,yerr=np.array([g.median_waveform_centroid_um[good]-g.bootstrap_low_um[good],g.bootstrap_high_um[good]-g.median_waveform_centroid_um[good]]),fmt='ko',ms=4,capsize=2,label='Lighthouse');ax.set(title=f'Unit{u} · {depth:.0f}µm',xlabel='Recording time (s)',ylabel='Relative displacement (µm)')
 axs[0,0].legend(fontsize=8);fig.suptitle('Continuous motion estimates across detection thresholds\nEach field offset by first10s temporal median; lighthouse points are10s summaries, not continuous verification')
 for ext in ['png','pdf']:fig.savefig(OUT/f'05_continuous_overlay.{ext}',dpi=140)
 (OUT/'audit.json').write_text(json.dumps(dict(status='passed',detections_and_fields=checks,baseline_reuse=reused,scope='Threshold eligibility, event/location correspondence and finite bounded fields checked; no neural purity claim'),indent=2));print(json.dumps(checks),flush=True);print(scores.loc[names,['peaks','overall_mae_um','movement_mae_um','step_p95_um']].to_string())
if __name__=='__main__':main()
