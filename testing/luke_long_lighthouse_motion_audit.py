"""Verify longer-run time alignment, cached reuse, and finite motion outputs."""
import json
import numpy as np
import pandas as pd
from testing.luke_long_lighthouse_motion import OUT,SRC
from testing.luke_epoch_corroboration import BASE

def main():
 m=json.loads((BASE/'recording/rescue_recording_manifest.json').read_text());fs=m['sampling_frequency_hz'];checks=[]
 for arm in ['original','compensated','screened']:
  p=np.load(OUT/f'{arm}_peaks.npy');y=np.load(OUT/f'{arm}_locations.npy');assert len(p)==len(y);assert np.all(np.diff(p['sample_index'])>=0);assert p['sample_index'].min()>=0 and p['sample_index'].max()<round(100*fs);assert np.isfinite(y['y']).all()
  a=np.load(OUT/f'{arm}_motion.npz');assert a['displacement_um'].shape==(len(a['time_s']),len(a['depth_um']));assert np.isfinite(a['displacement_um']).all();assert abs(a['D']).max()<=80
  start=round(4180*fs)-round(4160*fs);length=round(20*fs);selected=(p['sample_index']>=start)&(p['sample_index']<start+length);q=p[selected].copy();q['sample_index']-=start
  if arm=='original':prefix=SRC/'luke_peak_threshold_screen_v1';pp=np.load(prefix/'peaks_5sigma.npy');yy=np.load(prefix/'locations_5sigma.npy')
  elif arm=='compensated':prefix=SRC/'luke_compensated_peak_trial_v1';pp=np.load(prefix/'peaks.npy');yy=np.load(prefix/'locations.npy')
  else:prefix=SRC/'luke_motion_waveform_screen_v2';pp=np.load(prefix/'retained_peaks.npy');yy=np.load(prefix/'retained_locations.npy')
  assert np.array_equal(q,pp) and np.array_equal(y[selected],yy)
  checks.append(dict(arm=arm,peaks=len(p),central_cached_arrays_exact=True,finite_field=True,pairwise_bound_um=float(abs(a['D']).max()),max_abs_field_um=float(abs(a['displacement_um']).max())))
 d=pd.read_csv(OUT/'lighthouse_binned_comparison.csv');d['difference_from_centroid_um']=d.motion_bin_median_um-d.centroid_change_um
 stats=d[d.accepted_events>=10].groupby(['unit_id','depth_um','arm']).agg(supported_bins=('accepted_events','size'),median_absolute_difference_um=('difference_from_centroid_um',lambda v:float(np.median(abs(v)))),motion_min_um=('motion_bin_median_um','min'),motion_max_um=('motion_bin_median_um','max'))
 stats.to_csv(OUT/'descriptive_lighthouse_agreement.csv')
 (OUT/'audit.json').write_text(json.dumps(dict(status='passed',checks=checks,interpretation='Differences from provisional centroids are descriptive, not true displacement errors; baseline uncertainty not included in bin bootstrap bars.'),indent=2));print(json.dumps(checks),flush=True);print(stats.to_string())
if __name__=='__main__':main()
