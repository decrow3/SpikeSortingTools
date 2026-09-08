"""Check mask integrity and summarize retention without fitting motion."""
import json,hashlib
import numpy as np
import pandas as pd
from testing.luke_epoch_corroboration import ROOT,BASE
from testing.luke_multidepth_anchors_v2 import near
from testing.luke_motion_waveform_screen_v2 import OUT,SRC

def main():
 p=np.load(SRC/'luke_compensated_peak_trial_v1/peaks.npy');y=np.load(SRC/'luke_compensated_peak_trial_v1/locations.npy');k=np.load(OUT/'keep_mask.npy');d=pd.read_csv(OUT/'peak_decisions.csv')
 assert k.dtype==bool and len(k)==len(p)==len(y)
 assert np.array_equal(np.load(OUT/'retained_peaks.npy'),p[k])
 assert np.array_equal(np.load(OUT/'retained_locations.npy'),y[k])
 assert np.array_equal(np.load(OUT/'excluded_peaks.npy'),p[~k])
 assert d.peak_index.is_unique and np.array_equal(k[d.peak_index],d.keep)
 expected=(d.snr>=8)&(d.central_energy_fraction>=.65)&d.neighbor_coherent&(d.dominant_halfwidth_ms>=.067)&(d.dominant_halfwidth_ms<=.8)&(d.broad_residual_fraction<.2)&~((d.shared_explained>=.5)&(d.shared_cosine>=.8))
 assert np.array_equal(expected,d.keep)
 m=json.loads((BASE/'recording/rescue_recording_manifest.json').read_text());fs=m['sampling_frequency_hz'];geo=np.asarray(m['channel_locations_um'])
 d['probe_quarter']=np.minimum(3,(geo[d.channel,1]//960).astype(int));depth=d.groupby(['probe_quarter','half']).keep.agg(['size','sum']);depth['retained_fraction']=depth['sum']/depth['size'];depth.to_csv(OUT/'depth_time_retention.csv')
 light=pd.read_csv(SRC/'luke_lighthouse_gentle_v1/gentle_events.csv');dep=pd.read_csv(SRC/'luke_lighthouse_gentle_v1/gentle_tracks.csv').groupby('unit_id').depth_um.first();first=round(4180*fs);light=light[(light.frame>=first)&(light.frame<first+round(20*fs))];rows=[]
 for u,g in light.groupby('unit_id'):
  local=abs(geo[p['channel_index'],1]-dep[u])<=60;ev=g.frame.to_numpy()-first
  before=near(ev,p['sample_index'][local],.0008*fs);after=near(ev,p['sample_index'][local&k],.0008*fs)
  rows.append(dict(unit_id=int(u),reference_events=len(ev),coincident_before=int(before.sum()),coincident_after=int(after.sum())))
 pd.DataFrame(rows).to_csv(OUT/'lighthouse_coincidences.csv',index=False)
 snr=np.zeros(len(p),bool);snr[d.peak_index]=d.snr>=8
 from testing import luke_peak_population_review as hist
 hist.OUT=OUT
 hist.histogram([('Compensated input',p,y['y']),('Amplitude only: ≥8σ',p[snr],y['y'][snr]),('Amplitude + waveform screen',p[k],y['y'][k])],4180,4200,fs,'03_amplitude_control')
 files=[SRC/'luke_compensated_peak_trial_v1/peaks.npy',SRC/'luke_compensated_peak_trial_v1/locations.npy',SRC/'luke_common_event_screen_v1/shared_response_model.npz',ROOT/'testing/luke_motion_waveform_screen_v2.py']
 hashes={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
 r=dict(checks='passed: mask alignment, disjoint partition, stored arrays, explicit rule reconstruction',input_peaks=len(p),snr_only_retained=int(snr.sum()),waveform_retained=int(k.sum()),additional_excluded_beyond_snr=int((snr&~k).sum()),sha256=hashes,lighthouse_coincidences=rows,interpretation='Coincidences are descriptive only; no lighthouse-dependent thresholds or preservation gate')
 (OUT/'audit.json').write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True);print(depth.to_string())
if __name__=='__main__':main()
