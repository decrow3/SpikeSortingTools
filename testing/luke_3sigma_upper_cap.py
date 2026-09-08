"""Reject detected peaks above8sigma from the existing3sigma motion input."""
import json,hashlib
import numpy as np
import pandas as pd
from spikeinterface.core import NumpyRecording
from testing.luke_epoch_corroboration import ROOT,BASE
from testing.luke_dredge_bounded import estimate_bounded
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
SRC=ROOT/'testing/outputs';PRE=SRC/'luke_detection_threshold_sweep_v1';OUT=SRC/'luke_3sigma_upper_cap_v1'
def main():
 OUT.mkdir(exist_ok=False);(OUT/'fields').mkdir();m=json.loads((BASE/'recording/rescue_recording_manifest.json').read_text());fs=m['sampling_frequency_hz'];geo=np.asarray(m['channel_locations_um']);noise_path=SRC/'luke_peak_threshold_screen_v1/noise_uv.npy';noise=np.load(noise_path);p=np.load(PRE/'3sigma_peaks.npy');y=np.load(PRE/'3sigma_locations.npy');snr=abs(p['amplitude'])/noise[p['channel_index']];k=snr<=8.;assert np.all(snr>=3-1e-4);cfg=json.loads((PRE/'settings.json').read_text())['estimator']
 settings=dict(interval_s=[4160,4260],input='Existing compensated fresh3sigma negative detections and unchanged localized positions',upper_limit='Reject events whose absolute DETECTED negative peak amplitude / frozen original detector-channel noise exceeds8; not clipping histogram weights or screening maximum full-waveform voltage',estimator=cfg,strict_pairwise_bound_um=80,noise_sha256=hashlib.sha256(noise_path.read_bytes()).hexdigest(),scope='Only upper-cap diagnostic authorized after pause; low-pass experiment remains paused. No sort or production changes',resume='No within-estimator checkpoint; preserve incomplete output if interrupted')
 (OUT/'settings.json').write_text(json.dumps(settings,indent=2));np.save(OUT/'keep_mask.npy',k);np.save(OUT/'peak_sigma.npy',snr);np.save(OUT/'retained_peaks.npy',p[k]);np.save(OUT/'retained_locations.npy',y[k]);np.save(OUT/'excluded_peaks.npy',p[~k]);np.save(OUT/'excluded_locations.npy',y[~k])
 old=np.load(PRE/'fields/3sigma.npz');np.savez_compressed(OUT/'fields/uncapped_3sigma.npz',**{key:old[key] for key in old.files});record=NumpyRecording(np.broadcast_to(np.zeros((1,384),dtype='float32'),(round(100*fs),384)),fs);record.set_channel_locations(geo);motion,extra=estimate_bounded(record,p[k],y[k],cfg);np.savez_compressed(OUT/'fields/capped_3to8sigma.npz',time_s=motion.temporal_bins_s[0]+4160,depth_um=motion.spatial_bins_um,displacement_um=motion.displacement[0],D=extra['D'],C=extra['C'],U=extra['U'])
 pd.DataFrame([dict(name='uncapped_3sigma',peaks=len(p),retained_fraction=1),dict(name='capped_3to8sigma',peaks=int(k.sum()),retained_fraction=float(k.mean()))]).to_csv(OUT/'manifest.csv',index=False)
 from testing import luke_screen_sweep as scoring
 scoring.OUT=OUT;scoring.analyze();fig=plt.gcf();fig.suptitle('Upper-amplitude rejection on3σ detections ·100s\nUnchanged compensation, noise vector, localization and DREDGE; provisional lighthouse agreement')
 for ext in ['png','pdf']:fig.savefig(OUT/f'01_summary.{ext}',dpi=140)
 d=pd.read_csv(OUT/'event_matched_predictions.csv');tracks=pd.read_csv(SRC/'luke_lighthouse_gentle_v1/gentle_tracks.csv');fig,axs=plt.subplots(3,3,figsize=(16,11),sharex=True,layout='constrained')
 for ax,(u,g) in zip(axs.ravel(),tracks.groupby('unit_id')):
  g=g.sort_values('time_s');base=g.iloc[0].median_waveform_centroid_um
  for name,label,color in [('uncapped_3sigma','3σ uncapped','#2878b5'),('capped_3to8sigma','3–8σ','#b13775')]:
   q=d[(d.name==name)&(d.unit_id==u)];ax.plot(q.time_s,q.predicted_um,'.-',label=label,color=color)
  good=g.accepted_events>=10;ax.errorbar(g.time_s[good],g.median_waveform_centroid_um[good]-base,yerr=np.array([g.median_waveform_centroid_um[good]-g.bootstrap_low_um[good],g.bootstrap_high_um[good]-g.median_waveform_centroid_um[good]]),fmt='ko',ms=4,capsize=2,label='Lighthouse');ax.set(title=f'Unit{u} · {g.depth_um.iloc[0]:.0f}µm',xlabel='Recording time (s)',ylabel='Relative displacement (µm)')
 axs[0,0].legend(fontsize=8);fig.suptitle('Does removing detected peaks above8σ improve the3σ input?\nMotion sampled at unchanged lighthouse event times; no gain, sign or lag fitted')
 for ext in ['png','pdf']:fig.savefig(OUT/f'02_lighthouse_overlay.{ext}',dpi=140)
 from testing import luke_peak_population_review as hist
 hist.OUT=OUT;hist.histogram([('3σ uncapped',p,y['y']),('3–8σ retained',p[k],y['y'][k]),('Above8σ excluded',p[~k],y['y'][~k])],4160,4260,fs,'03_histograms')
 rows=[]
 for band in range(4):
  q=(geo[p['channel_index'],1]>=band*960)&(geo[p['channel_index'],1]<(band+1)*960);rows.append(dict(depth_quarter=band,total=int(q.sum()),excluded=int((q&~k).sum()),excluded_amplitude_fraction=float(abs(p['amplitude'][q&~k]).sum()/abs(p['amplitude'][q]).sum())))
 pd.DataFrame(rows).to_csv(OUT/'depth_exclusion.csv',index=False)
 result=dict(status='complete',input_peaks=len(p),retained=int(k.sum()),excluded=int((~k).sum()),excluded_fraction=float((~k).mean()),excluded_amplitude_mass_fraction=float(abs(p['amplitude'][~k]).sum()/abs(p['amplitude']).sum()),scope=settings['scope']);(OUT/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
