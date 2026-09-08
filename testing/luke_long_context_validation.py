"""Extend early interval to100s while preserving central cached peak inputs exactly."""
import json
import numpy as np
from scipy.signal import butter,sosfiltfilt
from testing.luke_epoch_corroboration import ROOT,BASE
from testing.luke_dredge_bounded import estimate_bounded
from spikeinterface.core import NumpyRecording
from spikeinterface.sortingcomponents.peak_detection import detect_peaks
from spikeinterface.sortingcomponents.peak_localization import localize_peaks
from spikeinterface.sortingcomponents.motion import estimate_motion
OUT=ROOT/'testing/outputs/luke_long_context_validation_v1';SRC=ROOT/'testing/outputs'
def main():
 OUT.mkdir(exist_ok=False);m=json.loads((BASE/'recording/rescue_recording_manifest.json').read_text());fs=m['sampling_frequency_hz'];loc=np.asarray(m['channel_locations_um']);model=np.load(SRC/'luke_common_event_screen_v1/shared_response_model.npz');noise=np.load(SRC/'luke_peak_threshold_screen_v1/noise_uv.npy');cfg=json.loads((SRC/'luke_compensation_validation_v1/settings.json').read_text())['estimator'];settings=dict(interval_s=[930,1030],central_interval_s=[970,990],central_inputs='Reuse exact cached peaks and localizations; only surrounding context added',outer_chunks_s=[[s,s+20] for s in [930,950,990,1010]],model_refit=False,estimator=cfg,localization_workers=8,resume='Completed chunk arrays persist; no automatic restart or within-localization checkpoint',scope='Longer motion-estimation context, not amplitude completeness or a spike sort');(OUT/'settings.json').write_text(json.dumps(settings,indent=2));allp={a:[] for a in ['original','compensated']};allpos={a:[] for a in allp};basefirst=round(930*fs)
 for start in [930,950,970,990,1010]:
  first=round(start*fs);n=round(20*fs);pad=round(.05*fs)
  if start!=970:
   with (BASE/'recording/traces_cached_seg0.raw').open('rb') as f:f.seek((first-pad)*768);buf=f.read((n+2*pad)*768)
   x=np.frombuffer(buf,dtype='<i2').reshape(-1,384).astype('float32')*m['gain_uv_per_count'];x=sosfiltfilt(butter(3,[300,6000],fs=fs,btype='bandpass',output='sos'),x,axis=0).astype('float32');del buf;ref=np.median(x,axis=1);clean=np.empty((n,384),dtype='float32')
   for lo in range(0,n,10000):
    e=np.arange(lo,min(n,lo+10000));clean[e]=x[pad+e]-ref[pad+e[:,None]+model['lag_samples']]@model['coefficients']
   post=x[pad:pad+n]-ref[pad:pad+n,None];del x,ref
  for arm in allp:
   if start==970:
    p=np.load(SRC/f'luke_distant_motion_validation_v1/s970_{arm}_peaks.npy');positions=np.load(SRC/f'luke_distant_motion_validation_v1/s970_{arm}_locations.npy')
   else:
    record=NumpyRecording(post if arm=='original' else clean,fs);record.set_channel_locations(loc);p=detect_peaks(record,method='locally_exclusive',peak_sign='neg',radius_um=50.,detect_threshold=5.,noise_levels=noise,n_jobs=1,chunk_duration='1s',progress_bar=False);p=p[(p['sample_index']>=50)&(p['sample_index']<n-50)];np.save(OUT/f's{start}_{arm}_peaks.npy',p);print(start,arm,'peaks',len(p),flush=True);positions=localize_peaks(record,p,method='monopolar_triangulation',radius_um=75.,n_jobs=8,mp_context='fork',chunk_duration='1s',progress_bar=False);assert np.isfinite(positions['y']).all();np.save(OUT/f's{start}_{arm}_locations.npy',positions)
   pp=p.copy();pp['sample_index']+=first-basefirst;allp[arm].append(pp);allpos[arm].append(positions);print(start,arm,'chunk ready',flush=True)
  if start!=970:del post,clean,record
 record=NumpyRecording(np.broadcast_to(np.zeros((1,384),dtype='float32'),(round(100*fs),384)),fs);record.set_channel_locations(loc)
 for arm in allp:
  p=np.concatenate(allp[arm]);positions=np.concatenate(allpos[arm]);assert np.all(np.diff(p['sample_index'])>=0);np.save(OUT/f'{arm}_peaks.npy',p);np.save(OUT/f'{arm}_locations.npy',positions);motion,extra=estimate_bounded(record,p,positions,cfg);np.savez_compressed(OUT/f'{arm}_dredge_motion.npz',time_s=motion.temporal_bins_s[0]+930,depth_um=motion.spatial_bins_um,displacement_um=motion.displacement[0],**extra);print(arm,'100s DREDGE complete',flush=True)
  icfg=dict(direction='y',rigid=False,win_shape='gaussian',win_step_um=200.,win_scale_um=300.,win_margin_um=50.,method='iterative_template',progress_bar=False,verbose=False);im=estimate_motion(record,p,positions,**icfg);assert np.isfinite(im.displacement[0]).all();np.savez_compressed(OUT/f'{arm}_iterative_motion.npz',time_s=im.temporal_bins_s[0]+930,depth_um=im.spatial_bins_um,displacement_um=im.displacement[0]);print(arm,'100s iterative complete',flush=True)
 (OUT/'complete.json').write_text(json.dumps(dict(status='complete',counts={a:sum(map(len,allp[a])) for a in allp}),indent=2))
if __name__=='__main__':main()
