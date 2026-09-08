"""Attribute retained local residuals and test neural-waveform preservation."""
from testing.luke_epoch_corroboration import ROOT,BASE
from testing.luke_multidepth_anchors_v2 import scores,near
import numpy as np,pandas as pd,json
from scipy.signal import butter,sosfiltfilt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=ROOT/'testing/outputs/luke_compensated_peak_trial_v1';PRE=ROOT/'testing/outputs/luke_common_event_screen_v1';DET=ROOT/'testing/outputs/luke_peak_threshold_screen_v1';LIGHT=ROOT/'testing/outputs/luke_lighthouse_gentle_v1'
def main():
 OUT.mkdir(exist_ok=False);assert json.loads((ROOT/'testing/outputs/luke_common_residual_review_v1/summary.json').read_text())['compensated_detection_gate_pass'];m=json.loads((BASE/'recording/rescue_recording_manifest.json').read_text());fs=m['sampling_frequency_hz'];loc=np.asarray(m['channel_locations_um']);n=round(20*fs);first=round(4180*fs);pad=round(.05*fs);model=np.load(PRE/'shared_response_model.npz');lags=model['lag_samples'];coeff=model['coefficients'];noise=model['residual_noise_uv'];original_noise=np.load(DET/'noise_uv.npy')
 settings=dict(interval_s=[4180,4200],model=str(PRE/'shared_response_model.npz'),model_refit=False,scope='Residual attribution plus protected neural waveform check; no classifier thresholds changed; original masks preserved',template_match='Existing independently trained lighthouse templates, weighted cosine>=0.9 and gain0.4–2.5; candidate templates within120um of residual maximum',compensation_gate='Every tracked lighthouse with >=10 events/half must retain median local waveform cosine>=0.9 and peak amplitude ratio0.8–1.2 in both halves before any compensated-input detection trial')
 (OUT/'settings.json').write_text(json.dumps(settings,indent=2))
 with (BASE/'recording/traces_cached_seg0.raw').open('rb') as f:f.seek((first-pad)*768);buf=f.read((n+2*pad)*768)
 x=np.frombuffer(buf,dtype='<i2').reshape(-1,384).astype('float32')*m['gain_uv_per_count'];sos=butter(3,[300,6000],fs=fs,btype='bandpass',output='sos');x=sosfiltfilt(sos,x,axis=0).astype('float32')[pad:pad+n].copy();del buf;reference=np.median(x,axis=1);prediction=np.zeros_like(x)
 for lo in range(16,n-16,10000):
  e=np.arange(lo,min(lo+10000,n-16));prediction[e]=reference[e[:,None]+lags]@coeff
 clean=x-prediction;post=x-reference[:,None];del x,prediction

 from spikeinterface.core import NumpyRecording
 from spikeinterface.sortingcomponents.peak_detection import detect_peaks
 from spikeinterface.sortingcomponents.peak_localization import localize_peaks
 from scipy.ndimage import gaussian_filter1d
 del post
 record=NumpyRecording(clean,fs);record.set_channel_locations(loc)
 (OUT/'trial_settings.json').write_text(json.dumps(dict(input='Shared-response-compensated voltage; frozen first-half model',detection='Fresh negative locally-exclusive, radius50um, threshold5 times ORIGINAL per-channel noise vector to isolate compensation',localization='Same monopolar triangulation radius75um ±0.5ms defaults',production_change=False),indent=2))
 peaks=detect_peaks(record,method='locally_exclusive',peak_sign='neg',radius_um=50.,detect_threshold=5.,noise_levels=original_noise,n_jobs=1,chunk_duration='1s',progress_bar=False);np.save(OUT/'peaks.npy',peaks);print(f'Compensated fresh detections: {len(peaks)}',flush=True)
 pos=localize_peaks(record,peaks,method='monopolar_triangulation',radius_um=75.,n_jobs=4,mp_context='fork',chunk_duration='1s',progress_bar=False);np.save(OUT/'locations.npy',pos);assert np.isfinite(pos['y']).all()
 original=np.load(DET/'peaks_5sigma.npy');original_pos=np.load(DET/'locations_5sigma.npy');light=pd.read_csv(LIGHT/'gentle_events.csv');light=light[(light.frame>=first)&(light.frame<first+n)];depths=pd.read_csv(LIGHT/'gentle_tracks.csv').groupby('unit_id').depth_um.first();checks=[]
 for cid,g in light.groupby('unit_id'):
  for name,p in [('Original',original),('Compensated',peaks)]:
   local=abs(loc[p['channel_index'],1]-depths[cid])<=60;hits=p['sample_index'][local]
   for half in [0,1]:
    ev=g.frame.to_numpy()-first;ev=ev[(ev>=half*10*fs)&(ev<(half+1)*10*fs)];count=int(near(ev,hits,.0008*fs).sum());checks.append(dict(unit_id=int(cid),half=half,arm=name,reference_events=len(ev),coincident_peaks=count))
 c=pd.DataFrame(checks);c.to_csv(OUT/'lighthouse_recovery.csv',index=False)
 rows=[];fig,axes=plt.subplots(1,3,figsize=(15,5));edges=np.arange(0,3841);depth=edges[:-1]+.5;shifts=np.arange(-20,21)
 for name,p,y in [('Original',original,original_pos['y']),('Compensated',peaks,pos['y'])]:
  profiles=[]
  for half in [0,1]:
   k=(p['sample_index']>=half*10*fs)&(p['sample_index']<(half+1)*10*fs);profiles.append(gaussian_filter1d(np.histogram(y[k],edges,weights=abs(p['amplitude'][k]))[0],1.))
  for ax,(lo,hi) in zip(axes[:2],[(1960,2560),(2260,2860)]):
   k=(depth>=lo)&(depth<hi);vals=[np.corrcoef(profiles[0][k],np.interp(depth[k]+s,depth,profiles[1]))[0,1] for s in shifts];rows.append(dict(arm=name,depth_lo_um=lo,depth_hi_um=hi,best_shift_um=int(shifts[np.argmax(vals)]),peak_correlation=max(vals),zero_correlation=vals[20]));ax.plot(shifts,vals,label=name);ax.set(title=f'Depth {lo}–{hi} µm',xlabel='After − before shift (µm)',ylabel='Profile correlation');ax.legend()
 g=c.groupby(['unit_id','arm']).coincident_peaks.sum().unstack();yy=np.arange(len(g));axes[2].barh(yy-.18,g.Original,height=.36,label='Original');axes[2].barh(yy+.18,g.Compensated,height=.36,label='Compensated');axes[2].set_yticks(yy,g.index.astype(str));axes[2].set(xlabel='Coincident peaks',ylabel='Independent lighthouse unit',title='Neural recovery with identical detection thresholds');axes[2].legend(fontsize=8);fig.suptitle('Fresh detection/localization after shared-response compensation\nFrozen model; original noise thresholds retained; pairwise profile diagnostic is not a full estimator replay',fontsize=11);fig.tight_layout(rect=[0,0,1,.9]);fig.savefig(OUT/'01_compensated_input.png',dpi=160);fig.savefig(OUT/'01_compensated_input.pdf');pd.DataFrame(rows).to_csv(OUT/'alignment_objective.csv',index=False)
 result=dict(status='complete',original_peaks=len(original),compensated_peaks=len(peaks),objective=rows,recovery=c.groupby('arm').coincident_peaks.sum().to_dict(),limitations=['Preservation assessed on a small preselected cohort; not a ground-truth false-rejection rate.','Same diagnostic epoch, not untouched prospective evaluation.','New locations/amplitudes are necessary because retained original artifact positions can be misleading.','Compensation may remove other biological shared components; no production adoption.']);(OUT/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
