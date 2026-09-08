"""Frozen original/compensated DREDGE comparison at new early and late holdouts."""
import json
import numpy as np
import pandas as pd
from scipy.signal import butter,sosfiltfilt
from testing.luke_epoch_corroboration import ROOT,BASE
from testing.luke_dredge_bounded import estimate_bounded
from spikeinterface.core import NumpyRecording
from spikeinterface.sortingcomponents.peak_detection import detect_peaks
from spikeinterface.sortingcomponents.peak_localization import localize_peaks
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=ROOT/'testing/outputs/luke_distant_motion_validation_v1';SRC=ROOT/'testing/outputs';HOLD=SRC/'luke_transfer_template_holdout_v1'
def main():
 OUT.mkdir(exist_ok=False);m=json.loads((BASE/'recording/rescue_recording_manifest.json').read_text());fs=m['sampling_frequency_hz'];loc=np.asarray(m['channel_locations_um']);model=np.load(SRC/'luke_common_event_screen_v1/shared_response_model.npz');noise=np.load(SRC/'luke_peak_threshold_screen_v1/noise_uv.npy');cfg=json.loads((SRC/'luke_compensation_validation_v1/settings.json').read_text())['estimator'];tracks=pd.read_csv(HOLD/'tracks.csv');evs=pd.read_csv(HOLD/'matched_events.csv');cohorts=pd.read_csv(SRC/'luke_independent_transfer_review_v1/candidate_screen.csv').set_index('candidate');rows=[];checks=[];counts=[]
 (OUT/'settings.json').write_text(json.dumps(dict(intervals_s=[[970,990],[9520,9540]],model_refit=False,estimator=cfg,search_bounds='Strict abs(lag)<=80 enforced and baseline D/C reconstruction verified in both arms',detection='Fresh negative locally exclusive50um,5sigma fixed development noise',localization='Monopolar triangulation75um defaults',waveform_check='Same independent heldout matched events in5s bins>=10events; cosine>=.9 amplitude ratio.8–1.2',scope='Independent provisional waveform observations; sparse identity competitors and deep-only coverage; no production correction or sort'),indent=2))
 fields={}
 for start in [970,9520]:
  first=round(start*fs);n=round(20*fs);pad=round(.05*fs)
  with (BASE/'recording/traces_cached_seg0.raw').open('rb') as f:f.seek((first-pad)*768);buf=f.read((n+2*pad)*768)
  x=np.frombuffer(buf,dtype='<i2').reshape(-1,384).astype('float32')*m['gain_uv_per_count'];x=sosfiltfilt(butter(3,[300,6000],fs=fs,btype='bandpass',output='sos'),x,axis=0).astype('float32');del buf;ref=np.median(x,axis=1);clean=np.empty((n,384),dtype='float32')
  for lo in range(0,n,10000):
   e=np.arange(lo,min(n,lo+10000));clean[e]=x[pad+e]-ref[pad+e[:,None]+model['lag_samples']]@model['coefficients']
  post=x[pad:pad+n]-ref[pad:pad+n,None];del x,ref
  for candidate,g in evs[evs.start_s==start].groupby('candidate'):
   depth=cohorts.loc[candidate,'depth_um'];ch=np.flatnonzero(abs(loc[:,1]-depth)<=60)
   for b in range(4):
    e=g.frame.to_numpy()-first;e=e[(e>=b*5*fs)&(e<(b+1)*5*fs)]
    if len(e)<10:continue
    off=np.arange(-30,31);a=np.median(post[e[:,None,None]+off[None,:,None],ch[None,None,:]],axis=0);v=np.median(clean[e[:,None,None]+off[None,:,None],ch[None,None,:]],axis=0);cos=float(np.sum(a*v)/(np.linalg.norm(a)*np.linalg.norm(v)));ratio=float(abs(v).max()/abs(a).max());checks.append(dict(start_s=start,candidate=candidate,bin=b,events=len(e),cosine=cos,amplitude_ratio=ratio,passes=cos>=.9 and .8<=ratio<=1.2))
  pd.DataFrame(checks).to_csv(OUT/'preservation.csv',index=False)
  assert all(r['passes'] for r in checks),'Preservation gate failed; do not proceed to compensated estimation'
  for arm,voltage in [('original',post),('compensated',clean)]:
   record=NumpyRecording(voltage,fs);record.set_channel_locations(loc);peaks=detect_peaks(record,method='locally_exclusive',peak_sign='neg',radius_um=50.,detect_threshold=5.,noise_levels=noise,n_jobs=1,chunk_duration='1s',progress_bar=False);peaks=peaks[(peaks['sample_index']>=50)&(peaks['sample_index']<n-50)];np.save(OUT/f's{start}_{arm}_peaks.npy',peaks);print(start,arm,'peaks',len(peaks),flush=True)
   positions=localize_peaks(record,peaks,method='monopolar_triangulation',radius_um=75.,n_jobs=4,mp_context='fork',chunk_duration='1s',progress_bar=False);assert np.isfinite(positions['y']).all();np.save(OUT/f's{start}_{arm}_locations.npy',positions);motion,extra=estimate_bounded(record,peaks,positions,cfg);a=motion.displacement[0];t=motion.temporal_bins_s[0]+start;dep=motion.spatial_bins_um;np.savez_compressed(OUT/f's{start}_{arm}_motion.npz',time_s=t,depth_um=dep,displacement_um=a,**extra);fields[(start,arm)]=(t,dep,a);counts.append(dict(start_s=start,arm=arm,peaks=len(peaks)));print(start,arm,'motion complete',flush=True)
   for candidate,g in tracks[(tracks.time_s>=start)&(tracks.time_s<start+20)].groupby('candidate'):
    vv=np.array([np.interp(g.depth_um.iloc[0],dep,w) for w in a])
    for r in g.itertuples():
     mask=(t>=r.time_s-2.5)&(t<r.time_s+2.5);rows.append(dict(start_s=start,arm=arm,candidate=candidate,time_s=r.time_s,depth_um=r.depth_um,events=r.events,centroid_um=r.centroid_um,dredge_um=float(np.median(vv[mask]))))
  del post,clean,record
 pd.DataFrame(rows).to_csv(OUT/'comparison.csv',index=False);pd.DataFrame(counts).to_csv(OUT/'peak_counts.csv',index=False);d=pd.DataFrame(rows);fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained')
 for ax,(candidate,g) in zip(axes.flat,d.groupby('candidate',sort=False)):
  valid=g[g.centroid_um.notna()];reftime=valid.time_s.min();refcent=valid[valid.time_s==reftime].centroid_um.iloc[0]
  for arm,h in g.groupby('arm',sort=False):
   refval=h[h.time_s==reftime].dredge_um.iloc[0];ax.plot(h.time_s,h.dredge_um-refval,label=arm,ls='--' if arm=='original' else '-')
  h=g[g.arm=='original'];ax.plot(h.time_s,h.centroid_um-refcent,'kx-',label='Waveform centroid');ax.set(title=candidate,xlabel='Recording time (s)',ylabel='Change from first valid waveform bin (µm)');ax.legend(fontsize=8)
 fig.suptitle('Original and compensated DREDGE in separate early/late intervals\nStrict±80um search; waveform observations provisional and not calibrated ground truth',fontsize=11)
 for ext in ['png','pdf']:fig.savefig(OUT/f'01_distant_comparison.{ext}',dpi=150)
 (OUT/'summary.json').write_text(json.dumps(dict(status='complete',preservation_checks=len(checks),preservation_pass=sum(r['passes'] for r in checks),counts=counts),indent=2));print('Complete',flush=True)
if __name__=='__main__':main()
