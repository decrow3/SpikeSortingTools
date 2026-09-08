"""Direct large-peak cohorts for early/late candidate qualification; no sort IDs."""
import json
import numpy as np
import pandas as pd
from scipy.signal import butter,sosfiltfilt
from testing.luke_epoch_corroboration import ROOT,BASE
from spikeinterface.core import NumpyRecording
from spikeinterface.sortingcomponents.peak_detection import detect_peaks
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=ROOT/'testing/outputs/luke_independent_transfer_candidates_v1'
REVIEW_ALL=False
def main():
 OUT.mkdir(exist_ok=False);m=json.loads((BASE/'recording/rescue_recording_manifest.json').read_text());fs=m['sampling_frequency_hz'];loc=np.asarray(m['channel_locations_um']);model=np.load(ROOT/'testing/outputs/luke_common_event_screen_v1/shared_response_model.npz');off=np.arange(-45,46);rows=[];saved={};eventsaved={};checks=[]
 settings=dict(intervals_s=[[960,970],[9510,9520]],detection='Both signs; locally exclusive50um; threshold max(150uV,10*original local MAD)',cohorts='Detection channel and sign; all qualifying events, no template-score filtering',min_events_each_half=15,max_sample_each_half=100,gates=dict(half_waveform_cosine=.9,local_energy_fraction=.7,event_cosine=.8,event_fraction=.8,edge_margin_um=120),shared_signal_gate='Reject if >=80% of channels in event-median PRE-reference waveform correlate >0.9 with event-median common reference',scope='Candidate cohorts, not verified units or trajectories; no use of sort IDs or motion agreement')
 (OUT/'settings.json').write_text(json.dumps(settings,indent=2))
 for start in [960,9510]:
  first=round(start*fs);n=round(10*fs);pad=round(.05*fs)
  with (BASE/'recording/traces_cached_seg0.raw').open('rb') as f:f.seek((first-pad)*768);buf=f.read((n+2*pad)*768)
  x=np.frombuffer(buf,dtype='<i2').reshape(-1,384).astype('float32')*m['gain_uv_per_count'];x=sosfiltfilt(butter(3,[300,6000],fs=fs,btype='bandpass',output='sos'),x,axis=0).astype('float32');del buf;ref=np.median(x,axis=1);pre=x[pad:pad+n].copy();post=pre-ref[pad:pad+n,None];clean=np.empty_like(post)
  for lo in range(0,n,10000):
   e=np.arange(lo,min(n,lo+10000));clean[e]=pre[e]-ref[pad+e[:,None]+model['lag_samples']]@model['coefficients']
  del x
  noise=np.median(abs(post[::5]-np.median(post[::5],axis=0)),axis=0)/.67448975;record=NumpyRecording(post,fs);record.set_channel_locations(loc)
  peaks=detect_peaks(record,method='locally_exclusive',peak_sign='both',radius_um=50.,detect_threshold=10.,noise_levels=np.maximum(noise,15.),n_jobs=1,chunk_duration='1s',progress_bar=False);peaks=peaks[(peaks['sample_index']>46)&(peaks['sample_index']<n-46)];np.save(OUT/f's{start}_peaks.npy',peaks)
  for ch in np.unique(peaks['channel_index']):
   for sign in [-1,1]:
    subset=peaks[(peaks['channel_index']==ch)&(np.sign(peaks['amplitude'])==sign)]['sample_index'];ev=[subset[(subset>=h*5*fs)&(subset<(h+1)*5*fs)] for h in [0,1]]
    if min(map(len,ev))<15:continue
    total=list(map(len,ev));ev=[e[np.linspace(0,len(e)-1,min(100,len(e)),dtype=int)] for e in ev];med=[np.median(post[e[:,None]+off],axis=0) for e in ev];depth=loc[ch,1];local=np.flatnonzero(abs(loc[:,1]-depth)<=60);a,b=[w[:,local] for w in med];amp=float(abs(med[0][:,ch]).max());snr=amp/noise[ch];cos=float(np.sum(a*b)/(np.linalg.norm(a)*np.linalg.norm(b)));compact=float(np.sum(a*a)/np.sum(med[0]**2));w=post[ev[1][:,None]+off,ch];template=med[0][:,ch];ec=w@template/(np.linalg.norm(w,axis=1)*np.linalg.norm(template));fraction=float((ec>=.8).mean())
    pm=np.median(pre[ev[0][:,None]+off],axis=0);rm=np.median(ref[pad+ev[0][:,None]+off],axis=0);coherence=pm.T@rm/(np.linalg.norm(pm,axis=0)*np.linalg.norm(rm)+1e-20);broad=float((coherence>.9).mean());passed=bool(amp>=150 and snr>=10 and cos>=.9 and compact>=.7 and fraction>=.8 and 120<=depth<=3700 and broad<.8)
    key=f's{start}_c{ch}_{"neg" if sign<0 else "pos"}';rows.append(dict(candidate=key,start_s=start,channel=int(ch),sign=sign,depth_um=float(depth),events_first=total[0],events_second=total[1],peak_uv=amp,snr=float(snr),half_cosine=cos,local_energy_fraction=compact,event_shape_fraction=fraction,common_coherent_fraction=broad,screen_pass=passed))
    if not passed and not REVIEW_ALL:continue
    saved[key+'_channels']=local;eventsaved[key]=subset+first
    for h,e in enumerate(ev):
     before=med[h][:,local];after=np.median(clean[e[:,None,None]+off[None,:,None],local[None,None,:]],axis=0);pc=float(np.sum(before*after)/(np.linalg.norm(before)*np.linalg.norm(after)));ratio=float(abs(after).max()/abs(before).max());checks.append(dict(candidate=key,half=h,cosine=pc,amplitude_ratio=ratio,passes=pc>=.9 and .8<=ratio<=1.2));saved[key+f'_h{h}_before']=before;saved[key+f'_h{h}_after']=after
  print('Epoch',start,'detections',len(peaks),'passing cohorts',sum(r['screen_pass'] for r in rows if r['start_s']==start),flush=True);del pre,post,clean,record
 d=pd.DataFrame(rows);d.to_csv(OUT/'candidate_screen.csv',index=False);c=pd.DataFrame(checks);c.to_csv(OUT/'preservation.csv',index=False);np.savez_compressed(OUT/'waveforms.npz',**saved);np.savez_compressed(OUT/'candidate_events.npz',**eventsaved)
 passes=d if REVIEW_ALL else d[d.screen_pass]
 for page in range((len(passes)+3)//4):
  group=passes.iloc[page*4:page*4+4];fig,axes=plt.subplots(len(group),2,figsize=(11,3*len(group)),squeeze=False,layout='constrained')
  for ax,r in zip(axes,group.itertuples()):
   ch=saved[r.candidate+'_channels'];peak=np.flatnonzero(ch==r.channel)[0];tt=off/fs*1000
   for h in [0,1]:
    a=saved[r.candidate+f'_h{h}_before'];b=saved[r.candidate+f'_h{h}_after'];ax[0].plot(tt,a[:,peak],label=f'Original half{h+1}');ax[0].plot(tt,b[:,peak],ls='--',label=f'Compensated half{h+1}')
   ax[0].set(title=f'{r.candidate} · {r.depth_um:.0f} µm · n={r.events_first}/{r.events_second}',xlabel='Time (ms)',ylabel='µV');ax[0].legend(fontsize=7);im=ax[1].imshow(a.T,aspect='auto',origin='lower',extent=[tt[0],tt[-1],0,len(ch)],cmap='RdBu_r',vmin=-r.peak_uv,vmax=r.peak_uv);ax[1].set(title='Original second-half spatial waveform',xlabel='Time (ms)',ylabel='Local channel index');fig.colorbar(im,ax=ax[1])
  fig.suptitle('Rejected-cohort review: all shown fail at least one fixed gate' if REVIEW_ALL else 'Direct-detection candidate review: identity not yet established',fontsize=12)
  for ext in ['png','pdf']:fig.savefig(OUT/f'01_candidates_{page+1}.{ext}',dpi=150)
 summary=dict(status='complete',eligible_cohorts=len(d),passing_cohorts=int(d.screen_pass.sum()),review_all=REVIEW_ALL,preservation_checks=len(c),preservation_pass=int(c.passes.sum()) if len(c) else 0);(OUT/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
if __name__=='__main__':
 import sys
 if '--review-all' in sys.argv:
  REVIEW_ALL=True;OUT=ROOT/'testing/outputs/luke_independent_transfer_review_v1'
 main()
