"""Original-voltage candidate screening and frozen-model preservation across time."""
import json,hashlib
import numpy as np
import pandas as pd
from scipy.signal import butter,sosfiltfilt
from testing.luke_epoch_corroboration import ROOT,BASE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=ROOT/'testing/outputs/luke_neural_transfer_validation_v1'
STARTS=[960,5710,9510]
def main():
 OUT.mkdir(exist_ok=False);m=json.loads((BASE/'recording/rescue_recording_manifest.json').read_text());fs=m['sampling_frequency_hz'];loc=np.asarray(m['channel_locations_um']);model_path=ROOT/'testing/outputs/luke_common_event_screen_v1/shared_response_model.npz';model=np.load(model_path);lags=model['lag_samples'];coeff=model['coefficients'];cur=BASE/'cur/cur_output';st=np.load(cur/'spike_times.npy',mmap_mode='r').ravel();cl=np.load(cur/'spike_clusters.npy',mmap_mode='r').ravel();offs=np.arange(-45,46);rows=[];preservation=[];payload={}
 settings=dict(intervals_s=[[s,s+10] for s in STARTS],selection='Early, middle and late previously surveyed epochs; candidates screened on ORIGINAL voltage only',screen=dict(min_events_per_half=15,min_peak_uv=150,min_peak_noise_ratio=10,min_half_cosine=.9,min_local_energy_fraction=.7,min_event_cosine=.8,min_event_fraction=.8,edge_margin_um=120),preservation=dict(min_cosine=.9,amplitude_ratio=[.8,1.2]),model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),model_refit=False,scope='Provisional existing-sort event times; preservation at fixed times, not independent identity tracking or motion accuracy')
 (OUT/'settings.json').write_text(json.dumps(settings,indent=2))
 for start in STARTS:
  first=round(start*fs);n=round(10*fs);pad=round(.05*fs)
  with (BASE/'recording/traces_cached_seg0.raw').open('rb') as f:f.seek((first-pad)*768);buf=f.read((n+2*pad)*768)
  x=np.frombuffer(buf,dtype='<i2').reshape(-1,384).astype('float32')*m['gain_uv_per_count'];x=sosfiltfilt(butter(3,[300,6000],fs=fs,btype='bandpass',output='sos'),x,axis=0).astype('float32');del buf
  ref=np.median(x,axis=1);clean=np.empty((n,384),dtype='float32')
  for lo in range(0,n,10000):
   e=np.arange(lo,min(n,lo+10000));clean[e]=x[pad+e]-ref[pad+e[:,None]+lags]@coeff
  post=x[pad:pad+n]-ref[pad:pad+n,None];del x,ref
  noise=np.median(abs(post[::5]-np.median(post[::5],axis=0)),axis=0)/.67448975
  idx=np.flatnonzero((st>=first+46)&(st<first+n-46));times=np.asarray(st[idx])-first;ids=np.asarray(cl[idx]);epoch_pass=[]
  for cid in np.unique(ids):
   events=[]
   for half in [0,1]:
    ev=times[(ids==cid)&(times>=half*5*fs)&(times<(half+1)*5*fs)];events.append(ev)
   if min(map(len,events))<15:continue
   events=[e[np.linspace(0,len(e)-1,min(100,len(e)),dtype=int)] for e in events]
   a=post[events[0][:,None]+offs];wa=np.median(a,axis=0);del a
   peak=np.max(abs(wa),axis=0).argmax();depth=loc[peak,1];channels=np.flatnonzero(abs(loc[:,1]-depth)<=60);amp=float(abs(wa[:,peak]).max());snr=amp/noise[peak]
   wb=np.median(post[events[1][:,None]+offs],axis=0);v=wa[:,channels];w=wb[:,channels];cos=float(np.sum(v*w)/(np.linalg.norm(v)*np.linalg.norm(w)));compact=float(np.sum(v*v)/np.sum(wa*wa));evw=post[events[1][:,None]+offs,peak];ec=evw@wa[:,peak]/(np.linalg.norm(evw,axis=1)*np.linalg.norm(wa[:,peak]));repeat=float((ec>=.8).mean());passed=bool(amp>=150 and snr>=10 and cos>=.9 and compact>=.7 and repeat>=.8 and 120<=depth<=3700)
   rows.append(dict(start_s=start,unit_id=int(cid),depth_um=float(depth),peak_channel=int(peak),peak_uv=amp,snr=float(snr),half_cosine=cos,local_energy_fraction=compact,event_fraction=repeat,screen_pass=passed))
   if not passed:continue
   epoch_pass.append(int(cid))
   for half,e in enumerate(events):
    before=[wa,wb][half][:,channels];after=np.median(clean[e[:,None,None]+offs[None,:,None],channels[None,None,:]],axis=0);pc=float(np.sum(before*after)/(np.linalg.norm(before)*np.linalg.norm(after)));ratio=float(abs(after).max()/abs(before).max());preservation.append(dict(start_s=start,unit_id=int(cid),depth_um=float(depth),half=half,events=len(e),cosine=pc,amplitude_ratio=ratio,passes=pc>=.9 and .8<=ratio<=1.2));key=f's{start}_u{cid}_h{half}';payload[key+'_before']=before;payload[key+'_after']=after;payload[key+'_channels']=channels
  print('Epoch',start,'passed original screen:',epoch_pass,flush=True);del clean,post
 df=pd.DataFrame(rows);df.to_csv(OUT/'candidate_screen.csv',index=False);p=pd.DataFrame(preservation);p.to_csv(OUT/'preservation.csv',index=False);np.savez_compressed(OUT/'selected_waveforms.npz',**payload)
 fig,axes=plt.subplots(1,3,figsize=(13,6),sharey=True,layout='constrained')
 for ax,start in zip(axes,STARTS):
  g=df[df.start_s==start];ax.scatter(g.peak_uv,g.depth_um,c='lightgray',s=15,label='Below screen');v=g[g.screen_pass];ax.scatter(v.peak_uv,v.depth_um,c='tab:blue',s=30,label='Original-voltage pass');ax.axvline(150,c='k',ls=':',lw=.7);ax.set(title=f'{start}–{start+10} s · {len(v)} pass',xlabel='Original median peak (µV)',ylim=(0,3820));ax.legend(fontsize=7)
 axes[0].set_ylabel('Depth (µm)');fig.suptitle('Neural candidate coverage at widely separated epochs\nFixed original-voltage screen; existing-sort event times remain provisional',fontsize=11)
 for ext in ['png','pdf']:fig.savefig(OUT/f'01_coverage.{ext}',dpi=150)
 if len(p):
  fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
  for start in STARTS:
   g=p[p.start_s==start];axes[0].scatter(g.cosine,g.depth_um,label=f'{start} s');axes[1].scatter(g.amplitude_ratio,g.depth_um,label=f'{start} s')
  axes[0].set(xlabel='Original / compensated waveform cosine',ylabel='Depth (µm)',xlim=(min(.9,p.cosine.min()-.01),1.002));axes[1].set(xlabel='Compensated / original peak amplitude',ylabel='Depth (µm)',xlim=(min(.8,p.amplitude_ratio.min()-.05),max(1.2,p.amplitude_ratio.max()+.05)));axes[0].legend();axes[1].axvline(1,c='gray',lw=.6);fig.suptitle('Frozen-model preservation on candidates selected before compensation\nTwo half-interval checks per candidate; no cross-epoch identity claim',fontsize=11)
  for ext in ['png','pdf']:fig.savefig(OUT/f'02_preservation.{ext}',dpi=150)
 result=dict(status='complete',eligible=len(df),screen_pass=int(df.screen_pass.sum()),checks=len(p),checks_pass=int(p.passes.sum()) if len(p) else 0,cosine_min=float(p.cosine.min()) if len(p) else None,amplitude_ratio_range=[float(p.amplitude_ratio.min()),float(p.amplitude_ratio.max())] if len(p) else None);(OUT/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
