"""Channel-agnostic shared-event explanation screen; validation-only masks."""
from testing.luke_epoch_corroboration import ROOT,BASE
import numpy as np,pandas as pd,json
from scipy.signal import butter,sosfiltfilt
from scipy.ndimage import gaussian_filter1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=ROOT/'testing/outputs/luke_common_event_screen_v1';PRE=ROOT/'testing/outputs/luke_peak_threshold_screen_v1'
def main():
 OUT.mkdir(exist_ok=False);m=json.loads((BASE/'recording/rescue_recording_manifest.json').read_text());fs=m['sampling_frequency_hz'];loc=np.array(m['channel_locations_um']);n=round(20*fs);first=round(4180*fs);pad=round(.05*fs);lags=np.arange(-15,16)
 settings=dict(training_s=[4180,4190],evaluation_s=[4190,4200],input_peaks='Existing fresh5sigma negative locally-exclusive peaks',model='Per-channel ridge FIR prediction from pre-reference global median; 31 taps ±0.5ms; fit first10s every3samples; ridge0.001*mean diagonal covariance; no channel blacklist',reject_rule='ALL: >=80% of pre-reference channel waveforms cosine>0.8 with same-time global-reference waveform; explained fraction>=0.8 of post-reference detection-channel energy; post-reference observed/predicted cosine>=0.95; no residual local extremum >=4 residual noise sigma within ±40um/±0.3ms.',scope='Diagnostic removal mask on peaks only. Compensation is used to assess remaining neural evidence, not applied to production voltage or sort.',limitations=['A learned shared component may include biological population signals.','Passing the rule supports shared-disturbance explanation, not definitive origin classification.','Leave unmodeled/ambiguous peaks eligible.','Same paired epoch is diagnostic development; model parameters alone are held out in its second half.'])
 (OUT/'settings.json').write_text(json.dumps(settings,indent=2))
 with (BASE/'recording/traces_cached_seg0.raw').open('rb') as f:f.seek((first-pad)*768);buf=f.read((n+2*pad)*768)
 x=np.frombuffer(buf,dtype='<i2').reshape(-1,384).astype('float32')*m['gain_uv_per_count'];sos=butter(3,[300,6000],fs=fs,btype='bandpass',output='sos');x=sosfiltfilt(sos,x,axis=0).astype('float32')[pad:pad+n].copy();del buf;reference=np.median(x,axis=1);train=np.arange(16,round(10*fs)-16,3);design=reference[train[:,None]+lags];gram=design.astype('float64').T@design;ridge=.001*np.trace(gram)/len(lags);coef=np.linalg.solve(gram+ridge*np.eye(len(lags)),design.astype('float64').T@x[train]).astype('float32');del design
 prediction=np.zeros_like(x)
 for lo in range(16,n-16,10000):
  e=np.arange(lo,min(lo+10000,n-16));prediction[e]=reference[e[:,None]+lags]@coef
 clean=x-prediction;noise=np.median(abs(clean[:round(10*fs)]-np.median(clean[:round(10*fs)],axis=0)),axis=0)/.67448975;np.savez_compressed(OUT/'shared_response_model.npz',coefficients=coef,lag_samples=lags,residual_noise_uv=noise,ridge=ridge);print('Shared response model fitted on first10s; coefficients frozen',flush=True)
 p=np.load(PRE/'peaks_5sigma.npy');positions=np.load(PRE/'locations_5sigma.npy');off=np.arange(-30,31);rows=[];examples={};rng=np.random.default_rng(14)
 for lo in range(0,len(p),200):
  inds=np.arange(lo,min(lo+200,len(p)));inds=inds[(p['sample_index'][inds]>50)&(p['sample_index'][inds]<n-50)];ev=p['sample_index'][inds];ch=p['channel_index'][inds];w=x[ev[:,None]+off];rw=reference[ev[:,None]+off];dot=np.einsum('ntc,nt->nc',w,rw);cos=dot/(np.linalg.norm(w,axis=1)*np.linalg.norm(rw,axis=1)[:,None]+1e-20);coherence=np.mean(cos>.8,axis=1)
  observed=w[np.arange(len(ev))[:,None],np.arange(61)[None,:],ch[:,None]]-rw;pred=prediction[ev[:,None]+off,ch[:,None]]-rw;residual=observed-pred;explained=1-np.sum(residual**2,axis=1)/(np.sum(observed**2,axis=1)+1e-20);shape=np.sum(observed*pred,axis=1)/(np.linalg.norm(observed,axis=1)*np.linalg.norm(pred,axis=1)+1e-20)
  for j,idx in enumerate(inds):
   channels=np.flatnonzero(abs(loc[:,1]-loc[ch[j],1])<=40);cw=clean[ev[j]+np.arange(-9,10)[:,None],channels[None,:]];remaining=float(np.max(abs(cw)/noise[channels]));flag=bool(coherence[j]>=.8 and explained[j]>=.8 and shape[j]>=.95 and remaining<4);r=dict(peak_index=int(idx),time_s=float(4180+ev[j]/fs),channel=int(ch[j]),depth_um=float(positions['y'][idx]),broad_coherence_fraction=float(coherence[j]),energy_explained=float(explained[j]),predicted_waveform_cosine=float(shape[j]),residual_local_max_snr=remaining,reject_shared_artifact=flag);rows.append(r)
   # Review representative high explained events, plus protected events in the same shared context.
   category='reject' if flag else 'protected_local_signal' if coherence[j]>=.8 and explained[j]>=.8 and shape[j]>=.95 else None
   key=(category,int(loc[ch[j],1]//1000))
   if category and key not in examples:examples[key]=(r,observed[j].copy(),pred[j].copy(),residual[j].copy(),w[j].copy(),rw[j].copy())
 d=pd.DataFrame(rows).set_index('peak_index');d.to_csv(OUT/'peak_classification.csv');reject=np.zeros(len(p),bool);reject[d.index]=d.reject_shared_artifact;np.save(OUT/'keep_mask.npy',~reject);np.save(OUT/'rejected_peaks.npy',p[reject]);np.save(OUT/'retained_peaks.npy',p[~reject]);np.save(OUT/'retained_locations.npy',positions[~reject]);print(f'Screen removed {reject.sum()} of {len(p)} peaks',flush=True)
 # Check independently matched lighthouse times from preceding local-control audit, rather than using them to fit screen.
 lighthouse=pd.read_csv(ROOT/'testing/outputs/luke_lighthouse_gentle_v1/gentle_events.csv');lighthouse=lighthouse[(lighthouse.frame>=first)&(lighthouse.frame<first+n)];depths=pd.read_csv(ROOT/'testing/outputs/luke_lighthouse_gentle_v1/gentle_tracks.csv').groupby('unit_id').depth_um.first();checks=[]
 for cid,g in lighthouse.groupby('unit_id'):
  events=g.frame.to_numpy()-first;local=abs(loc[p['channel_index'],1]-depths[cid])<=60
  for half in [0,1]:
   truths=events[(events>=half*10*fs)&(events<(half+1)*10*fs)];before=after=0
   for e in truths:
    k=local&(abs(p['sample_index']-e)<=.0008*fs);before+=int(k.any());after+=int((k&~reject).any())
   checks.append(dict(unit_id=int(cid),depth_um=float(depths[cid]),half=half,independent_lighthouse_events=len(truths),coincident_before=before,coincident_after=after,lost_coincidences=before-after))
 c=pd.DataFrame(checks);c.to_csv(OUT/'lighthouse_preservation.csv',index=False)
 objective=[];edges=np.arange(0,3841);depth=edges[:-1]+.5;shift=np.arange(-20,21);fig,axes=plt.subplots(1,3,figsize=(15,5))
 for name,mask in [('All peaks',np.ones(len(p),bool)),('Screen retained',~reject)]:
  profiles=[]
  for half in [0,1]:
   k=mask&(p['sample_index']>=half*10*fs)&(p['sample_index']<(half+1)*10*fs);profiles.append(gaussian_filter1d(np.histogram(positions['y'][k],edges,weights=abs(p['amplitude'][k]))[0],1.))
  for ax,(low,high) in zip(axes[:2],[(1960,2560),(2260,2860)]):
   mid=(depth>=low)&(depth<high);vals=[np.corrcoef(profiles[0][mid],np.interp(depth[mid]+s,depth,profiles[1]))[0,1] for s in shift];ax.plot(shift,vals,label=name);ax.set(title=f'Depth {low}–{high} µm',xlabel='After − before shift (µm)',ylabel='Profile correlation');ax.legend();objective.append(dict(arm=name,depth_lo_um=low,depth_hi_um=high,best_shift_um=int(shift[np.argmax(vals)]),peak_correlation=max(vals),zero_correlation=vals[20]))
 cg=c.groupby('unit_id')[['coincident_before','coincident_after']].sum();axes[2].barh(cg.index.astype(str),cg.coincident_before,color='#aaaaaa',label='Before screen');axes[2].barh(cg.index.astype(str),cg.coincident_after,color='#167a9a',label='After screen');axes[2].set(xlabel='Peaks coincident with lighthouse events',ylabel='Lighthouse unit',title='Existing independent tracks retained?');axes[2].legend(fontsize=8);fig.suptitle('Shared-event screen: no channel blacklist and no motion agreement in selection\nDiagnostic 4180–4200 s; first-half fit, second-half model evaluation; profile scores are not full estimator replay',fontsize=11);fig.tight_layout(rect=[0,0,1,.89]);fig.savefig(OUT/'01_screen_effect.png',dpi=160);fig.savefig(OUT/'01_screen_effect.pdf');pd.DataFrame(objective).to_csv(OUT/'alignment_objective.csv',index=False)
 keys=list(examples);fig,axes=plt.subplots(len(keys),2,figsize=(11,2.7*len(keys)),squeeze=False)
 for axs,key in zip(axes,keys):
  r,observed,pred,residual,w,rw=examples[key];tt=off/fs*1000;axs[0].plot(tt,observed,c='k',label='Observed referenced waveform');axs[0].plot(tt,pred,label='Predicted shared contribution');axs[0].plot(tt,residual,label='Unexplained remainder');axs[0].set(title=f'{key[0]} · ch{r["channel"]} · {r["time_s"]:.3f} s',xlabel='Time (ms)',ylabel='µV');axs[0].legend(fontsize=7);axs[1].imshow(w.T,aspect='auto',origin='lower',extent=[tt[0],tt[-1],0,384],cmap='RdBu_r',vmin=-250,vmax=250);axs[1].set(title=f'Pre-reference coherence {r["broad_coherence_fraction"]:.0%}; residual local SNR {r["residual_local_max_snr"]:.1f}',xlabel='Time (ms)',ylabel='Channel index')
 fig.suptitle('Event-level explanation review — retained local signal overrides shared-artifact score',fontsize=12);fig.tight_layout(rect=[0,0,1,.96]);fig.savefig(OUT/'02_event_examples.png',dpi=160);fig.savefig(OUT/'02_event_examples.pdf');result=dict(status='complete',peaks=len(p),rejected=int(reject.sum()),retained=int((~reject).sum()),lighthouse_coincidences_lost=int(c.lost_coincidences.sum()),objective=objective,limitations=settings['limitations']);(OUT/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
