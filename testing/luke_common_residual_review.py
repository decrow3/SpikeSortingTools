"""Attribute retained local residuals and test neural-waveform preservation."""
from testing.luke_epoch_corroboration import ROOT,BASE
from testing.luke_multidepth_anchors_v2 import scores,near
import numpy as np,pandas as pd,json
from scipy.signal import butter,sosfiltfilt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=ROOT/'testing/outputs/luke_common_residual_review_v1';PRE=ROOT/'testing/outputs/luke_common_event_screen_v1';DET=ROOT/'testing/outputs/luke_peak_threshold_screen_v1';LIGHT=ROOT/'testing/outputs/luke_lighthouse_gentle_v1'
def main():
 OUT.mkdir(exist_ok=False);m=json.loads((BASE/'recording/rescue_recording_manifest.json').read_text());fs=m['sampling_frequency_hz'];loc=np.asarray(m['channel_locations_um']);n=round(20*fs);first=round(4180*fs);pad=round(.05*fs);model=np.load(PRE/'shared_response_model.npz');lags=model['lag_samples'];coeff=model['coefficients'];noise=model['residual_noise_uv'];original_noise=np.load(DET/'noise_uv.npy')
 settings=dict(interval_s=[4180,4200],model=str(PRE/'shared_response_model.npz'),model_refit=False,scope='Residual attribution plus protected neural waveform check; no classifier thresholds changed; original masks preserved',template_match='Existing independently trained lighthouse templates, weighted cosine>=0.9 and gain0.4–2.5; candidate templates within120um of residual maximum',compensation_gate='Every tracked lighthouse with >=10 events/half must retain median local waveform cosine>=0.9 and peak amplitude ratio0.8–1.2 in both halves before any compensated-input detection trial')
 (OUT/'settings.json').write_text(json.dumps(settings,indent=2))
 with (BASE/'recording/traces_cached_seg0.raw').open('rb') as f:f.seek((first-pad)*768);buf=f.read((n+2*pad)*768)
 x=np.frombuffer(buf,dtype='<i2').reshape(-1,384).astype('float32')*m['gain_uv_per_count'];sos=butter(3,[300,6000],fs=fs,btype='bandpass',output='sos');x=sosfiltfilt(sos,x,axis=0).astype('float32')[pad:pad+n].copy();del buf;reference=np.median(x,axis=1);prediction=np.zeros_like(x)
 for lo in range(16,n-16,10000):
  e=np.arange(lo,min(lo+10000,n-16));prediction[e]=reference[e[:,None]+lags]@coeff
 clean=x-prediction;post=x-reference[:,None];del x,prediction
 light=pd.read_csv(LIGHT/'gentle_events.csv');light=light[(light.frame>=first)&(light.frame<first+n)];depths=pd.read_csv(LIGHT/'gentle_tracks.csv').groupby('unit_id').depth_um.first();z=np.load(LIGHT/'templates.npz');templates={};wavecheck=[];saved={};off=np.arange(-30,31)
 for cid,g in light.groupby('unit_id'):
  ch=z[f'unit_{cid}_channels'];v=z[f'unit_{cid}_template'];weights=v*v/(v*v+(2*original_noise[ch])**2);peak=ch[np.max(abs(v),axis=0).argmax()];templates[cid]=dict(channels=ch,template=v,weights=weights,depth=float(depths[cid]),offset=int(np.argmax(abs(v[:,np.where(ch==peak)[0][0]])))-30)
  for half in [0,1]:
   ev=g.frame.to_numpy()-first;ev=ev[(ev>=half*10*fs)&(ev<(half+1)*10*fs)];ev=ev[np.linspace(0,len(ev)-1,min(200,len(ev)),dtype=int)];before=np.median(post[ev[:,None,None]+off[None,:,None],ch[None,None,:]],axis=0);after=np.median(clean[ev[:,None,None]+off[None,:,None],ch[None,None,:]],axis=0);cos=float(np.sum(before*after)/(np.linalg.norm(before)*np.linalg.norm(after)));ratio=float(np.max(abs(after))/np.max(abs(before)));wavecheck.append(dict(unit_id=int(cid),depth_um=float(depths[cid]),half=half,events=len(ev),waveform_cosine=cos,peak_amplitude_ratio=ratio,passes=len(ev)>=10 and cos>=.9 and .8<=ratio<=1.2));saved[f'unit_{cid}_half{half}_before']=before;saved[f'unit_{cid}_half{half}_after']=after;saved[f'unit_{cid}_channels']=ch
 check=pd.DataFrame(wavecheck);check.to_csv(OUT/'lighthouse_waveform_preservation.csv',index=False);np.savez_compressed(OUT/'lighthouse_waveforms.npz',**saved);print(check.to_json(orient='records'),flush=True)
 d=pd.read_csv(PRE/'peak_classification.csv');p=np.load(DET/'peaks_5sigma.npy');selected=d[(~d.reject_shared_artifact)&(d.broad_coherence_fraction>=.8)&(d.energy_explained>=.8)&(d.predicted_waveform_cosine>=.95)];rows=[];examples={}
 for r in selected.itertuples():
  e=int(p['sample_index'][r.peak_index]);ch=int(r.channel);channels=np.flatnonzero(abs(loc[:,1]-loc[ch,1])<=40);ww=clean[e+np.arange(-9,10)[:,None],channels[None,:]];s=abs(ww)/noise[channels];it,ic=np.unravel_index(np.argmax(s),s.shape);rc=int(channels[ic]);re=e+it-9;best=-1;bid=-1;bg=np.nan
  for cid,c in templates.items():
   if abs(c['depth']-loc[rc,1])>120:continue
   ss=scores(clean,np.array([re-c['offset']]),c['channels'],c['template'],c['weights'])[0]
   if ss[0]>best:best=float(ss[0]);bid=int(cid);bg=float(ss[1])
  neural=best>=.9 and .4<=bg<=2.5;coincident=False
  if neural:
   truth=np.sort(light[light.unit_id==bid].frame.to_numpy());coincident=bool(near(np.array([first+re]),truth,.0008*fs)[0])
  category='template_and_time_supported' if neural and coincident else 'template_supported_only' if neural else 'unresolved_local_signal';rows.append(dict(peak_index=int(r.peak_index),time_s=float(r.time_s),detection_channel=ch,residual_peak_channel=rc,residual_time_offset_ms=(re-e)/fs*1000,residual_max_snr=float(s[it,ic]),best_template_id=bid,best_template_cosine=best,best_template_gain=bg,independent_time_match=coincident,category=category))
  if category not in examples:examples[category]=(post[e+off,ch].copy(),clean[e+off,ch].copy(),clean[re+off,rc].copy(),rows[-1])
 residual=pd.DataFrame(rows);residual.to_csv(OUT/'retained_residual_attribution.csv',index=False);print(residual.category.value_counts().to_json(),flush=True)
 fig,axes=plt.subplots(1,2,figsize=(12,6));labels=[f'{r.unit_id} / {"before" if r.half==0 else "after"}' for r in check.itertuples()];yy=np.arange(len(check));colors=np.where(check.passes,'#167a9a','#aaaaaa');axes[0].barh(yy,check.waveform_cosine,color=colors);axes[1].barh(yy,check.peak_amplitude_ratio,color=colors);axes[0].set_yticks(yy,labels);axes[1].set_yticks(yy,[]);axes[0].invert_yaxis();axes[1].invert_yaxis();axes[0].axvline(.9,c='k',ls=':');axes[1].axvline(.8,c='k',ls=':');axes[1].axvline(1.2,c='k',ls=':');axes[0].set(xlabel='Original vs compensated median waveform cosine',xlim=(0,1));axes[1].set(xlabel='Compensated / original peak amplitude',xlim=(0,max(1.3,check.peak_amplitude_ratio.max()+.1)));fig.suptitle('Does shared-response compensation preserve lighthouse waveforms?\nSame independently matched events; no refitting on evaluation half; blue passes both gates');fig.tight_layout();fig.savefig(OUT/'01_waveform_preservation.png',dpi=160);fig.savefig(OUT/'01_waveform_preservation.pdf')
 fig,axes=plt.subplots(len(examples),1,figsize=(11,3*len(examples)),squeeze=False)
 for ax,(category,(a,b,c,r)) in zip(axes[:,0],examples.items()):
  tt=off/fs*1000;ax.plot(tt,a,label='Original detection-channel waveform',c='gray');ax.plot(tt,b,label='Detection-channel remainder');ax.plot(tt,c,label='Strongest local remainder, recentered');ax.set(title=f'{category} · {r["time_s"]:.3f}s · detection ch{r["detection_channel"]}, residual ch{r["residual_peak_channel"]}\nBest lighthouse template {r["best_template_id"]}: cosine {r["best_template_cosine"]:.2f}',xlabel='Time (ms)',ylabel='µV');ax.legend(fontsize=8)
 fig.tight_layout();fig.savefig(OUT/'02_residual_examples.png',dpi=160);fig.savefig(OUT/'02_residual_examples.pdf');result=dict(status='complete',retained_shared_explained_events=len(residual),categories=residual.category.value_counts().to_dict(),waveform_checks_pass=int(check.passes.sum()),waveform_checks_total=len(check),compensated_detection_gate_pass=bool(check.passes.all()),next='Compensation does not pass preservation gate; keep original screening mask and diagnose distortion.' if not check.passes.all() else 'Compensation passes bounded waveform gate; eligible for a separate fresh-detection/localization trial.',limitations=['Template library is sparse; unmatched residuals cannot be called artifacts.','Original lighthouse identity checks are provisional.','The same diagnostic interval is not untouched prospective validation.']);(OUT/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
