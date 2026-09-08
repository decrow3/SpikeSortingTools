"""Cached-feature screen ablation and bounded 100s motion sweep."""
import json,hashlib
import numpy as np
import pandas as pd
from scipy.signal import butter,sosfiltfilt
from spikeinterface.core import NumpyRecording
from testing.luke_epoch_corroboration import ROOT,BASE
from testing.luke_dredge_bounded import estimate_bounded
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
SRC=ROOT/'testing/outputs';LONG=SRC/'luke_long_lighthouse_motion_v1';OUT=SRC/'luke_screen_sweep_v1'
RULES=['snr','center','neighbor','width','broad','shared']

def features(clean,post,p,geo,noise,fs):
 result={key:np.full(len(p),np.nan) for key in ['snr','center','neighbor','width','broad','explained','shared_cosine']};off=np.arange(-30,31);mid=abs(off)<=15
 for ch in range(384):
  ids=np.flatnonzero((p['channel_index']==ch)&(p['sample_index']>40)&(p['sample_index']<len(clean)-40))
  if not len(ids):continue
  ev=p['sample_index'][ids];channels=np.flatnonzero(abs(geo[:,1]-geo[ch,1])<=60);ci=np.flatnonzero(channels==ch)[0];w=clean[ev[:,None,None]+off[None,:,None],channels[None,None,:]];v=w[:,:,ci];original=post[ev[:,None]+off,ch];pred=original-v
  result['explained'][ids]=1-np.sum(v*v,axis=1)/(np.sum(original**2,axis=1)+1e-12);result['shared_cosine'][ids]=np.sum(original*pred,axis=1)/(np.linalg.norm(original,axis=1)*np.linalg.norm(pred,axis=1)+1e-12)
  amp=np.max(abs(v),axis=1);result['snr'][ids]=amp/noise[ch];result['center'][ids]=np.sum(v[:,mid]**2,axis=1)/(np.sum(v*v,axis=1)+1e-12);other=np.max(abs(w),axis=1);other[:,ci]=0
  nc=np.einsum('ntc,nt->nc',w,v)/(np.linalg.norm(w,axis=1)*np.linalg.norm(v,axis=1)[:,None]+1e-12);qualified=(other>=.25*amp[:,None])&(other/noise[channels]>=4);result['neighbor'][ids]=np.max(np.where(qualified,nc,-1),axis=1)
  dominant=np.argmax(abs(v),axis=1);width=np.zeros(len(ids));broad=np.zeros(len(ids))
  for j,k in enumerate(dominant):
   sign=np.sign(v[j,k]);lo=hi=int(k)
   while lo>0 and sign*v[j,lo-1]>=.5*amp[j]:lo-=1
   while hi<60 and sign*v[j,hi+1]>=.5*amp[j]:hi+=1
   width[j]=(hi-lo+1)/fs*1000
  for lo in range(0,len(ids),100):
   sl=slice(lo,min(lo+100,len(ids)));centers=ev[sl]+off[dominant[sl]];full=clean[centers[:,None]+np.arange(-3,4)];broad[sl]=np.mean(np.max(abs(full),axis=1)/noise>=4,axis=1)
  result['width'][ids]=width;result['broad'][ids]=broad
 return result

def mask(f,omit=(),only=None,snr=8,center=.65,neighbor=.8):
 gates=dict(snr=f['snr']>=snr,center=f['center']>=center,neighbor=f['neighbor']>=neighbor,width=(f['width']>=.067)&(f['width']<=.8),broad=f['broad']<.2,shared=~((f['explained']>=.5)&(f['shared_cosine']>=.8)))
 k=np.isfinite(f['snr'])
 for name,g in gates.items():
  if name not in omit and (only is None or name in only):k=k&g
 return k

def variants():
 v=[dict(name='compensated',kind='baseline'),dict(name='full_screen',kind='screen')]
 v += [dict(name='without_'+r,kind='screen',omit=[r]) for r in RULES]
 v += [dict(name=r+'_only',kind='screen',only=[r]) for r in ['snr','center','neighbor']]
 v += [dict(name='snr_6',kind='screen',snr=6),dict(name='snr_5',kind='screen',snr=5),dict(name='neighbor_06',kind='screen',neighbor=.6),dict(name='neighbor_07',kind='screen',neighbor=.7),dict(name='center_05',kind='screen',center=.5),dict(name='center_055',kind='screen',center=.55),dict(name='relaxed_combination',kind='screen',snr=6,center=.5,neighbor=.6)]
 v += [dict(name='random_count_seed14',kind='random',seed=14),dict(name='random_count_seed29',kind='random',seed=29),dict(name='random_depth_time',kind='stratified',seed=14)]
 return v

def main():
 OUT.mkdir(exist_ok=False);(OUT/'fields').mkdir();(OUT/'masks').mkdir()
 m=json.loads((BASE/'recording/rescue_recording_manifest.json').read_text());fs=m['sampling_frequency_hz'];geo=np.asarray(m['channel_locations_um']);model=np.load(SRC/'luke_common_event_screen_v1/shared_response_model.npz');cfg=json.loads((LONG/'settings.json').read_text())['estimator'];specs=variants()
 settings=dict(interval_s=[4160,4260],variants=specs,estimator=cfg,strict_pairwise_bound_um=80,features='Frozen v2 features; one read/filter/compensation per20s chunk; no redetection or relocalization',metrics=dict(event_sampling='Interpolate each field at unchanged lighthouse accepted frames, median within10s bins, offset by first bin',overall='Equal weight per cell: mean absolute deviation from provisional centroids, bins>=10events, baseline bin excluded',movement='Equal weight per cell: absolute error in4185→4195 drop and4195→4205 recovery, requiring>=10events at both endpoints',stability='Per-cell95th percentile absolute1s increment, then median across cells; diagnostic, not a selection target',flat_control='Constant-zero motion scored identically; never a candidate for adoption'),random_controls='Two uniform samples with exact full-screen total count; one sample matching full-screen counts in each1s×100um detector-depth stratum',interpretation='Exploratory tuning on a previously viewed interval, not independent validation. Do not select solely by overall error or flatness.',resume='Completed feature chunks, masks and fields persist; no within-stage checkpoint or automatic resume; no sort')
 (OUT/'settings.json').write_text(json.dumps(settings,indent=2));parts=[]
 for start in [4160,4180,4200,4220,4240]:
  n=round(20*fs);pad=round(.05*fs)
  with (BASE/'recording/traces_cached_seg0.raw').open('rb') as handle:handle.seek((round(start*fs)-pad)*768);buf=handle.read((n+2*pad)*768)
  x=np.frombuffer(buf,dtype='<i2').reshape(-1,384).astype('float32')*m['gain_uv_per_count'];del buf;x=sosfiltfilt(butter(3,[300,6000],fs=fs,btype='bandpass',output='sos'),x,axis=0).astype('float32');ref=np.median(x,axis=1);clean=np.empty((n,384),dtype='float32')
  for lo in range(0,n,10000):
   e=np.arange(lo,min(n,lo+10000));clean[e]=x[pad+e]-ref[pad+e[:,None]+model['lag_samples']]@model['coefficients']
  post=x[pad:pad+n]-ref[pad:pad+n,None];del x,ref
  p=np.load(LONG/f's{start}_compensated_peaks.npy');f=features(clean,post,p,geo,model['residual_noise_uv'],fs);assert np.array_equal(mask(f),np.load(LONG/f's{start}_keep_mask.npy'));np.savez_compressed(OUT/f's{start}_features.npz',**f);parts.append(f);del clean,post;print(start,'features complete; original mask exact',flush=True)
 f={key:np.concatenate([part[key] for part in parts]) for key in parts[0]};np.savez_compressed(OUT/'features.npz',**f)
 p=np.load(LONG/'compensated_peaks.npy');y=np.load(LONG/'compensated_locations.npy');assert len(p)==len(f['snr']);full=mask(f);nkeep=int(full.sum());assert nkeep==len(np.load(LONG/'screened_peaks.npy'))
 record=NumpyRecording(np.broadcast_to(np.zeros((1,384),dtype='float32'),(round(100*fs),384)),fs);record.set_channel_locations(geo)
 field_cache={};manifest=[];coverage=[]
 for spec in specs:
  name=spec['name'];kind=spec['kind'];k=np.ones(len(p),bool)
  if kind=='screen':k=mask(f,**{key:value for key,value in spec.items() if key not in ['name','kind']})
  elif kind in ['random','stratified']:
   rng=np.random.default_rng(spec['seed']);k=np.zeros(len(p),bool)
   if kind=='random':k[rng.choice(len(p),nkeep,replace=False)]=True
   else:
    group=(p['sample_index']/fs).astype(int)*40+(geo[p['channel_index'],1]//100).astype(int)
    for g in np.unique(group):
     ids=np.flatnonzero(group==g);number=int(full[ids].sum())
     if number:k[rng.choice(ids,number,replace=False)]=True
  np.save(OUT/'masks'/f'{name}.npy',k);digest=hashlib.sha256(k.tobytes()).hexdigest();path=OUT/'fields'/f'{name}.npz'
  if name in ['compensated','full_screen']:
   source=LONG/('compensated_motion.npz' if name=='compensated' else 'screened_motion.npz');old=np.load(source);np.savez_compressed(path,**{key:old[key] for key in old.files});field_cache[digest]=path
  elif digest in field_cache:
   old=np.load(field_cache[digest]);np.savez_compressed(path,**{key:old[key] for key in old.files})
  else:
   motion,extra=estimate_bounded(record,p[k],y[k],cfg);np.savez_compressed(path,time_s=motion.temporal_bins_s[0]+4160,depth_um=motion.spatial_bins_um,displacement_um=motion.displacement[0],D=extra['D'],C=extra['C'],U=extra['U']);field_cache[digest]=path
  manifest.append(dict(name=name,peaks=int(k.sum()),retained_fraction=float(k.mean()),mask_sha256=digest))
  # Fine grid for diagnosing changes in central information, not only whole-probe totals.
  te=np.arange(4160,4261,5);de=np.arange(0,3901,200);h=np.histogram2d(4160+p['sample_index'][k]/fs,y['y'][k],bins=[te,de])[0];mass=np.histogram2d(4160+p['sample_index'][k]/fs,y['y'][k],bins=[te,de],weights=abs(p['amplitude'][k]))[0];np.savez_compressed(OUT/f'{name}_coverage.npz',counts=h,amplitude_mass=mass,time_edges=te,depth_edges=de)
  coverage.append(dict(name=name,empty_cells=int((h==0).sum()),cells_below10=int((h<10).sum())))
  print(name,'motion complete',int(k.sum()),flush=True)
 pd.DataFrame(manifest).to_csv(OUT/'manifest.csv',index=False);pd.DataFrame(coverage).to_csv(OUT/'coverage_summary.csv',index=False)
 analyze()
 (OUT/'summary.json').write_text(json.dumps(dict(status='complete',variants=len(specs),feature_mask_regression='all5chunks exact',scope='Screen diagnosis and exploratory sweep only; no production adoption'),indent=2))

def analyze():
 fs=json.loads((BASE/'recording/rescue_recording_manifest.json').read_text())['sampling_frequency_hz'];ev=pd.read_csv(SRC/'luke_lighthouse_gentle_v1/gentle_events.csv');ev['actual_s']=ev.frame/fs;tracks=pd.read_csv(SRC/'luke_lighthouse_gentle_v1/gentle_tracks.csv');manifest=pd.read_csv(OUT/'manifest.csv');rows=[];metric=[]
 for name in list(manifest.name)+['flat_zero_control']:
  if name!='flat_zero_control':field=np.load(OUT/'fields'/f'{name}.npz');t=field['time_s'];dep=field['depth_um'];a=field['displacement_um'];assert np.isfinite(a).all()
  percell=[]
  for u,g in tracks.groupby('unit_id'):
   g=g.sort_values('time_s');depth=float(g.depth_um.iloc[0]);e=ev[ev.unit_id==u];v=np.zeros(100) if name=='flat_zero_control' else np.array([np.interp(depth,dep,w) for w in a]);tt=np.arange(4160.5,4260,1) if name=='flat_zero_control' else t
   baseframes=e[(e.actual_s>=4160)&(e.actual_s<4170)].actual_s.to_numpy();baseline=np.median(np.interp(baseframes,tt,v));values={};truth={};support={};basecent=float(g.iloc[0].median_waveform_centroid_um)
   for _,q in g.iterrows():
    times=e[(e.actual_s>=q.time_s-5)&(e.actual_s<q.time_s+5)].actual_s.to_numpy();pred=float(np.median(np.interp(times,tt,v))-baseline) if len(times) else np.nan;target=float(q.median_waveform_centroid_um-basecent);values[q.time_s]=pred;truth[q.time_s]=target;support[q.time_s]=int(q.accepted_events);rows.append(dict(name=name,unit_id=int(u),depth_um=depth,time_s=float(q.time_s),accepted_events=int(q.accepted_events),predicted_um=pred,centroid_um=target))
   errors=[abs(values[time]-truth[time]) for time in values if time>4165 and support[time]>=10 and np.isfinite(truth[time]) and np.isfinite(values[time])];moves=[]
   for lo,hi in [(4185,4195),(4195,4205)]:
    if support[lo]>=10 and support[hi]>=10:moves.append(abs((values[hi]-values[lo])-(truth[hi]-truth[lo])))
   percell.append(dict(name=name,unit_id=int(u),overall_mae_um=float(np.mean(errors)),movement_mae_um=float(np.mean(moves)),step_p95_um=float(np.quantile(abs(np.diff(v)),.95))))
  pd.DataFrame(percell).to_csv(OUT/f'{name}_cell_metrics.csv',index=False);metric.append(dict(name=name,overall_mae_um=float(np.mean([r['overall_mae_um'] for r in percell])),movement_mae_um=float(np.mean([r['movement_mae_um'] for r in percell])),step_p95_um=float(np.median([r['step_p95_um'] for r in percell]))))
 d=pd.DataFrame(rows);d.to_csv(OUT/'event_matched_predictions.csv',index=False);scores=pd.DataFrame(metric).merge(manifest,on='name',how='left');scores.to_csv(OUT/'scores.csv',index=False)
 fig,axs=plt.subplots(1,3,figsize=(17,10),layout='constrained');names=list(scores.name);yy=np.arange(len(scores));colors=['#2878b5' if n=='compensated' else '#b13775' if n=='full_screen' else '#b0b0b0' if 'random' in n or n=='flat_zero_control' else '#56a17a' for n in names]
 for ax,col,title in zip(axs,['overall_mae_um','movement_mae_um','retained_fraction'],['Equal-cell mean absolute difference','Drop + recovery difference','Fraction of compensated peaks retained']):
  ax.barh(yy,scores[col],color=colors);ax.set(yticks=yy,yticklabels=names,title=title,xlabel='µm' if col!='retained_fraction' else 'Fraction');ax.invert_yaxis()
 fig.suptitle('Screen ablation and small threshold sweep ·100s against provisional lighthouse centroids\nLower difference is descriptive agreement, not ground-truth accuracy. Flat control exposes stationary-bin bias.')
 for ext in ['png','pdf']:fig.savefig(OUT/f'01_sweep_summary.{ext}',dpi=140)
 print(scores[['name','overall_mae_um','movement_mae_um','retained_fraction']].to_string(index=False),flush=True)
if __name__=='__main__':main()
