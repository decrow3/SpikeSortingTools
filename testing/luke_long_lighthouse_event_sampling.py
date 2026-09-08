"""Compare motion and centroids using the same accepted-event times."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from testing.luke_long_lighthouse_motion import OUT,SRC
from testing.luke_epoch_corroboration import BASE

def main():
 fs=json.loads((BASE/'recording/rescue_recording_manifest.json').read_text())['sampling_frequency_hz'];events=pd.read_csv(SRC/'luke_lighthouse_gentle_v1/gentle_events.csv');events['actual_time_s']=events.frame/fs;tracks=pd.read_csv(SRC/'luke_lighthouse_gentle_v1/gentle_tracks.csv');fields={arm:np.load(OUT/f'{arm}_motion.npz') for arm in ['original','compensated','screened']};colors=['gray','#2878b5','#b13775'];rows=[]
 fig,axs=plt.subplots(3,3,figsize=(16,11),sharex=True,layout='constrained')
 for ax,(u,g) in zip(axs.ravel(),tracks.groupby('unit_id')):
  g=g.sort_values('time_s');depth=float(g.depth_um.iloc[0]);base=float(g.iloc[0].median_waveform_centroid_um);ev=events[events.unit_id==u]
  for (arm,f),color in zip(fields.items(),colors):
   t=f['time_s'];v=np.array([np.interp(depth,f['depth_um'],w) for w in f['displacement_um']]);btime=ev[(ev.actual_time_s>=4160)&(ev.actual_time_s<4170)].actual_time_s.to_numpy();baseline=np.median(np.interp(btime,t,v));values=[]
   for _,q in g.iterrows():
    et=ev[(ev.actual_time_s>=q.time_s-5)&(ev.actual_time_s<q.time_s+5)].actual_time_s.to_numpy();sampled=np.interp(et,t,v)-baseline;value=float(np.median(sampled)) if len(et) else np.nan;values.append(value)
    rows.append(dict(arm=arm,unit_id=int(u),time_s=float(q.time_s),events=len(et),motion_at_event_median_um=value,centroid_change_um=float(q.median_waveform_centroid_um-base)))
   ax.plot(g.time_s,values,'.-',color=color,label=arm)
  good=g.accepted_events>=10;ax.errorbar(g.time_s[good],g.median_waveform_centroid_um[good]-base,yerr=np.array([g.median_waveform_centroid_um[good]-g.bootstrap_low_um[good],g.bootstrap_high_um[good]-g.median_waveform_centroid_um[good]]),fmt='ko',ms=4,capsize=2,label='Lighthouse');ax.set(title=f'Unit{u} · {depth:.0f}µm',xlabel='Recording time (s)',ylabel='Relative displacement (µm)');ax.axhline(0,color='gray',lw=.4)
 axs[0,0].legend(fontsize=8);fig.suptitle('Motion sampled at each lighthouse’s accepted spike times, then summarized in the same10s bins\nEach series offset to its first bin only; this controls unequal time sampling, but not fixed-template selection bias.')
 for ext in ['png','pdf']:fig.savefig(OUT/f'03_event_matched_overlay.{ext}',dpi=140)
 pd.DataFrame(rows).to_csv(OUT/'event_matched_comparison.csv',index=False)
 # Event support during the largest shallow excursion in compensated motion.
 f=fields['compensated'];r=[]
 for u,depth in [(80,220),(154,620)]:
  v=np.array([np.interp(depth,f['depth_um'],w) for w in f['displacement_um']]);v-=np.median(v[f['time_s']<4170]);deep=v<-20;ev=events[events.unit_id==u];et=ev.actual_time_s.to_numpy();r.append(dict(unit_id=u,excursion_bin_centers_s=f['time_s'][deep].tolist(),events_during_estimated_below_minus20=int((np.interp(et,f['time_s'],v)<-20).sum())))
 print(json.dumps(r),flush=True);(OUT/'shallow_excursion_support.json').write_text(json.dumps(r,indent=2))
if __name__=='__main__':main()
