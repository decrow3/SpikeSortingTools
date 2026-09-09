"""Time-bin and spike-count lighthouse summaries from frozen accepted waveforms."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];SRC=ROOT/'testing/outputs';IN=SRC/'luke_early_lighthouse_events_v2';OUT=SRC/'luke_early_lighthouse_binning_v2'
CANDIDATES=['s960_c293_pos','s960_c338_pos']
def centroid(waves,depth):
 wave=np.median(waves,axis=0);energy=np.sum(wave**2,axis=0);return float(energy@depth/energy.sum())
def groups(t,mode):
 if isinstance(mode,float):
  for a in np.arange(930,1030,mode):yield a,min(a+mode,1030),np.flatnonzero((t>=a)&(t<min(a+mode,1030)))
 elif mode=='adaptive':
  a=930.
  while a<1030:
   b=min(a+.25,1030)
   while np.count_nonzero((t>=a)&(t<b))<10 and b-a<2-1e-8 and b<1030:b=min(b+.25,1030)
   yield a,b,np.flatnonzero((t>=a)&(t<b));a=b
 else:
  # Preserve observed gaps; no 100-spike group crosses an interspike gap >2s.
  for run in np.split(np.arange(len(t)),np.flatnonzero(np.diff(t)>2)+1):
   for first in range(0,len(run),100):
    ix=run[first:first+100]
    if len(ix):yield float(t[ix[0]]),float(t[ix[-1]]),ix

def main():
 OUT.mkdir(exist_ok=False)
 from testing.luke_epoch_corroboration import BASE
 m=json.loads((BASE/'recording/rescue_recording_manifest.json').read_text());fs=m['sampling_frequency_hz'];geo=np.asarray(m['channel_locations_um']);rows=[];chosen={};anchors={};provenance={}
 modes=[.25,.5,1.,2.,5.,'adaptive','100spikes']
 for candidate in CANDIDATES:
  path=IN/f'events_{candidate}.npz';provenance[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest();z=np.load(path);t=z['frames']/fs;waves=z['waveforms'];depth=geo[z['channels'],1];assert len(t)==len(waves) and np.all(np.diff(t)>0)
  anchorix=np.flatnonzero((t>=970)&(t<975));assert len(anchorix)>=10;anchor=centroid(waves[anchorix],depth);anchors[candidate]=t[anchorix]
  rng=np.random.default_rng(20260908+CANDIDATES.index(candidate));chosen[candidate]={}
  for mode in modes:
   name=str(mode);chosen[candidate][name]=[]
   for a,b,ix in groups(t,mode):
    count=len(ix);status='supported' if count>=(100 if mode=='100spikes' else 10) else 'insufficient_events'
    if mode=='100spikes' and b-a>20:status='span_over20s'
    if count>1 and np.diff(t[ix]).max()>2:status='gap_over2s'
    row=dict(candidate=candidate,mode=name,start_s=a,end_s=b,time_s=float(np.median(t[ix])) if count else (a+b)/2,events=count,span_s=b-a,status=status,centroid_relative_um=np.nan,low_um=np.nan,high_um=np.nan)
    if status=='supported':
     w=waves[ix];row['centroid_relative_um']=centroid(w,depth)-anchor
     values=[centroid(w[rng.integers(0,count,count)],depth)-anchor for _ in range(100)]
     row['low_um'],row['high_um']=map(float,np.quantile(values,[.025,.975]))
    chosen[candidate][name].append(t[ix]);rows.append(row)
 table=pd.DataFrame(rows);table.to_csv(OUT/'waveform_windows.csv',index=False)
 # Event windows are frozen before loading motion. DREDGE comparisons use identical spike times.
 field=np.load(SRC/'luke_early_screen_shortlist_v1/fields/compensated.npz');pred=[]
 for candidate in CANDIDATES:
  z=np.load(IN/f'events_{candidate}.npz');nominal=2920 if '293' in candidate else 3380;v=np.array([np.interp(nominal,field['depth_um'],d) for d in field['displacement_um']]);offset=np.median(np.interp(anchors[candidate],field['time_s'],v))
  for mode in modes:
   part=table[(table.candidate==candidate)&(table['mode']==str(mode))]
   for r,events in zip(part.itertuples(),chosen[candidate][str(mode)]):pred.append(float(np.median(np.interp(events,field['time_s'],v))-offset) if len(events) and r.status=='supported' else np.nan)
 table['dredge_relative_um']=pred;table.to_csv(OUT/'windows_with_motion.csv',index=False)
 summary=table.groupby(['candidate','mode']).agg(windows=('events','size'),supported=('status',lambda x:(x=='supported').sum()),median_events=('events','median'),median_span_s=('span_s','median'));summary.to_csv(OUT/'support_summary.csv')
 fig,axes=plt.subplots(3,2,figsize=(15,11),sharex=True,layout='constrained')
 for col,candidate in enumerate(CANDIDATES):
  for ri,mode in enumerate(['0.25','adaptive','100spikes']):
   ax=axes[ri,col];q=table[(table.candidate==candidate)&(table['mode']==mode)];good=q.status=='supported';v=q[good]
   ax.axvspan(960,970,color='#dddddd',alpha=.6,label='Template training period')
   if len(v):
    ax.errorbar(v.time_s,v.centroid_relative_um,xerr=np.array([v.time_s-v.start_s,v.end_s-v.time_s]),fmt='o',color='#0072B2',ms=3,elinewidth=.6,label='Median-waveform centroid')
    ax.vlines(v.time_s,v.low_um,v.high_um,color='#0072B2',lw=.8)
    ax.scatter(v.time_s,v.dredge_relative_um,s=18,marker='x',color='#D55E00',label='Baseline DREDGE at same spikes')
   else:ax.text(.5,.5,'No supported windows',ha='center',transform=ax.transAxes)
   ax.set(title=f'{candidate} · '+{'0.25':'0.25 s bins','adaptive':'Adaptive 0.25–2 s · ≥10 spikes','100spikes':'100 consecutive spikes · ≤20 s span'}[mode]+f'\n{int(good.sum())}/{len(q)} windows supported',xlim=(930,1030),xlabel='Recording time (s)',ylabel='Relative displacement (µm)');ax.grid(alpha=.15);ax.ticklabel_format(axis='x',style='plain',useOffset=False)
   if ri==0 and len(v):ax.legend(fontsize=7,loc='best')
 fig.suptitle('Early provisional lighthouse motion · finer time bins and spike-count windows\nHorizontal bars = temporal support; vertical bars = conditional bootstrap; no interpolation across gaps',fontsize=14)
 fig.supxlabel('Fixed 970–975 s reference offset. Identity and displacement sensitivity remain unqualified; bootstrap excludes reference/identity/model uncertainty.\n100-spike groups are non-overlapping and cannot cross >2 s interspike gaps. Grey interval was used to train templates.',fontsize=9)
 for ext in ['png','pdf']:fig.savefig(OUT/f'01_motion_binning.{ext}',dpi=150)
 plt.close(fig)
 fig,axes=plt.subplots(2,1,figsize=(13,6),sharex=True,layout='constrained')
 for ax,candidate in zip(axes,CANDIDATES):
  q=table[(table.candidate==candidate)&(table['mode']=='0.25')];ax.bar(q.start_s+.125,q.events,width=.25,color='#0072B2');ax.axhline(10,color='#333333',ls='--',label='10-spike minimum');ax.axvspan(960,970,color='#dddddd',alpha=.5);ax.set(title=candidate,ylabel='Accepted spikes / 0.25 s');ax.legend(fontsize=8)
 axes[-1].set_xlabel('Recording time (s)');fig.suptitle('Observed spike support · detector/template dropout can resemble biological silence')
 for ext in ['png','pdf']:fig.savefig(OUT/f'02_spike_support.{ext}',dpi=150)
 plt.close(fig)
 (OUT/'summary.json').write_text(json.dumps(dict(status='complete',minimum_spikes_time_window=10,adaptive_max_s=2,spike_count=100,spike_window_max_s=20,split_gap_s=2,bootstrap_resamples=100,source_sha256=provenance),indent=2))
 print(summary.to_string(),flush=True)
if __name__=='__main__':main()
