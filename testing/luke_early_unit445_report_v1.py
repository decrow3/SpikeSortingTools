"""Cached-only sparse unit445 transfer report with explicit ten-second support."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from testing.luke_early_unit445_v1 import OUT,SRC,BASE
from testing.luke_early_lighthouse_binning_v2 import centroid

def main():
 m=json.loads((BASE/'recording/rescue_recording_manifest.json').read_text());z=np.load(OUT/'events_unit445.npz');t=z['frames']/m['sampling_frequency_hz'];waves=z['waveforms'];depth=np.asarray(m['channel_locations_um'])[z['channels'],1];events=pd.read_csv(OUT/'event_centroids.csv');trace=pd.read_csv(OUT/'dredge_trace.csv');summary=json.loads((OUT/'summary.json').read_text());anchor=summary['anchor_um'];rng=np.random.default_rng(445);rows=[]
 for a in np.arange(930,1030,10):
  ix=np.flatnonzero((t>=a)&(t<a+10));r=dict(start_s=a,end_s=a+10,time_s=float(np.median(t[ix])) if len(ix) else a+5,events=len(ix),centroid_um=np.nan,low_um=np.nan,high_um=np.nan,dredge_at_spikes_um=np.nan)
  if len(ix)>=10:
   w=waves[ix];r['centroid_um']=centroid(w,depth);r['low_um'],r['high_um']=np.quantile([centroid(w[rng.integers(0,len(ix),len(ix))],depth) for _ in range(100)],[.025,.975]);r['dredge_at_spikes_um']=float(np.median(np.interp(t[ix],trace.time_s,trace.display_depth_um)))
  rows.append(r)
 table=pd.DataFrame(rows);table.to_csv(OUT/'ten_second_windows.csv',index=False);q=table[table.centroid_um.notna()];hist=np.load(SRC/'luke_peak_population_review_v1/02_long_histograms_histograms.npz');arrays=[np.log1p(hist[f'arm{i}_counts']) for i in range(2)];vmax=np.quantile(np.concatenate([a.ravel() for a in arrays]),.995)
 fig,axes=plt.subplots(3,2,figsize=(15,11),layout='constrained',gridspec_kw={'height_ratios':[1,1, .8]})
 for col,arr in enumerate(arrays):
  for row in range(2):
   ax=axes[row,col];im=ax.imshow(arr,origin='lower',extent=[930,1030,0,3840],aspect='auto',interpolation='nearest',cmap='magma',vmin=0,vmax=vmax)
   ax.scatter(events.time_s,events.event_centroid_um,s=12,c='#56DDE0',alpha=.65,linewidths=0,label='Accepted-event energy centroid')
   ax.errorbar(q.time_s,q.centroid_um,xerr=np.array([q.time_s-q.start_s,q.end_s-q.time_s]),fmt='D',ms=5,mfc='white',mec='#006F80',ecolor='white',elinewidth=1,label='10 s median-waveform centroid (≥10 spikes)',zorder=4);ax.vlines(q.time_s,q.low_um,q.high_um,colors='white',lw=1,zorder=4)
   ax.plot(trace.time_s,trace.display_depth_um,color='#00FF66',lw=1.8,ls='--',label='Compensated 5σ DREDGE at 2260 µm')
   ax.set(xlim=(930,1030),ylim=(1950,2400) if row==0 else (2190,2310),ylabel='Depth (µm)',title=('Original' if col==0 else 'Compensated')+' input · '+('regional view' if row==0 else 'unit 445 zoom'))
 axes[0,0].legend(fontsize=8,loc='lower left');fig.colorbar(im,ax=axes[:2,:],label='log(1 + peak count)',shrink=.8)
 ax=axes[2,0];ax.plot(trace.time_s,trace.display_depth_um-anchor,color='#009E73',ls='--',lw=1.6,label='DREDGE trace');ax.errorbar(q.time_s,q.centroid_um-anchor,xerr=np.array([q.time_s-q.start_s,q.end_s-q.time_s]),fmt='D',color='#0072B2',ms=4,elinewidth=.8,label='10 s waveform centroid');ax.vlines(q.time_s,q.low_um-anchor,q.high_um-anchor,color='#0072B2',lw=1);ax.scatter(q.time_s,q.dredge_at_spikes_um-anchor,color='#009E73',marker='x',s=30,label='DREDGE at same accepted spikes');ax.axhline(0,color='gray',lw=.5);ax.set(title='Relative motion · no interpolation of lighthouse gaps',ylabel='Displacement (µm)',xlim=(930,1030));ax.legend(fontsize=7,loc='best');ax.grid(alpha=.15)
 ax=axes[2,1];ax.bar(table.start_s+5,table.events,width=9,color=np.where(table.events>=10,'#0072B2','#aaaaaa'));ax.axhline(10,color='black',ls=':',label='10-spike summary minimum');ax.set(title='Accepted-match support · 7/10 ten-second bins',ylabel='Accepted matches / 10 s',xlim=(930,1030));ax.legend(fontsize=8)
 for ax in axes.ravel():ax.ticklabel_format(axis='x',style='plain',useOffset=False)
 for ax in axes[-1]:ax.set_xlabel('Recording time (s)')
 fig.suptitle('Unit 445 template transfer · 930–1,030 s · 141 accepted matches\nEarly identity unqualified; no supported 100-spike windows · peak background: 0.25 s × 10 µm',fontsize=14)
 fig.supxlabel('DREDGE offset: median prediction at all accepted spikes aligned to their median-waveform centroid. No sign, gain, or lag fitting.\n10 s summaries may contain gaps; bars mark bin extent / conditional 95% bootstrap. Fixed-support matching can favor small movement.',fontsize=9)
 for ext in ['png','pdf']:fig.savefig(OUT/f'02_sparse_support_dredge.{ext}',dpi=180)
 plt.close(fig)
 print(table.to_string(index=False))
if __name__=='__main__':main()
