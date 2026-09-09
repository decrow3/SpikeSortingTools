"""Review depth hypotheses without using DREDGE for matching or centering."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from testing.luke_early_unit445_depth_v2 import OUT,SRC
from testing.luke_epoch_corroboration import BASE

def main():
 decision_rows=[]
 for f in sorted(OUT.glob('s*_decisions.npz')):
  q=np.load(f);n=len(q['winner']);jj=q['winner'];sc=q['scores'];best=sc[jj,np.arange(n),0];gain=sc[jj,np.arange(n),1];passing=(best>=.885)&(gain>=.4)&(gain<=2.5);identity=passing&(best-q['rival_score']>=.03);unique=passing&(best-q['second_score']>=.03)
  decision_rows.append(pd.DataFrame(dict(frame=q['event_frames'],best_score=best,coarse_shift_um=q['shift_grid_um'][jj],identity_accepted=identity,unique_shift=unique,boundary=abs(q['shift_grid_um'][jj])==120,identity_margin=best-q['rival_score'],shift_margin=best-q['second_score'],status=q['status'])))
 decisions=pd.concat(decision_rows,ignore_index=True);decisions.to_csv(OUT/'decisions_flags.csv',index=False)
 (OUT/'decision_support_summary.json').write_text(json.dumps(dict(identity_accepted=int(decisions.identity_accepted.sum()),identity_accepted_shift_ambiguous=int((decisions.identity_accepted&~decisions.unique_shift).sum()),joint_unique=int((decisions.identity_accepted&decisions.unique_shift).sum()),note='Counts are candidate decisions, before duplicate suppression. Identity-accepted ambiguous-depth candidates retained in decisions, not assigned unique waveform depth.'),indent=2))
 events=pd.read_csv(OUT/'events.csv');hist=np.load(SRC/'luke_peak_population_review_v1/02_long_histograms_histograms.npz');dredge=pd.read_csv(SRC/'luke_early_unit445_v1/dredge_trace.csv');old=pd.read_csv(SRC/'luke_early_unit445_v1/event_centroids.csv');z=np.load(OUT/'events_unit445.npz');assert np.array_equal(events.frame.to_numpy(),z['frames']);geo=np.asarray(json.loads((BASE/'recording/rescue_recording_manifest.json').read_text())['channel_locations_um']);depth=geo[z['base_channels'],1];rows=[]
 # Median waveforms only within a single support/shift context; never mix footprints.
 for start in range(930,1030,10):
  for shift in np.unique(z['shift_um']):
   ix=np.flatnonzero((events.time_s>=start)&(events.time_s<start+10)&(z['shift_um']==shift))
   if len(ix)<10:continue
   wave=np.median(z['waveforms'][ix],axis=0);e=np.sum(wave.astype(float)**2,axis=0);cent=float(e@(depth+shift)/e.sum());rows.append(dict(start_s=start,end_s=start+10,time_s=events.time_s.iloc[ix].median(),events=len(ix),shift_um=shift,centroid_um=cent))
 windows=pd.DataFrame(rows,columns=['start_s','end_s','time_s','events','shift_um','centroid_um']);windows.to_csv(OUT/'windows.csv',index=False)
 counts=pd.read_csv(OUT/'chunk_counts.csv').fillna(0);fig,axes=plt.subplots(3,1,figsize=(15,11),sharex=True,layout='constrained');a=np.log1p(hist['arm1_counts']);vmax=np.quantile(np.concatenate([np.log1p(hist[f'arm{i}_counts']).ravel() for i in range(2)]),.995)
 ax=axes[0];im=ax.imshow(a,origin='lower',extent=[hist['time_edges'][0],hist['time_edges'][-1],hist['depth_edges'][0],hist['depth_edges'][-1]],aspect='auto',interpolation='nearest',cmap='magma',vmin=0,vmax=vmax)
 ax.scatter(old.time_s,old.event_centroid_um,s=14,c='#AAAAAA',marker='x',alpha=.6,label='Historical fixed matches')
 ax.scatter(events.time_s,events.centroid_um,s=10,c='#56DDE0',alpha=.65,label='Depth-search accepted event centroid')
 boundary=events.boundary.astype(bool);ax.scatter(events.loc[boundary,'time_s'],events.loc[boundary,'centroid_um'],s=45,facecolors='none',edgecolors='#FFFFFF',marker='^',label='Search boundary (±120 µm)')
 if len(windows):ax.errorbar(windows.time_s,windows.centroid_um,xerr=[windows.time_s-windows.start_s,windows.end_s-windows.time_s],fmt='D',color='white',ms=5,lw=1,label='≥10-event median waveform / 10s / shift')
 ax.plot(dredge.time_s,dredge.display_depth_um,'--',color='#00FF66',lw=1.7,label='DREDGE; historical vertical anchor unchanged');ax.set(ylim=(2080,2440),ylabel='Absolute descriptive depth (µm)',title='Compensated peak-count background · waveform matches from original referenced voltage');ax.legend(fontsize=8,ncol=2,loc='upper right');fig.colorbar(im,ax=ax,label='log(1 + peak count)')
 axes[1].scatter(events.time_s,events.shift_um,s=12,c='#0072B2',label='Accepted coarse depth hypothesis')
 for f in sorted(OUT.glob('s*_decisions.npz')):
  q=np.load(f);sel=np.isin(q['status'],['ambiguous_shift','identity_ambiguous']);fs=json.loads((BASE/'recording/rescue_recording_manifest.json').read_text())['sampling_frequency_hz'];axes[1].scatter(q['event_frames'][sel]/fs,q['shift_grid_um'][q['winner'][sel]],s=9,c='#D55E00',marker='x',alpha=.2)
 fs=json.loads((BASE/'recording/rescue_recording_manifest.json').read_text())['sampling_frequency_hz'];amb=decisions.identity_accepted&~decisions.unique_shift;axes[1].scatter(decisions.loc[amb,'frame']/fs,decisions.loc[amb,'coarse_shift_um'],s=22,facecolors='none',edgecolors='#CC79A7',label='Identity accepted, depth ambiguous')
 axes[1].plot([],[],color='#D55E00',marker='x',ls='',label='Above threshold but unresolved shift/identity');axes[1].axhline(0,color='gray',lw=.5);axes[1].set(ylim=(-140,140),yticks=np.arange(-120,121,40),ylabel='Coarse template shift (µm)',title='Exact geometry grid; 0 µm does NOT establish no fine movement');axes[1].legend(fontsize=8)
 bottom=np.zeros(len(counts))
 for key,col in [('accepted','#0072B2'),('ambiguous_shift','#E69F00'),('identity_ambiguous','#D55E00')]:
  val=counts[key] if key in counts else np.zeros(len(counts));axes[2].bar(counts.start_s+5,val,width=9,bottom=bottom,color=col,label=key.replace('_',' '));bottom+=val
 axes[2].set(ylabel='Events / candidate decisions per 10s',xlabel='Recording time (s)',title='Accepted: unique events after duplicate suppression; unresolved: candidate decisions before suppression');axes[2].legend(fontsize=8)
 for ax in axes:ax.set_xlim(930,1030);ax.ticklabel_format(axis='x',style='plain',useOffset=False)
 fig.suptitle(f'Unit 445 · exploratory depth-aware matching · {len(events)} accepted events\n±120 µm search, 40 µm grid; independent of DREDGE, no forced continuity',fontsize=14)
 fig.supxlabel('Historical thresholds are not recalibrated for the expanded search. Early identity and centroid motion sensitivity remain unqualified.\nCompetitor bank misses early-only cells; some templates have only 1–2 training events. Coarse grid and search boundaries limit interpretation; gaps remain gaps.',fontsize=9)
 for ext in ['png','pdf']:fig.savefig(OUT/f'01_depth_search_dredge.{ext}',dpi=180)
 plt.close(fig)
if __name__=='__main__':main()
