"""Frozen-reference direct and descriptive reporting for the early250um sweep."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from testing.luke_epoch_corroboration import ROOT,BASE
SRC=ROOT/'testing/outputs';OUT=SRC/'luke_early_sigma_screen250_v1';REF=SRC/'luke_waveform_only_expansion_v1'
COLORS={3:'#0072B2',4:'#009E73',5:'#D55E00',6:'#CC79A7',8:'#8C6D31'}
def references():
 fs=json.loads((BASE/'recording/rescue_recording_manifest.json').read_text())['sampling_frequency_hz'];c=pd.read_csv(REF/'candidate_audit.csv');c=c[c.selected].sort_values('seed_centroid_um');e=pd.read_csv(REF/'overlay_events.csv');z=np.load(SRC/'luke_population_depth_v2/templates.npz');return fs,c,e,z

def field_curve(z,c,fs,seeds):
 t=z['time_s'];v=np.array([np.interp(c.seed_centroid_um,z['depth_um'],x) for x in z['displacement_um']]);st=seeds[f'unit_{c.unit_id}_seed_frames']/fs;offset=np.nanmedian(np.interp(st,t,v,left=np.nan,right=np.nan));return t,v-offset

def observations(ax,e,color=False):
 for cls,marker in [('identity_ambiguous','x'),('lower_score','o'),('strict_accepted','o')]:
  p=e[e.evidence==cls];col=('#D55E00' if cls=='identity_ambiguous' else '#0072B2') if color else ('#777777' if cls!='strict_accepted' else '#111111');kw=dict(facecolors='none',edgecolors=col) if cls=='lower_score' else dict(c=col)
  ax.scatter(p.time_s,p.relative_um,s=9,marker=marker,alpha=.5 if cls!='strict_accepted' else .8,linewidths=.55,rasterized=True,**kw)
 ax.axvspan(930,940,color='gray',alpha=.1);ax.axhline(0,color='gray',lw=.5);ax.set(xlim=(930,1030),xlabel='Time (s)',ylabel='Seed-relative µm')

def quick_report():
 fs,c,e,seeds=references();old=np.load(SRC/'luke_early_screen_shortlist_v1/fields/compensated.npz');new=np.load(OUT/'cached_bound250/field.npz');assert np.array_equal(old['time_s'],new['time_s']);assert np.array_equal(old['depth_um'],new['depth_um'])
 with PdfPages(OUT/'00_cached_bound80_vs250.pdf') as pdf:
  for start in range(0,len(c),6):
   batch=c.iloc[start:start+6];fig,axs=plt.subplots(3,2,figsize=(15,12),layout='constrained')
   for ax,r in zip(axs.flat,batch.itertuples()):
    observations(ax,e[e.unit_id==r.unit_id],True)
    for z,col,ls,label in [(old,'#777777','--','±80µm'),(new,'#111111','-','±250µm')]:t,v=field_curve(z,r,fs,seeds);ax.plot(t,v,color=col,ls=ls,lw=1.2,label=label)
    ax.set_title(f'Unit {r.unit_id} · {r.seed_centroid_um:.0f}µm');ax.legend(fontsize=7)
   for ax in list(axs.flat)[len(batch):]:ax.axis('off')
   fig.suptitle('Pairwise-bound control · identical cached5σ peaks · 930–1030s',fontsize=15);fig.supxlabel('Dashed gray:±80µm; black:±250µm. Waveforms:blue strict/open lower-score; orange ambiguity. No gain/sign/lag fitting.\nFull cached peaks differ only by the common chunk-edge exclusion from the later fresh sweep. This control changes the bound only.',fontsize=10);pdf.savefig(fig);fig.savefig(OUT/f'bound_control_{start//6+1}.png',dpi=140);plt.close(fig)
 rows=[]
 for label,z in [('80',old),('250',new)]:rows.append(dict(bound_um=int(label),median_depth_peak_to_peak_um=float(np.median(np.ptp(z['displacement_um'],axis=0))),max_abs_pairwise_um=float(abs(z['D']).max()),field_min_um=float(z['displacement_um'].min()),field_max_um=float(z['displacement_um'].max())))
 pd.DataFrame(rows).to_csv(OUT/'cached_bound_control.csv',index=False)
 print('Quick bound comparison ready',rows,flush=True)

def report():
 fs,c,e,seeds=references();manifest=pd.read_csv(OUT/'manifest.csv');fields={r.name:np.load(OUT/'fields'/r.name/'field.npz') for r in manifest.itertuples()};metrics=[];eventrows=[];summ=[]
 for r in manifest.itertuples():
  z=fields[r.name];m=z['displacement_um'];t=z['time_s'];valid=(abs(t[:,None]-t[None,:])<=60)&~np.eye(len(t),dtype=bool);valid=np.broadcast_to(valid,z['D'].shape)&(z['U']>0)
  summ.append(dict(name=r.name,sigma=r.sigma,screen=r.screen,bound_um=r.bound_um,peaks=r.peaks,median_depth_peak_to_peak_um=float(np.median(np.ptp(m,axis=0))),median_depth_central95_range_um=float(np.median(np.quantile(m,.975,axis=0)-np.quantile(m,.025,axis=0))),bound_pair_fraction=float(np.mean(abs(z['D'][valid])==r.bound_um))))
  for cell in c.itertuples():
   tt,v=field_curve(z,cell,fs,seeds);q=e[e.unit_id==cell.unit_id].copy();q['name']=r.name;q['predicted_um']=np.interp(q.time_s,tt,v,left=np.nan,right=np.nan);q['difference_um']=q.relative_um-q.predicted_um;eventrows.append(q[['name','unit_id','time_s','evidence','relative_um','predicted_um','difference_um']]);held=q[(q.time_s>=940)&q.predicted_um.notna()]
   for cls in ['strict_accepted','lower_score','identity_ambiguous']:
    for scope in ['all_observations','large_excursions']:
     a=held[held.evidence==cls]
     if scope=='large_excursions':a=a[abs(a.relative_um)>=120]
     metrics.append(dict(name=r.name,unit_id=cell.unit_id,evidence=cls,scope=scope,events=len(a),supported=len(a)>=5,median_abs_difference_um=float(np.median(abs(a.difference_um))) if len(a)>=5 else np.nan,median_signed_difference_um=float(np.median(a.difference_um)) if len(a)>=5 else np.nan))
 metrics=pd.DataFrame(metrics);metrics.to_csv(OUT/'per_candidate_differences.csv',index=False);pd.concat(eventrows,ignore_index=True).to_csv(OUT/'event_matched_predictions.csv',index=False);summ=pd.DataFrame(summ);summ.to_csv(OUT/'field_amplitudes.csv',index=False)
 aggregate=metrics[metrics.supported].groupby(['name','evidence','scope']).agg(candidates=('unit_id','nunique'),median_candidate_abs_difference_um=('median_abs_difference_um','median')).reset_index();aggregate.to_csv(OUT/'descriptive_agreement.csv',index=False)
 screens=['none','center_only','relaxed','full'];labels={'none':'No screening','center_only':'Center only','relaxed':'Relaxed combined','full':'Full screen'}
 with PdfPages(OUT/'02_all17_sigma_screen_overlays.pdf') as pdf:
  for cell in c.itertuples():
   q=e[e.unit_id==cell.unit_id];fig,axs=plt.subplots(2,2,figsize=(17,12),sharex=True,sharey=True,layout='constrained')
   for ax,screen in zip(axs.flat,screens):
    observations(ax,q)
    for sigma,col in COLORS.items():
     name=f'{sigma}sigma_{screen}_bound250';tt,v=field_curve(fields[name],cell,fs,seeds);ax.plot(tt,v,c=col,lw=1,label=f'{sigma}σ')
    if screen=='none':tt,v=field_curve(fields['5sigma_none_bound80'],cell,fs,seeds);ax.plot(tt,v,c='#777777',ls='--',lw=.9,label='5σ ±80 control')
    ax.set_title(labels[screen]);ax.legend(fontsize=8,ncol=3)
   fig.suptitle(f'Unit {cell.unit_id} · seed {cell.seed_centroid_um:.0f}µm · ±250µm sigma/screen sweep',fontsize=16);fig.supxlabel('Black filled:strict waveform; gray open:lower score; gray crosses:identity ambiguity. Colored curves:fixed candidate-depth DREDGE.\nShaded930–940s:seed reference. Same observations in every panel; no fitted sign,gain,lag or candidate reselection. Screening does not screen lighthouse events.',fontsize=10);pdf.savefig(fig)
   if cell.unit_id in [106,161,555,632,657,673]:fig.savefig(OUT/f'unit_{cell.unit_id}_sweep.png',dpi=140)
   plt.close(fig)
 fig,axs=plt.subplots(1,2,figsize=(13,6),layout='constrained');sc={'none':'#0072B2','center_only':'#009E73','relaxed':'#D55E00','full':'#CC79A7'}
 for screen in screens:
  a=summ[(summ.screen==screen)&(summ.bound_um==250)].sort_values('sigma');axs[0].plot(a.sigma,a.median_depth_peak_to_peak_um,'o-',c=sc[screen],label=labels[screen]);axs[1].plot(a.sigma,a.peaks,'o-',c=sc[screen],label=labels[screen])
 axs[0].set(xlabel='Detection threshold (σ)',ylabel='Median across depths of temporal peak-to-peak (µm)',title='Estimated motion amplitude');axs[1].set(xlabel='Detection threshold (σ)',ylabel='Peaks retained',yscale='log',title='Input population (log scale)');axs[0].legend();fig.suptitle('930–1030s · all20 sweep arms use±250µm pairwise bounds');fig.supxlabel('Amplitude is not accuracy. Inspect individual waveform comparisons and evidence classes; no automatic winner is selected.',fontsize=10)
 for ext in ['pdf','png']:fig.savefig(OUT/f'01_amplitude_and_counts.{ext}',dpi=160)
 plt.close(fig)
 (OUT/'README.md').write_text('''# Early100s sigma and screening sweep with17 frozen waveform candidates\n\nOpen00_cached_bound80_vs250.pdf for the bound-only control;01_amplitude_and_counts.pdf for input counts and displacement amplitudes;02_all17_sigma_screen_overlays.pdf for all17 candidates across the four screens and five sigma thresholds.\n\nAll20 factorial arms use±250µm explicitly bounded pairwise correlation. A21st arm uses5σ/no screen/±80µm on the same fresh common-interior peaks. The initial cached5σ bound-only control preserves the earlier complete input exactly; do not confuse its edge population with the fresh common-interior population. All other DREDGE settings remain frozen.\n\nCandidates,seed references,and strict/lower-score/ambiguous classes remain frozen. DREDGE is sampled at fixed seed centroid depth and actual waveform event times,with the median field over supported seed-spike times removed. No field extrapolation beyond930.5–1029.5s. No fitted gain,sign,lag or offset to later waveform observations.\n\nPer-candidate differences are descriptive,not physical error. Reports separate evidence classes and all observations from predeclared|waveform displacement|≥120µm observations. Require≥5 events for each candidate/class/scope summary; aggregate is the median of candidate median absolute differences,not spike-count-weighted. Candidate families may overlap,so this is not a consensus or independent-cell confidence interval. Sparse coverage and stationary observations can favor flat fields; no automatic winner is adopted. The interval has been viewed during method development.\n\nDetection sigma uses frozen original-channel MAD; screening SNR uses frozen residual noise. Screening affects DREDGE input only. All peak arrays,locations,features,masks,fields,pairwise constraints,coverage and source hashes are saved. Completed chunks/fields are checksum-validated; interrupted stages require evidence review and restart,not within-stage resumption.\n''')
 print('Report complete',flush=True)
