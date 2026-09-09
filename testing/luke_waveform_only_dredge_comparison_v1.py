"""Cached direct17-candidate comparison; no refitting or consensus."""
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from testing.luke_epoch_corroboration import ROOT,BASE
SRC=ROOT/'testing/outputs';IN=SRC/'luke_waveform_only_expansion_v1';OUT=SRC/'luke_waveform_only_dredge_comparison_v1'
def main():
 OUT.mkdir(exist_ok=True);mp=BASE/'recording/rescue_recording_manifest.json';fs=json.loads(mp.read_text())['sampling_frequency_hz'];cp=IN/'candidate_audit.csv';ep=IN/'overlay_events.csv';fp=SRC/'luke_early_screen_shortlist_v1/fields/compensated.npz';sp=SRC/'luke_population_depth_v2/templates.npz';c=pd.read_csv(cp);c=c[c.selected].sort_values('seed_centroid_um');e=pd.read_csv(ep);f=np.load(fp);seeds=np.load(sp);tt=f['time_s'];curves={};rows=[];refs=[]
 for r in c.itertuples():
  absolute=np.array([np.interp(r.seed_centroid_um,f['depth_um'],v) for v in f['displacement_um']]);st=seeds[f'unit_{r.unit_id}_seed_frames']/fs;v=np.interp(st,tt,absolute,left=np.nan,right=np.nan);offset=float(np.nanmedian(v));trace=absolute-offset;curves[r.unit_id]=trace;refs.append(dict(unit_id=r.unit_id,seed_centroid_um=r.seed_centroid_um,seed_spikes=len(st),seed_spikes_with_field_support=int(np.isfinite(v).sum()),field_reference_um=offset))
  q=e[e.unit_id==r.unit_id].copy();q['dredge_relative_um']=np.interp(q.time_s,tt,trace,left=np.nan,right=np.nan);q['waveform_minus_dredge_um']=q.relative_um-q.dredge_relative_um;rows.append(q)
 e=pd.concat(rows,ignore_index=True);e.to_csv(OUT/'event_matched_comparison.csv',index=False);pd.DataFrame(refs).to_csv(OUT/'reference_support.csv',index=False);np.savez_compressed(OUT/'dredge_reference_curves.npz',time_s=tt,**{f'unit_{k}':v for k,v in curves.items()})
 h=e[(e.time_s>=940)&e.dredge_relative_um.notna()];summary=h.groupby(['unit_id','evidence']).agg(events=('time_s','size'),median_difference_um=('waveform_minus_dredge_um','median'),median_absolute_difference_um=('waveform_minus_dredge_um',lambda x:np.median(abs(x))),waveform_min_um=('relative_um','min'),waveform_max_um=('relative_um','max')).reset_index();summary.to_csv(OUT/'descriptive_differences.csv',index=False)
 config=dict(interval_s=[930,1030],units=c.unit_id.tolist(),field=str(fp),field_sampling='Spatial interpolation at each fixed seed centroid; temporal interpolation only inside930.5–1029.5s; no extrapolation',reference='DREDGE median at seed spike times within field support; waveform uses original frozen seed centroid. No fitted offset,gain,sign or lag to match later observations.',display='All strict,lower-score and ambiguous observations; full-time field plus same-event field samples; no consensus or smoothed waveform path',interpretation='Differences are descriptive, not motion ground-truth error. Existing development window, not independent method confirmation. Fixed-reference-depth field convention; no moving-depth resampling chosen from matches.')
 (OUT/'settings.json').write_text(json.dumps(config,indent=2));(OUT/'chart_contract.json').write_text(json.dumps(dict(question='Do17 waveform-only candidates at different depths show movements consistent with frozen DREDGE?',renderer='Matplotlib standalone PDF/PNG',grain='Individual waveform events and1s field samples,930–1030s',palette='Blue waveform strict/low-score,orange ambiguity,black DREDGE',limits='Per-unit y scales include every alternative; field sampled at actual events for differences'),indent=2))
 handles=[Line2D([],[],color='#0072B2',marker='o',lw=0,label='Strict waveform'),Line2D([],[],color='#0072B2',marker='o',mfc='none',lw=0,label='Lower-score waveform'),Line2D([],[],color='#D55E00',marker='x',lw=0,label='Identity ambiguous'),Line2D([],[],color='#222222',lw=1.4,label='DREDGE (fixed seed depth)')]
 def panel(ax,r,small=False,event_marks=False):
  q=e[e.unit_id==r.unit_id];size=8 if small else 21
  for cls in ['identity_ambiguous','lower_score','strict_accepted']:
   p=q[q.evidence==cls]
   if cls=='identity_ambiguous':ax.scatter(p.time_s,p.relative_um,s=size,marker='x',c='#D55E00',alpha=.5,linewidths=.5,rasterized=True)
   elif cls=='lower_score':ax.scatter(p.time_s,p.relative_um,s=size,facecolors='none',edgecolors='#0072B2',alpha=.65,linewidths=.65,rasterized=True)
   else:ax.scatter(p.time_s,p.relative_um,s=size,c='#0072B2',alpha=.8,linewidths=0,rasterized=True)
  ax.plot(tt,curves[r.unit_id],color='#222222',lw=1.1,zorder=4)
  if event_marks:
   a=q[(q.evidence=='strict_accepted')&q.dredge_relative_um.notna()];ax.scatter(a.time_s,a.dredge_relative_um,s=12,marker='+',c='#222222',linewidths=.65,zorder=5)
  ax.axvspan(930,940,color='gray',alpha=.1);ax.axhline(0,color='gray',lw=.5);ax.set(xlim=(930,1030),xlabel='Time (s)',ylabel='Seed-relative displacement (µm)',title=f'Unit {r.unit_id} · seed {r.seed_centroid_um:.0f} µm');ax.ticklabel_format(axis='x',style='plain',useOffset=False)
 with PdfPages(OUT/'01_all_17_depth_ordered.pdf') as pdf:
  for start in range(0,len(c),6):
   batch=c.iloc[start:start+6];fig,axs=plt.subplots(3,2,figsize=(15,12),layout='constrained')
   for ax,r in zip(axs.flat,batch.itertuples()):panel(ax,r,True)
   for ax in list(axs.flat)[len(batch):]:ax.axis('off')
   fig.suptitle(f'Waveform-only lighthouse candidates vs DREDGE · ordered by seed depth · page {start//6+1}/3',fontsize=14);fig.supxlabel('Filled blue:strict; open blue:lower score; orange crosses:ambiguous; black:DREDGE. Shaded930–940s:seed interval.\nIndividual vertical scales include all alternatives. No waveform smoothing, consensus, or fit to DREDGE. Field centered on seed-spike support.',fontsize=10);pdf.savefig(fig);fig.savefig(OUT/f'overview_{start//6+1}.png',dpi=150);plt.close(fig)
 with PdfPages(OUT/'02_individual_comparisons.pdf') as pdf:
  for r in c.itertuples():
   fig,axs=plt.subplots(2,1,figsize=(14,10),layout='constrained',gridspec_kw={'height_ratios':[2,1]});panel(axs[0],r,event_marks=True);axs[0].legend(handles=handles,fontsize=9,ncol=2,loc='best');q=e[e.unit_id==r.unit_id]
   for cls,color,marker in [('strict_accepted','#0072B2','o'),('lower_score','#0072B2','o'),('identity_ambiguous','#D55E00','x')]:
    p=q[(q.evidence==cls)&q.dredge_relative_um.notna()]
    kw=dict(facecolors='none',edgecolors=color) if cls=='lower_score' else dict(c=color)
    axs[1].scatter(p.time_s,p.waveform_minus_dredge_um,s=15,marker=marker,alpha=.6,linewidths=.6,rasterized=True,**kw)
   axs[1].axhline(0,color='#222222',lw=.9);axs[1].axvspan(930,940,color='gray',alpha=.1);axs[1].set(xlim=(930,1030),xlabel='Time (s)',ylabel='Waveform − DREDGE (µm)',title='Difference at actual waveform event times; no temporal binning')
   fig.suptitle(f'Unit {r.unit_id} · {r.cohort.replace("_"," ")} · waveform identities frozen before DREDGE comparison',fontsize=15);fig.supxlabel('Top black plus signs: DREDGE sampled at strict event times. All evidence classes remain separate.\nNo field extrapolation beyond930.5–1029.5s; missing edge comparisons remain missing. Descriptive disagreement is not physical error.',fontsize=10);pdf.savefig(fig)
   if r.unit_id in [161,555,657,673]:fig.savefig(OUT/f'unit_{r.unit_id}.png',dpi=150)
   plt.close(fig)
 assert len(c)==17 and len(e)==len(pd.read_csv(ep));assert e.loc[~e.time_s.between(tt[0],tt[-1]),'dredge_relative_um'].isna().all()
 (OUT/'README.md').write_text('''# Direct comparison of17 waveform-only candidates with DREDGE\n\nOpen01_all_17_depth_ordered.pdf for the3-page overview, or02_individual_comparisons.pdf for all17 detailed comparisons with event-time residuals. Candidates are ordered by seed centroid across the probe.\n\nThe frozen compensated-input DREDGE field is evaluated at each fixed seed centroid, with its median over available seed-spike times removed. Waveform displacement retains the original seed-centroid reference. No later offset,sign,gain,lag,identity or trajectory is fitted to the field. The first10s are training overlap; the epoch has already been inspected during development.\n\nStrict filled-blue,lower-score open-blue and ambiguous orange observations remain separate. Black line:1s DREDGE field samples connected for display. Black plus signs on detailed panels:field values at strict observation times. Residuals use each actual event time. The field spans930.5–1029.5s; no extrapolation is made, and reference_support.csv records excluded seed spikes at the edge. Peak-input compensation refers to the cached field's input; waveform matching used original referenced input.\n\nThere is no consensus,smoothing or gap-filled waveform trajectory. Full per-unit y extents preserve alternate depth hypotheses. Descriptive differences are not calibrated physical errors: waveform lookalikes,seed-reference uncertainty,coarse translation sensitivity and low coverage remain possible. Multiple candidate templates can describe the same biological family. This comparison is a direct visual check before any aggregation.\n''')
 paths=[mp,cp,ep,fp,sp,Path(__file__)];(OUT/'manifest.json').write_text(json.dumps(dict(status='complete',units=17,observations=len(e),event_time_comparisons=int(e.dredge_relative_um.notna().sum()),unchanged_waveform_rows=True,field_extrapolation=False,source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}),indent=2));print(OUT,flush=True)
if __name__=='__main__':main()
