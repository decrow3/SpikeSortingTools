"""Method comparison on frozen event support; no fitted identity or ranking winner."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from testing.luke_early_sigma_screen250_report_v1 import references,observations,COLORS
ROOT=Path(__file__).resolve().parents[1];SRC=ROOT/'testing/outputs';OUT=SRC/'luke_ap_methods_partial32_v1';SCREENS=['none','center_only','relaxed','full']
def report(baseline_only=False):
 fs,c,e,seeds=references();manifest=pd.read_csv(OUT/'manifest.csv');fields={(r.input,r.method):np.load(r.field) for r in manifest.itertuples()};lo=max(z['time_s'][0] for z in fields.values());hi=min(z['time_s'][-1] for z in fields.values());curves={};reference_rows=[]
 for key,z in fields.items():
  for cell in c.itertuples():
   assert z['depth_um'][0]<=cell.seed_centroid_um<=z['depth_um'][-1],'Candidate outside field depth support; do not extrapolate'
   t=z['time_s'];v=np.array([np.interp(cell.seed_centroid_um,z['depth_um'],a) for a in z['displacement_um']]);st=seeds[f'unit_{cell.unit_id}_seed_frames']/fs;st=st[(st>=lo)&(st<=hi)];assert len(st);offset=float(np.median(np.interp(st,t,v)));curves[key+(cell.unit_id,)]=(t,v-offset);reference_rows.append(dict(input=key[0],method=key[1],unit_id=cell.unit_id,reference_events=len(st),reference_offset_um=offset))
 stem='00_cached5sigma' if baseline_only else 'full';pd.DataFrame(reference_rows).to_csv(OUT/f'{stem}_reference_support.csv',index=False)
 if baseline_only:
  with PdfPages(OUT/'00_cached5sigma_lighthouse_comparison.pdf') as pdf:
   for start in range(0,len(c),6):
    batch=c.iloc[start:start+6];fig,axs=plt.subplots(3,2,figsize=(15,12),layout='constrained')
    for ax,cell in zip(axs.flat,batch.itertuples()):
     observations(ax,e[e.unit_id==cell.unit_id])
     for method,color in [('dredge','#0072B2'),('medicine','#CC79A7')]:t,v=curves[('5sigma_none_bound250',method,cell.unit_id)];ax.plot(t,v,c=color,lw=1.2,label='DREDGE v8' if method=='dredge' else 'MEDiCINe v6')
     ax.set_title(f'Unit {cell.unit_id} · {cell.seed_centroid_um:.0f}µm');ax.legend(fontsize=8)
    for ax in list(axs.flat)[len(batch):]:ax.axis('off')
    fig.suptitle('Documented implementations on exact cached unscreened5σ input · 930–1030s',fontsize=14);fig.supxlabel('Blue:DREDGE0.25s bins/0.5s histogram sigma; purple:MEDiCINe0.25s bins/1s kernel. Black strict waveform,gray open lower-score,gray crosses ambiguous.\nBoth referenced at identical seed-spike times within common field support. No sign,gain,lag or later offset fitting; individual vertical scales.',fontsize=9);pdf.savefig(fig);fig.savefig(OUT/f'baseline_{start//6+1}.png',dpi=140);plt.close(fig)
  print('Baseline implementation-vs-lighthouse report ready',flush=True);return
 assert len(fields)==32
 events=[];metrics=[];amplitudes=[]
 for key,z in fields.items():
  amplitudes.append(dict(input=key[0],method=key[1],median_depth_peak_to_peak_um=float(np.median(np.ptp(z['displacement_um'],axis=0)))))
  for cell in c.itertuples():
   q=e[e.unit_id==cell.unit_id].copy();t,v=curves[key+(cell.unit_id,)];q['predicted_um']=np.interp(q.time_s,t,v,left=np.nan,right=np.nan);q.loc[~q.time_s.between(lo,hi),'predicted_um']=np.nan;q['difference_um']=q.relative_um-q.predicted_um;q['input']=key[0];q['method']=key[1];events.append(q[['input','method','unit_id','time_s','evidence','relative_um','predicted_um','difference_um']]);q=q[(q.time_s>=940)&q.predicted_um.notna()]
   for cls in ['strict_accepted','lower_score','identity_ambiguous']:
    for scope in ['all_observations','large_excursions']:
     p=q[q.evidence==cls]
     if scope=='large_excursions':p=p[abs(p.relative_um)>=120]
     metrics.append(dict(input=key[0],method=key[1],unit_id=cell.unit_id,evidence=cls,scope=scope,events=len(p),supported=len(p)>=5,median_abs_difference_um=float(np.median(abs(p.difference_um))) if len(p)>=5 else np.nan,median_signed_difference_um=float(np.median(p.difference_um)) if len(p)>=5 else np.nan))
 pd.concat(events,ignore_index=True).to_csv(OUT/'event_matched_predictions.csv',index=False);metrics=pd.DataFrame(metrics);metrics.to_csv(OUT/'per_candidate_differences.csv',index=False);metrics[metrics.supported].groupby(['input','method','evidence','scope']).agg(candidates=('unit_id','nunique'),median_candidate_abs_difference_um=('median_abs_difference_um','median')).reset_index().to_csv(OUT/'descriptive_agreement.csv',index=False);pd.DataFrame(amplitudes).to_csv(OUT/'field_amplitudes.csv',index=False)
 for method in ['dredge','medicine']:
  with PdfPages(OUT/f'01_{method}_all17_sweep.pdf') as pdf:
   for cell in c.itertuples():
    fig,axs=plt.subplots(2,2,figsize=(17,12),sharex=True,sharey=True,layout='constrained')
    for ax,screen in zip(axs.flat,SCREENS):
     observations(ax,e[e.unit_id==cell.unit_id])
     for sigma,color in COLORS.items():
      if sigma==8:continue
      t,v=curves[(f'{sigma}sigma_{screen}_bound250',method,cell.unit_id)];ax.plot(t,v,c=color,lw=1,label=f'{sigma}σ')
     ax.set_title(screen.replace('_',' '));ax.legend(fontsize=8,ncol=3)
    fig.suptitle(f'{method.upper()} · unit {cell.unit_id} · seed {cell.seed_centroid_um:.0f}µm · 930–1030s · partial32/40 (8σ missing)',fontsize=16);fig.supxlabel('Colored:method curves; black filled:strict waveform; gray open:lower-score; gray crosses:identity ambiguous. Shading:seed interval.\nSame cached input arrays per combination and frozen17 candidates. Common seed-event reference; no matching to field or smoothed waveform path.',fontsize=10);pdf.savefig(fig)
    if cell.unit_id in [161,555,657,673]:fig.savefig(OUT/f'{method}_unit_{cell.unit_id}.png',dpi=140)
    plt.close(fig)
 with PdfPages(OUT/'02_head_to_head_5sigma_6sigma.pdf') as pdf:
  for cell in c.itertuples():
   fig,axs=plt.subplots(2,2,figsize=(17,12),sharex=True,sharey=True,layout='constrained')
   for ax,screen in zip(axs.flat,SCREENS):
    observations(ax,e[e.unit_id==cell.unit_id])
    for method,color in [('dredge','#0072B2'),('medicine','#CC79A7')]:
     for sigma,ls in [(5,'-'),(6,'--')]:t,v=curves[(f'{sigma}sigma_{screen}_bound250',method,cell.unit_id)];ax.plot(t,v,c=color,ls=ls,lw=1.2,label=f'{method} {sigma}σ')
    ax.set_title(screen.replace('_',' '));ax.legend(fontsize=8,ncol=2)
   fig.suptitle(f'Documented DREDGE/MEDiCINe · 5σ and6σ examples · partial32/40 · unit {cell.unit_id}',fontsize=16);fig.supxlabel('5σ baseline and6σ highest completed threshold shown;8σ missing after failed check. Four-threshold sweeps in separate PDFs.\nBlack/gray markers preserve all waveform evidence classes; models compared on common event-time support,without fitting to lighthouses.',fontsize=10);pdf.savefig(fig);plt.close(fig)
 (OUT/'README.md').write_text('Partial results:32/40 completed combinations;3,4,5,6 sigma across four screens and both methods. All8sigma fits absent after a DREDGE consistency-check failure. Completed arm hashes verified before plotting. Frozen17 candidate identities and evidence classes; common seed-event offsets and common temporal support; no fitted gain/sign/lag. Head-to-head shows5sigma baseline and6sigma highest completed threshold, not selected winners. Fields evaluated at fixed seed depths. Differences are descriptive, not validated motion accuracy or independent-cell statistics. DREDGE20 depth windows and MEDiCINe4 depth bins have different spatial bases. Native-grid peak-to-peak summaries are not accuracy scores.\n');(OUT/'report_validation.json').write_text(json.dumps(dict(status='partial',combinations=32,missing='All8sigma combinations',candidates=17,common_time_support_s=[float(lo),float(hi)],no_spatial_extrapolation=True,reference_events_common=True),indent=2));print('Partial32 method reports complete',flush=True)
if __name__=='__main__':report()
