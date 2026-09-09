"""Audit existing seed reference without altering fits or later-event alignment."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from testing.luke_early_sigma_screen250_report_v1 import references, observations
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'testing/outputs/luke_dredge_offset_audit_v1';OUT.mkdir(exist_ok=True)
P=ROOT/'testing/outputs/luke_ap_methods_partial32_v1'
fs,c,e,seeds=references();m=pd.read_csv(P/'manifest.csv');refs=pd.read_csv(P/'full_reference_support.csv');pred=pd.read_csv(P/'event_matched_predictions.csv');lo,hi=json.loads((P/'report_validation.json').read_text())['common_time_support_s'];rows=[];curves={}
for r in m[m.method=='dredge'].itertuples():
 z=np.load(r.field);t=z['time_s']
 for cell in c.itertuples():
  u=cell.unit_id;v=np.array([np.interp(cell.seed_centroid_um,z['depth_um'],x) for x in z['displacement_um']]);st=seeds[f'unit_{u}_seed_frames']/fs;assert np.all((st>=930)&(st<940));st=st[(st>=lo)&(st<=hi)];off=np.median(np.interp(st,t,v));saved=refs[(refs.input==r.input)&(refs.method=='dredge')&(refs.unit_id==u)].iloc[0];assert abs(off-saved.reference_offset_um)<1e-8
  q=e[e.unit_id==u];assert np.allclose(q.centroid_um-cell.seed_centroid_um,q.relative_um)
  p=pred[(pred.input==r.input)&(pred.method=='dredge')&(pred.unit_id==u)];a=p.predicted_um.notna();assert np.allclose(np.interp(p.loc[a,'time_s'],t,v-off),p.loc[a,'predicted_um'])
  for cls in ['strict_accepted','lower_score','identity_ambiguous']:
   s=p[(p.time_s>=lo)&(p.time_s<940)&(p.evidence==cls)&p.predicted_um.notna()];res=s.relative_um-s.predicted_um
   rows.append(dict(input=r.input,unit_id=u,evidence=cls,seed_matches=len(s),raw_field_offset_um=off,seed_median_residual_um=float(res.median()),seed_residual_q25_um=float(res.quantile(.25)),seed_residual_q75_um=float(res.quantile(.75))))
  curves[r.input,u]=(t,v-off)
a=pd.DataFrame(rows);a.to_csv(OUT/'seed_offset_audit.csv',index=False)
with PdfPages(OUT/'01_seed_offset_check.pdf') as pdf:
 for cell in c.itertuples():
  fig,axs=plt.subplots(1,2,figsize=(14,5),layout='constrained')
  for ax,screen in zip(axs,['none','full']):
   name=f'5sigma_{screen}_bound250';observations(ax,e[e.unit_id==cell.unit_id]);t,v=curves[name,cell.unit_id];ax.plot(t,v,color='#0072B2',label='Current DREDGE reference');row=a[(a.input==name)&(a.unit_id==cell.unit_id)&(a.evidence=='strict_accepted')].iloc[0]
   ax.set_xlim(930,940);ax.set_title(f'5σ {screen}: {row.seed_matches} strict seed matches');ax.legend(fontsize=8)
   ax.text(.02,.02,f'Strict seed median (waveform − trace): {row.seed_median_residual_um:.1f} µm',transform=ax.transAxes,fontsize=9)
  fig.suptitle(f'Unit {cell.unit_id}: offset audit · seed interval only');pdf.savefig(fig)
  if cell.unit_id in [657,632]:fig.savefig(OUT/f'unit_{cell.unit_id}.png',dpi=130)
  plt.close(fig)
(OUT/'validation.json').write_text(json.dumps(dict(fields=16,candidates=17,saved_offsets_reproduced=True,saved_event_predictions_reproduced=True,waveform_reference_verified=True,seed_times_verified=True,common_time_support_s=[lo,hi],fits_changed=False),indent=2))
print(a[(a.input=='5sigma_none_bound250')&(a.evidence=='strict_accepted')][['unit_id','seed_matches','raw_field_offset_um','seed_median_residual_um']].to_string(index=False))
