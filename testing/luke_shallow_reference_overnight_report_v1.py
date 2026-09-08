"""Postprocess completed shallow-reference stages; cached tables/fields only."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];SRC=ROOT/'testing/outputs'
INPUT=SRC/'luke_shallow_reference_overnight_v1';OUT=SRC/'luke_shallow_reference_overnight_report_v1'

def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def validate(stage,runhash):
 r=json.loads((stage/'receipt.json').read_text());assert r['status']=='complete' and r['run_sha256']==runhash
 actual={str(p.relative_to(stage)) for p in stage.rglob('*') if p.is_file() and p.name!='receipt.json'}
 assert actual==set(r['files'])
 for name,digest in r['files'].items():assert sha(stage/name)==digest,f'Corrupt {stage/name}'
 return r
def main():
 settings=json.loads((INPUT/'settings.json').read_text());runhash=hashlib.sha256(json.dumps(settings,sort_keys=True).encode()).hexdigest()
 validate(INPUT/'report',runhash);result=json.loads((INPUT/'report/summary.json').read_text());assert result['status']=='complete'
 qualified=pd.read_csv(INPUT/'report/qualification.csv');passed=result['qualified_contextual_ids'];assert set(passed)==set(qualified[(qualified['mode']=='contextual')&qualified.qualified].target)
 rows=[];provenance={str(INPUT/'report/receipt.json'):sha(INPUT/'report/receipt.json'),str(Path(__file__).resolve()):sha(Path(__file__).resolve())};fields_path=SRC/'luke_3sigma_lowpass_screen_v2/fields/broad_3sigma.npz';field=np.load(fields_path)
 provenance[str(fields_path)]=sha(fields_path)
 if passed:
  for unit in passed:
   depth=float(qualified.loc[qualified.target==unit,'depth_um'].iloc[0]);v=np.array([np.interp(depth,field['depth_um'],d) for d in field['displacement_um']]);parts=[]
   for start in settings['target_chunks_s']:
    stage=INPUT/f'track_u{unit}_{start}';validate(stage,runhash);provenance[str(stage/'receipt.json')]=sha(stage/'receipt.json')
    support=pd.read_csv(stage/'support_5s.csv');events=pd.read_csv(stage/'accepted_events.csv')
    for r in support.to_dict('records'):
     r.update(unit_id=unit,depth_um=depth,field_median_um=np.nan,comparison_events=0)
     if r['status']=='supported':
      q=events[(events.time_s>=r['start_s'])&(events.time_s<r['start_s']+5)&events.unique_shift];mode=q.shift_um.mode().iloc[0];q=q[q.shift_um==mode]
      assert len(q)>=10 and len(q)==r['bootstrap_events'];r['comparison_events']=len(q)
      r['field_median_um']=float(np.median(np.interp(q.time_s,field['time_s'],v)))
     parts.append(r)
   parts=pd.DataFrame(parts).sort_values('start_s');supported=parts[parts.status=='supported']
   base=supported.iloc[0] if len(supported) else None
   for r in parts.to_dict('records'):
    r['reference_start_s']=float(base.start_s) if base is not None else np.nan
    for key in ['centroid','field','bootstrap_low','bootstrap_high']:r[key+'_relative_um']=np.nan
    if r['status']=='supported' and base is not None:
     r.update(centroid_relative_um=r['centroid_um']-base.centroid_um,field_relative_um=r['field_median_um']-base.field_median_um,
      bootstrap_low_relative_um=r['bootstrap_low_um']-base.centroid_um,bootstrap_high_relative_um=r['bootstrap_high_um']-base.centroid_um)
    rows.append(r)
 OUT.mkdir(exist_ok=True);qualified.to_csv(OUT/'qualification.csv',index=False)
 for key in ['quiet_recall','injected_unique_shift_recovery','worst_rival_false_target']:
  if key not in qualified:qualified[key]=np.nan
 fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
 for ax,col,title in zip(axes,['quiet_recall','injected_unique_shift_recovery','worst_rival_false_target'],['Quiet detection recall','Injected correct unique shift','Worst tested rival false acceptance']):
  names=[f'{r.target} / {r.mode}' for r in qualified.itertuples()];ax.bar(np.arange(len(qualified)),qualified[col],color=['#2878b5' if m=='contextual' else '#aaaaaa' for m in qualified['mode']]);ax.set(xticks=np.arange(len(names)),xticklabels=names,title=title,ylim=(0,1));ax.tick_params(axis='x',rotation=65,labelsize=7)
 fig.suptitle('Reused development controls; frozen acceptance gates, no motion-guided selection')
 for ext in ['png','pdf']:fig.savefig(OUT/f'01_qualification.{ext}',dpi=130)
 plt.close(fig)
 if rows:
  d=pd.DataFrame(rows);d.to_csv(OUT/'event_matched_comparison.csv',index=False)
  fig,axes=plt.subplots(len(passed),2,figsize=(14,4*len(passed)),squeeze=False,layout='constrained')
  for axs,(unit,g) in zip(axes,d.groupby('unit_id')):
   g=g.sort_values('start_s');x=g.start_s+2.5;valid=g.status=='supported'
   axs[0].plot(x,np.where(valid,g.field_relative_um,np.nan),'o-',color='#2878b5',label='Baseline DREDGE at same events')
   axs[0].plot(x,np.where(valid,g.centroid_relative_um,np.nan),'ko-',ms=4,label='Centroid of median waveform')
   axs[0].fill_between(x,g.bootstrap_low_relative_um,g.bootstrap_high_relative_um,color='#555555',alpha=.2,label='Conditional bootstrap; reference fixed')
   for r in g[~valid].itertuples():axs[0].axvspan(r.start_s,r.start_s+5,color='#aaaaaa',alpha=.2)
   axs[0].set(title=f'Unit {unit} · {g.depth_um.iloc[0]:.0f} µm · gaps stay unsupported',xlabel='Recording time (s)',ylabel='Offset-aligned change (µm)');axs[0].legend(fontsize=7)
   axs[1].bar(x,g.events,width=4,color=['#2878b5' if s=='supported' else '#aaaaaa' for s in g.status]);axs[1].axhline(10,color='black',ls=':')
   for r in g.itertuples():
    if r.status!='supported':axs[1].annotate(r.status,(r.start_s+2.5,r.events),rotation=90,fontsize=6,va='bottom')
   axs[1].set(xlabel='Recording time (s)',ylabel='Unique-shift events / 5 s',title='Support: dominant count ≥10 and ≥80% agreement\nBoundary events remain gaps')
  fig.suptitle('Qualified contextual matches: descriptive local corroboration only\nBootstrap omits reference-offset, identity and measurement-model uncertainty; no physical accuracy claim',fontsize=12)
  for ext in ['png','pdf']:fig.savefig(OUT/f'02_motion_support.{ext}',dpi=130)
  plt.close(fig)
 if not passed:
  fig,ax=plt.subplots(figsize=(10,4),layout='constrained');ax.axis('off');ax.text(.5,.5,'No candidates passed contextual qualification.\nNo motion comparison or accuracy score is available.\nFailed qualification remains evidence; no thresholds were relaxed.',ha='center',va='center',transform=ax.transAxes)
  for ext in ['png','pdf']:fig.savefig(OUT/f'02_motion_support.{ext}',dpi=130)
  plt.close(fig)
 gaps='No candidates passed contextual qualification; no motion comparison was attempted.' if not passed else 'Only supported dominant-shift events enter each comparison; sparse, mixed, or boundary bins remain gaps.'
 (OUT/'README.md').write_text('# Shallow reference overnight postprocessing\n\n'+gaps+'\n\nFields are sampled at exactly the dominant-shift events used to form each median waveform. Each unit uses its first supported bin as a fixed offset. Bands are conditional percentile bootstrap intervals shifted by that fixed reference; they exclude reference uncertainty, identity errors, and injection-model bias. Qualification is development evidence, not session-wide validation. No summary ground-truth error score is computed.\n')
 summary=dict(status='complete',qualified_ids=passed,comparison_rows=len(rows),source_sha256=provenance,source_run_sha256=runhash,interpretation=gaps)
 (OUT/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
