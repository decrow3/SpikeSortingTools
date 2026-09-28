"""Finalize h1's first DE saved-output scorecard and resource receipt."""
from pathlib import Path
import hashlib,json,os
import pandas as pd, numpy as np

ROOT=Path('/mnt/NPX/Luke/DARTsort_motion_experiments/de_common_outcome_scorecard_20260928/host_h1')
RES=ROOT/'results'; DC=Path('/mnt/NPX/Luke/DARTsort_motion_experiments/dc_motion_domain_and_native_bank_review_20260928')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''): h.update(b)
 return h.hexdigest()
def wj(p,x):
 q=p.with_suffix(p.suffix+'.partial');q.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n');os.replace(q,p)
def main():
 a=pd.read_csv(DC/'DOMAIN_EVENT_COUNTS.csv');b=pd.read_csv(RES/'PRIMARY_COUNTS_RATES.csv')
 aa=a[(a.arm=='STATIC_S')&(a.domain=='negative_excursion')].iloc[0];bb=b[(b.window=='W2')&(b.arm=='STATIC_S')&(b.subset=='all_units')&(b.domain=='negative_excursion')].iloc[0]
 di=pd.read_csv(DC/'DOMAIN_INTERVALS.csv');ei=pd.read_csv(RES/'DOMAIN_INTERVALS.csv');w=ei[ei.window=='W2'].drop(columns='window').reset_index(drop=True)
 repro={'selected_static_row_exact':all(float(aa[x])==float(bb[y]) for x,y in [('exposure_s','exposure_s'),('all_events','all_events'),('assigned_events','assigned_events'),('noise_events','noise_events'),('assigned_event_rate_hz','assigned_rate_hz')]),'w2_interval_row_count':len(w),'all_boundary_values_exact':bool(np.array_equal(di[['start_s','end_s']].to_numpy(),w[['start_s','end_s']].to_numpy())),'all_boundary_labels_exact':bool(np.array_equal(di.domain.to_numpy(),w.domain.to_numpy()))}
 if not all(repro.values()): raise ValueError(repro)
 wj(ROOT/'DC_INDEPENDENT_REPRODUCTION.json',repro)
 primary=b[b.subset=='all_units'].copy(); primary.to_csv(ROOT/'EXISTING_ARM_PRIMARY_TABLE.csv',index=False)
 wj(ROOT/'RESOURCE_RECEIPT.json',{'actual_cpu_s':121.26,'actual_wall_s':134.35,'charged_cpu_s':150.0,'h1_cumulative_before_s':17633.22,'h1_cumulative_after_s':17783.22,'h1_operational_ceiling_s':26000.0,'raw_voltage_reads':0,'gpu_s':0,'threads':2,'readers':1})
 report='''# DE h1 existing-arm scorecard\n\n## Verdict first\n\nReady within the saved-output scope. The common source exactly reproduces DC's selected static row and all 67 W2 domain boundaries. It adds NATIVE_REMATCH0, W3 D2L, and the full-session rescue KS12/9 contextual rows without cross-arm unit joins.\n\nKilosort has no DARTsort-style negative-noise rows, and its full-session training context differs from the window sorts; its rates are therefore contextual rather than a like-for-like yield winner. `catalogue_outside_remainder` is exhaustive operational support, not true rest.\n\nD2L has stronger flat-to-excursion within-unit rank association than static in W2 (0.657 vs 0.209); native rematch is similar (0.663). These are descriptive arm-local associations, not identity tests. Exact assigned/noise/rate, catalogue cross-tab, segment-safe ISI and eligibility rows are in `results/`. The RF outer holdout remains sealed.\n'''
 (ROOT/'REPORT.md').write_text(report)
 products=[{'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(ROOT.rglob('*')) if p.is_file() and p.name not in {'MANIFEST.json','COMPLETE.json'}]
 wj(ROOT/'MANIFEST.json',{'products':products}); wj(ROOT/'COMPLETE.json',{'status':'complete_existing_arms','manifest_sha256':sha(ROOT/'MANIFEST.json'),'outer_rf_holdout_opened':False,'written_last':True})
if __name__=='__main__':main()
