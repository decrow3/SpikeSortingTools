"""DE v2 finite repair: affected schemas and per-unit controls only."""
from pathlib import Path
import hashlib,json,os,sys
import numpy as np,pandas as pd
from scipy.stats import spearmanr
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from testing import de_common_outcome_scorecard as de

ROOT=Path('/mnt/NPX/Luke/DARTsort_motion_experiments/de_common_outcome_scorecard_20260928/host_h1'); OLD=ROOT/'results'; OUT=ROOT/'revision_v2'; CFG=Path('/home/huklab/Documents/RyanSorting/SpikeSortingTools/testing/inputs/de_common_scorecard_h1_20260928.json')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def wj(p,x):
 q=p.with_suffix(p.suffix+'.partial');q.write_text(json.dumps(x,indent=2,sort_keys=True,default=str)+'\n');os.replace(q,p)
def main():
 if OUT.exists():raise FileExistsError(OUT)
 OUT.mkdir();cfg=json.loads(CFG.read_text());fs=float(cfg['sampling_frequency_hz']);old=pd.read_csv(OLD/'PRIMARY_COUNTS_RATES.csv')
 old=old.rename(columns={'exposure_s':'domain_exposure_s'});old['subset_exposure_s']=np.where(old['subset'].isin(['all_units','good_units']),old.domain_exposure_s,np.nan)
 dart=old.arm!='RESCUE_KS12_9';old=old[~(dart&(old.subset=='good_units'))].copy();old['good_label_availability']=np.where(old.arm=='RESCUE_KS12_9','available','unavailable');clock={a:('full_source' if a=='RESCUE_KS12_9' else 'window_local') for a in old.arm.unique()};old['clock_type']=old.arm.map(clock);old.to_csv(OUT/'PRIMARY_COUNTS_RATES_SCHEMA_CORRECTED.csv',index=False)
 with np.load(cfg['field'],allow_pickle=False) as z:ft=np.asarray(z['time_s'],float);disp=np.asarray(z['displacement_um'],float)[:,0]
 baseline=pd.Series(disp).rolling(480,center=True,min_periods=1).median().to_numpy();dev=disp-baseline;mask=pd.read_csv(cfg['mask'])[['start_s','end_s']].to_numpy(float);ivals_all=pd.read_csv(OLD/'DOMAIN_INTERVALS.csv');units=[];summary=[]
 for w in cfg['windows']:
  name=w['name'];start,end=map(int,w['frames_half_open']);iv=ivals_all[ivals_all.window==name];exposure=iv.groupby('domain').duration_s.sum().reindex(de.DOMAINS);ivals=list(iv[['start_s','end_s','domain']].itertuples(index=False,name=None))
  for arm,spec in w['arms'].items():
   times,labels,depth,assigned,good,noise,ctype=de.load_arm(spec,start,end,fs);sec=times/fs;domain=de.assign_domain(sec,ft,dev,mask)
   for d in de.DOMAINS:
    seg=de.segment_id(sec,ivals,d);inside=domain==d
    if np.any(seg[inside]<0) or np.any(seg[~inside]>=0):raise AssertionError(f'{name}/{arm}/{d}')
   rows=[]
   for u in np.unique(labels[assigned]):
    own=assigned&(labels==u);flat=own&(domain==de.DOMAINS[1]);row={'window':name,'arm':arm,'unit_id':int(u),'median_flat_event_depth_um':float(np.median(depth[flat])) if flat.any() else np.nan}
    for d in de.DOMAINS:
     n=int(np.sum(own&(domain==d)));row[f'count_{d}']=n;row[f'log_rate_{d}']=float(np.log((n+.5)/exposure[d]))
    row['negative_to_flat_rate_ratio']=(row['count_negative_excursion']/exposure[de.DOMAINS[0]])/(row['count_outside_mask_flat']/exposure[de.DOMAINS[1]]) if row['count_outside_mask_flat'] else np.nan;row['eligible_flat_count_ge_100']=row['count_outside_mask_flat']>=100;row['exclusion_reason']='' if row['eligible_flat_count_ge_100'] else 'flat_count_lt_100';rows.append(row)
   uf=pd.DataFrame(rows);eligible=uf[uf.eligible_flat_count_ge_100].copy();control=[]
   for r in eligible.itertuples():
    near=eligible[(np.abs(eligible.median_flat_event_depth_um-r.median_flat_event_depth_um)<=10)&(eligible.unit_id!=r.unit_id)];control.append(float(near.log_rate_outside_mask_flat.mean()) if len(near) else np.nan)
   eligible['same_row_other_unit_flat_log_rate']=control;uf=uf.merge(eligible[['unit_id','same_row_other_unit_flat_log_rate']],how='left',on='unit_id');units.extend(uf.to_dict('records'));ok=np.isfinite(eligible.same_row_other_unit_flat_log_rate);r=spearmanr(eligible.log_rate_outside_mask_flat,eligible.log_rate_negative_excursion);rc=spearmanr(eligible.loc[ok,'same_row_other_unit_flat_log_rate'],eligible.loc[ok,'log_rate_negative_excursion']) if ok.sum()>=3 else (np.nan,np.nan);summary.append({'window':name,'arm':arm,'all_units':len(uf),'eligible_units':len(eligible),'excluded_units':len(uf)-len(eligible),'rho_negative_vs_flat':float(r.statistic),'rho_negative_vs_same_row_other_flat':float(rc.statistic if hasattr(rc,'statistic') else rc[0]),'same_row_control_units':int(ok.sum()),'same_row_depth_source':'own outside_mask_flat events','exact_own_id_excluded':True})
 pd.DataFrame(units).to_csv(OUT/'ALL_UNIT_RATE_CONTROLS.csv',index=False);pd.DataFrame(summary).to_csv(OUT/'UNIT_RATE_ASSOCIATIONS_CORRECTED.csv',index=False);wj(OUT/'VALIDATION.json',{'status':'pass','unchanged_counts_reused_from':str(OLD/'PRIMARY_COUNTS_RATES.csv'),'domain_counts_recomputed':False,'rf_rerun':False,'outer_holdout_opened':False,'domain_segment_assertions':'pass','source_sha256':sha(Path(de.__file__))});products=[{'path':p.name,'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='COMPLETE.json'];wj(OUT/'COMPLETE.json',{'status':'complete_finite_revision','products':products,'written_last':True})
if __name__=='__main__':main()
