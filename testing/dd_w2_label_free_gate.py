"""Frozen T8 label-free correctness gate for the DD W2 lattice remap."""
from __future__ import annotations
import hashlib,json,os,sys
from pathlib import Path
import numpy as np,pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from testing import luke_imec1_medicine_m_reference_v1 as mref

FS=29999.759166666667;LEFT=26999783/FS;RIGHT=37199701/FS;BIN=.25
BASE=Path('/media/huklab/Data/NPX/Ryansorting/Luke/luke_imec1_medicine_reference_sweep_v2')
CAT=BASE/'stage3_q/episode_catalogue.csv'; KNOTS=Path('/mnt/NPX/Luke/DARTsort_motion_experiments/dd_lattice_w2_20260928/host_h1/preflight/W2_KNOT_Q.csv')
def sha(p):
 h=hashlib.sha256();
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def wj(p,x):
 q=p.with_suffix(p.suffix+'.partial');q.write_text(json.dumps(x,indent=2,sort_keys=True,default=str)+'\n');os.replace(q,p)
def load_peaks():
 rows=[]
 for i in range(7,11):
  with np.load(BASE/f'stage3_q/peak_cache/full_block_{i:03d}/extraction/population.npz',allow_pickle=False) as z:
   rows.append({k:np.asarray(z[k]) for k in ('time_s','depth_um','x_um')})
 t=np.concatenate([r['time_s']+120*i for i,r in zip(range(7,11),rows)]);d=np.concatenate([r['depth_um'] for r in rows]);x=np.concatenate([r['x_um'] for r in rows]);keep=(t>=LEFT)&(t<RIGHT)
 return {'time':t[keep],'depth':d[keep],'x':x[keep]}
def shifted(peaks,knots):
 right=knots.time_s.to_numpy()+.125;pos=np.searchsorted(right,peaks['time'],side='right');q=knots.q_um.to_numpy()[pos]
 return {'time':peaks['time'],'depth':peaks['depth']-q,'x':peaks['x']}
def measure(peaks,row,allcat,rng=None,targets=None):
 ep=(peaks['time']>=row.start_s)&(peaks['time']<row.end_s);rest=(peaks['time']>=row.start_s-4)&(peaks['time']<row.start_s-1)
 for c in allcat.itertuples(): rest&=~((peaks['time']>=c.start_s)&(peaks['time']<c.end_s))
 kw={} if targets is None else {'rng':rng,'episode_target':targets[0],'rest_target':targets[1]}
 return mref.map_shift(peaks,ep,rest,**kw)
def main():
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();out=a.output.resolve()
 if out.exists():raise FileExistsError(out)
 out.mkdir(parents=True);peaks=load_peaks();knots=pd.read_csv(KNOTS);corrected=shifted(peaks,knots);cat=pd.read_csv(CAT);inside=cat[(cat.start_s>=LEFT)&(cat.end_s<=RIGHT)].copy();accepted=inside[inside.status=='accepted'];rows=[];blocks=[]
 for ep in accepted.itertuples():
  for arm,p in [('KS_0_INPUT',peaks),('KS_L_INPUT',corrected)]:
   r=measure(p,ep,inside)
   # Original accepted references retain gain>=.05. After correction, a best
   # shift of zero necessarily has zero gain over the zero-shift correlation,
   # so residual resolution is finite shift plus the frozen peak support.
   ok=(np.isfinite(r['best_shift_um']) and r['episode_peaks']>=mref.MIN_PEAKS and r['rest_peaks']>=mref.MIN_PEAKS and (arm=='KS_L_INPUT' or r['corr_gain']>=.05))
   rows.append({'episode_id':ep.episode_id,'arm':arm,'catalogue_shift_um':ep.measured_shift_um,**{k:v for k,v in r.items() if k!='correlations'},'resolved':ok})
   if arm=='KS_L_INPUT':
    for bi,(lo,hi) in enumerate([(0,950),(950,1900),(1900,2850),(2850,3840)]):
     sel=(p['depth']>=lo-150)&(p['depth']<hi+150);rp={k:v[sel] for k,v in p.items()};br=measure(rp,ep,inside);blocks.append({'episode_id':ep.episode_id,'block':bi,'best_shift_um':br['best_shift_um'],'corr_gain':br['corr_gain'],'episode_peaks':br['episode_peaks'],'rest_peaks':br['rest_peaks'],'resolved':bool(np.isfinite(br['best_shift_um']) and br['episode_peaks']>=mref.MIN_PEAKS and br['rest_peaks']>=mref.MIN_PEAKS)})
 frame=pd.DataFrame(rows);frame.to_csv(out/'EPISODE_SHIFTS.csv',index=False);pd.DataFrame(blocks).to_csv(out/'DEPTH_BLOCK_RESIDUALS.csv',index=False)
 # Deterministic matched-duration/peak-count quiet null, first 20 valid placements.
 active=np.zeros(int(np.ceil((RIGHT-LEFT)/BIN)),bool)
 for c in inside.itertuples():
  i0=max(0,int(np.floor((c.start_s-LEFT)/BIN)));i1=min(len(active),int(np.ceil((c.end_s-LEFT)/BIN)));active[i0:i1]=True
 rng=np.random.default_rng(20260928);nulls=[];durations=accepted.duration_s.to_numpy();tries=0
 while len(nulls)<20 and tries<10000:
  tries+=1;dur=float(durations[(tries-1)%len(durations)]);start=LEFT+4+BIN*int(rng.integers(max(1,int((RIGHT-LEFT-4-dur)/BIN)))) ;stop=start+dur
  i0=int((start-LEFT)/BIN);i1=int(np.ceil((stop-LEFT)/BIN));r0=max(0,i0-16);r1=max(0,i0-4)
  if active[i0:i1].any() or active[r0:r1].any():continue
  pseudo=type('P',(),{'start_s':start,'end_s':stop})();base_ep=accepted.iloc[len(nulls)%len(accepted)];target=(int(base_ep.episode_peaks),int(base_ep.rest_peaks))
  r=measure(peaks,pseudo,inside,rng,target);nulls.append({'index':len(nulls),'start_s':start,**{k:v for k,v in r.items() if k!='correlations'},'resolved':bool(np.isfinite(r['best_shift_um']))})
 nf=pd.DataFrame(nulls);nf.to_csv(out/'MATCHED_NULL.csv',index=False);vals=nf.loc[nf.resolved,'best_shift_um'].to_numpy();unique,count=np.unique(vals,return_counts=True);mode=float(unique[np.argmax(count)]) if len(vals) else np.nan;median_abs=float(np.median(np.abs(vals))) if len(vals) else np.nan
 q0=frame[(frame.arm=='KS_0_INPUT')&frame.resolved];ql=frame[(frame.arm=='KS_L_INPUT')&frame.resolved];merged=q0.merge(ql,on='episode_id',suffixes=('_q0','_ql'));unreproduced=np.abs(merged.best_shift_um_q0-merged.catalogue_shift_um_q0)>40;residual_ok=np.abs(merged.best_shift_um_ql)<=40
 gate={'status':'pass' if len(vals)>=20 and mode==0 and median_abs<=10 and len(merged)>0 and residual_ok.mean()>=.8 and not unreproduced.any() else 'fail','resolved_nulls':len(vals),'null_mode_um':mode,'null_median_abs_shift_um':median_abs,'resolved_episode_pairs':len(merged),'remapped_residual_le40_fraction':float(residual_ok.mean()) if len(merged) else np.nan,'unremapped_catalogue_reproduction_failures':int(unreproduced.sum()),'criterion':'null mode 0 and median abs <=10; >=80% resolved episodes residual abs<=40; unremapped within40 of catalogue','interpretation':'input correctness only; not biological identity'}
 wj(out/'GATE.json',gate);wj(out/'COMPLETE.json',{'status':gate['status'],'gate_sha256':sha(out/'GATE.json'),'written_last':True})
if __name__=='__main__':main()
