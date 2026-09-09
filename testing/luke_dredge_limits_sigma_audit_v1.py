"""Read-only cached-field audit of displacement bounds and sigma amplitude trends."""
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];SRC=ROOT/'testing/outputs';OUT=SRC/'luke_dredge_limits_sigma_audit_v1'
def main():
 OUT.mkdir(exist_ok=True);rows=[];summary=[];paths=[]
 for label,path in [('early_5sigma',SRC/'luke_early_screen_shortlist_v1/fields/compensated.npz')]+[(f'{s}sigma',SRC/f'luke_detection_threshold_sweep_v1/fields/{s}sigma.npz') for s in [3,4,5,6]]:
  paths.append(path);z=np.load(path);m=z['displacement_um'];d=z['D'];t=z['time_s'];valid=(abs(t[:,None]-t[None,:])<=60)&~np.eye(len(t),dtype=bool);valid=np.broadcast_to(valid,d.shape);active=valid&(z['U']>0);ptp=np.ptp(m,axis=0);rob=np.quantile(m,.975,axis=0)-np.quantile(m,.025,axis=0)
  assert m.shape==(len(t),len(z['depth_um'])) and abs(d).max()<=80
  for depth,a,b in zip(z['depth_um'],ptp,rob):rows.append(dict(arm=label,depth_um=depth,peak_to_peak_um=a,central95_range_um=b))
  summary.append(dict(arm=label,start_center_s=t[0],end_center_s=t[-1],median_depth_peak_to_peak_um=float(np.median(ptp)),median_depth_central95_range_um=float(np.median(rob)),field_min_um=float(m.min()),field_max_um=float(m.max()),active_pairs=int(active.sum()),active_pairs_at_bound_fraction=float(np.mean(abs(d[active])==80)),pair_weight_at_bound_fraction=float(np.sum(z['U'][active]*(abs(d[active])==80))/np.sum(z['U'][active]))))
 table=pd.DataFrame(rows);table.to_csv(OUT/'by_depth.csv',index=False);s=pd.DataFrame(summary);s.to_csv(OUT/'summary.csv',index=False);base=table[table.arm=='5sigma'].set_index('depth_um').peak_to_peak_um;comparisons=[]
 for arm in ['3sigma','4sigma','6sigma']:
  a=table[table.arm==arm].set_index('depth_um').peak_to_peak_um;assert a.index.equals(base.index);comparisons.append(dict(arm=arm,depths_smaller_than5=int((a<base).sum()),depths_larger_than5=int((a>base).sum()),median_paired_percent_change=float(np.median(100*(a/base-1)))))
 pd.DataFrame(comparisons).to_csv(OUT/'paired_vs5sigma.csv',index=False);paths.append(Path(__file__));(OUT/'manifest.json').write_text(json.dumps(dict(status='complete',metric='Temporal maximum minus minimum at each depth, then median across20 depth windows; central95% range is97.5th minus2.5th temporal percentile. Offset invariant.',boundary='Off-diagonal pairs≤60s apart with positive stored U; counts include both directions. Boundary fractions do not measure lost large-shift matches.',new_estimation=False,source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}),indent=2));print(s.to_string(index=False));print(comparisons)
if __name__=='__main__':main()
