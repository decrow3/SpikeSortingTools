"""Direct cached measurement plots: no tracking, smoothing, or consensus fitting."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];SRC=ROOT/'testing/outputs';IN=SRC/'luke_population_depth_v2';OUT=SRC/'luke_lighthouse_direct_check_v1'
COLORS=['#0072B2','#D55E00','#009E73','#CC79A7','#E69F00','#333333','#56B4E9'];MARKERS=['o','x','^','s','+','D','v']

def nearest(t,s):
 i=np.searchsorted(s,t);a=np.clip(i,0,len(s)-1);b=np.clip(i-1,0,len(s)-1)
 return np.where(abs(t-s[a])<abs(t-s[b]),a,b)

def main():
 OUT.mkdir(exist_ok=True);cp=SRC/'luke_population_consensus_v1/candidate_consensus_eligibility.csv';c=pd.read_csv(cp);c=c[c.consensus_eligible].sort_values('depth_um');ep=IN/'events.csv';e=pd.read_csv(ep);e=e[(e.status=='accepted')&e.unit_id.isin(c.unit_id)].copy();e['relative_um']=e.centroid_um-e.unit_id.map(c.set_index('unit_id').template_centroid_um);e.to_csv(OUT/'accepted_measurements.csv',index=False)
 fig,axes=plt.subplots(6,2,figsize=(15,14),layout='constrained')
 for r in range(6):
  units=c[(c.depth_um>=r*640)&(c.depth_um<(r+1)*640)]
  for k,u in enumerate(units.itertuples()):
   q=e[e.unit_id==u.unit_id]
   for ax in axes[r]:ax.scatter(q.time_s,q.relative_um,s=11,c=COLORS[k%7],marker=MARKERS[k%7],alpha=.55,linewidths=.6,label=f'{u.unit_id} · {u.depth_um:g} µm · n={len(q)}',rasterized=True)
  for col,ax in enumerate(axes[r]):
   ax.axhline(0,color='gray',lw=.6);ax.set(xlim=(930,1030) if col==0 else (950,985),ylim=(-150,150),ylabel='Relative centroid (µm)',title=f'{r*640}–{(r+1)*640} µm'+(' · all observations' if col==0 else ' · transition detail'));ax.grid(alpha=.12);ax.ticklabel_format(axis='x',style='plain',useOffset=False)
   if col==0:ax.axvspan(930,940,color='gray',alpha=.12);ax.legend(fontsize=6,ncol=2,loc='lower left')
 for ax in axes[-1]:ax.set_xlabel('Recording time (s)')
 fig.suptitle('Do independently tracked cells visibly move together?\nRaw accepted-event scatter by cell · no binning, interpolation, DREDGE, or consensus',fontsize=14)
 fig.supxlabel('Each point is a transported-support waveform energy centroid, not calibrated physical displacement. Seed offsets use930–940s (shaded).\nRecognition bias and noisy individual-event centroids remain limitations; missing points do not establish no movement.',fontsize=9)
 for ext in ['png','pdf']:fig.savefig(OUT/f'01_direct_scatter.{ext}',dpi=150)
 plt.close(fig)
 # Reciprocal nearest spike pairs; each event contributes at most once per cell pair.
 pairs=[];stats=[]
 for r in range(6):
  units=c[(c.depth_um>=r*640)&(c.depth_um<(r+1)*640)]
  for i,u in enumerate(units.itertuples()):
   for v in list(units.itertuples())[i+1:]:
    if u.family_id==v.family_id or abs(u.depth_um-v.depth_um)>400:continue
    a=e[(e.unit_id==u.unit_id)&(e.time_s>=940)].sort_values('time_s');b=e[(e.unit_id==v.unit_id)&(e.time_s>=940)].sort_values('time_s')
    if not len(a) or not len(b):continue
    ta=a.time_s.to_numpy();tb=b.time_s.to_numpy();j=nearest(ta,tb);back=nearest(tb,ta);ix=np.flatnonzero((back[j]==np.arange(len(a)))&(abs(ta-tb[j])<=.25));jj=j[ix]
    if not len(ix):continue
    ya=a.relative_um.to_numpy()[ix];yb=b.relative_um.to_numpy()[jj];delta=ya-yb;pid=f'{u.unit_id}_{v.unit_id}'
    for k,l,aa,bb,dd in zip(ix,jj,ya,yb,delta):pairs.append(dict(pair=pid,region=r,unit_a=u.unit_id,unit_b=v.unit_id,time_a=ta[k],time_b=tb[l],time_s=(ta[k]+tb[l])/2,lag_s=abs(ta[k]-tb[l]),a_um=aa,b_um=bb,difference_um=dd))
    stats.append(dict(pair=pid,region=r,unit_a=u.unit_id,unit_b=v.unit_id,depth_a=u.depth_um,depth_b=v.depth_um,pairs=len(ix),median_abs_difference_um=np.median(abs(delta)),median_signed_difference_um=np.median(delta),difference_mad_um=np.median(abs(delta-np.median(delta))),time_span_s=max(ta[ix].max(),tb[jj].max())-min(ta[ix].min(),tb[jj].min()),median_time_separation_s=np.median(abs(ta[ix]-tb[jj]))))
 ps=pd.DataFrame(pairs);st=pd.DataFrame(stats);ps.to_csv(OUT/'matched_event_pairs.csv',index=False);st.to_csv(OUT/'pairwise_summary.csv',index=False)
 fig,axes=plt.subplots(6,2,figsize=(15,14),layout='constrained');selected=[]
 for r in range(6):
  candidates=st[st.region==r].sort_values(['pairs','pair'],ascending=[False,True])
  if candidates.empty:
   for ax in axes[r]:ax.text(.5,.5,'No matched pair',ha='center',transform=ax.transAxes)
   continue
  s=candidates.iloc[0];selected.append(s.pair);q=ps[ps.pair==s.pair]
  ax=axes[r,0];ax.scatter(q.time_a,q.a_um,s=13,c='#0072B2',marker='o',alpha=.65,label=f'Candidate {s.unit_a:g}');ax.scatter(q.time_b,q.b_um,s=15,c='#D55E00',marker='x',alpha=.65,label=f'Candidate {s.unit_b:g}');ax.legend(fontsize=8);ax.set(title=f'{s.depth_a:g}/{s.depth_b:g} µm · {len(q)} reciprocal-nearest spike pairs',ylabel='Seed-relative centroid (µm)')
  ax=axes[r,1];ax.scatter(q.time_s,q.difference_um,s=12,c='#333333',alpha=.6);ax.set(title=f'Paired difference · median |difference| {s.median_abs_difference_um:.1f} µm',ylabel='Candidate A − B (µm)')
  for ax in axes[r]:ax.axhline(0,color='gray',lw=.5);ax.set(xlim=(940,1030),ylim=(-150,150));ax.grid(alpha=.12);ax.ticklabel_format(axis='x',style='plain',useOffset=False)
 for ax in axes[-1]:ax.set_xlabel('Recording time (s)')
 fig.suptitle('Direct nearby-cell comparison outside training\nOne pair per depth region, selected only by number of matched events; ≤0.25s time separation',fontsize=14)
 fig.supxlabel('Reciprocal nearest matches use each event at most once per pair. All eligible pairs are saved in CSV.\nA small difference may reflect shared stationarity or selection bias; it does not by itself demonstrate recovery of motion.',fontsize=9)
 for ext in ['png','pdf']:fig.savefig(OUT/f'02_direct_pairs.{ext}',dpi=150)
 plt.close(fig)
 summary=dict(events=len(e),units_with_events=e.unit_id.nunique(),eligible_pairs=len(st),shown_pairs=selected,median_across_pair_median_abs_difference_um=float(st.median_abs_difference_um.median()),sources_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [cp,ep,Path(__file__)]},scope='Direct cached measurements only; no new tracking or consensus.')
 (OUT/'summary.json').write_text(json.dumps(summary,indent=2));print(st.to_string(index=False));print(summary)
if __name__=='__main__':main()
