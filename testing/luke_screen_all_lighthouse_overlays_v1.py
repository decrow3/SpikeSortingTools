"""Nine-lighthouse pages for every saved screening-sweep arm; no estimation."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
ROOT=Path(__file__).resolve().parents[1];SRC=ROOT/'testing/outputs';OUT=SRC/'luke_screen_all_lighthouse_overlays_v1'
TITLES={'compensated':'Compensated 5σ baseline','full_screen':'Full waveform screen','without_snr':'Full screen without amplitude gate','without_center':'Full screen without center-energy gate','without_neighbor':'Full screen without neighbor gate','without_width':'Full screen without width gate','without_broad':'Full screen without broad-residual gate','without_shared':'Full screen without extra shared-artifact gate','snr_only':'Amplitude gate only','center_only':'Center-energy gate only','neighbor_only':'Neighbor gate only','snr_6':'Full screen with amplitude gate at 6σ','snr_5':'Full screen with amplitude gate at 5σ','neighbor_06':'Full screen with neighbor cosine ≥ 0.6','neighbor_07':'Full screen with neighbor cosine ≥ 0.7','center_05':'Full screen with center-energy fraction ≥ 0.50','center_055':'Full screen with center-energy fraction ≥ 0.55','relaxed_combination':'Combined relaxation: 6σ / 0.50 center / 0.6 neighbor','random_count_seed14':'Count-matched random thinning · seed 14','random_count_seed29':'Count-matched random thinning · seed 29','random_depth_time':'Depth/time-matched random thinning'}
def main():
 OUT.mkdir(exist_ok=True)
 sources=[SRC/'luke_screen_sweep_v1/event_matched_predictions.csv',SRC/'luke_screen_sweep_v1/manifest.csv',SRC/'luke_lighthouse_gentle_v1/gentle_tracks.csv']
 d=pd.read_csv(sources[0]);manifest=pd.read_csv(sources[1]);tracks=pd.read_csv(sources[2]);units=sorted(tracks.unit_id.unique());names=manifest.name.tolist()
 assert len(units)==9 and len(names)==21 and set(names)<=set(d.name)
 d=d[d.name.isin(names)]
 limits={}
 for unit,g in tracks.groupby('unit_id'):
  g=g.sort_values('time_s');base=g.iloc[0].median_waveform_centroid_um;good=g.accepted_events>=10
  vals=np.r_[d.loc[d.unit_id==unit,'predicted_um'].to_numpy(),g.loc[good,'bootstrap_low_um']-base,g.loc[good,'bootstrap_high_um']-base];vals=vals[np.isfinite(vals)]
  lo,hi=vals.min(),vals.max();pad=max(.15,(hi-lo)*.07);limits[int(unit)]=(lo-pad,hi+pad)
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
 index=['# Nine-lighthouse overlays for every screening variant','','All 21 sweep arms, including the compensated baseline and three thinning controls. Each page compares one arm against the **same compensated 5σ baseline** and the same nine provisional references. All data are saved event-matched predictions; no estimator was rerun.','','[Complete 21-page PDF](all_21_screening_lighthouse_overlays.pdf)','','Colors and symbols: blue solid circles = baseline; orange dashed squares = selected variant; black open diamonds = provisional lighthouses. Per-unit y limits are identical across pages; unsupported bins remain gaps. Error bars reproduce the original reference bootstrap intervals, not physical motion uncertainty.','','| Page | Variant | Peaks retained | PNG | PDF |','|---|---|---:|---|---|']
 with PdfPages(OUT/'all_21_screening_lighthouse_overlays.pdf') as book:
  for page,row in enumerate(manifest.itertuples(),1):
   fig,axes=plt.subplots(3,3,figsize=(15,11),sharex=True);handles=[]
   for ax,unit in zip(axes.ravel(),units):
    g=tracks[tracks.unit_id==unit].sort_values('time_s');base=float(g.iloc[0].median_waveform_centroid_um)
    for arm,color,ls,marker,label in [('compensated','#0072B2','-','o','Compensated 5σ baseline'),(row.name,'#D55E00','--','s','Selected variant')]:
     if row.name=='compensated' and label=='Selected variant':continue
     q=d[(d.name==arm)&(d.unit_id==unit)].sort_values('time_s');assert len(q)==len(g)
     h,=ax.plot(q.time_s,q.predicted_um,color=color,ls=ls,marker=marker,ms=3.5,mfc='white' if marker=='s' else color,lw=1.7,label=label)
     if unit==units[0]:handles.append(h)
    good=g.accepted_events>=10
    h=ax.errorbar(g.time_s[good],g.median_waveform_centroid_um[good]-base,yerr=np.array([g.median_waveform_centroid_um[good]-g.bootstrap_low_um[good],g.bootstrap_high_um[good]-g.median_waveform_centroid_um[good]]),fmt='D',color='black',mfc='white',ms=4.5,capsize=3,zorder=5,label='Provisional lighthouse')
    if unit==units[0]:handles.append(h)
    ax.set(title=f'Unit {unit} · {g.depth_um.iloc[0]:.0f} µm',ylim=limits[int(unit)],xlim=(4160,4260),xticks=[4160,4180,4200,4220,4240,4260],xlabel='Recording time (s)',ylabel='Relative displacement (µm)');ax.ticklabel_format(axis='x',style='plain',useOffset=False);ax.grid(alpha=.16)
   fig.suptitle(f'{page:02d}/21 · {TITLES[row.name]}\n4,160–4,260 s · {row.peaks:,} peaks · {100*row.retained_fraction:.1f}% of baseline',fontsize=15)
   fig.subplots_adjust(left=.065,right=.99,bottom=.14,top=.89,hspace=.40,wspace=.24)
   fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.058),ncols=3,frameon=False,fontsize=11)
   fig.text(.5,.015,'Same per-unit y scale across all variants; unsupported bins remain gaps.\nOriginal reference estimates and bootstrap bars are provisional, not ground truth.',ha='center',fontsize=10)
   stem=f'{page:02d}_{row.name}'
   for ext in ['png','pdf']:fig.savefig(OUT/f'{stem}.{ext}',dpi=140,bbox_inches='tight')
   book.savefig(fig,bbox_inches='tight');plt.close(fig)
   index.append(f'| {page} | {TITLES[row.name]} | {row.retained_fraction:.1%} | [View]({stem}.png) | [PDF]({stem}.pdf) |')
 index+=['','The later 3σ gentle-screen / low-pass comparison is a separate experiment: [six-arm accessible overlay](../luke_lowpass_accessible_figures_v1/02_lighthouse_overlay_colorblind.png). Its 3σ baseline must not be confused with the 5σ sweep baseline.']
 (OUT/'README.md').write_text('\n'.join(index)+'\n')
 (OUT/'provenance.json').write_text(json.dumps({'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},'variants':names,'units':[int(u) for u in units],'per_unit_shared_ylim':limits,'render_only':True},indent=2))
 print(f'Created {len(names)} nine-unit pages and combined PDF in {OUT}')
if __name__=='__main__':main()
