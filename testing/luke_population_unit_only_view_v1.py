"""Make existing per-unit waveform estimates legible; cached rendering only."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages
ROOT=Path(__file__).resolve().parents[1];SRC=ROOT/'testing/outputs';IN=SRC/'luke_population_consensus_v1';OUT=SRC/'luke_population_unit_only_view_v1'
def main():
 OUT.mkdir(exist_ok=True);files=[IN/'candidate_consensus_eligibility.csv',IN/'adaptive_windows.csv',SRC/'luke_population_depth_v2/events.csv'];c=pd.read_csv(files[0]).sort_values('depth_um');b=pd.read_csv(files[1]);events=pd.read_csv(files[2]);blue='#0072B2'
 with PdfPages(OUT/'all_unit_waveform_estimates.pdf') as pdf:
  for page in range(2):
   batch=c.iloc[page*15:(page+1)*15];fig,axes=plt.subplots(5,3,figsize=(15,13),sharex=True,sharey=True,layout='constrained')
   for ax,u in zip(axes.flat,batch.itertuples()):
    e=events[(events.unit_id==u.unit_id)&(events.status=='accepted')];allbins=b[b.unit_id==u.unit_id].sort_values('start_s');q=allbins[allbins.status=='supported']
    ax.scatter(e.time_s,e.centroid_um-u.template_centroid_um,c='#888888',s=4,alpha=.25,rasterized=True)
    if len(q):
     ax.errorbar(q.time_s,q.relative_um,xerr=np.array([q.time_s-q.start_s,q.end_s-q.time_s]),fmt='o',ms=4,c=blue,elinewidth=1,lw=0);ax.vlines(q.time_s,q.low_um,q.high_um,color=blue,lw=.8)
    else:ax.text(.5,.08,'No supported adaptive bins',ha='center',transform=ax.transAxes,fontsize=8)
    ax.axhline(0,c='gray',lw=.5);ax.axvspan(930,940,color='gray',alpha=.12);ax.set(title=f'{u.unit_id} · {u.depth_um:g} µm · {len(q)} bins'+(' · seed review*' if not u.consensus_eligible else ''),xlim=(930,1030),ylim=(-150,150));ax.grid(alpha=.12);ax.ticklabel_format(axis='x',style='plain',useOffset=False)
   for ax in axes[:,0]:ax.set_ylabel('Seed-relative centroid (µm)')
   for ax in axes[-1]:ax.set_xlabel('Recording time (s)')
   fig.suptitle(f'Per-unit waveform motion estimates · page {page+1}/2\nBlue: existing adaptive estimates and temporal spans · gray dots: individual accepted events',fontsize=14)
   fig.supxlabel('Vertical bars: existing conditional bootstrap intervals. Shading:930–940s template training. No DREDGE curves in this view.\nGaps remain gaps; * flags seed-quality review. These are provisional centroid measurements, not calibrated continuous physical trajectories.',fontsize=9)
   for ext in ['png','pdf']:fig.savefig(OUT/f'{page+1:02d}_unit_waveform_estimates.{ext}',dpi=160)
   pdf.savefig(fig);plt.close(fig)
 (OUT/'manifest.json').write_text(json.dumps(dict(render_only=True,new_tracking=False,new_estimation=False,source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files+[Path(__file__)]}),indent=2))
if __name__=='__main__':main()
