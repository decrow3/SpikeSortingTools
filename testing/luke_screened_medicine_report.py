"""Direct longer-snippet coverage, field and frozen-lighthouse checks."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from testing.luke_early_sigma_screen250_report_v1 import references,observations

def report(out):
    out=Path(out);cfg=json.loads((out/'settings.json').read_text());fs,c,e,seeds=references()
    specs=[('5sigma_relaxed_10000','5σ relaxed · 10k','#0072B2'),('5sigma_relaxed_30000','5σ relaxed · 30k','#D55E00'),('6sigma_full_10000','6σ full · 10k','#009E73')]
    fields={k:np.load(out/f'fit_{k}/field.npz') for k,_,_ in specs};rows=[]
    with PdfPages(out/'01_depth_time_and_fields.pdf') as pdf:
        for k,label,color in specs:
            z=fields[k];variant=k.rsplit('_',1)[0];p=np.load(out/f'input_{variant}/peaks.npy');loc=np.load(out/f'input_{variant}/locations.npy');t=p['sample_index']/fs+cfg['start_s']
            fig,axs=plt.subplots(2,1,figsize=(16,9),layout='constrained');h,te,de=np.histogram2d(t,loc['y'],bins=[np.arange(cfg['start_s'],cfg['stop_s']+.25,.25),np.arange(-200,4101,10)])
            axs[0].pcolormesh(te,de,np.log1p(h.T),cmap='Greys',rasterized=True);axs[0].set(ylabel='Peak depth (µm)',title='Screened peak counts · log(1 + count), 0.25 s × 10 µm')
            for j,y in enumerate(z['depth_um']):axs[1].plot(z['time_s'],z['displacement_um'][:,j],label=f'{y:.0f} µm')
            axs[1].set(xlabel='Recording time (s)',ylabel='Displacement (µm)',title='Native field values; no lighthouse alignment');axs[1].legend(ncol=4)
            fig.suptitle(f'{label} · {cfg["start_s"]}–{cfg["stop_s"]} s');pdf.savefig(fig);fig.savefig(out/f'{k}.png',dpi=130);plt.close(fig)
            audit=json.loads((out/f'fit_{k}/fit_audit.json').read_text());rows.append(dict(variant=k,peaks=len(p),fit_seconds=audit['seconds'],max_rss_kib=audit['max_rss_kib'],cuda_max_allocated_bytes=audit['cuda_max_allocated_bytes'],median_depth_ptp_um=float(np.median(np.ptp(z['displacement_um'],axis=0)))))
    pd.DataFrame(rows).to_csv(out/'benchmark.csv',index=False)
    metrics=[];lo=max(z['time_s'][0] for z in fields.values());hi=min(z['time_s'][-1] for z in fields.values())
    with PdfPages(out/'02_lighthouse_overlap_only.pdf') as pdf:
        for cell in c.itertuples():
            fig,ax=plt.subplots(figsize=(14,5),layout='constrained');observations(ax,e[e.unit_id==cell.unit_id])
            for k,label,color in specs:
                z=fields[k];assert z['depth_um'][0]<=cell.seed_centroid_um<=z['depth_um'][-1]
                v=np.array([np.interp(cell.seed_centroid_um,z['depth_um'],x) for x in z['displacement_um']]);st=seeds[f'unit_{cell.unit_id}_seed_frames']/fs;st=st[(st>=lo)&(st<=hi)];assert len(st)>0
                v-=np.median(np.interp(st,z['time_s'],v));ax.plot(z['time_s'],v,color=color,label=label,lw=1)
                q=e[(e.unit_id==cell.unit_id)&(e.evidence=='strict_accepted')&(e.time_s>=940)&e.time_s.between(lo,hi)]
                for large in [False,True]:
                    s=q[abs(q.relative_um)>=120] if large else q;err=s.relative_um-np.interp(s.time_s,z['time_s'],v)
                    metrics.append(dict(variant=k,unit_id=cell.unit_id,large_excursions=large,events=len(s),median_abs_difference_um=float(np.median(abs(err))) if len(s)>=5 else np.nan))
            ax.legend();fig.suptitle(f'Unit {cell.unit_id} · longer fit, existing 930–1030s waveform evidence only');pdf.savefig(fig);plt.close(fig)
    pd.DataFrame(metrics).to_csv(out/'overlap_strict_differences.csv',index=False)
    (out/'README.md').write_text('Benchmark of screened MEDiCINe on a longer snippet. Fixed seed0;10000 steps and primary30000-step sensitivity control. Same model, precision, kernels and offsets. Native field plots and screened peak histograms cover the full snippet. Frozen lighthouse evidence exists only930–1030s; no claim of later identity validation. The first100s model reproduction is in reproduction.json. Higher-budget agreement or field amplitude is not accuracy. Complete chunk and fit seals support stage reuse; no within-fit optimizer resume.\n')
