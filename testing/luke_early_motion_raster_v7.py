from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
ROOT=Path(__file__).resolve().parents[1];SRC=ROOT/'testing/outputs';OUT=SRC/'luke_early_medicine_peak_comparison_v7'
def main():
    hist=np.load(SRC/'luke_peak_population_review_v1/02_long_histograms_histograms.npz')
    v=np.quantile(np.concatenate([np.log1p(hist[f'arm{i}_counts']).ravel() for i in range(2)]),.995)
    fig,axes=plt.subplots(4,2,figsize=(15,13),sharex=True,layout='constrained')
    rows=[(0,3840,[320,960,1600,2240,2920,3380]),(1950,2350,[2100,2250]),(2849,2989,[2919]),(3311,3451,[3381])]
    summaries=[]
    for col,arm in enumerate(['original','compensated']):
        z=np.load(OUT/f'{arm}_dredge_motion.npz')
        methods=[('DREDGE · 0.25 s Gaussian',z['time_s'],z['depth_um'],z['displacement_um'],'#56E0E8'),('MEDiCINe · 1 s kernel · 0.25 s time bins · 4 depth bins',np.load(SRC/'luke_early_medicine_peak_comparison_v6'/arm/'time_bins.npy'),np.load(SRC/'luke_early_medicine_peak_comparison_v6'/arm/'depth_bins.npy'),np.load(SRC/'luke_early_medicine_peak_comparison_v6'/arm/'motion.npy'),'#FFFFFF')]
        for row,(lo,hi,anchors) in enumerate(rows):
            ax=axes[row,col]
            ax.imshow(np.log1p(hist[f'arm{col}_counts']),origin='lower',aspect='auto',extent=[930,1030,0,3840],cmap='magma',vmin=0,vmax=v,rasterized=True)
            for name,t,y,d,color in methods:
                f=RegularGridInterpolator((t,y),d,bounds_error=False,fill_value=None)
                for anchor in anchors:
                    trace=f(np.column_stack([t,np.full(len(t),anchor)]))
                    trace-=np.median(trace[(t>=970)&(t<975)])
                    ax.plot(t,anchor+trace,color=color,lw=3.0 if name.startswith('DREDGE') else 1.6,label=name if anchor==anchors[0] else None)
                    summaries.append(dict(arm=arm,method=name,depth_um=anchor,peak_to_peak_um=float(np.ptp(trace))))
            ax.set(xlim=(930,1030),ylim=(lo,hi),title=f'{arm} · {lo}–{hi} µm',ylabel='Depth (µm)')
            if row==0: ax.legend(fontsize=8,facecolor='#222222',labelcolor='white',framealpha=.9)
        axes[-1,col].set_xlabel('Recording time (s)')
    fig.suptitle('DREDGE 0.25 s Gaussian / MEDiCINe 1 s kernel · same peaks · 0.25 s time bins\nCurves anchored by median displacement at 970–975 s; offsets are illustrative, not unit identities')
    for ext in ['png','pdf']: fig.savefig(OUT/f'01_peak_motion_overlay.{ext}',dpi=170)
    plt.close(fig)
    (OUT/'motion_spans.json').write_text(json.dumps(summaries,indent=2))

if __name__=='__main__':main()
