"""MEDiCINe on exact cached early raster peaks; cached DREDGE comparison."""
import json, hashlib, time
from pathlib import Path
import numpy as np
import torch
import medicine
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'testing/outputs'
OUT=SRC/'luke_early_medicine_peak_comparison_v2'
PRE=SRC/'luke_long_context_validation_v1'
FS=29999.835983263598
ARGS=dict(time_bin_size=1.,num_depth_bins=2,time_kernel_width=1.,amplitude_threshold_quantile=0.,training_steps=10000,plot_figures=False)
def main():
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    hist=np.load(SRC/'luke_peak_population_review_v1/02_long_histograms_histograms.npz')
    manifest=dict(interval_s=[930,1030],medicine_args=ARGS,amplitudes='absolute detector amplitudes',seed=0,device='cuda' if torch.cuda.is_available() else 'cpu',resume='Completed arm outputs reused; interrupted training restarts that arm from scratch.',sources={},arms={})
    for i,arm in enumerate(['original','compensated']):
        pp=PRE/f'{arm}_peaks.npy'; lp=PRE/f'{arm}_locations.npy'
        p=np.load(pp); loc=np.load(lp); times=p['sample_index']/FS+930
        counts=np.histogram2d(loc['y'],times,bins=(hist['depth_edges'],hist['time_edges']))[0]
        assert np.array_equal(counts,hist[f'arm{i}_counts']), 'Raster source mismatch'
        for path in [pp,lp,PRE/f'{arm}_dredge_motion.npz']:
            manifest['sources'][str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        manifest['arms'][arm]=dict(peaks=len(p),raster_exact_match=True)
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    for arm in ['original','compensated']:
        dest=OUT/arm
        if not (dest/'complete.json').exists():
            if dest.exists(): raise RuntimeError('Incomplete arm preserved: inspect and choose fresh output before restart')
            p=np.load(PRE/f'{arm}_peaks.npy'); loc=np.load(PRE/f'{arm}_locations.npy')
            np.random.seed(0); torch.manual_seed(0)
            start=time.monotonic()
            medicine.run_medicine(peak_times=p['sample_index']/FS+930,peak_depths=loc['y'],peak_amplitudes=np.abs(p['amplitude']),output_dir=dest,**ARGS)
            d=np.load(dest/'motion.npy'); assert np.isfinite(d).all()
            (dest/'complete.json').write_text(json.dumps(dict(runtime_s=time.monotonic()-start,shape=list(d.shape))))
    v=np.quantile(np.concatenate([np.log1p(hist[f'arm{i}_counts']).ravel() for i in range(2)]),.995)
    fig,axes=plt.subplots(4,2,figsize=(15,13),sharex=True,layout='constrained')
    rows=[(0,3840,[320,960,1600,2240,2920,3380]),(1950,2350,[2100,2250]),(2849,2989,[2919]),(3311,3451,[3381])]
    summaries=[]
    for col,arm in enumerate(['original','compensated']):
        z=np.load(OUT/f'{arm}_dredge_motion.npz')
        methods=[('DREDGE fresh fit',z['time_s'],z['depth_um'],z['displacement_um'],'#56E0E8'),('MEDiCINe · 1 s kernel',np.load(OUT/arm/'time_bins.npy'),np.load(OUT/arm/'depth_bins.npy'),np.load(OUT/arm/'motion.npy'),'#FFFFFF')]
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
    fig.suptitle('Same detected peaks · fresh DREDGE / MEDiCINe with 1 s kernel\nCurves anchored by median displacement at 970–975 s; offsets are illustrative, not unit identities')
    for ext in ['png','pdf']: fig.savefig(OUT/f'01_peak_motion_overlay.{ext}',dpi=170)
    plt.close(fig)
    (OUT/'motion_spans.json').write_text(json.dumps(summaries,indent=2))
    manifest['status']='complete'
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
if __name__=='__main__': main()
