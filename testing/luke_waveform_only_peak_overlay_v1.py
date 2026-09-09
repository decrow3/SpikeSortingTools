"""Cached waveform-only cell traces over independently cached localized peaks."""
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages
from testing.luke_epoch_corroboration import ROOT,BASE
SRC=ROOT/'testing/outputs';IN=SRC/'luke_waveform_only_global_v1';OUT=SRC/'luke_waveform_only_peak_overlay_v1'
COLORS={161:'#0072B2',283:'#D55E00',527:'#009E73',555:'#CC79A7',673:'#8C6D31'}

def segments(e):
    q=e[e.status=='accepted'].sort_values('time_s');v=q[['time_s','centroid_um']].to_numpy();keep=(np.diff(q.time_s)<=2)&(np.diff(q.patch_base_um)==0)
    return np.stack([v[:-1][keep],v[1:][keep]],axis=1) if len(q)>1 else np.empty((0,2,2))

def main():
    OUT.mkdir(exist_ok=True);mfile=BASE/'recording/rescue_recording_manifest.json';fs=json.loads(mfile.read_text())['sampling_frequency_hz'];files=[mfile,IN/'training_qualified_candidates.csv',IN/'seed_distinctness.csv',IN/'events.csv',Path(__file__)]
    q=pd.read_csv(files[1]);ids=sorted(q[q.selected].unit_id.tolist());seeds=pd.read_csv(files[2]).set_index('unit_id');all_events=pd.read_csv(files[3]);e=all_events[all_events.unit_id.isin(ids)&all_events.status.isin(['accepted','identity_ambiguous'])].copy();e.to_csv(OUT/'overlay_events.csv',index=False)
    hist=np.load(SRC/'luke_peak_population_review_v1/02_long_histograms_histograms.npz');files.append(SRC/'luke_peak_population_review_v1/02_long_histograms_histograms.npz');background={};checks={}
    for i,arm in enumerate(['original','compensated']):
        pf=SRC/f'luke_long_context_validation_v1/{arm}_peaks.npy';lf=SRC/f'luke_long_context_validation_v1/{arm}_locations.npy';files.extend([pf,lf]);p=np.load(pf);loc=np.load(lf);t=930+p['sample_index']/fs;y=loc['y'];assert len(t)==len(y);assert ((t>=930)&(t<1030)).all()
        h=np.histogram2d(t,y,bins=[hist['time_edges'],hist['depth_edges']])[0].T;assert np.array_equal(h,hist[f'arm{i}_counts']);background[arm]=(t,y);checks[arm]=dict(peaks=len(t),matches_existing_histogram=True)
    config=dict(interval_s=[930,1030],units=ids,background='All cached localized peak events, no subsampling; independently detected original and compensated input arms',overlay='Identical original-input waveform matches on both arms; absolute energy centroid on translated observed support',lines='Display only: connect consecutive accepted observations of same identity only when same40um patch and time gap<=2s. No connection across patch changes or longer gaps; no interpolation, smoothing or new identity decisions.',markers='Filled colored circles: accepted; open orange x: identity ambiguous. All classes retained in source data.',training_s=[930,940],new_extraction=False,new_estimator=False)
    (OUT/'settings.json').write_text(json.dumps(config,indent=2));(OUT/'chart_contract.json').write_text(json.dumps(dict(question='Do waveform-only displacement hypotheses follow the peak depth/time structure?',renderer='Matplotlib rasterized scatter in standalone PDF/PNG',grain='Every cached localized peak and every matched cell event,930–1030s',palette='Gray background; five fixed unit colors in overview; blue accepted/orange ambiguous in individual panels',trace_policy=config['lines'],surfaces=['whole-probe overview','five-page per-cell PDF'],scales='Absolute depth in micrometers; identical original/compensated limits within page'),indent=2))
    def base(ax,arm,ylim):
        t,y=background[arm];keep=np.isfinite(y)&(y>=ylim[0])&(y<=ylim[1]);ax.scatter(t[keep],y[keep],s=.55,c='#606060',alpha=.23,linewidths=0,rasterized=True,zorder=1);ax.axvspan(930,940,color='#777777',alpha=.10,zorder=0);ax.set(xlim=(930,1030),ylim=ylim,xlabel='Recording time (s)',ylabel='Absolute depth (µm)');ax.ticklabel_format(axis='x',style='plain',useOffset=False)
    def overlay(ax,p,color,show_amb=True):
        a=p[p.status=='accepted'];b=p[p.status=='identity_ambiguous'];ax.add_collection(LineCollection(segments(p),colors=color,linewidths=1.1,zorder=3));ax.scatter(a.time_s,a.centroid_um,s=15,c=color,edgecolors='white',linewidths=.25,zorder=4,rasterized=True)
        if show_amb:ax.scatter(b.time_s,b.centroid_um,s=15,c='#D55E00',marker='x',linewidths=.6,alpha=.75,zorder=3,rasterized=True)
    handles=[Line2D([],[],color=COLORS[c],marker='o',ms=4,lw=1,label=f'Unit {c}') for c in ids]
    fig,axs=plt.subplots(1,2,figsize=(17,10),sharey=True,layout='constrained')
    for ax,arm in zip(axs,background):
        base(ax,arm,(0,3840))
        for cid in ids:overlay(ax,e[e.unit_id==cid],COLORS[cid],False)
        ax.set_title(f'{arm.capitalize()} peak background · {checks[arm]["peaks"]:,} peaks')
    axs[0].legend(handles=handles,loc='upper left',fontsize=9);fig.suptitle('Waveform-only lighthouse traces over localized peak scatters · 930–1,030 s',fontsize=17);fig.supxlabel('Gray: all cached peaks. Color: accepted waveform matches; absolute depth was excluded from identity scoring. Shading: seed interval.\nSegments show local observed support only; patch changes and gaps >2 s are disconnected. Identities and displacement remain provisional.',fontsize=10)
    for ext in ['pdf','png']:fig.savefig(OUT/f'01_whole_probe_overlay.{ext}',dpi=180)
    plt.close(fig)
    stats=[]
    with PdfPages(OUT/'02_individual_cell_overlays.pdf') as pdf:
        for cid in ids:
            p=e[e.unit_id==cid];a=p[p.status=='accepted'];b=p[p.status=='identity_ambiguous'];seed=float(seeds.loc[cid,'seed_centroid_um']);ys=np.r_[p.centroid_um,seed];ylim=(max(0,float(ys.min())-100),min(3840,float(ys.max())+100))
            if ylim[1]-ylim[0]<400:mid=np.mean(ylim);ylim=(max(0,mid-200),min(3840,mid+200))
            fig,axs=plt.subplots(1,2,figsize=(17,8),sharey=True,layout='constrained')
            for ax,arm in zip(axs,background):
                base(ax,arm,ylim);overlay(ax,p,'#0072B2');ax.axhline(seed,c='#0072B2',ls=':',lw=.8,alpha=.65);ax.set_title(f'{arm.capitalize()} peak background')
            axs[0].legend(handles=[Line2D([],[],color='#0072B2',marker='o',lw=1,label='Accepted waveform match'),Line2D([],[],color='#D55E00',marker='x',lw=0,label='Identity ambiguous'),Line2D([],[],color='#0072B2',ls=':',lw=.8,label='Seed centroid')],fontsize=9,loc='best')
            fig.suptitle(f'Unit {cid} · depth against peak population · 930–1,030 s\n{len(a)} accepted matches ({int((a.time_s>=940).sum())} held out) · {len(b)} ambiguous · seed centroid {seed:.1f} µm',fontsize=16)
            fig.supxlabel('Gray dots: cached localized peaks. Identical original-waveform observations on both backgrounds; compensated refers to background input.\nLines connect only consecutive accepted events within the same patch and ≤2 s. No connection across depth-patch changes; no smoothed trajectory. Shading:930–940 s.',fontsize=10)
            pdf.savefig(fig)
            for ext in ['pdf','png']:fig.savefig(OUT/f'unit_{cid}_overlay.{ext}',dpi=180)
            plt.close(fig);stats.append(dict(unit_id=cid,accepted=len(a),ambiguous=len(b),heldout=int((a.time_s>=940).sum()),trace_segments=len(segments(p)),seed_centroid_um=seed,ylim_um=ylim))
    (OUT/'README.md').write_text('''# Waveform-only cell traces over peak depth/time scatters\n\nOpen01_whole_probe_overlay.pdf for all five cells or02_individual_cell_overlays.pdf for one page per cell (161,283,527,555,673). Individual page PDFs and PNGs are also supplied.\n\nEvery cached localized peak is plotted, without subsampling. Left: original input peaks. Right: compensated input peaks. Waveform identities and energy centroids were measured on original referenced input; the identical observations appear on both backgrounds. This is a rendering of existing evidence, not a new tracking run. Peak localization and waveform energy centroid are different position summaries and need not coincide exactly.\n\nFilled blue points are accepted identity hypotheses; orange crosses retain ambiguous ones on individual pages. Dotted blue lines show seed centroids, and930–940s is the training interval. Trace segments connect consecutive accepted events only within the same40µm support patch and at most2s apart. Depth changes and longer gaps remain disconnected; this display rule is not an identity prior or a fitted trajectory. Whole-probe overview colors distinguish cells and omit ambiguous marks for readability; individual pages preserve them. Full vertical extents include ambiguous alternatives.\n\nBackground time origin was verified by exact reconstruction of the existing930–1030s peak-count histograms for both arms. No DREDGE or peak background information entered selection or matching. Apparent excursions can reflect true movement or waveform lookalikes; this figure supports direct inspection without deciding between them.\n''')
    (OUT/'manifest.json').write_text(json.dumps(dict(status='complete',checks=checks,unit_stats=stats,source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}),indent=2));print(json.dumps(stats,indent=2))
if __name__=='__main__':main()
