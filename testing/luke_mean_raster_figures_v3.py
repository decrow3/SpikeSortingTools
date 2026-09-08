"""Lightweight saved-array renderer and accepted-frame sampling; no fitting or voltage."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]; SRC=ROOT/'testing/outputs'; OUT=SRC/'luke_mean_raster_diagnostic_v3'
def save(fig,name):
    for ext in ['png','pdf']:fig.savefig(OUT/f'{name}.{ext}',dpi=140)
    plt.close(fig)
def main():
    predictions=pd.read_csv(OUT/'event_matched_predictions.csv')
    tracks=pd.read_csv(SRC/'luke_lighthouse_gentle_v1/gentle_tracks.csv')
    fig,axes=plt.subplots(3,3,figsize=(13,10),sharex=True,layout='constrained')
    for ax,(unit,g) in zip(axes.flat,tracks.groupby('unit_id')):
        for name,color,style in [('amplitude_sum','#2878b5','-'),('amplitude_mean','#b13775','--')]:
            q=predictions[(predictions.name==name)&(predictions.unit_id==unit)]
            ax.plot(q.time_s,q.predicted_um,style,color=color,label=name.replace('_',' '))
        g=g.sort_values('time_s');base=g.iloc[0].median_waveform_centroid_um; good=g.accepted_events>=10
        ax.errorbar(g.time_s[good],g.median_waveform_centroid_um[good]-base,yerr=np.array([g.median_waveform_centroid_um[good]-g.bootstrap_low_um[good],g.bootstrap_high_um[good]-g.median_waveform_centroid_um[good]]),fmt='ko',ms=3,capsize=2,label='Provisional lighthouse')
        ax.axvspan(4245,4250,color='#aaaaaa',alpha=.18)
        ax.set(title=f'Unit {unit} · {g.depth_um.iloc[0]:.0f} µm',xlabel='Recording time (s)',ylabel='Relative displacement (µm)')
        ax.tick_params(axis='x',labelbottom=True,labelsize=8)
    axes.flat[0].legend(fontsize=7)
    fig.suptitle('Same AP events: mean amplitude helps some shallow tracks but worsens deeper agreement\nAccepted-event sampling; conditional lighthouse uncertainty is not motion accuracy',fontsize=12)
    save(fig,'02_all_unit_overlay')
    fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    for row,name in enumerate(['amplitude_sum','amplitude_mean']):
        z=np.load(OUT/f'{name}_raster.npz');r=z['raster'];t=z['time_edges_s']+4160;depth=z['depth_edges_um']
        for col,(lo,hi) in enumerate([(0,3820),(0,900)]):
            ids=(depth[:-1]>=lo)&(depth[:-1]<hi);v=np.log1p(r[ids]);hi_color=float(np.quantile(np.log1p(r),.995))
            ax=axes[row,col];im=ax.imshow(v,origin='lower',aspect='auto',extent=[t[0],t[-1],depth[:-1][ids][0],depth[1:][ids][-1]],vmin=0,vmax=hi_color,cmap='magma')
            fig.colorbar(im,ax=ax,label='log(1 + sum |amplitude|)' if row==0 else 'log(1 + mean |amplitude|)')
            ax.set(title=f'{name.replace("_"," ")} · {"full depth" if col==0 else "shallow detail"}',xlabel='Recording time (s)',ylabel='Localized depth (µm)')
    fig.suptitle('Actual DREDGE input rasters after the same 1 µm / 1 s smoothing\nSeparate scales reflect different units; row scales fixed across full/shallow views, clipped at row 99.5th percentile',fontsize=11)
    save(fig,'03_actual_rasters')
    accepted=pd.read_csv(SRC/'luke_shallow_transition_audit_v3/accepted_events.csv');rows=[]
    for name in ['amplitude_sum','amplitude_mean']:
        z=np.load(OUT/'fields'/f'{name}.npz');v=np.array([np.interp(620,z['depth_um'],x) for x in z['displacement_um']]);pred=np.interp(accepted.time_s,z['time_s'],v)
        baseline=np.median(pred[(accepted.time_s>=4240)&(accepted.time_s<4245)])
        for start in [4240,4245,4250,4255]:
            ix=(accepted.time_s>=start)&(accepted.time_s<start+5);n=int(ix.sum());rows.append(dict(name=name,start_s=start,unit_id=154,events=n,supported=n>=10,field_relative_um=float(np.median(pred[ix])-baseline) if n>=10 else np.nan))
    pd.DataFrame(rows).to_csv(OUT/'qualified_unit154_sampling.csv',index=False)
    print(pd.DataFrame(rows).to_string(index=False))
if __name__=='__main__':main()
