"""Brief early-interval comparison from completed saved fields/masks only."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];SRC=ROOT/'testing/outputs';RUN=SRC/'luke_early_screen_shortlist_v1';OUT=SRC/'luke_early_screen_report_v1'
ARMS=['compensated','center_only','relaxed_combination','without_snr','full_screen'];LABELS=['Unscreened baseline','Center-energy only','Relaxed combination','Without amplitude gate','Full screen'];COLORS=['#0072B2','#D55E00','#882255','#666666','#009E73']
def save(fig,name):
 for ext in ['png','pdf']:fig.savefig(OUT/f'{name}.{ext}',dpi=150,bbox_inches='tight')
 plt.close(fig)
def main():
 OUT.mkdir(exist_ok=False)
 from testing.luke_epoch_corroboration import BASE
 fs=json.loads((BASE/'recording/rescue_recording_manifest.json').read_text())['sampling_frequency_hz']
 tracks=pd.read_csv(SRC/'luke_transfer_template_holdout_v1/tracks.csv');tracks=tracks[(tracks.time_s>=930)&(tracks.time_s<1030)]
 events=pd.read_csv(SRC/'luke_transfer_template_holdout_v1/matched_events.csv');rows=[]
 fig,axes=plt.subplots(1,2,figsize=(14,5));handles=[]
 for ax,(candidate,g) in zip(axes,tracks.groupby('candidate')):
  g=g.sort_values('time_s');valid=(g.events>=10)&np.isfinite(g.centroid_um);anchor=g[valid].iloc[0];depth=anchor.depth_um;ev=events[events.candidate==candidate].frame.to_numpy()/fs
  for name,label,color in zip(ARMS,LABELS,COLORS):
   f=np.load(RUN/'fields'/f'{name}.npz');v=np.array([np.interp(depth,f['depth_um'],d) for d in f['displacement_um']]);ref=ev[(ev>=anchor.time_s-2.5)&(ev<anchor.time_s+2.5)];offset=np.median(np.interp(ref,f['time_s'],v));vals=[]
   for r in g.itertuples():
    use=ev[(ev>=r.time_s-2.5)&(ev<r.time_s+2.5)];pred=float(np.median(np.interp(use,f['time_s'],v))-offset) if len(use)>=10 else np.nan;obs=float(r.centroid_um-anchor.centroid_um) if r.events>=10 else np.nan;vals.append(pred);rows.append(dict(name=name,candidate=candidate,depth_um=depth,time_s=r.time_s,events=len(use),predicted_um=pred,reference_um=obs,absolute_disagreement_um=abs(pred-obs)))
   h,=ax.plot(g.time_s,vals,color=color,ls='-' if name=='compensated' else '--',marker='o',ms=4,label=label)
   if len(handles)<5:handles.append(h)
  ax.plot(g.time_s,np.where(valid,g.centroid_um-anchor.centroid_um,np.nan),'kD',mfc='white',ms=6,label='Provisional footprint');ax.set(title=f'{candidate} · {depth:.0f} µm',xlabel='Recording time (s)',ylabel='Relative displacement (µm)');ax.grid(alpha=.15)
 fig.suptitle('930–1,030 s screening shortlist · cached references cover only 970–990 s\nMatched at the same accepted event times; candidate identities remain unqualified',fontsize=13)
 fig.subplots_adjust(bottom=.23,top=.79,wspace=.24);fig.legend(handles=handles,loc='lower center',ncols=3,frameon=False);save(fig,'01_cached_reference_overlay')
 table=pd.DataFrame(rows);table.to_csv(OUT/'event_matched_predictions.csv',index=False);table[table.time_s!=table.groupby(['name','candidate']).time_s.transform('min')].groupby(['name','candidate']).absolute_disagreement_um.mean().unstack().reindex(ARMS).to_csv(OUT/'descriptive_disagreement.csv')
 peaks=np.load(SRC/'luke_long_context_validation_v1/compensated_peaks.npy');y=np.load(SRC/'luke_long_context_validation_v1/compensated_locations.npy')['y'];te=np.arange(930,1030+.25,.25);de=np.arange(0,3841,10);hist=[]
 for name in ARMS:
  k=np.load(RUN/'masks'/f'{name}.npy');hist.append([np.histogram2d(930+peaks['sample_index'][k]/fs,y[k],bins=[te,de],weights=None if col==0 else abs(peaks['amplitude'][k]))[0].T for col in range(2)])
 limits=[np.quantile(np.concatenate([np.log1p(h[col]).ravel() for h in hist]),.995) for col in range(2)]
 fig,axes=plt.subplots(5,2,figsize=(13,14),sharex=True,sharey=True,layout='constrained')
 for i,(label,h) in enumerate(zip(LABELS,hist)):
  for col in range(2):
   ax=axes[i,col];im=ax.imshow(np.log1p(h[col]),origin='lower',aspect='auto',extent=[930,1030,0,3840],cmap='magma',vmin=0,vmax=limits[col]);ax.set(title=label+' · '+('count' if col==0 else 'amplitude sum'),ylabel='Localized depth (µm)',xlabel='Recording time (s)');fig.colorbar(im,ax=ax,label='log(1 + bin total)')
 fig.suptitle('Early screening shortlist · 0.25 s × 10 µm bins\nShared 99.5th-percentile color limits; same cached 5σ detections and locations',fontsize=14);save(fig,'02_depth_time')
 (OUT/'summary.json').write_text(json.dumps(dict(status='complete',scope='Five prespecified arms; only two unqualified deep footprint references for20s; no winning physical accuracy claim',source_settings_sha256=hashlib.sha256((RUN/'settings.json').read_bytes()).hexdigest()),indent=2))
if __name__=='__main__':main()
