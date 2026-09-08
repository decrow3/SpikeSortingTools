"""Read-only peak/field fingerprint audit; no estimation or voltage rerun."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'testing/outputs/luke_existing_motion_failure_audit_v1';P=Path('/mnt/NPX/Luke/20250804/dredge_pipeline_results_Luke0804_V2V1_g0_imec0/motion');fs=29999.835983263598
p=np.load(P/'peaks.npy',mmap_mode='r');l=np.load(P/'peak_locations.npy',mmap_mode='r');i,j=np.searchsorted(p['sample_index'],np.array([4160,4260])*fs);a=p[i:j];y=l['y'][i:j];amp=abs(a['amplitude']);t=a['sample_index']/fs;rows=[]
for ch in np.unique(a['channel_index']):
 k=a['channel_index']==ch;rows.append(dict(channel=int(ch),events=int(k.sum()),saved_amplitude_sum=float(amp[k].sum()),median_depth_um=float(np.median(y[k])),depth_IQR_um=float(np.ptp(np.quantile(y[k],[.25,.75]))),median_abs_amplitude=float(np.median(amp[k]))))
d=pd.DataFrame(rows).sort_values('saved_amplitude_sum',ascending=False);d.to_csv(OUT/'channel_support.csv',index=False)
mid=(y>=2000)&(y<2900);dominant=a['channel_index']==252;second=a['channel_index']==238
result=dict(interval_s=[4160,4260],events=len(a),midband_events=int(mid.sum()),midband_channels252_238_event_fraction=float(np.mean((dominant|second)[mid])),midband_channels252_238_amplitude_mass_fraction=float(amp[mid&(dominant|second)].sum()/amp[mid].sum()),interpretation='Amplitude mass is a diagnostic, not a reconstruction of estimator weighting. Channel identity does not establish neural versus artifact origin.')
(OUT/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result));print(d.head(8).to_string(index=False))
fig,axes=plt.subplots(1,3,figsize=(15,7),sharey=True);edges=np.arange(0,3841,10);tt=np.arange(4160,4261,2)
for ax,k,title in [(axes[0],np.ones(len(a),bool),'All saved peaks'),(axes[1],~(dominant|second),'Diagnostic view omitting channels 252 and 238')]:
 h,_,_=np.histogram2d(t[k],y[k],bins=[tt,edges]);im=ax.pcolormesh(tt,edges,np.log1p(h.T),shading='auto',vmin=0,vmax=7,cmap='magma');ax.set_title(title);ax.set_xlabel('Recording time (s)');fig.colorbar(im,ax=ax,label='log(1 + event count)',fraction=.04)
for name in ['dredge-motion','decentralized-motion','ks-motion']:
 q=P/name;dep=np.load(q/'depth_bins.npy').ravel();times=np.load(q/'time_bins.npy').ravel()-3057.677050340359;m=np.load(q/'motion.npy');k=(times>=4160)&(times<4260);v=np.ptp(m[k],axis=0);axes[2].plot(v,dep,label=name);pd.DataFrame(dict(depth_um=dep,epoch_range_um=v,full_recording_range_um=np.ptp(m,axis=0))).to_csv(OUT/(name+'_ranges.csv'),index=False)
axes[2].set_xlabel('Saved displacement range (µm)');axes[2].set_title('Depth-dependent estimator response');axes[2].legend(fontsize=8);axes[0].set_ylabel('Depth (µm)');axes[0].set_ylim(0,3820);fig.suptitle('Where do existing motion estimates lose sensitivity?\nSame historical imec0 peak cache; omission is a diagnostic visualization, not a corrected estimate');fig.tight_layout(rect=[0,0,1,.93]);fig.savefig(OUT/'01_peak_population_and_field.png',dpi=160);fig.savefig(OUT/'01_peak_population_and_field.pdf')
