"""Compare saved Bacon KS4 fields with Fast MEDiCINe on common support."""
import json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'testing/outputs/cross_dataset_fast_motion_v2'
OUT=ROOT/'testing/outputs/bacon_saved_motion_crosscheck_20260914'

def metrics(v,dt):
    v=v-np.median(v,axis=0)
    rigid=np.median(v,axis=1)
    residual=v-rigid[:,None]
    excursion=lambda a:np.percentile(a,95,axis=0)-np.percentile(a,5,axis=0)
    spread=np.percentile(v,95,axis=1)-np.percentile(v,5,axis=1)
    return dict(rigid_excursion_um=float(excursion(rigid)),
        nonrigid_residual_excursion_um=float(np.median(excursion(residual))),
        median_depth_spread_um=float(np.median(spread)),p95_depth_spread_um=float(np.percentile(spread,95)),
        p99_local_speed_um_s=float(np.percentile(abs(np.diff(v,axis=0))/dt,99)))

def main():
    OUT.mkdir(exist_ok=False)
    cfg=json.loads((SOURCE/'config.json').read_text());rows=[];sources={};full=[]
    for probe in ['probeA','probeB']:
        path=Path('/mnt/NPX/Bacon halo_declan/20251016')/probe/'ops.npy'
        sources[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        ops=np.load(path,allow_pickle=True).item();ks=np.asarray(ops['dshift'],float);depth=np.asarray(ops['yblk'],float)
        dt=float(ops['batch_size']/ops['fs']);assert dt==2 and ops['tmin']==0
        assert ks.shape==(int(ops['Nbatches']),len(depth)) and np.isfinite(ks).all()
        start=np.arange(len(ks))*dt;stop=start+dt
        # Historical definitions: raw across-depth spread and median rigid, no per-depth centering.
        rigid=np.median(ks,axis=1);spread=np.percentile(ks,95,axis=1)-np.percentile(ks,5,axis=1)
        full.append(dict(probe=probe,method='saved_KS4_full_session',rigid_excursion_um=float(np.ptp(np.percentile(rigid,[5,95]))),
            median_depth_spread_um=float(np.median(spread)),p95_depth_spread_um=float(np.percentile(spread,95)),
            p99_rigid_step_um=float(np.percentile(abs(np.diff(rigid)),99)),time_step_s=dt,n_depth_bins=len(depth)))
        for w in [w for w in cfg['windows'] if w['dataset']=='Bacon' and w['probe']==probe]:
            p=SOURCE/w['id']/'fit/field.npz';sources[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
            seal=json.loads((p.parent/'complete.json').read_text());assert sources[str(p)]==seal['field.npz']
            f=np.load(p);t=f['session_time_s'];m=f['displacement_um'].astype(float);d=f['depth_um']
            keep=(start>=w['start_s'])&(stop<=w['stop_s'])
            # Require every KS batch to have all eight MEDiCINe samples, avoiding extrapolation.
            batches=[i for i in np.flatnonzero(keep) if ((t>=start[i])&(t<stop[i])).sum()==8]
            assert len(batches)>=58 and depth.min()>=d.min() and depth.max()<=d.max()
            med_on_depth=np.stack([np.interp(depth,d,row) for row in m])
            med_bins=np.stack([med_on_depth[(t>=start[i])&(t<stop[i])].mean(axis=0) for i in batches])
            base=dict(probe=probe,window=w['id'],start_s=float(start[batches[0]]),stop_s=float(stop[batches[-1]]),
                n_common_batches=len(batches),n_common_depths=len(depth))
            for method,v in [('KS4_common_2s_9depth',ks[batches]),('MEDiCINe_common_2s_9depth',med_bins)]:
                rows.append(dict(**base,method=method,**metrics(v,dt)))
            native=(t>=base['start_s'])&(t<base['stop_s'])
            rows.append(dict(**base,method='MEDiCINe_native_250ms_4depth',**metrics(m[native],.25)))
            np.savez_compressed(OUT/(w['id']+'.npz'),batch_start_s=start[batches],batch_stop_s=stop[batches],
                depth_um=depth,ks_um=ks[batches],medicine_2s_um=med_bins)
    frame=pd.DataFrame(rows);frame.to_csv(OUT/'matched_window_metrics.csv',index=False)
    pd.DataFrame(full).to_csv(OUT/'historical_full_session_metrics.csv',index=False)
    agg=frame.groupby(['probe','method']).median(numeric_only=True);agg.to_csv(OUT/'median_window_metrics.csv')
    fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
    for ax,key,title in zip(axes,['rigid_excursion_um','nonrigid_residual_excursion_um','p99_local_speed_um_s'],
        ['Rigid P95–P5 excursion (µm)','Nonrigid residual P95–P5 (µm)','P99 local speed at 2 s spacing (µm/s)']):
        for i,probe in enumerate(['probeA','probeB']):
            for method,offset,color in [('KS4_common_2s_9depth',-.1,'#0072B2'),('MEDiCINe_common_2s_9depth',.1,'#D55E00')]:
                v=frame[(frame.probe==probe)&(frame.method==method)][key]
                ax.scatter(i+offset+np.linspace(-.035,.035,len(v)),v,color=color,label=method.split('_')[0] if i==0 else None)
        ax.set_xticks([0,1],['Bacon A','Bacon B']);ax.set_ylabel(title);ax.set_ylim(bottom=0);ax.grid(axis='y',alpha=.2)
    axes[0].legend();fig.suptitle('Saved KS4 versus Fast MEDiCINe: identical 2 s batches and nine depth centers')
    fig.savefig(OUT/'comparison.png',dpi=160);plt.close(fig)
    (OUT/'provenance.json').write_text(json.dumps(dict(source_sha256=sources,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        time_mapping='KS4 batch i covers [i*batch_size/fs,(i+1)*batch_size/fs); tmin=0, dt=2 s. Only complete batches within pilot windows; average eight native MED samples/batch.',
        spatial_mapping='Linearly sample four-depth MED field at nine saved KS yblk centers, without extrapolation. Nine sampled values do not add spatial resolution.',
        metrics='Remove each depth temporal median; rigid=across-depth median; nonrigid=residual. Median across-depth temporal P95-P5 for residual excursion. Median across three windows for summaries.',
        limitation='Different frontends and estimator smoothing remain. This checks older saved estimates, not an independent validation of physical motion or exclusion of faster missed motion.'),indent=2))
    print(agg[['rigid_excursion_um','nonrigid_residual_excursion_um','p95_depth_spread_um','p99_local_speed_um_s']].round(2).to_string())
    print(frame[['window','method','rigid_excursion_um','nonrigid_residual_excursion_um']].round(2).to_string(index=False))

if __name__=='__main__':main()
