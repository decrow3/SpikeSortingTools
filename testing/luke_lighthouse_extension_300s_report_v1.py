"""Direct frozen-cell evidence over new200seconds; fit read only after matching."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from testing.luke_lighthouse_extension_300s_v1 import OUT,ROOT,COHORT,SRC,BASE,digest

def report():
    m=json.loads((BASE/'recording/rescue_recording_manifest.json').read_text());fs=m['sampling_frequency_hz'];c=pd.read_csv(COHORT/'candidate_audit.csv');c=c[c.selected].sort_values('seed_centroid_um');parts=[];counts=[]
    for start in range(1030,1230,5):
        for fn,h in json.loads((OUT/f'chunk_{start}.complete.json').read_text()).items():assert digest(OUT/fn)==h
        p=pd.read_csv(OUT/f'chunk_{start}.csv');counts.append(dict(start_s=start,detections=len(p),unmatched=int((p.status=='unmatched').sum())))
        q=p[p.unit_id.isin(c.unit_id)&(p.score>=.8)&p.gain.between(.35,3)].copy();q['evidence']=np.where(q.margin<.025,'identity_ambiguous',np.where(q.status=='accepted','strict_accepted','lower_score'));parts.append(q)
    new=pd.concat(parts,ignore_index=True);old=pd.read_csv(COHORT/'overlay_events.csv');events=pd.concat([old,new],ignore_index=True);events.to_csv(OUT/'overlay_events.csv',index=False);new.to_csv(OUT/'new_events.csv',index=False);pd.DataFrame(counts).to_csv(OUT/'detection_counts.csv',index=False)
    summary=[]
    for cell in c.itertuples():
        for cls in ['strict_accepted','lower_score','identity_ambiguous']:
            q=new[(new.unit_id==cell.unit_id)&(new.evidence==cls)];summary.append(dict(unit_id=cell.unit_id,evidence=cls,events=len(q),large_excursions=int((abs(q.relative_um)>=120).sum()),occupied_5s_bins=q.time_s.floordiv(5).nunique()))
    pd.DataFrame(summary).to_csv(OUT/'new_support.csv',index=False)
    # Estimator and screened-peak inputs are accessed only after identities are frozen.
    fit=ROOT/'testing/outputs/luke_screened_medicine_300s_v1';specs=[('5sigma_relaxed_10000','#0072B2'),('5sigma_relaxed_30000','#D55E00'),('6sigma_full_10000','#009E73')];fields={k:np.load(fit/f'fit_{k}/field.npz') for k,_ in specs};seeds=np.load(SRC/'templates.npz');lo=max(z['time_s'][0] for z in fields.values());hi=min(z['time_s'][-1] for z in fields.values())
    peak=np.load(fit/'input_5sigma_relaxed/peaks.npy');loc=np.load(fit/'input_5sigma_relaxed/locations.npy');pt=930+peak['sample_index']/fs;metrics=[];predictions=[]
    with PdfPages(OUT/'01_all17_depth_and_motion.pdf') as pdf:
        for cell in c.itertuples():
            fig,axs=plt.subplots(2,1,figsize=(16,10),layout='constrained');q=events[events.unit_id==cell.unit_id];low=min(q.centroid_um.min(),cell.seed_centroid_um)-80;high=max(q.centroid_um.max(),cell.seed_centroid_um)+80
            bg=(loc['y']>=low)&(loc['y']<=high);axs[0].scatter(pt[bg],loc['y'][bg],s=.3,c='#AAAAAA',alpha=.25,rasterized=True)
            for cls,marker,color in [('identity_ambiguous','x','#D55E00'),('lower_score','o','#0072B2'),('strict_accepted','o','#111111')]:
                a=q[q.evidence==cls];kw=dict(facecolors='none',edgecolors=color) if cls=='lower_score' else dict(c=color)
                axs[0].scatter(a.time_s,a.centroid_um,s=12,marker=marker,linewidths=.5,label=cls,**kw)
                axs[1].scatter(a.time_s,a.relative_um,s=12,marker=marker,linewidths=.5,**kw)
            for k,color in specs:
                z=fields[k];assert z['depth_um'][0]<=cell.seed_centroid_um<=z['depth_um'][-1]
                v=np.array([np.interp(cell.seed_centroid_um,z['depth_um'],row) for row in z['displacement_um']]);st=seeds[f'unit_{cell.unit_id}_seed_frames']/fs;st=st[(st>=lo)&(st<=hi)];assert len(st);v-=np.median(np.interp(st,z['time_s'],v));axs[1].plot(z['time_s'],v,c=color,lw=1,label=k)
                p=q[(q.time_s>=1030)&q.time_s.between(lo,hi)].copy();p['prediction_um']=np.interp(p.time_s,z['time_s'],v);p['difference_um']=p.relative_um-p.prediction_um;p['variant']=k;predictions.append(p)
                for cls in ['strict_accepted','lower_score','identity_ambiguous']:
                    for large in [False,True]:
                        a=p[p.evidence==cls];a=a[abs(a.relative_um)>=120] if large else a
                        metrics.append(dict(variant=k,unit_id=cell.unit_id,evidence=cls,large_excursions=large,events=len(a),median_abs_difference_um=float(np.median(abs(a.difference_um))) if len(a)>=5 else np.nan))
            for ax in axs:ax.axvline(1030,c='#777777',ls='--');ax.set_xlim(930,1230);ax.set_xlabel('Recording time (s)');ax.legend(fontsize=8,ncol=3)
            axs[0].set_ylim(low,high);axs[0].set_ylabel('Absolute waveform depth (µm)');axs[0].set_title('Gray:5σ/relaxed screened peaks. Waveforms measured independently on original referenced voltage.')
            axs[1].set_ylabel('Seed-relative displacement (µm)');fig.suptitle(f'Frozen unit {cell.unit_id} · new validation right of1030s · no identity reselection or offset fitting');pdf.savefig(fig)
            if cell.unit_id in [161,555,657,673]:fig.savefig(OUT/f'unit_{cell.unit_id}.png',dpi=130)
            plt.close(fig)
    pd.concat(predictions,ignore_index=True).to_csv(OUT/'new_event_predictions.csv',index=False);pd.DataFrame(metrics).to_csv(OUT/'new_differences.csv',index=False)
    (OUT/'summary.json').write_text(json.dumps(dict(status='complete',new_interval_s=[1030,1230],candidates=17,new_displayed_events=len(new),new_detections=sum(x['detections'] for x in counts),common_field_support_s=[lo,hi],identities_frozen=True,offsets_unchanged=True),indent=2))
    (OUT/'README.md').write_text('Frozen17 waveform-only candidate extension through1230s. Existing930–1030s events reused, only1030–1230s newly extracted. Whole-probe scoring against all247 frozen rivals, unchanged preprocessing/thresholds. Every detector event and rival score remains in chunk CSV/NPZ, including unmatched. No motion estimate enters extraction. Fields and screened-peak backgrounds loaded only after matching. No interpolation/smoothing of waveform observations; gaps and ambiguity remain. Figures show absolute waveform depths over screened peak scatters and unchanged seed-referenced MEDiCINe fields. New-window metrics separate evidence classes and large excursions; require5 events per summary. New interval is a temporal generalization check using the same provisional template inventory, not independent biological ground truth.\n')
    print('EXTENDED VALIDATION COMPLETE',len(new),flush=True)
if __name__=='__main__':report()
