"""Cached sensitivity/positive-control review; does not tune primary decisions."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from testing.luke_waveform_only_global_v1 import OUT, SRC, BASE, save

def near(t, reference, radius=.0003):
    if not len(reference):return np.zeros(len(t),bool)
    ref=np.sort(reference);i=np.searchsorted(ref,t);return np.minimum(abs(t-ref[np.clip(i,0,len(ref)-1)]),abs(t-ref[np.clip(i-1,0,len(ref)-1)]))<=radius

def main():
    assert json.loads((OUT/'summary.json').read_text())['status']=='complete'
    e=pd.read_csv(OUT/'events.csv');d=pd.read_csv(OUT/'seed_distinctness.csv');qualified=pd.read_csv(OUT/'training_qualified_candidates.csv');qualified_ids=set(qualified[qualified.selected].unit_id);d=d[d.selected|d.unit_id.isin(qualified_ids)];z=np.load(SRC/'templates.npz');fs=json.loads((BASE/'recording/rescue_recording_manifest.json').read_text())['sampling_frequency_hz'];rows=[]
    for c in d.itertuples():
        seed=z[f'unit_{c.unit_id}_seed_frames']/fs
        for threshold in [.70,.75,.80,.85,.86,.90]:
            q=e[(e.unit_id==c.unit_id)&(e.score>=threshold)&(e.margin>=.025)&e.gain.between(.35,3)].copy()
            train=q[q.time_s<940];held=q[q.time_s>=940]
            rows.append(dict(unit_id=c.unit_id,threshold=threshold,seed_events=len(seed),seed_recalled=int(near(seed,train.time_s.to_numpy()).sum()),training_matches=len(train),heldout_matches=len(held),heldout_beyond120=int((abs(held.relative_um)>120).sum())))
    s=pd.DataFrame(rows);s.to_csv(OUT/'threshold_sensitivity.csv',index=False)
    fig,axs=plt.subplots(int(np.ceil(len(d)/3)),3,figsize=(15,3*int(np.ceil(len(d)/3))),layout='constrained')
    for ax,c in zip(axs.flat,d.itertuples()):
        q=s[s.unit_id==c.unit_id];ax.plot(q.threshold,q.seed_recalled/q.seed_events,'o-',color='#0072B2');ax.axvline(.86,color='#777777',ls='--');ax.set(title=f'Unit {c.unit_id} · {int(q.seed_events.iloc[0])} cached seed spikes',ylim=(0,1),xlabel='Cosine threshold',ylabel='Seed-event recovery fraction')
    for ax in list(axs.flat)[len(d):]:ax.axis('off')
    fig.suptitle('Sensitivity check using cached seed spike times · 930–940 s\nSame global identity competition and gain/margin gates; temporal agreement ±0.3 ms; training control, not independent validation')
    save(fig,'03_seed_recovery_sensitivity')
    # No continuity prior: label simultaneous separated locations as contradictory identity evidence.
    a=e[(e.status=='accepted')&e.unit_id.isin(d.unit_id)].copy();a['simultaneous_separated_match']=False
    for cid,q in a.groupby('unit_id'):
        q=q.sort_values('time_s');t=q.time_s.to_numpy();b=q.patch_base_um.to_numpy();ix=q.index.to_numpy()
        for i in range(len(q)):
            j=i+1
            while j<len(q) and t[j]-t[i]<=.0008:
                if abs(b[j]-b[i])>80:a.loc[[ix[i],ix[j]],'simultaneous_separated_match']=True
                j+=1
    a.to_csv(OUT/'accepted_with_identity_conflict_audit.csv',index=False)
    # Report observed groups, not smooth trajectories; retain separated depths in one time bin.
    a['time_bin_s']=np.floor(a.time_s/5)*5;a['depth_patch_group_um']=a.patch_base_um
    groups=a.groupby(['unit_id','time_bin_s','depth_patch_group_um']).agg(events=('time_s','size'),median_relative_um=('relative_um','median'),min_time_s=('time_s','min'),max_time_s=('time_s','max'),median_score=('score','median'),conflicts=('simultaneous_separated_match','sum')).reset_index();groups.to_csv(OUT/'observed_support_groups.csv',index=False)
    primary=s[s.threshold==.86];result=dict(seed_recalled=int(primary.seed_recalled.sum()),seed_events=int(primary.seed_events.sum()),heldout_matches=int(primary.heldout_matches.sum()),heldout_beyond120=int(primary.heldout_beyond120.sum()),simultaneous_separated_matches=int(a.simultaneous_separated_match.sum()),five_second_same_patch_groups_at_least5=int(((groups.events>=5)&(groups.time_bin_s>=940)).sum()))
    result['training_qualified_units']=sorted(map(int,qualified_ids))
    result['qualified_heldout_matches']=int(((a.time_s>=940)&a.unit_id.isin(qualified_ids)).sum())
    result['qualified_heldout_beyond120']=int(((a.time_s>=940)&a.unit_id.isin(qualified_ids)&(abs(a.relative_um)>120)).sum())
    fig,axs=plt.subplots(3,2,figsize=(15,11),layout='constrained')
    for ax,cid in zip(axs.flat,sorted(qualified_ids)):
        q=e[e.unit_id==cid];g=q[q.status=='accepted'];amb=q[q.status=='identity_ambiguous'];ax.scatter(amb.time_s,amb.relative_um,s=9,marker='x',color='#D55E00',alpha=.35,label='Identity ambiguous');ax.scatter(g.time_s,g.relative_um,s=8,color='#0072B2',alpha=.6,label='Accepted');ax.axvspan(930,940,color='gray',alpha=.15);ax.axhline(0,color='gray',lw=.5);ax.set(title=f'Unit {cid} · {len(g)} accepted',xlim=(930,1030),xlabel='Time (s)',ylabel='Seed-relative centroid (µm)');ax.legend(fontsize=8)
    for ax in list(axs.flat)[len(qualified_ids):]:ax.axis('off')
    fig.suptitle('Seeds qualified using waveform distinctness and training recovery alone\nWhole-probe search · shaded930–940s training · no depth proximity or continuity prior · provisional identities')
    save(fig,'04_training_qualified_motion')
    (OUT/'review_summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
