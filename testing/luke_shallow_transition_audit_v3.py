"""Frozen qualified unit154 transition audit; matching never receives motion fields."""
import json
import time
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from testing.luke_shallow_identity_audit_v3 import detect, score, adjudicate, near, SHIFTS
from testing.luke_lowpass_waveform_preservation_v2 import reconstruct, digest, SRC, BASE

BANK = SRC/'luke_shallow_identity_audit_v3'
OUT = SRC/'luke_shallow_transition_audit_v3'
TARGET, DEPTH, START, STOP = 154, 620., 4240, 4260


def summarize(events, width):
    rows=[]
    for lo in np.arange(START,STOP,width):
        g=events[(events.time_s>=lo)&(events.time_s<lo+width)]
        unique=g[g.shift_unique]
        dominance=float(unique.best_shift_um.value_counts(normalize=True).max()) if len(unique) else 0.
        enough=len(unique)>=10 and dominance>=.8
        counts={f'shift_{shift}':int((unique.best_shift_um==shift).sum()) for shift in SHIFTS}
        rows.append(dict(start_s=lo,stop_s=lo+width,time_s=lo+width/2,
                         identity_events=len(g),unique_shift_events=len(unique),enough_support=enough,
                         existing_fixed_matches=int(g.near_existing_fixed.sum()),newly_recovered_events=int((~g.near_existing_fixed).sum()),
                         event_centroid_median_um=float(unique.centroid_um.median()) if enough else np.nan,
                         distinct_shifts=int(unique.best_shift_um.nunique()),dominant_shift_fraction=dominance,status='supported_single_population' if enough else 'mixed_hypotheses' if len(unique)>=10 else 'insufficient_support',
                         **counts))
    return pd.DataFrame(rows)


def main():
    begun=time.monotonic()
    qualification=pd.read_csv(BANK/'qualification.csv')
    q=qualification[qualification.target==TARGET]
    assert len(q)==1 and q.iloc[0].status=='eligible_for_bounded_transition_audit', 'Unit154 is not qualified'
    OUT.mkdir(exist_ok=False)
    bankfile=BANK/'u154_bank.npz'; infofile=BANK/'u154_hypotheses.csv'
    manifestfile=BASE/'recording/rescue_recording_manifest.json'
    modelfile=SRC/'luke_common_event_screen_v1/shared_response_model.npz'
    priorfile=SRC/'luke_lighthouse_gentle_v1/gentle_events.csv'
    m=json.loads(manifestfile.read_text());fs=m['sampling_frequency_hz'];geo=np.asarray(m['channel_locations_um'])
    bank=np.load(bankfile);hypotheses=bank['training'];channels=bank['channels'];weights=bank['common_weights']
    info=pd.read_csv(infofile).to_dict('records')
    settings=dict(target=TARGET,interval_s=[START,STOP],detection='Same5sigma both-sign independent detector and depth±120um as quiet qualification; no label-seeded candidates.',
                  matching='Frozen bank, commonweights, cosine>=.8,gain.4–2.5,other-identity margin>=.03; shiftunique additionally requires other-shift margin>=.03.',
                  centroid='Raw same-event squared-energy centroid within original±120um source footprint translated by winning exact40um shift; descriptive, background biased, not monopolar localization.',
                  summaries='1s counts;5s centroid medians only with>=10unique-shift events and dominant shiftfraction>=.8; otherwise mixed-hypothesis gap. Discrete shift populations kept; no interpolation across gaps or claim of calibrated fractional motion.',
                  field_comparison='Only after matching tables/matrices saved. Existing broad3sigma and lowpassadjusted at acceptedunique-shift frames; ownfirst5s baseline requires>=10events; no sign/gain/lag fitting.',
                  limitations=['Unit80 remains failed sensitivity qualification and is excluded.','Quiet/injection qualification does not prove identity in the transition.','40um grid does not exclude or measure smaller shifts; mixed winning shifts remain ambiguity.','Newly recovered means unmatched to prior fixed matcher within0.5ms; not newly established neural events.'],
                  sha256={str(p):digest(p) for p in [Path(__file__).resolve(),BANK/'qualification.csv',BANK/'settings.json',bankfile,infofile,manifestfile,modelfile,priorfile,Path(__file__).with_name('luke_shallow_identity_audit_v3.py'),Path(__file__).with_name('luke_lowpass_waveform_preservation_v2.py')]})
    (OUT/'settings.json').write_text(json.dumps(settings,indent=2))
    broad,lowpass,first=reconstruct(START,m,np.load(modelfile));del lowpass
    noise=np.full(384,np.inf);noise[channels]=bank['noise_uv']
    detected=detect(broad,np.flatnonzero(abs(geo[:,1]-DEPTH)<=120),noise,fs)
    blocks=[];gainblocks=[];lagblocks=[]
    for ev in np.array_split(detected,max(1,int(np.ceil(len(detected)/100)))):
        waves=broad[ev[:,None,None]+np.arange(-33,34)[None,:,None],channels[None,None,:]]
        sc,ga,la=score(waves,hypotheses,weights);blocks.append(sc);gainblocks.append(ga);lagblocks.append(la)
    scores=np.concatenate(blocks);gains=np.concatenate(gainblocks);lags=np.concatenate(lagblocks)
    accepted,col,top,margin,shiftmargin=adjudicate(scores,info,TARGET)
    rivalcols=np.flatnonzero([r['identity']!=f'label_{TARGET}' for r in info])
    rivalwin=rivalcols[np.argmax(scores[:,rivalcols],axis=1)]
    frames=detected+lags[np.arange(len(detected)),col]+first
    prior=pd.read_csv(priorfile);existing=prior[(prior.unit_id==TARGET)&(prior.frame>=first)&(prior.frame<first+len(broad))].frame.to_numpy(dtype='int64')
    decisions=pd.DataFrame(dict(frame=frames,time_s=frames/fs,detected_frame=detected+first,identity_accepted=accepted,
        shift_unique=shiftmargin>=.03,best_shift_um=[info[c]['shift_um'] for c in col],score=top,identity_margin=margin,shift_margin=shiftmargin,
        gain=gains[np.arange(len(detected)),col],hypothesis_index=col,rival_hypothesis_index=rivalwin,
        rival_identity=[info[c]['identity'] for c in rivalwin],rival_score=scores[np.arange(len(detected)),rivalwin],
        near_existing_fixed=near(frames,existing,.0005*fs)))
    decisions.to_csv(OUT/'all_detection_decisions.csv',index=False)
    np.savez_compressed(OUT/'all_detection_scores.npz',detected_frames=detected+first,scores=scores,gains=gains,timing_lags=lags)
    events=decisions[decisions.identity_accepted].sort_values('score',ascending=False).drop_duplicates('frame').sort_values('frame').copy()
    centroids=[];templates={};galleries=[]
    for row in events.itertuples():
        footprint=np.flatnonzero(abs(geo[:,1]-(DEPTH+row.best_shift_um))<=120)
        w=broad[int(row.frame)-first+np.arange(-30,31)[:,None],footprint[None,:]]
        energy=np.sum(w*w,axis=0);centroids.append(float(energy@geo[footprint,1]/np.maximum(energy.sum(),1e-20)))
    events['centroid_um']=centroids
    events['preceding_accepted_interval_ms']=np.r_[np.nan,np.diff(events.frame.to_numpy())/fs*1000] if len(events) else np.array([])
    events['at_shift_boundary']=abs(events.best_shift_um)==80
    events.to_csv(OUT/'accepted_events.csv',index=False)
    for width in [1,5]:
        table=summarize(events,width)
        for ix,row in table.iterrows():
            d=decisions[(decisions.time_s>=row.start_s)&(decisions.time_s<row.stop_s)]
            g=events[(events.time_s>=row.start_s)&(events.time_s<row.stop_s)]
            table.loc[ix,'all_detections']=len(d)
            table.loc[ix,'rejected_detections']=int((~d.identity_accepted).sum())
            table.loc[ix,'identity_ambiguous_detections']=int(((d.score>=.8)&(d.identity_margin<.03)).sum())
            table.loc[ix,'shift_ambiguous_identity_events']=int((~g.shift_unique).sum())
            table.loc[ix,'boundary_identity_events']=int(g.at_shift_boundary.sum())
            table.loc[ix,'boundary_unique_events']=int((g.at_shift_boundary&g.shift_unique).sum())
            table.loc[ix,'accepted_intervals_lt1ms']=int((g.preceding_accepted_interval_ms<1).sum())
        table.to_csv(OUT/f'support_{width}s.csv',index=False)
    (OUT/'matching_complete.json').write_text(json.dumps(dict(status='matching_complete_before_motion_read',identity_events=len(events),unique_shift_events=int(events.shift_unique.sum()),seconds=time.monotonic()-begun),indent=2))
    fig,axes=plt.subplots(4,5,figsize=(18,12),sharex=True,layout='constrained')
    for i,lo in enumerate(range(START,STOP,5)):
        for j,shift in enumerate(SHIFTS):
            ax=axes[i,j];g=events[(events.time_s>=lo)&(events.time_s<lo+5)&events.shift_unique&(events.best_shift_um==shift)]
            if not len(g):ax.text(.5,.5,'No unique-shift matches',ha='center',va='center',transform=ax.transAxes,fontsize=8);ax.set_title(f'{lo}–{lo+5}s · {shift:+d}µm');continue
            frames=g.frame.to_numpy(dtype='int64');frames=frames[np.linspace(0,len(frames)-1,min(100,len(frames)),dtype=int)]
            wave=np.median(broad[frames[:,None,None]-first+np.arange(-30,31)[None,:,None],channels[None,None,:]],axis=0)
            h=int(g.hypothesis_index.iloc[0]);reference=hypotheses[h];ch=int(np.argmax(np.max(abs(reference),axis=0)))
            ax.plot(np.arange(-30,31)/fs*1000,wave[:,ch],color='#2468a2',label='Observed median')
            ax.plot(np.arange(-30,31)/fs*1000,reference[:,ch]*g.gain.median(),color='#cc7722',ls='--',label='Frozen target × median gain')
            ax.set_title(f'{lo}–{lo+5}s · {shift:+d}µm · n={len(g)}\nscore={g.score.median():.2f}, rival={g.rival_score.median():.2f}',fontsize=9)
            ax.set_xlabel('Time (ms)');ax.set_ylabel('µV');key=f't{lo}_shift{shift}';templates[key]=wave
            galleries.append(dict(start_s=lo,shift_um=shift,events=len(g),channel=int(channels[ch]),median_score=g.score.median(),median_rival_score=g.rival_score.median()))
    handles,labels=next((ax.get_legend_handles_labels() for ax in axes.ravel() if ax.get_legend_handles_labels()[0]),([],[]))
    fig.legend(handles,labels,loc='outside lower center',ncol=2)
    fig.suptitle('Unit154: observed waveform by time and independently winning discrete shift\nNo motion-field-guided selection; median fitted gain comes from frozen template matching, not motion fitting')
    for ext in ['png','pdf']:fig.savefig(OUT/f'01_time_shift_waveforms.{ext}',dpi=150)
    pd.DataFrame(galleries).to_csv(OUT/'waveform_gallery.csv',index=False);np.savez_compressed(OUT/'observed_templates.npz',channels=channels,**templates)
    del broad
    # Motion files are first opened here, after immutable selection artifacts exist.
    unique=events[events.shift_unique];baseline=unique[unique.time_s<START+5]
    baseline_dominance=float(baseline.best_shift_um.value_counts(normalize=True).max()) if len(baseline) else 0.
    baseline_supported=len(baseline)>=10 and baseline_dominance>=.8
    comparisons=[];field_rows=[]
    for arm in ['broad_3sigma','lowpass_adjusted']:
        path=SRC/'luke_3sigma_lowpass_screen_v2/fields'/f'{arm}.npz'
        field=np.load(path);local=np.array([np.interp(DEPTH,field['depth_um'],row) for row in field['displacement_um']])
        pred=np.interp(unique.time_s,field['time_s'],local)
        offset=float(np.median(pred[unique.time_s.to_numpy()<START+5])) if baseline_supported else np.nan
        for frame,t,value in zip(unique.frame,unique.time_s,pred):field_rows.append(dict(arm=arm,frame=frame,time_s=t,prediction_um=value,prediction_relative_um=value-offset))
        for lo in range(START,STOP,5):
            mask=(unique.time_s>=lo)&(unique.time_s<lo+5);g=unique[mask];dominance=float(g.best_shift_um.value_counts(normalize=True).max()) if len(g) else 0.;enough=len(g)>=10 and dominance>=.8 and baseline_supported
            comparisons.append(dict(arm=arm,start_s=lo,time_s=lo+2.5,events=len(g),supported=enough,dominant_shift_fraction=dominance,baseline_supported=baseline_supported,
                field_relative_um=float(np.median(pred[mask])-offset) if enough else np.nan,
                event_centroid_relative_um=float(g.centroid_um.median()-baseline.centroid_um.median()) if enough else np.nan,
                field_sha256=digest(path)))
    pd.DataFrame(field_rows).to_csv(OUT/'event_matched_motion_predictions.csv',index=False)
    comparison=pd.DataFrame(comparisons);comparison.to_csv(OUT/'motion_comparison_5s.csv',index=False)
    fig,axes=plt.subplots(4,1,figsize=(13,12),sharex=True,layout='constrained')
    axes[0].scatter(events.time_s,events.best_shift_um,c=np.where(events.shift_unique,'#2468a2','#bbbbbb'),s=8);axes[0].set_ylabel('Best discrete shift (µm)');axes[0].set_yticks(SHIFTS)
    axes[1].scatter(events.time_s,events.centroid_um,c=np.where(events.shift_unique,'#2468a2','#bbbbbb'),s=8);axes[1].set_ylabel('Event energy centroid (µm)')
    one=summarize(events,1);axes[2].bar(one.time_s,one.identity_events,width=.8,color='#bbbbbb',label='Identity accepted');axes[2].bar(one.time_s,one.unique_shift_events,width=.5,color='#2468a2',label='Unique shift');axes[2].set_ylabel('Events / second');axes[2].legend()
    for arm,color in [('broad_3sigma','#2468a2'),('lowpass_adjusted','#cc7722')]:
        g=comparison[comparison.arm==arm];axes[3].plot(g.time_s,g.field_relative_um,'o',color=color,label=arm)
    g=comparison[comparison.arm=='broad_3sigma'];axes[3].plot(g.time_s,g.event_centroid_relative_um,'ks',label='Descriptive event-energy centroid')
    axes[3].set_ylabel('Relative to first5s (µm)');axes[3].set_xlabel('Recording time (s)');axes[3].legend()
    fig.suptitle('Qualified unit154 transition audit: identity support and discrete shift hypotheses\nNo lines across unsupported bins; energy centroid is not calibrated fractional displacement')
    for ext in ['png','pdf']:fig.savefig(OUT/f'02_support_motion_comparison.{ext}',dpi=150)
    summary=dict(status='complete',seconds=time.monotonic()-begun,target=TARGET,detections=len(detected),identity_events=len(events),unique_shift_events=int(events.shift_unique.sum()),
                 boundary_identity_events=int(events.at_shift_boundary.sum()),boundary_unique_events=int((events.at_shift_boundary&events.shift_unique).sum()),accepted_intervals_lt1ms=int((events.preceding_accepted_interval_ms<1).sum()),exact_aligned_duplicates_removed=int(accepted.sum()-len(events)),
                 fixed_matcher_events=len(existing),newly_recovered_events=int((~events.near_existing_fixed).sum()),
                 fixed_events_near_new_matcher=int(near(existing,events.frame.to_numpy(),.0005*fs).sum()),
                 baseline_unique_events=len(baseline),scope='Bounded conditional identity audit; no calibrated motion conclusion; unit80 and other depths excluded;620um evidence does not validate220um or410um.')
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
