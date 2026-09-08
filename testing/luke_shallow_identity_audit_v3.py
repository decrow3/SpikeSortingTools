"""Bounded quiet-data specificity gate for shallow lighthouse identities; no tracking."""
import json
import hashlib
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from testing.luke_lowpass_waveform_preservation_v2 import reconstruct, SRC, BASE

OUT = SRC / 'luke_shallow_identity_audit_v3'
TARGETS = {80: 220., 154: 620.}
SHIFTS = [-80, -40, 0, 40, 80]
LAGS = [-3, -2, -1, 0, 1, 2, 3]
OFF = np.arange(-30, 31)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def near(events, reference, tolerance):
    reference = np.sort(reference)
    if not len(reference): return np.zeros(len(events), bool)
    ix = np.searchsorted(reference, events)
    return np.minimum(abs(events-reference[np.clip(ix,0,len(reference)-1)]),
                      abs(events-reference[np.clip(ix-1,0,len(reference)-1)])) <= tolerance


def subsample(events, cap):
    events = np.sort(np.asarray(events, dtype=int))
    return events[np.linspace(0,len(events)-1,min(cap,len(events)),dtype=int)] if len(events) else events


def detect(x, channels, noise, fs):
    candidates=[]
    for ch in channels:
        ev=find_peaks(abs(x[:,ch]),height=5*noise[ch],distance=round(.0008*fs))[0]
        candidates.extend((int(e),float(abs(x[e,ch])/noise[ch])) for e in ev if 40<e<len(x)-40)
    # Cross-channel duplicates only; this is not a sorted-unit event extractor.
    chosen=[]
    blocked=np.zeros(len(x),bool)
    radius=round(.0002*fs)
    for e,_ in sorted(candidates,key=lambda z:-z[1]):
        if not blocked[e]:chosen.append(e);blocked[max(0,e-radius):e+radius+1]=True
    return np.sort(chosen)


def score(waves, hypotheses, weights):
    """Common spatial/time weights across every identity/shift; ±3-sample timing."""
    n,h=len(waves),len(hypotheses)
    best=np.full((n,h),-np.inf,dtype='float32');gains=np.zeros((n,h),dtype='float32');lags=np.zeros((n,h),dtype='int8')
    weighted=hypotheses*np.sqrt(weights)[None,:,:]
    vectors=weighted.reshape(h,-1)
    norm=np.sum(vectors*vectors,axis=1)
    for lag in LAGS:
        w=waves[:,3+lag:64+lag]*np.sqrt(weights)[None,:,:]
        flat=w.reshape(n,-1);dot=flat@vectors.T
        cosine=dot/np.sqrt(np.maximum(np.sum(flat*flat,axis=1)[:,None]*norm[None,:],1e-20))
        gain=dot/np.maximum(norm[None,:],1e-20)
        valid=(gain>=.4)&(gain<=2.5)
        update=valid&(cosine>best)
        best[update]=cosine[update];gains[update]=gain[update];lags[update]=lag
    return best,gains,lags


def adjudicate(scores, info, target, threshold=.8):
    ids=np.asarray([r['identity'] for r in info]);targetmask=ids==f'label_{target}'
    targetcols=np.flatnonzero(targetmask); rivalcols=np.flatnonzero(~targetmask)
    own=scores[:,targetcols];win=own.argmax(axis=1);col=targetcols[win];top=own[np.arange(len(scores)),win]
    rival=np.max(scores[:,rivalcols],axis=1) if len(rivalcols) else np.full(len(scores),-np.inf)
    second=np.partition(own,-2,axis=1)[:,-2] if len(targetcols)>1 else np.full(len(scores),-np.inf)
    with np.errstate(invalid='ignore'):
        identity_margin=top-rival;shift_margin=top-second
    accepted=(top>=threshold)&(identity_margin>=.03)
    return accepted,col,top,identity_margin,shift_margin


def main():
    begun=time.monotonic();OUT.mkdir(exist_ok=False)
    manifest_path=BASE/'recording/rescue_recording_manifest.json'
    model_path=SRC/'luke_common_event_screen_v1/shared_response_model.npz'
    m=json.loads(manifest_path.read_text());fs=m['sampling_frequency_hz'];geo=np.asarray(m['channel_locations_um'])
    model=np.load(model_path)
    config=dict(targets=TARGETS,training_s=[4080,4100],heldout_quiet_s=[4100,4110],target_transition_read=False,
        labeled_bank='Every cohort with >=20 training events and full-probe median peak within200um; no cosine pruning. Chronological interleaved half templates, cap100 per half.',
        independent_bank='Both-sign5sigma detections within200um, remove any within0.5ms of nearby label events; uniformly cap3000; normalized-waveform PCA20 + MiniBatchKMeans12 seed20260908. Include every family with>=20members; unknown families never count as certified identities.',
        shifts_um=SHIFTS,timing_lags_samples=LAGS,matching='Common target-union energy/noise weights across all competing identity/shift hypotheses; gain0.4–2.5; cosine>=.8, best-other-identity margin>=.03. Separate best-other-shift margin>=.03.',
        injections='Held-out real backgrounds plus independent-half templates, exact geometry shifts; target gains0.75,1,1.25;10backgrounds per shift/gain. Every label rival also injected10times at each allowed shift at gain1.',
        gates=dict(quiet_reference_events_min=15,quiet_recall_min=.5,quiet_unassigned_fraction_max=.1,
                   injected_target_identity_recovery_min=.9,injected_target_correct_unique_shift_min=.9,injected_rival_false_target_max=.01,injected_worst_rival_false_target_max=.01),
        limitations=['Labels are provisional. Independent clusters are conservative competing waveform families, not a complete cell inventory.',
                    'Exact40um translations do not validate fractional motion, unknown rivals, or transition-time identities.',
                    'No DREDGE access; no target tracking. A pass only permits a separately reviewed bounded transition audit.'],
        resume='No automatic resume; existing output refused.',
        sha256={str(p):digest(p) for p in [Path(__file__).resolve(),Path(__file__).with_name('luke_lowpass_waveform_preservation_v2.py'),manifest_path,model_path,BASE/'cur/cur_output/spike_times.npy',BASE/'cur/cur_output/spike_clusters.npy']},
        calibration='Thresholds fixed before execution; no tuning to pass quiet or injected tests.')
    (OUT/'settings.json').write_text(json.dumps(config,indent=2))
    train,unused,first=reconstruct(4080,m,model);del unused
    quiet,unused,qfirst=reconstruct(4100,m,model);del unused
    quiet=quiet[:round(10*fs)]
    noise=np.median(abs(train-np.median(train,axis=0)),axis=0)/.67448975
    times=np.load(BASE/'cur/cur_output/spike_times.npy',mmap_mode='r').ravel()
    labels=np.load(BASE/'cur/cur_output/spike_clusters.npy',mmap_mode='r').ravel()
    index=np.flatnonzero((times>=first+40)&(times<first+len(train)-40))
    trtime=np.asarray(times[index])-first;trlabel=np.asarray(labels[index])
    allbank={};depths={};refs={}
    for identity in np.unique(trlabel):
        ev=np.sort(trtime[trlabel==identity])
        if len(ev)<20:continue
        a,b=subsample(ev[::2],100),subsample(ev[1::2],100)
        wa=np.median(train[a[:,None]+OFF],axis=0);wb=np.median(train[b[:,None]+OFF],axis=0)
        depth=float(geo[np.argmax(np.max(abs(wa),axis=0)),1])
        if min(abs(depth-d) for d in TARGETS.values())>200:continue
        allbank[f'label_{identity}']=(wa,wb);depths[f'label_{identity}']=depth;refs[f'label_{identity}']=ev
    lookup={tuple(g):i for i,g in enumerate(geo)};results=[]
    for target,depth in TARGETS.items():
        target_key=f'label_{target}'
        if target_key not in allbank:results.append(dict(target=target,status='failed_missing_target_template'));continue
        bank={k:v for k,v in allbank.items() if abs(depths[k]-depth)<=200}
        channels=np.flatnonzero(abs(geo[:,1]-depth)<=360)
        local=np.flatnonzero(abs(geo[:,1]-depth)<=200)
        detected=detect(train,local,noise,fs)
        known=np.concatenate([refs[k] for k in bank]);unknown=detected[~near(detected,known,.0005*fs)]
        unknown=subsample(unknown,3000);familyrows=[]
        if len(unknown)>=40:
            w=train[unknown[:,None,None]+OFF[None,:,None],local[None,None,:]]
            whitened=w/noise[local][None,None,:];flat=whitened.reshape(len(w),-1)
            flat/=np.maximum(np.linalg.norm(flat,axis=1)[:,None],1e-12)
            pc=PCA(n_components=min(20,len(flat)-1),svd_solver='randomized',random_state=20260908).fit_transform(flat)
            cluster=MiniBatchKMeans(n_clusters=12,n_init=3,random_state=20260908,batch_size=512).fit_predict(pc)
            for family in range(12):
                ev=unknown[cluster==family]
                if len(ev)<20:continue
                a,b=subsample(ev[::2],100),subsample(ev[1::2],100)
                wa=np.median(train[a[:,None]+OFF],axis=0);wb=np.median(train[b[:,None]+OFF],axis=0)
                key=f'unknown_{target}_{family}';bank[key]=(wa,wb);depths[key]=float(geo[np.argmax(np.max(abs(wa),axis=0)),1]);familyrows.append(dict(identity=key,events=len(ev),depth_um=depths[key]))
        pd.DataFrame(familyrows).to_csv(OUT/f'u{target}_independent_families.csv',index=False)
        hypotheses=[];validation=[];info=[]
        for identity,(wa,wb) in bank.items():
            # Shift a local120um footprint, preserving actual probe columns.
            source=np.flatnonzero(abs(geo[:,1]-depths[identity])<=120)
            for shift in SHIFTS:
                if abs(depths[identity]+shift-depth)>200:continue
                destination=np.array([lookup.get((geo[c,0],geo[c,1]+shift),-1) for c in source])
                if (destination<0).any() or not np.isin(destination,channels).all():continue
                a=np.zeros((61,len(channels)),dtype='float32');b=a.copy();dest=np.searchsorted(channels,destination)
                a[:,dest]=wa[:,source];b[:,dest]=wb[:,source]
                hypotheses.append(a);validation.append(b);info.append(dict(identity=identity,shift_um=shift,depth_um=depths[identity]))
        hypotheses=np.asarray(hypotheses);validation=np.asarray(validation)
        own=np.asarray([r['identity']==target_key for r in info])
        if not own.any():
            results.append(dict(target=target,status='qualification_failed_no_target_hypotheses'))
            continue
        energy=np.max(hypotheses[own]**2,axis=0)
        weights=energy/(energy+(2*noise[channels])**2+1e-20)
        pd.DataFrame(info).to_csv(OUT/f'u{target}_hypotheses.csv',index=False)
        np.savez_compressed(OUT/f'u{target}_bank.npz',training=hypotheses,independent_half=validation,channels=channels,common_weights=weights,noise_uv=noise[channels])
        qidx=np.flatnonzero((times>=qfirst+40)&(times<qfirst+len(quiet)-40)&(labels==target))
        truth=np.asarray(times[qidx])-qfirst
        candidates=detect(quiet,np.flatnonzero(abs(geo[:,1]-depth)<=120),noise,fs)
        # Include label reference centers to distinguish detection misses from identity failures.
        independent_candidates=candidates.copy()
        candidates=np.unique(np.r_[candidates,truth]);blocks=[];gainblocks=[];lagblocks=[]
        for ev in np.array_split(candidates,max(1,int(np.ceil(len(candidates)/100)))):
            w=quiet[ev[:,None,None]+np.arange(-33,34)[None,:,None],channels[None,None,:]]
            sc,ga,la=score(w,hypotheses,weights);blocks.append(sc);gainblocks.append(ga);lagblocks.append(la)
        scores=np.concatenate(blocks);gains=np.concatenate(gainblocks);lags=np.concatenate(lagblocks)
        accepted,col,top,margin,shiftmargin=adjudicate(scores,info,target)
        hit=candidates[accepted]+lags[np.flatnonzero(accepted),col[accepted]]
        hit=np.unique(hit)
        detected_only=accepted&near(candidates,independent_candidates,.0002*fs)
        independent_hit=candidates[detected_only]+lags[np.flatnonzero(detected_only),col[detected_only]]
        detection_recall=float(near(truth,independent_hit,.0005*fs).mean()) if len(truth) else 0.
        recall=float(near(truth,hit,.0005*fs).mean()) if len(truth) else 0.
        unassigned=float((~near(hit,truth,.0005*fs)).mean()) if len(hit) else 1.
        np.savez_compressed(OUT/f'u{target}_quiet_scores.npz',frames=candidates+qfirst,scores=scores,gains=gains,timing_lags=lags,reference_frames=truth+qfirst)
        pd.DataFrame(dict(frame=candidates+qfirst,accepted_identity=accepted,score=top,identity_margin=margin,shift_margin=shiftmargin,best_shift_um=[info[c]['shift_um'] for c in col])).to_csv(OUT/f'u{target}_quiet_decisions.csv',index=False)
        rng=np.random.default_rng(20260908+target);injectionrows=[];isc=[];ig=[];il=[]
        for hypothesis_index,r in enumerate(info):
            if r['identity'].startswith('unknown_'):continue
            gains_to_test=[.75,1.,1.25] if r['identity']==target_key else [1.]
            for amplitude in gains_to_test:
                bg=rng.integers(40,len(quiet)-40,size=10)
                w=quiet[bg[:,None,None]+np.arange(-33,34)[None,:,None],channels[None,None,:]].copy()
                w[:,3:64]+=amplitude*validation[hypothesis_index]
                sc,ga,la=score(w,hypotheses,weights);ok,cols,tops,margins,sm=adjudicate(sc,info,target)
                for i in range(len(bg)):
                    injectionrows.append(dict(injected_identity=r['identity'],injected_shift_um=r['shift_um'],gain=amplitude,background_frame=int(bg[i]+qfirst),target_accepted=bool(ok[i]),target_best_shift_um=info[cols[i]]['shift_um'],target_shift_unique=bool(sm[i]>=.03),target_score=float(tops[i]),target_identity_margin=float(margins[i]),target_shift_margin=float(sm[i])))
                isc.append(sc);ig.append(ga);il.append(la)
        inj=pd.DataFrame(injectionrows);inj.to_csv(OUT/f'u{target}_real_background_injections.csv',index=False)
        np.savez_compressed(OUT/f'u{target}_injection_scores.npz',scores=np.concatenate(isc),gains=np.concatenate(ig),timing_lags=np.concatenate(il))
        owninj=inj[inj.injected_identity==target_key];rivalinj=inj[inj.injected_identity!=target_key]
        recovery=float(owninj.target_accepted.mean());shiftrecovery=float((owninj.target_accepted&owninj.target_shift_unique&(owninj.target_best_shift_um==owninj.injected_shift_um)).mean())
        false=float(rivalinj.target_accepted.mean()) if len(rivalinj) else 1.
        worst_rival=float(rivalinj.groupby('injected_identity').target_accepted.mean().max()) if len(rivalinj) else 1.
        injected_by_identity=inj.groupby(['injected_identity','injected_shift_um']).target_accepted.agg(['count','mean'])
        injected_by_identity.to_csv(OUT/f'u{target}_injection_by_identity_shift.csv')
        included_unknown=len(set(r['identity'] for r in info if r['identity'].startswith('unknown_')))
        allshifts=set(r['shift_um'] for r in info if r['identity']==target_key)==set(SHIFTS)
        passed=len(truth)>=15 and detection_recall>=.5 and unassigned<=.1 and recovery>=.9 and shiftrecovery>=.9 and false<=.01 and worst_rival<=.01 and included_unknown>0 and allshifts
        results.append(dict(target=target,status='eligible_for_bounded_transition_audit' if passed else 'qualification_failed_no_tracking',labeled_identities=sum(k.startswith('label_') for k in bank),independent_families=len(familyrows),unknown_training_events_sampled=len(unknown),hypotheses=len(info),quiet_reference_events=len(truth),quiet_label_seeded_identity_recall=recall,quiet_independent_detection_recall=detection_recall,quiet_unassigned_fraction=unassigned,injected_target_identity_recovery=recovery,injected_target_unique_correct_shift=shiftrecovery,injected_rival_false_target=false,injected_worst_rival_false_target=worst_rival,included_independent_families=included_unknown,target_all_shifts_present=allshifts))
        print(json.dumps(results[-1]),flush=True)
    pd.DataFrame(results).to_csv(OUT/'qualification.csv',index=False)
    (OUT/'summary.json').write_text(json.dumps(dict(status='complete',seconds=time.monotonic()-begun,results=results,target_tracking_run=False,interpretation='Quiet and injected specificity gate only; no trajectory or biological identity ground truth.'),indent=2))


if __name__=='__main__':main()
