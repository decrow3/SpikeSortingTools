"""Whole-probe translation-invariant waveform test, without absolute-depth priors.
Exact geometry translations at 40 um; no DREDGE, continuity, or depth proximity gates.
"""
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, find_peaks
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from testing.luke_epoch_corroboration import ROOT, BASE
SRC=ROOT/'testing/outputs/luke_population_depth_v2'
OUT=ROOT/'testing/outputs/luke_waveform_only_global_v1'
OFF=np.arange(-24,25)

def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(fig,name):
    for ext in ['pdf','png']: fig.savefig(OUT/f'{name}.{ext}',dpi=160)
    plt.close(fig)
def setup():
    OUT.mkdir(exist_ok=True)
    m=json.loads((BASE/'recording/rescue_recording_manifest.json').read_text()); geo=np.array(m['channel_locations_um'])
    # Each patch has identical absolute lateral geometry and relative vertical geometry.
    bases=np.arange(80,int(geo[:,1].max())-79,40)
    patches=np.array([np.flatnonzero((geo[:,1]>=b-60)&(geo[:,1]<=b+80)) for b in bases])
    assert patches.shape[1]==16
    rel=geo[patches]-np.stack([np.zeros(len(bases)),bases],axis=1)[:,None,:]
    assert np.all(rel==rel[0])
    inv=pd.read_csv(SRC/'candidates.csv'); z=np.load(SRC/'templates.npz'); waves=[]; rows=[]
    for r in inv.itertuples():
        b=40*np.floor(r.depth_um/40)
        if b not in bases: continue
        ch=patches[np.flatnonzero(bases==b)[0]]; w=z[f'unit_{r.unit_id}_full']; peak_t=np.unravel_index(abs(w).argmax(),w.shape)[0]
        # Keep all complete seed identities as rivals, even weak ones.
        if peak_t<24 or peak_t+24>=len(w):continue
        v=w[peak_t+OFF][:,ch].astype('float32'); en=(v*v).sum(axis=0)
        full_en=(w*w).sum(); local=float((w[:,ch]**2).sum()/full_en)
        far=np.max(abs(w[:,np.abs(geo[:,1]-r.depth_um)>100]))/np.max(abs(w))
        rows.append(dict(unit_id=r.unit_id,seed_base_um=b,seed_centroid_um=float(en@geo[ch,1]/en.sum()),repeatability=r.repeatability,snr=r.snr,local_fraction=local,far_ratio=far))
        waves.append(v)
    w=np.array(waves); flat=w.reshape(len(w),-1); norms=np.linalg.norm(flat,axis=1); normalized=flat/norms[:,None]
    # Timing-aligned rival comparisons allow +/-3 samples, without using location.
    sim=np.full((len(w),len(w)),-1.,dtype='float32')
    for lag in range(-3,4):
        shifted=np.zeros_like(w)
        if lag<0:shifted[:,:lag]=w[:,-lag:]
        elif lag>0:shifted[:,lag:]=w[:,:-lag]
        else:shifted=w.copy()
        f=shifted.reshape(len(w),-1);f/=np.maximum(np.linalg.norm(f,axis=1)[:,None],1e-20)
        sim=np.maximum(sim,normalized@f.T)
    np.fill_diagonal(sim,-1); d=pd.DataFrame(rows); rival=sim.argmax(axis=1)
    d['nearest_rival_id']=d.unit_id.to_numpy()[rival];d['nearest_rival_cosine']=sim.max(axis=1)
    d['eligible']=(d.repeatability>=.85)&(d.snr>=5)&(d.local_fraction>=.35)&(d.far_ratio<=.8)&(d.nearest_rival_cosine<.95)
    d['rank_score']=(1-d.nearest_rival_cosine)*d.repeatability
    chosen=d[d.eligible].sort_values('rank_score',ascending=False).head(12).index.to_numpy()
    d['selected']=d.index.isin(chosen);d.to_csv(OUT/'seed_distinctness.csv',index=False)
    np.savez_compressed(OUT/'seed_comparison.npz',waveforms=w,similarities=sim,unit_ids=d.unit_id.to_numpy(),relative_geometry=rel[0])
    config=dict(interval_s=[930,1030],training_s=[930,940],search='Every complete 16-channel patch across probe, exact 40um geometry translations; no absolute depth in scoring, selection, rivals or continuity',patch_relative_y_um=[-60,80],lateral_geometry='Preserved exactly, no interpolation or reflection',waveform_samples=OFF.tolist(),identity='Unweighted normalized multichannel cosine; all complete cached identities compete; +/-3 sample lag',detector='Both signs on every channel; max(30uV, 3sigma); keep strongest channel within +/-40um at peak time, 0.8ms channel spacing',acceptance=dict(cosine=.86,margin=.025,gain=[.35,3]),selection='Top12 repeatable localized seeds ranked by 1 - global closest-rival cosine; no depth strata',checkpoint='Hash-validated completed 5s chunks reused; interrupted chunk must restart after evidence review; no within-chunk checkpoint',limitations='Exploratory thresholds, not false-positive calibrated. Seed identities inherited from existing sorting. Exact40um lattice can miss intermediate shape changes. Probe edges without full support excluded. Unknown identities, collisions and waveform evolution remain possible.',sources={str(p):digest(p) for p in [Path(__file__),SRC/'templates.npz',SRC/'candidates.csv',BASE/'recording/rescue_recording_manifest.json']})
    path=OUT/'settings.json'
    if path.exists():assert json.loads(path.read_text())==config,'Use a new version for changed settings'
    else:path.write_text(json.dumps(config,indent=2))
    (OUT/'chart_contract.json').write_text(json.dumps(dict(question='Do globally distinctive relative waveforms recur at shifted probe positions?',surface='Standalone Matplotlib PDF/PNG',charts='Seed-versus-rival waveform panels; per-event depth/time scatter with uncertain matches and original bounded-search control',grain='One detected event; 930–1030s; no smoothing or gap filling',palette='Blue accepted, orange ambiguous, gray historical control',fallback='Show empty panels and counts if no accepted observations'),indent=2))
    fig,axs=plt.subplots(max(1,len(chosen)),2,figsize=(12,max(4,2*len(chosen))),squeeze=False,layout='constrained')
    for axpair,i in zip(axs,chosen):
        j=rival[i]
        for ax,k,title in zip(axpair,[i,j],[f'Seed {d.unit_id[i]}',f'Closest rival {d.unit_id[j]} · cosine {sim[i,j]:.3f}']):
            vv=w[k]/np.max(abs(w[k])); ax.plot(OFF, vv+np.arange(16)[None,:]*.7,color='#0072B2' if k==i else '#777777',lw=.7);ax.set(title=title,xlabel='Samples from peak',yticks=[])
    fig.suptitle(f'Absolute-depth-hidden seed comparison · {len(d)} competing templates\n930–940 s seeds; relative 16-channel footprints; selected without depth strata')
    save(fig,'01_seed_vs_global_rivals')
    print('SEEDS',len(d),'SELECTED',d.loc[chosen,'unit_id'].tolist(),flush=True)
    return m,geo,bases,patches,d,w,normalized,norms

def extract(state):
    m,geo,bases,patches,d,w,normalized,norms=state;fs=m['sampling_frequency_hz'];raw=BASE/'recording/traces_cached_seg0.raw';rawstat=raw.stat();sos=butter(3,[300,6000],fs=fs,btype='bandpass',output='sos');ids=d.unit_id.to_numpy();selected=set(d[d.selected].unit_id)
    # Explicit geometry sign check: a synthetic translated seed has unchanged features.
    for b,ch in zip(bases,patches):
        synthetic=np.zeros((49,384),dtype='float32');synthetic[:,ch]=w[0]
        assert np.allclose(synthetic[:,ch].ravel()@normalized[0]/norms[0],1,atol=1e-5)
        assert abs(((w[0]**2).sum(axis=0)@geo[ch,1]/(w[0]**2).sum())-(d.seed_centroid_um[0]+b-d.seed_base_um[0]))<.002
    (OUT/'geometry_checks.json').write_text(json.dumps(dict(patches=len(bases),identity_invariant=True,centroid_sign_verified=True)))
    for start in range(930,1030,5):
        seal=OUT/f'chunk_{start}.complete.json'
        if seal.exists():
            for fn,h in json.loads(seal.read_text()).items():assert digest(OUT/fn)==h
            continue
        p=OUT/f'chunk_{start}.csv'; scorepath=OUT/f'chunk_{start}_scores.npz'
        if p.exists() or scorepath.exists():raise RuntimeError('Unsealed chunk evidence exists; investigate before restart')
        tick=time.monotonic();first=round(start*fs);n=round(5*fs);pad=round(.05*fs)
        with raw.open('rb') as f:f.seek((first-pad)*768);buf=f.read((n+2*pad)*768)
        assert len(buf)==(n+2*pad)*768
        x=np.frombuffer(buf,dtype='<i2').reshape(-1,384).astype('float32')*m['gain_uv_per_count'];x=sosfiltfilt(sos,x,axis=0).astype('float32');x-=np.median(x,axis=1,keepdims=True);x=x[pad:pad+n]
        noise=np.median(abs(x[::10]-np.median(x[::10],axis=0)),axis=0)/.67448975
        events=[]
        for pk in range(384):
            base=40*np.floor(geo[pk,1]/40)
            if base not in bases:continue
            ev=find_peaks(abs(x[:,pk]),height=max(30,3*noise[pk]),distance=round(.0008*fs))[0];ev=ev[(ev>30)&(ev<n-30)]
            near=np.flatnonzero(abs(geo[:,1]-geo[pk,1])<=40)
            ev=ev[np.argmax(abs(x[ev[:,None],near]),axis=1)==np.flatnonzero(near==pk)[0]]
            events.extend((int(e),int(np.flatnonzero(bases==base)[0]),pk) for e in ev)
        events=np.array(sorted(events),dtype=int).reshape(-1,3);rows=[];allscores=[];allgain=[];alllag=[]
        for a in range(0,len(events),256):
            e,bi,pk=events[a:a+256].T;ch=patches[bi];best=np.full((len(e),len(d)),-1.,dtype='float32');gains=np.zeros_like(best);lags=np.zeros_like(best,dtype='int8')
            for lag in range(-3,4):
                wave=x[e[:,None,None]+OFF[None,:,None]+lag,ch[:,None,:]];f=wave.reshape(len(e),-1);wn=np.linalg.norm(f,axis=1);dots=f@normalized.T;sc=dots/np.maximum(wn[:,None],1e-20);gain=dots/norms[None,:];better=sc>best
                best[better]=sc[better];gains[better]=gain[better];lags[better]=lag
            order=np.argsort(best,axis=1);win=order[:,-1];rival=order[:,-2];ar=np.arange(len(e));s=best[ar,win];margin=s-best[ar,rival];gain=gains[ar,win]
            for k in range(len(e)):
                wi=win[k];frame=e[k]+lags[k,wi];wave=x[frame+OFF][:,ch[k]];energy=(wave.astype(float)**2).sum(axis=0);cent=energy@geo[ch[k],1]/energy.sum()
                status='accepted' if s[k]>=.86 and margin[k]>=.025 and .35<=gain[k]<=3 else ('identity_ambiguous' if s[k]>=.86 and .35<=gain[k]<=3 else 'unmatched')
                rows.append(dict(frame=int(first+frame),time_s=(first+frame)/fs,peak_channel=int(pk[k]),patch_base_um=bases[bi[k]],unit_id=ids[wi],rival_id=ids[rival[k]],score=s[k],margin=margin[k],gain=gain[k],status=status,selected=ids[wi] in selected,centroid_um=cent,relative_um=cent-d.seed_centroid_um[wi],training=start<940))
            allscores.append(best);allgain.append(gains);alllag.append(lags)
        table=pd.DataFrame(rows);table.to_csv(p,index=False)
        np.savez_compressed(scorepath,scores=np.concatenate(allscores),gains=np.concatenate(allgain),lags=np.concatenate(alllag),detections=events,unit_ids=ids)
        seal.write_text(json.dumps({p.name:digest(p),scorepath.name:digest(scorepath)},indent=2))
        print('CHUNK',start,'detections',len(rows),'selected accepted',int(((table.status=='accepted')&table.selected).sum()),'seconds',round(time.monotonic()-tick,1),flush=True)
    assert (raw.stat().st_size,raw.stat().st_mtime_ns)==(rawstat.st_size,rawstat.st_mtime_ns)

def report(state):
    d=state[4];ev=pd.concat([pd.read_csv(OUT/f'chunk_{s}.csv') for s in range(930,1030,5)],ignore_index=True);ev.to_csv(OUT/'events.csv',index=False)
    chosen=d[d.selected];fig,axs=plt.subplots(max(1,int(np.ceil(len(chosen)/3))),3,figsize=(16,max(4,3*int(np.ceil(len(chosen)/3)))),squeeze=False,layout='constrained');old=pd.read_csv(SRC/'events.csv');summary=[]
    for ax,r in zip(axs.flat,chosen.itertuples()):
        q=ev[ev.unit_id==r.unit_id];good=q[q.status=='accepted'];amb=q[q.status=='identity_ambiguous'];control=old[(old.unit_id==r.unit_id)&(old.status=='accepted')]
        ax.scatter(control.time_s,control.centroid_um-r.seed_centroid_um,s=3,color='#999999',alpha=.25,label='Old bounded search')
        ax.scatter(amb.time_s,amb.relative_um,s=8,color='#D55E00',marker='x',alpha=.3,label='Ambiguous')
        ax.scatter(good.time_s,good.relative_um,s=4,color='#0072B2',alpha=.5,label='Waveform accepted')
        ax.axvspan(930,940,color='gray',alpha=.12);ax.axhline(0,color='gray',lw=.5);ax.set(xlim=(930,1030),xlabel='Time (s)',ylabel='Seed-relative centroid (µm)',title=f'Unit {r.unit_id} · {len(good)} accepted');ax.legend(fontsize=6)
        held=good[good.time_s>=940];summary.append(dict(unit_id=r.unit_id,accepted=len(good),heldout_accepted=len(held),ambiguous=len(amb),heldout_beyond120=int((abs(held.relative_um)>120).sum()),heldout_min_um=held.relative_um.min(),heldout_max_um=held.relative_um.max()))
    for ax in list(axs.flat)[len(chosen):]:ax.axis('off')
    fig.suptitle('Whole-probe waveform identity test · 930–1030 s\nAbsolute depth excluded from matching; shaded seed epoch; individual events, no smoothing; identities provisional')
    save(fig,'02_global_waveform_motion');pd.DataFrame(summary).to_csv(OUT/'unit_summary.csv',index=False)
    result=dict(status='complete',detections=len(ev),selected_units=len(chosen),selected_summary=summary)
    (OUT/'summary.json').write_text(json.dumps(result,indent=2));(OUT/'README.md').write_text('''# Whole-probe waveform-only identity test\n\nSee 01_seed_vs_global_rivals.pdf and 02_global_waveform_motion.pdf. Selection and identity scores do not use absolute depth, displacement predictions, or temporal continuity. Geometry is retained as an exact translated 16-channel footprint. All complete cached seed identities compete, including those not selected for plotting. Depth is revealed after scoring to compute energy centroids on each observed patch.\n\nThis is an exploratory identity test, not calibrated motion estimation. A high cosine is not proof of identity. Whole-probe searches increase incidental matches; unknown cells and collisions remain unmodeled. Exact 40µm translations can miss intermediate waveform changes; incomplete edge patches are excluded. Local peak detection can miss overlapped or weak spikes. The template inventory comes from existing sorted labels and is not an independent discovery of all cells. Unmatched detections and all per-identity score/gain/lag alternatives are retained in chunk files. No continuous path is inferred across missing support; dispersed depth matches may be false identity matches. Historical bounded observations are gray where available.\n\nThe first10s are training overlap; use940–1030s for held-out evidence. Fixed thresholds cosine>=0.86, identity margin>=0.025 and gain0.35–3 are exploratory, not newly false-positive calibrated. Seed selection uses waveform repeatability, SNR, locality and global rival distinctness, with no depth quotas. Completed5s chunks are hash checked on restart; interrupted chunks require evidence review and restart, not within-chunk resumption.\n''')
    print(json.dumps(result),flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--seed-only',action='store_true');args=ap.parse_args();state=setup()
    if not args.seed_only:extract(state);report(state)
