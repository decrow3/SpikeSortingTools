"""Bounded shallow reference experiment with validated atomic stage checkpoints."""
import argparse
import importlib.metadata
import platform
import fcntl
import hashlib
import json
import os
import time
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from testing.luke_shallow_identity_audit_v3 import score, detect, near, SHIFTS
from testing.luke_lowpass_waveform_preservation_v2 import reconstruct, SRC, BASE

OUT=SRC/'luke_shallow_reference_overnight_v1'
SOURCE=SRC/'luke_shallow_identity_audit_v3'
CHUNKS=[4240,4180,4160,4200,4220]


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def atomic_json(path,value):
    temporary=Path(str(path)+'.tmp')
    with temporary.open('w') as f:
        json.dump(value,f,indent=2);f.flush();os.fsync(f.fileno())
    os.replace(temporary,path)


class StageStore:
    """Completed outputs require exact hashes; incomplete stages restart, never merge."""
    def __init__(self,root,settings):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self.lock=(self.root/'run.lock').open('a+')
        fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        self.digest=hashlib.sha256(json.dumps(settings,sort_keys=True).encode()).hexdigest()
        path=self.root/'settings.json'
        if path.exists():
            if json.loads(path.read_text())!=settings:raise RuntimeError('Settings/source/input hashes changed; preserve this run and use a new version.')
        else:atomic_json(path,settings)

    def run(self,name,work):
        final=self.root/name
        if final.exists():
            receipt=json.loads((final/'receipt.json').read_text())
            if receipt['run_sha256']!=self.digest:raise RuntimeError(f'{name}: receipt run hash mismatch')
            actual={str(p.relative_to(final)) for p in final.rglob('*') if p.is_file() and p.name!='receipt.json'}
            if actual!=set(receipt['files']):raise RuntimeError(f'{name}: missing or unexpected stage outputs')
            for path,digest in receipt['files'].items():
                if sha(final/path)!=digest:raise RuntimeError(f'{name}: corrupt output {path}; no automatic replacement')
            print(name,'validated checkpoint reused',flush=True);return final
        partial=self.root/(name+'.partial')
        if partial.exists():os.replace(partial,self.root/(name+f'.interrupted-{time.time_ns()}'))
        partial.mkdir();begun=time.monotonic()
        work(partial)
        files={str(p.relative_to(partial)):sha(p) for p in sorted(partial.rglob('*')) if p.is_file()}
        if not files:raise RuntimeError(f'{name}: no outputs')
        atomic_json(partial/'receipt.json',dict(run_sha256=self.digest,files=files,seconds=time.monotonic()-begun,status='complete'))
        os.replace(partial,final)
        directory=os.open(self.root,os.O_RDONLY);os.fsync(directory);os.close(directory)
        print(name,'checkpoint committed',flush=True);return final


def build_bank(templates,depths,geo,noise,target,depth):
    lookup={tuple(g):i for i,g in enumerate(geo)}
    channels=np.flatnonzero(abs(geo[:,1]-depth)<=360);info=[];train=[];test=[]
    for identity,(wa,wb) in templates.items():
        if abs(depths[identity]-depth)>200:continue
        source=np.flatnonzero(abs(geo[:,1]-depths[identity])<=120)
        for shift in SHIFTS:
            if abs(depths[identity]+shift-depth)>200:continue
            dest=np.array([lookup.get((geo[c,0],geo[c,1]+shift),-1) for c in source])
            if (dest<0).any() or not np.isin(dest,channels).all():continue
            a=np.zeros((61,len(channels)),np.float32);b=a.copy();ix=np.searchsorted(channels,dest)
            a[:,ix]=wa[:,source];b[:,ix]=wb[:,source]
            train.append(a);test.append(b);info.append(dict(identity=identity,shift_um=shift,depth_um=depths[identity]))
    train=np.asarray(train);test=np.asarray(test);own=np.array([r['identity']==f'label_{target}' for r in info])
    if own.sum()!=5:raise RuntimeError(f'{target}: missing target shifts')
    ownenergy=train[own]**2
    union=np.max(ownenergy,axis=0);union/=union+(2*noise[channels])**2+1e-20
    contexts=ownenergy/(ownenergy+(2*noise[channels])**2+1e-20)
    return train,test,channels,info,union,contexts


def infer(waves,hypotheses,info,union,contexts,target,mode):
    if not len(waves):
        columns={k:pd.Series(dtype='bool' if k in ['accepted_identity','unique_shift'] else 'int64' if k in ['timing_lag','shift_um'] else 'float64') for k in ['accepted_identity','unique_shift','score','identity_margin','shift_margin','shift_um','timing_lag','gain']}
        return pd.DataFrame(columns),np.empty((0,1 if mode=='union' else len(contexts),len(hypotheses)),np.float32)
    own=np.flatnonzero([r['identity']==f'label_{target}' for r in info]);rivals=np.flatnonzero([r['identity']!=f'label_{target}' for r in info])
    matrices=[];gains=[];lags=[]
    for weights in [union] if mode=='union' else contexts:
        sc,ga,la=score(waves,hypotheses,weights);matrices.append(sc);gains.append(ga);lags.append(la)
    matrices=np.stack(matrices,axis=1);gains=np.stack(gains,axis=1);lags=np.stack(lags,axis=1)
    ownscore=np.stack([matrices[:,0 if mode=='union' else k,c] for k,c in enumerate(own)],axis=1)
    best=ownscore.argmax(axis=1);column=own[best];context=np.zeros(len(waves),int) if mode=='union' else best
    ix=np.arange(len(waves));top=ownscore[ix,best];rival=np.max(matrices[ix,context][:,rivals],axis=1)
    with np.errstate(invalid='ignore'):
        identitymargin=top-rival;shiftmargin=top-np.partition(ownscore,-2,axis=1)[:,-2]
    accepted=(top>=.8)&(identitymargin>=.03);unique=shiftmargin>=.03
    table=pd.DataFrame(dict(accepted_identity=accepted,unique_shift=unique,score=top,identity_margin=identitymargin,shift_margin=shiftmargin,
        shift_um=[info[c]['shift_um'] for c in column],timing_lag=lags[ix,context,column],gain=gains[ix,context,column]))
    return table,matrices


def raw_fingerprint(raw,fs):
    before=raw.stat();windows={}
    for start in [4110]+CHUNKS:
        first=round(start*fs);pad=round(.05*fs);remaining=(round(20*fs)+2*pad)*768;h=hashlib.sha256()
        with raw.open('rb') as f:
            f.seek((first-pad)*768)
            while remaining:
                block=f.read(min(8*1024*1024,remaining))
                if not block:raise RuntimeError('Raw recording truncated')
                h.update(block);remaining-=len(block)
        windows[str(start)]=h.hexdigest()
    after=raw.stat()
    if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise RuntimeError('Raw source changed during fingerprint')
    return dict(path=str(raw),size=before.st_size,mtime_ns=before.st_mtime_ns,inode=before.st_ino,bounded_20s_window_sha256=windows)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=OUT);args=parser.parse_args()
    manifest=BASE/'recording/rescue_recording_manifest.json';model=SRC/'luke_common_event_screen_v1/shared_response_model.npz'
    inputs=[manifest,model,BASE/'cur/cur_output/spike_times.npy',BASE/'cur/cur_output/spike_clusters.npy']
    for target in [80,154]:inputs += [SOURCE/f'u{target}_bank.npz',SOURCE/f'u{target}_hypotheses.csv']
    m=json.loads(manifest.read_text());fs=m['sampling_frequency_hz'];raw=BASE/'recording/traces_cached_seg0.raw';rawproof=raw_fingerprint(raw,fs)
    code=[Path(__file__).resolve(),Path(__file__).with_name('luke_shallow_identity_audit_v3.py'),Path(__file__).with_name('luke_lowpass_waveform_preservation_v2.py')]
    settings=dict(version=1,runtime=dict(python=platform.python_version(),packages={name:importlib.metadata.version(name) for name in ['numpy','pandas','scipy','scikit-learn','matplotlib','spikeinterface']}),raw_source=rawproof,candidates='Unit80 plus at most one quiet-training-ranked label in each depthband[300,380),[380,440),[440,520); max4total; no holdout/motion selection.',
        weighting='Compare original union weights versus one shift-conditioned common-target weight. Every rival identity/shift scored under SAME weights as target in each context; choose highest targetscore, then apply unchanged margins.',
        gates=dict(cosine=.8,gain=[.4,2.5],identity_margin=.03,unique_shift_margin=.03,quiet_reference_min=15,quiet_independent_recall_min=.5,quiet_unassigned_max=.1,injected_identity_recovery_min=.9,injected_unique_correct_shift_min=.9,worst_rival_false_target_max=.01),
        reused_development_quiet_s=[4110,4120],training_source='Frozen4080–4100 interleaved-half banks+independent unknownfamilies from v3; no new clustering.',
        injections='Paired known-center real4110–4120backgrounds;10per shift/gain(.75,1,1.25)target;10per rivalshift gain1. Known-center scoring is not detector recovery.',
        target_chunks_s=CHUNKS,transition='Only contextualqualifiedunits; same5sigma both-sign detector±120um. 5sbins>=10uniqueevents & dominantshift>=.8; gaps/mixed shifts explicit; noDREDGEread.',
        checkpoint='Atomic stage directories+receipt outputhashes, source/input/settingshashes validated; incomplete stage restarts preserving partial evidence; no withinstage checkpoint.',
        bootstrap='200within-bin actualeventresamples and interleaved-halfcentroiddifference conditional on accepteddominantidentity; measurementuncertainty only, not physicaltruth.',
        resource='One numerical worker; external BLAS/OMP1; bounded cachedquiet10s+five20stransitionchunks about5.1GBfloat32; nofullrecordingread.',
        hashes={str(p):sha(p) for p in inputs+code})
    store=StageStore(args.output,settings)
    m=json.loads(manifest.read_text());fs=m['sampling_frequency_hz'];geo=np.asarray(m['channel_locations_um']);response=np.load(model)
    def prepare(path):
        templates={};depths={};noise=np.full(384,np.nan)
        for origin in [80,154]:
            z=np.load(SOURCE/f'u{origin}_bank.npz');info=pd.read_csv(SOURCE/f'u{origin}_hypotheses.csv');channels=z['channels'];noise[channels]=z['noise_uv']
            for i,r in info.iterrows():
                if r.shift_um!=0 or r.identity in templates:continue
                a=np.zeros((61,384),np.float32);b=a.copy();a[:,channels]=z['training'][i];b[:,channels]=z['independent_half'][i]
                templates[r.identity]=(a,b);depths[r.identity]=float(r.depth_um)
        noise[~np.isfinite(noise)]=float(np.nanmedian(noise))
        ranking=[]
        for identity,(a,b) in templates.items():
            if not identity.startswith('label_') or not 300<=depths[identity]<520:continue
            channels=np.flatnonzero(abs(geo[:,1]-depths[identity])<=120);aa=a[:,channels]/noise[channels];bb=b[:,channels]/noise[channels]
            co=float(np.sum(aa*bb)/np.maximum(np.linalg.norm(aa)*np.linalg.norm(bb),1e-20));snr=float(np.max(abs(aa)))
            ranking.append(dict(target=int(identity.split('_')[1]),depth_um=depths[identity],half_cosine=co,peak_snr=snr,rank_score=co*min(snr,30)))
        rank=pd.DataFrame(ranking,columns=['target','depth_um','half_cosine','peak_snr','rank_score']).sort_values(['rank_score','target'],ascending=[False,True]);rank.to_csv(path/'quiet_candidate_ranking.csv',index=False)
        candidates=[dict(target=80,depth_um=220.)]
        for low,high in [(300,380),(380,440),(440,520)]:
            eligible=rank[(rank.depth_um>=low)&(rank.depth_um<high)&(rank.half_cosine>=.9)]
            if len(eligible):candidates.append(dict(target=int(eligible.iloc[0].target),depth_um=float(eligible.iloc[0].depth_um)))
        for candidate in candidates:
            target,depth=candidate['target'],candidate['depth_um']
            try:a,b,ch,info,union,contexts=build_bank(templates,depths,geo,noise,target,depth)
            except RuntimeError as error:
                candidate['bank_ready']=False;candidate['failure']=str(error);continue
            candidate['bank_ready']=True;candidate['failure']=''
            np.savez_compressed(path/f'u{target}_bank.npz',training=a,independent_half=b,channels=ch,noise_uv=noise[ch],union=union,contexts=contexts)
            pd.DataFrame(info).to_csv(path/f'u{target}_hypotheses.csv',index=False)
        pd.DataFrame(candidates).to_csv(path/'candidates.csv',index=False)
    prepared=store.run('prepare',prepare);candidates=pd.read_csv(prepared/'candidates.csv')
    def cache(path,start,duration):
        before=raw.stat()
        if (before.st_size,before.st_mtime_ns)!=(rawproof['size'],rawproof['mtime_ns']):raise RuntimeError('Raw source changed before cache')
        broad,unused,first=reconstruct(start,m,response);del unused
        after=raw.stat()
        if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise RuntimeError('Raw source changed during cache')
        np.save(path/'voltage.npy',broad[:round(duration*fs)])
        atomic_json(path/'metadata.json',dict(start_s=start,first_frame=first,duration_s=duration))
    quietpath=store.run('quiet_4110',lambda p:cache(p,4110,10));quiet=np.load(quietpath/'voltage.npy',mmap_mode='r');qfirst=json.loads((quietpath/'metadata.json').read_text())['first_frame']
    times=np.load(BASE/'cur/cur_output/spike_times.npy',mmap_mode='r').ravel();labels=np.load(BASE/'cur/cur_output/spike_clusters.npy',mmap_mode='r').ravel()
    qualifications=[]
    for candidate in candidates.itertuples():
        target,depth=int(candidate.target),float(candidate.depth_um)
        def qualify(path):
            if not bool(candidate.bank_ready):
                pd.DataFrame([dict(target=target,depth_um=depth,mode=mode,qualified=False,failure=str(candidate.failure)) for mode in ['union','contextual']]).to_csv(path/'qualification.csv',index=False);return
            bank=np.load(prepared/f'u{target}_bank.npz');info=pd.read_csv(prepared/f'u{target}_hypotheses.csv').to_dict('records');ch=bank['channels'];noise=np.full(384,np.inf);noise[ch]=bank['noise_uv']
            events=detect(quiet,np.flatnonzero(abs(geo[:,1]-depth)<=120),noise,fs)
            if not len(events):
                pd.DataFrame([dict(target=target,depth_um=depth,mode=mode,qualified=False,failure='No independent quiet detections') for mode in ['union','contextual']]).to_csv(path/'qualification.csv',index=False);return
            references=np.asarray(times[(times>=qfirst+40)&(times<qfirst+len(quiet)-40)&(labels==target)])-qfirst
            rng=np.random.default_rng(20260909+target);injection_events=[];injection_meta=[]
            for h,r in enumerate(info):
                for amplitude in [.75,1.,1.25] if r['identity']==f'label_{target}' else [1.]:
                    centers=rng.integers(40,len(quiet)-40,size=10)
                    for center in centers:injection_events.append((int(center),h,amplitude));injection_meta.append(dict(injected_identity=r['identity'],injected_shift_um=r['shift_um'],amplitude=amplitude,background_frame=int(center+qfirst),provisional_family=r['identity'].startswith('unknown_')))
            results=[]
            for mode in ['union','contextual']:
                tables=[];matrices=[]
                for indices in np.array_split(np.arange(len(events)),max(1,int(np.ceil(len(events)/100)))):
                    centers=events[indices];waves=quiet[centers[:,None,None]+np.arange(-33,34)[None,:,None],ch[None,None,:]]
                    table,matrix=infer(waves,bank['training'],info,bank['union'],bank['contexts'],target,mode);tables.append(table);matrices.append(matrix)
                table=pd.concat(tables,ignore_index=True);table['frame']=events+qfirst+table.timing_lag
                table.to_csv(path/f'{mode}_quiet_events.csv',index=False);np.savez_compressed(path/f'{mode}_quiet_scores.npz',scores=np.concatenate(matrices),detection_frames=events+qfirst)
                hit=table[table.accepted_identity].frame.to_numpy()-qfirst
                recall=float(near(references,hit,.0005*fs).mean()) if len(references) else 0.
                unassigned=float((~near(hit,references,.0005*fs)).mean()) if len(hit) else 1.
                injecttables=[];injectmatrices=[]
                for indices in np.array_split(np.arange(len(injection_events)),max(1,int(np.ceil(len(injection_events)/50)))):
                    specification=[injection_events[i] for i in indices];centers=np.array([r[0] for r in specification]);waves=quiet[centers[:,None,None]+np.arange(-33,34)[None,:,None],ch[None,None,:]].copy()
                    for i,(_,h,amplitude) in enumerate(specification):waves[i,3:64]+=amplitude*bank['independent_half'][h]
                    t,mm=infer(waves,bank['training'],info,bank['union'],bank['contexts'],target,mode);injecttables.append(t);injectmatrices.append(mm)
                inj=pd.concat([pd.DataFrame(injection_meta),pd.concat(injecttables,ignore_index=True)],axis=1)
                inj.to_csv(path/f'{mode}_injections.csv',index=False);np.savez_compressed(path/f'{mode}_injection_scores.npz',scores=np.concatenate(injectmatrices))
                own=inj[inj.injected_identity==f'label_{target}'];rival=inj[inj.injected_identity!=f'label_{target}']
                recovery=float(own.accepted_identity.mean());shift=float((own.accepted_identity&own.unique_shift&(own.shift_um==own.injected_shift_um)).mean())
                rates=rival.groupby('injected_identity').accepted_identity.agg(['count','mean']);rates.to_csv(path/f'{mode}_rival_rates.csv')
                worst=float(rates['mean'].max()) if len(rates) else 1.
                passed=len(references)>=15 and recall>=.5 and unassigned<=.1 and recovery>=.9 and shift>=.9 and worst<=.01
                results.append(dict(target=target,depth_um=depth,mode=mode,quiet_reference_events=len(references),quiet_recall=recall,quiet_unassigned=unassigned,injected_identity_recovery=recovery,injected_unique_shift_recovery=shift,worst_rival_false_target=worst,qualified=passed))
            pd.DataFrame(results).to_csv(path/'qualification.csv',index=False)
            fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
            for ax,col,title in zip(axes,['quiet_recall','injected_unique_shift_recovery','worst_rival_false_target'],['Development quiet detection recall','Injected unique correct shift','Worst rival false-target fraction']):
                ax.bar([r['mode'] for r in results],[r[col] for r in results],color=['#888888','#2468a2']);ax.set_title(title);ax.set_ylim(0,1)
            axes[0].axhline(.5,color='black',ls=':');axes[1].axhline(.9,color='black',ls=':');axes[2].axhline(.01,color='black',ls=':')
            fig.suptitle(f'Unit{target}: frozen weighting comparison, reused4110–4120s development controls')
            fig.savefig(path/'qualification.png',dpi=140);plt.close(fig)
        qualifiedpath=store.run(f'qualify_u{target}',qualify);qualifications.append(pd.read_csv(qualifiedpath/'qualification.csv'))
    qualified=pd.concat(qualifications,ignore_index=True);passed=qualified[(qualified['mode']=='contextual')&qualified.qualified]
    # Any failed unit is excluded from all transition chunks without retuning.
    for start in CHUNKS if len(passed) else []:
        voltagepath=store.run(f'voltage_{start}',lambda p:cache(p,start,20));voltage=np.load(voltagepath/'voltage.npy',mmap_mode='r');first=json.loads((voltagepath/'metadata.json').read_text())['first_frame']
        for unit in passed.itertuples():
            target,depth=int(unit.target),float(unit.depth_um)
            def track(path):
                bank=np.load(prepared/f'u{target}_bank.npz');info=pd.read_csv(prepared/f'u{target}_hypotheses.csv').to_dict('records');ch=bank['channels'];noise=np.full(384,np.inf);noise[ch]=bank['noise_uv']
                ev=detect(voltage,np.flatnonzero(abs(geo[:,1]-depth)<=120),noise,fs);tables=[];matrices=[]
                for indices in np.array_split(np.arange(len(ev)),max(1,int(np.ceil(len(ev)/100)))):
                    waves=voltage[ev[indices,None,None]+np.arange(-33,34)[None,:,None],ch[None,None,:]]
                    t,sc=infer(waves,bank['training'],info,bank['union'],bank['contexts'],target,'contextual');tables.append(t);matrices.append(sc)
                table=pd.concat(tables,ignore_index=True);table['frame']=ev+first+table.timing_lag;table['time_s']=table.frame/fs
                table.to_csv(path/'all_events.csv',index=False);np.savez_compressed(path/'all_scores.npz',scores=np.concatenate(matrices),detection_frames=ev+first)
                accepted=table[table.accepted_identity].sort_values('score',ascending=False).drop_duplicates('frame').sort_values('frame')
                accepted.to_csv(path/'accepted_events.csv',index=False);rows=[];templates={}
                for lo in range(start,start+20,5):
                    g=accepted[(accepted.time_s>=lo)&(accepted.time_s<lo+5)&accepted.unique_shift];fraction=float(g.shift_um.value_counts(normalize=True).max()) if len(g) else 0.;edge=bool((abs(g.shift_um)==80).any());dominant_count=int(g.shift_um.value_counts().max()) if len(g) else 0;enough=dominant_count>=10 and fraction>=.8 and not edge
                    row=dict(start_s=lo,events=len(g),dominant_fraction=fraction,status='supported' if enough else 'edge' if edge else 'mixed' if len(g)>=10 else 'sparse',centroid_um=np.nan,bootstrap_low_um=np.nan,bootstrap_high_um=np.nan,half_difference_um=np.nan,bootstrap_events=0,boundary_events=int((abs(g.shift_um)==80).sum()),**{f'shift_{s}':int((g.shift_um==s).sum()) for s in SHIFTS})
                    if enough:
                        shift=int(g.shift_um.mode().iloc[0]);use=g[g.shift_um==shift];frames=use.frame.to_numpy(dtype=int)-first;foot=np.flatnonzero(abs(geo[:,1]-(depth+shift))<=120)
                        waves=voltage[frames[:,None,None]+np.arange(-30,31)[None,:,None],foot[None,None,:]]
                        def center(w):
                            energy=np.sum(np.median(w,axis=0)**2,axis=0);return float(energy@geo[foot,1]/energy.sum())
                        waveform=np.median(waves,axis=0);row['centroid_um']=center(waves);rng=np.random.default_rng(20260909+target+lo)
                        boot=np.array([center(waves[rng.integers(0,len(waves),len(waves))]) for _ in range(200)]);row['bootstrap_low_um'],row['bootstrap_high_um']=map(float,np.quantile(boot,[.025,.975]));row['half_difference_um']=center(waves[::2])-center(waves[1::2]);row['bootstrap_events']=len(waves);templates[f't{lo}_waveform']=waveform;templates[f't{lo}_channels']=foot
                    rows.append(row)
                pd.DataFrame(rows).to_csv(path/'support_5s.csv',index=False);np.savez_compressed(path/'median_templates.npz',**templates)
                fig,axes=plt.subplots(2,1,figsize=(11,6),sharex=True,layout='constrained');axes[0].scatter(accepted.time_s,accepted.shift_um,c=np.where(accepted.unique_shift,'#2468a2','#bbbbbb'),s=8);axes[0].set_ylabel('Winning discrete shift(µm)')
                d=pd.DataFrame(rows);axes[1].bar(d.start_s+2.5,d.events,width=4);axes[1].axhline(10,color='black',ls=':');axes[1].set(ylabel='Unique-shift events/5s',xlabel='Recording time(s)');fig.suptitle(f'Unit{target}: frozen contextual matcher · sparse/mixed bins remain gaps');fig.savefig(path/'support.png',dpi=140);plt.close(fig)
            store.run(f'track_u{target}_{start}',track)
    def report(path):
        qualified.to_csv(path/'qualification.csv',index=False)
        atomic_json(path/'summary.json',dict(status='complete',qualified_contextual_ids=[int(i) for i in passed.target],candidate_ids=[int(i) for i in candidates.target],scope='Bounded reference qualification only; no DREDGE input, motion accuracy conclusion, or full-session scan.'))
    if raw_fingerprint(raw,fs)!=rawproof:raise RuntimeError('Raw source changed: final bounded-content verification failed')
    for path,digest in settings['hashes'].items():
        if sha(path)!=digest:raise RuntimeError(f'Source/input changed before final report: {path}')
    store.run('report',report)


if __name__=='__main__':main()
