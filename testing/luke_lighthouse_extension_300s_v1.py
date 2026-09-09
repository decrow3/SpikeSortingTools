"""Frozen whole-probe lighthouse tracking extension,1030–1230s; no motion priors."""
from testing.luke_waveform_only_global_v1 import *
from testing.luke_ap_methods_sweep_v1 import sha
OUT=ROOT/'testing/outputs/luke_lighthouse_extension_300s_v1'
FROZEN=ROOT/'testing/outputs/luke_waveform_only_global_v1'
COHORT=ROOT/'testing/outputs/luke_waveform_only_expansion_v1'
def extract(state):
    m,geo,bases,patches,d,w,normalized,norms=state;fs=m['sampling_frequency_hz'];raw=BASE/'recording/traces_cached_seg0.raw';rawstat=raw.stat();sos=butter(3,[300,6000],fs=fs,btype='bandpass',output='sos');ids=d.unit_id.to_numpy();selected=set(d[d.selected].unit_id)
    # Explicit geometry sign check: a synthetic translated seed has unchanged features.
    for b,ch in zip(bases,patches):
        synthetic=np.zeros((49,384),dtype='float32');synthetic[:,ch]=w[0]
        assert np.allclose(synthetic[:,ch].ravel()@normalized[0]/norms[0],1,atol=1e-5)
        assert abs(((w[0]**2).sum(axis=0)@geo[ch,1]/(w[0]**2).sum())-(d.seed_centroid_um[0]+b-d.seed_base_um[0]))<.002
    (OUT/'geometry_checks.json').write_text(json.dumps(dict(patches=len(bases),identity_invariant=True,centroid_sign_verified=True)))
    for start in range(1030,1230,5):
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

def main():
    OUT.mkdir(exist_ok=True)
    mp=BASE/'recording/rescue_recording_manifest.json';m=json.loads(mp.read_text());geo=np.array(m['channel_locations_um'])
    bases=np.arange(80,int(geo[:,1].max())-79,40);patches=np.array([np.flatnonzero((geo[:,1]>=b-60)&(geo[:,1]<=b+80)) for b in bases]);assert patches.shape[1]==16
    rel=geo[patches]-np.stack([np.zeros(len(bases)),bases],axis=1)[:,None,:];assert np.all(rel==rel[0])
    bank=np.load(FROZEN/'seed_comparison.npz');d=pd.read_csv(FROZEN/'seed_distinctness.csv');c=pd.read_csv(COHORT/'candidate_audit.csv');chosen=c[c.selected].unit_id.to_numpy();assert len(chosen)==17
    assert np.array_equal(bank['unit_ids'],d.unit_id) and np.array_equal(bank['relative_geometry'],rel[0]);d['selected']=d.unit_id.isin(chosen)
    w=bank['waveforms'];flat=w.reshape(len(w),-1);norms=np.linalg.norm(flat,axis=1);normalized=flat/norms[:,None]
    paths=[Path(__file__),ROOT/'testing/luke_lighthouse_extension_300s_report_v1.py',FROZEN/'seed_comparison.npz',FROZEN/'seed_distinctness.csv',COHORT/'candidate_audit.csv',COHORT/'overlay_events.csv',mp]
    cfg=dict(interval_s=[1030,1230],seed_interval_s=[930,940],candidate_ids=chosen.tolist(),competitors=len(d),matching='Exact frozen matcher; all complete whole-probe16channel patches;49samples,lag+-3; no depth or motion priors',acceptance='strict score>=.86 margin>=.025 gain.35–3; lower score.80–.86 margin>=.025; ambiguous score>=.80 margin<.025; preserve all detections and competitor scores',preprocessing='Original bandpassed globally median-referenced voltage; detector unchanged,5second chunks',restart='Sealed5s chunks hash validated; interrupted chunk must restart after investigation; no within-chunk resume',source_sha256={str(p):sha(p) for p in paths})
    sp=OUT/'settings.json'
    if sp.exists():assert json.loads(sp.read_text())==cfg
    else:sp.write_text(json.dumps(cfg,indent=2))
    extract((m,geo,bases,patches,d,w,normalized,norms))
    assert cfg['source_sha256']=={str(p):sha(p) for p in paths}
    from testing.luke_lighthouse_extension_300s_report_v1 import report
    report()
if __name__=='__main__':main()
