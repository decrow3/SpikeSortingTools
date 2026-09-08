"""Existing-field alignment and bounded voltage footprint corroboration; no sort."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import butter,sosfiltfilt,find_peaks
ROOT=Path(__file__).resolve().parents[1]; DATA=Path('/mnt/NPX/Luke/20250804')
BASE=DATA/'rescue_pipeline_results_Luke0804_V2V1_g0_imec0'
OUT=ROOT/'testing/outputs/luke_epoch_corroboration_local_v2'
SHARED=DATA/'shared_analysis/luke_next_stage_coordination_20260907_v1/huklaban5_registration_v1/testing/outputs/luke_full_probe_rigid_registration_v1'
SWEEP=DATA/'dredge_pipeline_results_Luke0804_V2V1_g0_imec1/motion_scale_sweep/runs'

def savefig(fig,name):
    fig.savefig(OUT/(name+'.png'),dpi=170,bbox_inches='tight');fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight');plt.close(fig)

def main():
    assert not (OUT/'summary.json').exists()
    sources={}
    def load(path):
        sources[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest();return np.load(path)
    field=load(SHARED/'native_rigid_field.npz'); t=field['time_s']; physical=field['physical_displacement_um'].ravel(); origin=3057.677050340359
    series=[]
    def add(name,tt,v,depth,kind):
        mask=(tt>=8160)&(tt<8280); times=tt[mask]; values=v[mask];assert len(times)
        # Same overlapping baseline interval, never optimize offset against another field.
        baseline=values[(times>=8160)&(times<8170)]; assert len(baseline)
        offset=float(np.median(baseline));series.append(dict(name=name,time=times,value=values-offset,offset=offset,depth=depth,kind=kind))
    add('Native KS rigid',t,physical,1909,'native')
    for est in ['dredge-motion','decentralized-motion','ks-motion']:
        p=DATA/'dredge_pipeline_results_Luke0804_V2V1_g0_imec0/motion'/est
        tt=load(p/'time_bins.npy').ravel()-origin;yy=load(p/'depth_bins.npy').ravel();mm=load(p/'motion.npy')
        for depth in [1000,1909,3000]:add(est,tt,np.array([np.interp(depth,yy,row) for row in mm]),depth,'AP')
    for f in sorted((SWEEP/'imec0').glob('*lfp*/full_*/manifest.json')):
        m=json.loads(f.read_text());p=f.parent;tt=load(p/'time_bins.npy').ravel()-origin;v=load(p/'motion.npy').ravel();add(m['candidate']['name'],tt,v,1909,'LFP')
    fig,axes=plt.subplots(3,1,figsize=(12,9),sharex=True)
    for ax,depth in zip(axes,[1000,1909,3000]):
        for s in series:
            if s['kind']=='native' or s['depth']==depth: ax.plot(s['time'],s['value'],label=s['name'],lw=.9,alpha=.85)
        ax.set_ylabel('Displacement (µm)');ax.set_title(f'imec0: local AP estimates at {depth} µm; native rigid is global');ax.axhline(0,color='gray',lw=.5);ax.legend(fontsize=8,ncol=2)
    axes[-1].set_xlabel('Seconds from recording frame zero');fig.suptitle('Existing estimates disagree during the same epoch\nEach trace referenced to its own median over 8160–8170 s; no sign/lag optimization');fig.tight_layout(rect=[0,0,1,.94]);savefig(fig,'01_existing_fields')
    pd.DataFrame([dict(method=s['name'],depth_um=s['depth'],time_s=tt,displacement_um=v,baseline_offset_um=s['offset']) for s in series for tt,v in zip(s['time'],s['value'])]).to_csv(OUT/'aligned_fields.csv',index=False)
    band=DATA/'dredge_pipeline_results_Luke0804_V2V1_g0_imec1/motion_estimator_band_ablation'
    fig,axes=plt.subplots(1,2,figsize=(11,4));peak_summary=[]
    for name,color in [('ap_300_3000','tab:blue'),('ap_300_6000','tab:orange')]:
        p=band/'estimators'/name; peaks=load(p/'peaks.npy');amp=np.abs(peaks['amplitude']); peak_summary.append(dict(branch=name,peaks=len(peaks),median_saved_amplitude=float(np.median(amp)),p95_saved_amplitude=float(np.quantile(amp,.95))))
        axes[0].bar(name,len(peaks),color=color);axes[1].hist(amp,bins=np.linspace(0,np.quantile(amp,.99),80),density=True,histtype='step',label=name,color=color)
    axes[0].set_ylabel('Detected peaks');axes[1].set_xlabel('Absolute saved amplitude (extractor units)');axes[1].set_ylabel('Density');axes[1].legend();fig.suptitle('Historical imec1 band comparison, 8160–8220 s\nSame nominal detection threshold; not current imec0 native detections');fig.tight_layout(rect=[0,0,1,.88]);savefig(fig,'02_historical_peak_population');pd.DataFrame(peak_summary).to_csv(OUT/'historical_peak_population.csv',index=False)
    print('Existing-field and historical-peak figures complete',flush=True)
    rec=json.loads((BASE/'recording/rescue_recording_manifest.json').read_text());fs=rec['sampling_frequency_hz'];loc=np.array(rec['channel_locations_um']);raw=BASE/'recording/traces_cached_seg0.raw';before=raw.stat();sos=butter(3,[300,6000],fs=fs,btype='bandpass',output='sos');read_bytes=0
    def voltage(start,duration):
        nonlocal read_bytes
        first=int(round(start*fs));n=int(round(duration*fs));pad=int(round(.05*fs));lo=first-pad
        with raw.open('rb') as h:h.seek(lo*384*2); data=h.read((n+2*pad)*384*2)
        read_bytes+=len(data);assert len(data)==(n+2*pad)*384*2
        x=np.frombuffer(data,dtype='<i2').reshape(-1,384).astype('float32')*rec['gain_uv_per_count'];x=sosfiltfilt(sos,x,axis=0).astype('float32');x-=np.median(x,axis=1,keepdims=True)
        return x[pad:pad+n],first
    # Qualification uses only stable-control data, before target snippets are read.
    train,first=voltage(8160,10)
    cur=BASE/'cur/cur_output';ts=load(cur/'spike_times.npy').reshape(-1);cl=load(cur/'spike_clusters.npy').reshape(-1); labels=pd.read_csv(cur/'cluster_KSLabel.tsv',sep='\t');candidates=[]
    for cid in labels.loc[labels.KSLabel=='good','cluster_id']:
        events=ts[(cl==cid)&(ts>=first+31)&(ts<first+len(train)-31)]-first
        a=events[events<5*fs];b=events[events>=5*fs]
        if min(len(a),len(b))<10:continue
        a=a[np.linspace(0,len(a)-1,min(100,len(a)),dtype=int)];b=b[np.linspace(0,len(b)-1,min(100,len(b)),dtype=int)]
        wa=np.median(train[a[:,None]+np.arange(-30,31)],axis=0);wb=np.median(train[b[:,None]+np.arange(-30,31)],axis=0);peak=int(np.ptp(wa,axis=0).argmax());depth=loc[peak,1]
        if not 320<=depth<=3500:continue
        channels=np.flatnonzero(np.abs(loc[:,1]-depth)<=60);v=wa[:,channels];vv=wb[:,channels];corr=float(np.sum(v*vv)/np.sqrt(np.sum(v*v)*np.sum(vv*vv)));energy=(wa**2).sum(axis=0);compact=float(energy[channels].sum()/energy.sum());amplitude=float(np.max(np.abs(v)))
        if corr<.9 or compact<.65 or amplitude<70:continue
        # Align template's largest absolute extremum to temporal center.
        trough=int(np.argmax(np.abs(wa[:,peak])));shift=trough-30
        if abs(shift)>5:continue
        candidates.append(dict(cid=int(cid),peak=peak,depth=float(depth),corr=corr,compact=compact,amplitude=amplitude,channels=channels,template=v,shift=shift,validation_events=len(b)))
    selected=[]
    for lo,hi in zip([320,1100,1900,2700],[1100,1900,2700,3501]):
        pool=sorted([c for c in candidates if lo<=c['depth']<hi],key=lambda c:-c['amplitude'])
        for c in pool:
            if all(abs(c['depth']-d['depth'])>=80 for d in selected):selected.append(c)
            if sum(lo<=d['depth']<hi for d in selected)>=2:break
    assert len(selected)>=2,'Insufficient qualified stable-control templates; inspect evidence before extending'
    template_payload={};qual=[]
    for c in selected:
        template_payload[f"unit_{c['cid']}_template"]=c['template'];template_payload[f"unit_{c['cid']}_channels"]=c['channels'];qual.append({k:v for k,v in c.items() if k not in ['channels','template']})
    np.savez_compressed(OUT/'stable_templates.npz',**template_payload);pd.DataFrame(qual).to_csv(OUT/'template_qualification.csv',index=False);print(json.dumps({'qualified_candidates':len(candidates),'selected_templates':len(selected)}),flush=True)
    del train
    epoch=np.flatnonzero((t>=8162)&(t<=8278));large=epoch[np.argsort(-np.abs(np.diff(physical,prepend=physical[0]))[epoch])];chosen=[]
    for i in large:
        if all(abs(t[i]-t[j])>12 for j in chosen):chosen.append(int(i))
        if len(chosen)==3:break
    blocks=[dict(name='stable_validation',start=8165.,duration=5.,native_relative=0.)]
    for k,i in enumerate(chosen):
        for label,j in [('before',i-1),('after',i)]:blocks.append(dict(name=f'jump{k+1}_{label}',start=float(t[j]-1),duration=2.,native_relative=float(physical[j]-physical[chosen[k]-1]),pair=k+1,batch=j))
    (OUT/'tracking_plan.json').write_text(json.dumps(dict(blocks=blocks,shifts_um=list(range(-280,281,40)),method='Template shape matching on independently detected extrema; exact geometry translations every 40um, plus waveform energy centroid. No target cluster IDs used. Thresholds exploratory; not validated neuron identities.'),indent=2)+'\n')
    matches=[];profiles={}
    shifts=list(range(-280,281,40));geometry={(float(x),float(y)):i for i,(x,y) in enumerate(loc)}
    for block in blocks:
        x,start=voltage(block['start'],block['duration']);noise=np.median(np.abs(x-np.median(x,axis=0)),axis=0)/.67448975
        # Independent detections on every channel, both polarities, no reference-cluster gating.
        detections={ch:find_peaks(np.abs(x[:,ch]),height=max(50,5*float(noise[ch])),distance=int(.0008*fs))[0] for ch in range(384)}
        for c in selected:
            template=c['template']; channels=c['channels'];norm=np.linalg.norm(template);peakoffset=int(np.argmax(np.abs(template[:,np.where(channels==c['peak'])[0][0]])))-30
            found=[]
            for shift in shifts:
                mapped=np.array([geometry.get((float(loc[ch,0]),float(loc[ch,1]+shift)),-1) for ch in channels]);pch=geometry.get((float(loc[c['peak'],0]),float(loc[c['peak'],1]+shift)),-1)
                if pch<0 or (mapped<0).any():continue
                events=detections[pch]-peakoffset;events=events[(events>=31)&(events<len(x)-31)]
                for chunk in np.array_split(events,max(1,int(np.ceil(len(events)/500)))):
                    if not len(chunk):continue
                    waves=x[chunk[:,None,None]+np.arange(-30,31)[None,:,None],mapped[None,None,:]];dot=np.einsum('ntc,tc->n',waves,template);corr=dot/(np.linalg.norm(waves,axis=(1,2))*norm+1e-20);gain=dot/(norm**2)
                    for ii in np.flatnonzero((corr>=.85)&(gain>=.4)&(gain<=2.5)):
                        energy=np.sum(waves[ii]**2,axis=0);center=float(np.sum(loc[mapped,1]*energy)/energy.sum());found.append(dict(frame=int(start+chunk[ii]),score=float(corr[ii]),gain=float(gain[ii]),shift_um=shift,centroid_um=center))
            # Nonmax suppression across shift candidates; report runner-up shifts as ambiguity.
            accepted=[]
            for event in sorted(found,key=lambda e:-e['score']):
                if all(abs(event['frame']-v['frame'])>int(.0008*fs) for v in accepted):accepted.append(event)
            for event in accepted:
                rivals=[v['score'] for v in found if abs(v['frame']-event['frame'])<=int(.0008*fs) and v['shift_um']!=event['shift_um']]
                event.update(unit_id=c['cid'],block=block['name'],runner_up_score=max(rivals) if rivals else np.nan,ambiguous=bool(rivals and event['score']-max(rivals)<.05));matches.append(event)
        print(json.dumps({'block_complete':block['name'],'matches_so_far':len(matches)}),flush=True)
    m=pd.DataFrame(matches);m.to_csv(OUT/'template_matches.csv',index=False)
    # Prevent one detected event from being claimed by multiple template identities.
    m['cross_template_ambiguous']=False
    for block,g in m.groupby('block'):
        idx=g.sort_values('frame').index.to_numpy(); frames=m.loc[idx,'frame'].to_numpy();units=m.loc[idx,'unit_id'].to_numpy()
        for j in range(len(idx)):
            k=j+1
            while k<len(idx) and frames[k]-frames[j]<=.0008*fs:
                if units[k]!=units[j]:m.loc[[idx[j],idx[k]],'cross_template_ambiguous']=True
                k+=1
    m.to_csv(OUT/'template_matches.csv',index=False);summaries=[]
    for c in selected:
        for block in blocks:
            rows=m[(m.unit_id==c['cid'])&(m.block==block['name'])]; valid=rows[~rows.ambiguous&~rows.cross_template_ambiguous]
            summaries.append(dict(unit_id=c['cid'],depth_um=c['depth'],block=block['name'],matches=len(rows),unambiguous_matches=len(valid),median_shift_um=float(valid.shift_um.median()) if len(valid)>=5 else np.nan,median_centroid_um=float(valid.centroid_um.median()) if len(valid)>=5 else np.nan,status='provisional_track' if len(valid)>=5 else 'unresolved'))
    sums=pd.DataFrame(summaries);sums.to_csv(OUT/'tracking_summary.csv',index=False)
    fig,axes=plt.subplots(1,3,figsize=(13,4),sharey=True);comparisons=[]
    for k,i in enumerate(chosen):
        ax=axes[k];a=sums[sums.block==f'jump{k+1}_before'].set_index('unit_id');b=sums[sums.block==f'jump{k+1}_after'].set_index('unit_id');valid=(a.status=='provisional_track')&(b.status=='provisional_track');delta=b.loc[valid,'median_centroid_um']-a.loc[valid,'median_centroid_um'];depth=a.loc[valid,'depth_um'];native=float(physical[i]-physical[i-1]);ax.axvline(native,color='red',label='Native predicted step');ax.axvline(0,color='gray',lw=.7);ax.scatter(delta,depth,color='black',label='Provisional footprint step');ax.set_xlim(-220,220);ax.set_ylim(0,3820);ax.set_title(f'Jump {k+1}: {t[i]:.1f} s\n{valid.sum()}/{len(selected)} templates trackable both sides');ax.set_xlabel('After − before displacement (µm)');ax.legend(fontsize=7)
        for cid,value in delta.items():comparisons.append(dict(pair=k+1,unit_id=int(cid),footprint_step_um=float(value),native_step_um=native))
    axes[0].set_ylabel('Stable-template depth (µm)');fig.suptitle('Exploratory footprint corroboration — unresolved templates excluded, not stationary\n40 µm translation grid; centroid is descriptive, without a calibrated uncertainty model');fig.tight_layout(rect=[0,0,1,.86]);savefig(fig,'03_footprint_steps');pd.DataFrame(comparisons).to_csv(OUT/'step_comparisons.csv',index=False)
    assert (raw.stat().st_size,raw.stat().st_mtime_ns)==(before.st_size,before.st_mtime_ns)
    result=dict(status='bounded_existing_data_and_footprint_pilot_complete',selected_templates=len(selected),qualified_candidates=len(candidates),matches=len(m),paired_track_observations=len(comparisons),raw_bytes_read=read_bytes,source_hashes=sources,preprocessing='Accepted conditioned reference, 300–6000Hz 3rd-order Butterworth forward-backward with 50ms margins, then global median reference; µV gain from verified manifest.',limitations=['Template qualification is stable-period consistency, not biological identity proof.','Target tracks use independently detected events but template matches can still confuse similar neurons.','Exact 40um geometry shifts do not resolve arbitrary fine displacement; centroid changes are descriptive.','Only selected strong templates and three events; no population recovery conclusion.','Historical band comparison is imec1, not imec0.','LFP preprocessing/provenance must be reviewed before treating its field as physical truth.'])
    (OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='source_hashes'}),flush=True)
if __name__=='__main__':main()
