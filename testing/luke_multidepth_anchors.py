"""Calibrate noise-weighted event tracking on control data before target tracking."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
from scipy.signal import butter,sosfiltfilt,find_peaks
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from testing.luke_epoch_corroboration import BASE,ROOT,SHARED
OUT=ROOT/'testing/outputs/luke_multidepth_anchors_v1'
OLD=ROOT/'testing/outputs/luke_epoch_corroboration_v1'
SHIFTS=np.arange(-280,281,40)
LAGS=np.arange(-3,4)

def scores(x,events,channels,template,weights):
    """Optimize only ±3 sample timing; return weighted cosine, gain, aligned index."""
    events=np.asarray(events,dtype=int);result=np.full((len(events),3),np.nan);norm=float(np.sum(weights*template**2))
    for chunk in np.array_split(np.arange(len(events)),max(1,int(np.ceil(len(events)/250)))):
        if not len(chunk):continue
        best=np.full(len(chunk),-np.inf);gain=np.zeros(len(chunk));center=events[chunk].copy()
        for lag in LAGS:
            e=events[chunk]+lag;wave=x[e[:,None,None]+np.arange(-30,31)[None,:,None],channels[None,None,:]]
            dot=np.einsum('ntc,tc->n',wave,weights*template);den=np.einsum('ntc,tc,ntc->n',wave,weights,wave)
            c=dot/np.sqrt(np.maximum(den*norm,1e-20));update=c>best;best[update]=c[update];gain[update]=dot[update]/norm;center[update]=e[update]
        result[chunk]=np.c_[best,gain,center]
    return result

def near(events,reference,radius):
    i=np.searchsorted(reference,events);d=np.full(len(events),np.inf)
    if len(reference):
        a=np.clip(i,0,len(reference)-1);b=np.clip(i-1,0,len(reference)-1);d=np.minimum(abs(events-reference[a]),abs(events-reference[b]))
    return d<=radius

def main():
    OUT.mkdir(exist_ok=False);rec=json.loads((BASE/'recording/rescue_recording_manifest.json').read_text());fs=rec['sampling_frequency_hz'];loc=np.array(rec['channel_locations_um']);geometry={tuple(v):i for i,v in enumerate(loc)};sos=butter(3,[300,6000],fs=fs,btype='bandpass',output='sos');raw=BASE/'recording/traces_cached_seg0.raw';before=raw.stat();bytes_read=0
    config=dict(training_templates=str(OLD/'stable_templates.npz'),template_training_s=[4080,4090],calibration_s=[4090,4100],validation_s=[4100,4110],target_s=[8160,8280],shift_grid_um=SHIFTS.tolist(),lag_samples=LAGS.tolist(),min_control_events=10,min_validation_recall=.5,max_validation_unassigned_fraction=.1,calibration_negative_quantile=.999,score_floor=.65,min_target_events=5,method='Template-energy/noise weighting; calibrated threshold from competing detections at the anchor channel; separate control validation; target IDs unused.',checkpoint='None; preserve evidence and use a new output directory for a justified restart.')
    (OUT/'settings.json').write_text(json.dumps(config,indent=2)+'\n')
    def read(start,duration):
        nonlocal bytes_read
        first=round(start*fs);n=round(duration*fs);pad=round(.05*fs)
        with raw.open('rb') as h:h.seek((first-pad)*768);data=h.read((n+2*pad)*768)
        bytes_read+=len(data);assert len(data)==(n+2*pad)*768
        x=np.frombuffer(data,dtype='<i2').reshape(-1,384).astype('float32')*rec['gain_uv_per_count'];x=sosfiltfilt(sos,x,axis=0).astype('float32');x-=np.median(x,axis=1,keepdims=True)
        return x[pad:pad+n],first
    cur=BASE/'cur/cur_output';times=np.load(cur/'spike_times.npy',mmap_mode='r').reshape(-1);clusters=np.load(cur/'spike_clusters.npy',mmap_mode='r').reshape(-1)
    def refs(cid,first,n):
        a,b=np.searchsorted(times,[first+40,first+n-40]);return np.array(times[a:b][clusters[a:b]==cid])-first
    prior=pd.read_csv(OLD/'template_qualification.csv');z=np.load(OLD/'stable_templates.npz');candidates=[];diagnostics=[]
    cal,calfirst=read(4090,10);val,valfirst=read(4100,10);noise=np.median(abs(cal-np.median(cal,axis=0)),axis=0)/.67448975
    for row in prior.itertuples():
        cid=int(row.cid);template=z[f'unit_{cid}_template'];channels=z[f'unit_{cid}_channels'];peak=int(row.peak);peakoffset=int(np.argmax(abs(template[:,np.where(channels==peak)[0][0]])))-30
        weights=template**2/(template**2+(2*noise[channels][None,:])**2+1e-20)
        truth=refs(cid,calfirst,len(cal));vtruth=refs(cid,valfirst,len(val))
        positive=scores(cal,truth,channels,template,weights)
        # Original unweighted score at sorted-event centers, diagnosing the old cutoff.
        old=[]
        for e in truth:
            wave=cal[e-30:e+31,channels];old.append(float(np.sum(wave*template)/(np.linalg.norm(wave)*np.linalg.norm(template))))
        detect=find_peaks(abs(cal[:,peak]),height=max(30,3*noise[peak]),distance=round(.0008*fs))[0]-peakoffset;detect=detect[(detect>40)&(detect<len(cal)-40)];control=scores(cal,detect,channels,template,weights);notknown=~near(control[:,2],truth,.0005*fs)
        negatives=control[notknown&(control[:,1]>=.4)&(control[:,1]<=2.5),0]
        threshold=max(.65,float(np.quantile(negatives,.999))) if len(negatives)>=100 else 1.
        vdetect=find_peaks(abs(val[:,peak]),height=max(30,3*noise[peak]),distance=round(.0008*fs))[0]-peakoffset;vdetect=vdetect[(vdetect>40)&(vdetect<len(val)-40)];vscore=scores(val,vdetect,channels,template,weights);keep=(vscore[:,0]>=threshold)&(vscore[:,1]>=.4)&(vscore[:,1]<=2.5);accepted=vscore[keep,2].astype(int)
        # NMS after lag optimization.
        accepted=np.sort(np.unique(accepted));accepted=accepted[np.r_[True,np.diff(accepted)>.0008*fs]] if len(accepted) else accepted
        recalled=near(vtruth,accepted,.0005*fs);assigned=near(accepted,vtruth,.0005*fs);recall=float(recalled.mean()) if len(vtruth) else 0;unassigned=float((~assigned).mean()) if len(accepted) else 1
        eligible=len(truth)>=10 and len(vtruth)>=10 and len(negatives)>=100 and recall>=.5 and unassigned<=.1
        diagnostics.append(dict(unit_id=cid,depth_um=row.depth,calibration_reference_events=len(truth),validation_reference_events=len(vtruth),old_positive_median_cosine=float(np.median(old)) if old else np.nan,weighted_positive_median_cosine=float(np.median(positive[:,0])) if len(positive) else np.nan,negative_detections=len(negatives),threshold=threshold,validation_accepted=len(accepted),validation_recall=recall,validation_unassigned_fraction=unassigned,qualified=eligible))
        if eligible:
            candidates.append(dict(cid=cid,depth=float(row.depth),peak=peak,channels=channels,template=template,weights=weights,threshold=threshold,peakoffset=peakoffset))
    pd.DataFrame(diagnostics).to_csv(OUT/'anchor_validation.csv',index=False);print(json.dumps({'stage':'qualification_complete','qualified':len(candidates),'candidates':len(prior)}),flush=True)
    fig,axes=plt.subplots(1,2,figsize=(11,5));d=pd.DataFrame(diagnostics);axes[0].scatter(d.old_positive_median_cosine,d.weighted_positive_median_cosine,c=d.depth_um,cmap='viridis');axes[0].plot([0,1],[0,1],color='gray');axes[0].axvline(.85,color='red',ls='--');axes[0].set(xlabel='Old median score on reference events',ylabel='Noise-weighted median score',xlim=(0,1),ylim=(0,1));axes[1].barh(d.unit_id.astype(str),d.validation_recall,color=np.where(d.qualified,'#228866','#aaaaaa'));axes[1].set(xlabel='Separate validation recall against reference labels',ylabel='Reference cluster',xlim=(0,1));fig.suptitle('Anchor qualification before target tracking\nGreen passes both recall and competing-event checks; reference labels are provisional');fig.tight_layout(rect=[0,0,1,.90]);fig.savefig(OUT/'01_anchor_qualification.png',dpi=170);fig.savefig(OUT/'01_anchor_qualification.pdf');plt.close(fig)
    del cal,val
    matches=[];summaryrows=[];field=np.load(SHARED/'native_rigid_field.npz');t=field['time_s'];native=field['physical_displacement_um'].ravel();indices=np.flatnonzero((t>=8160)&(t<8280))
    for i in indices if candidates else []:
        start=float(t[i]-1);x,first=read(start,2);sig=np.median(abs(x-np.median(x,axis=0)),axis=0)/.67448975
        detections={};found=[]
        for c in candidates:
            for shift in SHIFTS:
                mapped=np.array([geometry.get((loc[ch,0],loc[ch,1]+shift),-1) for ch in c['channels']]);peak=geometry.get((loc[c['peak'],0],loc[c['peak'],1]+shift),-1)
                if peak<0 or (mapped<0).any():continue
                if peak not in detections:detections[peak]=find_peaks(abs(x[:,peak]),height=max(30,3*sig[peak]),distance=round(.0008*fs))[0]
                events=detections[peak]-c['peakoffset'];events=events[(events>40)&(events<len(x)-40)];sc=scores(x,events,mapped,c['template'],c['weights']);keep=(sc[:,0]>=c['threshold'])&(sc[:,1]>=.4)&(sc[:,1]<=2.5)
                for score,gain,e in sc[keep]:
                    wave=x[int(e)-30:int(e)+31,mapped];energy=np.maximum(np.sum(wave**2,axis=0)-61*sig[mapped]**2,0);centroid=float(np.sum(energy*loc[mapped,1])/energy.sum()) if energy.sum()>0 else np.nan
                    found.append(dict(unit_id=c['cid'],time_s=float(t[i]),frame=int(first+e),score=float(score),gain=float(gain),shift_um=int(shift),centroid_um=centroid))
        # Resolve competing shift and identity explanations jointly, without duplicate event assignment.
        accepted=[]
        for event in sorted(found,key=lambda v:-v['score']):
            if any(abs(event['frame']-v['frame'])<=.0008*fs for v in accepted):continue
            rivals=[v for v in found if abs(event['frame']-v['frame'])<=.0008*fs and (v['unit_id']!=event['unit_id'] or v['shift_um']!=event['shift_um'])]
            event['ambiguous']=bool(rivals and event['score']-max(v['score'] for v in rivals)<.05);accepted.append(event)
        matches.extend(accepted)
        for c in candidates:
            good=[v for v in accepted if v['unit_id']==c['cid'] and not v['ambiguous'] and np.isfinite(v['centroid_um'])];values=np.array([v['centroid_um'] for v in good]);enough=len(values)>=5
            summaryrows.append(dict(unit_id=c['cid'],depth_um=c['depth'],time_s=float(t[i]),matches=len(good),status='provisional_track' if enough else 'unresolved',median_depth_um=float(np.median(values)) if enough else np.nan,q25_depth_um=float(np.quantile(values,.25)) if enough else np.nan,q75_depth_um=float(np.quantile(values,.75)) if enough else np.nan,native_physical_um=float(native[i])))
        if (i-indices[0])%10==0:print(json.dumps({'stage':'tracking','bins_complete':int(i-indices[0]+1),'bins':len(indices)}),flush=True)
    pd.DataFrame(matches).to_csv(OUT/'matches.csv',index=False);tracks=pd.DataFrame(summaryrows);tracks.to_csv(OUT/'tracks.csv',index=False)
    if candidates:
        fig,axes=plt.subplots(len(candidates),1,figsize=(12,max(4,len(candidates)*2.5)),sharex=True,squeeze=False)
        for ax,c in zip(axes[:,0],candidates):
            g=tracks[tracks.unit_id==c['cid']];good=g.status=='provisional_track';baseline=g.loc[(g.time_s<8170)&good,'median_depth_um'];offset=float(baseline.median()) if len(baseline) else c['depth'];nativebase=float(np.median(g.loc[g.time_s<8170,'native_physical_um']));ax.plot(g.time_s,g.native_physical_um-nativebase,color='gray',alpha=.6,label='Native rigid');ax.plot(g.time_s,g.median_depth_um-offset,'o-',ms=3,label='Provisional footprint');ax.fill_between(g.time_s,g.q25_depth_um-offset,g.q75_depth_um-offset,alpha=.2,label='Event IQR (not confidence interval)');ax.scatter(g.loc[~good,'time_s'],np.zeros((~good).sum()),marker='x',color='red',label='Unresolved, shown at zero for visibility');ax.set_ylabel('Displacement (µm)');ax.set_title(f"Cluster {c['cid']}, reference depth {c['depth']:.0f} µm; {good.sum()}/{len(g)} bins trackable");ax.legend(fontsize=7,ncol=2)
        axes[-1,0].set_xlabel('Seconds from recording frame zero');fig.tight_layout();fig.savefig(OUT/'02_tracks_across_depth.png',dpi=170);fig.savefig(OUT/'02_tracks_across_depth.pdf');plt.close(fig)
    assert (raw.stat().st_size,raw.stat().st_mtime_ns)==(before.st_size,before.st_mtime_ns)
    result=dict(status='completed_with_qualified_anchors' if candidates else 'qualification_failed_no_target_tracking',qualified_anchors=len(candidates),depths_um=[c['depth'] for c in candidates],target_bins=60,target_tracks=int((tracks.status=='provisional_track').sum()) if len(tracks) else 0,bytes_read=bytes_read,limitations=['Control labels are provisional; competing-event rejection does not establish neuronal identity.','Reference controls calibrate each anchor at its original depth; false matches over the expanded target search need further adjudication.','40um shift grid and descriptive centroid cannot establish calibrated fine displacement.','Missing tracks remain unresolved.'],script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
