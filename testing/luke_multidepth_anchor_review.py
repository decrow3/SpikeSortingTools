"""Full-search held-out validation and figures for multidepth anchor tracking."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.signal import butter,sosfiltfilt,find_peaks
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from testing.luke_multidepth_anchors_v2 import OUT,OLD,BASE,SHARED,scores,near,SHIFTS

def main():
    destination=OUT/'review';destination.mkdir(exist_ok=False)
    rec=json.loads((BASE/'recording/rescue_recording_manifest.json').read_text());fs=rec['sampling_frequency_hz'];loc=np.array(rec['channel_locations_um']);geometry={tuple(v):i for i,v in enumerate(loc)};sos=butter(3,[300,6000],fs=fs,btype='bandpass',output='sos')
    def read(start,duration):
        first=round(start*fs);n=round(duration*fs);pad=round(.05*fs)
        with (BASE/'recording/traces_cached_seg0.raw').open('rb') as h:h.seek((first-pad)*768);data=h.read((n+2*pad)*768)
        x=np.frombuffer(data,dtype='<i2').reshape(-1,384).astype('float32')*rec['gain_uv_per_count'];x=sosfiltfilt(sos,x,axis=0).astype('float32');x-=np.median(x,axis=1,keepdims=True);return x[pad:pad+n],first
    cal,_=read(4090,10);noise=np.median(abs(cal-np.median(cal,axis=0)),axis=0)/.67448975;del cal
    x,first=read(4100,10);sig=np.median(abs(x-np.median(x,axis=0)),axis=0)/.67448975;d=pd.read_csv(OUT/'anchor_validation.csv');prior=pd.read_csv(OLD/'template_qualification.csv').set_index('cid');z=np.load(OLD/'stable_templates.npz');found=[];detections={}
    for row in d[d.qualified].itertuples():
        cid=row.unit_id;channels=z[f'unit_{cid}_channels'];template=z[f'unit_{cid}_template'];weights=template**2/(template**2+(2*noise[channels][None,:])**2+1e-20);peak=int(prior.loc[cid,'peak']);offset=int(np.argmax(abs(template[:,np.where(channels==peak)[0][0]])))-30
        for shift in SHIFTS:
            mapped=np.array([geometry.get((loc[ch,0],loc[ch,1]+shift),-1) for ch in channels]);pch=geometry.get((loc[peak,0],loc[peak,1]+shift),-1)
            if pch<0 or (mapped<0).any():continue
            if pch not in detections:detections[pch]=find_peaks(abs(x[:,pch]),height=max(30,3*sig[pch]),distance=round(.0008*fs))[0]
            events=detections[pch]-offset;events=events[(events>40)&(events<len(x)-40)];sc=scores(x,events,mapped,template,weights);keep=(sc[:,0]>=row.threshold)&(sc[:,1]>=.4)&(sc[:,1]<=2.5)
            for score,gain,e in sc[keep]:found.append(dict(unit_id=int(cid),frame=int(first+e),score=float(score),shift_um=int(shift)))
    accepted=[]
    for event in sorted(found,key=lambda v:-v['score']):
        if any(abs(event['frame']-v['frame'])<=.0008*fs for v in accepted):continue
        rivals=[v for v in found if abs(event['frame']-v['frame'])<=.0008*fs and (v['unit_id']!=event['unit_id'] or v['shift_um']!=event['shift_um'])];event['ambiguous']=bool(rivals and event['score']-max(v['score'] for v in rivals)<.05);accepted.append(event)
    control=pd.DataFrame(accepted);control.to_csv(destination/'full_search_control_events.csv',index=False)
    ts=np.load(BASE/'cur/cur_output/spike_times.npy',mmap_mode='r').reshape(-1);cl=np.load(BASE/'cur/cur_output/spike_clusters.npy',mmap_mode='r').reshape(-1);a,b=np.searchsorted(ts,[first+40,first+len(x)-40]);rows=[]
    for row in d[d.qualified].itertuples():
        truth=ts[a:b][cl[a:b]==row.unit_id];hits=control[(control.unit_id==row.unit_id)&~control.ambiguous];frames=np.sort(hits.frame.to_numpy());recall=float(near(truth,frames,.0005*fs).mean());extra=float((~near(frames,truth,.0005*fs)).mean()) if len(frames) else 1;passed=recall>=.5 and extra<=.1
        rows.append(dict(unit_id=row.unit_id,depth_um=row.depth_um,reference_events=len(truth),full_search_accepted=len(frames),full_search_recall=recall,full_search_unassigned_fraction=extra,full_search_pass=passed,nonzero_shift_events=int((hits.shift_um!=0).sum())))
    validation=pd.DataFrame(rows);validation.to_csv(destination/'full_search_validation.csv',index=False);print(validation.to_json(orient='records'),flush=True)
    # A larger target search must pass control before appearing in the primary trajectory figure.
    tracks=pd.read_csv(OUT/'tracks.csv');validids=validation.loc[validation.full_search_pass,'unit_id'].tolist();matches=pd.read_csv(OUT/'matches.csv');metrics=[]
    for cid in validids:
        g=tracks[tracks.unit_id==cid];hits=matches[(matches.unit_id==cid)&~matches.ambiguous].sort_values('frame');interval=np.diff(hits.frame)/fs;metrics.append(dict(unit_id=int(cid),depth_um=float(g.depth_um.iloc[0]),trackable_bins=int((g.status=='provisional_track').sum()),total_bins=len(g),accepted_events=len(hits),fraction_isi_lt1p5ms=float(np.mean(interval<.0015)),nonzero_shift_events=int((hits.shift_um!=0).sum())))
    pd.DataFrame(metrics).to_csv(destination/'target_coverage.csv',index=False)
    fig,axes=plt.subplots(2,1,figsize=(12,7),sharex=True,gridspec_kw={'height_ratios':[3,1]});field=np.load(SHARED/'native_rigid_field.npz');mask=(field['time_s']>=8160)&(field['time_s']<8280);t=field['time_s'][mask];native=field['physical_displacement_um'].ravel()[mask];axes[0].plot(t,native-np.median(native[t<8170]),color='gray',lw=1,label='Native global rigid')
    for j,cid in enumerate(validids):
        g=tracks[tracks.unit_id==cid];baseline=g.loc[(g.time_s<8170)&(g.status=='provisional_track'),'median_depth_um'];offset=float(baseline.median()) if len(baseline) else np.nan;color=plt.get_cmap('tab10')(j);axes[0].plot(g.time_s,g.median_depth_um-offset,'o-',ms=3,color=color,label=f"Cluster {cid}, {g.depth_um.iloc[0]:.0f} µm");good=g.status=='provisional_track';axes[1].scatter(g.loc[good,'time_s'],np.full(good.sum(),j),color=color,s=16);axes[1].scatter(g.loc[~good,'time_s'],np.full((~good).sum(),j),color='lightgray',marker='x',s=16)
    axes[0].set_ylabel('Relative displacement (µm)');axes[0].legend(fontsize=8,ncol=2);axes[0].axhline(0,color='black',lw=.4);axes[1].set_yticks(range(len(validids)),[str(c) for c in validids]);axes[1].set_ylabel('Cluster');axes[1].set_xlabel('Seconds from recording frame zero');axes[1].set_title('Coverage: colored = ≥5 unambiguous events; gray × = unresolved');fig.suptitle('Control-qualified footprint trajectories across depth\nEach footprint referenced to its first-10-second median; gaps are not interpolated');fig.tight_layout(rect=[0,0,1,.91]);fig.savefig(destination/'01_combined_tracks.png',dpi=170);fig.savefig(destination/'01_combined_tracks.pdf');plt.close(fig)
    plan=json.loads((OLD/'tracking_plan.json').read_text());pairs=[]
    for pair in [1,2,3]:
        before=next(v for v in plan['blocks'] if v['name']==f'jump{pair}_before');after=next(v for v in plan['blocks'] if v['name']==f'jump{pair}_after')
        for cid in validids:
            g=tracks[tracks.unit_id==cid];a=g.iloc[np.argmin(abs(g.time_s-(before['start']+1)))];b=g.iloc[np.argmin(abs(g.time_s-(after['start']+1)))];ok=a.status==b.status=='provisional_track';pairs.append(dict(pair=pair,unit_id=cid,depth_um=a.depth_um,before_count=int(a.matches),after_count=int(b.matches),status='paired' if ok else 'unresolved',footprint_step_um=float(b.median_depth_um-a.median_depth_um) if ok else np.nan,native_step_um=after['native_relative']))
    pd.DataFrame(pairs).to_csv(destination/'paired_steps.csv',index=False)
    result=dict(status='review_complete',control_qualified_anchors=len(validids),control_qualified_ids=validids,depths_um=validation.loc[validation.full_search_pass,'depth_um'].tolist(),target_coverage=metrics,paired_observations=sum(v['status']=='paired' for v in pairs),limitations=['Control matching uses existing curated labels; this is evidence of consistency, not proven biological identity.','No known-motion injection calibration yet; centroid precision and fine motion remain uncalibrated.','A repeated biological waveform cannot be distinguished conclusively from a localized artifact by these checks alone.','Agreement across a few selected depth locations does not establish whole-probe rigidity or full-session recovery.'])
    (destination/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
