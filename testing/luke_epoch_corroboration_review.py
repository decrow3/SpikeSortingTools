"""Validate a bounded tracking pilot and export its waveform evidence."""
import json
import numpy as np
import pandas as pd
from scipy.signal import butter,sosfiltfilt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from testing.luke_epoch_corroboration import BASE,OUT,savefig

def main():
    destination=OUT/'review';destination.mkdir(exist_ok=False)
    rec=json.loads((BASE/'recording/rescue_recording_manifest.json').read_text());fs=rec['sampling_frequency_hz'];loc=np.array(rec['channel_locations_um']);sos=butter(3,[300,6000],fs=fs,btype='bandpass',output='sos')
    matches=pd.read_csv(OUT/'template_matches.csv');plan=json.loads((OUT/'tracking_plan.json').read_text());c=matches[(matches.unit_id==478)&~matches.ambiguous&~matches.cross_template_ambiguous];waveforms={};rows=[]
    for block in plan['blocks']:
        events=c[c.block==block['name']].sort_values('frame');n=len(events);assert n>=5
        frames=events.frame.to_numpy(); selection=frames[np.linspace(0,n-1,min(64,n),dtype=int)]
        first=int(round(block['start']*fs));length=int(round(block['duration']*fs));pad=int(round(.05*fs))
        with (BASE/'recording/traces_cached_seg0.raw').open('rb') as h:h.seek((first-pad)*384*2);data=h.read((length+2*pad)*384*2)
        x=np.frombuffer(data,dtype='<i2').reshape(-1,384).astype('float32')*rec['gain_uv_per_count'];x=sosfiltfilt(sos,x,axis=0).astype('float32');x-=np.median(x,axis=1,keepdims=True);x=x[pad:pad+length]
        allwaves=x[(selection-first)[:,None]+np.arange(-30,31)];median=np.median(allwaves,axis=0);waveforms[block['name']]=median
        original=matches[(matches.unit_id==478)&(matches.block==block['name'])];rows.append(dict(block=block['name'],accepted_events=n,sampled_events=len(selection),median_score=float(events.score.median()),fraction_isi_lt1p5ms=float((np.diff(frames)<.0015*fs).mean()),median_gain=float(events.gain.median())))
    metrics=pd.DataFrame(rows);metrics.to_csv(destination/'tracked_signal_metrics.csv',index=False);np.savez_compressed(destination/'tracked_signal_median_waveforms_uv.npz',**waveforms)
    # A geometry/sign check, not an injection/recovery validation: translate a measured waveform exactly 160um.
    geometry={(float(xx),float(yy)):i for i,(xx,yy) in enumerate(loc)};channels=np.flatnonzero(np.abs(loc[:,1]-2520)<=60);mapped=np.array([geometry[(float(loc[ch,0]),float(loc[ch,1]+160))] for ch in channels]);assert np.allclose(loc[mapped]-loc[channels],[0,160]);v=waveforms['stable_validation'][:,channels];e=np.sum(v*v,axis=0);before=float(np.sum(loc[channels,1]*e)/e.sum());after=float(np.sum(loc[mapped,1]*e)/e.sum());assert np.isclose(after-before,160)
    fig,axes=plt.subplots(3,2,figsize=(10,10),sharex=True,sharey=True);depths=np.unique(loc[:,1]);mask=(depths>=2200)&(depths<=2840)
    images=[]
    for k in range(1,4):
        pair=[]
        for name in ['before','after']:
            wave=waveforms[f'jump{k}_{name}'];raster=np.stack([wave[:,loc[:,1]==y].mean(axis=1) for y in depths]);pair.append(raster[mask])
        limit=max(np.abs(x).max() for x in pair)
        for j,name in enumerate(['before','after']):
            ax=axes[k-1,j];im=ax.imshow(pair[j],origin='lower',aspect='auto',extent=[-30/fs*1000,30/fs*1000,depths[mask][0]-10,depths[mask][-1]+10],cmap='RdBu_r',vmin=-limit,vmax=limit);ax.axhline(2520,color='black',lw=.5,ls=':');ax.set_title(f"Jump {k}, {name}: {int(metrics.set_index('block').loc[f'jump{k}_{name}','accepted_events'])} events");fig.colorbar(im,ax=ax,label='µV')
        axes[k-1,0].set_ylabel('Fixed recording depth (µm)')
    for ax in axes[-1]:ax.set_xlabel('Time from matched event (ms)')
    fig.suptitle('One trackable signal retains its localized footprint across all three jumps\nProvisional reference template 478; fixed channels, up to 64 events per median');fig.tight_layout(rect=[0,0,1,.94]);savefig(fig,'04_tracked_waveform_evidence')
    summary=dict(status='bounded_pilot_reviewed',matched_signal=478,paired_regions=3,geometry_translation_control_um=after-before,validation_scope='Exact geometry/sign check and event/waveform reconstruction; not validated single-neuron identity or realistic injection/recovery sensitivity.',stable_event_tracking_templates=5,unresolved_in_stable_control_templates=3,target_pair_trackable_templates=1,limitations=['Only one signal is trackable on both sides; no full-depth physical motion estimate is established.','The selected stable interval is over an hour before the target, which can reduce template transfer.','Multiple templates fail stable-control event matching despite similar averaged waveforms.','Untracked signals remain unresolved; do not count them as stationary.','No full-sort or new motion estimator run.'])
    (destination/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
