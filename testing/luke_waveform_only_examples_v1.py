"""Cheap direct waveform inspection of supported held-out match groups."""
import json
import numpy as np
import pandas as pd
from scipy.signal import butter,sosfiltfilt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from testing.luke_waveform_only_global_v1 import OUT,BASE,OFF,save

def main():
    m=json.loads((BASE/'recording/rescue_recording_manifest.json').read_text());geo=np.array(m['channel_locations_um']);fs=m['sampling_frequency_hz'];sos=butter(3,[300,6000],fs=fs,btype='bandpass',output='sos');raw=BASE/'recording/traces_cached_seg0.raw';pad=round(.05*fs)
    z=np.load(OUT/'seed_comparison.npz');q=pd.read_csv(OUT/'training_qualified_candidates.csv');ids=q[q.selected].unit_id.tolist();events=pd.read_csv(OUT/'accepted_with_identity_conflict_audit.csv');groups=pd.read_csv(OUT/'observed_support_groups.csv');fig,axs=plt.subplots(len(ids),2,figsize=(13,2.7*len(ids)),squeeze=False,layout='constrained');rows=[];arrays={}
    events['time_bin_s']=np.floor(events.time_s/5)*5
    events['depth_patch_group_um']=events.patch_base_um
    for axpair,cid in zip(axs,ids):
        seed=z['waveforms'][np.flatnonzero(z['unit_ids']==cid)[0]];g=groups[(groups.unit_id==cid)&(groups.time_bin_s>=940)&(groups.events>=5)]
        if g.empty:
            axpair[0].text(.1,.5,f'Unit {cid}: no held-out group with ≥5 events',transform=axpair[0].transAxes);axpair[0].axis('off');axpair[1].axis('off');continue
        r=g.loc[abs(g.median_relative_um).idxmax()];e=events[(events.unit_id==cid)&(events.time_bin_s==r.time_bin_s)&(events.depth_patch_group_um==r.depth_patch_group_um)];e=e.iloc[np.linspace(0,len(e)-1,min(30,len(e)),dtype=int)];ch=np.flatnonzero((geo[:,1]>=r.depth_patch_group_um-60)&(geo[:,1]<=r.depth_patch_group_um+80));wave=[]
        for fr in e.frame:
            with raw.open('rb') as f:f.seek((int(fr)-pad)*768);buf=f.read((2*pad+1)*768)
            x=np.frombuffer(buf,dtype='<i2').reshape(-1,384).astype('float32')*m['gain_uv_per_count'];x=sosfiltfilt(sos,x,axis=0).astype('float32');x-=np.median(x,axis=1,keepdims=True);wave.append(x[pad+OFF][:,ch])
        w=np.median(wave,axis=0);cos=float(np.sum(w*seed)/(np.linalg.norm(w)*np.linalg.norm(seed)));gain=float(np.sum(w*seed)/np.sum(seed*seed));norm=np.max(abs(seed));offset=np.arange(16)[None,:]*.7
        for ax,v,title,color in zip(axpair,[seed,w/max(gain,1e-10)],[f'Unit {cid} seed',f'{r.time_bin_s:g}–{r.time_bin_s+5:g}s · {len(e)}/{int(r.events)} events · Δ{r.median_relative_um:+.1f}µm'],['#777777','#0072B2']):
            ax.plot(OFF/fs*1000,v/norm+offset,color=color,lw=.8);ax.set(title=title,xlabel='Time from aligned peak (ms)',yticks=[])
        axpair[1].set_title(axpair[1].get_title()+f'\nMedian cosine {cos:.3f}; gain {gain:.2f}')
        rows.append(dict(unit_id=cid,time_bin_s=r.time_bin_s,patch_base_um=r.depth_patch_group_um,events_total=int(r.events),events_shown=len(e),relative_um=r.median_relative_um,median_waveform_cosine=cos,gain=gain));arrays[f'unit_{cid}_waveforms']=np.array(wave);arrays[f'unit_{cid}_frames']=e.frame.to_numpy();arrays[f'unit_{cid}_channels']=ch
    fig.suptitle('Direct check: seed and held-out median relative waveform\nPer cell: largest absolute displacement among five-second/same-patch groups with ≥5 accepted events; exploratory selection')
    save(fig,'05_heldout_waveform_examples');pd.DataFrame(rows).to_csv(OUT/'waveform_examples.csv',index=False);np.savez_compressed(OUT/'waveform_examples.npz',**arrays)
if __name__=='__main__':main()
