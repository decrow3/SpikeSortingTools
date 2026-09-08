"""Sparse voltage inspection on selected reference events; no full-file scan."""
from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from testing.luke_full_reference_panel_analysis import BASE, OUT

def main():
    target=OUT/'analysis'; destination=target/'voltage'; destination.mkdir(exist_ok=False)
    manifest=json.loads((OUT/'manifest.json').read_text()); units=json.loads((target/'waveform_selection.json').read_text()); plan=np.load(target/'waveform_inspection_plan.npz')
    rec=json.loads((BASE/'recording/rescue_recording_manifest.json').read_text()); raw=BASE/'recording'/rec['recording_binary_files'][0]['name']
    before=raw.stat(); assert before.st_size==rec['expected_binary_bytes']
    nchan=rec['num_channels']; fs=rec['sampling_frequency_hz']; gain=rec['gain_uv_per_count']; waves={}; rows=[]; requests=[]
    for unit in units:
        cid=unit['unit_id']; channels=unit['channels']
        for w in manifest['windows']:
            seg=w['segment_id']; frames=plan[f'unit_{cid}_{seg}_frames']; assert np.all((frames>=np.ceil(w['anchor_start_s']*fs))&(frames<np.ceil(w['anchor_end_s']*fs)))
            key=f'unit_{cid}_{seg}'; waves[key]=np.full((len(frames),82,len(channels)),np.nan,dtype=np.float32)
            for j,frame in enumerate(frames): requests.append((int(frame),key,j,channels))
    total=0
    with raw.open('rb') as handle:
        for frame,key,j,channels in sorted(requests):
            assert 41<=frame<rec['num_samples']-41
            handle.seek((frame-41)*nchan*2); data=handle.read(82*nchan*2); assert len(data)==82*nchan*2; total+=len(data)
            voltage=np.frombuffer(data,dtype='<i2').reshape(82,nchan)[:,channels].astype('float32')*gain
            voltage-=np.median(voltage[:10],axis=0,keepdims=True)
            waves[key][j]=voltage
    assert (raw.stat().st_size,raw.stat().st_mtime_ns)==(before.st_size,before.st_mtime_ns)
    fig,axes=plt.subplots(4,3,figsize=(13,11)); colors=plt.get_cmap('tab10').colors
    for ax,unit in zip(axes.flat,units):
        cid=unit['unit_id']
        for k,w in enumerate(manifest['windows']):
            key=f"unit_{cid}_{w['segment_id']}"; snippets=waves[key]
            rows.append(dict(unit_id=cid,segment_id=w['segment_id'],sampled_events=len(snippets),peak_channel=unit['peak_channel'],peak_channel_ptp_uv=float(np.ptp(np.median(snippets,axis=0)[:,0])) if len(snippets) else np.nan))
            if len(snippets): ax.plot((np.arange(82)-41)/fs*1000,np.median(snippets,axis=0)[:,0],color=colors[k],lw=1,label=w['segment_id'])
        ax.set_title(f"Cluster {cid}, {unit['depth_um']:.0f} µm"); ax.set_xlabel('Time from detected event (ms)'); ax.set_ylabel('Voltage (µV)'); ax.axhline(0,color='gray',lw=.4)
    for ax in list(axes.flat)[len(units):]: ax.set_visible(False)
    handles=[plt.Line2D([0],[0],color=colors[k],label=w['segment_id']) for k,w in enumerate(manifest['windows'])]
    fig.legend(handles=handles,loc='lower center',ncol=2,fontsize=9)
    fig.suptitle('Reference event-triggered voltage: fixed peak channel per cluster\nUp to 32 events per anchor; absent windows have no trace',fontsize=13)
    fig.tight_layout(rect=[0,.09,1,.95]); fig.savefig(destination/'waveform_panel.png',dpi=160); plt.close(fig)
    np.savez_compressed(destination/'snippets_uv.npz',**waves); pd.DataFrame(rows).to_csv(destination/'waveform_metrics.csv',index=False)
    fits=pd.read_csv(target/'anchor_fits_120s.csv'); rates=pd.read_csv(target/'counts_30s.csv')
    fig,axes=plt.subplots(1,2,figsize=(13,5)); order=[w['segment_id'] for w in manifest['windows']]; statuses=['zero_events','insufficient_events','fit_failed','fit_at_50pct_ceiling','fit_valid']
    for ax,df,title in [(axes[0],fits,'All 710 reference clusters'),(axes[1],fits[fits.ks_good],'301 KS-good reference clusters')]:
        counts=pd.crosstab(df.segment_id,df.status).reindex(index=order,columns=statuses,fill_value=0)
        counts.plot.barh(stacked=True,ax=ax,color=['#333333','#b8bec5','#d54a43','#d6a13c','#3c968b']); ax.set_title(title); ax.set_xlabel('Number of clusters'); ax.set_ylabel(''); ax.legend().remove()
    axes[1].set_yticklabels([]); fig.legend(statuses,loc='lower center',ncol=3); fig.suptitle('Fixed 120-second anchor fits: retain every reference cluster'); fig.tight_layout(rect=[0,.12,1,.94]); fig.savefig(destination/'fit_coverage.png',dpi=160); plt.close(fig)
    byseg=[]
    for seg in order:
        f=fits[(fits.segment_id==seg)&fits.ks_good]; r=rates[(rates.segment_id==seg)&rates.ks_good]
        byseg.append(dict(segment_id=seg,good_clusters=301,zero_event_anchors=int((f['count']==0).sum()),valid_fits=int((f.status=='fit_valid').sum()),valid_fit_median_missing_pct=float(f.loc[f.status=='fit_valid','missing_pct'].median()),zero_event_30s_bins=int((r['count']==0).sum()),total_30s_bins=len(r)))
    summary=dict(status='sparse_voltage_inspection_complete',source=str(raw),source_content_receipt=rec['recording_content_sha256'],requested_bytes_read=total,snippets=len(requests),units=len(units),preprocessing='Accepted cached int16 recording: phase correction, blanking, channel interpolation already applied; gain from manifest; subtract first 10 samples median separately per channel. No added temporal filter, reference or motion correction.',sampling='At most 32 evenly spaced event ranks per anchor per selected cluster; detected events only, not missing-event evidence.',by_segment_ks_good=byseg,limitations=['No trace for a zero-event anchor; never impute a zero waveform.','Raw voltage includes background and overlapping spikes; this does not establish cross-sort identity.','Cached full-session waveforms used only for spatial selection; voltage scale taken from recording manifest.'],script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (destination/'summary.json').write_text(json.dumps(summary,indent=2)+'\n'); print(json.dumps(summary),flush=True)
if __name__=='__main__': main()
