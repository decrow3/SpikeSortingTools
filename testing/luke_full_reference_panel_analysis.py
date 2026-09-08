"""Read-only fixed-window reference diagnostics; no sorting or input changes."""
from pathlib import Path
import contextlib, io, json, hashlib, warnings
import numpy as np
import pandas as pd
from pipeline.truncation import fit_amp_cdf, is_saturated
ROOT=Path(__file__).resolve().parents[1]
BASE=Path('/mnt/NPX/Luke/20250804/rescue_pipeline_results_Luke0804_V2V1_g0_imec0')
OUT=ROOT/'testing/outputs/luke_full_reference_diagnostic_panel_v1'

def classify_fit(amps):
    if not len(amps): return 'zero_events', np.nan, ''
    if len(amps)<1000: return 'insufficient_events', np.nan, ''
    stream=io.StringIO()
    try:
        with contextlib.redirect_stdout(stream), warnings.catch_warnings():
            warnings.simplefilter('ignore')
            pars, pct=fit_amp_cdf(amps)
        if stream.getvalue() or not np.isfinite(pars).all() or not np.isfinite(pct):
            return 'fit_failed',np.nan,stream.getvalue().strip()
        if is_saturated(pct): return 'fit_at_50pct_ceiling',float(pct),''
        return 'fit_valid',float(pct),''
    except Exception as exc: return 'fit_failed',np.nan,str(exc)

def main():
    target=OUT/'analysis'; target.mkdir(exist_ok=False)
    manifest=json.loads((OUT/'manifest.json').read_text())
    audit=json.loads((ROOT/'testing/outputs/luke_full_reference_reuse_audit_v1/summary.json').read_text())
    assert audit['status']=='reference_reusable' and manifest['reference_identity']==audit['sort_identity']['identity_digest']
    rec=json.loads((BASE/'recording/rescue_recording_manifest.json').read_text())
    fs=manifest['sampling_frequency_hz']; ids=np.array(manifest['reference_unit_ids']); good=set(manifest['ks_good_ids'])
    cur=BASE/'cur/cur_output'
    ts=np.load(cur/'spike_times.npy',mmap_mode='r').reshape(-1)
    cl=np.load(cur/'spike_clusters.npy',mmap_mode='r').reshape(-1)
    full=np.load(cur/'full_st.npy',mmap_mode='r')
    kept=np.load(cur/'kept_spikes.npy',mmap_mode='r'); indices=np.flatnonzero(kept) if kept.dtype.kind=='b' else kept
    assert len(ts)==len(cl)==len(indices)==audit['curated_spikes']
    rates=[]; fits=[]; selected_events={}; anchor_counts=[]
    for w in manifest['windows']:
        seg=w['segment_id']; edges=np.ceil(np.arange(w['start_s'],w['end_s']+1,30)*fs).astype('int64')
        for lo,hi in zip(edges[:-1],edges[1:]):
            a,b=np.searchsorted(ts,[lo,hi]); counts=np.bincount(np.searchsorted(ids,cl[a:b]),minlength=len(ids))
            assert counts.sum()==b-a
            rates.extend(dict(segment_id=seg,unit_id=int(cid),ks_good=int(cid) in good,start_frame=int(lo),end_frame=int(hi),count=int(n),rate_hz=float(n*fs/(hi-lo))) for cid,n in zip(ids,counts))
        lo,hi=np.ceil(np.array([w['anchor_start_s'],w['anchor_end_s']])*fs).astype('int64'); a,b=np.searchsorted(ts,[lo,hi])
        cc=np.asarray(cl[a:b]); tt=np.asarray(ts[a:b]); amps=np.asarray(full[indices[a:b],2]); counts=[]
        assert np.array_equal(full[indices[a:b],0],tt)
        for cid in ids:
            take=cc==cid; amp=amps[take]; t=tt[take]; status,pct,error=classify_fit(amp)
            maxgap=float(np.diff(t).max()/fs) if len(t)>1 else np.nan
            fits.append(dict(segment_id=seg,unit_id=int(cid),ks_good=int(cid) in good,start_frame=int(lo),end_frame=int(hi),count=len(t),status=status,missing_pct=pct,median_sort_amplitude=float(np.median(amp)) if len(amp) else np.nan,max_internal_gap_s=maxgap,error=error))
            selected_events[(seg,int(cid))]=t[np.linspace(0,len(t)-1,min(32,len(t)),dtype=int)] if len(t) else t
            counts.append(len(t))
        assert sum(counts)==b-a
        anchor_counts.append(counts)
        print(json.dumps({'stage':'anchor_complete','segment':seg}),flush=True)
    rates=pd.DataFrame(rates); fits=pd.DataFrame(fits)
    assert len(rates)==6*20*710 and len(fits)==6*710
    rates.to_csv(target/'counts_30s.csv',index=False); fits.to_csv(target/'anchor_fits_120s.csv',index=False)
    coverage=fits.groupby(['segment_id','ks_good','status']).size().rename('unit_count').reset_index(); coverage.to_csv(target/'fit_coverage.csv',index=False)
    z=np.load(BASE/'qc/waveforms/waveforms.npz'); cids=z['cids']; waves=z['waveforms']; assert np.array_equal(cids,ids)
    loc=np.array(rec['channel_locations_um']); peak=np.ptp(waves,axis=1).argmax(axis=1); depth=loc[peak,1]
    counts=np.array(anchor_counts); mean=counts.mean(axis=0); variability=counts.std(axis=0)/np.maximum(mean,1)
    chosen=[]
    for low,high in zip(np.linspace(40,3780,5)[:-1],np.linspace(40,3780,5)[1:]):
        eligible=np.flatnonzero((depth>=low)&(depth<high)&np.isin(ids,list(good))&(mean>=100))
        if not len(eligible): continue
        ordered=eligible[np.argsort(mean[eligible],kind='stable')]
        chosen.append(int(ordered[len(ordered)//2]))
        chosen.extend(int(x) for x in eligible[np.argsort(-variability[eligible],kind='stable')[:2]])
    chosen=list(dict.fromkeys(chosen)); selection=[]; payload={}
    for i in chosen:
        cid=int(ids[i]); channels=np.argsort(np.linalg.norm(loc-loc[peak[i]],axis=1),kind='stable')[:12]
        selection.append(dict(unit_id=cid,depth_um=float(depth[i]),peak_channel=int(peak[i]),mean_anchor_count=float(mean[i]),anchor_count_cv=float(variability[i]),channels=channels.tolist()))
        payload[f'unit_{cid}_cached_waveform']=waves[i,:,channels].T
        payload[f'unit_{cid}_channels']=channels
        for w in manifest['windows']: payload[f"unit_{cid}_{w['segment_id']}_frames"]=selected_events[(w['segment_id'],cid)]
    np.savez_compressed(target/'waveform_inspection_plan.npz',**payload)
    (target/'waveform_selection.json').write_text(json.dumps(selection,indent=2)+'\n')
    summary=dict(status='reference_counts_and_fixed_anchor_fits_complete',rate_rows=len(rates),fit_rows=len(fits),zero_event_30s_bins=int((rates['count']==0).sum()),fit_status_counts=fits.status.value_counts().to_dict(),ks_good_fit_status_counts=fits[fits.ks_good].status.value_counts().to_dict(),waveform_units=len(chosen),waveform_status='selection_and_event_frames_saved_raw_extraction_pending',fit_definition='Legacy amplitude CDF fitter on every retained event in each fixed 120s anchor, minimum 1000 events. No continuous-block exclusion; maximum internal gap reported. Thus this is a fixed-time diagnostic, not a rerun of adaptive production QC.',limitations=['Conditional amplitude fits cannot measure undetected events.','Six existing discovery windows are not an independent holdout.','Waveform selections use reference rate variation; illustrative, not population estimates.','No corrected-sort comparison exists yet.'],manifest_sha256=hashlib.sha256((OUT/'manifest.json').read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (target/'summary.json').write_text(json.dumps(summary,indent=2)+'\n'); print(json.dumps(summary),flush=True)
if __name__=='__main__': main()
