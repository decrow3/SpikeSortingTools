"""Full-recording count-supported truncation with explicit excluded data."""
from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
from testing.luke_full_reference_panel_analysis import BASE, ROOT, classify_fit
OUT=ROOT/'testing/outputs/luke_full_reference_truncation_v1'
N=1000
GAP=10.

def partition(t, fs):
    """Return inclusive-exclusive index windows and residuals; no event omitted."""
    edges=np.r_[0,np.flatnonzero(np.diff(t)>GAP*fs)+1,len(t)]
    windows=[]; residuals=[]
    for a,b in zip(edges[:-1],edges[1:]):
        if a==b: continue
        num=(b-a)//N; start=a+((b-a)-num*N)//2
        if not num: residuals.append((int(a),int(b))); continue
        windows.extend((int(j),int(j+N)) for j in range(start,start+num*N,N))
        if a<start: residuals.append((int(a),int(start)))
        if start+num*N<b: residuals.append((int(start+num*N),int(b)))
    return windows,residuals

def main():
    OUT.mkdir(exist_ok=False)
    audit=json.loads((ROOT/'testing/outputs/luke_full_reference_reuse_audit_v1/summary.json').read_text()); assert audit['status']=='reference_reusable'
    rec=json.loads((BASE/'recording/rescue_recording_manifest.json').read_text()); fs=rec['sampling_frequency_hz']; end=rec['num_samples']
    labels=pd.read_csv(BASE/'cur/cur_output/cluster_KSLabel.tsv',sep='\t'); ids=labels.cluster_id.to_numpy(); good=set(labels.loc[labels.KSLabel=='good','cluster_id'])
    cur=BASE/'cur/cur_output'; ts=np.load(cur/'spike_times.npy',mmap_mode='r').reshape(-1); cl=np.load(cur/'spike_clusters.npy',mmap_mode='r').reshape(-1); full=np.load(cur/'full_st.npy',mmap_mode='r'); kept=np.load(cur/'kept_spikes.npy',mmap_mode='r'); retained=np.flatnonzero(kept) if kept.dtype.kind=='b' else kept
    assert len(ts)==len(cl)==len(retained)==audit['curated_spikes']
    order=np.argsort(cl,kind='stable'); sorted_ids=np.asarray(cl[order]); boundaries=np.searchsorted(sorted_ids,ids); stops=np.searchsorted(sorted_ids,ids,side='right')
    rows=[]; gaps=[]; leftover=[]; units=[]; bins=[]
    edges=np.unique(np.r_[np.ceil(np.arange(0,end/fs,30)*fs).astype('int64'),end])
    for ui,(cid,a,b) in enumerate(zip(ids,boundaries,stops)):
        take=order[a:b]; t=np.asarray(ts[take]); amplitude=np.asarray(full[retained[take],2]); assert np.array_equal(t,full[retained[take],0]) and np.all(np.diff(t)>=0)
        windows,residuals=partition(t,fs); assert sum(j-i for i,j in windows+residuals)==len(t)
        spans=sorted(windows+residuals); assert all(spans[k][1]==spans[k+1][0] for k in range(len(spans)-1))
        covered_frames=0; valid=[]; ceiling=failed=0
        for wi,(i,j) in enumerate(windows):
            amp=amplitude[i:j]; assert len(amp)==N
            status,pct,error=classify_fit(amp); duration=(int(t[j-1])+1-int(t[i]))/fs; covered_frames+=int(t[j-1])+1-int(t[i])
            if status=='fit_valid': valid.append(pct)
            ceiling+=status=='fit_at_50pct_ceiling'; failed+=status=='fit_failed'
            rows.append(dict(unit_id=int(cid),ks_good=int(cid) in good,window_id=wi,start_frame=int(t[i]),end_frame=int(t[j-1])+1,start_s=float(t[i]/fs),end_s=float((t[j-1]+1)/fs),duration_s=duration,spikes=N,rate_hz=N/duration,status=status,missing_pct=pct,error=error))
        for i,j in residuals: leftover.append(dict(unit_id=int(cid),start_frame=int(t[i]),end_frame=int(t[j-1])+1,spikes=j-i,reason='fewer_than_1000_remaining_in_continuous_block'))
        endpoints=np.r_[0,t,end]; differences=np.diff(endpoints); long=np.flatnonzero(differences>GAP*fs)
        for gi in long: gaps.append(dict(unit_id=int(cid),start_frame=int(endpoints[gi]),end_frame=int(endpoints[gi+1]),duration_s=float(differences[gi]/fs),kind='leading' if gi==0 else 'trailing' if gi==len(endpoints)-2 else 'internal'))
        counts=np.histogram(t,bins=edges)[0]; assert counts.sum()==len(t)
        bins.extend(dict(unit_id=int(cid),ks_good=int(cid) in good,start_frame=int(lo),end_frame=int(hi),count=int(n)) for lo,hi,n in zip(edges[:-1],edges[1:],counts))
        units.append(dict(unit_id=int(cid),ks_good=int(cid) in good,spikes=len(t),windows=len(windows),fit_valid=len(valid),fit_at_ceiling=int(ceiling),fit_failed=int(failed),median_valid_missing_pct=float(np.median(valid)) if valid else np.nan,fit_input_spikes=len(windows)*N,residual_spikes=sum(j-i for i,j in residuals),fit_time_coverage_fraction=covered_frames/end,long_gap_count=len(long),long_gap_seconds=float(differences[long].sum()/fs),zero_event_30s_bins=int((counts==0).sum()),bins=len(counts)))
        if ui%50==0: print(json.dumps({'units_complete':ui+1,'total_units':len(ids),'fit_windows':len(rows)}),flush=True)
    fits=pd.DataFrame(rows); u=pd.DataFrame(units)
    fits.to_csv(OUT/'windows.csv',index=False); u.to_csv(OUT/'units.csv',index=False); pd.DataFrame(gaps).to_csv(OUT/'gaps.csv',index=False); pd.DataFrame(leftover).to_csv(OUT/'residual_spikes.csv',index=False); pd.DataFrame(bins).to_csv(OUT/'counts_30s.csv',index=False)
    strata={}
    for label,subset in [('all',u),('ks_good',u[u.ks_good])]:
        eligible=subset[subset.fit_valid>0]; ww=fits[fits.unit_id.isin(subset.unit_id)]
        strata[label]=dict(units=len(subset),units_with_any_valid_fit=len(eligible),units_without_valid_fit=int((subset.fit_valid==0).sum()),fit_status_counts=ww.status.value_counts().to_dict(),window_duration_s_quantiles=ww.duration_s.quantile([0,.05,.5,.95,1]).to_dict(),median_of_unit_median_valid_missing_pct=float(eligible.median_valid_missing_pct.median()),fit_input_spikes=int(subset.fit_input_spikes.sum()),residual_spikes=int(subset.residual_spikes.sum()),median_fit_time_coverage_fraction=float(subset.fit_time_coverage_fraction.median()))
    assert len(u)==710 and (u.fit_input_spikes+u.residual_spikes).sum()==len(ts)
    result=dict(status='complete',reference_identity=audit['sort_identity']['identity_digest'],duration_s=end/fs,sampling_frequency_hz=fs,spikes_per_window=N,max_internal_gap_s=GAP,strata=strata,method='For each cluster across the full recording, split only at internal gaps >10s; center as many disjoint 1000-event windows as fit within each continuous block. Record every residual event and all >10s gaps including leading/trailing. Fit every 1000-event window with explicit numerical failure and ceiling status.',legacy_difference='Production stores inclusive last-event indices but slices exclusively, fitting 999 of each nominal 1000 events. This diagnostic explicitly fits all 1000; source QC is unchanged. Physical intervals use [first event, last event + one sample).',comparison_rule='Reference-derived physical intervals are frozen for future corrected-sort/family comparisons. Fit both arms in identical intervals; report insufficient candidate counts without resizing intervals to favor either arm. Family identity must be independently validated first.',limitations=['Conditional amplitude distributions cannot quantify undetected events.','Long count-supported windows can mix waveform states; inspect durations and time-varying fits.','Gaps are inter-event intervals, not evidence of missing biological spikes.','Numerical fit success does not validate the distributional model.'],source_audit_sha256=hashlib.sha256((ROOT/'testing/outputs/luke_full_reference_reuse_audit_v1/summary.json').read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result),flush=True)
if __name__=='__main__': main()
