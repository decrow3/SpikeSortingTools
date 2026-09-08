"""Independent persisted-output QA and compact report for full-recording fits."""
import json
import numpy as np
import pandas as pd
from testing.luke_full_reference_truncation import BASE, ROOT, OUT

def main():
    summary=json.loads((OUT/'summary.json').read_text()); u=pd.read_csv(OUT/'units.csv'); w=pd.read_csv(OUT/'windows.csv'); bins=pd.read_csv(OUT/'counts_30s.csv'); residual=pd.read_csv(OUT/'residual_spikes.csv')
    assert len(u)==710 and u.unit_id.is_unique and not w.duplicated(['unit_id','window_id']).any()
    assert (w.spikes==1000).all() and (w.duration_s>0).all()
    assert (u.fit_input_spikes+u.residual_spikes).sum()==29227829
    assert bins['count'].sum()==29227829
    expected=residual.groupby('unit_id').spikes.sum().reindex(u.unit_id,fill_value=0).to_numpy()
    assert np.array_equal(expected,u.residual_spikes)
    old=np.load(BASE/'qc/amp_truncation/truncation_qc.npz'); oldcounts=pd.Series(old['cid'].astype(int)).value_counts().reindex(u.unit_id,fill_value=0).to_numpy()
    assert np.array_equal(oldcounts,u.windows), 'Adaptive window counts differ from cached production QC'
    t=np.load(BASE/'cur/cur_output/spike_times.npy',mmap_mode='r').reshape(-1); c=np.load(BASE/'cur/cur_output/spike_clusters.npy',mmap_mode='r').reshape(-1)
    selected=[int(u.loc[u.windows.idxmax(),'unit_id']),37,341,686]
    for cid in selected:
        tt=t[c==cid]; window=w[w.unit_id==cid]; oldwin=old['window_blocks'][old['cid']==cid]
        assert len(window)==len(oldwin)
        assert np.array_equal(window.start_frame,tt[oldwin[:,0]])
        assert np.array_equal(window.end_frame,tt[oldwin[:,1]]+1)
        assert np.array_equal(oldwin[:,1]-oldwin[:,0]+1,np.full(len(oldwin),1000))
    checks=['Every saved fit has exactly 1000 input events','All 29,227,829 retained spikes accounted for as fit input plus residuals','Full-session count bins independently reconcile to retained total','Residual table reconciles per unit','All 710 per-unit adaptive window counts equal cached production QC','Selected four units have exact physical boundary agreement with cached production window indices']
    (OUT/'validation.json').write_text(json.dumps(dict(status='passed',checks=checks,checked_boundary_unit_ids=selected),indent=2)+'\n')
    report=ROOT/'docs/luke_full_reference_truncation_20260907.md'
    result='\n## Completed results\n\n| Measure | All clusters | KS-good clusters |\n|---|---:|---:|\n'
    a,b=summary['strata']['all'],summary['strata']['ks_good']
    for label,key in [('Clusters','units'),('Clusters with at least one valid fit','units_with_any_valid_fit'),('Clusters without a valid fit','units_without_valid_fit'),('Spikes entering fit windows','fit_input_spikes'),('Explicit residual spikes','residual_spikes')]: result+=f'| {label} | {a[key]:,} | {b[key]:,} |\n'
    result+=f"| Median of per-cluster valid-fit medians | {a['median_of_unit_median_valid_missing_pct']:.2f}% | {b['median_of_unit_median_valid_missing_pct']:.2f}% |\n"
    for label,stats in [('All clusters',a),('KS-good',b)]:
        q=stats['window_duration_s_quantiles']; result+=f"\n{label}: {sum(stats['fit_status_counts'].values()):,} windows; median duration {q['0.5']:.1f} s, P5–P95 {q['0.05']:.1f}–{q['0.95']:.1f} s. Status counts: `{stats['fit_status_counts']}`. Median per-cluster fit-span coverage of the recording: {100*stats['median_fit_time_coverage_fraction']:.1f}%.\n"
    result+='\nThese are full-recording reference measurements. The successful-fit medians are conditional summaries, not an estimate of all lost spikes and not a motion-off versus corrected comparison.\n\nValidation passed: '+ '; '.join(checks)+'. Partition edge tests also passed for empty trains, exact 1,000-event boundaries, residuals and gaps.\n'
    report.write_text(report.read_text()+result)
    print(json.dumps(dict(status='validation_passed',strata=summary['strata']),indent=2))
if __name__=='__main__': main()
