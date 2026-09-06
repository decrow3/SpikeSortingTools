"""Descriptive, common-time comparison for the full-session rigid experiment."""
from pathlib import Path
import json
import hashlib
import io

import numpy as np
import pandas as pd
from numba import njit

from testing.luke_full_session_rigid import BASE, OUT
from testing.luke_amplitude_dropout_audit import read_curated_arrays, curated_arrays_from_raw, read_cached_truncation_qc, build_windows_table


@njit(cache=True)
def coincidence_counts(at, ac, bt, bc, na, nb, tolerance):
    """All temporally possible edges; no cross-cluster exclusive assignment."""
    counts = np.zeros((na, nb), dtype=np.int64)
    left = 0
    for i in range(len(at)):
        while left < len(bt) and bt[left] < at[i] - tolerance:
            left += 1
        j = left
        while j < len(bt) and bt[j] <= at[i] + tolerance:
            counts[ac[i], bc[j]] += 1
            j += 1
    return counts


@njit(cache=True)
def exclusive_count(a, b, tolerance):
    i = j = count = 0
    while i < len(a) and j < len(b):
        if a[i] < b[j] - tolerance:
            i += 1
        elif b[j] < a[i] - tolerance:
            j += 1
        else:
            count += 1
            i += 1
            j += 1
    return count


def correspondence(a, b, tolerance):
    aid, ac = np.unique(a['cl'], return_inverse=True)
    bid, bc = np.unique(b['cl'], return_inverse=True)
    an, bn = np.bincount(ac), np.bincount(bc)
    possible = coincidence_counts(a['st'], ac, b['st'], bc, len(aid), len(bid), tolerance)
    # Group once, preserving chronological ordering within each cluster.
    at = np.split(a['st'][np.argsort(ac, kind='stable')], np.cumsum(an)[:-1])
    bt = np.split(b['st'][np.argsort(bc, kind='stable')], np.cumsum(bn)[:-1])
    edges = []
    for i,j in np.argwhere(possible >= .1*np.minimum(an[:,None], bn[None,:])):
        n = exclusive_count(at[i], bt[j], tolerance)
        if n < .1*min(an[i], bn[j]):
            continue
        edges.append(dict(baseline_cluster=int(aid[i]), candidate_cluster=int(bid[j]),
            matched_events=n, baseline_events=int(an[i]), candidate_events=int(bn[j]),
            baseline_retention=n/an[i], candidate_retention=n/bn[j],
            jaccard=n/(an[i]+bn[j]-n)))
    columns = ['baseline_cluster','candidate_cluster','matched_events','baseline_events',
               'candidate_events','baseline_retention','candidate_retention','jaccard']
    frame = pd.DataFrame(edges, columns=columns)
    if frame.empty:
        frame['primary_match'] = pd.Series(dtype=bool)
        return frame
    # Ties are ambiguous, never resolved by cluster ID.
    abest = frame.groupby('baseline_cluster').jaccard.transform('max')
    bbest = frame.groupby('candidate_cluster').jaccard.transform('max')
    am = frame.jaccard == abest
    bm = frame.jaccard == bbest
    au = am.groupby(frame.baseline_cluster).transform('sum') == 1
    bu = bm.groupby(frame.candidate_cluster).transform('sum') == 1
    frame['primary_match'] = am & bm & au & bu & (frame.baseline_retention >= .5) & (frame.candidate_retention >= .5)
    return frame


def common_time(a, b):
    """Intersect nonoverlapping [start,end] windows; no interpolation over gaps."""
    a, b = np.asarray(a).reshape(-1,3), np.asarray(b).reshape(-1,3)
    for v in (a,b):
        if len(v) and (not np.isfinite(v).all() or np.any(v[:,1] < v[:,0]) or np.any(v[1:,0] < v[:-1,1])):
            raise ValueError('invalid or overlapping fit intervals')
    i = j = 0
    seconds = weighted = 0.
    pieces = []
    while i < len(a) and j < len(b):
        lo, hi = max(a[i,0],b[j,0]), min(a[i,1],b[j,1])
        if hi > lo:
            d = a[i,2]-b[j,2]
            seconds += hi-lo
            weighted += (hi-lo)*d
            pieces.append((lo,hi,a[i,2],b[j,2]))
        if a[i,1] <= b[j,1]:
            i += 1
        else:
            j += 1
    return seconds, weighted/seconds if seconds else np.nan, pieces


def load_population(name, base, fs, dest):
    curated = base / ('cur/cur_sorter_output' if name == 'legacy' else 'cur/cur_output')
    raw, hashes = read_curated_arrays(curated)
    arrays = curated_arrays_from_raw(name, raw)
    del raw
    label_bytes=(curated/'cluster_KSLabel.tsv').read_bytes()
    labels=pd.read_csv(io.BytesIO(label_bytes),sep='\t')
    hashes['cluster_KSLabel.tsv']=hashlib.sha256(label_bytes).hexdigest()
    label_column=next(c for c in labels if c!='cluster_id')
    data=dict(st=arrays.times,cl=arrays.clusters,label=dict(zip(labels.cluster_id,labels[label_column].astype(str).str.lower())))
    cached,qc_hash=read_cached_truncation_qc(name,base/'qc')
    windows = build_windows_table(arrays, cached, fs)
    (dest/f'{name}_inputs.json').write_text(json.dumps(dict(curated=str(curated),hashes=hashes,qc_sha256=qc_hash),indent=2)+'\n')
    windows.to_csv(dest/f'{name}_windows.csv', index=False)
    del arrays
    stats = []
    for cid in np.unique(data['cl']):
        samples = data['st'][data['cl'] == cid]
        fits = windows[(windows.cluster_id == cid) & (windows.status == 'finite_interior')].sort_values('start_s')
        stats.append(dict(cluster_id=int(cid), label=data['label'].get(cid,'unknown'),
            spike_count=len(samples), rv_fraction=float(np.mean(np.diff(samples)/fs < .0015)) if len(samples)>1 else np.nan,
            n_valid_fits=len(fits), valid_seconds=float((fits.end_s-fits.start_s).sum())))
    units = pd.DataFrame(stats).set_index('cluster_id', drop=False)
    units.to_csv(dest/f'{name}_units.csv', index=False)
    return data, windows, units


def evaluate_pair(name, a, b, aw, bw, au, bu, fs, duration, dest):
    edges = correspondence(a,b,round(.0005*fs))
    edges.to_csv(dest/f'{name}_edges.csv', index=False)
    summaries, intervals = [], []
    for edge in edges[edges.primary_match].to_dict('records'):
        ca, cb = edge['baseline_cluster'], edge['candidate_cluster']
        def valid(w,c):
            return w[(w.cluster_id==c)&(w.status=='finite_interior')].sort_values('start_s')[['start_s','end_s','missing_pct']].to_numpy()
        av,bv = valid(aw,ca),valid(bw,cb)
        seconds, delta, pieces = common_time(av,bv)
        measurable = len(av)>=2 and len(bv)>=2 and seconds>0
        summaries.append(dict(**edge, baseline_label=au.loc[ca,'label'], candidate_label=bu.loc[cb,'label'],
            baseline_valid_fits=len(av),candidate_valid_fits=len(bv),
            baseline_coverage=au.loc[ca,'valid_seconds']/duration,candidate_coverage=bu.loc[cb,'valid_seconds']/duration,
            common_coverage=seconds/duration, measurable=measurable,
            improvement_pp=delta if measurable else np.nan,
            baseline_rv_fraction=au.loc[ca,'rv_fraction'],candidate_rv_fraction=bu.loc[cb,'rv_fraction']))
        intervals.extend(dict(baseline_cluster=ca,candidate_cluster=cb,start_s=p[0],end_s=p[1],baseline_missing_pct=p[2],candidate_missing_pct=p[3]) for p in pieces)
    paired = pd.DataFrame(summaries)
    paired.to_csv(dest/f'{name}_paired.csv',index=False)
    pd.DataFrame(intervals).to_csv(dest/f'{name}_common_intervals.csv',index=False)
    matched_a = set(paired.baseline_cluster) if len(paired) else set()
    matched_b = set(paired.candidate_cluster) if len(paired) else set()
    au[~au.cluster_id.isin(matched_a)].to_csv(dest/f'{name}_unmatched_baseline.csv',index=False)
    bu[~bu.cluster_id.isin(matched_b)].to_csv(dest/f'{name}_unmatched_candidate.csv',index=False)
    return paired


def plot_comparison(paired, aw, bw, duration, dest):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    named=[37,553,452,36]
    fig,axes=plt.subplots(len(named),1,figsize=(13,10),sharex=True)
    for ax,cid in zip(axes,named):
        match=paired[paired.baseline_cluster==cid] if len(paired) else pd.DataFrame()
        plots=[(aw,cid,'#315ba8','rescue')]
        if len(match):
            plots.append((bw,int(match.candidate_cluster.iloc[0]),'#d36b23','rigid'))
        for w,c,color,label in plots:
            v=w[(w.cluster_id==c)&(w.status=='finite_interior')]
            lines=[[(r.start_s/60,r.missing_pct),(r.end_s/60,r.missing_pct)] for r in v.itertuples()]
            ax.add_collection(LineCollection(lines,colors=color,linewidths=2,label=f'{label} c{c}'))
            pinned=w[(w.cluster_id==c)&(w.status=='boundary_pinned')]
            ax.scatter((pinned.start_s+pinned.end_s)/120,np.full(len(pinned),50),marker='x',color=color,s=10)
        ax.set(title=f'Baseline cluster {cid}'+(' — no primary correspondence' if match.empty else ''),ylim=(-2,53),ylabel='Missing (%)')
        ax.legend(loc='upper left')
    axes[-1].set(xlim=(0,duration/60),xlabel='Recording time (minutes); gaps have no valid fit; crosses are boundary fits')
    fig.tight_layout()
    fig.savefig(dest/'named_case_trajectories.png',dpi=150)
    plt.close(fig)
    if paired.empty:
        return
    good=paired[(paired.baseline_label=='good')&paired.measurable]
    fig,ax=plt.subplots(figsize=(8,5))
    ax.scatter(good.common_coverage*100,good.improvement_pp,s=18,alpha=.7)
    ax.axhline(0,color='black',linewidth=.7)
    ax.set(xlabel='Common valid-fit time (% of full session)',ylabel='Baseline − candidate missing estimate (pp)',title='Matched baseline-good clusters; positive indicates lower candidate estimate')
    fig.tight_layout()
    fig.savefig(dest/'paired_completeness_coverage.png',dpi=150)
    plt.close(fig)
    if len(good):
        ordered=good.sort_values('baseline_cluster')
        bins=np.arange(0,duration+60,60.)
        heat=np.full((len(ordered),len(bins)-1),np.nan)
        for k,row in enumerate(ordered.itertuples()):
            def valid(w,c):
                return w[(w.cluster_id==c)&(w.status=='finite_interior')].sort_values('start_s')[['start_s','end_s','missing_pct']].to_numpy()
            _,_,pieces=common_time(valid(aw,row.baseline_cluster),valid(bw,row.candidate_cluster))
            sums=np.zeros(len(bins)-1)
            weights=np.zeros_like(sums)
            for lo,hi,x,y in pieces:
                overlap=np.maximum(0,np.minimum(bins[1:],hi)-np.maximum(bins[:-1],lo))
                sums+=overlap*(x-y)
                weights+=overlap
            np.divide(sums,weights,out=heat[k],where=weights>0)
        fig,ax=plt.subplots(figsize=(13,7))
        im=ax.pcolormesh(bins/60,np.arange(len(ordered)+1),np.ma.masked_invalid(heat),cmap='RdBu',vmin=-30,vmax=30,rasterized=True)
        ax.set(xlabel='Recording minutes',ylabel='Matched baseline-good clusters (cluster-ID order)',title='Missing estimate: baseline − rigid (pp); blank = no shared valid fit')
        fig.colorbar(im,ax=ax,label='Percentage points; positive favors candidate')
        fig.tight_layout()
        fig.savefig(dest/'paired_time_course.png',dpi=150)
        plt.close(fig)


def main():
    dest=OUT/'comparison'
    dest.mkdir(exist_ok=True)
    request=json.loads((OUT/'request.json').read_text())
    fs=314204894/request['duration_s']
    print('loading full baseline QC',flush=True)
    a,aw,au=load_population('baseline',BASE,fs,dest)
    print('loading full candidate QC',flush=True)
    b,bw,bu=load_population('candidate',OUT,fs,dest)
    print('matching full trains and comparing common-time QC',flush=True)
    paired=evaluate_pair('rigid_vs_rescue',a,b,aw,bw,au,bu,fs,request['duration_s'],dest)
    plot_comparison(paired,aw,bw,request['duration_s'],dest)
    summary=dict(baseline_units=len(au),candidate_units=len(bu),baseline_good=int((au.label=='good').sum()),
        candidate_good=int((bu.label=='good').sum()),primary_pairs=len(paired),
        request_sha256=hashlib.sha256((OUT/'request.json').read_bytes()).hexdigest())
    if len(paired):
        good=paired[paired.baseline_label=='good']
        summary.update(matched_baseline_good=len(good),measurable_baseline_good=int(good.measurable.sum()),
            baseline_good_common_coverage_ge_50pct=int((good.common_coverage>=.5).sum()),
            median_paired_good_improvement_pp=float(good.improvement_pp.median()),
            median_paired_good_rv_change=float((good.candidate_rv_fraction-good.baseline_rv_fraction).median()))
    (dest/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)
    print('comparing against historical legacy context',flush=True)
    legacy=Path('/mnt/NPX/Luke/20250804/pipeline_results_Luke0804_V2V1_g0_imec0')
    l,lw,lu=load_population('legacy',legacy,fs,dest)
    evaluate_pair('rigid_vs_legacy',l,b,lw,bw,lu,bu,fs,request['duration_s'],dest)
    (dest/'complete.json').write_text(json.dumps(dict(complete=True,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))+'\n')


if __name__=='__main__':
    import argparse
    import os
    import time
    parser=argparse.ArgumentParser()
    parser.add_argument('--wait-for-pid',type=int)
    args=parser.parse_args()
    if args.wait_for_pid:
        while True:
            status=json.loads((OUT/'status.json').read_text())
            if status['stage']=='sort and QC complete; comparison pending':
                break
            try:
                os.kill(args.wait_for_pid,0)
            except ProcessLookupError:
                raise RuntimeError('sorting/QC process exited before completion; inspect run.log')
            time.sleep(30)
    try:
        main()
    except Exception as exc:
        (OUT/'comparison_failure.json').write_text(json.dumps(dict(error=repr(exc)))+'\n')
        raise
    (OUT/'status.json').write_text(json.dumps(dict(stage='full-session comparison complete',updated_unix=time.time()))+'\n')
