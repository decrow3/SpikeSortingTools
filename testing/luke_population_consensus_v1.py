"""Cached-only, spike-supported adaptive summaries and regional consensus.

No DREDGE values enter selection, grouping, anchoring, or consensus. A consensus
point aggregates window observations containing actual spikes in that grid cell;
it is not an independent 0.25-second motion measurement.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'testing/outputs'
SETTINGS = dict(interval_s=[930, 1030], target_events=20, minimum_events=10,
                maximum_span_s=5., split_gap_s=2., grid_s=.25,
                depth_edges_um=[0, 640, 1280, 1920, 2560, 3200, 3840],
                minimum_families=3, bootstrap_resamples=100,
                template_quality='Primary consensus excludes local energy fraction <0.35 OR far peak ratio >0.8; all candidates remain plotted.',
                grouping='Nonoverlapping event-count windows; split on every coarse shift change or >2s gap; max5s.',
                consensus='Equal family votes; only grid cells containing actual accepted events; median and leave-one-family-out spread. Bin spans are retained; grid is not temporal resolution.',
                anchor='Template energy centroid, without fitted DREDGE offsets. Common seed epoch required for regional consensus.')


def adaptive_groups(t, shifts):
    """Yield indices, retaining short/unsupported runs for the audit."""
    if not len(t):
        return
    breaks = np.flatnonzero((np.diff(t) > SETTINGS['split_gap_s']) | (np.diff(shifts) != 0)) + 1
    for run in np.split(np.arange(len(t)), breaks):
        i = 0
        while i < len(run):
            j = i + 1
            while j < len(run) and j-i < SETTINGS['target_events'] and t[run[j]]-t[run[i]] <= SETTINGS['maximum_span_s']:
                j += 1
            yield run[i:j]
            i = j


def waveform_centroid(w, depths):
    e = np.sum(np.median(w.astype(float), axis=0)**2, axis=0)
    return float(e @ depths/e.sum())


def consensus(votes):
    """A duplicate family has one vote; range describes sensitivity, not CI."""
    if votes.empty:
        return pd.DataFrame()
    result = []
    for (region, grid), q in votes.groupby(['region', 'grid']):
        # A family cannot gain influence through several templates/windows.
        families = q.groupby('family_id', sort=True).agg(value=('value', 'median'),
            dredge=('dredge', 'median'), span=('span', 'max'))
        v = families.value.to_numpy()
        n = len(v)
        supported = n >= SETTINGS['minimum_families']
        loo = [np.median(np.delete(v, i)) for i in range(n)] if n > 1 else [np.nan]
        result.append(dict(region=region, grid=int(grid), time_s=930+(grid+.5)*.25,
            families=n, status='supported' if supported else 'insufficient_families',
            consensus_um=float(np.median(v)) if supported else np.nan,
            loo_low_um=float(np.min(loo)) if supported else np.nan,
            loo_high_um=float(np.max(loo)) if supported else np.nan,
            unit_q25_um=float(np.quantile(v,.25)), unit_q75_um=float(np.quantile(v,.75)),
            dredge_same_support_um=float(families.dredge.median()) if supported else np.nan,
            max_span_s=float(families.span.max())))
    return pd.DataFrame(result)


def savefig(fig, output, stem):
    for ext in ['png', 'pdf']:
        fig.savefig(output/f'{stem}.{ext}', dpi=180)
    plt.close(fig)


def main(input_dir=None, output_dir=None):
    source = Path(input_dir or SRC/'luke_population_depth_v2')
    output = Path(output_dir or SRC/'luke_population_consensus_v1')
    assert json.loads((source/'summary.json').read_text())['status'] == 'complete'
    output.mkdir(exist_ok=True)
    from testing.luke_epoch_corroboration import BASE
    manifest = BASE/'recording/rescue_recording_manifest.json'
    m = json.loads(manifest.read_text()); fs = m['sampling_frequency_hz']; geo=np.asarray(m['channel_locations_um'])
    candidates = pd.read_csv(source/'candidates.csv')
    if 'selected' in candidates:
        candidates = candidates[candidates.selected.astype(str).str.lower().isin(['true', '1'])].copy()
    candidates = candidates.sort_values('depth_um')
    quality_path=source/'template_quality_audit.csv'
    quality=pd.read_csv(quality_path)
    flagged=set(quality.loc[quality.broad_shared_suspect.astype(str).str.lower().isin(['true','1']),'unit_id'])
    candidates['consensus_eligible']=~candidates.unit_id.isin(flagged)
    candidates.to_csv(output/'candidate_consensus_eligibility.csv',index=False)
    events = pd.read_csv(source/'events.csv')
    common_epoch = candidates.training_start_s.nunique() == 1
    templates = np.load(source/'templates.npz')
    files = [quality_path, Path(__file__), source/'candidates.csv', source/'events.csv', source/'templates.npz', manifest, SRC/'luke_peak_population_review_v1/02_long_histograms_histograms.npz']
    rows=[]; members={}; rng=np.random.default_rng(20260908)
    for c in candidates.itertuples():
        path=source/f'events_unit_{int(c.unit_id)}.npz'
        if not path.exists():
            continue
        files.append(path); z=np.load(path); t=z['frames']/fs
        ix=np.flatnonzero(z['status'].astype(str)=='accepted')
        assert np.all(np.diff(t)>=0)
        for local in adaptive_groups(t[ix], z['shift_um'][ix]):
            ids=ix[local]; times=t[ids]; wave=z['waveforms'][ids]
            channels=z['channels'][ids] if z['channels'].ndim==2 else z['channels']
            if channels.ndim==2:
                assert np.all(channels==channels[0]), 'One bin must use one translated support'
                channels=channels[0]
            supported=len(ids)>=SETTINGS['minimum_events']
            center=waveform_centroid(wave,geo[channels,1])
            boot=[waveform_centroid(wave[rng.integers(0,len(ids),len(ids))],geo[channels,1]) for _ in range(100)] if supported else [np.nan]
            lo,hi=np.quantile(boot,[.025,.975])
            bid=len(rows); members[bid]=times
            rows.append(dict(bin_id=bid,unit_id=int(c.unit_id),family_id=str(c.family_id),depth_um=c.depth_um,
                region=int(np.clip(np.searchsorted(SETTINGS['depth_edges_um'],c.depth_um,side='right')-1,0,5)),
                start_s=times.min(),end_s=times.max(),time_s=np.median(times),events=len(ids),span_s=np.ptp(times),
                shift_um=z['shift_um'][ids[0]],status='supported' if supported else 'insufficient_events',
                centroid_um=center,relative_um=center-c.template_centroid_um,
                low_um=lo-c.template_centroid_um,high_um=hi-c.template_centroid_um))
    table=pd.DataFrame(rows)
    if table.empty:
        table=pd.DataFrame(columns=['bin_id','unit_id','family_id','depth_um','region','start_s','end_s','time_s','events','span_s','shift_um','status','centroid_um','relative_um','low_um','high_um'])
    table.to_csv(output/'adaptive_windows_before_dredge.csv',index=False)
    np.savez_compressed(output/'adaptive_window_event_times.npz', **{f'bin_{k}': v for k,v in members.items()})
    (output/'settings.json').write_text(json.dumps(SETTINGS,indent=2))
    # Freeze every waveform observation and its support before loading DREDGE.
    field_path=SRC/'luke_early_screen_shortlist_v1/fields/compensated.npz'
    files.append(field_path); f=np.load(field_path); votes=[]; dredge={}
    for c in candidates.itertuples():
        trace=np.array([np.interp(c.depth_um,f['depth_um'],d) for d in f['displacement_um']])
        # Fixed seed epoch reference (not optimized against waveform estimates).
        seed_key=f'unit_{int(c.unit_id)}_seed_frames'
        if seed_key not in templates:
            raise ValueError(f'Missing {seed_key}: DREDGE reference requires actual seed-event support')
        seed_times=templates[seed_key]/fs
        offset=float(np.median(np.interp(seed_times,f['time_s'],trace)))
        dredge[int(c.unit_id)]=(trace-offset)
    for r in table.itertuples():
        times=members[r.bin_id]; d=np.interp(times,f['time_s'],dredge[r.unit_id])
        table.loc[table.bin_id==r.bin_id,'dredge_same_support_um']=np.median(d)
        if r.status!='supported' or not common_epoch or r.unit_id in flagged:
            continue
        for grid in np.unique(np.floor((times-930)/.25).astype(int)):
            votes.append(dict(region=r.region,grid=int(grid),unit_id=r.unit_id,family_id=r.family_id,
                bin_id=r.bin_id,value=r.relative_um,dredge=float(np.median(d)),span=r.span_s))
    vote_table=pd.DataFrame(votes); cons=consensus(vote_table)
    table.to_csv(output/'adaptive_windows.csv',index=False); vote_table.to_csv(output/'consensus_votes.csv',index=False)
    cons.to_csv(output/'regional_consensus.csv',index=False)
    span_summary=table[table.status=='supported'].groupby('region').agg(supported_bins=('bin_id','size'),median_span_s=('span_s','median'),maximum_span_s=('span_s','max'))
    span_summary.to_csv(output/'regional_bin_spans.csv')
    hist=np.load(SRC/'luke_peak_population_review_v1/02_long_histograms_histograms.npz')
    fig, ax=plt.subplots(figsize=(15,9),layout='constrained'); a=np.log1p(hist['arm1_counts'])
    ax.imshow(a,origin='lower',aspect='auto',extent=[930,1030,hist['depth_edges'][0],hist['depth_edges'][-1]],cmap='magma',vmin=0,vmax=np.quantile(a,.995))
    for c in candidates.itertuples():
        p=events[events.unit_id==c.unit_id]; good=p.status=='accepted'
        ax.scatter(p.loc[~good,'time_s'],p.loc[~good,'centroid_um'],s=2,c='gray',alpha=.25,rasterized=True)
        ax.scatter(p.loc[good,'time_s'],p.loc[good,'centroid_um'],s=3,c='#56DDE0',alpha=.35,rasterized=True)
        q=table[(table.unit_id==c.unit_id)&(table.status=='supported')]
        ax.errorbar(q.time_s,q.centroid_um,xerr=[q.time_s-q.start_s,q.end_s-q.time_s],fmt='D',ms=2.5,color='white',elinewidth=.65)
        ax.text(1030.3,c.depth_um,str(c.unit_id)+('*' if c.unit_id in flagged else ''),fontsize=6,va='center')
    ax.axvspan(930,940,facecolor='none',edgecolor='white',hatch='///',alpha=.3,label='Template training')
    ax.set(xlim=(930,1030),xlabel='Time (s)',ylabel='Depth (µm)',title=f'{len(candidates)} candidate cells through depth · provisional identities\nCyan: accepted events; gray: uncertain alternatives; white: adaptive centroids/spans; * = seed-quality review, no consensus vote')
    savefig(fig,output,'01_population_depth_time')
    fig,axes=plt.subplots(6,2,figsize=(16,17),sharex=True,layout='constrained',gridspec_kw={'width_ratios':[3,1]})
    for region,(ax,coverage) in enumerate(axes):
        q=table[(table.region==region)&(table.status=='supported')]
        for unit,part in q.groupby('unit_id'):
            ax.errorbar(part.time_s,part.relative_um,xerr=[part.time_s-part.start_s,part.end_s-part.time_s],fmt='o',ms=2,alpha=.35,lw=.5)
        cc=cons[cons.region==region] if len(cons) else pd.DataFrame()
        if len(cc):
            good=cc.status=='supported'; support=cc[good]
            ax.scatter(support.time_s,support.consensus_um,c='#0072B2',s=8,label='Family-median consensus')
            ax.vlines(support.time_s,support.loo_low_um,support.loo_high_um,color='#0072B2',alpha=.35)
            ax.scatter(support.time_s,support.dredge_same_support_um,c='#D55E00',marker='x',s=9,label='DREDGE, same spike support')
            coverage.scatter(cc.time_s,cc.families,s=5,c='#0072B2'); coverage.axhline(3,c='gray',ls='--')
        mids=(SETTINGS['depth_edges_um'][region]+SETTINGS['depth_edges_um'][region+1])/2
        local=candidates[(candidates.depth_um>=SETTINGS['depth_edges_um'][region])&(candidates.depth_um<SETTINGS['depth_edges_um'][region+1])]
        if len(local):
            traces=np.array([dredge[int(u)] for u in local.unit_id]); ax.plot(f['time_s'],np.median(traces,axis=0),c='#D55E00',lw=.7,alpha=.5,label='DREDGE full-time, median candidate depths')
        ax.axvspan(930,940,color='gray',alpha=.15)
        coverage.axvspan(930,940,color='gray',alpha=.15)
        ax.set(title=f'{SETTINGS["depth_edges_um"][region]}–{SETTINGS["depth_edges_um"][region+1]} µm · {len(local)} candidates',ylabel='Template-relative centroid (µm)',xlim=(930,1030))
        coverage.set(title='Independent families',ylabel='Observed families / 0.25s',ylim=(0,max(4,len(local)+.5)))
        ax.legend(fontsize=6,loc='best')
    axes[-1,0].set_xlabel('Time (s)');axes[-1,1].set_xlabel('Time (s)')
    fig.suptitle('Depth-resolved consensus · minimum 3 families · seed-quality flags excluded · no rigid global trace assumed',fontsize=15)
    fig.supxlabel('Points repeat bin observations only in grid cells with actual accepted spikes. Horizontal bars = full adaptive span; 0.25s grid is NOT temporal resolution.\nVertical blue bars = leave-one-family-out sensitivity, not confidence intervals. Gaps are unsupported; waveform/identity calibration remains provisional.',fontsize=9)
    savefig(fig,output,'02_regional_consensus')
    # Small multiples retain each cell, including cells that never reach support.
    n=len(candidates); fig,axes=plt.subplots(int(np.ceil(n/4)),4,figsize=(16,2.5*int(np.ceil(n/4))),layout='constrained',squeeze=False)
    for ax,c in zip(axes.flat,candidates.itertuples()):
        p=events[(events.unit_id==c.unit_id)&(events.status=='accepted')]
        ax.scatter(p.time_s,p.centroid_um-c.template_centroid_um,s=2,alpha=.2,c='#0072B2')
        q=table[(table.unit_id==c.unit_id)&(table.status=='supported')]
        ax.errorbar(q.time_s,q.relative_um,xerr=[q.time_s-q.start_s,q.end_s-q.time_s],fmt='o',ms=3,color='#0072B2',lw=.5)
        ax.plot(f['time_s'],dredge[int(c.unit_id)],c='#D55E00',lw=.8)
        ax.axvspan(930,940,color='gray',alpha=.15)
        ax.set(title=f'Unit {c.unit_id} · {c.depth_um:g} µm · {len(p)} events / {len(q)} bins',xlim=(930,1030),ylabel='Relative µm')
    for ax in list(axes.flat)[n:]: ax.axis('off')
    fig.suptitle('Every selected candidate · adaptive waveform support (blue) and frozen DREDGE (orange)')
    savefig(fig,output,'03_all_unit_motion')
    fig,axes=plt.subplots(int(np.ceil(n/4)),4,figsize=(16,2.4*int(np.ceil(n/4))),layout='constrained',squeeze=False)
    for ax,c in zip(axes.flat,candidates.itertuples()):
        wave=templates[f'unit_{int(c.unit_id)}_template']; ch=templates[f'unit_{int(c.unit_id)}_channels']
        strengths=np.max(np.abs(wave),axis=0); show=np.argsort(strengths)[-min(5,len(ch)):]
        for j in show:
            ax.plot(np.arange(-30,31)/fs*1000,wave[:,j],lw=.8,label=f'{geo[ch[j],1]:g} µm')
        ax.set(title=f'Unit {c.unit_id} · {c.depth_um:g} µm · family {c.family_id}',xlabel='Time (ms)',ylabel='µV')
        ax.legend(fontsize=5,ncol=2)
    for ax in list(axes.flat)[n:]:ax.axis('off')
    fig.suptitle('Frozen seed templates · five strongest channels per cell · original referenced AP')
    savefig(fig,output,'04_template_waveforms')
    result=dict(status='complete',candidates=len(candidates),consensus_eligible_candidates=int(candidates.consensus_eligible.sum()),quality_flagged_units=sorted(map(int,flagged)),accepted_events=int((events.status=='accepted').sum()),supported_bins=int((table.status=='supported').sum()),common_training_epoch=bool(common_epoch),heldout_consensus_cells=int(((cons.status=='supported')&(cons.time_s>=940)).sum()) if len(cons) else 0,training_consensus_cells=int(((cons.status=='supported')&(cons.time_s<940)).sum()) if len(cons) else 0,supported_consensus_cells=int((cons.status=='supported').sum()) if len(cons) else 0,settings=SETTINGS,source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    (output/'summary.json').write_text(json.dumps(result,indent=2))
    (output/'README.md').write_text(f'''# Provisional population lighthouse consensus\n\n{len(candidates)} selected candidates ({result['consensus_eligible_candidates']} eligible for primary consensus; seed-quality flags {result['quality_flagged_units']} remain plotted but do not vote); {result['accepted_events']} accepted events; {result['supported_bins']} supported adaptive windows; {result['supported_consensus_cells']} supported region/grid entries. Of these, {result['heldout_consensus_cells']} entries are held out (940–1030s) and {result['training_consensus_cells']} are in the shaded930–940s training interval. Repeated entries from one adaptive window are not independent observations. No supported consensus means insufficient corroboration, not no motion. Per-region median/max spans are saved in regional_bin_spans.csv.\n\n{span_summary.to_string()}\n\nAdaptive windows target20 events, require10, span at most5s, and split at >2s gaps or any coarse depth-hypothesis change. Every family has one equal consensus vote. At least3 families are needed. No global rigid-motion trace is assumed.\n\nOffsets use seed-template waveform centroids. Candidate seed epochs common: {common_epoch}. DREDGE is loaded only after windows are frozen and uses each window's identical accepted spike times, relative to its actual seed-spike support. Seed phases may differ across units; no reference uncertainty correction is fitted. Full-time DREDGE is separately drawn. Gray events preserve uncertain identities/depths; these do not vote.\n\nThe grid samples observed spike support, not continuous coverage; bin spans determine effective resolution. Leave-one-family-out ranges are sensitivity diagnostics, not confidence intervals. Bootstrap spans exclude identity, reference-template and localization-model uncertainty. Coarse shifts and centroid changes are not calibrated physical displacement. This exploratory expansion does not yet prove DREDGE smoothing or recording-wide validity.\n''')
    print(json.dumps({k:v for k,v in result.items() if k not in ['source_sha256','settings']},indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--input-dir');ap.add_argument('--output-dir');args=ap.parse_args();main(args.input_dir,args.output_dir)
