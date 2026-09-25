"""Matched-sample motion-field QC scoring and Markdown reports."""
from __future__ import annotations

import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy import stats

from .field import MotionField, file_sha256
from .reference import unit_common_mode

OKABE_ITO = ("#E69F00","#56B4E9","#009E73","#F0E442","#0072B2","#D55E00","#CC79A7","#000000")
LINESTYLES = ("-","--","-.",":")


def load_peaks(path: str | Path) -> pd.DataFrame:
    path=Path(path)
    if path.suffix in (".csv",".gz"):return pd.read_csv(path)
    with np.load(path,allow_pickle=False) as z:return pd.DataFrame({k:np.asarray(z[k]) for k in z.files if np.asarray(z[k]).ndim==1})


def mask_from_intervals(time_s, intervals, margin_s=0):
    result=np.zeros(len(time_s),bool)
    for row in pd.DataFrame(intervals).itertuples(index=False):result|=(time_s>=row.start_s-margin_s)&(time_s<row.end_s+margin_s)
    return result


def matched_grid(fields: list[MotionField], grid_s=.25):
    start=max(field.time_s[0] for field in fields);stop=min(field.time_s[-1] for field in fields)
    if stop<=start:raise ValueError("fields have no shared time support")
    return np.arange(np.ceil(start/grid_s)*grid_s,np.floor(stop/grid_s)*grid_s+grid_s/2,grid_s)


def score_fields(fields: list[MotionField], episodes: pd.DataFrame, mask_intervals: pd.DataFrame,
                 *, grid_s=.25, increment_s=5.0):
    grid=matched_grid(fields,grid_s);masked=mask_from_intervals(grid,mask_intervals);step=round(increment_s/grid_s)
    quiet_pairs=(~masked[:-step])&(~masked[step:]);rows=[];episode_rows=[]
    accepted=episodes[episodes.get("status",pd.Series("accepted",index=episodes.index)).eq("accepted")]
    measured_column="measured_shift_um" if "measured_shift_um" in accepted else "best_shift_um"
    for field in fields:
        rigid=field.at(grid,np.median(field.depth_um));rigid=rigid-np.nanmedian(rigid[~masked])
        errors=[];ratios=[];block_errors=[]
        for episode in accepted.itertuples(index=False):
            start=float(getattr(episode,"start_s"));end=float(episode.end_s if hasattr(episode,"end_s") else episode.stop_s)
            core=(grid>=start)&(grid<end);rest=(grid>=start-4)&(grid<start-1)&(~masked)
            if not core.any() or not rest.any():continue
            measured=float(getattr(episode,measured_column));pred=float(np.nanmedian(rigid[core])-np.nanmedian(rigid[rest]))
            errors.append(abs(pred-measured));ratios.append(pred/measured if measured else np.nan)
            episode_rows.append({"field":field.source,"start_s":start,"end_s":end,"measured_shift_um":measured,"field_shift_um":pred})
            for i in range(4):
                key=f"block{i}_shift_um"
                if hasattr(episode,key) and np.isfinite(getattr(episode,key)):
                    z=(i+.5)*960;value=field.at(grid,z);block_pred=float(np.nanmedian(value[core])-np.nanmedian(value[rest]));block_errors.append(abs(block_pred-float(getattr(episode,key))))
        inc=rigid[step:]-rigid[:-step]
        finite_ratios=np.asarray(ratios,float);finite_ratios=finite_ratios[np.isfinite(finite_ratios)]
        rows.append({"field":field.source,"episode_err":float(np.nanmedian(errors)) if errors else np.nan,
                     "episode_ratio":float(np.median(finite_ratios)) if len(finite_ratios) else np.nan,
                     "block_err":float(np.nanmedian(block_errors)) if block_errors else np.nan,
                     "quiet_abs":float(np.nanmedian(np.abs(rigid[~masked]))),
                     "quiet_inc":float(np.sqrt(np.nanmean(inc[quiet_pairs]**2))),
                     "false_motion_frac":float(np.nanmean(np.abs(rigid[~masked])>50)),
                     "accepted_episodes":len(errors),"quiet_samples":int((~masked).sum()),"quiet_pairs":int(quiet_pairs.sum())})
    return pd.DataFrame(rows),pd.DataFrame(episode_rows),{"grid_s":grid_s,"shared_start_s":float(grid[0]),"shared_end_s":float(grid[-1]),"quiet_samples":int((~masked).sum()),"quiet_pairs":int(quiet_pairs.sum())}


def bias_diagnostics(fields, peaks, mask_intervals, *, grid_s=.25):
    grid=matched_grid(fields,grid_s);masked=mask_from_intervals(grid,mask_intervals);edges=np.r_[grid-grid_s/2,grid[-1]+grid_s/2]
    counts=np.histogram(peaks.time_s,edges)[0];distance=np.full(len(grid),np.inf);episode_density=np.zeros(len(grid))
    for row in mask_intervals.itertuples(index=False):
        distance=np.minimum(distance,np.minimum(abs(grid-row.start_s),abs(grid-row.end_s)))
        episode_density+=((grid>=row.start_s-60)&(grid<row.end_s+60)).astype(float)
    rows=[];profiles=[]
    def spearman(a,b):
        finite=np.isfinite(a)&np.isfinite(b);a=np.asarray(a)[finite];b=np.asarray(b)[finite]
        return np.nan if len(a)<2 or np.std(a)==0 or np.std(b)==0 else float(stats.spearmanr(a,b).statistic)
    for field in fields:
        value=field.at(grid,np.median(field.depth_um));value-=np.nanmedian(value[~masked]);quiet=~masked&np.isfinite(value)
        rows.append({"field":field.source,"spearman_rate":spearman(value[quiet],counts[quiet]),
                     "spearman_episode_density":spearman(value[quiet],episode_density[quiet]),
                     "spearman_distance":spearman(value[quiet],distance[quiet])})
        distance_edges=(0,1,2,3,5,10,30,60,np.inf)
        for lo,hi in zip(distance_edges[:-1],distance_edges[1:]):
            take=quiet&(distance>=lo)&(distance<hi)
            profiles.append({"field":field.source,"profile":"distance","distance_lo_s":lo,"distance_hi_s":hi,
                             "episode_density":np.nan,"n":int(take.sum()),
                             "median_level_um":float(np.nanmedian(value[take])) if take.any() else np.nan})
        for density in np.unique(episode_density[quiet]):
            take=quiet&(episode_density==density)
            profiles.append({"field":field.source,"profile":"episode_density","distance_lo_s":np.nan,
                             "distance_hi_s":np.nan,"episode_density":density,"n":int(take.sum()),
                             "median_level_um":float(np.nanmedian(value[take]))})
    return pd.DataFrame(rows),pd.DataFrame(profiles)


def _figures(fields,peaks,out):
    import matplotlib;matplotlib.use("Agg");import matplotlib.pyplot as plt
    t0=max(min(f.time_s) for f in fields);t1=min(max(f.time_s) for f in fields);p=peaks[(peaks.time_s>=t0)&(peaks.time_s<=t1)]
    fig,ax=plt.subplots(figsize=(12,6));take=np.ones(len(p),bool)
    if "amplitude" in p and len(p):take=np.abs(p.amplitude)>=np.quantile(np.abs(p.amplitude),.8)
    ax.scatter(p.time_s[take],p.depth_um[take],s=.5,color="0.65",lw=0,rasterized=True)
    depths=np.linspace(np.nanpercentile(p.depth_um,10),np.nanpercentile(p.depth_um,90),5) if len(p) else np.linspace(0,3840,5)
    tt=np.linspace(t0,t1,min(4000,max(100,int((t1-t0)*4))))
    for i,field in enumerate(fields):
        for j,z in enumerate(depths):
            d=field.at(tt,z);d-=np.nanmedian(d);ax.plot(tt,z+d,color=OKABE_ITO[i%len(OKABE_ITO)],ls=LINESTYLES[i%len(LINESTYLES)],lw=1,label=field.source if j==0 else None)
    ax.set(xlabel="Seconds from AP frame zero",ylabel="Depth (µm)",title="Peak raster and field-predicted band paths");ax.legend(fontsize=8);fig.tight_layout();fig.savefig(out/"raster_overlay.png",dpi=160);plt.close(fig)
    if len(p):
        tb=np.linspace(t0,t1,min(1601,max(101,int((t1-t0)/.05)+1)));zb=np.linspace(float(p.depth_um.min()),float(p.depth_um.max()),257)
        h,_,_=np.histogram2d(p.time_s,p.depth_um,bins=(tb,zb));fig,ax=plt.subplots(figsize=(12,6))
        ax.imshow(np.log1p(h.T),origin="lower",aspect="auto",extent=[tb[0],tb[-1],zb[0],zb[-1]],cmap="magma_r")
        for i,field in enumerate(fields):
            for j,z in enumerate(depths):
                d=field.at(tt,z);d-=np.nanmedian(d);ax.plot(tt,z+d,color=OKABE_ITO[i%len(OKABE_ITO)],ls=LINESTYLES[i%len(LINESTYLES)],lw=1,label=field.source if j==0 else None)
        ax.set(xlabel="Seconds from AP frame zero",ylabel="Depth (µm)",title="Log peak-density overlay");ax.legend(fontsize=8);fig.tight_layout();fig.savefig(out/"density_overlay.png",dpi=160);plt.close(fig)


def build_report(fields: list[MotionField], peaks: pd.DataFrame, windows: pd.DataFrame,
                 mask_intervals: pd.DataFrame, out: str | Path, units: pd.DataFrame | None = None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);warning_rows=[]
    for field in fields:
        warning_rows += [{"field":field.source,"warning":message} for message in field.validate()]
    scores,episodes,support=score_fields(fields,windows,mask_intervals);bias,profiles=bias_diagnostics(fields,peaks,mask_intervals)
    scores.to_csv(out/"scores.csv",index=False);episodes.to_csv(out/"episode_values.csv",index=False);bias.to_csv(out/"bias_diagnostics.csv",index=False)
    profiles.to_csv(out/"leakage_profile.csv",index=False)
    _figures(fields,peaks,out)
    unit_table=None
    if units is not None:
        start=min(units.time_s);stop=max(units.time_s);centres=np.arange(start,stop,.05)+.025;quiet=~mask_from_intervals(centres,mask_intervals)
        displacement=fields[0].at(centres,np.median(fields[0].depth_um));displacement=np.where(quiet,displacement,0)
        try:unit_table=unit_common_mode(units,quiet,displacement=displacement);unit_table.to_csv(out/"unit_common_mode.csv",index=False)
        except ValueError as exc:warning_rows.append({"field":"units","warning":str(exc)})
    pd.DataFrame(warning_rows,columns=("field","warning")).to_csv(out/"warnings.csv",index=False)
    text="# motionqc field-QC report\n\nAll field scores use identical episodes, quiet samples, and 5 s endpoint pairs on shared temporal support.\n\n"
    text+="## Scores\n\n"+scores.to_markdown(index=False,floatfmt=".3f")+"\n\n## Bias diagnostics\n\n"+bias.to_markdown(index=False,floatfmt=".3f")+"\n\n"
    text+=f"Matched support: {support}.\n\n![Raster overlay](raster_overlay.png)\n\n![Density overlay](density_overlay.png)\n\n"
    if unit_table is not None:text+="## Unit common-mode spectrum\n\n"+unit_table.to_markdown(index=False,floatfmt=".3f")+"\n\n"
    if warning_rows:text+="## Warnings\n\n"+"\n".join(f"- {r['field']}: {r['warning']}" for r in warning_rows)+"\n"
    (out/"REPORT.md").write_text(text)
    manifest={"schema":"motionqc-report-v1","fields":[f.source for f in fields],"support":support,
              "outputs":{p.name:file_sha256(p) for p in out.iterdir() if p.is_file()}}
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
    return scores
