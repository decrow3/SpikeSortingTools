#!/usr/bin/env python
"""Gentle-period comparison for frozen sorter-free imec1 candidates."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from testing.luke_imec1_sorterfree_top20_motion_comparison import (
    DISCOVERY, ESTIMATORS, ORIGIN_S, SEED_INTERVAL, WINDOWS as HEAVY_WINDOWS,
    load_field, sample,
)

ROOT=Path(__file__).resolve().parents[1]
GENTLE=ROOT/"testing/outputs/luke_imec1_sorterfree_gentle_match_v1"
OUT=ROOT/"testing/outputs/luke_imec1_sorterfree_gentle_motion_comparison_v1"
GENTLE_WINDOWS=[(70,80),(110,120),(420,430),(560,570),(820,830),(850,860)]


def window_id(time_s: float) -> int:
    for index,(start,stop) in enumerate(GENTLE_WINDOWS):
        if start <= time_s < stop: return index
    return -1


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    families=pd.read_csv(DISCOVERY/"families_after_depth_reveal.csv").sort_values("candidate_rank_depth_blind").head(20)
    events=pd.read_csv(GENTLE/"gentle_events.csv"); strict=events[events.status.eq("strict")].copy()
    fields={name:load_field(folder) for name,folder in ESTIMATORS.items()}
    inventory=[]; tracks=[]; metrics=[]
    for family in families.itertuples(index=False):
        q=strict[strict.family_id.eq(family.family_id)].copy(); q["second_bin"]=np.floor(q.time_s).astype(int) if len(q) else pd.Series(dtype=int)
        b=q.groupby("second_bin",as_index=False).agg(time_s=("time_s","median"),observed_depth_um=("waveform_centroid_um","median"),strict_events=("time_s","size")) if len(q) else pd.DataFrame(columns=["time_s","observed_depth_um","strict_events"])
        if len(b): b["window_id"]=[window_id(t) for t in b.time_s]
        observed=b.observed_depth_um.to_numpy(dtype=float)-family.seed_depth_median_um
        inventory.append({"rank":int(family.candidate_rank_depth_blind),"family_id":family.family_id,"tier":family.candidate_tier,"seed_depth_um":family.seed_depth_median_um,"seed_p90_span_um":family.seed_depth_p90_span_um,"strict_events":len(q),"one_second_bins":len(b),"gentle_windows":int(b.window_id.nunique()) if len(b) else 0})
        predictions={"No motion":np.zeros(len(b))}
        for estimator,(_,_,_,interp) in fields.items():
            seed_times=np.arange(SEED_INTERVAL[0]+.5,SEED_INTERVAL[1],1.0)
            baseline=float(np.nanmedian(sample(interp,seed_times,family.seed_depth_median_um)))
            predictions[estimator]=sample(interp,b.time_s.to_numpy(dtype=float),family.seed_depth_median_um)-baseline if len(b) else np.array([])
        for estimator,predicted in predictions.items():
            centered_obs=[]; centered_pred=[]
            for wid in sorted(b.window_id.unique()) if len(b) else []:
                k=b.window_id.to_numpy()==wid
                centered_obs.extend(observed[k]-np.median(observed[k])); centered_pred.extend(predicted[k]-np.median(predicted[k]))
            centered_obs=np.asarray(centered_obs); centered_pred=np.asarray(centered_pred)
            valid=np.isfinite(observed)&np.isfinite(predicted); n=int(valid.sum())
            corr=float(np.corrcoef(observed[valid],predicted[valid])[0,1]) if estimator!="No motion" and n>=5 and np.std(observed[valid])>0 and np.std(predicted[valid])>0 else np.nan
            valid_center=np.isfinite(centered_obs)&np.isfinite(centered_pred)
            within_rmse=float(np.sqrt(np.mean((centered_obs[valid_center]-centered_pred[valid_center])**2))) if valid_center.any() else np.nan
            metrics.append({"rank":int(family.candidate_rank_depth_blind),"family_id":family.family_id,"estimator":estimator,"one_second_bins":n,"gentle_windows":int(b.window_id.nunique()) if len(b) else 0,"seed_referenced_pearson_r":corr,"seed_referenced_rmse_um":float(np.sqrt(np.mean((observed[valid]-predicted[valid])**2))) if n else np.nan,"within_window_rmse_um":within_rmse,"within_window_observed_rms_um":float(np.sqrt(np.mean(centered_obs[valid_center]**2))) if valid_center.any() else np.nan})
            for row,obs,pred in zip(b.itertuples(index=False),observed,predicted): tracks.append({"rank":int(family.candidate_rank_depth_blind),"family_id":family.family_id,"time_s":row.time_s,"window_id":row.window_id,"strict_events":row.strict_events,"observed_displacement_um":obs,"estimator":estimator,"predicted_displacement_um":pred})
    inventory=pd.DataFrame(inventory); tracks=pd.DataFrame(tracks); metrics=pd.DataFrame(metrics)
    inventory.to_csv(OUT/"top20_gentle_inventory.csv",index=False); tracks.to_csv(OUT/"gentle_tracks_and_predictions.csv",index=False); metrics.to_csv(OUT/"gentle_candidate_estimator_metrics.csv",index=False)
    eligible=inventory[(inventory.one_second_bins>=5)&(inventory.gentle_windows>=2)]
    aggregate=metrics[metrics.family_id.isin(eligible.family_id)].groupby("estimator",as_index=False).agg(families=("family_id","nunique"),median_within_window_rmse_um=("within_window_rmse_um","median"),median_seed_referenced_rmse_um=("seed_referenced_rmse_um","median"),median_seed_referenced_r=("seed_referenced_pearson_r","median"))
    aggregate.to_csv(OUT/"family_balanced_aggregate.csv",index=False)
    colors={"No motion":"#555555","DREDGE":"#007f73","KS sidecar":"#2265ac","Decentralized":"#aa7600","MEDiCINe":"#6e59a5"}
    fig,axes=plt.subplots(3,2,figsize=(14,11),sharex=True,layout="constrained")
    for ax,candidate in zip(axes.flat,eligible.itertuples(index=False)):
        g=tracks[tracks.family_id.eq(candidate.family_id)]; observed=g.drop_duplicates("time_s")
        ax.scatter(observed.time_s,observed.observed_displacement_um,c="black",s=17,label="Waveform depth")
        for estimator in ["DREDGE","KS sidecar","Decentralized","MEDiCINe"]:
            h=g[g.estimator.eq(estimator)]
            for start,stop in GENTLE_WINDOWS:
                z=h[(h.time_s>=start)&(h.time_s<stop)]
                if len(z): ax.plot(z.time_s,z.predicted_displacement_um,marker=".",lw=1,color=colors[estimator],label=estimator if start==GENTLE_WINDOWS[0][0] else None)
        ax.axhline(0,color="gray",lw=.5); ax.set_title(f"#{candidate.rank} {candidate.family_id}: {candidate.strict_events} events / {candidate.one_second_bins} bins"); ax.set_ylabel("Seed-relative depth (µm)")
    for ax in axes.flat[len(eligible):]: ax.axis("off")
    axes.flat[0].legend(fontsize=7,ncol=2); fig.supxlabel("Recording-relative time (s)"); fig.suptitle("Frozen top-20 candidates in estimator-selected relative-gentle windows\nWindow selection did not use candidate tracks; fixed sign and no fitted lag/scale")
    fig.savefig(OUT/"01_gentle_motion_overlay.png",dpi=170); fig.savefig(OUT/"01_gentle_motion_overlay.pdf"); plt.close(fig)
    order=["No motion","DREDGE","KS sidecar","Decentralized","MEDiCINe"]
    fig,ax=plt.subplots(figsize=(10,5),layout="constrained")
    x=np.arange(len(eligible)); width=.15
    for j,estimator in enumerate(order):
        z=metrics[(metrics.family_id.isin(eligible.family_id))&metrics.estimator.eq(estimator)].set_index("family_id").reindex(eligible.family_id)
        ax.bar(x+(j-2)*width,z.within_window_rmse_um,width,label=estimator,color=colors[estimator])
    ax.set_xticks(x,[f"#{r} {f}" for r,f in zip(eligible['rank'],eligible.family_id)]); ax.set_ylabel("Within-window RMSE (µm)"); ax.set_title("Gentle-period local tracking error; no-motion is the stability baseline"); ax.legend(fontsize=8,ncol=3)
    fig.savefig(OUT/"02_within_window_rmse.png",dpi=170); fig.savefig(OUT/"02_within_window_rmse.pdf"); plt.close(fig)
    summary={"schema":"luke0804-imec1-sorterfree-gentle-motion-comparison-v1","status":"complete","candidate_ranking_used_motion":False,"window_selection_used_candidate_tracks":False,"window_selection_conditional_on_motion_estimators":True,"gentle_windows_s":GENTLE_WINDOWS,"top20_strict_events":int(inventory.strict_events.sum()),"top20_active_families":int((inventory.strict_events>0).sum()),"top20_evaluable_families":len(eligible),"comparison":"One-second medians. Cross-window values use the seed reference; local error median-centers observations and predictions separately inside each 10-s window. Saved physical sign; no lag, sign, or scale fit.","limitations":["Selecting low-excursion windows from the evaluated estimators biases this toward apparent stability and cannot independently validate estimator magnitude.","Low motion reduces correlation identifiability; within-window error versus no motion is primary.","Candidates remain waveform families rather than verified cells."]}
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2))


if __name__=="__main__": main()
