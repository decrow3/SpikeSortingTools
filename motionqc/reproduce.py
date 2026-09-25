"""Frozen Luke0804/Bacon phase-1 reproduction checks for AH.5/AH.6."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

from .field import load_field, file_sha256
from .reference import canonical_mask, episode_candidates, matched_null, shift_test, write_canonical_csv
from .report import build_report, load_peaks

REPO=Path(__file__).resolve().parents[1]
SWEEP=Path("/media/huklab/Data/NPX/Ryansorting/Luke/luke_imec1_medicine_reference_sweep_v2")
HUB=Path("/media/huklab/Data/NPX/Ryansorting/Luke/incoming/hub_analysis_v1")
HUB_MASK=Path("/media/huklab/Data/NPX/Ryansorting/Luke/incoming/censor_mask_v1/censor_mask_v1.csv")
GABOR_START=2091.250454760595


def absolute_population(window):
    cfg=json.loads((SWEEP/"extraction_config.json").read_text());spec=next(x for x in cfg["windows"] if x["id"]==f"{window}_240")
    path=SWEEP/f"peak_cache/{window}_240/extraction/population.npz"
    with np.load(path,allow_pickle=False) as z:
        return pd.DataFrame({"time_s":np.asarray(z["time_s"])+spec["start_s"],"depth_um":z["depth_um"],"x_um":z["x_um"],"amplitude":z["amplitude"]})


def stage1_m_reproduction(out):
    expected=pd.read_csv(SWEEP/"references_m/measured_episodes.csv");candidates=pd.read_csv(SWEEP/"references_m/candidate_episodes.csv");rows=[]
    for window,group in expected.groupby("window",sort=False):
        peaks=absolute_population(window);all_candidates=candidates[candidates.window.eq(window)]
        active=np.zeros(len(peaks),bool)
        for ep in all_candidates.itertuples(index=False):active|=(peaks.time_s>=ep.start_s)&(peaks.time_s<ep.stop_s)
        for ep in group.itertuples(index=False):
            episode=(peaks.time_s>=ep.start_s)&(peaks.time_s<ep.stop_s);rest=(peaks.time_s>=ep.start_s-4)&(peaks.time_s<ep.start_s-1)&(~active)
            result=shift_test(peaks,episode,rest)
            accepted=bool(ep.rest_fully_in_window and result["accepted"])
            rows.append({"window":window,"episode_id":ep.episode_id,"expected_accepted":bool(ep.accepted),"actual_accepted":accepted,
                         "expected_shift_um":ep.best_shift_um,"actual_shift_um":result["best_shift_um"]})
    frame=pd.DataFrame(rows);frame.to_csv(out/"stage1_m_reproduction.csv",index=False)
    expected_set=set(map(tuple,frame.loc[frame.expected_accepted,["window","episode_id"]].to_numpy()))
    actual_set=set(map(tuple,frame.loc[frame.actual_accepted,["window","episode_id"]].to_numpy()))
    shifts=frame[frame.expected_accepted&frame.actual_accepted]
    return {"expected_accepted":len(expected_set),"actual_accepted":len(actual_set),"accepted_set_exact":expected_set==actual_set,
            "best_shifts_exact":bool(np.array_equal(shifts.expected_shift_um.to_numpy(),shifts.actual_shift_um.to_numpy()))}


def mask_reproduction(out):
    q=SWEEP/"stage3_q";z=np.load(q/"luke0804_imec1_medicine_deployable_motion.npz");cat=pd.read_csv(q/"episode_catalogue.csv")
    frame,mask,source,_=canonical_mask(z["time_s"],z["displacement_um"],cat);path=out/"censor_mask_v1.csv";digest=write_canonical_csv(frame,path)
    return {"intervals":len(frame),"seconds":float(mask.sum()*.25),"sha256":digest,
            "expected_sha256":"86425e8a6627a10ac8932ba01b799cb7be7141f731ebb4ecc01e303326b9da55","exact":digest==file_sha256(HUB_MASK)}


def t8_reproduction(out):
    peaks=pd.read_csv(HUB/"imec1_ap_peaks_2091_2431.csv.gz");field=load_field(HUB/"340_native_field.npz",source="gabor_native")
    # Portable field time is relative to the Gabor window.
    if field.time_s[-1]<1000:field.time_s=field.time_s+GABOR_START
    tg=np.arange(GABOR_START,GABOR_START+340,.25)+.125;r=field.at(tg,2920);r-=np.median(r);core=r<=np.percentile(r,5);core=ndimage.binary_closing(core,structure=np.ones(5));labels,n=ndimage.label(core)
    intervals=[];rows=[]
    for k in range(1,n+1):
        ix=np.flatnonzero(labels==k);a=tg[ix[0]]-.125;b=tg[ix[-1]]+.125;intervals.append((a,b))
    peak_core=np.zeros(len(peaks),bool)
    for a,b in intervals:peak_core|=(peaks.time_s>=a)&(peaks.time_s<b)
    for a,b in intervals:
        ep=(peaks.time_s>=a)&(peaks.time_s<b);rest=(peaks.time_s>=a-4)&(peaks.time_s<a-1)&(~peak_core)
        value=shift_test(peaks,ep,rest,depth_range_um=(2020,3820),x_range_um=(-8,64),min_gain=0)
        rows.append({"start_s":a,"end_s":b,"best_shift_um":value["best_shift_um"],"episode_peaks":value["episode_peaks"],"rest_peaks":value["rest_peaks"]})
    frame=pd.DataFrame(rows);frame.to_csv(out/"t8_gabor_native.csv",index=False);scored=frame.dropna(subset=["best_shift_um"])
    quiet=np.abs(r)<20;bad=[]
    # Convert nonquiet field samples to intervals for the generic matched null.
    lab,nn=ndimage.label(~quiet)
    for k in range(1,nn+1):
        ix=np.flatnonzero(lab==k);bad.append((tg[ix[0]]-.125,tg[ix[-1]]+.125))
    counts=[int(((peaks.time_s>=a)&(peaks.time_s<b)).sum()) for a,b in intervals]
    null,gate=matched_null(peaks,[b-a for a,b in intervals],bad,(GABOR_START,GABOR_START+340),n=60,seed=0,
                           peak_targets=[int(np.median(counts))],placement_margin_s=0,quantisation_um=10,
                           depth_range_um=(2020,3820),x_range_um=(-8,64),min_gain=0)
    null.to_csv(out/"t8_gabor_null.csv",index=False)
    return {"episodes":len(frame),"scored":len(scored),"in_expected_range":int(scored.best_shift_um.between(-240,-160).sum()),
            "shifts":scored.best_shift_um.tolist(),"null_zero":int((null.best_shift_um==0).sum()),"null_n":len(null),"null_gate":gate,
            "mismatch_cause":"The supplied hub t8_episode_shift.py run on the supplied NPX peaks and portable native field also gives -80 um for the final 2429.000-2430.250 s episode; the stated 10/10 target says -160..-240. motionqc matches the supplied implementation and does not tune this away."}


def existing_counts_and_q():
    s=pd.read_csv(SWEEP/"stage2_o/s_correction/scores_episode_pool_corrected.csv");q=pd.read_csv(SWEEP/"stage3_q/matched_quiet_increment_differences.csv")
    q=q[(q["set"]=="quiet_heldout_s_corrected")&(q["mode"]=="frozen_nonoverlap")]
    stitched=float(np.sqrt(np.mean(q.stitched_increment_um**2)));per=float(np.sqrt(np.mean(q.per_window_increment_um**2)))
    return {"s_episode_pool_count":int(s.accepted_episodes.max()),"s_expected":45,
            "q_matched_pairs":len(q),"q_stitched_quiet_inc":stitched,"q_per_window_quiet_inc":per,
            "q_within_0p1":bool(abs(stitched-6.307240)<.1 and abs(per-6.064876)<.1)}


def generality_reports(out):
    mask=pd.read_csv(HUB_MASK)
    # Luke p10: Q deployable field, Q catalogue episodes, cached pilot peaks.
    q=SWEEP/"stage3_q";luke_field=load_field(q/"luke0804_imec1_medicine_deployable_motion.npz",source="Luke Q")
    luke_peaks=absolute_population("p10");episodes=pd.read_csv(q/"episode_catalogue.csv");episodes=episodes[(episodes.start_s<1107.4)&(episodes.end_s>987.3)]
    build_report([luke_field],luke_peaks,episodes,mask,out/"luke_p10")
    # Bacon p50 cached pilot. It has no x coordinate, so depth-only null uses x=0.
    root=REPO/"testing/outputs/cross_dataset_fast_motion_v2/bacon_probeA_p50";bacon_field=load_field(root/"fit/field.npz",source="Bacon MEDiCINe")
    bacon=load_peaks(root/"extraction/population.npz");bacon["x_um"]=0.0
    grid=np.arange(bacon_field.time_s[0],bacon_field.time_s[-1],.25);candidate,_=episode_candidates(grid,fields=[bacon_field])
    intervals=[]
    lab,n=ndimage.label(candidate)
    for k in range(1,n+1):ix=np.flatnonzero(lab==k);intervals.append((grid[ix[0]],grid[ix[-1]]+.25))
    empty=pd.DataFrame(columns=["start_s","end_s","status","measured_shift_um"]);empty_mask=pd.DataFrame(columns=["start_s","end_s","source"])
    build_report([bacon_field],bacon,empty,empty_mask,out/"bacon_p50")
    null,gate=matched_null(bacon,[1.0],intervals,(float(bacon.time_s.min()),float(bacon.time_s.max())),n=20,seed=4,min_gain=0)
    null.to_csv(out/"bacon_p50/null.csv",index=False)
    return {"bacon_candidate_episodes":len(intervals),"bacon_null":gate,"bacon_rigid_p95_p5_um":float(np.ptp(np.percentile(bacon_field.rigid(),[5,95]))),
            "luke_accepted_episodes":int((episodes.status=="accepted").sum())}


def run(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);result={}
    result["stage1_m"]=stage1_m_reproduction(out);result["s_and_q"]=existing_counts_and_q();result["canonical_mask"]=mask_reproduction(out);result["t8"]=t8_reproduction(out);result["generality"]=generality_reports(out)
    rows=[
        ("Stage-1 M accepted set",result["stage1_m"]["accepted_set_exact"],result["stage1_m"]),
        ("Stage-1 M best shifts",result["stage1_m"]["best_shifts_exact"],result["stage1_m"]),
        ("S corrected 45-episode pool",result["s_and_q"]["s_episode_pool_count"]==45,result["s_and_q"]),
        ("AE canonical mask hash",result["canonical_mask"]["exact"],result["canonical_mask"]),
        ("Hub T8 episodes",result["t8"]["scored"]==10 and result["t8"]["in_expected_range"]==10,result["t8"]),
        ("Hub T8 null",result["t8"]["null_zero"]>=59,result["t8"]),
        ("Q.2 matched quiet_inc",result["s_and_q"]["q_within_0p1"],result["s_and_q"]),
        ("Bacon no episodes/pass null",result["generality"]["bacon_candidate_episodes"]==0 and result["generality"]["bacon_null"]["pass"],result["generality"]),
        ("Luke p10 has episodes",result["generality"]["luke_accepted_episodes"]>0,result["generality"]),]
    table=pd.DataFrame([{"target":name,"pass":passed,"details":json.dumps(details,default=str)} for name,passed,details in rows]);table.to_csv(out/"reproduction_table.csv",index=False)
    (out/"REPRODUCTION.md").write_text("# motionqc phase-1 reproductions\n\n"+table.to_markdown(index=False)+"\n")
    (out/"result.json").write_text(json.dumps(result,indent=2,sort_keys=True,default=str)+"\n")
    return table


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--out",type=Path,required=True);args=parser.parse_args();print(run(args.out).to_string(index=False))


if __name__=="__main__":main()
