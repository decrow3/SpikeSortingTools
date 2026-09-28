#!/usr/bin/env python
"""Correction S: rebuild quiet references and re-apply frozen P.4."""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from testing import luke_imec1_medicine_reference_sweep_v1 as stage1
from testing import luke_imec1_medicine_stage2_o_v1 as old
from testing import luke_imec1_medicine_stage2_p_v1 as pstage

OUTPUT=pstage.DEFAULT_OUTPUT

def read_json(p): return json.loads(Path(p).read_text())

def extra_shift_segments(measured,nulls):
    import pandas as pd
    a=measured.loc[measured.best_shift_um.notna() & (measured.best_shift_um!=0),["window","start_s","stop_s"]].copy()
    b=nulls.loc[nulls.resolved & nulls.best_shift_um.notna() & (nulls.best_shift_um!=0),["window","start_s","stop_s"]].copy()
    a["source"]="candidate_shift_nonzero"; b["source"]="null_shift_nonzero"
    return pd.concat([a,b],ignore_index=True)

def field_flag_shift_tests(output,candidates):
    """Union >50 um field segments, then adjudicate with raw peak maps."""
    import pandas as pd
    flag_rows=[]; measured_parts=[]; combined_candidates=[]; combined_measured=[]
    for wi,(w,(start,stop)) in enumerate(old.HELDOUT_WINDOWS.items()):
        centers=np.arange(start+stage1.BIN/2,stop,stage1.BIN); flag=np.zeros(len(centers),bool)
        for config in old.CONFIGS:
            field=pstage.load_field(pstage.field_path(output,config,w)); ft=np.asarray(field["session_time_s"],float); fy=np.asarray(field["displacement_um"],float)
            sampled=np.column_stack([np.interp(centers,ft,fy[:,j],left=np.nan,right=np.nan) for j in range(fy.shape[1])])
            sampled-=np.nanmedian(sampled,axis=0,keepdims=True); rigid=np.nanmedian(sampled,axis=1); flag|=np.abs(rigid)>50
        base=candidates.loc[candidates.window==w].copy()
        for x in base.itertuples(): flag&=~((centers>=x.start_s-2)&(centers<=x.stop_s+2))
        flags=[]
        for j,(i0,i1,a,b) in enumerate(pstage.mref.intervals(flag,centers)):
            flags.append({"window":w,"episode_id":10000+j,"start_s":a,"stop_s":b,"duration_s":b-a,"n_bins":i1-i0,
                "field_p5_source":False,"low_rate_source":False,"invalid_eye_source":False,"source":"union_config_abs_centered_rigid_gt50"})
        fdf=pd.DataFrame(flags)
        if len(fdf): flag_rows.append(fdf)
        base=base.copy(); base["episode_id"]=np.arange(len(base)); base["source"]="shared_or_rate_candidate"
        combined=pd.concat([base,fdf],ignore_index=True); combined_candidates.append(combined)
        measured,_=old.measure_window(pstage.peaks(output,w),combined,(start,stop),20260930+wi); combined_measured.append(measured)
        if len(fdf): measured_parts.append(measured.loc[measured.episode_id>=10000])
    flags=pd.concat(flag_rows,ignore_index=True) if flag_rows else pd.DataFrame(columns=["window","episode_id","start_s","stop_s","duration_s","n_bins"])
    fmeas=pd.concat(measured_parts,ignore_index=True) if measured_parts else pd.DataFrame()
    return flags,fmeas,pd.concat(combined_candidates,ignore_index=True),pd.concat(combined_measured,ignore_index=True)

def exclusions(candidates,segments,window):
    import pandas as pd
    a=candidates.loc[candidates.window==window,["window","start_s","stop_s"]].copy(); a["source"]="candidate_union"
    return pd.concat([a,segments.loc[segments.window==window]],ignore_index=True)

def quiet_scores(output,candidates,measured,nulls):
    import pandas as pd
    segments=extra_shift_segments(measured,nulls); rows=[]
    for config in old.CONFIGS:
        qabs,incs,falsefracs=[],[],[]
        for w,bounds in old.HELDOUT_WINDOWS.items():
            field=pstage.load_field(pstage.field_path(output,config,w)); exc=exclusions(candidates,segments,w)
            # A nonzero shift in a pseudo-episode is real-motion evidence, not
            # a quiet reference.  Retain only resolved zero-shift nulls.
            for n in nulls.loc[(nulls.window==w)&nulls.resolved&(nulls.best_shift_um==0)].itertuples():
                qabs.append(abs(pstage.mref.episode_prediction(field,n.start_s,n.stop_s)))
            values,_=old.corrected_quiet_values(field,exc,bounds); incs.extend(values)
            falsefracs.append(pstage.false_motion(field,exc,bounds))
        rows.append({"config":config,"quiet_abs":float(np.median(qabs)) if qabs else np.nan,
            "quiet_inc":float(np.sqrt(np.mean(np.square(incs)))) if incs else np.nan,
            "false_motion_frac":float(np.mean(falsefracs)) if falsefracs else np.nan,
            "resolved_zero_shift_nulls":len(qabs),"quiet_increment_values":len(incs)})
    return pd.DataFrame(rows),segments

def run(output):
    import pandas as pd
    target=output/"s_correction"; target.mkdir(exist_ok=False)
    # S.2 uses exactly the already-frozen episode-window candidate builder:
    # rate below half median OR centred shared-recovery rigid <= -40 um.
    cand,bins=pstage.candidates(output,old.HELDOUT_WINDOWS)
    measured=pstage.measure_only(output,cand,old.HELDOUT_WINDOWS)
    nulls=pstage.make_nulls(output,cand,measured,old.HELDOUT_WINDOWS,20260927)
    validation=pstage.validate_nulls_r(nulls,old.HELDOUT_WINDOWS)
    if validation["status"]!="pass": raise RuntimeError("S quiet null validation failed")
    flags,flag_measured,combined_candidates,combined_measured=field_flag_shift_tests(output,cand)
    blocks=pstage.block_references(output,combined_candidates,combined_measured,old.HELDOUT_WINDOWS)
    cand.to_csv(target/"candidate_episodes.csv",index=False); bins.to_csv(target/"candidate_source_bins.csv",index=False)
    measured.to_csv(target/"measured_episodes.csv",index=False); flags.to_csv(target/"shift_test_segments.csv",index=False); flag_measured.to_csv(target/"shift_test_segment_measurements.csv",index=False)
    combined_measured.to_csv(target/"all_quiet_measurements.csv",index=False); nulls.to_csv(target/"null_pseudo_episodes.csv",index=False); blocks.to_csv(target/"block_measurements.csv",index=False)
    stage1.atomic_json(target/"null_validation.json",validation)
    # Candidate intervals are always excluded.  Outside them, exclude only
    # field-flagged or pseudo-episode segments whose raw-map shift is nonzero.
    flag_nonzero=flag_measured.loc[flag_measured.best_shift_um.notna()&(flag_measured.best_shift_um!=0),["window","start_s","stop_s"]].copy(); flag_nonzero["source"]="field_flag_shift_nonzero"
    quiet_measured=pd.concat([measured,flag_measured],ignore_index=True)
    quiet,segments=quiet_scores(output,cand,quiet_measured,nulls); segments=pd.concat([segments,flag_nonzero],ignore_index=True).drop_duplicates()
    # Re-score with the complete exclusion table rather than rebuilding it in
    # quiet_scores, so union-field adjudications apply to every config fairly.
    rows=[]
    for config in old.CONFIGS:
        qabs=[]; incs=[]; fms=[]
        for w,bounds in old.HELDOUT_WINDOWS.items():
            field=pstage.load_field(pstage.field_path(output,config,w)); exc=exclusions(cand,segments,w)
            for n in nulls.loc[(nulls.window==w)&nulls.resolved&(nulls.best_shift_um==0)].itertuples(): qabs.append(abs(pstage.mref.episode_prediction(field,n.start_s,n.stop_s)))
            vals,_=old.corrected_quiet_values(field,exc,bounds); incs.extend(vals); fms.append(pstage.false_motion(field,exc,bounds))
        rows.append({"config":config,"quiet_abs":float(np.median(qabs)),"quiet_inc":float(np.sqrt(np.mean(np.square(incs)))) if incs else np.nan,
            "false_motion_frac":float(np.mean(fms)),"resolved_zero_shift_nulls":len(qabs),"quiet_increment_values":len(incs)})
    quiet=pd.DataFrame(rows); quiet.to_csv(target/"scores_quiet_corrected.csv",index=False); segments.to_csv(target/"additional_nonzero_shift_exclusions.csv",index=False)

    eroot=output/"references_episode"; ec=pd.read_csv(eroot/"candidate_episodes.csv"); em=pd.read_csv(eroot/"measured_episodes.csv"); en=pd.read_csv(eroot/"null_pseudo_episodes.csv"); eb=pd.read_csv(eroot/"block_measurements.csv")
    allcand=pd.concat([ec,combined_candidates],ignore_index=True); allmeas=pd.concat([em,combined_measured],ignore_index=True); allnull=pd.concat([en,nulls],ignore_index=True); allblocks=pd.concat([eb,blocks],ignore_index=True)
    windows={**pstage.episode_windows(output),**old.HELDOUT_WINDOWS}
    episode=pstage.score_set(output,windows,allcand,allmeas,allnull,allblocks,False)
    episode.to_csv(target/"scores_episode_pool_corrected.csv",index=False)
    choice=pstage.select(quiet,episode)
    if choice["outcome"]!="selected": choice={**choice,"outcome":"provisional","config":"depth2","basis":"R.4 development-selected fallback after S"}
    accepted=combined_measured.loc[combined_measured.accepted].groupby("window").size().to_dict()
    stage1.atomic_json(target/"selection_s.json",{**choice,"quiet_accepted_episodes":accepted,
        "episode_pool_accepted":int(allmeas.accepted.sum()),"null_validation":validation,
        "reference_rule":"rate <0.5x median OR saved shared-recovery centred rigid <=-40um; gaps <=1s; raw-map shift test",
        "quiet_exclusion":"candidate episodes +/-2s plus every resolved nonzero-shift candidate/null segment",
        "created_at":time.time(),"no_sort":True,"voltage_modified":False})
    lines=["# Correction S — quiet held-out references and P.4 reselection","",
        "No sorting or voltage modification was performed.","",
        f"Selection outcome: **{choice['outcome']} `{choice['config']}`**.",
        f"Accepted episodes gained from quiet windows: {accepted}; corrected pooled total: {int(allmeas.accepted.sum())}.",
        f"All rebuilt nulls passed: {validation['status']}.","",
        "Artifacts: `candidate_episodes.csv`, `measured_episodes.csv`, `null_pseudo_episodes.csv`, `block_measurements.csv`, `scores_quiet_corrected.csv`, `scores_episode_pool_corrected.csv`, and `selection_s.json`.",""]
    (target/"README.md").write_text("\n".join(lines))

def report_services(output):
    target=output/"s_correction"; services={
        "held_depth2_full":{"unit":"luke-medicine-q-full-20260924-v2.service","action":"stopped before S rescoring; no full-session fit ran"},
        "depth2_machinery_pilot":{"unit":"luke-medicine-q-pilot-20260924-v3.service","action":"stopped pre-fit when a 600 s frontend materialization reached its 48 GB memory bound; partial diagnostics preserved, temporary preprocessing audited then discarded"},
        "selected_pilot":{"unit":"luke-medicine-q-amp50-d1-pilot-20260924.service","action":"running Q.1 for S-selected amp50_d1 with 240 s bounded frontend sources"},
        "selected_full":{"unit":"luke-medicine-q-amp50-d1-full-20260924.service","action":"queued; proceeds only after the amp50_d1 Q.1 gate passes"}}
    stage1.atomic_json(target/"q_services_s.json",{"services":services,"no_sort":True,"voltage_modified":False,"recorded_at":time.time()})
    with (target/"README.md").open("a") as f:
        f.write("\n## Q services\n\n")
        for x in services.values(): f.write(f"- `{x['unit']}`: {x['action']}.\n")

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--output",type=Path,default=OUTPUT); ap.add_argument("--services-only",action="store_true"); a=ap.parse_args()
    report_services(a.output.resolve()) if a.services_only else run(a.output.resolve())
if __name__=="__main__": main()
