#!/usr/bin/env python
"""Correction P continuation of the no-sort Luke0804 imec1 MEDiCINe sweep."""
from __future__ import annotations

import argparse, json, shutil, subprocess, sys, time, zipfile
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from testing import luke_imec1_medicine_reference_sweep_v1 as stage1
from testing import luke_imec1_medicine_m_reference_v1 as mref
from testing import luke_imec1_medicine_stage2_o_v1 as old

PARENT, DEFAULT_OUTPUT = old.PARENT, old.DEFAULT_OUTPUT
QUIET_WINDOWS, DEV_WINDOWS, CONFIGS = old.HELDOUT_WINDOWS, old.DEV_WINDOWS, old.CONFIGS
DEV_REUSE = {"amp50", "amp25", "depth2"}
SHARED_FIELD = ROOT / "testing/outputs/motion_saved_array_audit_20260923/export/full_session/shared_recovery/motion/fields.npz"
GPU_BUDGET_S, MAX_EPISODE_WINDOWS = 7 * 3600.0, 8

def read_json(p): return json.loads(Path(p).read_text())

def dynamic_windows(output):
    return {x["id"]: (float(x["start_s"]), float(x["stop_s"])) for x in read_json(output/"extraction_config.json")["windows"]}

def episode_windows(output): return {k:v for k,v in dynamic_windows(output).items() if k.startswith("e")}

def population(output, window, reference=True):
    if window in DEV_WINDOWS: return stage1.population_source(PARENT, window, 240 if reference else 120)
    row=next(x for x in read_json(output/"extraction_config.json")["windows"] if x["id"]==window)
    return output/f"peak_cache/{window}/extraction/population.npz",float(row["start_s"])

def peaks(output, window):
    path,start=population(output,window)
    with np.load(path,allow_pickle=False) as z:
        return {"time":np.asarray(z["time_s"],float)+start,"depth":np.asarray(z["depth_um"],float),"x":np.asarray(z["x_um"],float)}

def tile_ranking():
    import pandas as pd
    with np.load(SHARED_FIELD,allow_pickle=False) as z:
        times=np.asarray(z["time_bin_centers_s"],float); rigid=np.nanmedian(np.asarray(z["displacement"],float),axis=0)
    excluded=list(DEV_WINDOWS.values())+list(QUIET_WINDOWS.values()); rows=[]
    for i,start in enumerate(np.arange(0.,old.SESSION_DURATION_S-120+1e-9,120.)):
        stop=start+120; overlap=any(max(start,a-60)<min(stop,b+60) for a,b in excluded)
        mask=(times>=start)&(times<stop); centred=rigid[mask]-np.nanmedian(rigid[mask])
        rows.append({"tile_index":i,"start_s":start,"stop_s":stop,"center_s":start+60,
            "session_third":min(3,1+int((start+60)//(old.SESSION_DURATION_S/3))),
            "excluded_margin_overlap":overlap,"bins_at_or_below_minus40":int(np.sum(centred<=-40)),"field_bins":int(mask.sum())})
    t=pd.DataFrame(rows); e=t.loc[~t.excluded_margin_overlap].sort_values(["bins_at_or_below_minus40","tile_index"],ascending=[False,True])
    t["eligible_rank"]=np.nan; t.loc[e.index,"eligible_rank"]=np.arange(1,len(e)+1)
    return t.sort_values("tile_index").reset_index(drop=True)

def initial_tiles(ranking):
    chosen=[]; counts={1:0,2:0,3:0}
    for row in ranking.loc[~ranking.excluded_margin_overlap].sort_values("eligible_rank").itertuples():
        if counts[row.session_third]>=2: continue
        chosen.append(int(row.tile_index)); counts[row.session_third]+=1
        if len(chosen)==4: break
    return chosen

def append_tile(output,tile_index):
    cfgp=output/"extraction_config.json"; cfg=read_json(cfgp)
    previous=[x for x in cfg["windows"] if int(x.get("tile_index",-1))==tile_index]
    if previous: return previous[0]["id"]
    row=tile_ranking().loc[lambda x:x.tile_index==tile_index].iloc[0]
    used=[int(x["id"][1:]) for x in cfg["windows"] if x["id"].startswith("e")]; name=f"e{max(used,default=0)+1:02d}"
    record=next(x for x in cfg["records"] if x["dataset"]=="Luke" and x["probe"]=="imec1"); fs=float(record["sampling_frequency_hz"])
    a,b=round(row.start_s*fs),round(row.stop_s*fs)
    cfg["windows"].append({"id":name,"dataset":"Luke","probe":"imec1","fraction":None,"tile_index":tile_index,
        "start_frame":a,"stop_frame":b,"start_s":a/fs,"stop_s":b/fs})
    stage1.atomic_json(cfgp,cfg); return name

def prepare(output):
    receipts=list((output/"fields").glob("*/*/receipt.json")) if (output/"fields").exists() else []
    if receipts: raise RuntimeError("Cannot revise window selection after a stage-2 fit")
    ranking=tile_ranking(); ranking.to_csv(output/"episode_tile_ranking.csv",index=False)
    for i in initial_tiles(ranking): append_tile(output,i)
    if (output/"preregistration.json").exists() and not (output/"preregistration_o.json").exists(): shutil.copy2(output/"preregistration.json",output/"preregistration_o.json")
    stage1.atomic_json(output/"preregistration.json",{
        "schema":"luke0804-imec1-medicine-stage2-p-v1","created_before_any_stage2_fit":True,
        "shared_recovery_field":str(SHARED_FIELD),"shared_recovery_sha256":stage1.sha256(SHARED_FIELD),
        "tile_rule":"120s tiles; stage1/quiet +/-60s excluded; rank centred rigid bins <=-40um",
        "initial_third_cap":2,"initial_tile_indices":initial_tiles(ranking),
        "extension_rule":"if <10 accepted, append next global rank up to 8 total",
        "quiet_windows":QUIET_WINDOWS,"configs":CONFIGS,"development_reuse":sorted(DEV_REUSE),
        "quiet_gate":{"quiet_abs_max_um":10,"quiet_inc_max_um":8,"false_motion_frac_max":.05,"threshold_um":50},
        "gpu_budget_s":GPU_BUDGET_S,"no_sort":True,"voltage_modified":False,"status":"initial_episode_windows_frozen"})

def candidates(output,windows):
    import pandas as pd
    with np.load(SHARED_FIELD,allow_pickle=False) as z:
        shared_t=np.asarray(z["time_bin_centers_s"],float)
        shared_rigid=np.nanmedian(np.asarray(z["displacement"],float),axis=0)
    rows,bins=[],[]
    for window,(start,stop) in windows.items():
        pk=peaks(output,window); centers=np.arange(start+stage1.BIN/2,stop,stage1.BIN)
        counts=np.histogram(pk["time"],bins=np.r_[centers-stage1.BIN/2,stop])[0]; threshold=.5*float(np.median(counts))
        raw=counts<threshold
        field=np.interp(centers,shared_t,shared_rigid); field-=np.nanmedian(field)
        field_mask=field<=-40.0
        joined=mref.join_short_gaps(raw|field_mask,round(mref.MAX_JOIN_GAP_S/stage1.BIN))
        bins.extend({"window":window,"bin_center_s":centers[i],"source":"rate_below_half_median","value":counts[i],"threshold":threshold} for i in np.flatnonzero(raw))
        bins.extend({"window":window,"bin_center_s":centers[i],"source":"shared_recovery_centered_le_minus40",
            "value":field[i],"threshold":-40.0} for i in np.flatnonzero(field_mask))
        for eid,(i0,i1,a,b) in enumerate(mref.intervals(joined,centers)):
            rows.append({"window":window,"episode_id":eid,"start_s":a,"stop_s":b,"duration_s":b-a,"n_bins":i1-i0,
                "field_p5_source":bool(field_mask[i0:i1].any()),"field_candidate_rule":"shared_recovery_centered_le_minus40",
                "low_rate_source":bool(raw[i0:i1].any()),"invalid_eye_source":False})
    return pd.DataFrame(rows),pd.DataFrame(bins)

def measure_only(output,cand,windows):
    import pandas as pd
    parts=[]
    for i,(w,bounds) in enumerate(windows.items()):
        measured,_=old.measure_window(peaks(output,w),cand.loc[cand.window==w],bounds,20260924+i); parts.append(measured)
    return pd.concat(parts,ignore_index=True) if parts else pd.DataFrame()

def make_nulls(output,cand,measured,windows,seed):
    import pandas as pd
    pool=pd.read_csv(PARENT/"references_m/measured_episodes.csv").loc[lambda x:x.accepted]
    rows=[]; rng=np.random.default_rng(seed)
    for window,(start,stop) in windows.items():
        pk=peaks(output,window); group=cand.loc[cand.window==window]; centers=np.arange(start+stage1.BIN/2,stop,stage1.BIN)
        # R.1: both pseudo episode and its rest window must avoid every
        # candidate expanded by +/-2 s. Every window uses the pooled accepted
        # stage-1 duration/count templates, irrespective of candidate count.
        forbidden=np.zeros(len(centers),bool)
        for x in group.itertuples(): forbidden|=(centers>=x.start_s-2)&(centers<x.stop_s+2)
        templates=pool.iloc[rng.permutation(len(pool))]; used_starts=set(); resolved=0
        for j,x in enumerate(templates.itertuples()):
            nbin=max(1,int(round(float(x.duration_s)/stage1.BIN))); ne,nr=int(x.episode_peaks),int(x.rest_peaks); places=[]
            for i0 in range(round(4/stage1.BIN),len(forbidden)-nbin+1):
                i1=i0+nbin; r0=i0-round(4/stage1.BIN); r1=i0-round(1/stage1.BIN)
                if i0 in used_starts or forbidden[i0:i1].any() or forbidden[r0:r1].any(): continue
                a=centers[i0]-stage1.BIN/2; b=centers[i1-1]+stage1.BIN/2; ra,rb=a-4,a-1
                if ((pk["time"]>=a)&(pk["time"]<b)).sum()>=ne and ((pk["time"]>=ra)&(pk["time"]<rb)).sum()>=nr: places.append((a,b,ra,rb))
            base={"window":window,"matched_episode_id":int(getattr(x,"episode_id",j)),"template_source":"stage1_accepted_pool",
                "template_window":str(getattr(x,"window",window)),"target_episode_peaks":ne,"target_rest_peaks":nr,"duration_s":nbin*stage1.BIN}
            if not places: rows.append({**base,"resolved":False,"reason":"no matched quiet placement"}); continue
            a,b,ra,rb=places[int(rng.integers(len(places)))]; result=mref.map_shift(pk,(pk["time"]>=a)&(pk["time"]<b),(pk["time"]>=ra)&(pk["time"]<rb),rng,ne,nr)
            used_starts.add(int(round((a-start)/stage1.BIN)))
            is_resolved=bool(np.isfinite(result["best_shift_um"])); resolved+=int(is_resolved)
            rows.append({**base,"start_s":a,"stop_s":b,"rest_start_s":ra,"rest_stop_s":rb,
                **{k:v for k,v in result.items() if k!="correlations"},"resolved":is_resolved})
            if resolved>=20: break
    return pd.DataFrame(rows)

def validate_nulls_r(nulls,windows):
    result={"status":"pass","criterion":"per-window mode=0um and median(abs(shift))<=10um","windows":{}}
    for window in windows:
        values=nulls.loc[(nulls.window==window)&nulls.resolved,"best_shift_um"].to_numpy(float)
        if not len(values):
            result["windows"][window]={"n":0,"status":"fail_no_resolved_nulls"}; result["status"]="fail"; continue
        unique,counts=np.unique(values,return_counts=True); mode=float(unique[np.argmax(counts)]); medabs=float(np.median(np.abs(values)))
        status="pass" if mode==0 and medabs<=10 else "fail"
        if status!="pass": result["status"]="fail"
        result["windows"][window]={"n":len(values),"mode_um":mode,"median_abs_um":medabs,
            "exact_zero_fraction":float(np.mean(values==0)),"status":status,
            "counts":{str(int(k)):int(v) for k,v in zip(unique,counts)}}
    return result

def block_references(output,cand,measured,windows):
    import pandas as pd
    rows=[]
    for window in windows:
        pk=peaks(output,window); group=cand.loc[cand.window==window]; active=np.zeros(len(pk["time"]),bool)
        for x in group.itertuples(): active|=(pk["time"]>=x.start_s)&(pk["time"]<x.stop_s)
        for ep in measured.loc[(measured.window==window)&measured.accepted].itertuples():
            for bi,(lo,hi) in enumerate(old.BLOCKS):
                support=(pk["depth"]>=lo-150)&(pk["depth"]<hi+150); bp={k:v[support] for k,v in pk.items()}
                value=mref.map_shift(bp,(bp["time"]>=ep.start_s)&(bp["time"]<ep.stop_s),
                    (bp["time"]>=ep.start_s-4)&(bp["time"]<ep.start_s-1)&~active[support])
                accepted=value["episode_peaks"]>=mref.MIN_PEAKS and value["rest_peaks"]>=mref.MIN_PEAKS and np.isfinite(value["corr_gain"]) and value["corr_gain"]>=mref.MIN_CORR_GAIN
                rows.append({"window":window,"episode_id":int(ep.episode_id),"start_s":ep.start_s,"stop_s":ep.stop_s,
                    "block_index":bi,"block_low_um":lo,"block_high_um":hi,"block_center_um":(lo+hi)/2,
                    **{k:v for k,v in value.items() if k!="correlations"},"accepted":bool(accepted)})
    return pd.DataFrame(rows)

def extract(output,window):
    if (output/f"peak_cache/{window}/extraction/complete.json").exists(): return
    r=subprocess.run([str(stage1.DSPY),str(Path(__file__).resolve()),"extract-one","--output",str(output),"--window",window],timeout=old.EXTRACTION_TIMEOUT_S)
    if r.returncode: raise RuntimeError(f"Extraction failed: {window}")

def quiet_null_gate(output):
    import pandas as pd
    ref=output/"references_heldout"
    for name in ("null_pseudo_episodes.csv","null_validation.json"):
        src=ref/name; dst=ref/name.replace(".","_o_failed.",1)
        if src.exists() and not dst.exists(): shutil.copy2(src,dst)
    cand=pd.read_csv(ref/"candidate_episodes.csv"); measured=pd.read_csv(ref/"measured_episodes.csv")
    nulls=make_nulls(output,cand,measured,QUIET_WINDOWS,20260925); nulls.to_csv(ref/"null_pseudo_episodes.csv",index=False)
    check=validate_nulls_r(nulls,QUIET_WINDOWS); stage1.atomic_json(ref/"null_validation.json",check)

def build_episode_references(output):
    import pandas as pd
    ranking=pd.read_csv(output/"episode_tile_ranking.csv")
    while True:
        windows=episode_windows(output)
        for w in windows: extract(output,w)
        cand,bins=candidates(output,windows); measured=measure_only(output,cand,windows)
        accepted=int(measured.accepted.sum()) if len(measured) else 0
        if accepted>=10 or len(windows)>=MAX_EPISODE_WINDOWS: break
        used={int(x["tile_index"]) for x in read_json(output/"extraction_config.json")["windows"] if x["id"].startswith("e")}
        remaining=ranking.loc[(~ranking.excluded_margin_overlap)&~ranking.tile_index.isin(used)].sort_values("eligible_rank")
        if not len(remaining): break
        append_tile(output,int(remaining.iloc[0].tile_index))
    ref=output/"references_episode"; ref.mkdir(exist_ok=False)
    cand.to_csv(ref/"candidate_episodes.csv",index=False); bins.to_csv(ref/"candidate_source_bins.csv",index=False); measured.to_csv(ref/"measured_episodes.csv",index=False)
    nulls=make_nulls(output,cand,measured,windows,20260926); nulls.to_csv(ref/"null_pseudo_episodes.csv",index=False)
    check=validate_nulls_r(nulls,windows); stage1.atomic_json(ref/"null_validation.json",check)
    valid={w:b for w,b in windows.items() if check["windows"][w]["status"]=="pass"}
    excluded={w:check["windows"][w] for w in windows if w not in valid}
    accepted_valid=int(measured.loc[measured.window.isin(valid)&measured.accepted].shape[0])
    sufficient=len(valid)>=5 and accepted_valid>=20
    blocks=block_references(output,cand,measured,valid); blocks.to_csv(ref/"block_measurements.csv",index=False)
    cfg=read_json(output/"extraction_config.json"); chosen=[{k:x[k] for k in ("id","tile_index","start_s","stop_s")} for x in cfg["windows"] if x["id"].startswith("e")]
    stage1.atomic_json(output/"chosen_episode_windows.json",{"windows":chosen,"accepted_total":accepted,
        "valid_windows":sorted(valid),"excluded_windows":excluded,"accepted_in_valid_windows":accepted_valid,
        "r2_sufficient":sufficient,"stopped_because":"accepted>=10" if accepted>=10 else "maximum_8_windows"})
    plan=read_json(output/"preregistration.json"); plan.update(chosen_episode_windows=chosen,accepted_episode_references=accepted,status="all_references_frozen_before_fit")
    stage1.atomic_json(output/"preregistration.json",plan)
    stage1.atomic_json(output/"reference_decision_r.json",{"pathway":"p_full_sweep" if sufficient else "r4_depth2_fallback",
        "valid_episode_windows":sorted(valid),"excluded_episode_windows":excluded,"accepted_in_valid_windows":accepted_valid})

def stage1_rescore(output):
    import pandas as pd
    dc=pd.read_csv(PARENT/"references_m/candidate_episodes.csv"); dm=pd.read_csv(PARENT/"references_m/measured_episodes.csv"); dn=pd.read_csv(PARENT/"references_m/null_pseudo_episodes.csv")
    db=old.block_references(output,dc,dm,DEV_WINDOWS); db.to_csv(output/"stage1_block_measurements.csv",index=False)
    old.score_fields(PARENT/"fields",stage1.CONFIGS,DEV_WINDOWS,dc,dm,dn,db,{"p50"},output/"scores_stage1_rescored.csv")

def reference_all(output):
    receipts=list((output/"fields").glob("*/*/receipt.json")) if (output/"fields").exists() else []
    if receipts: raise RuntimeError("Stage-2 fit exists before P reference freeze")
    plan=read_json(output/"preregistration.json")
    plan["candidate_union_episode"]={
        "field":"runs of the ranking field at centred rigid <= -40um",
        "low_rate":"raw pilot-frontend count <0.5x tile median; gaps <=1s joined",
        "invalid_eye":"unavailable for Luke0804"}
    plan["candidate_union_frozen_before_fit_receipts"]=True
    stage1.atomic_json(output/"preregistration.json",plan)
    quiet_null_gate(output); build_episode_references(output); stage1_rescore(output)
    decision=read_json(output/"reference_decision_r.json")
    qcheck=read_json(output/"references_heldout/null_validation.json")
    decision["valid_quiet_windows"]=[w for w in QUIET_WINDOWS if qcheck["windows"][w]["status"]=="pass"]
    decision["excluded_quiet_windows"]={w:qcheck["windows"][w] for w in QUIET_WINDOWS if qcheck["windows"][w]["status"]!="pass"}
    stage1.atomic_json(output/"reference_decision_r.json",decision)
    files=[output/"preregistration.json",output/"episode_tile_ranking.csv",output/"chosen_episode_windows.json",
        output/"references_heldout/null_pseudo_episodes.csv",output/"references_heldout/null_validation.json",
        output/"references_episode/measured_episodes.csv",output/"references_episode/null_pseudo_episodes.csv",
        output/"references_episode/block_measurements.csv",output/"scores_stage1_rescored.csv",output/"reference_decision_r.json"]
    stage1.atomic_json(output/"references_complete_before_fit_p.json",{"status":"complete_before_stage2_fit","completed_at":time.time(),
        "stage2_fit_receipts_at_freeze":0,"artifacts":{str(p.relative_to(output)):stage1.sha256(p) for p in files}})

def fit_one(output,config,window):
    import medicine,torch
    if read_json(output/"references_complete_before_fit_p.json")["status"]!="complete_before_stage2_fit": raise RuntimeError("P references not frozen")
    target=output/f"fields/{config}/{window}"; target.mkdir(parents=True,exist_ok=False); source,start=population(output,window,False)
    with np.load(source,allow_pickle=False) as z: times,depths,amps=(np.asarray(z[k]) for k in ("time_s","depth_um","amplitude"))
    change=dict(CONFIGS[config]); seed=int(change.pop("seed",0)); settings=dict(stage1.BASE_MED); settings.update(change)
    torch.set_num_threads(4)
    if not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
    np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); begun=time.monotonic()
    trainer=medicine.run_medicine(peak_times=times,peak_depths=depths,peak_amplitudes=amps,output_dir=target/"medicine",optimizer=torch.optim.Adam,**settings)
    runtime=time.monotonic()-begun; med=target/"medicine"; ft,fz,fm=(np.load(med/n) for n in ("time_bins.npy","depth_bins.npy","motion.npy"))
    np.savez_compressed(target/"field.npz",time_s=ft,session_time_s=ft+start,depth_um=fz,displacement_um=fm,
        sign_contract=np.asarray("observed_minus_registered; corrected=observed-displacement")); np.save(target/"loss.npy",np.asarray(trainer.losses))
    stage1.atomic_json(target/"receipt.json",{"status":"complete","config":config,"window":window,"runtime_s":runtime,"settings":settings,"seed":seed,
        "peaks":len(times),"source_population":str(source),"reference_receipt_sha256":stage1.sha256(output/"references_complete_before_fit_p.json"),
        "field_sha256":stage1.sha256(target/"field.npz"),"no_sort":True,"voltage_modified":False})

def field_path(output,config,window):
    return PARENT/f"fields/{config}/{window}/field.npz" if window in DEV_WINDOWS and config in DEV_REUSE else output/f"fields/{config}/{window}/field.npz"

def planned_pairs(output):
    held={**QUIET_WINDOWS,**episode_windows(output)}
    decision=read_json(output/"reference_decision_r.json") if (output/"reference_decision_r.json").exists() else {"pathway":"p_full_sweep"}
    if decision["pathway"]=="r4_depth2_fallback": return [("depth2",w) for w in held]
    return [(c,w) for c in CONFIGS for w in held]+[(c,w) for c in CONFIGS if c not in DEV_REUSE for w in DEV_WINDOWS]

def sweep(output):
    if not (output/"references_complete_before_fit_p.json").exists(): raise RuntimeError("P references must complete first")
    cfg=read_json(output/"extraction_config.json"); runtime=sum(read_json(output/f"peak_cache/{x['id']}/extraction/audit.json")["seconds"] for x in cfg["windows"])
    for c,w in planned_pairs(output):
        receipt=output/f"fields/{c}/{w}/receipt.json"
        if receipt.exists(): runtime+=read_json(receipt)["runtime_s"]; continue
        if runtime>=GPU_BUDGET_S: raise RuntimeError("Stopped at P 7 GPU-hour budget")
        r=subprocess.run([str(stage1.MEDPY),str(Path(__file__).resolve()),"fit-one","--output",str(output),"--config",c,"--window",w],timeout=old.FIT_TIMEOUT_S)
        if r.returncode: raise RuntimeError(f"Fit failed: {c}/{w}")
        runtime+=read_json(receipt)["runtime_s"]
    stage1.atomic_json(output/"sweep_complete_p.json",{"status":"complete","new_fits":len(planned_pairs(output)),
        "reused_development_fits":len(DEV_REUSE)*len(DEV_WINDOWS),"budget_accounted_runtime_s":runtime,"budget_s":GPU_BUDGET_S,
        "no_sort":True,"voltage_modified":False})

def load_field(path):
    with np.load(path,allow_pickle=False) as z: return {k:np.asarray(z[k]) for k in z.files}

def false_motion(field,cand,bounds):
    start,stop=bounds; t=np.arange(start+stage1.BIN/2,stop,stage1.BIN); allowed=np.ones(len(t),bool)
    for x in cand.itertuples(): allowed&=~((t>=x.start_s-2)&(t<=x.stop_s+2))
    tt,zz=np.meshgrid(t,np.asarray(field["depth_um"],float),indexing="ij"); rigid=np.nanmedian(stage1.sample_field(field,tt,zz),axis=1)
    rigid-=np.nanmedian(rigid)
    return float(np.mean(np.abs(rigid[allowed])>50)) if allowed.any() else np.nan

def score_set(output,windows,cand,measured,nulls,blocks,quiet,configs=CONFIGS):
    import pandas as pd
    rows=[]
    for config in configs:
        er,ratios,br,qa,qi,fm=[],[],[],[],[],[]; qpair=0
        for window,bounds in windows.items():
            field=load_field(field_path(output,config,window))
            for ep in measured.loc[(measured.window==window)&measured.accepted].itertuples():
                pred=mref.episode_prediction(field,ep.start_s,ep.stop_s); er.append(abs(pred-ep.best_shift_um)); ratios.append(pred/ep.best_shift_um)
            if len(blocks):
                for b in blocks.loc[(blocks.window==window)&blocks.accepted].itertuples():
                    pred=old.depth_prediction(field,b.start_s,b.stop_s,b.block_center_um); br.append(abs(pred-b.best_shift_um))
            for n in nulls.loc[(nulls.window==window)&nulls.resolved].itertuples(): qa.append(abs(mref.episode_prediction(field,n.start_s,n.stop_s)))
            if quiet:
                inc,pairs=old.corrected_quiet_values(field,cand.loc[cand.window==window],bounds); qi.extend(inc); qpair+=pairs
                fm.append(false_motion(field,cand.loc[cand.window==window],bounds))
        rows.append({"config":config,"episode_err":float(np.median(er)) if er else np.nan,"episode_ratio":float(np.median(ratios)) if ratios else np.nan,
            "accepted_episodes":len(er),"block_err":float(np.median(br)) if br else np.nan,"accepted_blocks":len(br),
            "quiet_abs":float(np.median(qa)) if qa else np.nan,"quiet_inc":float(np.sqrt(np.mean(np.square(qi)))) if qi else np.nan,
            "false_motion_frac":float(np.mean(fm)) if fm else np.nan,"quiet_pairs":qpair})
    return pd.DataFrame(rows)

def select(quiet,episode):
    table=quiet[["config","quiet_abs","quiet_inc","false_motion_frac"]].merge(episode[["config","episode_err","episode_ratio","block_err"]],on="config")
    gate=table.loc[(table.quiet_abs<=10)&(table.quiet_inc<=8)&(table.false_motion_frac<=.05)]
    ratio=gate.loc[gate.episode_ratio.between(.8,1.2)&gate.block_err.notna()]
    best_block=float(ratio.block_err.min()) if len(ratio) else np.nan
    eligible=ratio.loc[ratio.block_err<=1.2*best_block] if len(ratio) else ratio
    if not len(eligible): return {"outcome":"no_selection","reason":"no configuration passed all gates","table":table.to_dict("records")}
    best=float(eligible.episode_err.min()); tied=eligible.loc[eligible.episode_err<=best+3]
    def simple(c):
        x=CONFIGS[c]; return (x.get("num_depth_bins",4),x.get("amplitude_threshold_quantile",0),c)
    chosen=min(tied.config,key=simple)
    return {"outcome":"selected","config":chosen,"best_episode_err":best,"best_block_err":best_block,
        "eligible":eligible.config.tolist(),"within_3um":tied.config.tolist(),"table":table.to_dict("records")}

def seed_stability(output):
    rows=[]
    for w in {**DEV_WINDOWS,**QUIET_WINDOWS,**episode_windows(output)}:
        a=load_field(field_path(output,"amp50_d2",w)); b=load_field(field_path(output,"amp50_d2_s1",w))
        tt,zz=np.meshgrid(a["session_time_s"],a["depth_um"],indexing="ij"); delta=stage1.sample_field(b,tt,zz)-a["displacement_um"]
        rows.append({"window":w,"field_rms_delta_um":float(np.sqrt(np.nanmean(delta**2)))})
    return {"per_window":rows,"median_field_rms_delta_um":float(np.median([x["field_rms_delta_um"] for x in rows]))}

def report(output):
    import pandas as pd
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    qr,er=output/"references_heldout",output/"references_episode"
    qc=pd.read_csv(qr/"candidate_episodes.csv"); qm=pd.read_csv(qr/"measured_episodes.csv"); qn=pd.read_csv(qr/"null_pseudo_episodes.csv")
    ec=pd.read_csv(er/"candidate_episodes.csv"); em=pd.read_csv(er/"measured_episodes.csv"); en=pd.read_csv(er/"null_pseudo_episodes.csv"); eb=pd.read_csv(er/"block_measurements.csv")
    decision=read_json(output/"reference_decision_r.json")
    qwin={w:QUIET_WINDOWS[w] for w in decision["valid_quiet_windows"]}
    eall=episode_windows(output); ewin={w:eall[w] for w in decision["valid_episode_windows"]}
    scored_configs=list(CONFIGS) if decision["pathway"]=="p_full_sweep" else ["depth2"]
    empty=pd.DataFrame(columns=["window","accepted"]); quiet=score_set(output,qwin,qc,qm,qn,empty,True,scored_configs); episode=score_set(output,ewin,ec,em,en,eb,False,scored_configs)
    quiet.to_csv(output/"scores_stage2_quiet_heldout.csv",index=False); episode.to_csv(output/"scores_stage2_episode_heldout.csv",index=False)
    if decision["pathway"]=="p_full_sweep":
        dc=pd.read_csv(PARENT/"references_m/candidate_episodes.csv"); dm=pd.read_csv(PARENT/"references_m/measured_episodes.csv"); dn=pd.read_csv(PARENT/"references_m/null_pseudo_episodes.csv"); db=pd.read_csv(output/"stage1_block_measurements.csv")
        mixed=output/"mixed_fields"; mixed.mkdir(exist_ok=True)
        for c in CONFIGS:
            for w in DEV_WINDOWS:
                dst=mixed/c/w; dst.mkdir(parents=True,exist_ok=True); link=dst/"field.npz"
                # A prior interrupted report can leave a valid symlink whose
                # target fit was not complete yet.  Preserve and reuse it.
                if not link.exists() and not link.is_symlink(): link.symlink_to(field_path(output,c,w))
        old.score_fields(mixed,CONFIGS,DEV_WINDOWS,dc,dm,dn,db,{"p50"},output/"scores_stage2_development.csv")
        choice=select(quiet,episode)
        if choice["outcome"]!="selected": choice={**choice,"outcome":"provisional","config":"depth2","basis":"R.4 development-selected fallback"}
        stability=seed_stability(output)
    else:
        pd.read_csv(output/"scores_stage1_rescored.csv").loc[lambda x:x.config=="depth2"].to_csv(output/"scores_stage2_development.csv",index=False)
        choice={"outcome":"provisional","config":"depth2","basis":"R.4 insufficient surviving held-out references"}
        stability={"status":"not_run_in_depth2_fallback"}
    stage1.atomic_json(output/"selection_p.json",choice); stage1.atomic_json(output/"seed_stability_p.json",stability)
    centers=[475,1425,2375,3325]
    for w in ewin:
        fig,ax=plt.subplots(figsize=(9,5),layout="constrained"); refs=eb.loc[(eb.window==w)&eb.accepted]
        for _,g in refs.groupby("episode_id"): ax.plot(g.block_center_um,g.best_shift_um,"o-",color=".75",lw=.8,ms=3)
        for c in scored_configs:
            field=load_field(field_path(output,c,w)); group=em.loc[(em.window==w)&em.accepted]
            vals=[np.nanmedian([old.depth_prediction(field,x.start_s,x.stop_s,d) for x in group.itertuples()]) for d in centers]
            ax.plot(centers,vals,"o-",label=c,lw=1.2)
        ax.set(title=f"{w}: episode-heldout per-depth shifts",xlabel="Depth-block centre (um)",ylabel="Rest-centred displacement (um)"); ax.legend(ncol=4,fontsize=7)
        fig.savefig(output/f"figure_{w}_per_depth.png",dpi=180); plt.close(fig)
    archive=output/"stage2_fields.zip"; allw={**DEV_WINDOWS,**QUIET_WINDOWS,**episode_windows(output)}
    with zipfile.ZipFile(archive,"w",compression=zipfile.ZIP_DEFLATED) as z:
        for c in scored_configs:
            for w in allw:
                path=field_path(output,c,w)
                if path.exists(): z.write(path,arcname=f"{c}/{w}/field.npz")
    chosen=read_json(output/"chosen_episode_windows.json"); qnull=read_json(qr/"null_validation.json"); enull=read_json(er/"null_validation.json")
    outcome=f"Selected `{choice['config']}`." if choice["outcome"]=="selected" else "No configuration passed; see `selection_p.json`."
    lines=["# Luke0804 imec1 MEDiCINe stage 2 — correction P","","No sort was run and source voltage was not modified. All references and episode-window choices were frozen before fitting.","","## Outcome","",outcome,
        (f"Seed stability median field RMS: {stability['median_field_rms_delta_um']:.3f} um." if "median_field_rms_delta_um" in stability else "Seed stability was not run in the R.4 depth2-only fallback."),"","## Held-out design","",
        f"Quiet-null validation: **{qnull['status']}**. Episode-null validation: **{enull['status']}**.",
        f"Episode windows: {len(chosen['windows'])}; accepted raw-map references: {chosen['accepted_total']}.",
        "Tile ranking used only the saved full-session shared-recovery rigid field; reference shifts use raw unlabeled pilot-frontend maps.","","## Artifacts","",
        "- `episode_tile_ranking.csv` and `chosen_episode_windows.json`","- `scores_stage1_rescored.csv`","- `scores_stage2_development.csv`",
        "- `scores_stage2_quiet_heldout.csv`","- `scores_stage2_episode_heldout.csv`","- `references_heldout/` and `references_episode/`",
        "- `figure_e*_per_depth.png`","- `stage2_fields.zip`",""]
    (output/"README.md").write_text("\n".join(lines))
    stage1.atomic_json(output/"validation_p.json",{"status":"complete","selection":choice,"seed_stability":stability,
        "new_fits":len(planned_pairs(output)),"archive_entries":len(zipfile.ZipFile(archive).namelist()),"quiet_null":qnull,"episode_null":enull,
        "no_sort":True,"voltage_modified":False})

def main():
    p=argparse.ArgumentParser(); p.add_argument("phase",choices=["prepare","extract-one","reference-all","fit-one","sweep","report"])
    p.add_argument("--output",type=Path,default=DEFAULT_OUTPUT); p.add_argument("--window"); p.add_argument("--config"); a=p.parse_args(); output=a.output.resolve()
    if a.phase=="prepare": prepare(output)
    elif a.phase=="extract-one": stage1.extract_one(output,a.window)
    elif a.phase=="reference-all": reference_all(output)
    elif a.phase=="fit-one": fit_one(output,a.config,a.window)
    elif a.phase=="sweep": sweep(output)
    elif a.phase=="report": report(output)

if __name__=="__main__": main()
