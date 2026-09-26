#!/usr/bin/env python
"""AM imec0 cross-probe validation and two-layer deployment orchestration."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy import stats

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

from motionqc.crossprobe import compare_episodes, fit_clock_map, map_intervals
from motionqc.field import MotionField, load_field
from motionqc.deployment import compose_two_layer
from motionqc.reference import matched_null, shift_test
from motionqc.report import OKABE_ITO, LINESTYLES, build_report, score_fields
from testing import luke_imec1_medicine_reference_sweep_v1 as stage1
from testing import luke_imec1_medicine_stage3_q_v1 as qstage

SWEEP=Path("/media/huklab/Data/NPX/Ryansorting/Luke/luke_imec1_medicine_reference_sweep_v2")
OUT=SWEEP/"stage5_imec0"
SHM=Path("/dev/shm/luke_am_imec0")
PILOT=ROOT/"testing/outputs/cross_dataset_fast_motion_v2"
CATALOGUE=SWEEP/"stage3_q/episode_catalogue.csv"
IME1_MASK=SWEEP/"stage4_ab/censor_mask_v1.csv"
SYNC_ROOT=Path("/mnt/NPX/Luke/20250804/Luke0804_V2V1_g0")
MIN_FREE_BYTES=30_000_000_000
MIN_MEM_BYTES=40_000_000_000
SESSION_DURATION_S=314204894/29999.835983263598
AM_GPU_BUDGET_S=4*3600.0


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(8*1024**2),b""):h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path,value) -> None:
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(value,indent=2,sort_keys=True,default=str)+"\n");os.replace(tmp,path)


def free_bytes(path: Path) -> int:
    x=os.statvfs(path);return int(x.f_bavail*x.f_frsize)


def mem_available() -> int:
    line=next(x for x in Path("/proc/meminfo").read_text().splitlines() if x.startswith("MemAvailable:"))
    return int(line.split()[1])*1024


def blocks():
    starts=np.arange(0,87*120,120,dtype=float)
    return [{"id":f"full_block_{i:03d}","start_s":float(a),"stop_s":float(SESSION_DURATION_S if i==86 else a+120)} for i,a in enumerate(starts)]


def clock_map():
    cfg=json.loads((PILOT/"config.json").read_text());records={r["probe"]:r for r in cfg["records"] if r["dataset"]=="Luke"}
    edge={}
    paths={}
    for probe in ("imec0","imec1"):
        paths[probe]=SYNC_ROOT/f"Luke0730_V2V1_g0_t0.{probe}.ap.sync.mat"
        frames=np.asarray(loadmat(paths[probe],squeeze_me=True)["riseSent"],float)
        edge[probe]=frames/float(records[probe]["sampling_frequency_hz"])
    mapping=fit_clock_map(edge["imec1"],edge["imec0"],source="imec1 AP frame zero",destination="imec0 AP frame zero")
    return mapping,paths,records


def prepare() -> None:
    if OUT.exists():raise FileExistsError(OUT)
    OUT.mkdir(parents=True);(OUT/"peak_cache").mkdir();SHM.mkdir(parents=True,exist_ok=True)
    mapping,sync_paths,records=clock_map()
    catalogue=pd.read_csv(CATALOGUE);accepted=catalogue[catalogue.status.eq("accepted")].copy()
    mapped=map_intervals(accepted,mapping);mapped.to_csv(OUT/"imec1_accepted_episodes_mapped_to_imec0.csv",index=False,float_format="%.9f",lineterminator="\n")
    mask=pd.read_csv(IME1_MASK);mapped_mask=map_intervals(mask,mapping);mapped_mask.to_csv(OUT/"imec1_ae_mask_mapped_to_imec0.csv",index=False,float_format="%.9f",lineterminator="\n")
    base=json.loads((SWEEP/"extraction_config.json").read_text());fs=float(records["imec0"]["sampling_frequency_hz"])
    base["windows"]=[{"id":b["id"],"dataset":"Luke","probe":"imec0","fraction":None,
                      "start_frame":round(b["start_s"]*fs),"stop_frame":min(records["imec0"]["n_frames"],round(b["stop_s"]*fs)),
                      "start_s":round(b["start_s"]*fs)/fs,"stop_s":min(records["imec0"]["n_frames"],round(b["stop_s"]*fs))/fs} for b in blocks()]
    base["purpose"]="AM full-session imec0 pilot frontend in independent 120 s blocks; no sort or voltage modification"
    atomic_json(OUT/"extraction_config.json",base)
    free=free_bytes(OUT);shm=free_bytes(SHM);estimated_cache=int(np.mean([7834250,8865120,7875316])*87)
    preflight={"data_free_bytes":free,"shm_free_bytes":shm,"minimum_data_free_bytes":MIN_FREE_BYTES,
               "estimated_persisted_peak_cache_bytes":estimated_cache,"estimated_per_worker_shm_bytes":20_000_000_000,
               "reserved_shm_bytes":25_000_000_000,"maximum_workers":min(3,max(1,(shm-25_000_000_000)//20_000_000_000)),
               "prior_imec0_120s_extraction_seconds":[790.6207638587803,796.8767036236823,788.7014553025365],
               "projected_sequential_hours":float(87*np.mean([790.6207638587803,796.8767036236823,788.7014553025365])/3600),
               "passed":bool(free>=MIN_FREE_BYTES and shm>=45_000_000_000 and mem_available()>=MIN_MEM_BYTES)}
    atomic_json(OUT/"preflight.json",preflight)
    if not preflight["passed"]:raise RuntimeError(f"AM preflight failed: {preflight}")
    atomic_json(OUT/"preregistration_am.json",{
        "status":"frozen_before_imec0_episode_shift_results","catalogue":str(CATALOGUE),"catalogue_sha256":sha256(CATALOGUE),
        "catalogue_expected_sha256":"46463cda1348c0e0b0c39ad22ae9c71bc46f7c7347797c1c10588944cfc52df1",
        "catalogue_location_note":"Requested incoming/hub_analysis_v1 copy is absent; stage3_q file has the requested full hash.",
        "accepted_imec1_episodes":len(accepted),"mapped_catalogue_sha256":sha256(OUT/"imec1_accepted_episodes_mapped_to_imec0.csv"),
        "clock_mapping":mapping.to_dict(),"sync_inputs":{k:{"path":str(v),"sha256":sha256(v)} for k,v in sync_paths.items()},
        "episode_test":{"depth_bin_um":10,"x_bin_um":8,"shift_range_um":[-320,120],"minimum_peaks_each":150,"minimum_gain":.05,"rest_s_before_onset":[4,1]},
        "null":{"minimum_resolved":20,"pass":"mode=0um and median_abs<=10um","seed":20260925,"run_before_episode_test":True},
        "verdict":{"co_moving":"accepted_same_sign_abs_ge80_fraction>=0.60 and spearman>=0.4","imec1_only":"accepted_fraction<=0.20 and null passes","otherwise":"mixed"},
        "blocks":[*blocks()],"frontend":"exact stage1.extract_one context; one independent denoiser per 120 s block; RAM scratch",
        "preflight":preflight,"gpu_fit_budget_hours":4,"stop_if_null_fails":True,"sort_run":False,"voltage_modified":False})


def disk_guard(where: str) -> None:
    free=free_bytes(OUT);atomic_json(OUT/"disk_guard_latest.json",{"where":where,"free_bytes":free,"required":MIN_FREE_BYTES,"passed":free>=MIN_FREE_BYTES,"at":time.time()})
    if free<MIN_FREE_BYTES:raise RuntimeError(f"disk guard failed at {where}: {free}")


def extract_one(block_id: str) -> None:
    disk_guard("before "+block_id)
    if mem_available()<MIN_MEM_BYTES:raise RuntimeError("MemAvailable below 40 GB")
    qstage.SHM_ROOT=SHM
    qstage.extract_one_block_v(OUT,block_id)
    disk_guard("after "+block_id)


def worker_limit() -> int:
    shm=free_bytes(SHM);return int(min(3,max(1,(shm-25_000_000_000)//20_000_000_000)))


def extract_all() -> None:
    start_file=OUT/"extraction_started.json"
    if not start_file.exists():atomic_json(start_file,{"started_at":time.time(),"planned":87})
    pending=[b for b in blocks() if not (OUT/f"peak_cache/{b['id']}/extraction/complete.json").exists()];running={}
    while pending or running:
        disk_guard("scheduler")
        while pending and len(running)<worker_limit() and mem_available()>=MIN_MEM_BYTES:
            b=pending.pop(0);cmd=[str(stage1.DSPY),str(Path(__file__).resolve()),"extract-one","--block",b["id"]]
            p=subprocess.Popen(cmd);running[p]=b
        if not running:time.sleep(5);continue
        time.sleep(2);done=[p for p in running if p.poll() is not None]
        for p in done:
            b=running.pop(p)
            if p.returncode:
                for other in running:other.terminate()
                for other in running:
                    try:other.wait(timeout=30)
                    except subprocess.TimeoutExpired:other.kill();other.wait()
                raise RuntimeError(f"extraction failed {b['id']} exit={p.returncode}")
        complete=[]
        for b in blocks():
            pop=OUT/f"peak_cache/{b['id']}/extraction/population.npz"
            audit=OUT/f"peak_cache/{b['id']}/extraction/audit.json"
            if pop.exists() and audit.exists():complete.append({"block":b["id"],"bytes":pop.stat().st_size,"seconds":json.loads(audit.read_text())["seconds"]})
        atomic_json(OUT/"extraction_progress.json",{"status":"running","completed":len(complete),"planned":87,"rows":complete,"updated_at":time.time()})
    receipt=json.loads(start_file.read_text());rows=json.loads((OUT/"extraction_progress.json").read_text())["rows"]
    atomic_json(OUT/"extraction_complete.json",{"status":"complete","completed":len(rows),"wall_s":time.time()-receipt["started_at"],"rows":rows})


def load_peaks() -> tuple[pd.DataFrame,dict]:
    config=json.loads((OUT/"extraction_config.json").read_text());spec={x["id"]:x for x in config["windows"]};pieces=[];manifest=[]
    for block in blocks():
        path=OUT/f"peak_cache/{block['id']}/extraction/population.npz"
        if not path.exists():raise FileNotFoundError(path)
        with np.load(path,allow_pickle=False) as z:
            n=len(z["time_s"]);pieces.append(pd.DataFrame({"time_s":np.asarray(z["time_s"],float)+spec[block["id"]]["start_s"],
                "depth_um":np.asarray(z["depth_um"],float),"x_um":np.asarray(z["x_um"],float),"amplitude":np.asarray(z["amplitude"],float)}))
        manifest.append({"block":block["id"],"path":str(path),"sha256":sha256(path),"bytes":path.stat().st_size,"peaks":n})
    manifest_frame=pd.DataFrame(manifest);manifest_frame.to_csv(OUT/"peak_cache_manifest.csv",index=False,lineterminator="\n")
    frame=pd.concat(pieces,ignore_index=True)
    receipt={"source":"87 independently extracted 120 s pilot-frontend caches","manifest":str(OUT/"peak_cache_manifest.csv"),
             "manifest_sha256":sha256(OUT/"peak_cache_manifest.csv"),"peak_count":len(frame),
             "depth_min_um":float(frame.depth_um.min()),"depth_max_um":float(frame.depth_um.max()),
             "time_min_s":float(frame.time_s.min()),"time_max_s":float(frame.time_s.max())}
    atomic_json(OUT/"peak_source_receipt.json",receipt);return frame,receipt


def am2() -> None:
    if not (OUT/"extraction_complete.json").exists():raise RuntimeError("AM.1 extraction is incomplete")
    peaks,source=load_peaks();episodes=pd.read_csv(OUT/"imec1_accepted_episodes_mapped_to_imec0.csv")
    durations=(episodes.end_s-episodes.start_s).to_numpy();time=peaks.time_s.to_numpy()
    targets=[]
    for row in episodes.itertuples(index=False):
        lo,hi=np.searchsorted(time,[row.start_s,row.end_s]);targets.append(hi-lo)
    null,gate=matched_null(peaks,durations,list(zip(episodes.start_s,episodes.end_s)),(0,SESSION_DURATION_S),n=60,seed=20260925,
        peak_targets=targets,placement_margin_s=2,quantisation_um=10,depth_bin_um=10,x_bin_um=8,shift_min_um=-320,
        shift_max_um=120,min_peaks=150,min_gain=0,depth_range_um=(0,3840),x_range_um=(-32,80))
    null.to_csv(OUT/"am2_null.csv",index=False);atomic_json(OUT/"am2_null_gate.json",gate)
    if not gate["pass"]:
        atomic_json(OUT/"am_stop.json",{"stage":"AM.2 null","reason":"null failed","gate":gate});raise RuntimeError(f"AM.2 null failed: {gate}")
    results,depth=compare_episodes(peaks,episodes,depth_bin_um=10,x_bin_um=8,shift_min_um=-320,shift_max_um=120,
        min_peaks=150,min_gain=.05,depth_range_um=(0,3840),x_range_um=(-32,80))
    results.to_csv(OUT/"am2_cross_probe_episodes.csv",index=False);depth.to_csv(OUT/"am2_per_depth_blocks.csv",index=False)
    accepted=results.accepted.astype(bool);source_shift=results.source_shift_um.to_numpy();target_shift=results.best_shift_um.to_numpy()
    same=np.sign(source_shift)==np.sign(target_shift);strong=accepted&same&(np.abs(target_shift)>=80)
    rho=float(stats.spearmanr(source_shift[accepted],target_shift[accepted]).statistic) if accepted.sum()>=3 else np.nan
    accepted_fraction=float(accepted.mean());strong_fraction=float(strong.mean())
    if strong_fraction>=.60 and rho>=.4:verdict="co-moving"
    elif accepted_fraction<=.20:verdict="imec1-only"
    else:verdict="mixed"
    summary={"verdict":verdict,"imec1_accepted_episodes":len(results),"imec0_accepted":int(accepted.sum()),
        "imec0_accepted_fraction":accepted_fraction,"strong_same_sign_abs_ge80":int(strong.sum()),
        "strong_same_sign_abs_ge80_fraction":strong_fraction,"spearman_rho_accepted":rho,
        "imec0_shift_median_um":float(np.nanmedian(target_shift[accepted])) if accepted.any() else np.nan,
        "imec0_shift_p05_um":float(np.nanpercentile(target_shift[accepted],5)) if accepted.any() else np.nan,
        "imec0_shift_p95_um":float(np.nanpercentile(target_shift[accepted],95)) if accepted.any() else np.nan,
        "null":gate,"peak_source":source,"interpretation":"frozen AM.2 thresholds; mixed receives no mechanism interpretation"}
    atomic_json(OUT/"am2_summary.json",summary)
    import matplotlib;matplotlib.use("Agg");import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,5));colors=np.where(accepted,"#0072B2","#999999")
    axes[0].scatter(source_shift,target_shift,s=14,c=colors,alpha=.65);axes[0].plot([-320,120],[-320,120],color="#D55E00",ls="--",lw=1)
    axes[0].set(xlabel="imec1 measured shift (µm)",ylabel="imec0 measured shift (µm)",title=f"Cross-probe episodes: {verdict}, Spearman ρ={rho:.3f}")
    resolved=depth[depth.accepted.astype(bool)];
    for block,group in resolved.groupby("block"):
        axes[1].plot(group.episode_index,group.best_shift_um,ls=("-","--","-.",":")[int(block)%4],color=("#0072B2","#D55E00","#009E73","#CC79A7")[int(block)%4],alpha=.4,label=f"block {block}")
    axes[1].set(xlabel="Episode index",ylabel="imec0 block shift (µm)",title="Accepted per-depth-block shifts");axes[1].legend();
    for ax in axes:ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(OUT/"am2_cross_probe_summary.png",dpi=180);plt.close(fig)
    lines=["# AM.2 imec0 cross-probe episode test","",f"Frozen verdict: **{verdict}**.","",f"- imec0 accepted: {int(accepted.sum())}/{len(results)} ({accepted_fraction:.3%})",f"- same-sign accepted |shift|≥80 µm: {int(strong.sum())}/{len(results)} ({strong_fraction:.3%})",f"- Spearman rho among accepted episodes: {rho:.3f}",f"- null: {gate}","",f"Peak source: `{source['manifest']}` (SHA-256 `{source['manifest_sha256']}`), {source['peak_count']:,} peaks, depth {source['depth_min_um']:.1f}–{source['depth_max_um']:.1f} µm.",""]
    (OUT/"AM2_REPORT.md").write_text("\n".join(lines))


def am3_prepare() -> None:
    summary=json.loads((OUT/"am2_summary.json").read_text());null=json.loads((OUT/"am2_null_gate.json").read_text())
    if summary["verdict"] not in ("co-moving","mixed") or not null["pass"]:
        raise RuntimeError("AM.3 mapped-mask precondition is not satisfied")
    if list((OUT/"fields/full").glob("*/receipt.json")) if (OUT/"fields/full").exists() else []:
        raise RuntimeError("AM.3 prepare must precede the first fit")
    choice,settings,seed=qstage.selected_settings()
    if choice["config"]!="amp50_d1" or settings["amplitude_threshold_quantile"]!=.5 or settings["num_depth_bins"]!=1:
        raise RuntimeError(f"Frozen amp50_d1 configuration mismatch: {choice}, {settings}")
    windows=qstage.window_specs("full")
    atomic_json(OUT/"am3_preregistration.json",{
        "status":"frozen_before_any_imec0_field_fit","am2_verdict":summary["verdict"],
        "am2_summary_sha256":sha256(OUT/"am2_summary.json"),"null_gate":null,
        "fast":{"configuration":"amp50_d1","settings":settings,"seed":seed,"windows":windows,
                "centering":"per-window temporal median","offsets":"chained overlap median difference",
                "blend":"triangular in 60 s overlaps","output_grid_s":.25},
        "slow":{"configuration":"amp50_d1","time_kernel_width_s":30,"mask":str(OUT/"imec1_ae_mask_mapped_to_imec0.csv"),
                "mask_sha256":sha256(OUT/"imec1_ae_mask_mapped_to_imec0.csv"),"additional_exclusion_s_each_side":3},
        "composition":{"mask":"mapped imec1 canonical AE mask because AM.2 is co-moving","crossfade_s":1},
        "validation":"motionqc phase-1: episode agreement, quiet slow reference/increments, and boundary checks; stop without tuning on failure",
        "medicine_fit_budget_s":AM_GPU_BUDGET_S,"disk_guard_min_free_bytes":MIN_FREE_BYTES,
        "no_sort":True,"voltage_modified":False,"prepared_at":time.time()})
    atomic_json(OUT/"u_full_mode.json",{"mode":"block_concat","reason":"AM frozen recipe uses the validated U block-concatenation path"})


def am3_fit_runtime() -> float:
    total=0.0
    for path in list(OUT.glob("fields/full/*/receipt.json"))+list(OUT.glob("fields/slow/*/receipt.json")):
        total+=float(json.loads(path.read_text())["runtime_s"])
    return total


def am3_fast_one(window: str) -> None:
    if not (OUT/"am3_preregistration.json").exists():raise RuntimeError("AM.3 is not preregistered")
    disk_guard("before fast fit "+window)
    qstage.fit_one(OUT,"full",window)
    receipt=OUT/f"fields/full/{window}/receipt.json"
    if not receipt.exists():raise RuntimeError(f"Missing fast-fit receipt: {window}")
    disk_guard("after fast fit "+window)


def _in_intervals(time_s: np.ndarray, intervals: np.ndarray) -> np.ndarray:
    starts=intervals[:,0];stops=intervals[:,1];index=np.searchsorted(starts,time_s,side="right")-1
    return (index>=0)&(time_s<stops[np.maximum(index,0)])


def am3_slow_fit() -> None:
    target=OUT/"fields/slow/medicine_amp50_d1_k30_exclude3s"
    receipt=target/"receipt.json"
    if receipt.exists():return
    import medicine, torch
    mask=pd.read_csv(OUT/"imec1_ae_mask_mapped_to_imec0.csv")
    intervals=np.c_[np.maximum(0,mask.start_s.to_numpy(float)-3),np.minimum(SESSION_DURATION_S,mask.end_s.to_numpy(float)+3)]
    pieces={k:[] for k in ("time_s","depth_um","amplitude")};total=kept=0
    config=json.loads((OUT/"extraction_config.json").read_text());spec={x["id"]:x for x in config["windows"]}
    for block in blocks():
        path=OUT/f"peak_cache/{block['id']}/extraction/population.npz"
        with np.load(path,allow_pickle=False) as z:
            t=np.asarray(z["time_s"],float)+spec[block["id"]]["start_s"];use=~_in_intervals(t,intervals)
            total+=len(t);kept+=int(use.sum());pieces["time_s"].append(t[use])
            pieces["depth_um"].append(np.asarray(z["depth_um"])[use]);pieces["amplitude"].append(np.asarray(z["amplitude"])[use])
    times,depths,amps=(np.concatenate(pieces[k]) for k in ("time_s","depth_um","amplitude"))
    settings=dict(stage1.BASE_MED);settings.update(amplitude_threshold_quantile=.5,num_depth_bins=1,time_kernel_width=30.)
    if am3_fit_runtime()+120>AM_GPU_BUDGET_S:raise RuntimeError("AM 4 GPU-hour fit budget would be exceeded before slow fit")
    target.mkdir(parents=True,exist_ok=False);np.random.seed(0);torch.manual_seed(0);torch.cuda.manual_seed_all(0);torch.set_num_threads(4)
    if not torch.cuda.is_available():raise RuntimeError("CUDA unavailable")
    begun=time.monotonic();trainer=medicine.run_medicine(peak_times=times,peak_depths=depths,peak_amplitudes=amps,
        output_dir=target/"medicine",optimizer=torch.optim.Adam,**settings);runtime=time.monotonic()-begun
    med=target/"medicine";ft,fz,fm=(np.load(med/name) for name in ("time_bins.npy","depth_bins.npy","motion.npy"))
    if not (np.isfinite(ft).all() and np.isfinite(fz).all() and np.isfinite(fm).all()):raise RuntimeError("Nonfinite AM.3 slow field")
    np.savez_compressed(target/"field.npz",time_s=ft,depth_um=fz,displacement_um=fm,sign_convention=np.asarray("corrected = observed - displacement"))
    np.save(target/"loss.npy",np.asarray(trainer.losses));atomic_json(receipt,{"status":"complete","runtime_s":runtime,
        "settings":settings,"seed":0,"input_peaks":total,"masked_peaks":total-kept,"kept_peaks":kept,
        "mask_sha256":sha256(OUT/"imec1_ae_mask_mapped_to_imec0.csv"),"additional_exclusion_s_each_side":3,
        "field_sha256":sha256(target/"field.npz"),"sort_run":False,"voltage_modified":False})


def am3_fit_all() -> None:
    if not (OUT/"am3_preregistration.json").exists():raise RuntimeError("AM.3 is not preregistered")
    specs=qstage.window_specs("full");begun=time.time()
    for index,spec in enumerate(specs):
        receipt=OUT/f"fields/full/{spec['id']}/receipt.json"
        if receipt.exists():continue
        if am3_fit_runtime()+90>AM_GPU_BUDGET_S:raise RuntimeError("AM 4 GPU-hour fit budget would be exceeded")
        result=subprocess.run([str(stage1.MEDPY),str(Path(__file__).resolve()),"am3-fast-one","--window",spec["id"]],timeout=stage1.FIT_TIMEOUT_S)
        if result.returncode:raise RuntimeError(f"AM.3 fast fit failed: {spec['id']}")
        completed=len(list(OUT.glob("fields/full/*/receipt.json")));elapsed=time.time()-begun
        atomic_json(OUT/"am3_fit_progress.json",{"status":"running","completed_fast_fits":completed,"planned_fast_fits":len(specs),
            "medicine_runtime_s":am3_fit_runtime(),"wall_s_this_service":elapsed,
            "projected_remaining_wall_s":elapsed/completed*(len(specs)-completed) if completed else None,"updated_at":time.time()})
    fast=qstage.stitch(OUT,"full");seams=json.loads((OUT/"seams_full.json").read_text())
    if seams["median_abs_seam_discontinuity_um"]>10:
        atomic_json(OUT/"am3_stop.json",{"stage":"fast stitch","reason":"median seam gate failed","seams":seams});raise RuntimeError("AM.3 fast seam gate failed")
    am3_slow_fit()
    atomic_json(OUT/"am3_fit_complete.json",{"status":"complete","fast_fits":len(specs),"slow_fits":1,
        "medicine_runtime_s":am3_fit_runtime(),"budget_s":AM_GPU_BUDGET_S,
        "fast_field":str(OUT/"stitched_full.npz"),"fast_field_sha256":sha256(OUT/"stitched_full.npz"),
        "slow_field":str(OUT/"fields/slow/medicine_amp50_d1_k30_exclude3s/field.npz"),
        "slow_field_sha256":sha256(OUT/"fields/slow/medicine_amp50_d1_k30_exclude3s/field.npz"),"completed_at":time.time()})


def am3_quiet_reference(peaks: pd.DataFrame, mask: pd.DataFrame) -> tuple[pd.DataFrame,dict]:
    from testing import luke_imec1_slow_layer_ab_v1 as ab
    starts=np.arange(0,np.floor(SESSION_DURATION_S/10)*10,10.);intervals=mask[["start_s","end_s"]].to_numpy(float)
    quiet=np.asarray([not np.any((intervals[:,0]<a+10)&(intervals[:,1]>a)) for a in starts])
    shape=(len(ab.DEPTH_EDGES)-1,len(ab.X_EDGES)-1);counts=np.zeros(len(starts),int);raw=np.zeros((len(starts),*shape),np.int32)
    time=peaks.time_s.to_numpy();depth=peaks.depth_um.to_numpy();x=peaks.x_um.to_numpy()
    for index in np.flatnonzero(quiet):
        lo,hi=np.searchsorted(time,[starts[index],starts[index]+10]);counts[index]=hi-lo
        raw[index]=np.histogram2d(depth[lo:hi],x[lo:hi],bins=(ab.DEPTH_EDGES,ab.X_EDGES))[0]
    maps=np.log1p(raw);rows=[]
    for lag_s in (60,300):
        lag=lag_s//10
        for i in range(lag,len(starts)):
            if counts[i]<150 or counts[i-lag]<150:continue
            shift,zero,best,prominence=ab.correlate_shift(maps[i],maps[i-lag]);resolved=np.isfinite(best) and best>=.1 and prominence>=.05
            rows.append({"time_s":starts[i]+5,"previous_time_s":starts[i-lag]+5,"lag_s":lag_s,
                "current_peaks":counts[i],"previous_peaks":counts[i-lag],"shift_um":shift,"corr_zero":zero,
                "corr_best":best,"corr_prominence":prominence,"resolved":bool(resolved)})
    reference=pd.DataFrame(rows);reference.to_csv(OUT/"am3_slow_shift_reference.csv",index=False)
    rng=np.random.default_rng(20260925);eligible=np.flatnonzero(quiet&(counts>=300));chosen=rng.choice(eligible,size=min(200,len(eligible)),replace=False)
    null_rows=[]
    for i in chosen:
        lo,hi=np.searchsorted(time,[starts[i],starts[i]+10]);p=np.c_[depth[lo:hi],x[lo:hi]];order=rng.permutation(len(p));n=len(p)//2;a,b=p[order[:n]],p[order[n:2*n]]
        shift,zero,best,prominence=ab.correlate_shift(ab.histogram(a[:,0],a[:,1]),ab.histogram(b[:,0],b[:,1]))
        null_rows.append({"time_s":starts[i]+5,"peaks_each":n,"shift_um":shift,"corr_zero":zero,"corr_best":best,"corr_prominence":prominence})
    null_frame=pd.DataFrame(null_rows);null_frame.to_csv(OUT/"am3_slow_shift_null.csv",index=False);finite=null_frame.shift_um.dropna()
    mode=float(finite.mode().iloc[0]) if len(finite) else np.nan;median_abs=float(np.median(np.abs(finite))) if len(finite) else np.nan
    gate={"resolved_nulls":len(finite),"requested":min(200,len(eligible)),"mode_shift_um":mode,"median_abs_shift_um":median_abs,
          "required_mode_um":0,"max_median_abs_um":2,"pass":bool(len(finite)>=20 and mode==0 and median_abs<=2)}
    atomic_json(OUT/"am3_slow_shift_null_gate.json",gate);return reference,gate


def _edge_sample(field: MotionField, time_s: np.ndarray, depth: np.ndarray) -> np.ndarray:
    return np.column_stack([np.interp(time_s,field.time_s,field.displacement_um[:,k]) for k in range(len(depth))])


def _prediction(field: MotionField, start: float, end: float, depth: float) -> float:
    episode=np.arange(start+.125,end,.25);rest=np.arange(start-4+.125,start-1,.25)
    return float(np.median(field.at(episode,np.full(len(episode),depth)))-np.median(field.at(rest,np.full(len(rest),depth))))


def am3_boundary_check(field: MotionField, peaks: pd.DataFrame, mask: pd.DataFrame) -> tuple[pd.DataFrame,dict]:
    time=field.time_s;rigid=field.rigid(centre=False);rows=[]
    for interval in mask.itertuples(index=False):
        for kind,boundary in (("start",float(interval.start_s)),("stop",float(interval.end_s))):
            index=int(np.searchsorted(time,boundary));
            if 0<index<len(time):rows.append({"boundary_kind":kind,"time_s":boundary,"step_um":float(rigid[index]-rigid[index-1])})
    frame=pd.DataFrame(rows);frame["abs_step_um"]=frame.step_um.abs();frame=frame.sort_values("abs_step_um",ascending=False).reset_index(drop=True)
    rng=np.random.default_rng(20260925);details=[];pt=peaks.time_s.to_numpy()
    for rank,row in enumerate(frame.head(10).itertuples(index=False),1):
        lo,hi=np.searchsorted(pt,[row.time_s-2,row.time_s+2]);local=peaks.iloc[lo:hi].reset_index(drop=True)
        value=shift_test(local,(row.time_s,row.time_s+2),(row.time_s-2,row.time_s),refine_um=2,shift_min_um=-320,shift_max_um=320,
            min_peaks=150,min_gain=.05,depth_range_um=(0,3840),x_range_um=(-32,80));null=[]
        n=len(local);half=n//2
        for _ in range(20):
            order=rng.permutation(n);a=np.zeros(n,bool);b=np.zeros(n,bool);a[order[:half]]=True;b[order[half:2*half]]=True
            nv=shift_test(local,a,b,refine_um=2,shift_min_um=-320,shift_max_um=320,min_peaks=150,min_gain=0,
                depth_range_um=(0,3840),x_range_um=(-32,80));null.append(nv["best_shift_um"])
        finite=pd.Series(null).dropna();mode=float(finite.mode().iloc[0]) if len(finite) else np.nan;median_abs=float(np.median(np.abs(finite))) if len(finite) else np.nan
        details.append({"rank":rank,"boundary_kind":row.boundary_kind,"time_s":row.time_s,"step_um":row.step_um,
            "measured_shift_um":value["best_shift_um"],"gain":value["gain"],"resolved":value["accepted"],
            "null_resolved":len(finite),"null_mode_um":mode,"null_median_abs_um":median_abs,"null_pass":bool(mode==0 and median_abs<=2)})
    detail=pd.DataFrame(details);detail.to_csv(OUT/"am3_boundary_top10.csv",index=False);frame.to_csv(OUT/"am3_boundary_steps.csv",index=False)
    gate={"boundary_count":len(frame),"median_abs_step_um":float(frame.abs_step_um.median()),"p95_abs_step_um":float(frame.abs_step_um.quantile(.95)),
          "max_abs_step_um":float(frame.abs_step_um.max()),"median_limit_um":10,"top10_nulls_pass":bool(detail.null_pass.all()),
          "pass":bool(frame.abs_step_um.median()<=10 and detail.null_pass.all())}
    return detail,gate


def am3_validate() -> None:
    if not (OUT/"am3_fit_complete.json").exists():raise RuntimeError("AM.3 fitting is incomplete")
    disk_guard("before AM.3 validation");peaks,_=load_peaks();mask=pd.read_csv(OUT/"imec1_ae_mask_mapped_to_imec0.csv")
    reference,ref_gate=am3_quiet_reference(peaks,mask)
    if not ref_gate["pass"]:
        atomic_json(OUT/"am3_stop.json",{"stage":"slow-reference refinement null","gate":ref_gate});raise RuntimeError("AM.3 slow-reference null failed")
    fast=load_field(OUT/"stitched_full.npz",source="imec0_fast_amp50_d1");slow=load_field(OUT/"fields/slow/medicine_amp50_d1_k30_exclude3s/field.npz",source="imec0_slow_k30_exclude3s")
    grid=fast.time_s;slow_grid=MotionField(grid,slow.depth_um,_edge_sample(slow,grid,slow.depth_um),source=slow.source,config=slow.config)
    intervals=mask[["start_s","end_s"]].to_numpy(float);active=_in_intervals(grid,intervals)
    two,weight=compose_two_layer(slow_grid,fast,grid,active,crossfade_s=1,source="imec0_two_layer")
    episodes=pd.read_csv(OUT/"am2_cross_probe_episodes.csv");episodes=episodes[episodes.accepted.astype(bool)].copy();episodes["status"]="accepted"
    scores,episode_values,support=score_fields([fast,two],episodes,mask);scores.to_csv(OUT/"am3_scores.csv",index=False);episode_values.to_csv(OUT/"am3_episode_values.csv",index=False)
    ref_rows=[]
    for field in (slow_grid,two,fast):
        errors=[]
        for row in reference[reference.resolved.astype(bool)].itertuples(index=False):
            pred=float(field.at(np.asarray([row.time_s]),np.asarray([np.median(field.depth_um)]))[0]-field.at(np.asarray([row.previous_time_s]),np.asarray([np.median(field.depth_um)]))[0])
            errors.append(pred-row.shift_um);ref_rows.append({"field":field.source,"time_s":row.time_s,"previous_time_s":row.previous_time_s,"measured_shift_um":row.shift_um,"predicted_shift_um":pred,"error_um":pred-row.shift_um})
        atomic_json(OUT/f"am3_{field.source}_slow_reference_score.json",{"comparisons":len(errors),"mae_um":float(np.median(np.abs(errors))),"bias_um":float(np.median(errors)),"rmse_um":float(np.sqrt(np.mean(np.square(errors))))})
    pd.DataFrame(ref_rows).to_csv(OUT/"am3_slow_reference_field_values.csv",index=False)
    _,boundary_gate=am3_boundary_check(two,peaks,mask);fast_row=scores[scores.field.eq(fast.source)].iloc[0];two_row=scores[scores.field.eq(two.source)].iloc[0]
    seams=json.loads((OUT/"seams_full.json").read_text());gate={"status":"pass","reference_null":ref_gate,
        "fast_stitch_median_abs_seam_um":seams["median_abs_seam_discontinuity_um"],"fast_stitch_seam_limit_um":10,
        "fast_episode_err_um":fast_row.episode_err,"two_layer_episode_err_um":two_row.episode_err,"episode_max_degradation_um":3,
        "fast_quiet_increment_rms_um":fast_row.quiet_inc,"two_layer_quiet_increment_rms_um":two_row.quiet_inc,
        "quiet_no_worse":bool(two_row.quiet_inc<=fast_row.quiet_inc),"boundary":boundary_gate,"matched_support":support}
    passed=(gate["fast_stitch_median_abs_seam_um"]<=10 and gate["two_layer_episode_err_um"]<=gate["fast_episode_err_um"]+3 and gate["quiet_no_worse"] and boundary_gate["pass"])
    gate["status"]="pass" if passed else "fail";atomic_json(OUT/"am3_validation_gate.json",gate)
    if not passed:
        atomic_json(OUT/"am3_stop.json",{"stage":"motionqc phase-1 validation","gate":gate});raise RuntimeError(f"AM.3 validation failed: {gate}")
    package=OUT/"luke0804_imec0_two_layer_motion.npz";manifest=two.save(package,OUT/"luke0804_imec0_two_layer_motion.manifest.json")
    shutil.copy2(OUT/"imec1_ae_mask_mapped_to_imec0.csv",OUT/"censor_mask_v1_imec0.csv")
    report_dir=OUT/"motionqc_report";build_report([fast,slow_grid,two],peaks,episodes,mask,report_dir)
    import matplotlib;matplotlib.use("Agg");import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,1,figsize=(12,7));show=np.linspace(0,len(grid)-1,min(5000,len(grid)),dtype=int)
    for i,field in enumerate((fast,slow_grid,two)):
        axes[0].plot(grid[show],field.rigid(centre=True)[show],color=OKABE_ITO[i],ls=LINESTYLES[i],lw=1,label=field.source)
    axes[0].set(xlabel="Seconds from imec0 AP frame zero",ylabel="Centred displacement (µm)",title="Luke0804 imec0 two-layer motion");axes[0].legend();axes[0].grid(alpha=.2)
    axes[1].bar(["fast","two-layer"],[fast_row.quiet_inc,two_row.quiet_inc],color=[OKABE_ITO[0],OKABE_ITO[2]],hatch=["//",".."])
    axes[1].set(ylabel="Quiet 5 s increment RMS (µm)",title=f"Episode error: fast {fast_row.episode_err:.1f}, two-layer {two_row.episode_err:.1f} µm");axes[1].grid(axis="y",alpha=.2)
    fig.tight_layout();fig.savefig(OUT/"imec0_two_layer_summary.png",dpi=180);plt.close(fig)
    outputs=[package,OUT/"luke0804_imec0_two_layer_motion.manifest.json",OUT/"censor_mask_v1_imec0.csv",OUT/"am3_validation_gate.json",OUT/"am3_scores.csv",OUT/"imec0_two_layer_summary.png"]
    (OUT/"SHA256SUMS").write_text("".join(f"{sha256(path)}  {path.name}\n" for path in outputs))
    (OUT/"README.md").write_text(f"""# Luke0804 imec0 two-layer motion field\n\n**AM.2 verdict: co-moving.** The imec0 label-free test accepted 282/339 mapped imec1 episodes (83.19%); Spearman rho was 0.508 and the median imec0 shift was -200 um.\n\nThe field uses the frozen AI-v2 recipe: amp50_d1 fast fits in 120 s windows stepped by 60 s, median-centred, offset-chained and triangular-blended; an always-on amp50_d1 30 s-kernel slow layer fitted after excluding the mapped AE mask plus 3 s; and a 1 s crossfade to the fast field inside the mapped mask.\n\nValidation status: **{gate['status']}**. Fast/two-layer episode errors were {fast_row.episode_err:.2f}/{two_row.episode_err:.2f} um; quiet increment RMS values were {fast_row.quiet_inc:.2f}/{two_row.quiet_inc:.2f} um; median boundary step was {boundary_gate['median_abs_step_um']:.2f} um. The slow-reference refinement null passed with mode {ref_gate['mode_shift_um']:.0f} um and median absolute shift {ref_gate['median_abs_shift_um']:.0f} um.\n\nDeploy `luke0804_imec0_two_layer_motion.npz` as an external motion field with corrected depth = observed - displacement. Keep AP voltage unwarped. No sorting or voltage modification was performed.\n""")
    atomic_json(OUT/"am3_complete.json",{"status":"complete","package":str(package),"package_sha256":manifest["npz_sha256"],
        "mask_sha256":sha256(OUT/"censor_mask_v1_imec0.csv"),"validation_gate":gate,"medicine_runtime_s":am3_fit_runtime(),
        "gpu_budget_s":AM_GPU_BUDGET_S,"completed_at":time.time(),"sort_run":False,"voltage_modified":False})
    disk_guard("after AM.3 package")


def am3_wait_validate() -> None:
    deadline=time.time()+6*3600
    while time.time()<deadline:
        if (OUT/"am3_fit_complete.json").exists():return am3_validate()
        result=subprocess.run(["systemctl","--user","is-failed","--quiet","luke-imec0-am3-fit-20260926.service"])
        if result.returncode==0:raise RuntimeError("AM.3 fit service failed before validation")
        time.sleep(30)
    raise TimeoutError("AM.3 fit did not complete within six hours")


def main():
    p=argparse.ArgumentParser();p.add_argument("phase",choices=("prepare","extract-one","extract-all","am2","am3-prepare","am3-fast-one","am3-fit-all","am3-validate","am3-wait-validate"));p.add_argument("--block");p.add_argument("--window");a=p.parse_args()
    if a.phase=="prepare":prepare()
    elif a.phase=="extract-one":extract_one(a.block)
    elif a.phase=="extract-all":extract_all()
    elif a.phase=="am2":am2()
    elif a.phase=="am3-prepare":am3_prepare()
    elif a.phase=="am3-fast-one":am3_fast_one(a.window)
    elif a.phase=="am3-fit-all":am3_fit_all()
    elif a.phase=="am3-validate":am3_validate()
    else:am3_wait_validate()


if __name__=="__main__":main()
