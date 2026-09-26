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
from motionqc.reference import matched_null
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


def main():
    p=argparse.ArgumentParser();p.add_argument("phase",choices=("prepare","extract-one","extract-all","am2","am3-prepare","am3-fast-one","am3-fit-all"));p.add_argument("--block");p.add_argument("--window");a=p.parse_args()
    if a.phase=="prepare":prepare()
    elif a.phase=="extract-one":extract_one(a.block)
    elif a.phase=="extract-all":extract_all()
    elif a.phase=="am2":am2()
    elif a.phase=="am3-prepare":am3_prepare()
    elif a.phase=="am3-fast-one":am3_fast_one(a.window)
    else:am3_fit_all()


if __name__=="__main__":main()
