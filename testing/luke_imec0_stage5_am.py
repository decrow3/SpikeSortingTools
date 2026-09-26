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

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

from motionqc.crossprobe import fit_clock_map, map_intervals
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


def main():
    p=argparse.ArgumentParser();p.add_argument("phase",choices=("prepare","extract-one","extract-all"));p.add_argument("--block");a=p.parse_args()
    if a.phase=="prepare":prepare()
    elif a.phase=="extract-one":extract_one(a.block)
    else:extract_all()


if __name__=="__main__":main()
