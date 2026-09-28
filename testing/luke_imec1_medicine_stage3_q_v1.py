#!/usr/bin/env python
"""No-sort Luke0804 imec1 MEDiCINe sliding-window deployment (Q).

The expensive frontend is run once per covered stretch with the frozen pilot
settings.  Its label-free peak population is then materialized as a cache for
each 120 s / 60 s-step MEDiCINe window.  The 30 minute pilot is a hard gate:
the full-session phase refuses to run until the seam and frozen-score checks
pass.
"""
from __future__ import annotations

import argparse, json, os, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from testing import luke_imec1_medicine_reference_sweep_v1 as stage1
from testing import luke_imec1_medicine_m_reference_v1 as mref
from testing import luke_imec1_medicine_stage2_o_v1 as old
from testing import luke_imec1_medicine_stage2_p_v1 as pstage

PARENT = ROOT / "testing/outputs/luke_imec1_medicine_reference_sweep_v2"
STAGE2 = PARENT / "stage2_o"
DEFAULT_OUTPUT = PARENT / "stage3_q"
SESSION_DURATION_S = old.SESSION_DURATION_S
PILOT_START_S, PILOT_STOP_S = 1500.0, 3300.0
STEP_S, WINDOW_S, GRID_S = 60.0, 120.0, 0.25
GPU_BUDGET_S = 10 * 3600.0
MIN_FREE_BYTES = 50_000_000_000
BLOCK_TRANSIENT_BYTES = 19_000_000_000
SHM_ROOT = Path("/dev/shm/luke_q")
SHM_RESERVE_BYTES = 25_000_000_000
SHM_WORKER_BYTES = 20_000_000_000
MIN_MEM_AVAILABLE_BYTES = 40_000_000_000
U_VALIDATION_WINDOWS = ("pilot_w010", "pilot_w012", "pilot_w008", "pilot_w014")

def read_json(path): return json.loads(Path(path).read_text())

def selection_path():
    corrected=STAGE2/"s_correction/selection_s.json"
    return corrected if corrected.exists() else STAGE2/"selection_p.json"

def selected_choice(): return read_json(selection_path())

def window_specs(scope):
    if scope == "pilot": start, stop = PILOT_START_S, PILOT_STOP_S
    else: start, stop = 0.0, SESSION_DURATION_S
    starts = list(np.arange(start, stop - WINDOW_S + 1e-8, STEP_S))
    terminal = stop - WINDOW_S
    if not np.isclose(starts[-1], terminal): starts.append(terminal)
    return [{"id": f"{scope}_w{i:03d}", "start_s": float(a),
             "stop_s": float(a + WINDOW_S), "terminal_shifted": bool(i == len(starts)-1 and not np.isclose(a % STEP_S, 0))}
            for i, a in enumerate(starts)]

def block_specs():
    """The 87 non-overlapping extraction units covering the complete session.

    The acquisition is 33.55 s longer than 87 ordinary 120 s blocks would
    cover.  Preserve the requested count and full coverage by extending only
    the terminal block rather than dropping the tail.
    """
    starts = np.arange(0.0, 87 * WINDOW_S, WINDOW_S)
    return [{"id": f"full_block_{i:03d}", "start_s": float(a),
             "stop_s": float(SESSION_DURATION_S if i == 86 else a + WINDOW_S),
             "terminal_extended": bool(i == 86)}
            for i, a in enumerate(starts)]

def free_bytes(path):
    st=os.statvfs(path); return int(st.f_bavail*st.f_frsize)

def disk_guard(output,where):
    available=free_bytes(output)
    # Block workers call this concurrently. Serialize the shared latest receipt
    # because atomic_json intentionally uses a stable `.tmp` sibling name.
    import fcntl
    with open(output/"disk_guard.lock","a+b") as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
        stage1.atomic_json(output/"disk_guard_latest.json",{"where":where,"available_bytes":available,"required_bytes":MIN_FREE_BYTES,"passed":available>=MIN_FREE_BYTES,"checked_at":time.time()})
        fcntl.flock(lock.fileno(),fcntl.LOCK_UN)
    if available<MIN_FREE_BYTES: raise RuntimeError(f"Q T.3 disk guard failed at {where}: {available} < {MIN_FREE_BYTES} bytes")
    return available

def prepare(output):
    selection = selected_choice()
    if selection.get("outcome") not in {"selected", "provisional"}: raise RuntimeError("Q.0: P has no selected or R.4 provisional configuration")
    output.mkdir(parents=True, exist_ok=True)
    if (output/"preregistration_q.json").exists(): raise FileExistsError("Q output was already prepared")
    for x in ("runtime_cache/tmp", "runtime_cache/numba", "runtime_cache/mpl"): (output/x).mkdir(parents=True)
    base = read_json(PARENT / "extraction_config.json")
    rec = next(x for x in base["records"] if x["dataset"] == "Luke" and x["probe"] == "imec1")
    fs = float(rec["sampling_frequency_hz"]); sources=[]
    for scope in ("pilot","full"):
      for item in window_specs(scope):
        a,b=item["start_s"],item["stop_s"]
        sources.append({"id":item["id"],"dataset":"Luke","probe":"imec1","fraction":None,
            "start_frame":round(a*fs),"stop_frame":round(b*fs),"start_s":round(a*fs)/fs,"stop_s":round(b*fs)/fs})
    base["windows"] = sources; base["purpose"] = "Q pilot/full pilot-frontend label-free peak caches; no sort or voltage modification"
    stage1.atomic_json(output/"extraction_config.json",base)
    stage1.atomic_json(output/"preregistration_q.json",{
        "schema":"luke0804-imec1-medicine-stage3-q-v1","configuration":selection["config"],
        "selection_status":selection["outcome"],"selection_source":str(selection_path()),"selection_sha256":stage1.sha256(selection_path()),
        "pilot_interval_s":[PILOT_START_S,PILOT_STOP_S],"pilot_development_windows":["gaborA","gaborB"],
        "window_s":WINDOW_S,"step_s":STEP_S,"grid_s":GRID_S,
        "frontend":"one independent stage1.extract_one call per 120 s MEDiCINe window; identical denoiser-fit context to stages 1/2",
        "disk_guard_min_free_bytes":MIN_FREE_BYTES,
        "terminal_rule":"append one 120 s window ending at AP duration when the 60 s lattice does not cover the tail",
        "centering":"subtract each window's temporal median independently at each depth",
        "offsets":"chain per-depth median differences over adjacent 60 s overlaps; anchor final per-depth session median",
        "blend":"triangular weights in overlaps","pilot_gate":{"median_abs_overlap_disagreement_um":10,"score_degradation_max_um":3},
        "medicine_fit_budget_s":GPU_BUDGET_S,"no_sort":True,"voltage_modified":False})

def extract_one_window(output,window):
    disk_guard(output,f"before {window}")
    if window.startswith("full_block_"):
        extract_one_block_v(output, window)
    else:
        stage1.extract_one(output,window)
    stage=output/f"peak_cache/{window}/extraction"; pop=stage/"population.npz"
    if not (stage/"complete.json").exists() or not pop.exists(): raise RuntimeError(f"Unsealed extraction: {window}")
    with np.load(pop,allow_pickle=False) as z:
        if set(("time_s","depth_um","x_um","amplitude"))-set(z.files) or len(z["time_s"])<=100: raise RuntimeError(f"Invalid population cache: {window}")
    if (stage/"temporary_preprocessed").exists() or (stage/"detection").exists(): raise RuntimeError(f"Temporary extraction files survived validation: {window}")
    disk_guard(output,f"after {window}")

def mem_available_bytes():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("MemAvailable is missing from /proc/meminfo")

def validate_population(path, window):
    with np.load(path, allow_pickle=False) as z:
        missing=set(("time_s","depth_um","x_um","amplitude"))-set(z.files)
        if missing or len(z["time_s"]) <= 100:
            raise RuntimeError(f"Invalid population cache for {window}: missing={sorted(missing)}")

def extract_one_block_v(output, window):
    """Run the unchanged stage-1 extractor wholly in tmpfs, then publish its cache."""
    from testing.cross_dataset_fast_motion import seal
    scratch = SHM_ROOT/window
    shadow = scratch/"output"
    published = output/f"peak_cache/{window}/extraction"
    if published.exists():
        raise FileExistsError(published)
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True)
    try:
        shadow.mkdir(parents=True, exist_ok=True)
        shutil.copy2(output/"extraction_config.json", shadow/"extraction_config.json")
        old_tmp=os.environ.get("TMPDIR"); old_lock=os.environ.get("DARTSORT_RAW_PREPROCESS_LOCK")
        old_gpu_lock=os.environ.get("DARTSORT_GPU_DETECTION_LOCK")
        old_gpu_counter=os.environ.get("DARTSORT_GPU_PROBE_COUNTER")
        old_gpu_log=os.environ.get("DARTSORT_GPU_LOG_DIR")
        old_gpu_mode=os.environ.get("DARTSORT_GPU_MODE_FILE")
        old_gpu_trial=os.environ.get("DARTSORT_GPU_TRIAL_COUNTER")
        os.environ["TMPDIR"]=str(scratch/"tmp")
        os.environ["DARTSORT_RAW_PREPROCESS_LOCK"]=str(SHM_ROOT/"raw_preprocess.lock")
        os.environ["DARTSORT_GPU_DETECTION_LOCK"]=str(SHM_ROOT/"gpu_detection.lock")
        os.environ["DARTSORT_GPU_PROBE_COUNTER"]=str(output/"w_gpu_probe_counter.txt")
        os.environ["DARTSORT_GPU_LOG_DIR"]=str(output/"w_gpu_logs")
        os.environ["DARTSORT_GPU_MODE_FILE"]=str(output/"w_gpu_mode.json")
        os.environ["DARTSORT_GPU_TRIAL_COUNTER"]=str(output/"w_gpu_trial_counter.txt")
        (scratch/"tmp").mkdir()
        try:
            stage1.extract_one(shadow, window)
        finally:
            if old_tmp is None: os.environ.pop("TMPDIR",None)
            else: os.environ["TMPDIR"]=old_tmp
            if old_lock is None: os.environ.pop("DARTSORT_RAW_PREPROCESS_LOCK",None)
            else: os.environ["DARTSORT_RAW_PREPROCESS_LOCK"]=old_lock
            if old_gpu_lock is None: os.environ.pop("DARTSORT_GPU_DETECTION_LOCK",None)
            else: os.environ["DARTSORT_GPU_DETECTION_LOCK"]=old_gpu_lock
            if old_gpu_counter is None: os.environ.pop("DARTSORT_GPU_PROBE_COUNTER",None)
            else: os.environ["DARTSORT_GPU_PROBE_COUNTER"]=old_gpu_counter
            if old_gpu_log is None: os.environ.pop("DARTSORT_GPU_LOG_DIR",None)
            else: os.environ["DARTSORT_GPU_LOG_DIR"]=old_gpu_log
            if old_gpu_mode is None: os.environ.pop("DARTSORT_GPU_MODE_FILE",None)
            else: os.environ["DARTSORT_GPU_MODE_FILE"]=old_gpu_mode
            if old_gpu_trial is None: os.environ.pop("DARTSORT_GPU_TRIAL_COUNTER",None)
            else: os.environ["DARTSORT_GPU_TRIAL_COUNTER"]=old_gpu_trial
        source=shadow/f"peak_cache/{window}/extraction"
        validate_population(source/"population.npz", window)
        audit=read_json(source/"audit.json")
        audit.update({"scratch_root":str(scratch),"scratch_files_deleted_after_validation":True,
                      "published_files":["population.npz","audit.json","complete.json"],
                      "raw_preprocess_lock":str(SHM_ROOT/"raw_preprocess.lock")})
        published.mkdir(parents=True)
        shutil.copy2(source/"population.npz", published/"population.npz.tmp")
        os.replace(published/"population.npz.tmp", published/"population.npz")
        stage1.atomic_json(published/"audit.json",audit)
        seal(published)
    except BaseException:
        if published.exists() and not (published/"complete.json").exists():
            shutil.rmtree(published)
        raise
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

def population_path(output, item_id):
    return output/f"peak_cache/{item_id}/extraction/population.npz"

def load_population(path, absolute_offset_s=0.0):
    with np.load(path, allow_pickle=False) as z:
        result = {k: np.asarray(z[k]) for k in ("time_s", "depth_um", "x_um", "amplitude")}
    result["time_s"] = np.asarray(result["time_s"], float) + absolute_offset_s
    return result

def population_for_window_from_blocks(output, spec):
    arrays = {k: [] for k in ("time_s", "depth_um", "x_um", "amplitude")}
    used = []
    for block in block_specs():
        if block["stop_s"] <= spec["start_s"] or block["start_s"] >= spec["stop_s"]:
            continue
        path = population_path(output, block["id"])
        pop = load_population(path, block["start_s"])
        keep = (pop["time_s"] >= spec["start_s"]) & (pop["time_s"] < spec["stop_s"])
        arrays["time_s"].append(pop["time_s"][keep] - spec["start_s"])
        for key in ("depth_um", "x_um", "amplitude"):
            arrays[key].append(pop[key][keep])
        used.append({"id": block["id"], "sha256": stage1.sha256(path)})
    if not used:
        raise RuntimeError(f"No extraction block overlaps {spec['id']}")
    result = {key: np.concatenate(parts) for key, parts in arrays.items()}
    order = np.argsort(result["time_s"], kind="stable")
    return {key: values[order] for key, values in result.items()}, used

def extract_windows(output,scope):
    disk_guard(output,f"before {scope} extraction launch")
    rows=[]
    for spec in window_specs(scope):
        stage=output/f"peak_cache/{spec['id']}/extraction"
        if not (stage/"complete.json").exists():
            cmd=[str(stage1.DSPY),str(Path(__file__).resolve()),"extract-one-window","--output",str(output),"--window",spec["id"]]
            result=subprocess.run(cmd)
            if result.returncode: raise RuntimeError(f"Per-window extraction failed: {spec['id']}")
        audit=read_json(stage/"audit.json"); rows.append({"window":spec["id"],"seconds":audit["seconds"],"population_bytes":(stage/"population.npz").stat().st_size})
        disk_guard(output,f"between windows after {spec['id']}")
        stage1.atomic_json(output/f"{scope}_extraction_progress.json",{"status":"running","completed":len(rows),"planned":len(window_specs(scope)),"rows":rows,
            "total_frontend_seconds":sum(x["seconds"] for x in rows),"gpu_budget_counted":False})
    stage1.atomic_json(output/f"{scope}_extraction_complete.json",{"status":"complete","completed":len(rows),"rows":rows,
        "total_frontend_seconds":sum(x["seconds"] for x in rows),"gpu_budget_counted":False})

def selected_settings():
    choice=selected_choice(); change=dict(old.CONFIGS[choice["config"]]); seed=int(change.pop("seed",0)); settings=dict(stage1.BASE_MED); settings.update(change)
    return choice,settings,seed

def replan_per_window_t(output):
    """T pre-fit correction: exact independent 120 s frontend per fit window."""
    if list(output.glob("fields/*/*/receipt.json")): raise RuntimeError("Cannot replan Q after a fit")
    import shutil
    prereg=output/"preregistration_q.json"; cfgp=output/"extraction_config.json"
    if not (output/"preregistration_q_source_chunks_prefit_superseded_t.json").exists(): shutil.copy2(prereg,output/"preregistration_q_source_chunks_prefit_superseded_t.json")
    if not (output/"extraction_config_source_chunks_prefit_superseded_t.json").exists(): shutil.copy2(cfgp,output/"extraction_config_source_chunks_prefit_superseded_t.json")
    cfg=read_json(cfgp); record=next(x for x in cfg["records"] if x["dataset"]=="Luke" and x["probe"]=="imec1"); fs=float(record["sampling_frequency_hz"]); windows=[]
    for scope in ("pilot","full"):
        for item in window_specs(scope):
            a,b=item["start_s"],item["stop_s"]
            windows.append({"id":item["id"],"dataset":"Luke","probe":"imec1","fraction":None,"start_frame":round(a*fs),"stop_frame":round(b*fs),"start_s":round(a*fs)/fs,"stop_s":round(b*fs)/fs})
    cfg["windows"]=windows; cfg["purpose"]="Q correction T: independent stage1.extract_one for every 120 s / 60 s-step window; no sorting or voltage modification"; stage1.atomic_json(cfgp,cfg)
    audits=list(STAGE2.glob("peak_cache/*/extraction/audit.json")); seconds=[read_json(x)["seconds"] for x in audits]
    pops=list(STAGE2.glob("peak_cache/*/extraction/population.npz")); sizes=[x.stat().st_size for x in pops]
    estimate={"basis":"12 validated stage-2 120 s extractions","n_basis":len(seconds),"median_extraction_s_per_window":float(np.median(seconds)),
        "mean_extraction_s_per_window":float(np.mean(seconds)),"median_population_cache_bytes":int(np.median(sizes)),
        "estimated_peak_transient_bytes":19_000_000_000,"components":"about 5.5 GB preprocessed recording + about 11 GB denoiser-fit scratch + about 1.9 GB detection HDF5, rounded up",
        "pilot_windows":len(window_specs("pilot")),"full_windows":len(window_specs("full")),"frontend_time_not_counted_against_gpu_budget":True}
    stage1.atomic_json(output/"extraction_estimate_t.json",estimate)
    choice=selected_choice(); plan=read_json(prereg); plan.update(configuration=choice["config"],selection_status=choice["outcome"],selection_source=str(selection_path()),
        selection_sha256=stage1.sha256(selection_path()),frontend="one independent stage1.extract_one invocation per 120 s fit window, identical to stages 1/2",
        frontend_windows={"pilot":window_specs("pilot"),"full":window_specs("full")},disk_guard_min_free_bytes=MIN_FREE_BYTES,
        tmpdir=str(output/"runtime_tmp"),extraction_estimate=estimate,t_replan_before_any_q_fit=True,t_replan_at=time.time())
    for key in ("frontend_source_chunks_s","frontend_source_chunks","frontend_reason"): plan.pop(key,None)
    stage1.atomic_json(prereg,plan)

def replan_blocks_u(output):
    """Pre-gate U amendment: retain Q.1, replace only the full extraction plan."""
    if list(output.glob("fields/full/*/receipt.json")):
        raise RuntimeError("Cannot apply U after a full-session fit")
    prereg = output/"preregistration_q.json"
    cfgp = output/"extraction_config.json"
    backup = output/"preregistration_q_exact_full_prefit_superseded_u.json"
    cfgbackup = output/"extraction_config_exact_full_prefit_superseded_u.json"
    if not backup.exists(): shutil.copy2(prereg, backup)
    if not cfgbackup.exists(): shutil.copy2(cfgp, cfgbackup)
    cfg = read_json(cfgp)
    record = next(x for x in cfg["records"] if x["dataset"] == "Luke" and x["probe"] == "imec1")
    fs = float(record["sampling_frequency_hz"])
    # Retain exact window records for the pre-committed validation fallback,
    # even though the passing path extracts only the block records.
    existing = {}
    for scope in ("pilot", "full"):
        for item in window_specs(scope):
            a, b = item["start_s"], item["stop_s"]
            existing[item["id"]] = {"id": item["id"], "dataset": "Luke", "probe": "imec1", "fraction": None,
                "start_frame": round(a*fs), "stop_frame": round(b*fs),
                "start_s": round(a*fs)/fs, "stop_s": round(b*fs)/fs}
    for item in block_specs():
        a, b = item["start_s"], item["stop_s"]
        existing[item["id"]] = {"id": item["id"], "dataset": "Luke", "probe": "imec1", "fraction": None,
            "start_frame": round(a*fs), "stop_frame": round(b*fs),
            "start_s": round(a*fs)/fs, "stop_s": round(b*fs)/fs}
    cfg["windows"] = list(existing.values())
    cfg["purpose"] = "Q/U: unchanged exact Q.1 pilot plus one extraction per non-overlapping full-session block; no sorting or voltage modification"
    stage1.atomic_json(cfgp, cfg)
    plan = read_json(prereg)
    plan.update({
        "u_replan_before_full_fit": True,
        "u_replan_at": time.time(),
        "pilot_frontend_unchanged": True,
        "full_frontend": "stage1.extract_one once per non-overlapping block; construct each 120 s/60 s-step fit population in memory from overlapping block caches",
        "full_extraction_blocks": block_specs(),
        "full_extraction_block_count": len(block_specs()),
        "terminal_block_rule": "86 ordinary 120 s blocks plus one terminal block extended to 153.55 s, preserving 87 units and complete AP coverage",
        "block_concat_validation_windows": list(U_VALIDATION_WINDOWS),
        "block_concat_gate": {"abs_delta_episode_err_max_um": 3.0, "median_abs_field_difference_max_um": 5.0,
                              "failure_action": "fall back to exact per-window extraction"},
        "block_worker_rule": "max(1,min(4,floor((free_bytes-50e9)/19e9))); recompute before every launch",
        "estimated_peak_transient_bytes_per_worker": BLOCK_TRANSIENT_BYTES,
    })
    stage1.atomic_json(prereg, plan)
    stage1.atomic_json(output/"u_plan.json", {"status": "armed_before_q1_release", "pilot_unchanged": True,
        "blocks": block_specs(), "validation_windows": list(U_VALIDATION_WINDOWS),
        "worker_rule": plan["block_worker_rule"], "created_at": time.time(), "no_sort": True, "voltage_modified": False})

def worker_limit(output):
    disk_guard(output, "before block worker launch")
    memory = mem_available_bytes()
    shm_free = free_bytes(SHM_ROOT.parent)
    if memory < MIN_MEM_AVAILABLE_BYTES:
        raise RuntimeError(f"V memory guard failed: MemAvailable {memory} < {MIN_MEM_AVAILABLE_BYTES}")
    limit=min(3, int((shm_free-SHM_RESERVE_BYTES)//SHM_WORKER_BYTES))
    if limit < 1:
        raise RuntimeError(f"V tmpfs guard failed: /dev/shm free {shm_free} leaves no 20 GB worker above 25 GB reserve")
    checkpoint=output/"v_first_10_more_blocks.json"
    if checkpoint.exists() and read_json(checkpoint).get("drop_to_one"):
        return 1
    return limit

def completed_block_rows(output):
    rows = []
    for block in block_specs():
        stage = output/f"peak_cache/{block['id']}/extraction"
        if not (stage/"complete.json").exists():
            continue
        audit = read_json(stage/"audit.json")
        rows.append({"block": block["id"], "start_s": block["start_s"], "stop_s": block["stop_s"],
                     "seconds": audit["seconds"], "population_bytes": population_path(output, block["id"]).stat().st_size})
    return rows

def maybe_first_ten_report(output):
    report = output/"u_first_10_blocks.json"
    rows = completed_block_rows(output)
    if len(rows) < 10 or report.exists():
        return
    started = read_json(output/"u_block_extraction_started.json")["started_at"]
    now = time.time()
    first = rows[:10]
    median_block_s = float(np.median([x["seconds"] for x in first]))
    current_workers = worker_limit(output)
    pilot_fit_s = [read_json(p)["runtime_s"] for p in output.glob("fields/pilot/*/receipt.json")]
    median_fit_s = float(np.median(pilot_fit_s)) if pilot_fit_s else np.nan
    remaining_block_s = (len(block_specs())-10)*median_block_s/current_workers
    remaining_fit_s = len(window_specs("full"))*median_fit_s if np.isfinite(median_fit_s) else np.nan
    projected = (now-started) + remaining_block_s + remaining_fit_s
    stage1.atomic_json(report, {"status": "continue", "completed_blocks": 10,
        "actual_wall_s_from_first_block_launch": now-started, "median_extraction_s_per_block": median_block_s,
        "workers_at_projection": current_workers, "projected_remaining_extraction_s": remaining_block_s,
        "median_pilot_medicine_fit_s": median_fit_s, "projected_full_medicine_fit_s": remaining_fit_s,
        "projected_total_wall_s_from_first_block_launch": projected,
        "projected_over_24h": bool(projected > 24*3600),
        "gpu_hours_used": fit_runtime(output)/3600.0,
        "note": "Frontend extraction time is excluded from the GPU budget; continue even if projected over 24 h per U.3.",
        "reported_at": now})

def replan_v(output):
    """Record V before resuming; completed U caches and fits remain reusable."""
    SHM_ROOT.mkdir(parents=True, exist_ok=True)
    baseline=len(completed_block_rows(output))
    plan=read_json(output/"preregistration_q.json")
    plan.update({"v_replan_at":time.time(),"v_baseline_completed_blocks":baseline,
        "v_scratch_root":str(SHM_ROOT),
        "v_raw_preprocess_lock":str(SHM_ROOT/"raw_preprocess.lock"),
        "v_worker_rule":"min(3, floor((shm_free-25GB)/20GB)); MemAvailable>=40GB; disk free>=50GB",
        "v_publish":"validated population.npz plus small audit and complete receipts only",
        "v_cleanup":"per-block tmpfs tree removed after success and on failure",
        "v_checkpoint_after_additional_blocks":10})
    stage1.atomic_json(output/"preregistration_q.json",plan)
    stage1.atomic_json(output/"v_plan.json",{"status":"armed","baseline_completed_blocks":baseline,
        "started_at":time.time(),"checkpoint_at_completed_blocks":baseline+10,
        "single_worker_baseline_blocks_per_hour":4.1,"minimum_parallel_blocks_per_hour":8.2,
        "shm_free_bytes":free_bytes(SHM_ROOT.parent),"mem_available_bytes":mem_available_bytes(),
        "disk_free_bytes":disk_guard(output,"V prelaunch"),"reuse_completed_blocks":baseline,
        "reuse_completed_validation_fits":len(list(output.glob("fields/u_validation/*/receipt.json")))})

def apply_w(output):
    """Document W and release the live controller to pipeline future blocks."""
    checkpoint=output/"v_first_10_more_blocks.json"
    backup=output/"v_first_10_more_blocks_frozen_pre_w.json"
    if not checkpoint.exists(): raise FileNotFoundError(checkpoint)
    if not backup.exists(): shutil.copy2(checkpoint,backup)
    completed={int(p.parts[-3].split("_")[-1]) for p in output.glob("peak_cache/full_block_*/extraction/complete.json")}
    inflight=sorted(int(p.name.split("_")[-1]) for p in SHM_ROOT.glob("full_block_*") if p.is_dir())
    first_pending=next(i for i in range(len(block_specs())) if i not in completed and i not in inflight)
    value=read_json(checkpoint)
    value.update({"drop_to_one":False,"status":"overridden_by_W_pipeline",
        "w_override_at":time.time(),"w_reason":"V fallback was a throughput heuristic; W authorizes serialized raw/preprocess plus serialized GPU detection with at most 3 processes",
        "w_original_checkpoint":str(backup)})
    stage1.atomic_json(checkpoint,value)
    (output/"w_gpu_logs").mkdir(exist_ok=True)
    for path in (output/"w_gpu_probe_counter.txt",output/"w_gpu_probe_counter.txt.lock"):
        if path.exists(): path.unlink()
    stage1.atomic_json(output/"w_gpu_mode.json",{"mode":"single_locked","reason":"W.2 probe pending","updated_at":time.time()})
    stage1.atomic_json(output/"w_schedule.json",{"status":"armed_between_blocks","applied_at":time.time(),
        "completed_at_apply":sorted(completed),"inflight_at_apply":inflight,
        "first_pipeline_block":first_pending,"five_pipeline_blocks":list(range(first_pending,first_pending+5)),
        "raw_preprocess_lock":str(SHM_ROOT/"raw_preprocess.lock"),"gpu_detection_lock":str(SHM_ROOT/"gpu_detection.lock"),
        "max_processes":3,"scientific_parameters_changed":False,"rerun_completed_blocks":False,
        "gpu_probe":{"sample_interval_s":1,"stages":2,"dual_trial_condition":{"mean_utilization_below_percent":60,"peak_memory_below_mib":10240}}})

def maybe_v_checkpoint(output):
    path=output/"v_first_10_more_blocks.json"
    if path.exists() or not (output/"v_plan.json").exists(): return
    plan=read_json(output/"v_plan.json"); rows=completed_block_rows(output)
    if len(rows)<plan["checkpoint_at_completed_blocks"]: return
    baseline=plan["baseline_completed_blocks"]
    new=rows[baseline:plan["checkpoint_at_completed_blocks"]]
    wall=time.time()-plan["started_at"]
    phases={k:[] for k in ("raw_read","preprocess","detection_and_denoiser_fit","lock_wait")}
    for row in new:
        audit=read_json(output/f"peak_cache/{row['block']}/extraction/audit.json")
        for key in phases: phases[key].append(float(audit.get("phase_seconds",{}).get(key,np.nan)))
    means={k:float(np.nanmean(v)) for k,v in phases.items()}
    throughput=len(new)/wall*3600
    drop=throughput<8.2
    remaining=len(block_specs())-len(rows)
    projected=time.time()+remaining/max(throughput,1e-9)*3600+len(window_specs("full"))*float(np.median(
        [read_json(p)["runtime_s"] for p in output.glob("fields/pilot/*/receipt.json")]))
    gate=read_json(output/"u_block_concat_gate.json") if (output/"u_block_concat_gate.json").exists() else {"status":"not_yet_run"}
    stage1.atomic_json(path,{"status":"drop_to_one" if drop else "continue_parallel","new_blocks":len(new),
        "completed_blocks":len(rows),"wall_s":wall,"blocks_per_hour":throughput,
        "mean_phase_seconds":means,"mean_medicine_fit_s":float(np.mean(
            [read_json(p)["runtime_s"] for p in output.glob("fields/u_validation/*/receipt.json")])),
        "projected_finish_unix":projected,"projected_finish_local":time.strftime("%Y-%m-%d %H:%M:%S %Z",time.localtime(projected)),
        "u_validation":gate,"drop_to_one":drop,
        "criterion":"drop to one worker when throughput < 2x 4.1 block/h single-worker baseline",
        "gpu_hours_used":fit_runtime(output)/3600.0,"reported_at":time.time()})

def extract_block_subset(output, wanted):
    wanted = set(wanted)
    marker = output/"u_block_extraction_started.json"
    if not marker.exists():
        stage1.atomic_json(marker, {"started_at": time.time(), "planned_blocks": len(block_specs())})
    running = {}
    pending = [b for b in block_specs() if b["id"] in wanted and not (output/f"peak_cache/{b['id']}/extraction/complete.json").exists()]
    while pending or running:
        launched = False
        while pending:
            limit = worker_limit(output)
            if len(running) >= limit:
                break
            block = pending.pop(0)
            cmd = [str(stage1.DSPY), str(Path(__file__).resolve()), "extract-one-window", "--output", str(output), "--window", block["id"]]
            proc = subprocess.Popen(cmd)
            running[proc] = block
            launched = True
        if not running:
            continue
        if launched:
            time.sleep(1)
        finished = [p for p in running if p.poll() is not None]
        if not finished:
            time.sleep(5)
            continue
        for proc in finished:
            block = running.pop(proc)
            if proc.returncode:
                # Do not rely on systemd's cgroup termination to run each
                # child's Python finally block. Stop siblings deliberately and
                # remove only this launch's per-block tmpfs trees.
                for sibling in running:
                    sibling.terminate()
                for sibling in running:
                    try: sibling.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        sibling.kill(); sibling.wait()
                for sibling_block in list(running.values())+[block]:
                    shutil.rmtree(SHM_ROOT/sibling_block["id"],ignore_errors=True)
                raise RuntimeError(f"Block extraction failed: {block['id']} (exit {proc.returncode})")
            maybe_first_ten_report(output)
            maybe_v_checkpoint(output)
        rows = completed_block_rows(output)
        stage1.atomic_json(output/"full_block_extraction_progress.json", {"status": "running", "completed": len(rows),
            "planned": len(block_specs()), "rows": rows, "gpu_budget_counted": False, "updated_at": time.time()})

def extract_blocks_u(output):
    extract_block_subset(output, [x["id"] for x in block_specs()])
    rows = completed_block_rows(output)
    stage1.atomic_json(output/"full_block_extraction_complete.json", {"status": "complete", "completed": len(rows),
        "planned": len(block_specs()), "rows": rows, "total_frontend_seconds": sum(x["seconds"] for x in rows),
        "actual_wall_s": time.time()-read_json(output/"u_block_extraction_started.json")["started_at"],
        "gpu_budget_counted": False})

def run_fit_population(output, spec, target, times, depths, amps, scope, source):
    import medicine, torch
    target.mkdir(parents=True,exist_ok=False)
    choice,settings,seed=selected_settings(); torch.set_num_threads(4)
    if not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
    np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); begun=time.monotonic()
    trainer=medicine.run_medicine(peak_times=times,peak_depths=depths,peak_amplitudes=amps,output_dir=target/"medicine",optimizer=torch.optim.Adam,**settings)
    runtime=time.monotonic()-begun; med=target/"medicine"; ft,fz,fm=(np.load(med/x) for x in ("time_bins.npy","depth_bins.npy","motion.npy"))
    np.savez_compressed(target/"field.npz",time_s=ft,session_time_s=ft+spec["start_s"],depth_um=fz,displacement_um=fm,
        sign_contract=np.asarray("observed_minus_registered; corrected=observed-displacement"))
    np.save(target/"loss.npy",np.asarray(trainer.losses))
    stage1.atomic_json(target/"receipt.json",{"status":"complete","scope":scope,"window":spec["id"],"start_s":spec["start_s"],"stop_s":spec["stop_s"],
        "config":choice["config"],"selection_status":choice["outcome"],"settings":settings,"seed":seed,"runtime_s":runtime,"peaks":len(times),
        "population_source":source,"field_sha256":stage1.sha256(target/"field.npz")})

def fit_one(output,scope,window):
    spec=next(x for x in window_specs(scope) if x["id"]==window); target=output/f"fields/{scope}/{window}"
    mode = read_json(output/"u_full_mode.json")["mode"] if scope == "full" and (output/"u_full_mode.json").exists() else "exact_window"
    if scope == "full" and mode == "block_concat":
        pop, used = population_for_window_from_blocks(output, spec)
        source = {"mode": mode, "blocks": used}
        times,depths,amps=(pop[k] for k in ("time_s","depth_um","amplitude"))
    else:
        path=population_path(output, window)
        with np.load(path,allow_pickle=False) as z: times,depths,amps=(np.asarray(z[k]) for k in ("time_s","depth_um","amplitude"))
        source={"mode":"exact_window","path":str(path),"sha256":stage1.sha256(path)}
    run_fit_population(output,spec,target,times,depths,amps,scope,source)

def fit_runtime(output):
    return sum(read_json(p)["runtime_s"] for p in output.glob("fields/*/*/receipt.json"))

def sweep(output,scope):
    if scope=="full" and read_json(output/"pilot_gate.json")["status"]!="pass": raise RuntimeError("Q.1 pilot gate did not pass")
    for spec in window_specs(scope):
        receipt=output/f"fields/{scope}/{spec['id']}/receipt.json"
        if receipt.exists(): continue
        if fit_runtime(output)>=GPU_BUDGET_S: raise RuntimeError("Stopped at Q 10 GPU-hour MEDiCINe budget")
        cmd=[str(stage1.MEDPY),str(Path(__file__).resolve()),"fit-one","--output",str(output),"--scope",scope,"--window",spec["id"]]
        result=subprocess.run(cmd,timeout=old.FIT_TIMEOUT_S)
        if result.returncode: raise RuntimeError(f"fit failed: {scope}/{spec['id']}")
    stage1.atomic_json(output/f"{scope}_sweep_complete.json",{"status":"complete","fits":len(window_specs(scope)),"cumulative_medicine_runtime_s":fit_runtime(output),"budget_s":GPU_BUDGET_S})

def load_field(path):
    with np.load(path,allow_pickle=False) as z:return {k:np.asarray(z[k]) for k in z.files}

def stitch(output,scope):
    specs=window_specs(scope); fields=[load_field(output/f"fields/{scope}/{x['id']}/field.npz") for x in specs]
    depth=np.asarray(fields[0]["depth_um"],float); aligned=[]; offsets=[]; seams=[]
    for i,(spec,f) in enumerate(zip(specs,fields)):
        t=np.asarray(f["session_time_s"],float); y=np.asarray(f["displacement_um"],float); y=y-np.nanmedian(y,axis=0,keepdims=True)
        if i==0: off=np.zeros(len(depth))
        else:
            pt,py=aligned[-1]; a=max(t.min(),pt.min()); b=min(t.max(),pt.max()); grid=np.arange(a,b+1e-8,GRID_S)
            cur=np.column_stack([np.interp(grid,t,y[:,k]) for k in range(len(depth))]); prev=np.column_stack([np.interp(grid,pt,py[:,k]) for k in range(len(depth))])
            off=np.nanmedian(prev-cur,axis=0); residual=prev-(cur+off)
            seams.append({"left":specs[i-1]["id"],"right":spec["id"],"median_abs_um":float(np.nanmedian(np.abs(residual))),"max_abs_um":float(np.nanmax(np.abs(residual)))})
        yy=y+off; aligned.append((t,yy)); offsets.append(off)
    start=PILOT_START_S if scope=="pilot" else 0.; stop=PILOT_STOP_S if scope=="pilot" else SESSION_DURATION_S
    grid=np.arange(start,stop,GRID_S); num=np.zeros((len(grid),len(depth))); den=np.zeros_like(num)
    for spec,(t,y) in zip(specs,aligned):
        center=(spec["start_s"]+spec["stop_s"])/2; w=np.maximum(0.,1.-np.abs(grid-center)/(WINDOW_S/2)); inside=(grid>=spec["start_s"])&(grid<=spec["stop_s"])
        w=np.where(inside,np.maximum(w,1e-6),0.)
        for k in range(len(depth)):
            vals=np.interp(grid,t,y[:,k],left=np.nan,right=np.nan); good=np.isfinite(vals)&(w>0); num[good,k]+=w[good]*vals[good]; den[good,k]+=w[good]
    result=np.divide(num,den,out=np.full_like(num,np.nan),where=den>0); result-=np.nanmedian(result,axis=0,keepdims=True)
    boundary_fills=[]
    for k in range(len(depth)):
        finite=np.flatnonzero(np.isfinite(result[:,k]))
        if not len(finite): raise RuntimeError(f"stitched {scope} depth {k} has no finite support")
        first,last=int(finite[0]),int(finite[-1])
        if np.isnan(result[first:last+1,k]).any(): raise RuntimeError(f"stitched {scope} depth {k} has an interior support gap")
        result[:first,k]=result[first,k]; result[last+1:,k]=result[last,k]
        boundary_fills.append({"depth_index":k,"leading_samples":first,"trailing_samples":len(grid)-last-1})
    out=output/f"stitched_{scope}.npz"; np.savez_compressed(out,time_s=grid,session_time_s=grid,depth_um=depth,displacement_um=result,
        sign_convention=np.asarray("corrected = observed - displacement"))
    stage1.atomic_json(output/f"seams_{scope}.json",{"median_abs_seam_discontinuity_um":float(np.median([x['median_abs_um'] for x in seams])),
        "seams":seams,"offsets_um":[x.tolist() for x in offsets],
        "boundary_fill":{"method":"nearest finite stitched sample; boundaries only","per_depth":boundary_fills}})
    return load_field(out)

def score_field(field,windows,cand,measured,nulls):
    errs,qabs,inc=[] ,[],[]
    for w,bounds in windows.items():
        for ep in measured.loc[(measured.window==w)&measured.accepted].itertuples(): errs.append(abs(mref.episode_prediction(field,ep.start_s,ep.stop_s)-ep.best_shift_um))
        for n in nulls.loc[(nulls.window==w)&nulls.resolved].itertuples(): qabs.append(abs(mref.episode_prediction(field,n.start_s,n.stop_s)))
        vals,_=old.corrected_quiet_values(field,cand.loc[cand.window==w],bounds); inc.extend(vals)
    return {"episode_err":float(np.median(errs)) if errs else None,"quiet_abs":float(np.median(qabs)) if qabs else None,
        "quiet_inc":float(np.sqrt(np.mean(np.square(inc)))) if inc else None,"accepted_episodes":len(errs),"resolved_nulls":len(qabs)}

def pilot_report(output):
    import pandas as pd
    stitched=stitch(output,"pilot"); windows={k:stage1.WINDOWS[k] for k in ("gaborA","gaborB")}
    cand=pd.read_csv(PARENT/"references_m/candidate_episodes.csv"); measured=pd.read_csv(PARENT/"references_m/measured_episodes.csv"); nulls=pd.read_csv(PARENT/"references_m/null_pseudo_episodes.csv")
    stitched_score=score_field(stitched,windows,cand,measured,nulls)
    choice=selected_choice(); per=[]
    for w in windows:
        per.append(load_field(pstage.field_path(STAGE2,choice["config"],w)))
    # Pool exact per-window predictions using one field at a time.
    es,qa,iv=[],[],[]
    for w,f in zip(windows,per):
        one=score_field(f,{w:windows[w]},cand,measured,nulls); 
        for ep in measured.loc[(measured.window==w)&measured.accepted].itertuples(): es.append(abs(mref.episode_prediction(f,ep.start_s,ep.stop_s)-ep.best_shift_um))
        for n in nulls.loc[(nulls.window==w)&nulls.resolved].itertuples(): qa.append(abs(mref.episode_prediction(f,n.start_s,n.stop_s)))
        x,_=old.corrected_quiet_values(f,cand.loc[cand.window==w],windows[w]); iv.extend(x)
    baseline={"episode_err":float(np.median(es)) if es else None,"quiet_abs":float(np.median(qa)) if qa else None,"quiet_inc":float(np.sqrt(np.mean(np.square(iv)))) if iv else None}
    degradation={k:stitched_score[k]-baseline[k] for k in baseline}; seam=read_json(output/"seams_pilot.json")["median_abs_seam_discontinuity_um"]
    passed=seam<=10 and all(v<=3 for v in degradation.values())
    stage1.atomic_json(output/"pilot_gate.json",{"status":"pass" if passed else "fail","seam_median_abs_um":seam,"stitched":stitched_score,"per_window":baseline,"degradation_um":degradation,
        "criterion":"seam<=10um and stitched minus per-window <=3um for episode_err, quiet_abs, corrected quiet_inc"})
    if not passed: raise RuntimeError("Q.1 gate failed; full session is not authorized")

def centered_field_difference(a, b):
    ta=np.asarray(a["session_time_s"],float); tb=np.asarray(b["session_time_s"],float)
    da=np.asarray(a["depth_um"],float); db=np.asarray(b["depth_um"],float)
    rigid = len(da)==len(db)==1
    if not rigid and (len(da)!=len(db) or not np.allclose(da,db)):
        raise RuntimeError("U validation fields have different depth grids")
    ya=np.asarray(a["displacement_um"],float); yb=np.asarray(b["displacement_um"],float)
    ya=ya-np.nanmedian(ya,axis=0,keepdims=True); yb=yb-np.nanmedian(yb,axis=0,keepdims=True)
    lo=max(ta.min(),tb.min()); hi=min(ta.max(),tb.max()); grid=np.arange(lo,hi+1e-8,GRID_S)
    ai=np.column_stack([np.interp(grid,ta,ya[:,k]) for k in range(ya.shape[1])])
    bi=np.column_stack([np.interp(grid,tb,yb[:,k]) for k in range(yb.shape[1])])
    return float(np.nanmedian(np.abs(ai-bi)))

def fit_validation_u(output, window):
    spec=next(x for x in window_specs("pilot") if x["id"]==window)
    target=output/f"fields/u_validation/{window}"
    pop,used=population_for_window_from_blocks(output,spec)
    run_fit_population(output,spec,target,pop["time_s"],pop["depth_um"],pop["amplitude"],"u_validation",
                       {"mode":"block_concat_validation","blocks":used})

def validate_block_concat_u(output):
    """Validate separately denoised block populations against exact Q.1 fits."""
    if read_json(output/"pilot_gate.json")["status"] != "pass":
        raise RuntimeError("Q.1 must pass before U validation")
    specs={x["id"]:x for x in window_specs("pilot")}
    needed=set()
    for window in U_VALIDATION_WINDOWS:
        spec=specs[window]
        needed.update(b["id"] for b in block_specs() if b["stop_s"]>spec["start_s"] and b["start_s"]<spec["stop_s"])
    extract_block_subset(output, needed)
    comparisons=[]
    for window in U_VALIDATION_WINDOWS:
        spec=specs[window]; exact_path=output/f"fields/pilot/{window}/field.npz"
        if not exact_path.exists(): raise FileNotFoundError(exact_path)
        target=output/f"fields/u_validation/{window}"
        if not (target/"receipt.json").exists():
            if fit_runtime(output)>=GPU_BUDGET_S: raise RuntimeError("Stopped at Q 10 GPU-hour MEDiCINe budget during U validation")
            cmd=[str(stage1.MEDPY),str(Path(__file__).resolve()),"fit-validation-u","--output",str(output),"--window",window]
            result=subprocess.run(cmd,timeout=old.FIT_TIMEOUT_S)
            if result.returncode: raise RuntimeError(f"U validation fit failed: {window}")
        exact=load_field(exact_path); concat=load_field(target/"field.npz")
        comparisons.append({"window":window,"median_abs_field_difference_um":centered_field_difference(exact,concat)})
    import pandas as pd
    measured=pd.read_csv(PARENT/"references_m/measured_episodes.csv")
    exact_err=[]; concat_err=[]; episode_rows=[]
    for window in U_VALIDATION_WINDOWS:
        spec=specs[window]; exact=load_field(output/f"fields/pilot/{window}/field.npz"); concat=load_field(output/f"fields/u_validation/{window}/field.npz")
        eligible=measured[(measured.accepted)&(measured.start_s-4>=spec["start_s"])&(measured.stop_s<=spec["stop_s"])]
        for ep in eligible.itertuples():
            ep_exact=abs(mref.episode_prediction(exact,ep.start_s,ep.stop_s)-ep.best_shift_um)
            ep_concat=abs(mref.episode_prediction(concat,ep.start_s,ep.stop_s)-ep.best_shift_um)
            exact_err.append(ep_exact); concat_err.append(ep_concat)
            episode_rows.append({"fit_window":window,"reference_window":ep.window,"episode_id":ep.episode_id,
                                 "exact_abs_err_um":ep_exact,"block_concat_abs_err_um":ep_concat})
    if not exact_err: raise RuntimeError("U validation selected no accepted episodes")
    exact_score=float(np.median(exact_err)); concat_score=float(np.median(concat_err)); delta=abs(concat_score-exact_score)
    field_diff=float(np.median([x["median_abs_field_difference_um"] for x in comparisons]))
    passed=delta<=3.0 and field_diff<=5.0
    gate={"status":"pass" if passed else "fail","validation_windows":list(U_VALIDATION_WINDOWS),
          "accepted_episode_predictions":len(exact_err),"exact_episode_err_um":exact_score,
          "block_concat_episode_err_um":concat_score,"abs_delta_episode_err_um":delta,
          "median_abs_field_difference_um":field_diff,"per_window":comparisons,"episodes":episode_rows,
          "criteria":{"abs_delta_episode_err_max_um":3.0,"median_abs_field_difference_max_um":5.0},
          "failure_action":"exact per-window extraction","completed_at":time.time()}
    stage1.atomic_json(output/"u_block_concat_gate.json",gate)
    stage1.atomic_json(output/"u_full_mode.json",{"mode":"block_concat" if passed else "exact_window_fallback",
        "gate_status":gate["status"],"decided_at":time.time()})

def extract_full_u(output):
    mode=read_json(output/"u_full_mode.json")["mode"]
    if mode=="block_concat":
        extract_blocks_u(output)
    elif mode=="exact_window_fallback":
        extract_windows(output,"full")
    else:
        raise RuntimeError(f"Unknown U full extraction mode: {mode}")

def shared_field():
    with np.load(pstage.SHARED_FIELD,allow_pickle=False) as z:
        t=np.asarray(z["time_bin_centers_s"],float); d=np.asarray(z["spatial_bin_centers_um"],float); y=np.asarray(z["displacement"],float).T
    return {"time_s":t,"session_time_s":t,"depth_um":d,"displacement_um":y}

def score_collection(provider,windows,cand,measured,nulls,blocks,quiet_windows):
    errs,ratios,berr,qabs,incs,falsefracs=[],[],[],[],[],[]
    for w,bounds in windows.items():
        f=provider(w)
        for ep in measured.loc[(measured.window==w)&measured.accepted].itertuples():
            pred=mref.episode_prediction(f,ep.start_s,ep.stop_s); errs.append(abs(pred-ep.best_shift_um)); ratios.append(pred/ep.best_shift_um)
        if len(blocks):
            for b in blocks.loc[(blocks.window==w)&blocks.accepted].itertuples():
                berr.append(abs(old.depth_prediction(f,b.start_s,b.stop_s,b.block_center_um)-b.best_shift_um))
        for n in nulls.loc[(nulls.window==w)&nulls.resolved].itertuples(): qabs.append(abs(mref.episode_prediction(f,n.start_s,n.stop_s)))
        if w in quiet_windows:
            vals,_=old.corrected_quiet_values(f,cand.loc[cand.window==w],bounds); incs.extend(vals)
            falsefracs.append(pstage.false_motion(f,cand.loc[cand.window==w],bounds))
    return {"episode_err":float(np.median(errs)) if errs else np.nan,"episode_ratio":float(np.median(ratios)) if ratios else np.nan,
        "block_err":float(np.median(berr)) if berr else np.nan,"quiet_abs":float(np.median(qabs)) if qabs else np.nan,
        "quiet_inc":float(np.sqrt(np.mean(np.square(incs)))) if incs else np.nan,"false_motion_frac":float(np.mean(falsefracs)) if falsefracs else np.nan,
        "accepted_episodes":len(errs),"accepted_blocks":len(berr),"resolved_nulls":len(qabs),"quiet_pairs":len(incs)}

def _field_series(field,times,depth=None):
    """Sample a field as a time series, respecting rigid one-bin fields."""
    times=np.asarray(times,float); depths=np.asarray(field["depth_um"],float)
    if depth is not None:
        z=float(depths[0]) if len(depths)==1 else float(depth)
        return stage1.sample_field(field,times,np.full(len(times),z))
    values=stage1.sample_field(field,*np.meshgrid(times,depths,indexing="ij"))
    finite=np.isfinite(values); count=finite.sum(axis=1)
    return np.divide(np.nansum(values,axis=1),count,out=np.full(len(times),np.nan),where=count>0)

def _paired_prediction(target,per_window,start,stop,depth=None):
    episode_t=np.arange(start+GRID_S/2,stop,GRID_S)
    rest_t=np.arange(start-4+GRID_S/2,start-1,GRID_S)
    if not len(episode_t) or not len(rest_t): return np.nan,np.nan
    te=_field_series(target,episode_t,depth); pe=_field_series(per_window,episode_t,depth)
    tr=_field_series(target,rest_t,depth); pr=_field_series(per_window,rest_t,depth)
    egood=np.isfinite(te)&np.isfinite(pe); rgood=np.isfinite(tr)&np.isfinite(pr)
    if not egood.any() or not rgood.any(): return np.nan,np.nan
    return float(np.median(te[egood])-np.median(tr[rgood])),float(np.median(pe[egood])-np.median(pr[rgood]))

def _quiet_allowed(times,candidates):
    allowed=np.ones(len(times),bool)
    for row in candidates.itertuples():
        allowed&=~((times>=row.start_s-2)&(times<=row.stop_s+2))
    return allowed

def _paired_quiet_increments(target,per_window,candidates,bounds,step_s):
    """Return matched 5 s increments on either frozen or overlapping starts."""
    start,stop=bounds
    if np.isclose(step_s,GRID_S):
        # Y stability check: use the saved per-window field grid itself.  Its
        # timestamps have a small frontend-dependent offset from the nominal
        # window boundary, so a synthetic global grid changes edge support.
        grid=np.asarray(per_window["session_time_s"],float)
        lag=int(round(5.0/float(np.median(np.diff(grid)))))
        starts=grid[:-lag]; ends=grid[lag:]
    else:
        starts=np.arange(start,stop-5+1e-6,step_s); ends=starts+5
    times=np.unique(np.r_[starts,ends])
    tv=_field_series(target,times); pv=_field_series(per_window,times)
    finite=np.isfinite(tv)&np.isfinite(pv); allowed=_quiet_allowed(times,candidates)
    lookup={round(float(t),8):i for i,t in enumerate(times)}
    rows=[]
    for a,b in zip(starts,ends):
        ia=lookup[round(float(a),8)]; ib=lookup[round(float(b),8)]
        if finite[ia] and finite[ib] and allowed[ia] and allowed[ib]:
            rows.append((float(a),float(b),float(tv[ib]-tv[ia]),float(pv[ib]-pv[ia])))
    return rows

def score_matched_pair(target,per_window_provider,windows,cand,measured,nulls,blocks,quiet_windows):
    """Score two fields only on their common evaluable samples (Y.1)."""
    terr,perr,tratio,pratio,tberr,pberr,tqabs,pqabs=[],[],[],[],[],[],[],[]
    tinc,pinc,tover,pover,false_t,false_p=[],[],[],[],[],[]; detail=[]
    for w,bounds in windows.items():
        per=per_window_provider(w)
        for ep in measured.loc[(measured.window==w)&measured.accepted].itertuples():
            tp,pp=_paired_prediction(target,per,ep.start_s,ep.stop_s)
            if np.isfinite(tp) and np.isfinite(pp):
                terr.append(abs(tp-ep.best_shift_um)); perr.append(abs(pp-ep.best_shift_um))
                tratio.append(tp/ep.best_shift_um); pratio.append(pp/ep.best_shift_um)
        if len(blocks):
            for b in blocks.loc[(blocks.window==w)&blocks.accepted].itertuples():
                tp,pp=_paired_prediction(target,per,b.start_s,b.stop_s,b.block_center_um)
                if np.isfinite(tp) and np.isfinite(pp):
                    tberr.append(abs(tp-b.best_shift_um)); pberr.append(abs(pp-b.best_shift_um))
        for n in nulls.loc[(nulls.window==w)&nulls.resolved].itertuples():
            tp,pp=_paired_prediction(target,per,n.start_s,n.stop_s)
            if np.isfinite(tp) and np.isfinite(pp): tqabs.append(abs(tp)); pqabs.append(abs(pp))
        if w in quiet_windows:
            wcand=cand.loc[cand.window==w]
            frozen=_paired_quiet_increments(target,per,wcand,bounds,5.0)
            overlap=_paired_quiet_increments(target,per,wcand,bounds,GRID_S)
            for mode,values in (("frozen_nonoverlap",frozen),("overlapping_0p25",overlap)):
                for a,b,td,pd in values:
                    detail.append({"window":w,"mode":mode,"start_s":a,"stop_s":b,
                        "stitched_increment_um":td,"per_window_increment_um":pd,
                        "stitched_minus_per_window_um":td-pd,
                        "abs_stitched_minus_per_window_um":abs(td-pd)})
            tinc.extend(x[2] for x in frozen); pinc.extend(x[3] for x in frozen)
            tover.extend(x[2] for x in overlap); pover.extend(x[3] for x in overlap)
            times=np.arange(bounds[0]+GRID_S/2,bounds[1],GRID_S)
            tv=_field_series(target,times); pv=_field_series(per,times)
            common=np.isfinite(tv)&np.isfinite(pv); allowed=_quiet_allowed(times,wcand)&common
            if allowed.any():
                tv=tv-np.median(tv[common]); pv=pv-np.median(pv[common])
                false_t.append(float(np.mean(np.abs(tv[allowed])>50)))
                false_p.append(float(np.mean(np.abs(pv[allowed])>50)))
    def result(err,ratio,berr,qabs,inc,over,false):
        return {"episode_err":float(np.median(err)) if err else np.nan,
            "episode_ratio":float(np.median(ratio)) if ratio else np.nan,
            "block_err":float(np.median(berr)) if berr else np.nan,
            "quiet_abs":float(np.median(qabs)) if qabs else np.nan,
            "quiet_inc":float(np.sqrt(np.mean(np.square(inc)))) if inc else np.nan,
            "quiet_inc_overlapping":float(np.sqrt(np.mean(np.square(over)))) if over else np.nan,
            "false_motion_frac":float(np.mean(false)) if false else np.nan,
            "accepted_episodes":len(err),"accepted_blocks":len(berr),"resolved_nulls":len(qabs),
            "quiet_pairs":len(inc),"quiet_pairs_overlapping":len(over)}
    return result(terr,tratio,tberr,tqabs,tinc,tover,false_t),result(perr,pratio,pberr,pqabs,pinc,pover,false_p),detail

def evaluate_full(output):
    import pandas as pd
    stitched=load_field(output/"stitched_full.npz"); shared=shared_field(); choice=selected_choice()
    decision=read_json(STAGE2/"reference_decision_r.json")
    sets=[]
    dc=pd.read_csv(PARENT/"references_m/candidate_episodes.csv"); dm=pd.read_csv(PARENT/"references_m/measured_episodes.csv"); dn=pd.read_csv(PARENT/"references_m/null_pseudo_episodes.csv"); db=pd.read_csv(STAGE2/"stage1_block_measurements.csv")
    sets.append(("development",stage1.WINDOWS,dc,dm,dn,db,{"p50"}))
    # Correction S superseded the original rate-only quiet references before Q
    # began.  Use its shared-recovery candidates, adjudicated nonzero-shift
    # exclusions, and rebuilt zero-shift nulls for every quiet metric.
    sroot=STAGE2/"s_correction"
    qc=pd.read_csv(sroot/"candidate_episodes.csv")
    qx=pd.read_csv(sroot/"additional_nonzero_shift_exclusions.csv")
    qs=pd.read_csv(sroot/"shift_test_segments.csv")
    qexclude=pd.concat([qc[["window","start_s","stop_s"]],qx[["window","start_s","stop_s"]],
        qs[["window","start_s","stop_s"]]],ignore_index=True).drop_duplicates()
    qm=pd.read_csv(sroot/"all_quiet_measurements.csv")
    qn=pd.read_csv(sroot/"null_pseudo_episodes.csv")
    qn=qn.loc[qn.resolved & (qn.best_shift_um==0)].copy()
    qb=pd.read_csv(sroot/"block_measurements.csv")
    qwin=dict(old.HELDOUT_WINDOWS)
    sets.append(("quiet_heldout_s_corrected",qwin,qexclude,qm,qn,qb,set(qwin)))
    ec=pd.read_csv(STAGE2/"references_episode/candidate_episodes.csv"); em=pd.read_csv(STAGE2/"references_episode/measured_episodes.csv"); en=pd.read_csv(STAGE2/"references_episode/null_pseudo_episodes.csv"); eb=pd.read_csv(STAGE2/"references_episode/block_measurements.csv")
    eall=pstage.episode_windows(STAGE2); ewin={w:eall[w] for w in decision["valid_episode_windows"]}
    # S's frozen selection pool contains the surviving episode windows plus
    # the corrected quiet windows, which contribute 14 accepted episodes.
    poolwin={**ewin,**qwin}
    poolcand=pd.concat([ec,qc],ignore_index=True,sort=False)
    poolmeasured=pd.concat([em,qm],ignore_index=True,sort=False)
    poolnulls=pd.concat([en,pd.read_csv(sroot/"null_pseudo_episodes.csv")],ignore_index=True,sort=False)
    poolblocks=pd.concat([eb,qb],ignore_index=True,sort=False)
    sets.append(("episode_pool_s_corrected",poolwin,poolcand,poolmeasured,poolnulls,poolblocks,set()))
    rows=[]; regressions=[]; increment_detail=[]
    for name,windows,cand,measured,nulls,blocks,quiet in sets:
        per_provider=lambda w:pstage.load_field(pstage.field_path(STAGE2,choice["config"],w))
        stitched_score,per_score,detail=score_matched_pair(stitched,per_provider,windows,cand,measured,nulls,blocks,quiet)
        shared_score,_,_=score_matched_pair(shared,per_provider,windows,cand,measured,nulls,blocks,quiet)
        for row in detail: row["set"]=name
        increment_detail.extend(detail)
        scored={"stitched":stitched_score,"full_session_shared_recovery":shared_score,"per_window_selected":per_score}
        for method,result in scored.items(): rows.append({"set":name,"method":method,"config":choice["config"],**result})
        for metric in ("episode_err","block_err","quiet_abs","quiet_inc"):
            a=scored["stitched"][metric]; b=scored["per_window_selected"][metric]
            if np.isfinite(a) and np.isfinite(b) and a-b>3: regressions.append({"set":name,"metric":metric,"stitched":a,"per_window":b,"degradation":a-b})
    pd.DataFrame(rows).to_csv(output/"scores_full_session_q.csv",index=False)
    detail_table=pd.DataFrame(increment_detail)
    detail_table.to_csv(output/"matched_quiet_increment_differences.csv",index=False)
    seam=float(read_json(output/"seams_full.json")["median_abs_seam_discontinuity_um"])
    status="pass" if seam<=10 and not regressions else "fail"
    stage1.atomic_json(output/"full_score_gate.json",{"status":status,"reference_version":"S-corrected",
        "reference_root":str(sroot),"episode_pool_accepted":int(poolmeasured.accepted.sum()),
        "quiet_reference_windows":sorted(qwin),"resolved_zero_shift_quiet_nulls":int(len(qn)),
        "matched_quiet_pairs":{
            "frozen_nonoverlap":int(((detail_table["set"]=="quiet_heldout_s_corrected")&(detail_table["mode"]=="frozen_nonoverlap")).sum()),
            "overlapping_0p25":int(((detail_table["set"]=="quiet_heldout_s_corrected")&(detail_table["mode"]=="overlapping_0p25")).sum())},
        "median_abs_seam_discontinuity_um":seam,"seam_limit_um":10.0,
        "regressions_over_3um":regressions,
        "criterion":"median seam <=10um; stitched may not exceed per-window selected by >3um on any available frozen um metric"})
    if regressions: raise RuntimeError("Q.2 frozen score gate failed")

def full_peak_population(output):
    """Load one unique full-session population without a disk duplicate."""
    mode=read_json(output/"u_full_mode.json")["mode"] if (output/"u_full_mode.json").exists() else "exact_window_fallback"
    if mode=="block_concat":
        arrays={k:[] for k in ("time_s","depth_um","x_um","amplitude")}; hashes={}
        for block in block_specs():
            path=population_path(output,block["id"]); hashes[block["id"]]=stage1.sha256(path)
            pop=load_population(path,block["start_s"])
            for key in arrays: arrays[key].append(pop[key])
        return {k:np.concatenate(v) for k,v in arrays.items()},hashes
    arrays={k:[] for k in ("time_s","depth_um","x_um","amplitude")}; specs=window_specs("full"); hashes={}
    for i,spec in enumerate(specs):
        path=output/f"peak_cache/{spec['id']}/extraction/population.npz"; hashes[spec["id"]]=stage1.sha256(path)
        with np.load(path,allow_pickle=False) as z: zarr={k:np.asarray(z[k]) for k in arrays}
        absolute=np.asarray(zarr["time_s"],float)+spec["start_s"]
        left=spec["start_s"] if i==0 else .5*((specs[i-1]["start_s"]+specs[i-1]["stop_s"])/2+(spec["start_s"]+spec["stop_s"])/2)
        right=spec["stop_s"] if i==len(specs)-1 else .5*((spec["start_s"]+spec["stop_s"])/2+(specs[i+1]["start_s"]+specs[i+1]["stop_s"])/2)
        keep=(absolute>=left)&(absolute<right)
        arrays["time_s"].append(absolute[keep])
        for k in ("depth_um","x_um","amplitude"): arrays[k].append(zarr[k][keep])
    return {k:np.concatenate(v) for k,v in arrays.items()},hashes

def episode_catalogue(output):
    import pandas as pd
    field=load_field(output/"stitched_full.npz"); shared=shared_field()
    population,source_hashes=full_peak_population(output); pk={"time":np.asarray(population["time_s"],float),"depth":np.asarray(population["depth_um"],float),"x":np.asarray(population["x_um"],float)}
    centers=np.arange(GRID_S/2,SESSION_DURATION_S,GRID_S); edges=np.r_[centers-GRID_S/2,SESSION_DURATION_S]
    counts=np.histogram(pk["time"],bins=edges)[0]; low=counts<.5*np.median(counts)
    rigid=np.nanmedian(field["displacement_um"],axis=1); sr=np.nanmedian(shared["displacement_um"],axis=1); sr=np.interp(centers,shared["session_time_s"],sr); sr-=np.nanmedian(sr)
    fr=np.interp(centers,field["session_time_s"],rigid); threshold=float(np.nanpercentile(fr,5)); motion=fr<=threshold; shared_runs=sr<=-40
    active=mref.join_short_gaps(low|motion|shared_runs,round(1/GRID_S)); intervals=mref.intervals(active,centers)
    rows=[]; rng=np.random.default_rng(20260924)
    for eid,(i0,i1,a,b) in enumerate(intervals):
        lo=np.searchsorted(pk["time"],a-4); hi=np.searchsorted(pk["time"],b); sub={k:v[lo:hi] for k,v in pk.items()}; st=sub["time"]
        ep=(st>=a)&(st<b); rest=(st>=a-4)&(st<a-1)
        # Exclude all other candidate activity from the rest map.
        cidx=np.clip(np.floor(st/GRID_S).astype(int),0,len(active)-1); rest &= ~active[cidx]
        val=mref.map_shift(sub,ep,rest,rng); accepted=bool(val["episode_peaks"]>=mref.MIN_PEAKS and val["rest_peaks"]>=mref.MIN_PEAKS and np.isfinite(val["corr_gain"]) and val["corr_gain"]>=mref.MIN_CORR_GAIN)
        pred=mref.episode_prediction(field,a,b); row={"episode_id":eid,"start_s":a,"end_s":b,"duration_s":b-a,
            "candidate_stitched_p5":bool(motion[i0:i1].any()),"candidate_shared_le_minus40":bool(shared_runs[i0:i1].any()),"candidate_low_rate":bool(low[i0:i1].any()),
            "measured_shift_um":val["best_shift_um"],"corr_zero":val["corr_zero"],"corr_best":val["corr_best"],"corr_gain":val["corr_gain"],
            "episode_peaks":val["episode_peaks"],"rest_peaks":val["rest_peaks"],"field_displacement_um":pred,"status":"accepted" if accepted else "unresolved"}
        for bi,(blo,bhi) in enumerate(old.BLOCKS):
            support=(sub["depth"]>=blo-150)&(sub["depth"]<bhi+150); bp={k:v[support] for k,v in sub.items()}; bidx=np.clip(np.floor(bp["time"]/GRID_S).astype(int),0,len(active)-1)
            bv=mref.map_shift(bp,(bp["time"]>=a)&(bp["time"]<b),(bp["time"]>=a-4)&(bp["time"]<a-1)&~active[bidx],rng)
            bok=bv["episode_peaks"]>=mref.MIN_PEAKS and bv["rest_peaks"]>=mref.MIN_PEAKS and np.isfinite(bv["corr_gain"]) and bv["corr_gain"]>=mref.MIN_CORR_GAIN
            row[f"block{bi}_shift_um"]=bv["best_shift_um"] if bok else np.nan
        rows.append(row)
    cat=pd.DataFrame(rows); cat.to_csv(output/"episode_catalogue.csv",index=False)
    cat["session_tenth"]=np.minimum(9,(cat.start_s/(SESSION_DURATION_S/10)).astype(int))+1
    summary=cat.groupby("session_tenth").agg(candidates=("episode_id","size"),accepted=("status",lambda x:int((x=="accepted").sum())),
        unresolved=("status",lambda x:int((x=="unresolved").sum())),median_duration_s=("duration_s","median"),median_measured_shift_um=("measured_shift_um","median"),median_field_displacement_um=("field_displacement_um","median")).reset_index()
    summary.to_csv(output/"episode_catalogue_by_tenth.csv",index=False)
    stage1.atomic_json(output/"catalogue_manifest.json",{"candidate_union":"stitched rigid <= session P5 OR shared-recovery centred rigid <= -40um OR population rate <0.5x median; join gaps <=1s",
        "stitched_p5_um":threshold,"episodes":len(cat),"accepted":int((cat.status=="accepted").sum()),"unresolved":int((cat.status=="unresolved").sum()),"source_window_population_sha256":source_hashes})

def handoff(output):
    import subprocess
    src=load_field(output/"stitched_full.npz"); package=output/"luke0804_imec1_medicine_deployable_motion.npz"
    np.savez_compressed(package,time_s=src["time_s"],depth_um=src["depth_um"],displacement_um=src["displacement_um"],
        sign_convention=np.asarray("corrected = observed - displacement"))
    choice=selected_choice(); rev=subprocess.run(["git","rev-parse","HEAD"],cwd=ROOT,text=True,capture_output=True).stdout.strip()
    artifacts=[package,output/"seams_pilot.json",output/"seams_full.json",output/"scores_full_session_q.csv",output/"episode_catalogue.csv",output/"episode_catalogue_by_tenth.csv"]
    extraction_seconds=sum(read_json(x)["seconds"] for x in output.glob("peak_cache/*/extraction/audit.json"))
    stage1.atomic_json(output/"manifest.json",{"schema":"luke0804-imec1-deployable-motion-v1","configuration":choice["config"],"selection_status":choice["outcome"],
        "windows":window_specs("full"),"seams":read_json(output/"seams_full.json"),"git_commit":rev,"code":str(Path(__file__).resolve()),"code_sha256":stage1.sha256(Path(__file__)),
        "artifacts":{p.name:stage1.sha256(p) for p in artifacts},"medicine_runtime_s":fit_runtime(output),"medicine_budget_s":GPU_BUDGET_S,
        "per_window_frontend_seconds":extraction_seconds,"frontend_time_counted_against_gpu_budget":False,"no_sort":True,"voltage_modified":False})
    (output/"DEPLOYMENT.md").write_text("""# Luke0804 imec1 motion-field deployment\n\nUse `luke0804_imec1_medicine_deployable_motion.npz` as the external `dredge_motion_est`. The sign convention is **corrected = observed - displacement**. Keep AP voltage unwarped.\n\nDARTsort's internal `speed_limit_um_per_s` and median-distance post-filters do not run on an externally supplied field, and the external field is not resampled to `temporal_bin_length_s` (K.3). The supplied grid is already 0.25 s from AP frame zero.\n\nA single motion state per 1 s matching chunk is too coarse for the fast approximately 200 um episodes. For the huklaban5 sort, use matching chunks of 0.25 s or less, or motion sub-intervals within longer chunks.\n\nStitched and per-window 5 s increments can differ slightly during motion-dominated time because triangular blending across overlapping windows changes those increments. The U block-concatenation equivalence check passed: the absolute change in `episode_err` was 2.38 um (limit 3 um).\n\nKnown limitations: the label-free reference is quantised to 10 um; episodes shorter than roughly 0.5 s are often unresolved; depth-block estimates can be unresolved when peak support is sparse; the terminal sliding window is shifted off the 60 s lattice by 33.55 s to cover the recording tail; and this field has not yet been validated by a sorting comparison.\n""")
    (output/"README.md").write_text("# Luke0804 imec1 MEDiCINe stage 3 (Q)\n\nThe deployable field passed the Q.1 seam/per-window pilot gate and the Q.2 frozen-score gate. No spike sort was run and AP voltage was not modified. See `DEPLOYMENT.md`, `manifest.json`, `scores_full_session_q.csv`, and the episode catalogue files.\n")
    stage1.atomic_json(output/"validation_q.json",{"status":"complete","pilot_gate":read_json(output/"pilot_gate.json"),"full_gate":read_json(output/"full_score_gate.json"),"package_sha256":stage1.sha256(package),"no_sort":True,"voltage_modified":False})

def report_t(output):
    cleanup=read_json(output/"cleanup_t.json"); estimate=read_json(output/"extraction_estimate_t.json")
    services={
        "failed_source_pilot":"luke-medicine-q-amp50-d1-pilot-20260924.service: failed ENOSPC/exit 139; evidence retained",
        "failed_source_full":"luke-medicine-q-amp50-d1-full-20260924.service: gate follower failed; no full-session work ran",
        "exact_window_pilot":"luke-medicine-q-t-pilot-20260924.service: running exact independent 120 s stage1.extract_one contexts",
        "exact_window_full":"luke-medicine-q-t-full-20260924.service: waiting for Q.1 pass"}
    stage1.atomic_json(output/"q_services_t.json",{"services":services,"recorded_at":time.time(),"no_sort":True,"voltage_modified":False})
    (output/"T_STATUS.md").write_text(f"""# Correction T status

No sorting or voltage modification was performed.

## Cleanup

- Free before: {cleanup['before_available_bytes']:,} bytes.
- Free after: {cleanup['after_available_bytes']:,} bytes.
- Freed: {cleanup['freed_bytes']:,} bytes.
- No process held either deleted path. Only `stage3_q/runtime_tmp` orphaned files and a superseded whole-source `stage3_q/peak_cache` without a population, completed extraction, field, or fit receipt were removed.

## Exact per-window extraction

Every 120 s / 60 s-step window now runs its own `stage1.extract_one`, matching stages 1/2. `TMPDIR` is inside `stage3_q/runtime_tmp`. Successful extraction validates and seals `population.npz`, then removes the temporary preprocessed recording and detection HDF5.

- Estimated median extraction: {estimate['median_extraction_s_per_window']:.1f} s/window.
- Estimated peak transient disk: {estimate['estimated_peak_transient_bytes']:,} bytes.
- Retained population cache: median {estimate['median_population_cache_bytes']:,} bytes/window.
- Disk guard: at least {MIN_FREE_BYTES:,} bytes before launch and between windows.
- Frontend time is reported separately and does not count against the Q MEDiCINe GPU budget.

## Services

"""+"\n".join(f"- {x}" for x in services.values())+"\n")

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("phase",choices=["prepare","replan-per-window-t","replan-blocks-u","replan-v","apply-w","extract-windows","extract-one-window","validate-block-concat-u","fit-validation-u","extract-full-u","fit-one","sweep","pilot-report","stitch-full","evaluate-full","catalogue","handoff","report-t"])
    ap.add_argument("--output",type=Path,default=DEFAULT_OUTPUT); ap.add_argument("--scope",choices=["pilot","full"]); ap.add_argument("--window"); a=ap.parse_args(); out=a.output.resolve()
    if a.phase=="prepare":prepare(out)
    elif a.phase=="replan-per-window-t":replan_per_window_t(out)
    elif a.phase=="replan-blocks-u":replan_blocks_u(out)
    elif a.phase=="replan-v":replan_v(out)
    elif a.phase=="apply-w":apply_w(out)
    elif a.phase=="extract-windows":extract_windows(out,a.scope)
    elif a.phase=="extract-one-window":extract_one_window(out,a.window)
    elif a.phase=="validate-block-concat-u":validate_block_concat_u(out)
    elif a.phase=="fit-validation-u":fit_validation_u(out,a.window)
    elif a.phase=="extract-full-u":extract_full_u(out)
    elif a.phase=="fit-one":fit_one(out,a.scope,a.window)
    elif a.phase=="sweep":sweep(out,a.scope)
    elif a.phase=="pilot-report":pilot_report(out)
    elif a.phase=="stitch-full":stitch(out,"full")
    elif a.phase=="evaluate-full":evaluate_full(out)
    elif a.phase=="catalogue":episode_catalogue(out)
    elif a.phase=="handoff":handoff(out)
    elif a.phase=="report-t":report_t(out)

if __name__=="__main__":main()
