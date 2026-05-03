#%%
# curation_single_run_test.py
#
# Runs KS4 once with default params on the shallow channel subset, then tests
# all four curation strategies.  Reuses the existing shallow binary so channel
# selection / motion correction do not need to re-run.
#
# Intended for rapid iteration on curation_postpatch.py.
# tF is loaded via torch throughout curation — keep torch installed in this env.

from pipeline import sort_ks4, run_qc, KilosortResults, save_binary_recording
from pipeline.qc import truncation_qc
from pipeline.curation_postpatch import (
    run_cur,
    run_cur_cosine,
    run_cur_amp_bic,
    run_cur_no_merge,
)
from spikeinterface.core import load_extractor
from spikeinterface.sorters import get_default_sorter_params
from pathlib import Path
import numpy as np
import os

# =============================================================================
# Configuration — match the paths used in shallow_param_sweep.py
# =============================================================================

pipeline_dir = Path("/mnt/NPX/Luke/20260316/dredge_pipeline_results_Luke03162026_V2V1_RH_g0_imec1/")
_DEFAULT_SWEEP_DIRNAME = "shallow_sweep_claimmask"
sweep_dir    = pipeline_dir / os.environ.get("SHALLOW_SWEEP_DIRNAME", _DEFAULT_SWEEP_DIRNAME).strip()

# Single run — all four curation strategies will be applied to this one KS4 result
RUN_NAME     = os.environ.get("CUR_TEST_RUN", "default").strip()
RECALC_KS4   = os.environ.get("CUR_RECALC_KS4",  "0").strip() == "1"
RECALC_CUR   = os.environ.get("CUR_RECALC_CUR",  "0").strip() == "1"
RECALC_QC    = os.environ.get("CUR_RECALC_QC",   "0").strip() == "1"
FS           = 30_000

run_dir = sweep_dir / f"run_{RUN_NAME}"

print(f"Run dir: {run_dir}")

# =============================================================================
# Step 1: Load the saved shallow recording (must already exist)
# =============================================================================

shallow_binary_path = sweep_dir / "preprocessed_recording_shallow"
assert shallow_binary_path.exists(), (
    f"Shallow binary not found at {shallow_binary_path}. "
    "Run shallow_param_sweep.py first to create it."
)
seg_shallow = load_extractor(shallow_binary_path)
print(f"Loaded shallow recording: {seg_shallow.get_num_samples()} samples, "
      f"{seg_shallow.get_num_channels()} channels")

# =============================================================================
# Step 2: Run KS4 with default params (cached)
# =============================================================================

base_params = get_default_sorter_params("kilosort4")
base_params.update(
    do_correction=False,
    save_extra_vars=True,
    Th_universal=12,
    Th_learned=9,
    max_peels=100,
    duplicate_spike_ms=0.25,
    ccg_threshold=0.75,
    nearest_chans=20,
    nearest_templates=200,
    max_channel_distance=64,
    cross_peel_claim_ms=0.0,
    cross_peel_claim_um=0.0,
    clear_cache=True,
)

print(f"\n{'='*60}")
print(f"Sorting: {RUN_NAME}")
print(f"{'='*60}")

ks4_results, ks4_sorter = sort_ks4(
    seg_shallow,
    run_dir / "kilosort4",
    sorter_params=base_params,
    recalc=RECALC_KS4,
)
print(f"KS4 done: {len(np.unique(ks4_results.spike_clusters))} clusters, "
      f"{len(ks4_results.spike_times)} spikes")

# Pre-curation QC (uses full_st.npy, only available on raw KS4 output)
run_qc(seg_shallow, ks4_results, run_dir / "qc_pre", recalc=RECALC_QC)

# =============================================================================
# Step 3: Run all four curation strategies
# =============================================================================

STRATEGIES = {
    "no_merge": (run_cur_no_merge, {}),
    "posthoc":  (run_cur,          {"posthoc_score_thresh": 3, "posthoc_ccg_thresh": 0.5,
                                     "posthoc_min_spikes_seed": 500, "posthoc_min_spikes_pair": 100}),
    "cosine":   (run_cur_cosine,   {"cosine_thresh": 0.90, "ccg_thresh": 0.5,
                                     "min_spikes_seed": 500, "min_spikes_pair": 100}),
    "amp_bic":  (run_cur_amp_bic,  {"bic_margin": 0.0, "ccg_thresh": 0.5,
                                     "min_spikes_seed": 100, "min_spikes_pair": 100}),
}

curation_results = {}

for strat_name, (fn, kwargs) in STRATEGIES.items():
    print(f"\n--- Curation: {strat_name} ---")
    try:
        cur_res = fn(
            ks4_sorter,
            ks4_results,
            run_dir / "cur",
            recalc=RECALC_CUR,
            fs=FS,
            **kwargs,
        )
        curation_results[strat_name] = cur_res
        n_units = len(np.unique(cur_res.spike_clusters))
        print(f"  {strat_name}: {n_units} units")
    except Exception as e:
        import traceback
        print(f"  ERROR in {strat_name}: {e}")
        traceback.print_exc()
        continue

# =============================================================================
# Step 4: Run QC on each curation output
# =============================================================================
# post-curation outputs have amplitudes.npy (from save_sorting) but not
# full_st.npy, so we call truncation_qc directly instead of run_qc.

print("\n--- QC on curation outputs ---")
qc_results = {}
for strat_name, cur_res in curation_results.items():
    qc_dir = run_dir / f"qc_{strat_name}" / "amp_truncation"
    qc_dir.mkdir(parents=True, exist_ok=True)
    try:
        trunc, pres = truncation_qc(
            cur_res.spike_times,
            cur_res.spike_clusters,
            cur_res.spike_amplitudes,
            cache_dir=qc_dir,
            recalc=RECALC_QC,
        )
        qc_results[strat_name] = (trunc, pres)
        print(f"  [{strat_name}] QC done")
    except Exception as e:
        print(f"  [{strat_name}] QC failed: {e}")
        qc_results[strat_name] = None

# =============================================================================
# Step 5: Quick comparison printout
# =============================================================================

print("\n--- Results summary ---")
print(f"{'Strategy':<12}  {'N units':>8}  {'N spikes':>10}  {'Median amp':>12}")
for strat_name, cur_res in curation_results.items():
    n_units  = len(np.unique(cur_res.spike_clusters))
    n_spikes = len(cur_res.spike_times)
    med_amp  = float(np.median(cur_res.spike_amplitudes))
    print(f"  {strat_name:<12}  {n_units:>8d}  {n_spikes:>10d}  {med_amp:>12.1f}")

print(f"\nOutputs in: {run_dir}")
print("Done.")
