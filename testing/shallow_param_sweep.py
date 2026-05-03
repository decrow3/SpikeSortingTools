#%%
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]  # .../SpikeSortingTools/SpikeSortingTools
sys.path.insert(0, str(REPO_ROOT))
from pipeline import condition_signal, correct_motion, sort_ks4, save_binary_recording, run_qc, KilosortResults
from pipeline.qc import truncation_qc
from pipeline.curation_postpatch import run_cur
from spikeinterface.sorters import get_default_sorter_params
from spikeinterface.core import load_extractor
import os
import numpy as np
import matplotlib.pyplot as plt
import gc
import spikeinterface.full as si

# =============================================================================
# Configuration
# =============================================================================

#%% Data + pipeline paths
data_dir     = r"/mnt/NPX/Luke/20260316/Luke03162026_V2V1_RH_g0/"
stream_id    = "imec1.ap"

pipeline_dir = Path("/media/huklaban5/Data/Testing/dredge_pipeline_results_Luke03162026_V2V1_RH_g0_imec1/")
patched_dir = Path("/media/huklaban5/Data/Testing/patched_pipeline_results_Luke03162026_V2V1_RH_g0_imec1/")

# Read motion/peaks from the dredge run; write all new outputs elsewhere.
SOURCE_DIR = pipeline_dir  # <-- switch this to patched_dir if needed
OUTPUT_DIR = patched_dir
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

_DEFAULT_SWEEP_DIRNAME = "shallow_sweep_testing_1000um"
# sweep_dir    = pipeline_dir / os.environ.get("SHALLOW_SWEEP_DIRNAME", _DEFAULT_SWEEP_DIRNAME).strip()
sweep_dir = OUTPUT_DIR / os.environ.get("SHALLOW_SWEEP_DIRNAME", _DEFAULT_SWEEP_DIRNAME).strip()
sweep_dir.mkdir(parents=True, exist_ok=True)
print(f"Sweep outputs will be written to {sweep_dir}")

# Channel selection settings
MARGIN_UM       = 500#175    # µm either side of dense zone to include
SURFACE_EXCL_UM = 200    # µm to exclude from surface end (high-y end of probe)
DENSITY_BIN_UM  = 40     # histogram bin width for density search (2× row pitch)
TOP_FRACTION    = 1 / 3  # search within top fraction of probe depth range

# =============================================================================
#%% Step 1: Load motion-corrected recording
# =============================================================================
# Try to reuse the binary from the main dredge pipeline run.
# If it doesn't exist yet, re-run conditioning + motion correction and save.
motion_corrected_source_path = SOURCE_DIR / "preprocessed_recording"
# motion_corrected_source_path = pipeline_dir / "preprocessed_recording"

if motion_corrected_source_path.exists():
    print(f"Loading motion-corrected recording from {motion_corrected_source_path}")
    seg_motion_saved = load_extractor(motion_corrected_source_path)
else:
    motion_corrected_output_path = OUTPUT_DIR / "preprocessed_recording"
    print(
        "Motion-corrected binary not found in SOURCE_DIR — "
        f"running conditioning + motion correction and saving to {motion_corrected_output_path}"
    )
    seg_raw = si.read_spikeglx(folder_path=data_dir, load_sync_channel=False, stream_id=stream_id)
    seg_pre_motion_est, seg_pre_sorting = condition_signal(
        seg_raw,
        cache_dir=OUTPUT_DIR / "conditioning",
        noise_thresh=0.3,
        uV_thresh=0.5e3,
        recalc=False,
    )
    seg_motion = correct_motion(
        seg_pre_motion_est,
        rec_for_sorting=seg_pre_sorting,
        cache_dir=OUTPUT_DIR / "motion",
        recalc=False,
        method="dredge",
    )
    del seg_raw, seg_pre_motion_est, seg_pre_sorting
    gc.collect()
    seg_motion_saved = save_binary_recording(seg_motion, motion_corrected_output_path, recalc=False)
    del seg_motion
    gc.collect()

# =============================================================================
#%% Step 2: Find peak spike density in the top third of the probe
# =============================================================================
# peak_locations were estimated on the pre-motion signal but depth offsets are
# small enough (<< MARGIN_UM) that they don't affect the density search.
# Depth convention: y=0 is probe tip (deepest), y increases toward brain surface.
peak_locs_path = SOURCE_DIR / "motion" / "peak_locations.npy"
# peak_locs_path = pipeline_dir / "motion" / "peak_locations.npy"
if not peak_locs_path.exists():
    alt_peak_locs_path = OUTPUT_DIR / "motion" / "peak_locations.npy"
    if alt_peak_locs_path.exists():
        peak_locs_path = alt_peak_locs_path
    else:
        raise FileNotFoundError(
            "Missing peak locations file. Looked for:\n"
            f"- {SOURCE_DIR / 'motion' / 'peak_locations.npy'}\n"
            f"- {OUTPUT_DIR / 'motion' / 'peak_locations.npy'}"
        )

peak_locs = np.load(peak_locs_path)
peak_y    = peak_locs["y"]

probe      = seg_motion_saved.get_probe()
ch_depths  = probe.contact_positions[:, 1]
depth_min  = float(ch_depths.min())
depth_max  = float(ch_depths.max())
depth_range = depth_max - depth_min

# Top third of probe, stepping back SURFACE_EXCL_UM from the surface edge
search_lo = depth_min + depth_range * (1 - TOP_FRACTION)
search_hi = depth_max - SURFACE_EXCL_UM

print(f"Probe depth range: {depth_min:.0f}–{depth_max:.0f} µm")
print(f"Density search region: {search_lo:.0f}–{search_hi:.0f} µm")

search_peaks = peak_y[(peak_y >= search_lo) & (peak_y <= search_hi)]
if len(search_peaks) == 0:
    raise RuntimeError(
        f"No peaks found in {search_lo:.0f}–{search_hi:.0f} µm. "
        "Adjust TOP_FRACTION or SURFACE_EXCL_UM."
    )

bins  = np.arange(search_lo, search_hi + DENSITY_BIN_UM, DENSITY_BIN_UM)
hist, edges = np.histogram(search_peaks, bins=bins)
dense_depth  = float(edges[np.argmax(hist)]) + DENSITY_BIN_UM / 2
print(f"Peak spike density at {dense_depth:.0f} µm — selecting ±{MARGIN_UM} µm window")

# --- Diagnostic plot: peak density vs depth, marked selection window ----------
fig, ax = plt.subplots(figsize=(4, 6))
ax.barh(edges[:-1], hist, height=DENSITY_BIN_UM * 0.9, align='edge', color='steelblue', alpha=0.7)
ax.axhline(dense_depth, color='red', lw=1.5, label=f'dense centre {dense_depth:.0f} µm')
ax.axhspan(dense_depth - MARGIN_UM, dense_depth + MARGIN_UM, color='red', alpha=0.1, label=f'±{MARGIN_UM} µm crop')
ax.set_xlabel("Peak count")
ax.set_ylabel("Depth (µm, 0=tip)")
ax.set_title("Top-third spike density")
ax.legend(fontsize=8)
plt.tight_layout()
fig.savefig(sweep_dir / "density_channel_selection.png", dpi=150)
plt.close(fig)
print(f"Saved density plot to {sweep_dir / 'density_channel_selection.png'}")

# =============================================================================
#%% Step 3: Slice recording to channel subset
# =============================================================================

crop_lo = dense_depth - MARGIN_UM
crop_hi = dense_depth + MARGIN_UM

ch_ids    = seg_motion_saved.get_channel_ids()
ch_mask   = (ch_depths >= crop_lo) & (ch_depths <= crop_hi)
ch_subset = ch_ids[ch_mask]
print(f"Selected {int(ch_mask.sum())} channels in {crop_lo:.0f}–{crop_hi:.0f} µm")

seg_shallow = seg_motion_saved.channel_slice(channel_ids=ch_subset)

np.savez(
    sweep_dir / "channel_subset_info.npz",
    channel_ids=ch_subset,
    channel_depths=ch_depths[ch_mask],
    dense_depth=dense_depth,
    crop_lo=crop_lo,
    crop_hi=crop_hi,
    search_lo=search_lo,
    search_hi=search_hi,
)

# Save once; every sweep run reuses this binary
shallow_binary_path = sweep_dir / "preprocessed_recording_shallow"
seg_shallow_saved   = save_binary_recording(seg_shallow, shallow_binary_path, recalc=False)
del seg_shallow
gc.collect()

# =============================================================================
#%% Step 4: Parameter sweep
# =============================================================================
# Base Kilosort4 parameters — same as the main dredge pipeline
# Claim-mask runs require launching this script from the spikeinterface-claimmask
# environment so SpikeInterface imports the patched editable Kilosort 4.0.27 package.
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

# Each entry: "run_name" labels the subfolder; remaining keys override base_params.
# Th_universal and Th_learned are swept independently so their effects are separable.
# ccg_threshold controls how aggressively similar units are merged — lower = more merging
# = fewer but more certain units (directly targets the oversplitting concern).
# param_sweeps = [
#     {"run_name": "default"},
#     # Universal threshold (first-pass detection) — Th_learned held at default 9
#     {"run_name": "Thu_lo",  "Th_universal": 9},
#     {"run_name": "Thu_hi",  "Th_universal": 15},
#     # Learned threshold (second-pass template matching) — Th_universal held at default 12
#     {"run_name": "Thl_lo",  "Th_learned": 6},
#     {"run_name": "Thl_hi",  "Th_learned": 12},
#     # Peeling sensitivity test — if duplicate fitting is causal, fewer residual
#     # passes should progressively reduce daughter units created by later peels.
#     {"run_name": "peel1",   "max_peels": 1},
#     {"run_name": "peel2",   "max_peels": 2},
#     {"run_name": "peel3",   "max_peels": 3},
#     # Cross-peel claim-mask tests — evaluate whether duplicate suppression can
#     # recover peel1-like behavior without disabling later peels entirely.
#     {"run_name": "claim_tonly",   "cross_peel_claim_ms": 0.25, "cross_peel_claim_um": 0.0},
#     {"run_name": "claim_spatial", "cross_peel_claim_ms": 0.25, "cross_peel_claim_um": 75.0},
#     # CCG merge threshold — lower = more aggressive merging of similar units
#     {"run_name": "ccg_lo",  "ccg_threshold": 0.40},
#     {"run_name": "ccg_mid", "ccg_threshold": 0.60},
#     {"run_name": "ccg_hi",  "ccg_threshold": 0.90},
# ]

#Testing with just the default run to make sure everything works before running the full sweep
param_sweeps = [
    {"run_name": "default"},
]

#%%
for sweep_config in param_sweeps:
    run_name  = sweep_config["run_name"]
    overrides = {k: v for k, v in sweep_config.items() if k != "run_name"}
    run_dir   = sweep_dir / f"run_{run_name}"

    print(f"\n{'='*60}")
    print(f"Sweep: {run_name}  overrides={overrides}")
    print(f"{'='*60}")

    sorter_params = dict(base_params)
    sorter_params.update(overrides)

    try:
        ks4_results, ks4_sorter = sort_ks4(
            seg_shallow_saved,
            run_dir / "kilosort4",
            sorter_params=sorter_params,
            recalc=False,
        )

        run_qc(seg_shallow_saved, ks4_results, run_dir / "qc_pre", recalc=False)

        cur_results = run_cur(
            ks4_sorter,
            ks4_results,
            run_dir / "cur",
            recalc=False,
            split_depth_export=False,
        )

        def _cur_qc(res, qc_dir):
            qc_dir = qc_dir / 'amp_truncation'
            qc_dir.mkdir(parents=True, exist_ok=True)
            truncation_qc(
                res.spike_times, res.spike_clusters, res.spike_amplitudes,
                cache_dir=qc_dir, recalc=False,
            )

        if isinstance(cur_results, dict):
            for part_name, part_results in cur_results.items():
                if part_results is not None:
                    _cur_qc(part_results, run_dir / f"qc_{part_name}")
        else:
            _cur_qc(cur_results, run_dir / "qc")

        print(f"Completed: {run_name}")

    except Exception as e:
        print(f"ERROR in run '{run_name}': {e}")
        continue

print("\nAll sweeps finished.")
