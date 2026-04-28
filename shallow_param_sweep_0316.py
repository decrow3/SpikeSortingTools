#%%
from pipeline import condition_signal, correct_motion, sort_ks4, save_binary_recording, run_qc, KilosortResults, run_cur
from spikeinterface.sorters import get_default_sorter_params
from spikeinterface.core import load_extractor
from pathlib import Path
import os
import numpy as np
import matplotlib.pyplot as plt
import gc
import spikeinterface.full as si


def _parse_float_list_env(var_name: str, default: list[float]) -> list[float]:
    """Parse comma-separated float list from env var; fallback to default."""
    raw = os.environ.get(var_name, '').strip()
    if not raw:
        return list(default)
    out: list[float] = []
    for part in raw.split(','):
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
    return out


def _run_name_for_claim(ms: float, um: float) -> str:
    def _fmt(x: float) -> str:
        if abs(x - round(x)) < 1e-12:
            s = str(int(round(x)))
        else:
            s = f"{x:g}"
        return s.replace('-', 'm').replace('.', 'p')

    return f"claim_ms{_fmt(ms)}_um{_fmt(um)}"

# =============================================================================
# Configuration
# =============================================================================

#%% Data + pipeline paths
data_dir     = r"/mnt/NPX/Luke/20260316/Luke03162026_V2V1_RH_g0/"
stream_id    = "imec0.ap"

pipeline_dir = Path("/mnt/NPX/Luke/20260316/dredge_pipeline_results_Luke03162026_V2V1_RH_g0_imec0/")
_DEFAULT_SWEEP_DIRNAME = "shallow_sweep_claimmask"
sweep_dir    = pipeline_dir / os.environ.get("SHALLOW_SWEEP_DIRNAME", _DEFAULT_SWEEP_DIRNAME).strip()
sweep_dir.mkdir(parents=True, exist_ok=True)
print(f"Sweep outputs will be written to {sweep_dir}")

# Channel selection settings
MARGIN_UM       = 175    # µm either side of dense zone to include
SURFACE_EXCL_UM = 200    # µm to exclude from surface end (high-y end of probe)
DENSITY_BIN_UM  = 40     # histogram bin width for density search (2× row pitch)
TOP_FRACTION    = 1 / 3  # search within top fraction of probe depth range

# =============================================================================
#%% Step 1: Load motion-corrected recording
# =============================================================================
# Try to reuse the binary from the main dredge pipeline run.
# If it doesn't exist yet, re-run conditioning + motion correction and save.

motion_corrected_path = pipeline_dir / "preprocessed_recording"

if motion_corrected_path.exists():
    print(f"Loading motion-corrected recording from {motion_corrected_path}")
    seg_motion_saved = load_extractor(motion_corrected_path)
else:
    print("Motion-corrected binary not found — running conditioning and motion correction")
    seg_raw = si.read_spikeglx(folder_path=data_dir, load_sync_channel=False, stream_id=stream_id)
    seg_pre_motion_est, seg_pre_sorting = condition_signal(
        seg_raw,
        cache_dir=pipeline_dir / "conditioning",
        noise_thresh=0.3,
        uV_thresh=0.5e3,
        recalc=False,
    )
    seg_motion = correct_motion(
        seg_pre_motion_est,
        rec_for_sorting=seg_pre_sorting,
        cache_dir=pipeline_dir / "motion",
        recalc=False,
        method="dredge",
    )
    del seg_raw, seg_pre_motion_est, seg_pre_sorting
    gc.collect()
    seg_motion_saved = save_binary_recording(seg_motion, motion_corrected_path, recalc=False)
    del seg_motion
    gc.collect()

# =============================================================================
#%% Step 2: Find peak spike density in the top third of the probe
# =============================================================================
# peak_locations were estimated on the pre-motion signal but depth offsets are
# small enough (<< MARGIN_UM) that they don't affect the density search.
# Depth convention: y=0 is probe tip (deepest), y increases toward brain surface.

peak_locs = np.load(pipeline_dir / "motion" / "peak_locations.npy")
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
# Allow overriding directory name to avoid clobbering an in-use cache.
# If SHALLOW_BINARY_SUFFIX is set, use that; otherwise use a timestamp suffix.
_suffix = os.environ.get('SHALLOW_BINARY_SUFFIX', '').strip()
if not _suffix:
    from datetime import datetime
    _suffix = datetime.now().strftime('%Y%m%d%H%M%S')
shallow_binary_path = sweep_dir / f"preprocessed_recording_shallow{('_' + _suffix) if _suffix else ''}"
print(f"Using shallow binary cache directory: {shallow_binary_path}")
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
param_sweeps = [
    {"run_name": "default"},
    # Legacy names used by downstream analysis scripts
    #{"run_name": "claim_tonly",   "cross_peel_claim_ms": 0.25, "cross_peel_claim_um": 0.0},
    {"run_name": "claim_spatial", "cross_peel_claim_ms": 0.25, "cross_peel_claim_um": 75.0},
]

# Claim-mask parameter grid.
# Override with e.g.
#   SHALLOW_CLAIM_MS_LIST="0,0.1,0.25,0.5"  SHALLOW_CLAIM_UM_LIST="0,25,50,75,100"
# or set SHALLOW_INCLUDE_DEFAULT_ONLY=1 to skip claim sweeps.
_include_claim = os.environ.get('SHALLOW_INCLUDE_DEFAULT_ONLY', '').strip() not in ('1', 'true', 'True')
if _include_claim:
    claim_ms_values = _parse_float_list_env('SHALLOW_CLAIM_MS_LIST', default=[0.0, 0.10, 0.25, 0.50])
    claim_um_values = _parse_float_list_env('SHALLOW_CLAIM_UM_LIST', default=[0.0, 25.0, 50.0, 75.0, 100.0])

    # Build ms×um grid, but skip the (0,0) combo since that's identical to default.
    existing_names = {d['run_name'] for d in param_sweeps if 'run_name' in d}
    for ms in claim_ms_values:
        for um in claim_um_values:
            if abs(ms) < 1e-12 and abs(um) < 1e-12:
                continue
            run_name = _run_name_for_claim(ms, um)
            if run_name in existing_names:
                continue
            existing_names.add(run_name)
            param_sweeps.append({
                'run_name': run_name,
                'cross_peel_claim_ms': float(ms),
                'cross_peel_claim_um': float(um),
            })

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
            seg_shallow_saved,
            ks4_sorter,
            ks4_results,
            run_dir / "cur",
            recalc=False,
            split_depth_export=False,
        )

        if isinstance(cur_results, dict):
            for part_name, part_results in cur_results.items():
                if part_results is not None:
                    run_qc(seg_shallow_saved, part_results, run_dir / f"qc_{part_name}", recalc=False)
        else:
            run_qc(seg_shallow_saved, cur_results, run_dir / "qc", recalc=False)

        print(f"Completed: {run_name}")

    except Exception as e:
        print(f"ERROR in run '{run_name}': {e}")
        continue

print("\nAll sweeps finished.")
