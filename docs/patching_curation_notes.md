# Patching & Post-Curation Pipeline Notes

## Directory Structure

```
/mnt/NPX/<subj>/<sess>/<sess_name>_g0/                    ← raw SpikeGLX data (server)
/mnt/NPX/<subj>/<sess>/dredge_pipeline_results_<sess>_<stream>/
    conditioning/          ← bad channel detection, CMR, bandpass cache
    motion/                ← dredge motion correction cache
    preprocessed_recording/ ← motion-corrected binary (input to KS4)

/media/huklaban5/Data/Patched/patched_pipeline_results_<sess>_<stream>/
    kilosort4/             ← KS4 sorter outputs
        sorter_output/     ← phy-compatible files (spike_times.npy, templates.npy, etc.)
    cur/
        cur_output/        ← curation results (ops.npy, spike_clusters.npy, curation_moves.csv)
        cur_comparison/    ← diagnostic CSVs / figures (sweep mode only)
    qc/
        waveforms/         ← waveforms.npz
        refractory/        ← refractory_qc.npz
        amp_truncation/    ← truncation_qc.npz, present_qc.npz
        *.mat              ← MATLAB export of the above
```

Old server patched dir (`/mnt/NPX/.../patched_pipeline_results_...`) may also contain a
`kilosort4/` folder from earlier runs — the scripts fall back to this if the harddrive KS4
dir does not yet exist.

---

## Step-by-Step Pipeline Logic

### 1. Motion correction (NEVER rerun)

- Results are loaded from `dredge_dir` on the server.
- If `dredge_dir` does not exist the script **raises `FileNotFoundError`** immediately.
- To generate `dredge_dir` for a new session, run the pre-patch dredge pipeline first
  (`SpikeGLX_ext_ref_dredge.py` or equivalent).

### 2. KS4 sorting (run once, cache on harddrive)

Priority order for `ks4_dir`:

1. `pipeline_dir / 'kilosort4'` (harddrive) — if present, load and skip sort
2. `dredge_dir.parent / 'patched_pipeline_results_...' / 'kilosort4'` (server old run) — if present, load
3. Neither found → run KS4 fresh, saving to `pipeline_dir / 'kilosort4'` (harddrive)

`sort_ks4(..., recalc=False)` respects this: if the target dir already contains a completed
sort it loads rather than re-sorting.

### 3. Curation (always rerun with `recalc=True`)

- `run_cur_final(ks4_sorter, ks4_results, pipeline_dir / 'cur', recalc=True, ks4_out_path=ks4_dir / 'sorter_output')`
- `ks4_out_path` decouples the KS4 source dir from the curation cache dir — needed when KS4
  lives on the server but curation writes to the harddrive.
- Outputs land in `pipeline_dir / 'cur' / 'cur_output'`.
- `recalc=True` ensures stale curation from before the cosine-only strategy update is never used.

### 4. QC (always rerun with `recalc=True`)

- `run_qc(seg_saved, cur_results, pipeline_dir / 'qc', recalc=True)`
- `recalc=True` for the same reason as curation.

---

## Curation Strategy: Cosine-Only

### Why cosine only?

Three merge strategies were evaluated on a 629-unit full-probe dataset:

| Strategy | Merges proposed | Merges accepted | Notes |
|---|---|---|---|
| `cosine` | 16 groups | 16 | depth <7 µm, cosine 0.90–0.99, post-ISI <5% |
| `amp_bic` | large groups spanning up to 575 µm | 0 | all co-active, negative handoff score |
| `posthoc` | groups spanning up to 2241 µm | 0 | same pathology at larger scale |

**amp_bic false positive pattern:** amplitude similarity across units with cosine ≈ 0 and
depth up to 575 µm. `time_overlap_frac` ≈ 0.80–1.0 with negative handoff scores — these
are amplitude coincidences, not the same neuron at different times.

**posthoc false positive pattern:** transitivity collapses spatially dispersed units into
giant groups. Cross-strategy gate (posthoc accepted only if cosine or amp_bic agree on ≥1
pair) collapses posthoc to 0 merges at full-probe scale.

**Temporal handoff diagnostics** (`pipeline/curation_temporal_diag.py`) confirmed:
- Of 1015 candidate pairs examined, only 1 passed `looks_temporal_split` criteria.
- That pair (units 54, 55; cosine=0.956, depth=1.86 µm, handoff=0.628) was already caught
  by the cosine strategy.

### Cosine merge acceptance criteria

Configured in `curation_postpatch.py` → `run_cur_cosine`:

| Parameter | Default | Meaning |
|---|---|---|
| `cosine_threshold` | 0.90 | minimum template cosine similarity |
| `max_depth_um` | 200 µm | depth gate on cosine candidate pairs |
| `max_isi_viol_frac` | 0.05 | post-merge ISI violation rate |
| `min_strategies_agree` | 0 | cross-strategy agreement (0 = disabled) |

`min_strategies_agree=0` was set after confirming that cosine and amp_bic propose different
pairs — requiring agreement was blocking all cosine merges even for well-supported pairs.

---

## KS4 Parameters (as of 2026-04)

```python
sorter_params['do_correction']        = False   # drift handled by dredge
sorter_params['save_extra_vars']      = True    # required for truncation QC
sorter_params['Th_universal']         = 9
sorter_params['Th_learned']           = 8
sorter_params['duplicate_spike_ms']   = 0.25
sorter_params['ccg_threshold']        = 0.75   # increased from 0.25 for long recordings
sorter_params['nearest_chans']        = 20     # up from 10
sorter_params['nearest_templates']    = 200    # up from 100
sorter_params['max_channel_distance'] = 64     # up from 32
sorter_params['clear_cache']          = True   # prevents CUDA OOM on large files
sorter_params['cross_peel_claim_ms']  = 0.25
sorter_params['cross_peel_claim_um']  = 75.0
```

**Reference type differences:**

| | ext_ref (2026+) | tip_ref (2024) |
|---|---|---|
| `uV_thresh` | 500 µV | 1200 µV |
| `stream_id` | `imec1.ap` (default) | `imec0.ap` (default) |

`stream_id` should be verified per session — imec0 is typically the first-inserted probe.

---

## How to Rerun from Scratch

If you need to fully redo a session (e.g., KS4 was corrupted):

1. Delete `pipeline_dir / 'kilosort4'` on harddrive.
2. Delete the server old patched dir's `kilosort4/` if present.
3. Run the script — it will detect no KS4 dir and sort fresh.
4. Curation and QC always rerun regardless.

If you need to redo motion correction (unusual):

1. Delete `dredge_dir` on server.
2. Re-run the pre-patch dredge pipeline to regenerate it.
3. Then run the patching script.

---

## Files Produced Per Session

After a successful run:

```
cur/cur_output/
    spike_clusters.npy    ← post-curation cluster assignments
    ops.npy               ← KS4 ops struct (channel positions etc.)
    curation_moves.csv    ← per-merge evidence table
    ops.mat               ← MATLAB export of channel positions

qc/
    waveforms/waveforms.npz
    refractory/refractory_qc.npz
    amp_truncation/truncation_qc.npz
    amp_truncation/present_qc.npz
    waveforms_data.mat
    refractory_data.mat
    truncation_data.mat
    presence_data.mat
```
