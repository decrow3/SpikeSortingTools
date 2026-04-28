# SpikeSortingTools

Spike sorting pipeline for the Huk Lab forked from the version for the Yates Lab authored by Ryan Ressmeyer. Wraps [Kilosort4](https://github.com/MouseLand/Kilosort) and [SpikeInterface](https://spikeinterface.readthedocs.io/) with custom preprocessing, motion correction, curation, and QC steps tuned for Neuropixels recordings acquired with SpikeGLX.

Currently maintained by Declan Rowley
---

## Installation

### 1. Create the Conda environment

```bash
conda env create -f environment.yml
conda activate spikeinterface
```

This installs Python 3.12.3 with all pinned dependencies, including:
- `kilosort==4.0.27`
- `spikeinterface==0.102.1`
- `torch==2.6.0` with CUDA 11/12 support
- `medicine-neuro==1.5` (MEDiCINe motion correction)
- `probeinterface==0.2.25`

A CUDA-capable GPU is required to run Kilosort4.

### 2. (Optional) Apply the cross-peel claim mask patch

The patch adds two new Kilosort4 parameters that suppress redundant spike candidates during the matching-pursuit peeling loop — candidates that fall too close in time and/or space to an already-accepted spike. This can reduce false positives, particularly at higher firing rates or with dense probes.

```bash
python patch_kilosort_claimmask.py
```

**Options:**

| Flag | Effect |
|------|--------|
| *(none)* | Apply the patch in-place |
| `--dry-run` | Show a unified diff of all changes without modifying any files |
| `--reverse` | Restore original Kilosort files from `.bak` backups |

The script auto-locates the installed `kilosort` package, backs up `parameters.py` and `template_matching.py` as `.bak` files, then inserts the patch. It is idempotent — re-running it on an already-patched install is safe.

**New parameters exposed by the patch:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `cross_peel_claim_ms` | `0.0` | Suppress candidates within this many ms of an accepted spike. `0` disables the rule. |
| `cross_peel_claim_um` | `0.0` | Spatial radius paired with `cross_peel_claim_ms`. Candidates are suppressed only if they are also within this many µm. `0` applies the time-only rule. |

Use these parameters in the `_patching` pipeline scripts (e.g. `SpikeGLX_ext_ref_2026_patching.py`):

```python
sorter_params['cross_peel_claim_ms'] = 0.25
sorter_params['cross_peel_claim_um'] = 75.0
```

---

## Running a SpikeGLX session — `SpikeGLX_ext_ref_2025.py`

This is the primary end-to-end pipeline script for external-reference Neuropixels recordings. Open it as a script or run it cell-by-cell in an IDE (VS Code, Spyder). The `#%%` markers define logical cells.

### Step 0 — Set your data path

```python
data_dir = r"/mnt/NPX/Luke/20260224/Luke02242026_V1V2_RH_g0/"
stream_id = "imec0.ap"   # imec0 = first inserted probe, imec1 = second probe
```

`stream_id` follows SpikeGLX naming. For a single-probe session use `imec0.ap`. The recording is loaded with SpikeInterface's `read_spikeglx`.

All pipeline outputs are saved to a directory derived from the data path:
```
/mnt/NPX/Luke/20260224/dredge_pipeline_results_<session>_<stream>/
```

---

### Step 1 — Signal conditioning (`condition_signal`)

```python
seg_pre_motion_est, seg_pre_sorting = condition_signal(
    seg,
    cache_dir=pipeline_dir / 'conditioning',
    noise_thresh=0.3,   # SpikeGLX external ref
    uV_thresh=500,      # saturation threshold in µV (use 1200 for tip-ref)
    recalc=False,
)
```

This step performs the following operations in order:

1. **Phase shift correction** — if the probe reports per-channel `inter_sample_shift` values (common on some Neuropixels versions), a fractional-sample correction is applied to align all channels to the same time base before any further processing.

2. **Saturation blanking** — samples exceeding `uV_thresh` (in either polarity) are replaced with zeros. External reference recordings saturate around 500 µV; tip-reference recordings have higher dynamic range and typically use 1200 µV. Blanking before filtering prevents ringing artefacts from spreading saturation events across the band.

3. **Bad channel detection** — per-channel metrics are computed over 50 randomly sampled 2-second batches:
   - *Similarity to median*: correlation of each channel to the spatial median across the probe. Dead or disconnected channels have anomalously low values (threshold: `similarity < -0.5`).
   - *High-frequency noise power*: PSD energy above 80% of Nyquist. Noisy channels have anomalously high values (threshold controlled by `noise_thresh`; 0.3 is appropriate for SpikeGLX external ref). Bad channels are **interpolated** from neighbours rather than dropped, so the channel count remains constant.

4. **Dual-branch filtering and referencing** — two output branches are produced from the same interpolated signal:
   - **Sorting branch** (300–6000 Hz, 12th-order Butterworth, zero-phase): used as input to Kilosort4. The wider band preserves fast spike transients.
   - **Motion estimation branch** (300–3000 Hz, same filter order): used for DREDGE motion estimation. The narrower band reduces high-frequency noise that can bias peak localisation.
   
   Both branches apply **local common median referencing** (radius 40–140 µm), which cancels correlated noise between nearby channels without attenuating genuine single-unit signals that are spatially compact.

Results are cached in `conditioning/channel_metrics.npy`. Set `recalc=True` to recompute.

---

### Step 2 — Motion correction (`correct_motion`)

```python
seg_motion = correct_motion(
    seg_pre_motion_est,
    rec_for_sorting=seg_pre_sorting,
    cache_dir=pipeline_dir / 'motion',
    recalc=False,
    method='dredge',
)
plot_motion_output(seg_motion, cache_dir=pipeline_dir / 'motion')
```

Motion correction estimates and compensates for probe drift along the depth axis.

**Why motion correction matters:** Slow drift (tens of µm over a session) causes the same neuron to appear at different depths in different segments of the recording. Without correction, Kilosort either splits one unit into multiple clusters or merges neighbouring units — both outcomes degrade yield and contamination metrics.

**Why external DREDGE rather than Kilosort's internal correction:** Kilosort4's built-in drift correction (`do_correction=True`) estimates motion from its own detected spikes, which means it can only correct drift after template matching has already been biased by it. Running DREDGE first produces a corrected binary that Kilosort4 sees as nearly stationary (`do_correction=False`).

**Supported methods** (passed via `method=`):

| Method | Notes |
|--------|-------|
| `'dredge'` | Default. Decentralised robust estimation. Generally best for chronic recordings. |
| `'decentralized'` | Varol 2021 algorithm, good alternative. |
| `'medicine'` | MEDiCINe library. |
| `'kilosort'` | Mimics Kilosort-style iterative template matching. |

The pipeline uses DREDGE peak detection (locally exclusive, 50 µm radius, 5σ threshold) with monopolar triangulation for spike localisation. Results are cached; `recalc=False` reuses a previous run.

`plot_motion_output` generates 6 diagnostic plots comparing the estimated drift traces and the effect of correction.

---

### Step 3 — Kilosort4 sorting (`sort_ks4`)

```python
sorter_params['do_correction']        = False   # motion handled by DREDGE above
sorter_params['save_extra_vars']      = True    # required for truncation QC
sorter_params['Th_universal']         = 12
sorter_params['Th_learned']           = 9
sorter_params['duplicate_spike_ms']   = 0.25
sorter_params['ccg_threshold']        = 0.75
sorter_params['nearest_chans']        = 20
sorter_params['nearest_templates']    = 200
sorter_params['max_channel_distance'] = 64
sorter_params['clear_cache']          = True    # prevents CUDA OOM on large files
```

**Parameter rationale:**

- `Th_universal = 12`, `Th_learned = 9` — slightly higher than Kilosort defaults to reduce noise units on longer recordings where template drift can lower effective SNR at the end of the session.
- `duplicate_spike_ms = 0.25` — spikes from the same template within 0.25 ms are removed as hardware/software duplicates. Note: CCG-based metrics should not be interpreted below ~1 ms regardless.
- `ccg_threshold = 0.75` — raised from the default 0.25 to handle long recordings where two distinct units may share a few coincident spikes (due to temporal drift in waveform shape), which would otherwise trigger a spurious merge.
- `nearest_chans = 20` / `nearest_templates = 200` / `max_channel_distance = 64` — increased from defaults to improve template matching in high-density regions of the probe and to allow detection of spatially extended units (e.g. wide-field neurons near the probe shank).
- `save_extra_vars = True` — saves per-spike template amplitudes needed for the downstream truncation QC step.

The preprocessed recording is cached to disk as a binary file before sorting (`save_binary_recording`). Kilosort4 reads from this file directly. Results are cached in `kilosort4/`; re-running with `recalc=False` skips the sort if outputs already exist.

---

### Step 4 — Curation (`run_cur`)

```python
cur_results = run_cur(
    seg_saved, ks4_sorter, ks4_results,
    pipeline_dir / 'cur',
    recalc=True,
    split_depth_export=False,
    depth_overlap_um=75.0,
    depth_split_um=None,
)
```

Curation removes artefactual units and merges over-split clusters:

1. **Duplicate spike removal** — spikes from different units detected within 1 ms and 150 µm depth are deduplicated. This handles cases where Kilosort assigns the same physical spike to two templates.
2. **Unit merge detection** — cross-correlograms (CCGs) and feature projections are used to identify pairs of units that likely represent the same neuron. A union-find algorithm resolves transitive merge groups.
3. **Redundant unit removal** — units that are explained by linear combinations of other units are flagged and removed.

`split_depth_export` / `depth_split_um` allow large files to be processed as top/bottom depth halves and saved separately, useful when RAM is limited.

---

### Step 5 — QC (`run_qc`)

```python
qc_results = run_qc(seg_saved, cur_results, pipeline_dir / 'qc', recalc=True)
```

Three QC metrics are computed per unit:

| Metric | File | Description |
|--------|------|-------------|
| Waveform statistics | `waveforms/waveforms.npz` | Mean waveform shape, peak channel, amplitude |
| Refractory period violations | `refractory/refractory_qc.npz` | RVL tensor — fraction of spike pairs within the refractory period |
| Amplitude truncation | `amp_truncation/truncation_qc.npz` | Detects units whose amplitude distribution is clipped by the detection threshold, indicating incomplete spike capture |
| Unit presence | `amp_truncation/present_qc.npz` | Temporal stability of firing rate across the session |

---

### Step 6 — Export to MATLAB

All QC results are saved as `.mat` files in `qc/` for downstream analysis in MATLAB:

```
qc/waveforms_data.mat
qc/refractory_data.mat
qc/truncation_data.mat
qc/presence_data.mat
```

---

## Pipeline output directory structure

```
dredge_pipeline_results_<session>_<stream>/
├── conditioning/
│   └── channel_metrics.npy          # per-channel similarity & noise
├── motion/
│   ├── motion_info.pkl              # drift traces and interpolator
│   └── *.png                        # diagnostic plots
├── preprocessed_recording/          # cached binary (motion-corrected)
├── kilosort4/
│   ├── sorter_output/               # Kilosort4 native output (ops, templates, etc.)
│   └── sorter/                      # SpikeInterface sorting object
├── cur/
│   └── cur_sorter_output/           # curated spike trains and merge info
└── qc/
    ├── waveforms/
    ├── refractory/
    ├── amp_truncation/
    └── *.mat
```

---

## Repository structure

```
SpikeSortingTools/
├── environment.yml                      # Conda environment (pinned)
├── patch_kilosort_claimmask.py          # Kilosort cross-peel claim mask patch
├── SpikeGLX_ext_ref_2025.py            # Main pipeline — external reference
├── SpikeGLX_ext_ref_2026_patching.py   # Extended pipeline with claim mask params
├── SpikeGLX_tip_ref_2024.py            # Tip reference pipeline
├── SpikeGLX_tip_ref_2024_patching.py   # Tip ref + claim mask
├── example.py                           # Quick test on synthetic MEArec data
└── pipeline/
    ├── preprocess.py                    # condition_signal, bad channel detection
    ├── motion.py                        # correct_motion, DREDGE/MEDiCINe/etc.
    ├── sorting.py                       # sort_ks4, KilosortResults
    ├── qc.py                            # run_qc, waveform/refractory/truncation
    ├── refractory.py                    # RVL computation
    ├── truncation.py                    # Amplitude saturation analysis
    ├── curation.py                      # run_cur, duplicate removal, merging
    └── curation_postpatch.py            # Advanced curation (cosine, BIC, no-merge)
```

## Quick test

To verify the environment without real data, run the synthetic MEArec pipeline:

```bash
python example.py
```

This downloads a short synthetic recording and runs the full pipeline on a 10-second snippet.
