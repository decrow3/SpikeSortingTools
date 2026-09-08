# Bounded overnight transfer diagnostic: frozen compensated3σ baseline

Prepared three100-second AP-only epochs: early940–1040s, middle6000–6100s, late9480–9580s. These are development transfer checks, not independent validation or a full-session scan. The middle interval is prespecified for coverage with no cached biological reference. Early and late include previously inspected candidate holdouts at970–990s and9520–9540s.

The single retained input is300–6000Hz, third-order zero-phase Butterworth with50ms margins, followed by the frozen31-tap shared-response compensation. Detection uses negative locally exclusive3σ peaks,50µm radius, and the original frozen channel-noise vector. Localization uses monopolar triangulation with75µm radius. The estimator copies the saved baseline settings, including enforced±80µm pairwise search. There is no low-pass arm, screening sweep, sort, or production correction.

## Execution and restart contract

Command from repository root:

```bash
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 environments/rescue-production/.venv/bin/python -m testing.luke_motion_transfer_overnight_v1
```

Run only through the root's independently managed launcher, with persisted command, service identifier, logs and final receipt. Recommended allocation: CPU only, at most4localization workers, numerical kernels/Torch restricted to1thread, memory limit24GiB. No GPU is needed. The script uses an exclusive output lock to reject simultaneous launchers.

Each20-second chunk reads only its bounded padded voltage region. Its settings/code/library provenance and actual input-byte SHA256 are bound to the checkpoint. Completed peaks and locations are validated for shape, finite values, index bounds, ordering and uniqueness; both output hashes are recorded before an atomic completion-marker replacement. A restart rereads and hashes the bounded source bytes, verifies settings and saved arrays, and skips only a valid committed chunk. A killed/incomplete chunk restarts from its beginning; its attempt directory remains preserved. Corrupt or missing committed outputs and changed inputs/settings cause a hard failure requiring investigation.

There is no within-chunk or within-DREDGE checkpoint. Motion/figures rerun from validated chunk arrays after interruption, in a new preserved epoch attempt directory. Each epoch's `complete.json` and final run summary point to the actual artifact directory. A completed chunk is not described as within-sort checkpointing; no sort is involved. The raw recording is not hashed in full: file identity/size/mtime and per-chunk content hashes provide bounded verification.

## Outputs and interpretation

Outputs live under `testing/outputs/luke_motion_transfer_overnight_v1/`. Each epoch attempt saves peaks, localizations, DREDGE fields and D/C/U matrices; count/amplitude rasters; waveform-reference observations; and PNG/PDF figures. The three principal figures are `01_motion_raster`, `02_cached_waveform_corroboration` and `03_pairwise_support` inside the attempt named by the epoch receipt.

Pairwise support tables show positive outgoing edges, weight mass, weighted correlation and weight fraction hitting±80µm by time/depth, excluding diagonal self-comparisons. A deterministic sample of2000triangles per depth reports absolute cycle residuals only where all three directed edges have positive weight. Support, high correlation and cycle consistency are internal diagnostics, not evidence of physical correctness. No extra lag-curve replay or parameter tuning is performed.

Early/late plots overlay the fixed candidate centroids with DREDGE sampled at the actual cached event times. Both are referenced to the first valid candidate holdout bin; low-count bins remain gaps. Only20seconds and a few deep candidate locations have waveform observations, with no shallow reference. Their fixed support, positive waveform similarity, sparse competitors and failed translated-matcher identity checks prevent a ground-truth error score. Cached original-voltage waveforms are labeled as such and do not certify compensation preservation in these epochs. The middle plot explicitly states that biological corroboration is unavailable. All-invalid candidate bins also produce an explicit gap rather than aborting the run.

## Cheap prelaunch checks

`testing/test_luke_motion_transfer_overnight_v1.py` covers a subprocess killed by SIGKILL before commit, subsequent restart from a completed synthetic checkpoint while preserving killed-attempt evidence, input/settings mismatch rejection, corrupted/missing output rejection, and a full synthetic three-figure path with no cached references. These checks never read recording voltage or run DREDGE. The parent launcher separately owns service-disconnection and durable job-status verification.
