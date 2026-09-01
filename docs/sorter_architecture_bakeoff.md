# Sorter architecture bake-off

This comparison asks whether drift-aware neuron or template tracking improves
the accepted unwarped Luke recording. It is not another voltage-registration
sweep.

## Invariants

- Every candidate starts from the accepted rescue recording manifest.
- A bounded comparison uses one fingerprinted interval relative to that
  materialized recording. Receipts also retain original-source frame bounds.
- Spatial resampling of voltage for motion correction is forbidden.
- Sorter-native filtering, referencing, standardization, and whitening are
  allowed but must be recorded. This is a comparison of complete sorting
  architectures, not an assertion that their frontends are identical.
- A challenger cannot advance on unit count alone. Reviewed-event recovery,
  refractory violations, and near-coincident duplicate burden are guardrails
  before longitudinal continuity is interpreted.

## Candidates

### KS4 without motion

The existing accepted KS4 sort is the production reference. For a bounded
window, the bake-off extracts spikes in the half-open interval and rebases
their sample indices to the start of the window. This is explicitly recorded
as `accepted_full_sort_window_extraction`: it is not misrepresented as a fresh
KS4 run on a short recording.

### DARTsort native motion tracking

DARTsort receives the same accepted unwarped recording, runs its explicitly
recorded native frontend, estimates motion, and uses its native drift-aware
sorting pipeline. The repository describes DARTsort as work in progress and
states that it is not currently recommended for production spike sorting.
Accordingly, every DARTsort receipt is marked `experimental=True`; it is a
challenger, not a replacement for the KS4 reference.

The initial pinned challenger version is `dartsort==0.5.16`. Its native output
must contain `dartsort_sorting.npz` with `times_samples`, `channels`, and
`labels`, and the runner atomically accepts the result only after validating
those arrays.

That pin is now an important interpretation limitation. DARTsort 0.5.17 fixed
an invalid-label bug in the GMM and incorrect overlap handling in neural-network
subtraction, and later 0.5.x releases include additional bug fixes and
preprocessing changes. The 0.5.16 run also used the explicitly requested
`ibllikecmr` frontend, but this bake-off did not independently validate the
standardized voltage scale against the SNR-unit thresholds used by DARTsort or
test whether its early temporal subsampling adequately represented this
deliberately unusual 120 s rapid-motion window. The accepted run is therefore
useful historical and integration evidence, but its sparsity is not an
algorithmic verdict on a current, correctly tuned DARTsort.

### KIASORT through its upstream SpikeInterface wrapper

KIASORT's own repository includes `SpikeInterface_wrapper/` with a Python
`run_kiasort()` adapter around its MATLAB no-GUI entrypoint. This is separate
from SpikeInterface's built-in sorter registry. Set `KIASORT_PATH` to a pinned
KIASORT checkout (or pass `--kiasort-path`). The runner fingerprints both the
upstream Python wrapper and `run_kiasort_nogui.m`, invokes MATLAB, normalizes the
returned sorting to `spike_times.npy` and `spike_labels.npy`, and atomically
records the result.

KIASORT performs its own frontend processing. That frontend is part of the
candidate architecture and must not be described as identical to KS4's
frontend.

Machine audit on 2026-08-31: MATLAB R2022b is installed, and the Signal
Processing, Statistics and Machine Learning, Parallel Computing, Image
Processing, and Curve Fitting toolbox license checks all pass. The complete
KIASORT repository is installed at `/home/huklab/Documents/KIASORT` and pinned
to commit `4caecd25132da06485b6b2dde267fe1ecac5b895`; hashes are recorded in
`third_party/kiasort_pin.json`. Both MATLAB entrypoints and the Python wrapper
are required by the installation audit.

To test KIASORT graphically, open desktop MATLAB and run:

```matlab
pyenv('Version', '/home/huklab/anaconda3/envs/kiasort-python/bin/python');
addpath(genpath('/home/huklab/Documents/KIASORT'));
kiaSort
```

Alternatively, run `third_party/launch_kiasort_gui.sh`; it opens desktop MATLAB
with the same checkout and isolated Python 3.10 environment. Both paths can be
overridden with `KIASORT_PATH` and `KIASORT_PYTHON_EXECUTABLE`.

The graphical entrypoint resolves to the pinned checkout's `kiaSort.m`. The
pipeline separately uses the root `run_kiasort_nogui.m` through KIASORT's
upstream SpikeInterface wrapper. Current upstream also contains a mirrored
copy under `No_GUI/`; the adapter deliberately selects the root copy and
fingerprints it to avoid MATLAB path-order ambiguity.

This machine uses MATLAB R2022b. The pinned checkout carries a narrow,
documented compatibility diff: numeric `omitmissing` options are changed to
R2022b's equivalent `omitnan`, and the Python wrapper passes its generated
statements directly to `matlab -batch` rather than through R2022b's `run()`
helper. The GPU shared-noise decomposition also uses the R2022b-compatible
one-input `eig` form followed by `diag`, which is equivalent to the newer
`eig(C, 'vector')` result.
Duration-derived sample counts are rounded before indexing so the exact
non-integer SpikeGLX sampling rate remains supported. The base commit and
SHA-256 of the complete tracked diff are included
in every KIASORT request. This patch does not change sorter parameters or
algorithmic decisions.

For interrupted bounded runs, the opt-in `resumeCompletedStages` override may
reuse sample extraction only when `channel_info.mat` and at least one result
file for every configured channel are present. The override is part of the
request digest and receipt. New partial directories persist that digest before
MATLAB starts; older unguarded partials and parameter-mismatched partials are
refused rather than reused.

KIASORT's documented Python dependencies are installed in the isolated
`kiasort-python` environment (Python 3.10). The adapter requires an explicit
`--kiasort-python-executable` or `KIASORT_PYTHON_EXECUTABLE`, configures MATLAB
with out-of-process `pyenv`, disables user-site package leakage, exposes only
the isolated environment's library directory to the Python worker, and records
the executable in the run receipt.

The adapter defaults `NUMBA_NUM_THREADS` to 2 because a same-matrix benchmark
on this host completed in 25 seconds at two threads, while unconstrained nested
parallelism remained in the first UMAP fit for more than 15 minutes. Override
with `--kiasort-numba-threads`; the selected count is receipt-recorded.
Numba cache files default to `/tmp/kiasort-numba-cache` (or `NUMBA_CACHE_DIR`)
so MATLAB's isolated Python worker always has a writable cache location.

The upstream Python wrapper emits one-dimensional SciPy channel-map vectors,
which MATLAB otherwise reads as rows and concatenates into a malformed
`1 x 3N` geometry. The repository adapter reshapes `chanMap`, `connected`,
`xcoords`, `ycoords`, and optional `shankInd` to MATLAB column vectors before
the wrapper writes `channel_map.mat`. Every native candidate must then pass a
receipt-recorded geometry check (`N x >=2` locations and `N` mapping entries),
with both classic MAT and v7.3/HDF5 files supported. This is an API/storage
adapter, not a sorter parameter change, and it also leaves the resulting
native tree readable by the graphical MATLAB curation path.

For feasibility pilots, `--kiasort-channel-start-index` and
`--kiasort-channel-count` select a contiguous channel-index band before the
wrapper exports voltage. The band is request-digested and receipt-recorded,
and its output directory includes the half-open index range. Such a result is
not a full-probe bake-off condition.

The geometry-valid KIASORT feasibility run used the rapid-motion window,
channel indices `[82, 114)`, 20 distributed one-second sample chunks, serial
sample modeling, and two Numba threads. It completed native sorting on
2026-08-31 and normalized 274,715 in-bounds spikes from 90 units. Its accepted
output is `kiasort_channels_82_114/` under the window directory. The run proves
the adapter and native MATLAB path work; it does not establish a quality win or
provide a full-probe comparison. Full-probe sample modeling was not operationally
bounded, and MATLAB parallel workers aborted when each initialized Python/UMAP,
so neither route is a production default.

An earlier 284,980-spike/93-unit result was found to contain flattened
`1 x 96` channel geometry caused by the wrapper boundary above. It is retained
as `kiasort_channels_82_114_legacy_flat_geometry_20260831/` for diagnosis but
is excluded from all advancement evidence. Its earlier comparison artifacts
must likewise be treated as superseded.

Source and log audit distinguishes two cleanup layers. The
no-GUI entrypoint sets `run_auto_curation=false`, so the separate GUI-equivalent
`kiaSort_auto_curate_nogui` stage did not run. However, the result is not a raw
pre-cleanup sort: `kiaSort_main_sortData` ran with `postHocProcessing=true`,
`postHocDriftMerge=true`, `postHocRemoval=true`, and the default
`postHocMerging=false`. In the geometry-valid run, `KIASort_log.txt` records
completion of the iterative drift merge, zero in-place merges, five CCG
strips, 8,566 overlap strips, and successful SNR recomputation. KIASORT then
hit an upstream rollback bug after deleting `unifiedLabels.h5`: its backup was
complete but the native restore call failed. The adapter accepted recovery
only after current/backed-up spike arrays were byte-identical, current/backed-up
`sorted_samples.mat` were byte-identical, backed-up labels had exactly 274,715
events, and the log recorded post-hoc completion. The receipt explicitly
records that rollback; ambiguous partial transactions are refused.

The non-destructive GUI-equivalent auto-curation replay is now complete in
`kiasort_channels_82_114_auto_curated/`. It reused no extraction, UMAP, sample
sorting, or full-data sorting; canonical replay inputs remained hash-identical.
Native auto-curation reported zero unit merges, 295 overlap strips, and all 90
units retained. Stable-interval gating then excluded 46,314 assigned events,
leaving 228,106 normalized spikes. Waveforms were reconstructed from the same
accepted unwarped voltage rather than exported by the curation stage.

Auto-curation did not pass the fairness gate: it retained every label, raised
the qualified one-KS4-to-many-KIASORT family count from five to eight, reduced
median 10-second presence from 1.0 to 0.5, and reduced KS4 event support from
92.4% to 80.6%. Therefore this default auto-curation output is not the preferred
KIASORT condition. One bounded merge/sampling-focused tuning pass remains
permitted; representative sampling comes before template-versus-SVM, and
detection-threshold changes come last. A longer quiet--motion--quiet window is
deferred until fragmentation is controlled.

### SpikeInterface motion-aware peeler benchmark

The production environment remains pinned to SpikeInterface 0.102.1. The
isolated challenger environment pins 0.104.8, where the TDC peeler exposes
`motion_aware`, `motion`, `interpolation_time_bin_size_s`, and `motion_step_um`.
The external qualified-field adapter remains available for future work, but the
present experiment deliberately uses KS4's own rigid `dshift` as an
experimental latent trajectory. This does not treat `dshift` as physical
ground truth or spatially resample the voltage.

The benchmark is a paired KS4-seeded experiment, not KS4's native
registration-oriented correction:

1. `ks4_seeded_static_peeler`: accepted KS4 unit IDs and seed events, shared
   re-estimated templates, original unwarped voltage, and the TDC peeler with
   motion awareness disabled.
2. Two motion-aware arms use the same templates, conditioning, thresholds, and
   original voltage, with only the native or stabilized KS4 rigid `Motion`
   object enabled to move templates during matching.

All arms use the KS4-to-SpikeInterface template adapter, identical
train/evaluation policy, normalized sorting output, and atomic receipts. Their
paired difference isolates template motion from changes in detection,
clustering, or voltage interpolation. `ks4_no_motion` remains the production
reference, while the static peeler arm controls for changing the matching
engine itself.

The adapter is now implemented as one atomic three-arm run:

1. `ks4_seeded_static_peeler` uses no motion.
2. `ks4_seeded_motion_native_peeler` uses the negative median-centered native
   rigid `dshift` trajectory. The sign inversion is required because KS4 stores
   the voltage-correction shift while SpikeInterface `Motion` represents the
   observed displacement used to move templates on stationary voltage.
3. `ks4_seeded_motion_stabilized_peeler` uses the same trajectory after a
   frozen causal unwrapping rule rejects batch-to-batch steps larger than
   20 um. A rejected step contributes zero increment; the rule is applied
   before looking at peeler output.

All arms use the same accepted KS4 unit IDs and deterministic seed-spike set.
Templates are re-estimated in matcher units from the accepted unwarped voltage
after a shared 300 Hz high-pass and global median reference, then sparsified to
a 100 um radius. This avoids mixing exported KS4 template scaling with a
different matcher frontend. The run saves the seed spikes, templates, sparsity,
noise levels, native and stabilized trajectories, and normalized outputs for
each arm. The root receipt hashes the accepted recording, accepted KS4 sort,
motion-source `ops.npy`, window, and every matching parameter. It also reports
label-preserving KS4 event recovery, median unit refractory burden, 10 s
presence, first/last-20 s persistence, and cross-unit near-coincident pairs for
each arm. Those are immediate guardrails, not substitutes for reviewed events
or the raw-waveform residual audit.

SpikeInterface 0.104.8 has a motion-branch warm-up assertion that compares a
margin-padded trace with an unpadded time vector. The adapter disables only that
unnecessary warm-up and therefore requires `n_jobs=1`; actual matching code is
unchanged and the workaround is receipt-recorded.

This paired benchmark is the next architecture priority. It is referred to
informally as "motion-aware Kilosort4," but the experimental intervention is
specifically motion-aware matching of KS4-seeded templates in the
SpikeInterface TDC peeler; it is not a claim that stock KS4 exposes this mode.
DARTsort reruns and further KIASORT tuning are deferred unless this benchmark
creates a concrete reason to reopen them.

## Isolated environment

Create a separate environment; do not upgrade the accepted production
environment in place:

```bash
conda env create -f environment-challengers.yml
conda activate spike-sort-challengers
```

The environment pins SpikeInterface 0.104.8, DARTsort 0.5.16, and the
CUDA-12.4 build of Torch 2.6.0. This avoids pip's current CUDA-13 default,
which cannot initialize against this host's NVIDIA 550 driver. On 2026-08-31,
the host reports an RTX A5000 with 24,564 MiB total memory; a successful Torch
device-allocation smoke test is still required before a DARTsort data run.

## Commands

```bash
python sorter_bakeoff.py --rescue-output-dir /path/to/results --plan
python sorter_bakeoff.py --rescue-output-dir /path/to/results --plan \
  --window-name rapid_motion --start-s 5910 --duration-s 120
python sorter_bakeoff.py --rescue-output-dir /path/to/results \
  --run ks4_no_motion --window-name rapid_motion --start-s 5910 --duration-s 120
python sorter_bakeoff.py --rescue-output-dir /path/to/results \
  --run dartsort_native --window-name rapid_motion --start-s 5910 --duration-s 120
python sorter_bakeoff.py --rescue-output-dir /path/to/results \
  --run ks4_seeded_peeler_pair \
  --ks4-motion-ops /path/to/rigid_ks4/sorter_output/ops.npy \
  --ks4-motion-time-reference window_start \
  --window-name rapid_motion --start-s 5910 --duration-s 120
python sorter_bakeoff.py --rescue-output-dir /path/to/results --run kiasort \
  --kiasort-path /home/huklab/Documents/KIASORT \
  --window-name rapid_motion --start-s 5910 --duration-s 120
```

Window outputs live under
`sorter_bakeoff/windows/<name>-<window-digest>/<candidate>`, so intervals with
the same human label cannot overwrite one another.

Run the same paired command on a prespecified duration-matched quiet window as
the negative control. The intended interaction is a continuity benefit in the
rapid-motion window with little or no change in quiet data. The quiet interval
must be chosen independently of peeler results. Use
`--ks4-motion-time-reference selected_recording_start` only when the supplied
rigid `ops.npy` was estimated across the complete selected recording; use
`window_start` for a motion-source run materialized on the challenge window.

## First bounded execution

On 2026-08-31, the imec1 `rapid_motion` window (nominal start 5910 s,
duration 120 s; 3,599,971 samples) completed under DARTsort 0.5.16. The atomic
receipt reports 488,236 detected spikes, 478,251 assigned spikes, and 763 unit
labels. The normalized arrays contain 478,251 sorted, window-relative sample
indices ranging from 46 to 3,599,881, all inside the requested interval.
DARTsort's native timing report gives 1,531.6 s for its pipeline stages; total
wall time was longer because the accepted recording slice was first cached as
5.53 GB of float32 data.

This is an execution/integrity result, not evidence that DARTsort beats KS4.
The accepted full-session KS4 extraction for the same window contains 888,641
spikes and 514 labels, but raw spike or unit counts are not comparable quality
metrics across architectures. Event recovery, refractory/duplicate burden,
and longitudinal waveform-family continuity remain the advancement gates.

## First KS4-seeded motion-aware execution

Terminology correction: the static control is a **KS4-seeded TDC replay**, not
“KS4 without motion.” It re-estimates sparse templates from KS4 seeds but uses
TriDesClous detection, nearest-template selection, amplitude fitting, and
iterative subtraction. Static versus motion therefore isolates motion within
TDC, not within KS4's own learned-template matching machinery.

The rapid-motion three-arm run completed under SpikeInterface 0.104.8 and was
atomically accepted as schema `ks4-seeded-motion-aware-peeler-v2`. The motion
adapter uses the negative median-centered KS4 `dshift`: KS4 stores a
voltage-correction shift, whereas SpikeInterface `Motion` represents observed
template displacement on stationary voltage. An initial v1 run used the wrong
sign; it is preserved as
`ks4_seeded_peeler_pair_legacy_wrong_sign_20260831/` for diagnosis and excluded
from advancement evidence.

All v2 arms used byte-identical shared seed spikes, templates, sparsity, and
noise levels. The eligible KS4 reference contained 887,910 events from 403
units; template training used a deterministic cap of 415,879 seed events.

| Arm | Spikes | Units | KS4 label-preserving recovery | Median 1.5 ms refractory fraction | First/last 20 s | Cross-unit pairs/spike within 0.2 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Static peeler | 148,684 | 381 | 0.0783 | 0.00515 | 0.8263 | 0.4219 |
| Native rigid motion | 162,948 | 381 | 0.0572 | 0.00758 | 0.8486 | 0.4725 |
| Stabilized rigid motion | 154,644 | 380 | 0.0707 | 0.00562 | 0.8213 | 0.4362 |

These recovery and near-coincident values were recomputed from the unchanged
saved arrays after fixing one-to-one event matching and all-pairs duplicate
counting. The original canonical manifest contains the pre-fix summaries.

Neither motion-aware arm passes. Native motion adds 9.6% more events and a
small endpoint-presence gain, but KS4-label recovery falls by 2.10 percentage
points while refractory and near-coincident burdens rise. Stabilization limits
the damage but still lowers recovery by 0.76 points, slightly worsens both
guardrails, and does not improve endpoint presence.

Under corrected one-to-one matching, 84.9% of native-motion events and 90.8%
of stabilized-motion events have a distinct static-peeler counterpart within
0.5 ms. Label-preserving support is only 57.8% and 78.2%, respectively. Thus
motion mostly changes template identity assignments, while native motion also
adds a material redundant or unpaired event surplus. The static peeler itself
recovers only 7.8% of eligible accepted
KS4 events with the same label, so it is not yet a faithful replacement for
KS4's matcher. A repeat static arm with byte-identical shared inputs had 96.4%
same-label and 96.6% any-label one-to-one support when the larger run is the
denominator (99.2% and 99.3% in the reverse direction), revealing additional
matcher nondeterminism that matters for small effects.

A fixed-event gate trace then localized a major configuration mismatch: 79.7%
of eligible KS4 events belong to positive-dominant re-estimated templates,
while the TDC control used its default negative-only detector.
Positive-dominant replay was 2.31%, versus 29.54% for negative-dominant events.
Changing the local trace to both polarities and threshold 4 raised correct
template eligibility among missed controls from 22.5% to 84.5%, but correct
nearest-template selection reached only 49.5%. Thus polarity explains a large
part of the failure but does not reduce it to one threshold fix. See
`docs/luke_static_tdc_fidelity_trace.md`.

This configuration does not advance to a quiet control or full session. Any
future reopening must first solve static TDC replay and determinism; motion
tuning is not the next step. At most one static-only both-polarity,
threshold-4 control is justified before this branch is closed.

## First channel-matched descriptive comparison

The reproducible analysis in `testing/luke_sorter_band_comparison.py` compares
the accepted outputs on the same rapid-motion window and physical band:
channel indices `[82, 114)`, corresponding to depth rows 41--56. KIASORT was
run on that 32-channel band; KS4 and DARTsort were restricted to it after their
full-probe sorts. The source manifests must be complete, refer to the same
recording and window, and declare `raw_voltage_warp=false`.

| Sorter | Band spikes | Units with at least 20 spikes | Median 1.5 ms refractory fraction | Units present in first and last 20 s | Cross-sort event coverage |
| --- | ---: | ---: | ---: | ---: | --- |
| KS4 no motion | 186,001 | 51 | 0.0076 | 0.902 | 65.8% by DARTsort; 92.4% by KIASORT |
| DARTsort native | 99,279 | 76 | 0.0028 | 0.789 | 98.7% by KS4; 94.8% by KIASORT |
| KIASORT geometry-valid native | 274,715 | 90 | 0.0165 | 1.000 | 68.7% by KS4; 46.0% by DARTsort |
| KIASORT auto-curated | 228,106 | 88 | 0.0205 | 0.284 | 67.4% by KS4; 44.0% by DARTsort |

Cross-sort coverage means that an event from the row's sorter has an event
from the named sorter within 0.5 ms and 60 um. It is symmetric event support,
not unit matching. Geometry-valid native KIASORT has broad temporal persistence
and recovers most KS4 events, but also has the highest native refractory burden
and many additional events. Its auto-curated condition is shown separately:
the `88` analysis units exclude two retained labels with fewer than 20 final
events. DARTsort is a sparse, low-refractory subset that is almost entirely
supported by both alternatives in this run. A shifted-unit null shows no excess
near-coincident burden for KIASORT under the present event-level test, but that
does not rule out split or duplicate units.

These results permit at most one targeted KIASORT tuning pass if that branch is
reopened; they do not justify a full-probe or production replacement. They also
do not justify rejecting DARTsort: the run used 0.5.16, preceded relevant
upstream fixes, and did not close the preprocessing-scale and short-window
sampling checks described above. Neither follow-up is the current priority.
The next effort is the paired KS4-seeded static versus motion-aware peeler
benchmark, which directly tests the template-motion hypothesis while holding
the KS4 seed population fixed.

### Unit-family and raw-waveform arbitration

`testing/luke_sorter_unit_families.py` constructs bidirectional event-overlap
edges and connected unit families. Under the recorded baseline edge rule, the
51 eligible KS4 units and 90 eligible geometry-valid KIASORT units resolve into
eight 1-to-1 families, five simple 1-KS4-to-several-KIASORT candidates, one
simple several-KS4-to-1-KIASORT candidate, 25 isolated KIASORT units, 15
isolated KS4 units, and several complex many-to-many neighborhoods. Counts vary under
permissive and stringent edge rules, so connected-component shape is a
hypothesis generator rather than a biological unit count.

The raw-waveform audit in `testing/luke_sorter_waveform_arbitration.py` reads
the exact accepted unwarped `int16` voltage for the band. Beginning, middle,
and end sample checks reproduce the corresponding channels in the full
accepted recording exactly. Every sorter is evaluated after the same
32-channel common-median reference and 300--6000 Hz zero-phase filter;
alternating events form and score independent templates, with 17 ms-shifted
times as a background control.

The five simple 1-KS4-to-many-KIASORT families have median cross-sort template
cosines from 0.86 to 0.91 (except individual minima down to 0.81), similar
KIASORT child templates, and up to six KIASORT identity switches across 10 s
bins. This remains consistent with KIASORT splitting temporally complementary
portions of related waveform families. The sole simple reverse candidate,
KIASORT unit 83 versus KS4 units 249 and 256, is weak: median cross-sort cosine
is 0.53 and within-KS4 cosine is 0.34.

The 25 isolated geometry-valid KIASORT families contain 81,979 spikes. The same
conservative waveform gate retains only units 46 and 82 (2,788 spikes):
refractory fraction at most 0.02, split-half and early/late template cosine at
least 0.65, and median event-centered explained-fraction excess at least 0.05.
Auto-curation retains the same two review targets (2,511 spikes) but does not
improve the family-level result. These are review targets, not accepted added
yield.

The ignored analysis artifact `family_waveform_examples.png` shows the dominant
fragmentation family, the remaining merge hypothesis, and the two-unit
isolated-waveform shortlist with the recorded gate lines.

If KIASORT is reopened, the present evidence supports at most one
merge/sampling-focused tuning pass rather than a full-probe runtime effort. It
does not demonstrate that KIASORT maintains a coherent identity where KS4
switches; most apparent extra yield remains waveform-family fragmentation or
low-confidence isolated output. Default auto-curation has now been tested and
does not rescue this configuration, so this branch is deferred behind the
paired KS4-seeded motion-aware benchmark.

The run emitted a Numba warning that its installed TBB was too old for Numba's
TBB threading backend, so that backend was disabled. The run nevertheless
completed. DARTsort also reported 104 initially uncovered spikes during a
neighborhood-coverage check and expanded coverage to zero uncovered spikes.
