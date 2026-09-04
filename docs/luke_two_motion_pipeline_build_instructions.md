# Build instructions: Luke two-option motion pipeline bakeoff

**Status:** implementation specification; not a result and not a frozen numeric
prespec  
**Governing plan:**
[`pipeline_improvement_plan.md`](pipeline_improvement_plan.md)  
**Purpose:** build two credible motion-handling pipelines quickly enough that
C2 v4 can trigger a fair bakeoff immediately:

- **Option A — external voltage registration:** estimate and qualify a motion
  field, resample the accepted voltage once on the supported full geometry, then
  sort with KS4 motion correction disabled.
- **Option B — unwarped motion-aware identity:** leave accepted voltage and
  spike times unchanged; use motion only to express coordinates in a tissue
  frame and to track identity through time.

The current uncorrected rescue pipeline is the **shared control**, not a third
development option. Historical external-warp failures reject those particular
fields, operators and operating points; they do not reject external voltage
registration as a class.

## 1. Outcome and time discipline

The build is complete when one command can run the same accepted input through
the control, Option A and Option B and emit the same score schema for all three.
Do not turn this into another general framework project.

Time-box the work:

1. one smoke window per option;
2. C2 v4 causal calibration;
3. L1 on one discriminating window;
4. L2 only for candidates that pass L1;
5. L2L only for candidates that pass L2;
6. held-out and production-scale materialization only after the option is frozen.

Implementation, unit tests and smoke tests may run while C2 v4 runs. Numeric
tuning, candidate selection and promotion may not. Within the existing
six-configuration D2 cap, start with no more than three external-warp policies:
the field-standard full correction, one conservative low-order/partial policy,
and one supported non-rigid policy if the field diagnostics justify it.

## 2. Inputs and storage

### Shared immutable input

The 24 frozen real-data windows are available to both machines at:

```text
/mnt/NPX/Luke/20250804/shared_analysis/rigid_dose_snippets/
```

Expected inventory: 24 `rigid_dose_iv*` directories, 384 files and
66,382,318,589 file bytes. The copied SpikeInterface recordings use relative
paths to `traces_cached_seg0.raw` and can be loaded directly from `/mnt`.

Treat this tree as read-only. Never write a sort, cache, corrected recording,
temporary file or summary beneath `/mnt`. Each host uses a distinct local root,
for example:

```bash
export MOTION_BAKEOFF_ROOT=/media/huklab/Data/luke_motion_bakeoff_huklaban1
export LADDER_L1_ROOT="$MOTION_BAKEOFF_ROOT/l1"
```

On huklaban5, use its own local data volume and a host-specific directory name.
Do not point two hosts at the same writable output root.

### Required repository inputs

- `docs/luke_within_rigid_motion_windows.frozen.json`
- the new frozen C2 v4 prespec, once written
- the 14 hash-frozen compact donors from D2b-2
- `testing/ladder_l1.py`
- `testing/ladder_sorter.py`
- `testing/ladder_score.py`
- `testing/ladder_motion.py`
- `testing/ladder_motion_estimate.py`
- `testing/luke_rescue_c2_drift_challenge.py` as historical v3 scaffolding only

Record the Git commit, Python environment, SpikeInterface version, Kilosort
version, CUDA version and GPU model in every run manifest.

## 3. Shared invariants

Both options and the control must share:

- the same accepted preprocessing input;
- identical channel set and channel order at the comparison point;
- identical KS4 detection thresholds, CAR/high-pass/whitening choices and
  curation settings unless the frozen comparison explicitly varies one;
- identical injected donor, spike train, placement and static/moving pair;
- identical truth scorer and guardrails;
- content-bound cache identity;
- one output namespace per option and configuration.

Only motion handling may differ. Never combine external voltage correction with
KS4 internal correction in the primary comparison. Never compare an externally
warped recording using legacy detection thresholds against an unwarped recording
using rescue thresholds and call the difference a motion effect.

The control is:

```text
accepted recording
  -> no external spatial resampling
  -> KS4 do_correction=False, nblocks=0
  -> frozen curation
  -> common scorer
```

### Required orchestration contract

Add one thin top-level runner rather than separate ad hoc notebooks:

```text
testing/luke_two_motion_pipeline_bakeoff.py
```

It must expose at least:

```text
--option control|external_warp|unwarped_identity
--snippet-dir PATH
--out-root PATH
--motion-info-dir PATH        # required for motion-aware arms
--truth PATH                  # required for injected-truth runs
--config PATH                 # frozen option configuration
--mode verify|smoke|l1|l2|l2l
```

The runner must print the resolved configuration and digests before work starts,
refuse an output below `/mnt`, refuse an existing incompatible output, and write
one top-level run manifest even when a downstream stage fails. `verify` performs
no sort and checks inputs, hashes, environment, write location and option
invariants. `smoke` runs exactly one frozen window. Do not add an implicit
"latest field" lookup or default to whichever cache happens to exist.

Freeze the first scientific comparison in a separate versioned file, suggested
name:

```text
docs/luke_two_motion_pipeline_bakeoff.v1.frozen.json
```

The frozen file owns option IDs, motion-field receipt, interpolation policy,
epoch/link rule, input windows, donor cohort hash, scorer schema and stop gates.

## 4. Common C2 v4 calibration

C2 v4 must be frozen in a new script/prespec/output namespace. Do not edit or
reuse the v3 result namespace.

### Required design

- All 14 compact D2b-2 donors, both polarities, 73–295 µV.
- Static plus Luke-calibrated rigid trajectories near 4–5, 10–12 and
  20–25 µm.
- Geometry-aware forward motion using
  `paired_geometry_motion_injection`.
- Same physical-y operator family and sign convention for forward motion and
  inverse/oracle correction.
- Content-bound accepted-recording digest.
- Per-cluster exclusive truth matching and the chance-null split/merge gate.
- Static qualification: accuracy at least 0.8 under both rescue and
  legacy-style configurations before a donor enters the primary drift summary.

Static rescoring has already removed the old ~0.78 scorer floor. Preserve its
result as a validation receipt; do not interpret the stale v3 moving arms.

### Operator tests before sorting

1. Zero trajectory returns the input within the declared numerical tolerance.
2. Forward trajectory followed by exact inverse returns the donor waveform near
   the measured interpolation floor.
3. The same test passes at both polarities and at shallow/mid/deep placements.
4. A deliberately reversed sign fails clearly.
5. A real bin-edge truth event keeps all accuracy values in `[0, 1]`.
6. Cache digests change when voltage content, trajectory, interpolation policy
   or field content changes.

The C2 output is the common causal reference. It asks how much identity is lost
because of imposed Luke-like motion; it does not by itself promote either
pipeline.

## 5. Option A — external voltage-registration pipeline

### 5.1 Stage graph

```text
accepted full-geometry recording
  -> float32 estimator view (300–6000 Hz unless a frozen alternative says otherwise)
  -> full-duration motion estimation
  -> independent field diagnostics and qualification receipt
  -> apply field once to the original accepted sorter input
  -> geometry-aware interpolation with explicit boundary policy
  -> save a content-bound local accepted recording
  -> KS4 with do_correction=False and nblocks=0
  -> frozen curation
  -> common scorer and waveform guardrails
```

Motion may be estimated on a filtered float view, but correction is applied to
the accepted unfiltered sorter input. Estimate on the full-duration recording,
not independently on each 120 s snippet, when testing a real field. Apply on the
full supported channel geometry before any depth crop.

### 5.2 Existing implementation anchors

Use and extend, rather than duplicate:

- `estimate_full_session_motion()` in `testing/ladder_motion_estimate.py`
- `qualify_field()` and `FieldGate`
- `materialize_qualified_correction()`
- `oracle_corrected_recording()` in `testing/ladder_motion.py`
- `l1_run()` in `testing/ladder_l1.py`
- the rescue `SorterConfig`, with KS4 internal correction explicitly disabled

Keep the option-specific implementation in:

```text
testing/luke_external_warp_pipeline.py
```

It should accept explicit input, motion-info and local output paths. It must not
contain a hidden `/mnt` write, silently estimate a new field, or fall back to an
unqualified field.

### 5.3 Field qualification

Before materialization, require a receipt covering:

- source recording content SHA-256 and request digest;
- motion-array digest;
- acquisition-time versus recording-time mapping;
- displacement sign convention;
- supported time/depth fraction;
- split-half or otherwise independent field reproducibility;
- displacement range and temporal bandwidth;
- spatial gradient/non-rigid range;
- real voltage support at both spatial boundaries;
- estimated error relative to the C2/D2b tolerance envelope once available.

Missing support, reproducibility or error evidence fails closed. Field shape
alone is not accuracy evidence.

### 5.4 Application contract

The materialized manifest must record:

- source content digest and motion digest;
- interpolation method and every parameter;
- `border_mode`;
- dtype before interpolation and final dtype;
- correction strength/gain;
- channel positions before and after application;
- selected time range and time origin;
- whether any spatial crop occurred, which must be after correction;
- proof that KS4 internal correction was disabled.

The current implementation anchor uses float32 interpolation, geometry-aware
SpikeInterface `InterpolateMotionRecording`, `force_extrapolate`, kriging and an
explicit sigma. Treat that as a recorded starting policy, not an eternally fixed
winner. Any alternative must consume one of the preregistered configuration
slots.

### 5.5 Required arms

For injected known motion:

1. static, uncorrected control;
2. moving, uncorrected control;
3. moving, exact-oracle external correction;
4. moving, deliberately imperfect fields for the coarse D2b tolerance test.

For real data after the oracle gate:

1. no external correction;
2. best qualified full correction;
3. one simple partial/selective policy only if its rule was frozen from D2b.

Do not optimize against KS-good count or total spikes.

### 5.6 Option-A unit tests

- refuse output paths beneath `/mnt`;
- refuse an unqualified field;
- reject source, field or application digest mismatch;
- reject external plus internal double correction;
- reject correction after an unsupported crop;
- detect time-origin mismatch;
- verify zero-motion identity and forward/inverse sign;
- verify changed field content invalidates the recording and sort cache;
- verify the saved applied KS4 settings have `do_correction=False`, `nblocks=0`.

## 6. Option B — unwarped motion-aware identity pipeline

### 6.1 Minimum viable architecture

Option B must be genuinely unwarped: it may read a motion field, but it may not
spatially interpolate the accepted voltage or alter spike times.

```text
accepted recording
  -> unchanged KS4 rescue sort
  -> partition observations into overlapping time epochs
  -> estimate unit waveform/location in each epoch
  -> express locations in a motion-corrected tissue coordinate frame
  -> build evidence-limited links between adjacent epochs
  -> solve longitudinal identity tracks
  -> emit original spikes with a separate family/track identity
  -> common scorer plus union-refractory and continuity guardrails
```

Suggested implementation modules:

```text
testing/ladder_unwarped_identity.py
testing/luke_unwarped_identity_pipeline.py
```

Do not overwrite `spike_clusters.npy` during development. Emit a separate
mapping from `(original_cluster, epoch)` to `track_id`, plus a manifest. This
keeps every proposed merge reversible and auditable.

### 6.2 Epoch observations

For each original unit and epoch, record:

- original cluster ID and epoch bounds;
- spike count and firing rate;
- waveform on a fixed physical channel set;
- peak channel and observed depth;
- tissue-frame depth derived from the frozen motion field;
- amplitude and waveform cosine to adjacent epochs;
- refractory and contamination evidence;
- estimator support/confidence at that time and depth.

Freeze epoch duration and overlap before L2. Do not select them by maximizing
the number of linked units.

### 6.3 Link gates

A candidate link must satisfy all frozen evidence dimensions:

- spatially plausible observed or tissue-frame displacement;
- waveform similarity on the same physical channels;
- amplitude compatibility;
- temporal continuity or complementarity;
- adequate motion-estimator support if motion is used;
- no unacceptable refractory violation on the proposed union;
- no better conflicting one-to-one assignment.

Use maximum-weight one-to-one matching between adjacent epochs by default.
Branches, many-to-one links and gap bridging require explicit evidence and must
remain separately flagged. The Phase A2 result forbids treating temporal
complementarity alone as proof that clusters are one neuron.

### 6.4 Outputs and reversibility

Write:

- `epoch_observations.parquet` or CSV;
- `candidate_links.parquet` or CSV with each gate component;
- `accepted_links.parquet` or CSV;
- `track_membership.parquet` or CSV;
- `unwarped_identity_manifest.json`;
- a score adapter that presents track identities to `score_sort()` without
  deleting the original assignment.

The manifest must hash the original sort, motion field, epoch definition, link
rule and output mapping. A reviewer must be able to reconstruct why every link
was accepted or rejected.

### 6.5 Option-B unit tests

- raw recording bytes are never written or changed;
- spike times and original cluster IDs are preserved exactly;
- zero-motion stable units retain one identity;
- a synthetic coherent moving unit links across epochs;
- simultaneous duplicate clusters do not become one track merely because their
  waveforms are similar;
- an apparently complementary link fails when the union refractory gate fails;
- ambiguous competing links remain unresolved rather than greedily merged;
- reversing motion sign worsens the synthetic tissue-frame trajectory test;
- changed motion or link-rule content invalidates the track cache;
- the score adapter uses the same chance-null split/merge scorer.

## 7. Smoke test on each host

Use the median-dose frozen window (`iv3360`) unless the active frozen prespec
names another smoke window. The smoke test is engineering validation, not a
scientific result.

Before launch:

```bash
git status --short
git rev-parse HEAD
find /mnt/NPX/Luke/20250804/shared_analysis/rigid_dose_snippets \
  -maxdepth 1 -mindepth 1 -type d -name 'rigid_dose_iv*' | wc -l
```

Then verify that SpikeInterface loads the shared recording and can read first,
middle and final trace blocks. Write all outputs beneath the host-local
`MOTION_BAKEOFF_ROOT`.

The smoke receipt must state:

- host and GPU;
- Git commit and environment versions;
- input path, spec digest and accepted-recording digest;
- output root;
- option/configuration label;
- runtime by stage;
- whether every expected manifest and score file exists;
- whether the shared `/mnt` input remained unchanged.

Do not interpret smoke-window unit counts.

## 8. Common evaluation ladder

### L1 — one discriminating window

Run the control and both options on the same input. All arms must emit the
`luke-ladder-score-sort-v3` score structure. Stop an option for:

- worse injected accuracy/recall without a compensating prespecified benefit;
- new chance-null-supported splits or merges;
- waveform or localization regression;
- refractory/coincidence/boundary guardrail failure;
- broken provenance or cache identity;
- runtime already incompatible with the 1.25× promotion ceiling.

### L2 — frozen panel

Advance each fixed L1 survivor. Use the frozen panel and report every donor and
waveform stratum, not only the aggregate. No parameter changes after seeing L2.

### L2L — longitudinal identity

This is the decisive real-data tier for Option B and an essential guardrail for
Option A. Measure identity continuity across the 2.9-hour recording, ownership
switches, waveform drift, union refractory behaviour and the symmetric legacy
accounting. A short-window winner that fragments longitudinally stops here.

### Held-out and replication

Open the blinded holdout only for a frozen finalist. Then replicate on imec1 and
a second session before changing the production default. Luke–Yates biological
comparison remains downstream of pipeline lock.

## 9. Decision table

| Result | Pipeline consequence |
|---|---|
| External warp beats control and unwarped option on truth and L2L without waveform cost | Promote Option A to held-out testing |
| Unwarped option preserves identity better and external warp retains interpolation cost | Promote Option B to held-out testing |
| Both beat control in different strata | Freeze a simple physical selective rule, then validate once |
| Neither beats control | Keep the current control; move to peeling/collision and saturation handling |
| Oracle warp helps but estimated warp does not | Motion estimation/qualification is the blocker, not interpolation in principle |
| Oracle warp cannot help compact donors | Close the tested voltage-registration regime and prioritize unwarped identity handling |

No row authorizes promotion by yield alone.

## 10. Handoff checklist

Before another agent starts a long run, it must report:

- [ ] exact Git commit and clean/dirty worktree state;
- [ ] selected option and frozen configuration ID;
- [ ] source and output paths;
- [ ] confirmation that `/mnt` is read-only input;
- [ ] input/spec/content hashes;
- [ ] static qualification receipt and scorer schema;
- [ ] operator/link-rule unit tests passing;
- [ ] smoke receipt passing;
- [ ] expected runtime and disk use;
- [ ] confirmation that no other host owns the same output namespace or
      donor/window arm.

If any item is missing, stop before the long run. That is a useful fast failure,
not a reason to invent a local exception.
