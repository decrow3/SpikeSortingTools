# Luke KS2 native waveform-state tracking validation

## Status

**Installation gate failed on 2026-08-31; the six-segment panel is not
authorized.** The pinned upstream source, CUDA build, deterministic fixtures,
and two rapid-motion smoke configurations were executed. The published
v2.0.2-minus grid created a complete 28-ms phase gap, while a coverage-aligned
grid removed that complete gap but retained a significant periodic detection
trough. See
[`luke_ks2_native_tracking_installation_result.md`](luke_ks2_native_tracking_installation_result.md).

This was designed as the next bounded alternative-sorter test on the frozen,
unwarped rescue recording. MATLAB is available: on 2026-08-31 the same
`/usr/local/bin/matlab` executable recorded by the successful KIASORT run
reported R2022b and passed both MATLAB and Parallel Computing Toolbox license
checks when run with normal host-ID visibility. The earlier failure occurred
only inside the restricted execution context and is not an installation
blocker.

The same audit confirmed licensed Signal Processing, Statistics and Machine
Learning, and Parallel Computing toolboxes; a visible NVIDIA RTX A5000 with
compute capability 8.6; driver 12.4; and Linux `mexa64` support.

Source provenance is now resolved to MouseLand tag `v2.0.2`, commit
`0ce102799e69b97e3364ae47b403a809712d7e15`, with clean tracked source and
frozen source/MEX hashes. Legacy KS2/KS2.5 trees under
`/home/huklab/Documents/NPX_pilot/` were not used.

The installation audit and integration smoke showed that source provenance is
not sufficient: the executed KS2 configurations did not satisfy the empirical
batch-boundary gate. No sorter-quality conclusion is drawn from either smoke
output.

## Primary question

> Does Kilosort 2's native waveform-state tracking and time-varying template
> machinery preserve neuronal identity through Luke's rapid motion without
> spatially interpolating the voltage?

This test is motivated by a specific residual KS4 failure mode: supported
events are usually present in the accepted KS4 event table, but many land only
in MUA or are distributed across temporally complementary fragments. The
predicted KS2 win is therefore not more clusters or more spikes. It is:

- one coherent KS2 identity corresponding to multiple KS4 fragments or a
  difficult KS4 MUA family;
- continuous temporal support through a known rapid-motion episode;
- smooth raw-waveform evolution across time;
- acceptable refractory, duplicate, merge, and residual behavior; and
- preservation of already-clean, stable KS4 families.

## Relationship to the active validation ladder

KS2 is a bounded sorter-architecture experiment under step 4 of
[`luke_pipeline_stage_local_validation_strategy.md`](../docs/luke_pipeline_stage_local_validation_strategy.md).
It does not advance or replace the independent motion-estimator and
coordinate-only application gates.

A KS2 win would support motion-aware template tracking as a useful architecture
for Luke. It would not validate a DREDGE trajectory, establish that KS2's latent
state is physical displacement in microns, or authorize voltage resampling.

## Why KS2 is a mechanism-matched challenger

The archived MouseLand KS2 documentation describes an approach that does not
explicitly register the voltage. KS2 summarizes each batch with waveform
templates, compares batches through a batch-similarity matrix, reorders batches
so adjacent batches have similar waveform states during fast drift, and then
tracks templates through incremental updates. The inferred one-dimensional
position is a waveform-state coordinate and is not guaranteed to be literal
probe displacement.

This is materially different from the rejected KS4 native voltage-registration
operator and from the historical externally warped SpikeInterface binary. It
directly targets identity preservation without requiring a spatial inverse for
undersampled extracellular voltage.

Primary references:

- [MouseLand: More on drift correction](https://github.com/MouseLand/Kilosort/wiki/3.-More-on-drift-correction)
- [MouseLand Kilosort warning about the KS2/2.5/3 batch-boundary bugs](https://github.com/MouseLand/Kilosort#readme)
- [MouseLand issue 594: spike holes at batching edges](https://github.com/MouseLand/Kilosort/issues/594)

## Frozen input and comparator

### KS2 input

Use the same accepted **unwarped rescue recording** that supplied the current
production KS4 reference. Keep all accepted recording-construction choices
fixed upstream, including:

- phase correction;
- artifact blanking and its sidecar/provenance;
- channel-191 handling;
- channel order, geometry, gain, dtype, sample rate, and clock origin; and
- no voltage motion correction or spatial interpolation.

Pass that recording to upstream KS2 and allow KS2 to perform its native
filtering, CAR, whitening, batch representation, reordering, and time-varying
template tracking. Do not precompute a KS4-like frontend and do not set
`skip_kilosort_preprocessing=true`.

### Comparator

The production comparator is the already accepted full-session **KS4
no-motion** sort. For every bounded interval, slice KS4 units and events from
that accepted sort. Do not run a fresh short KS4 sort, because doing so would
confound sorter architecture with recording duration and initialization.

### Execution architecture

Run the pinned upstream MATLAB KS2 entrypoint as the scientific implementation.
SpikeInterface may prepare an input binary and normalize outputs, but the exact
generated MATLAB config must be saved and upstream KS2 remains the source of
truth.

The installed SpikeInterface 0.102.1 wrapper is only a reference for the first
config. Its local source currently exposes:

```text
detect_threshold = 6
projection_threshold = [10, 4]
preclust_threshold = 8
whiteningRange = 32
momentum = [20, 400]
car = true
freq_min = 150
sigmaMask = 30
lam = 10
AUCsplit = 0.9
nPCs = 3
ntbuff = 64
NT = 64 * 1024 + ntbuff when unspecified
reorder = 1 (hard-coded by the wrapper)
```

These are reference values, not authority to run an unpatched source tree.
The final frozen config is the generated MATLAB `ops` after the installation
audit.

## Mandatory installation and provenance gate

Kilosort 2, 2.5, and 3 had two bugs that reduced detection during roughly 7 ms
around batch boundaries. MouseLand instructs users to use the `patch1` releases
and their new default `NT`/`ntbuff` settings. Because the failure is periodic
and subtle, source provenance alone is insufficient.

Before the smoke test, record:

- exact upstream KS2 release/tag and Git commit;
- SHA-256 hashes of the KS2 source tree manifest and all modified files;
- generated channel map and complete MATLAB config;
- MATLAB release, license status, Parallel Computing Toolbox status;
- CUDA toolkit, GPU model, driver, and KS2 CUDA binary build provenance;
- SpikeInterface version if it is used for input/output plumbing;
- input binary size/hash, dtype, channel count, sample rate, and time origin;
- `NT`, `ntbuff`, effective batch stride, and expected boundary phase; and
- the entrypoint command and complete stdout/stderr log.

The installation gate passes only when:

1. the pinned KS2 source is demonstrably a patched release or an audited
   equivalent;
2. its exact `NT`/`ntbuff` behavior is confirmed from executed code, not only a
   wrapper default;
3. a deterministic tiny test completes twice with identical output hashes or a
   documented nondeterminism envelope; and
4. the empirical batch-phase check below finds no material periodic detection
   trough.

MATLAB licensing does not currently block this gate. Run MATLAB with normal
host-ID visibility, as in the successful KIASORT execution; a sandbox-only
license failure must not be recorded as evidence that the host license is
inactive. Pin and audit a patched upstream KS2 source before scientific
execution. Do not silently substitute one of the unversioned legacy trees or
replace MATLAB KS2 with another sorter.

## Empirical batch-phase check

Run this check after every KS2 execution, including smoke, bounded, tuned, and
full-session runs.

For each spike time `s`, define batch phase using the executed KS2 batch stride
and clock:

\[
\phi_s = (s-s_0) \bmod S,
\]

where `S` is the effective stride and `s0` is recovered from the actual reader.
Bin phase at no coarser than 0.5 ms. Report pooled spikes and separately report
high-rate units, low-rate units, KS2 labels, depth quartiles, and quiet versus
motion epochs.

The primary statistic is the minimum event-rate ratio in the preregistered
boundary neighborhood relative to matched interior phase bins:

\[
R_{boundary} = \frac{\text{rate within the boundary neighborhood}}
                    {\text{median rate in matched interior bins}}.
\]

Use circularly shifted pseudo-boundaries that preserve each unit's spike train
as a null distribution. Freeze the boundary width from the executed patched
implementation before viewing the histogram; include the historical 7-ms
width as a secondary diagnostic.

The installation gate fails if either:

- the pooled boundary ratio is below 0.98 and below the 1st percentile of the
  pseudo-boundary null; or
- any preregistered major subgroup is below 0.95 with the same null criterion.

Always retain the phase histogram and exact phase calculation, even on a pass.

## Three-stage execution ladder

### Stage 0: installation fixture

Use a tiny deterministic synthetic binary with spikes deliberately placed at
interior and boundary phases. This is a software-integrity test, not biological
validation. Require output readability, clock agreement, expected channel map,
and a passing batch-phase check.

### Stage 1: rapid-motion smoke test

Run the existing 120-s rapid-motion interval beginning at 5910 s on the primary
Luke probe. Its purpose is to verify end-to-end execution and preservation of
native diagnostics. It is not a sorter-quality verdict and cannot authorize
tuning or a full-session run.

Required outputs include:

- standard Phy/NumPy outputs and the full internal `rez` structure;
- batch-by-batch waveform similarity matrix before and after ordering;
- original batch order, inferred order, and latent waveform-state position;
- time-varying template state sufficient to reproduce tracking;
- whitening and preprocessing state;
- per-batch support/template counts; and
- batch-phase audit tables and figures.

### Stage 2: bounded multi-segment scientific panel

Run the frozen six-segment panel in
[`Luke identity through motion segment panel plan.md`](Luke%20identity%20through%20motion%20segment%20panel%20plan.md).
It contains three non-overlapping 10-minute segments spanning a cleaner
relative-quiet, moderate, and large supported-motion gradient; one large-motion
plus input-anomaly interaction segment; and sustained-noise and support-dropout
controls. Each segment contains its original 120-s anchor for event-level
localization.

The panel was selected only from existing sorter-blind input and motion-
estimator features. Its exact boundaries and per-probe summaries are generated
by [`luke_motion_identity_segment_panel.py`](luke_motion_identity_segment_panel.py).
Do not change boundaries after viewing KS2 output.

Ten minutes provides roughly 275 default KS2 batch strides at 30 kHz. Confirm
adequate per-batch support on the smoke test before treating that duration as a
scientific comparison. Run each segment as its own contiguous recording; do not
concatenate separated intervals and allow KS2 to track across artificial joins.

Run exactly one default KS2 condition across the complete panel first. Complete
the batch-phase and native diagnostic review for every segment before any
family/yield review. Report segment-specific results and the ordered trend
across the three cleaner motion-gradient segments, with the motion-plus-anomaly
segment kept separate; do not let a strong high-motion win hide a quiet/control
regression.

### Stage 3: full selected recording

Authorize a full-session KS2 run only if the bounded default condition, or the
single allowed diagnosis-driven condition, passes every advancement gate.

## Native diagnostic gate

Preserve and score the batch-similarity/reordering diagnostic before examining
unit-family wins.

Report at minimum:

- number of batches and spikes/templates supporting each batch;
- missing or low-support batch fraction;
- adjacent-batch similarity in chronological and reordered sequences;
- lower-tail adjacent similarity, not only the median;
- largest adjacent discontinuity before and after reordering;
- path length through similarity space before and after reordering;
- ordering stability under a fixed split-half spike support perturbation; and
- correspondence of latent state with time, supported motion epochs, input
  anomaly epochs, and firing-rate support as descriptive diagnostics only.

A coherent diagnostic requires all of the following frozen conditions:

- at least 95% of batches meet the prespecified minimum template/spike support;
- reordered median adjacent similarity is no worse than chronological;
- reordered 5th-percentile adjacent similarity is no worse than chronological;
- the maximum reordered discontinuity does not increase by more than 10%; and
- split-half reordered positions have absolute Spearman correlation at least
  0.8 after allowing a global reversal.

If the default run fails only because batches are undersupported, the one
allowed tuning pass may increase batch duration. Do not lower detection
thresholds to make this gate pass.

## Unit-family comparison

Do not compare raw cluster counts or KS2/KS4 `good` labels as equivalent
biological outcomes. Build bidirectional, reversible unit-family links using
raw unwarped voltage and the same clock.

### Primary endpoints

- reviewed neural-event recovery;
- 1.5-ms refractory violation and contamination burden;
- raw-waveform early/middle/late stability;
- temporal presence in frozen 30-s bins, with 10-s bins as sensitivity;
- first/last-flank persistence;
- duplicate and near-zero-lag coincidence burden;
- event-centered residual explanation;
- bidirectional KS2-to-KS4 and KS4-to-KS2 family matching; and
- continuity specifically through the nested rapid-motion interval.

Waveform matching must use raw unwarped snippets under one shared extraction
path. A family win requires morphology, timing, refractory, residual, and
temporal-complementarity evidence; template cosine alone is insufficient.

### Targeted positive controls

Evaluate these before broad yield summaries:

1. KS4 unit 389, the strongest reviewed MUA problem case: 22 reviewed neural
   events, full-session presence, but 20.8% estimated contamination.
2. KIASORT-nominated family 46, whose events map across 16 KS4 units and whose
   naive KS4 target union has a 67.4% refractory burden.
3. KIASORT-nominated family 82, whose events map across 19 KS4 units and whose
   naive KS4 target union has a 75.4% refractory burden.

The positive-control question is whether KS2 yields one coherent identity with
clean union refractory behavior and smooth temporal waveform evolution—not
whether it merely captures more of the nominated events.

### Frozen negative controls

Before running KS2, select up to 12 already-excellent KS4 units using only the
accepted full-sort metrics:

- high 30-s temporal presence;
- low contamination and refractory burden;
- stable early/late raw templates;
- adequate event count in the bounded interval; and
- coverage across depth quartiles and firing-rate strata.

Save the unit IDs and selection table before KS2 family matching. KS2 must not
systematically split, merge, lose, or contaminate these controls.

## Primary family-win definition

A KS2 family counts as a real win over a KS4 fragment set only when all of the
following hold:

1. bidirectional event matching links it to the same raw-waveform family;
2. its presence spans both quiet flanks and the rapid-motion epoch;
3. the KS4-linked pieces are temporally complementary rather than predominantly
   duplicate;
4. KS2's 1.5-ms refractory burden is no worse than 1 percentage point above the
   best defensible KS4 comparison and is below 2%;
5. the KS2 family explains more event-centered residual energy than every
   single KS4 fragment and than any KS4 union that passes the same refractory
   rule;
6. early/middle/late raw-template cosine remains at least 0.90 or stays within
   the frozen same-family recurrent-variability envelope; and
7. blinded waveform/CCG review does not identify a merge, artifact, or duplicate
   pathology.

Report borderline families, but do not count them toward advancement.

## Advancement gates for a full-session run

All gates are conjunctive.

### 1. Software integrity

- installation/provenance gate passes;
- empirical batch-phase gate passes;
- clocks, channel map, and output normalization are exact; and
- native KS2 diagnostics required by this plan are preserved.

### 2. No regression

- reviewed-event recovery is no worse than KS4 by more than 5 percentage
  points, with paired event outcomes reported;
- median refractory burden among matched clean families is no worse than KS4
  by more than 1 percentage point;
- at least 10 of the 12 frozen negative controls remain one-to-one defensible
  families; and
- no more than one negative control shows a large merge, split, or loss
  pathology after blinded arbitration.

### 3. Tracking evidence

- the native diagnostic gate passes; and
- failures are not concentrated at batch boundaries or low-support batches.

### 4. Mechanism-specific benefit

- at least three prespecified or independently discovered KS4 fragment/MUA
  families satisfy the complete family-win definition;
- wins replicate in at least two supported-motion segments;
- at least one win spans the 5910-s rapid-motion episode;
- fragmentation or ambiguity improvement is larger in supported high-motion
  segments than in the relative-quiet and control segments; and
- at least one win comes from the targeted positive-control set unless all
  three controls are shown not to be valid single-neuron hypotheses.

### 5. No large duplicate/merge pathology

- extra events survive raw-waveform, CCG, coincidence, and residual
  arbitration;
- the upper tail of near-zero-lag burden is not materially worse than KS4; and
- no apparent gain depends on naively unioning contaminated families.

Failure of any gate stops the full-session run. The result should still be
reported as a bounded architectural characterization.

## One allowed diagnosis-driven tuning pass

Do not tune before completing and freezing the default bounded analysis.

Allow at most one additional KS2 run, chosen by a deterministic decision rule:

- if more than 5% of batches fail the frozen support minimum, increase `NT` to
  the smallest prespecified patched-compatible value that doubles effective
  batch duration;
- otherwise, if diagnostic ordering is coherent but raw template identities
  systematically lag smooth state changes, change only the template-update
  momentum to one prespecified slower/faster candidate justified from the
  internal diagnostic;
- otherwise, do not tune.

Do not change detection threshold, projection thresholds, `AUCsplit`, `lam`,
`sigmaMask`, CAR, whitening range, or multiple mechanisms in this pass.
The tuned condition must clear the same gates and is reported alongside, not in
place of, the default result.

## Conditional within-KS2 ablation

If native KS2 passes the bounded advancement gates, run one bounded mechanistic
ablation:

```text
native reorder = 1  versus  identical config with reorder = 0
```

This ablation asks whether fast-drift batch reordering, rather than only slow
incremental template updating, explains continuity through Luke's 2–6-s motion
episodes. It must use the identical frozen segment panel, input binaries,
frontend, thresholds, random state where applicable, and family evaluation.

The ablation is explanatory, not an additional advancement requirement. If the
two conditions are similar, KS2 may still advance if the native architecture
passes all production gates.

## Frozen analysis order

1. Verify installation provenance, clocks, config, and batch phase.
2. Inspect native batch support and similarity/reordering diagnostics.
3. Score frozen negative controls.
4. Score the three targeted positive controls.
5. Perform blinded bidirectional family matching across every frozen segment.
6. Compute broad sorter-neutral endpoint summaries.
7. Apply the conjunctive advancement gates.
8. Only then decide whether the single tuning pass or full-session run is
   authorized.

## Required artifacts

Each run must emit:

- immutable run manifest and exact config;
- source/version/hash and hardware/software provenance;
- input and output hashes;
- complete logs and runtime/resource summary;
- batch-phase event table, null draws, summary, and figure;
- native batch similarity/order/support arrays and figures;
- normalized spike/unit tables with original labels retained;
- frozen positive/negative control tables;
- bidirectional family-edge and family-component tables;
- waveform, temporal-presence, CCG, coincidence, and residual metrics;
- blinded review decisions with reviewer-independent raw evidence; and
- a machine-readable gate decision naming every failed condition.

## Compute bound

The initial authorization covers one tiny installation fixture, one 120-s smoke
run, six non-overlapping 10-minute default scientific runs, and at most one
diagnosis-driven repeat of the same complete panel. It does not authorize a
full-session run, a broad threshold sweep, or a preprocessing-by-sorter
factorial.

Record wall time, peak host RAM, GPU model, peak GPU memory, temporary disk,
final disk, and estimated full-session cost after the smoke and bounded runs.

## Stop rules

Stop before scientific interpretation if:

- patched source provenance cannot be established;
- MATLAB/CUDA execution is not reproducible;
- the batch-phase gate fails;
- clocks or channel maps disagree with the accepted KS4 reference; or
- native similarity/order/template state cannot be preserved.

Stop before a full-session run if any bounded advancement gate fails. Do not
rescue a failure by lowering thresholds, externally warping voltage, forcing a
KS4 frontend, comparing label counts, or interpreting the latent state as
physical microns.
