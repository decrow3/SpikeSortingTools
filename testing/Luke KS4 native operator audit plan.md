# Luke KS4 native voltage-registration operator audit

## Status

Executed on 2026-08-31 as a parallel, sorter-independent operator
characterization. The preregistered native-operator and zero-tax gates failed,
so advancement to a supplied-`dshift` sort is not authorized. The bounded
smoothness companion was also unvalidated because no recurrent family met the
frozen eligibility rule. The edge challenge and supplied-trajectory sort were
stopped because they cannot rescue a failed necessary operator condition.

See
[`luke_20250804_ks4_native_operator_audit_result.md`](../docs/luke_20250804_ks4_native_operator_audit_result.md)
and the machine-readable
[`final_decision.json`](outputs/luke_ks4_native_operator_audit/final_decision.json).

## Relationship to the active validation ladder

This audit is **not the next load-bearing pipeline gate** and must not displace
the active order in
[`luke_pipeline_stage_local_validation_strategy.md`](../docs/luke_pipeline_stage_local_validation_strategy.md):

1. complete the motion-estimator bakeoff on held-out motion observables;
2. demonstrate benefit from coordinate-only motion application;
3. only then consider voltage interpolation as a separate operator.

The work here is parallel, CPU-scale characterization on already-open discovery
templates and small background arrays. It must not consume the raw-data, GPU,
or review resources allocated to the estimator bakeoff. A result from this
audit can characterize or reject an application operator, but it cannot
authorize a voltage warp by itself. The conditional supplied-trajectory sort
also requires a separately qualified motion estimate and completion of the
coordinate-only gates.

## Primary question

Can Kilosort 4's native post-whitening rigid voltage-registration operator undo
plausible physical motion in Luke better than leaving the motion uncorrected,
after accounting for the intrinsic cost of enabling the KS4 registration
operator?

The experiment also asks two mechanistic questions:

1. Does the placement of interpolation relative to KS4 preprocessing explain
   any benefit?
2. Do differences between the KS4 and SpikeInterface spatial kernels explain
   the historical failure of the externally warped binary?

Neither mechanistic contrast is itself required to advance. The essential gate
is useful native recovery relative to both the moved/no-correction condition and
the stationary reference.

## Motivation

The rejected historical path was approximately:

\[
X \rightarrow \text{SpikeInterface voltage interpolation}
\rightarrow \text{materialized binary} \rightarrow \text{KS4 preprocessing}.
\]

That is not the same operation as native KS4 registration. In the installed
`kilosort==4.0.27` implementation, each batch is channel-centered, median-CAR
referenced, and high-pass filtered before whitening and drift are applied
together as:

```python
X = (M @ whiten_mat) @ X
```

The CAR stage contains a channel median, so the complete preprocessing path is
not a purely linear operator that can be rearranged algebraically. Let `P`
denote the exact channel-centering, CAR, and temporal-filtering path and let `W`
denote a frozen whitening matrix. The native branch is then described
schematically as:

\[
M_d W P(X),
\]

whereas applying the same spatial matrix before KS4 preprocessing is:

\[
W P(M_d X).
\]

KS4 4.0.27 also constructs:

\[
iK_{xx}=(K_{xx}+0.01I)^{-1}.
\]

At zero displacement, the motion matrix is therefore:

\[
M_0=K_{xx}(K_{xx}+0.01I)^{-1},
\]

not the identity. Enabling motion correction can consequently impose a spatial
regularization or smoothing cost even when `dshift == 0`. The stationary
zero-shift arm measures that cost directly on Luke.

## Scope and invariants

The initial audit must keep the following fixed across arms:

- released `kilosort==4.0.27` behavior;
- the same full physical probe geometry and channel ordering;
- rigid displacement only: one displacement per KS4 batch;
- the same float32 source voltages;
- the same exact KS4 channel-centering, CAR, and high-pass implementation;
- one frozen whitening matrix estimated once from the prespecified unwarped
  reference data;
- the same donor templates, background snippets, imposed displacement values,
  and scoring windows;
- no intermediate int16 export;
- no pre-cropping before preprocessing or spatial interpolation;
- separate reporting for interior and edge-affected contacts;
- no re-estimation or tuning of the motion trajectory between arms.

Whitening must not be re-estimated after each warp. Otherwise, the comparison
would mix operator placement with changes in the estimated whitening solution.

The first audit does not run clustering and does not attempt to optimize a
motion estimator. It evaluates the application operator under known motion.

Any successful result is conditional on the forward generators' shared
assumption that the waveform varies sufficiently smoothly with physical depth.
The bounded empirical smoothness companion below is therefore required before
an operator-level pass can authorize a supplied-trajectory sort.

## Input material

Reuse the sealed discovery material and machinery from:

- [`luke_injected_ground_truth_pilot.py`](luke_injected_ground_truth_pilot.py)
- [`luke_injected_ground_truth_benchmark.py`](luke_injected_ground_truth_benchmark.py)
- [`luke_synthetic_motion_residual_lab.py`](luke_synthetic_motion_residual_lab.py)
- [`luke_interpolation_implementation_audit.py`](luke_interpolation_implementation_audit.py)

The existing donor templates and real Luke background snippets remain discovery
material. Prospective holdout material must remain sealed during operator
development.

## Forward imposed-motion model

Do not generate motion with the exact regularized KS4 operator that will later
be asked to undo it. That would give KS4 an inverse-crime advantage and answer
whether KS4 approximately inverts itself rather than whether it can undo
plausible physical motion.

For the primary screen, reuse the multiple forward generators already defined
in `luke_synthetic_motion_residual_lab.py`:

- SpikeInterface kriging, `p=2`, `sigma_um=10`;
- SpikeInterface kriging, `p=2`, `sigma_um=20`;
- four-neighbor inverse-distance weighting.

These generators represent different assumptions about the waveform between
contacts and none is the exact KS4 4.0.27 matrix, whose kernel definition,
regularization, and normalization differ.

Use known rigid displacements that span sub-contact and one-contact movement.
The prespecified starting grid is:

```text
0, +/-1, +/-2, +/-4, +/-6, +/-10, +/-20 um
```

Both displacement signs must be included because the four-column geometry and
probe edges need not behave symmetrically.

For each moved condition, apply the forward generator first and then apply the
candidate inverse correction. Score recovery against the original stationary
waveform. Applying an inverse matrix directly to an unshifted waveform measures
interpolation cost, not motion recovery.

Observed donor waveforms from different inferred depth states may be added as a
bounded companion analysis. They are not substituted for exact synthetic
ground truth because same-neuron identity and biological stability across depth
states remain uncertain.

## Bounded empirical waveform-versus-depth companion

The synthetic audit can rank operators only under its assumed continuous
waveform models. Run a small discovery-only companion to test whether actual
Luke waveform families are at least compatible with a smooth depth-dependent
description.

Freeze at most six high-confidence recurrent unit families before evaluating
smoothness. A family is eligible only when it has:

- independent recurrent events already labeled as neural;
- at least three populated inferred depth states spanning a nonzero depth
  range;
- stable polarity and gross morphology;
- no overlap with prospective holdout material.

For each eligible family, hold out each interior depth state in turn. Predict
its amplitude-normalized waveform from the two bracketing depth states using a
prespecified linear depth interpolation; do not select an interpolation kernel
per family. Compare held-out prediction error with the family's empirical
same-state recurrent variability and with a nearest-observed-state baseline.

Report held-out residual, cosine, amplitude error, peak-depth error, and
pairwise separability. Do not use these outcomes to tune the synthetic forward
generators. If fewer than three families qualify, or if held-out interpolation
is not consistently better than the nearest-state baseline within the
prespecified tolerance, label waveform smoothness **unvalidated**. The operator
audit may still report a conditional ranking, but it cannot authorize the
bounded supplied-trajectory sort.

## Experimental arms

| Arm | Input state | Correction path | Purpose |
| --- | --- | --- | --- |
| `stationary_no_correction` | Stationary | Exact KS4 preprocessing, no motion matrix | Reference ceiling |
| `stationary_ks4_d0` | Stationary | Native KS4 path with `dshift=0` | Measure the intrinsic KS4 regularization tax |
| `moved_no_correction` | Forward-moved | Exact KS4 preprocessing, no inverse | Measure the motion penalty |
| `moved_ks4_native_inverse` | Forward-moved | `M_d @ W @ P(X)` | Test native KS4 recovery |
| `moved_ks4_external_order_inverse` | Forward-moved | `W @ P(M_d @ X)` using the identical KS4 matrix | Isolate operator placement/order |
| `moved_si_inverse` | Forward-moved | Prespecified SpikeInterface inverse path | Isolate kernel, regularization, and normalization differences |

The sign convention for forward motion and inverse `dshift` must be validated
with a one-template, one-displacement test before the full factorial is run.
The direction is correct only if the inverse moves the recovered peak toward
the stationary reference depth.

## Exact KS4 state required

The native operator must call or faithfully reuse the installed KS4 4.0.27
implementation rather than reimplementing its formula approximately.

For the rigid experiment, construct and record at least:

- batch-indexed `dshift` with shape `(n_batches, 1)`;
- one rigid `yblk` entry in the convention expected by KS4;
- `Kxx` and `iKxx` from the exact probe geometry and `sig_interp`;
- the frozen `whiten_mat`;
- KS4 batch size, padding, sample-to-batch mapping, and time reference;
- the explicit displacement sign convention.

Replacing only `ops["dshift"]` is insufficient unless the remaining
registration state and batch alignment have already been initialized
consistently.

## Primary waveform and voltage metrics

Compute paired metrics against the original stationary reference for every
template, background, forward generator, displacement, sign, and arm:

- amplitude retention and absolute amplitude error;
- centered waveform cosine;
- scaled residual fraction;
- recovered peak channel and depth error;
- detection statistic or peak SNR under the frozen preprocessing path;
- local and global noise RMS;
- local noise covariance change;
- zero or unsupported-signal fraction;
- interior-versus-edge results;
- positive-versus-negative displacement asymmetry.

Cosine, apparent smoothness, and scalar SNR are not sufficient. Spatial
interpolation can reduce both signal and noise while erasing dimensions that
distinguish nearby units.

### Frozen detection metric

The primary detection metric is **reference-template matched-filter SNR**, not
KS4's learned or universal template detector. For each donor, use its stationary
reference waveform after the exact frozen `P` and `W` path as a fixed matched
filter over a prespecified temporal and channel support. Define the event score
as its inner product with that fixed unit-norm filter. Define noise as the robust
standard deviation of the same filter score at prespecified event-free times in
the paired real background. The reported SNR is event score divided by that
noise estimate.

The filter, support, event-free samples, centering convention, and robust-scale
estimator must be frozen before arm evaluation. This statistic measures
retention of a known discriminative waveform direction; it must not be
described as an exact reproduction of KS4 detection.

## Template separability and signal-to-noise geometry

For nearby donor-template pairs, add two prespecified discriminability metrics.

### 1. Nearest-neighbor template-distance retention

After the frozen KS4 preprocessing/whitening path, calculate pairwise distances
between amplitude-normalized templates. For each reference template, identify
its nearest competing template using only the stationary reference arm. Freeze
those pairs before evaluating correction arms.

Report:

- corrected/reference distance ratio for each frozen pair;
- median and lower-tail distance retention;
- the fraction of pairs whose distance shrinks beyond the prespecified
  tolerance;
- changes in nearest-neighbor identity or rank.

Pair selection must not be recomputed per arm.

### 2. Noise projected onto discriminative directions

For each frozen template pair, let the stationary reference difference vector
define the discriminative direction. Project real background noise from each arm
onto that fixed direction and report:

- retained signal separation along the reference direction;
- projected noise standard deviation;
- the resulting separation-to-noise ratio;
- change relative to `stationary_no_correction` and
  `moved_no_correction`.

Using a fixed reference direction prevents an interpolated arm from appearing
better merely because its own smoothed representation defines an easier axis.

## Prespecified causal contrasts

### Intrinsic native registration tax

```text
stationary_ks4_d0 - stationary_no_correction
```

This measures the cost of the regularized KS4 spatial matrix in the absence of
motion.

### Uncorrected motion penalty

```text
moved_no_correction - stationary_no_correction
```

### Native recovery benefit

```text
moved_ks4_native_inverse - moved_no_correction
```

Signs must be expressed so that positive values consistently mean improvement
when summaries are produced.

### Remaining gap to the stationary ceiling

```text
moved_ks4_native_inverse - stationary_no_correction
```

### Ordering effect

```text
moved_ks4_native_inverse - moved_ks4_external_order_inverse
```

This is a mechanism contrast, not an absolute advancement requirement.

### Kernel/normalization effect

```text
moved_ks4_external_order_inverse - moved_si_inverse
```

Interpret this contrast only after verifying that preprocessing placement,
dtype, geometry, displacement, and edge policy match.

## Frozen whitening policy for the SI arm

The primary `moved_si_inverse` arm intentionally uses the same whitening matrix
estimated once from `stationary_no_correction`. This is not meant to reproduce
the natural end-to-end SI-warped pipeline, which would allow KS4 to estimate a
new whitening solution from the warped recording. The frozen matrix is required
to isolate differences in the spatial kernel, regularization, normalization,
and edge behavior.

An optional descriptive `moved_si_pipeline_rewhitened` arm may later reproduce
the ecological external pipeline with whitening re-estimated after SI
interpolation. It is not part of the six-arm causal audit, must not enter the
primary gate, and cannot be used for the kernel-only contrast.

## Primary aggregation and frozen thresholds

Reuse the robust aggregation already implemented in
`luke_synthetic_motion_residual_lab.py`. The unit of aggregation is one paired
case: template, background, and signed displacement. Within each forward
generator, take the median paired delta across all prespecified cases. Across
generators, use the least favorable generator median:

- maximum generator-median delta residual;
- maximum generator-median delta absolute amplitude error;
- minimum generator-median delta cosine.

For `moved_ks4_native_inverse` relative to `moved_no_correction`, the frozen
primary screen thresholds are exactly:

```text
worst_generator_median_delta_residual <= -0.005
worst_generator_median_delta_absolute_amplitude_error <= 0.005
worst_generator_median_delta_cosine >= 0.0
```

Do not replace the worst-generator summary with a pooled median after results
are inspected. Report the same statistics separately by displacement magnitude,
sign, template, background, and edge status as robustness strata, but do not
select a favorable stratum as the headline result.

For the zero-shift tax comparison, calculate the same three paired metrics for
`stationary_ks4_d0` relative to `stationary_no_correction`. Native recovery must
exceed any adverse zero-shift change metric by metric. For residual specifically,
the tax-adjusted improvement must still clear `0.005`; equivalently, the
generator-median reduction in moved residual minus the stationary median
zero-shift residual increase must be at least `0.005` for every generator.

## Advancement gate

Advance to a bounded supplied-`dshift` KS4 sort only if all of the following are
true on the prespecified discovery screen:

1. Native KS4 correction improves the forward-moved condition relative to
   `moved_no_correction` under the frozen worst-generator aggregation and exact
   residual/amplitude/cosine thresholds above.
2. The native recovery benefit is meaningfully larger than the degradation
   measured in `stationary_ks4_d0` relative to
   `stationary_no_correction`, including the tax-adjusted residual threshold
   above.
3. The corrected condition remains acceptably close to the stationary reference
   in amplitude, waveform residual, cosine, peak depth, and detection statistic.
4. Nearest-neighbor template separability and projected separation-to-noise do
   not show a material lower-tail collapse.
5. Interior contacts pass separately; success driven by edge extrapolation is
   not acceptable.
6. Results are not dependent on one displacement sign or one donor template.
7. The empirical waveform-versus-depth companion supports the smoothness
   assumption; an unvalidated companion permits only a conditional operator
   result, not advancement to a supplied-trajectory sort.

The native-order arm does **not** have to outperform the external-order KS4 arm
to advance. If both work, the historical failure is more likely explained by
the old SI kernel, normalization, border behavior, int16 materialization, or
repeated conditioning. The native-versus-external contrast explains mechanism;
it does not define usefulness.

All remaining numeric tolerances, including separability and empirical
smoothness thresholds, must be frozen in the implementation configuration
before the full results are inspected.

## Bounded run size and compute plan

The primary run is capped at:

- six sealed donor templates;
- three prespecified real Luke discovery backgrounds;
- three forward generators;
- thirteen total signed displacement values, including zero;
- the six causal arms above.

A rectangular upper bound is `6 * 3 * 3 * 13 * 6 = 4,212` waveform-arm
evaluations. Stationary arms and zero displacement do not need to be duplicated
across forward generators, so the implementation should execute fewer unique
cases while retaining a rectangular results table if useful. These are small
array operations: the primary audit must not run a sorter, open prospective
holdout data, materialize a recording, or require a GPU.

Before the full run, execute a one-template, one-background dry run to validate
matrix orientation, displacement sign, batch mapping, output hashes, peak
memory, and projected runtime. Record that estimate in the frozen configuration.
If the full CPU run would compete with the motion-estimator bakeoff, defer it.

## Failure interpretation

If native KS4 correction cannot improve known rigid displacement beyond the
uncorrected condition after paying its zero-shift tax, voltage registration is
difficult to justify for Luke without a substantially different forward model
or spatial operator.

If correction improves individual-waveform metrics but collapses pairwise
separability or projected separation-to-noise, apparent waveform preservation
is not sufficient to advance.

If native and external ordering differ strongly with the same KS4 matrix, the
application point is causal.

If native and external ordering are similar but both outperform SI, the KS4
kernel, regularization, normalization, edge behavior, or prior materialization
path is the more likely explanation.

If all inverse arms fail for one or more forward generators, Luke may violate
the assumed smooth waveform-versus-depth relationship strongly enough that
spatial interpolation is not reliable.

## Conditional next stage: bounded supplied-trajectory KS4 sort

Only after the operator gate passes:

1. Freeze one conservative rigid trajectory from an already qualified motion
   estimate.
2. Stabilize implausible jumps before application, using a prespecified rule.
3. Map the trajectory explicitly to KS4 batch centers.
4. Run paired bounded sorts with no motion and supplied rigid native motion,
   leaving all downstream detection and clustering settings unchanged.
5. Evaluate reviewed-event recovery, injected-event recovery, unit continuity,
   duplicate burden, waveform fidelity, and edge effects.
6. Do not introduce nonrigid correction until rigid native registration passes
   both the operator and bounded-sort gates.

## Deferred and parallel work

The following questions are useful but must not enlarge the first operator
audit:

- broader waveform-family smoothness mapping beyond the capped six-family
  companion;
- observed donor-depth analyses that require new identity adjudication or
  prospective material;
- tuning `sig_interp` by actual matrix behavior rather than nominal sigma;
- nonrigid supplied trajectories;
- prospective holdout validation;
- a KS2 comparison as a parallel sorter strategy that does not answer the
  native-KS4 operator question.

## Expected artifacts

The implementation should write a self-contained output directory containing:

- a frozen JSON configuration and software-version record;
- hashes of donor templates, backgrounds, probe geometry, whitening matrix, and
  displacement arrays;
- serialized KS4 and SI matrices for every unique displacement;
- per-case waveform and voltage metrics;
- per-pair separability and projected-noise metrics;
- summaries stratified by generator, displacement magnitude/sign, template,
  and edge status;
- a machine-readable gate decision;
- figures showing recovery curves, zero-shift tax, pairwise distance retention,
  projected separation-to-noise, and edge asymmetry.

## References

- Pachitariu M, Sridhar S, Pennington J, Stringer C. [Spike sorting with
  Kilosort4](https://www.nature.com/articles/s41592-024-02232-7). *Nature
  Methods* (2024).
- [Kilosort 4 documentation](https://kilosort.readthedocs.io/en/latest/).
- [Kilosort `datashift.py`](https://github.com/MouseLand/Kilosort/blob/main/kilosort/datashift.py).
- [Kilosort `io.py`](https://github.com/MouseLand/Kilosort/blob/main/kilosort/io.py).
- [SpikeInterface motion preprocessing](https://github.com/SpikeInterface/spikeinterface/blob/main/src/spikeinterface/preprocessing/motion.py).
- Garcia S et al. [A modular implementation to handle and benchmark drift
  correction for high-density extracellular
  recordings](https://pmc.ncbi.nlm.nih.gov/articles/PMC10897502/).
