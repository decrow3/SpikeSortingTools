# Luke pipeline development strategy: stage-local validation

> **PRINCIPLE RETAINED; ACTIVE SEQUENCE SUPERSEDED 2026-09-03.** Stage-local
> validation remains the governing method. The specific priority order and
> status statements below are the 2026-08-31 snapshot; use
> [`pipeline_improvement_plan.md`](pipeline_improvement_plan.md) for current
> execution and decisions 0011–0015 for current evidence limits.

**Decision date:** 2026-08-31

**Applies to:** Luke pipeline development after the replicated full-session
rescue and bounded pinned-AIND downstream comparison

## Current decision

Preprocessing is conditionally frozen. The frozen rescue graph is the current
production and downstream reference; pinned AIND preprocessing is retained as
an independent, competent comparator. Do not reopen a broad reference/filter/
blanking search unless a downstream experiment demonstrates a reproducible,
preprocessing-specific failure.

This is not a claim that rescue is biologically optimal or universally best.
It is a resource-allocation decision: preprocessing no longer appears to be
the dominant limitation under the current evidence, while motion estimation,
motion application and sorter behavior retain clear unanswered questions.

Do not run a full-session pinned-AIND challenger under the tested settings. The
bounded comparison did not beat rescue. A rescue-preserving AIND-style hybrid
remains an optional targeted experiment, not a prerequisite for production and
not the next priority.

## Why preprocessing is no longer the leading search axis

Two substantially different preprocessing architectures now converge on
broadly similar bounded Kilosort 4 outcomes:

- rescue and production-like pinned AIND both recovered 470/720 sealed events;
- pooled continuity and good-unit firing-rate distributions were similar;
- AIND improved refractory and coincidence diagnostics consistently;
- rescue retained better yield and lower normalized similar-template burden;
- neither architecture showed a general collapse;
- the frozen rescue graph already produced strong full-session results on both
  Luke probes.

The remaining differences are tradeoffs rather than evidence that another
broad preprocessing sweep is likely to reproduce the large gains obtained by
fixing the earlier pipeline failures. The inference is bounded and
sorter-specific: another sorter may interact differently with the two voltage
representations.

## Governing methodological principle

> **A stage must be validated with an observable that the next stage cannot
> rescue or manufacture. A better final sort does not retroactively prove that
> every upstream choice was correct.**

This is the stage-local validation rule. It prevents a downstream algorithm
from compensating for an upstream defect in a way that improves headline
metrics while creating hidden pathology. For example, a claim mask can suppress
an upstream detection explosion without establishing that the voltage or
motion representation is correct.

| Stage | Stage-local observable | What authorizes advancement |
|---|---|---|
| Preprocessing | Injected and recurrent-waveform preservation; nuisance, ringing and common-mode behavior | Preserves known events while reducing the specific nuisance, before sorting |
| Motion estimation | Held-out raster residual; independent recurrent trajectories; field support and nearby-parameter stability | Improves held-out geometry across probes/windows without using sorter labels |
| Motion application | Coordinate-only improvement; known-waveform and operator preservation; edge and zero-fill behavior | Coordinate correction helps first; voltage resampling separately passes preservation gates |
| Detection | Sealed-event recovery; learned/final detection expansion; artifact proximity | Recovers known events without disproportionate false-event or coincidence expansion |
| Clustering/template learning | Continuity, residuals, normalized duplication, refractory structure and morphology | Improves separation and stability without relying on KS-good count alone |
| Merging/peeling/suppression | Pairwise residual reduction, CCG/refractory evidence and retained sealed events | Removes demonstrated redundancy without merely reducing spikes or units |

Metrics from adjacent stages may be reported together, but correlated endpoint
families must not be described as statistically independent evidence.

## Active development order

1. **Keep preprocessing fixed.** Use rescue as the default reference and
   pinned AIND only as a fixed comparator or when testing a specific interaction.
2. **Bake off motion estimators without voltage resampling.** Determine whether
   any estimator improves held-out motion observables beyond the current DREDGE
   fit.
3. **Test motion application separately.** Establish benefit from corrected
   coordinates before considering voltage interpolation. A good field does not
   automatically authorize a warp.
4. **Run bounded sorter and parameter experiments.** Hold the input graph fixed
   while testing detection thresholds, template learning, peeling, merging,
   duplicate handling, whitening/CAR interactions and, where justified,
   alternative sorters.
5. **Use compensatory or suppressive mechanisms last.** Claim masks, aggressive
   duplicate suppression and unusual thresholds require an identified failure
   mechanism plus retained-event and residual evidence.

## Motion-estimation gate

The next load-bearing question is:

> Does a better motion estimator measurably improve held-out motion observables
> beyond the current DREDGE fit?

A candidate must lower held-out raster residuals, agree better with independent
recurrent-event trajectories, and remain supported under nearby parameters.
The effect should replicate across probes or windows, or have a documented
probe-specific explanation. Sorting output and KS labels must not select the
estimator. If no candidate clears these gates, the structured residual may
reflect non-motion variability or limitations of the observable rather than a
deficient estimator.

## Motion-application authorization ladder

1. Demonstrate that corrected coordinates improve motion-localized observables.
2. Demonstrate that coordinate correction improves bounded downstream sorting.
3. Test voltage interpolation as a separate operator.
4. Require injected/recurrent-waveform preservation, correct clocks, real
   boundary support and no pathological zero filling before a voltage warp can
   enter production.

Historical failure of the tested external warp therefore does not reject motion
handling in general. It rejects that field/application combination.

The subsequent exact KS4 4.0.27 native-operator audit also failed its
preregistered waveform, amplitude and zero-shift-tax gates. Do not advance that
operator to a supplied-trajectory sort. This result remains application-local:
the estimator bakeoff and coordinate-only ladder are still active. See
[`luke_20250804_ks4_native_operator_audit_result.md`](luke_20250804_ks4_native_operator_audit_result.md).

A narrower exact-identity/selective-correction hypothesis may be tested only as
a new operator experiment. The existing grid has no complete residual,
amplitude, and cosine crossover, so it cannot yet authorize a real trajectory
or bounded sort. See
[`Luke KS4 selective correction crossover plan.md`](../testing/Luke%20KS4%20selective%20correction%20crossover%20plan.md).

## Sorter experiment rules

- Start with the frozen rescue input and frozen metric-blind windows.
- Change one mechanism or a tightly bounded parameter family at a time.
- Prioritize sealed recovery, detection expansion, coincidence, refractory
  structure, continuity, residuals and normalized duplicate burden.
- Treat KS-good count and contamination as secondary diagnostics.
- Do not optimize polarity toward Yates; positive-dominant units must fail
  independent artifact, residual, morphology or stability checks before they
  are rejected.
- Require residual-supported evidence before merging, peeling or suppression is
  credited as beneficial.
- Advance to a full session only after a bounded candidate wins across endpoint
  families without boundary accumulation or hidden detection expansion.

### Next bounded sorter-architecture test

The next mechanism-matched alternative-sorter test is upstream Kilosort 2 on
the frozen unwarped rescue recording. It asks whether KS2's native batch
reordering and time-varying template tracking preserve identities that KS4
leaves fragmented or MUA, without spatial voltage interpolation. This is a
step-4 sorter experiment and does not validate a physical motion estimate or
advance the voltage-application ladder.

The test must begin with patched-source and batch-boundary integrity gates, then
progress from a 120-s integration smoke test to a frozen panel of six
non-overlapping 10-minute segments spanning supported-motion magnitude, input
anomaly, and support dropout. Full-session execution requires coherent native
tracking diagnostics, preserved clean KS4 control families, and several
residual/refractory-supported family wins replicated across motion segments.
See
[`Luke KS2 native tracking validation plan.md`](../testing/Luke%20KS2%20native%20tracking%20validation%20plan.md).

The shared central endpoint is identity through motion: whether a candidate
neuron's time-resolved waveform path approaches competing identities and is
split into temporally consecutive labels. The segment panel and frozen
time-ordered family analysis are specified in
[`Luke identity through motion segment panel plan.md`](../testing/Luke%20identity%20through%20motion%20segment%20panel%20plan.md).

## When to reopen preprocessing

Reopen preprocessing only for a specific, reproducible failure such as:

- known-waveform attenuation or morphology distortion before motion/sorting;
- artifact/ringing burden demonstrably seeding detections or templates;
- a probe- or sorter-specific failure that differs between rescue and pinned
  AIND under otherwise fixed conditions;
- an independently validated sorter whose performance is consistently limited
  by one voltage representation.

If sorter dependence becomes plausible, use a small fixed preprocessing ×
sorter matrix. Do not co-optimize both layers simultaneously. Rescue and pinned
AIND provide the two characterized baselines needed for that interaction test.

## Evidence links

- [`luke_20250804_full_probe_rescue_result.md`](luke_20250804_full_probe_rescue_result.md)
- [`luke_20250804_imec0_rescue_result.md`](luke_20250804_imec0_rescue_result.md)
- [`luke_20250804_aind_downstream_bounded_result.md`](luke_20250804_aind_downstream_bounded_result.md)
- [`luke_20250804_presort_motion_handoff.md`](luke_20250804_presort_motion_handoff.md)
- [`luke_20250804_rescue_status_and_test_plan.md`](luke_20250804_rescue_status_and_test_plan.md)
- [`luke_20250804_ks4_native_operator_audit_result.md`](luke_20250804_ks4_native_operator_audit_result.md)
