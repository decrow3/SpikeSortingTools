# Luke bounded pinned-AIND downstream result

> **SCOPE NOTE — 2026-09-03.** This bounded comparison supports retaining rescue
> as the operational comparator instead of advancing the tested AIND branch. It
> does not establish that rescue is better than the historical legacy pipeline,
> that its extra KS-good units are biological neurons, or that it handles motion
> better. Those questions require the current known-truth plan.

## Technical summary

The bounded downstream experiment does **not** support replacing the frozen
rescue graph with the pinned AIND preprocessing branch or advancing that branch
to a full-session Luke sort. Production-like AIND preprocessing with Kilosort
CAR enabled exactly tied rescue on total sealed-event recovery (470/720), but
the tie concealed a probe interaction: AIND lost four net events on imec0 and
gained four on imec1. It improved refractory and coincidence diagnostics, but
did so with 1.82% fewer assigned spikes, a median paired deficit of 15 KS-good
units, no pooled continuity improvement, and worse normalized nearby-similar
good-pair burden in four of six probe/window cells.

The mechanistic CAR-off ablation was not a better challenger. It recovered
467/720 sealed events, assigned 0.82% fewer spikes than rescue, and produced the
experiment's only residual-supported redundant-template hypothesis. That pair
was artifact-enriched, so the sidecar remains useful as an annotation, but it
does not authorize merging or voltage intervention.

The operational decision is therefore:

> **Retain the frozen rescue graph as Luke's downstream reference. Do not run a
> full-session pinned-AIND challenger from these settings.**

This is a downstream decision, not a contradiction of the preprocessing-only
25/25 result. The upstream AIND branch improves bounded voltage nuisance and
preservation metrics, but those gains did not translate into a superior sorting
outcome against the now-replicated rescue baseline.

## Primary endpoint-family review

| Family | Pinned AIND + KS CAR versus rescue | Interpretation |
|---|---|---|
| Sealed-event recovery | 470/720 versus 470/720; 10 gains and 10 losses | Aggregate tie with an imec0/imec1 interaction, not general improvement |
| Detection expansion | 3,789,907 learned detections and 3,767,376 final spikes versus 3,878,835 and 3,837,354 | 2.29% fewer learned detections and 1.82% fewer final spikes |
| Refractory behavior | Favorable in 6/6 cells; median paired relative change -35.9% | Real and fully consistent improvement, although absolute violation fractions were already small |
| Coincidence | Favorable in 6/6 cells; median paired relative change -4.6% | Real and fully consistent improvement |
| Continuity | Pooled stable-good fraction 87.10% versus 87.31%; 2 favorable, 1 tied and 3 unfavorable cells | Essentially neutral overall; late-window losses offset T2 gains |
| Duplicate burden | Normalized similar-pair burden favorable in 2/6 and unfavorable in 4/6 cells; median paired relative change +17.0% | The lower unpaired raw median is misleading; AIND did not solve the similar-template problem |
| Residuals | No residual-supported redundant pair in AIND+CAR or rescue | No differentiating evidence; absence of support is not proof that all pairs are distinct |

The secondary diagnostics point in opposite directions. AIND+CAR had a median
paired deficit of 15 KS-good units (-10.2%) and fewer good units in four of six
cells, including all three imec1 windows (-15, -15 and -16). Median good-unit
contamination improved or tied in every cell, but contamination was frequently
zero and remains secondary. The median-of-sort median good-unit firing rate was
3.15 Hz for AIND+CAR and 3.10 Hz for rescue; the corresponding median p10/p90
rates were 0.65/12.67 Hz and 0.64/12.27 Hz. The yield difference is therefore
not explained by a large tail of near-silent rescue units.

## The KS-CAR ablation does not identify a better AIND configuration

AIND+CAR recovered three more sealed events than AIND without CAR (470 versus
467). The two AIND conditions agreed on 465 recovered events; CAR-on uniquely
recovered five and CAR-off uniquely recovered two. CAR-on generally improved
refractory behavior, while CAR-off usually had slightly lower coincidence and
similar-pair burden. Neither condition dominated the other, and CAR-off created
the only residual-supported redundant-template hypothesis. If AIND were ever
revisited, the production-like CAR-on condition remains the more defensible
starting point, but neither bounded result beats rescue.

## Artifact annotation localizes the one residual-supported pair

The CAR-off imec0 T3 pair 199/201 had 97 one-to-one coincident events. Of those
events, 51.5% fell within 0.5 ms of the >500 µV sidecar claim samples and 60.8%
fell within 2 ms. In contrast, only 6.8% and 10.0% of the two units' individual
spikes were within 0.5 ms. This is strong artifact enrichment of the coincident
subset, but the sidecar remains observational: it neither proves causality nor
authorizes deletion, merging, blanking changes or claim exclusion.

## Scope, metric definitions and design

The result covers two probes, three independently frozen windows per probe and
three sorter conditions, for 18 sorts. Each condition covers 600 seconds per
probe and 720 sealed events total. All sorts used the frozen rescue Kilosort
settings, internal 300 Hz high-pass, motion correction off and
`skip_kilosort_preprocessing=false`. The production-like AIND condition used
external phase shift, 300 Hz high-pass, frozen AP191 removal, global median
reference and KS CAR on. The CAR-off condition differed from it only in
`do_CAR`.

The comparison is of the complete pinned AIND preprocessing branch, not CMR
alone. Unlike rescue, it omitted the bilateral ±500 µV blanker and removed
AP191 rather than interpolating it. The artifact sidecar was annotation-only.

Continuity is the fraction of KS-good units with spikes in at least 90% of four
or eight equal-duration 30-second bins. Duplicate burden is the count of nearby
similar good-good template pairs normalized per 100 KS-good units. Endpoint
families are correlated and are not treated as statistically independent
replicates; the review reports paired direction and magnitude across the six
fixed probe/window cells rather than a pooled p-value.

## Data-quality and robustness audit

The final score table has the expected 18 unique probe/window/condition rows,
all six paired cells are complete, and all sort manifests share config digest
`98b676c514e7229da3bbf042919983805d9f80c1f2676cd065e0ba063ac41434`.
Within every probe/window, AIND CAR-on and CAR-off used the same prepared
recording and differed in sorter parameters only by `do_CAR`. Saved critical
settings were `nblocks=0`, `highpass_cutoff=300`, and the requested CAR state.

The review found and corrected three scoring-edge defects before making the
decision:

1. Kilosort exported 0-3 final spikes beyond the half-open recording boundary
   in affected sorts. The scorer now removes and reports them synchronously
   across time, cluster and depth arrays. Twelve of 11.41 million final spikes
   were removed across the panel; this is too small to change any conclusion.
2. A one-frame sampling-rounding difference created a ninth, nearly empty bin
   in some nominal 240-second continuity windows. Continuity now uses exactly
   four or eight equal-duration bins; the prior 0.8889 presence artifact is no
   longer used.
3. Per-event recovery vectors were assigned to filtered DataFrames by pandas
   index rather than position, producing NaNs in the annotation CSVs. Summary
   recovery rates were already computed from the correct vectors, but the CSVs
   were regenerated with complete positional Boolean labels before the
   gained/lost event analysis.

An empty coincident-event subset in the residual audit is now recorded as
"evidence unavailable" rather than raising or being counted as duplicate
support. Two aggregate residual-improvement values remain structurally missing
because no usable residual events existed; the discrete supported-pair count is
complete. Eight targeted tests pass after these repairs.

## Limitations and remaining uncertainty

- These are 120/240-second bounded sorts, not full-session stability tests.
- One of six sealed windows per probe was omitted by the frozen bounded plan.
- KS-good labels and contamination are sorter diagnostics, not biological
  ground truth.
- Similar-template counts depend on the fixed 0.8 similarity and 100 µm rules;
  normalization removes good-count scale but not all template-count coupling.
- Residual audits evaluate at most the top three similar good-good pairs per
  sort and do not authorize automatic curation.
- The experiment cannot attribute any effect to CMR alone because blanking,
  high-pass placement and AP191 handling also differ.

## Recommended next steps

1. Keep the frozen rescue graph as the production/downstream Luke reference.
2. Do not spend the compute on a full-session pinned-AIND sort under either CAR
   condition.
3. Preserve the AIND preprocessing-only result as evidence that nuisance
   metrics can improve without guaranteeing a better sorter outcome.
4. Treat preprocessing as conditionally frozen. Move the primary testing effort
   to motion-estimator validation, then coordinate-only motion application and
   bounded sorter experiments.
5. Apply stage-local validation: an estimator must win on held-out motion
   observables before sorting; a coordinate field must help before voltage
   resampling; and merging, peeling or suppression must have residual-supported
   evidence rather than merely reducing spikes or units.
6. Retain a rescue-preserving AIND-style hybrid only as an optional targeted
   follow-up if later downstream evidence exposes a preprocessing-specific
   weakness. It is not the next priority.

The cross-stage strategy and authorization gates are recorded in
[`luke_pipeline_stage_local_validation_strategy.md`](luke_pipeline_stage_local_validation_strategy.md).

Machine-readable evidence and Markdown mirrors of every result/audit artifact
are indexed in
[`testing/outputs/luke_aind_downstream_bounded_endpoint_review/README.md`](../testing/outputs/luke_aind_downstream_bounded_endpoint_review/README.md);
the complete sort and per-condition score outputs remain under the frozen
experiment root.
