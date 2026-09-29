# 0017 — Time-resolved completeness is a screening sidecar, never an automatic mask

**Status:** Adopted 2026-09-28  
**Evidence:**
[`luke_dropout_anchor_and_detection_lineage_20260928.md`](../luke_dropout_anchor_and_detection_lineage_20260928.md),
[`luke_cluster553_local_template_rescue_result_20260928.md`](../luke_cluster553_local_template_rescue_result_20260928.md), and
[`luke_completeness_timeline_sidecar_result_20260928.md`](../luke_completeness_timeline_sidecar_result_20260928.md)

## Decision

Every completed production QC pass writes two additive identity-bound outputs:

1. standard per-unit quality metrics, definitions, flags, and policy; and
2. `amplitude-completeness-timeline-v2`, including every curated unit,
   physical-time fit windows, measurement status, screening nominations,
   explicit unsupported units, policy, summary, request, and receipt.

The timeline is triage and provenance. `deterioration_screen=true` nominates a
window for review; it never alters spikes or labels, automatically masks an
interval, imputes missing spikes, or establishes biological event loss. A row
absent for a unit/time means unsupported, not zero missingness. A 50% result is
boundary-censored, not a measured severity. Fits whose two stored estimators
disagree by more than five percentage points are poor-fit rows, not reliable
missingness values.

## Evidence for the restriction

Independent anchors confirm material event loss in only two of seven fitted
deterioration cases. The sidecar correctly nominates the confirmed cluster-21
and cluster-553 failing windows, but also nominates cluster 452, whose anchored
loss is only about 2.2 points. On the complete imec0 rescue output, 396/710
units have no fit window; among 26,227 fitted windows, only 9,204 are trustworthy
measurements, 14,394 are boundary-censored, and 2,629 fail estimator agreement.

Cluster 553 demonstrates why the sidecar must not imply a repair. Its missed
times retain a weaker population-median waveform, but a frozen local
peak/template detector, score-only calibration, wider-lag calibration, wider
depth matching, and direct depth tracing do not yield a defensible recoverable
single-event signal. The pipeline reports this loss boundary rather than
fabricating spikes.

## Reopening condition

Automatic masking or imputation requires a separately frozen method with
known-truth or independent event validation, preserved healthy intervals,
acceptable contamination/refractory behavior, and held-out transfer. Better
fit values or more screening nominations alone cannot satisfy that condition.
