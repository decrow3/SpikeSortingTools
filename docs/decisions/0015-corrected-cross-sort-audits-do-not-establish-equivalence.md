# 0015 — Corrected cross-sort audits show no confirmed detection difference, not equivalence

**Status:** Adopted 2026-09-03  
**Updates the current consequence of:**
[0010](0010-rescue-yield-is-relabelling-not-detection.md) and
[0011](0011-cross-sort-event-matching-and-detection-evidence.md)  
**Evidence:**
[`luke_20250804_rescue_unique_units_audit.md`](../luke_20250804_rescue_unique_units_audit.md),
[`luke_20250804_rescue_lost_units_audit.md`](../luke_20250804_rescue_lost_units_audit.md), and
[`luke_20250804_rescue_repartition_motion_audit.md`](../luke_20250804_rescue_repartition_motion_audit.md)

## Decision

The corrected v2 cross-sort audits support this bounded statement:

> On Luke0804 imec0, the null-controlled audit found no **confirmed**
> rescue-only detection and no **confirmed** legacy detection lost by rescue.
> It did not establish detection equivalence, biological-unit identity, or that
> either pipeline is better.

The exclusive cross-sort cohort is `+210 / -137` KS-good units. On the rescue
side, 208/210 units have spatially plausible event overlap with the complete
legacy sort that exceeds the circular-shift null; two are unresolved. On the
legacy side, 132/137 have corresponding support in the complete rescue sort;
five are unresolved. The seven unresolved units must not be silently assigned
to either “shared” or “different.”

Accordingly, the following phrasings are not authorized:

- “the rescue detects nothing new”;
- “no legacy neuron is lost”;
- “the entire yield difference is relabelling/re-clustering”; or
- “preprocessing and detection have been ruled out.”

The supported interpretation is that **most of the observed difference is
expressed at clustering and curation**, while seven units remain unadjudicated.
That localizes the observed output difference, not its upstream cause:
preprocessing or motion representation may preserve overlapping event pools
while changing how events are partitioned into cluster identities.

## Phase A2 consequence

The v2 repartition audit found no families meeting its prespecified
**coexisting-fragment** signature of ordinary over-peeling. That rejects that
specific signature; it does not prove that over-splitting is absent in every
sense. Only 6% of imec0 and 13% of imec1 fragment unions were
refractory-clean, and about 90% of families were classified `ambiguous`.

Phase A2 therefore establishes neither clean motion fragmentation nor ordinary
coexisting-fragment over-peeling as the dominant mechanism. It does not show
that stitching is beneficial, that fragmentation must be prevented upstream,
or that motion caused the repartitioning.

## What remains unresolved

- Whether rescue detects any real neurons that legacy misses, or vice versa.
- Whether either clustering better represents biological neurons.
- Whether the new pipeline handles motion better than legacy.
- The corrected exclusive matched-unit completeness comparison required by
  [0009](0009-cross-sort-comparisons-must-be-unit-matched.md).
- The mechanism behind the 27 similar good-good pairs.

Pipeline superiority remains a known-truth, held-out question. The corrected
C2 static qualification and paired static-versus-moving experiment are the
next relevant tests; raw KS-good yield and cross-sort coincidence cannot answer
it.
