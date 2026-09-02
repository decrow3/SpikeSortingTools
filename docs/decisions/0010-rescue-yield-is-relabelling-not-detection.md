# 0010 — The rescue pipeline's yield difference is relabelling, not detection

**Status:** Adopted 2026-09-02
**Evidence:** [`luke_20250804_rescue_unique_units_audit.md`](../luke_20250804_rescue_unique_units_audit.md)
**Bears on:** [0001](0001-ks4-unwarped-is-the-production-sorter.md),
[0006](0006-recovery-axis-is-post-sort-mua-reconciliation.md),
[0009](0009-cross-sort-comparisons-must-be-unit-matched.md)

## Finding

On Luke0804 imec0, **the rescue pipeline detects no spikes that the legacy
pipeline did not detect.** All 200 KS-good units that rescue has and legacy does
not are built from spikes already present in the legacy sort — median 100% of
each unit's spikes locatable there within 0.5 ms.

| | count |
|---|---:|
| rescue good matched to a legacy good unit | 101 |
| rescue good unique to rescue | **200** |
| legacy good not reproduced by rescue | **127** |
| net | +73 |

The reported "+73 KS-good units (+32%)" is a two-way relabelling of +200/−127,
not an incremental gain.

The 200 decompose as 80 legacy-MUA clusters relabelled good, 85 units assembled
from spikes dispersed across many legacy clusters, and 35 splits of legacy good
units. None is a new detection.

## Decision

**Differences between these configurations are attributable to clustering and
curation, not to preprocessing or detection.** Work aimed at improving yield or
resolving the gate failures must target those stages.

Concretely:

- **Do not run preprocessing sweeps to resolve this.** Preprocessing changes
  what is available to detect. Detection is not where these configurations
  differ, so a sweep cannot address the actual difference.
- **The 80 automatic MUA promotions require an explicit decision.**
  [0006](0006-recovery-axis-is-post-sort-mua-reconciliation.md) concluded that
  under a conservative screen no unit was safe to promote, and that any
  promotion should be reversible and backed by family-link evidence. The rescue
  configuration performs exactly this promotion automatically and at scale. The
  refractory evidence is reassuring (median 0.16% violations, 6.3% above 1%,
  against 0.09% and 1.0% for units both sorts agree on) but it is not the screen
  0006 specified. Either 0006's screen is relaxed deliberately and on the
  record, or these promotions are not yet authorized.
- **The 127 lost legacy good units must be examined before promotion.** They are
  a symmetric risk to the 200 gained and no current gate measures them. A
  configuration that silently drops 127 previously-good units is not obviously
  an improvement, whatever its net count.

## What this does not resolve

The similar-pair gate failure. The 27 nearby similar good–good pairs
(similarity ≥ 0.8 within 100 µm, reproduced exactly) are **not** concentrated in
the extra units: involvement is 20.0% for splits, 14.9% for units shared with
legacy, 13.8% for MUA promotions and 10.6% for dispersed units, with dispersed
versus rest at odds ratio 0.66, p = 0.357.

The prediction that the extra units cause the gate failure was tested and
failed. The 27 pairs are a property of the curated population as a whole,
including units rescue shares with legacy. This remains open.

## Consequence for `reject_universal_default`

Unchanged. The verdict still rests on the gates that failed. This record
narrows *where* to look for the cause: curation and clustering, not
preprocessing.

## Reopening conditions

This is a single-probe, single-session result. The +200/−127 decomposition
should be reproduced on imec1 and on at least one other session before it is
treated as a general property of the configuration rather than of this
recording.

## Limits

Spike-time coincidence establishes that spikes were shared, not that a
grouping is correct. KS-good is an automated label in both sorts. Refractory
cleanliness is necessary but not sufficient for a single unit.
