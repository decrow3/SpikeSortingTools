# The 127 legacy-good units rescue drops: all relabelling, none lost (Luke0804 imec0)

> **RETRACTED PENDING V2 RERUN — 2026-09-03.** The 127-unit cohort came from a
> non-exclusive matcher, and the 100% “present in rescue” statistic had an
> approximately 87% whole-probe chance baseline. No claim about detection loss,
> curation loss, or relabelling survives. V2 fails closed unless spatial
> coincidence exceeds a fixed circular-shift null.

**Date:** 2026-09-02
**Closes:** Phase A / Checkpoint A of
[`pipeline_improvement_plan.md`](pipeline_improvement_plan.md)
**Question left open by**
[`decisions/0010`](decisions/0010-rescue-yield-is-relabelling-not-detection.md):
the rescue-vs-legacy KS-good difference is +200 / −127. The +200 gained units
were classified in
[`luke_20250804_rescue_unique_units_audit.md`](luke_20250804_rescue_unique_units_audit.md)
— none is a new detection. The −127 **lost** legacy-good units had never been
examined and no gate measures them.
**Answer:** none of the 127 is a neuron lost at detection or by curation. Every
one of the 127 has 100% of its spikes present in the rescue sort. The −127, like
the +200, is entirely re-clustering and re-labelling.

Reproduce with `python testing/luke_rescue_lost_units_audit.py`. Outputs to
`testing/outputs/luke_rescue_lost_units_audit/` (untracked, local). Nothing is
written under `/mnt`. The coincidence machinery is imported unchanged from
`luke_rescue_unique_units_audit.py`, so both sides of the table are computed
identically.

## Method

The 127 are the legacy KS-good units with no mutual-best spike-time match
(±0.5 ms, coincident fraction ≥ 0.5) to any rescue KS-good unit — the same
matching that produced the +200 / −127 split. Each one's spikes were then
located in the **complete** rescue sort (all 710 clusters, MUA included), both
after curation and in the pre-curation `full_clu` set, and classified by where
they went:

| Class | Rule |
|---|---|
| absent at detection | < 25% of spikes found anywhere in the rescue sort |
| removed by curation | absent after curation but ≥ 25% present pre-curation |
| preserved as MUA | one rescue cluster holds ≥ 50%, and it is labelled `mua` |
| merged into a rescue good unit | one rescue cluster holds ≥ 50% and also dominates another legacy good unit |
| split across rescue clusters | top two rescue clusters each hold ≥ 25% |
| dispersed across rescue clusters | found, but no cluster holds ≥ 25% |

## Result: the −127 is relabelling, symmetric to the +200

| Classification | n | spikes found in rescue | best rescue partner | median rate | median amp | median refractory | n rv > 1% |
|---|---:|---:|---:|---:|---:|---:|---:|
| dispersed across rescue clusters | 82 | 100% | 21% | 0.12 Hz | 35 µV | 0.26% | 17 |
| preserved as MUA | 27 | 100% | 64% | 0.65 Hz | 24 µV | 0.14% | 0 |
| split across rescue clusters | 10 | 100% | 38% | 0.61 Hz | 22 µV | 0.18% | 0 |
| merged into a rescue good unit | 8 | 100% | 69% | 0.27 Hz | 19 µV | 0.06% | 0 |
| **absent at detection** | **0** | — | — | — | — | — | — |
| **removed by curation** | **0** | — | — | — | — | — | — |

**Checkpoint A is met and the decision is clear-cut.** The checkpoint asked
whether a substantial share of the 127 are genuine neurons lost at detection or
curation, which would be a regression the yield narrative hid. Zero are. There
is no detection regression and no curation-drop regression on this probe.

## The two sides are the same operation run in reverse

| + gained (200) | | − lost (127) | |
|---|---:|---|---:|
| legacy `mua` relabelled good | 80 | rescue relabels legacy good → `mua` (preserved as MUA) | 27 |
| dispersed across legacy clusters | 85 | dispersed across rescue clusters | 82 |
| split of a legacy good unit | 35 | split across rescue clusters + merged | 18 |
| genuinely new detection | 0 | absent at detection + removed by curation | 0 |

The rescue configuration is applying a **bidirectional curation threshold
shift** — it promotes 80 legacy-MUA clusters to good and demotes 27 legacy-good
clusters to MUA — on top of a **re-clustering** that redistributes low-rate
spikes among many small clusters in both directions. Neither stage that differs
is preprocessing or detection. This confirms
[`decisions/0010`](decisions/0010-rescue-yield-is-relabelling-not-detection.md)
from the other side.

## The classes, read out

- **27 preserved as MUA.** The cleanest sub-story. Rescue keeps these as
  coherent single clusters (best partner 0.5–0.85, refractory 0.05–0.5%,
  several at 3–5 Hz) but labels them MUA. They are the exact mirror of the 80
  MUA→good promotions: the same curation boundary, moved, cutting the other way.
  A promotion decision on the 80 (per
  [`0006`](decisions/0006-recovery-axis-is-post-sort-mua-reconciliation.md))
  must also account for these 27 demotions.
- **82 dispersed.** Low rate (median 0.12 Hz), the worst refractory of any
  class (17 of 82 over 1%), and the lowest-amplitude high-rate members
  (clusters 372, 482: ~10 µV, 12–22 Hz, shredded across 500+ rescue clusters
  with a ≤ 8% best partner). These mirror the +85 dispersed gains and are
  marginal on both sides — the re-clustering is churning a low-SNR spike pool,
  not moving neurons.
- **10 split, 8 merged.** Small counts. Legacy good units that rescue
  fragmented (10) or absorbed into a larger shared cluster (8).

## Consequences

1. **The −127 is not a regression to prioritise.** Phase A's stated trigger —
   "a substantial share of the 127 are genuine neurons lost at detection or
   curation" — did not fire. No detection or curation-drop defect exists here.
2. **Curation and clustering remain the only stages that differ**, now
   confirmed from both the +200 and the −127 sides. Phase D's priority order
   (curation/clustering first) stands.
3. **The MUA boundary must be decided as one thing, not two.** 80 promotions
   and 27 demotions are one moved threshold. Evaluating the promotions in
   isolation would miss that the same change costs 27 previously-good units.
4. **A guardrail for the −127 direction is cheap and should exist.** This audit
   is the Phase A secondary metric ("lost good units") and now runs in ~35 s on
   existing outputs; wire it into the ladder's L2/L4 scoring alongside the
   `+N / −M` from `luke_rescue_unique_units_audit.py`.

## Limits

- Same as the +200 audit: ±0.5 ms coincidence identifies shared spikes, not
  shared identity. With 29 M rescue spikes, `n_rescue_clusters_touched` is
  inflated by chance coincidence; only the dominant-partner fractions carry
  weight.
- KS-good / KS-`mua` is an automated label in both sorts, not adjudicated
  identity.
- One probe, one session. Reopening conditions in
  [`0010`](decisions/0010-rescue-yield-is-relabelling-not-detection.md) apply
  equally here: reproduce on imec1 and one other session before treating the
  symmetric decomposition as a property of the configuration.
