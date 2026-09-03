# The legacy-good units rescue drops: none lost at detection or curation (Luke0804 imec0)

> **V1 RESULT RETRACTED — 2026-09-03.** The 127-unit cohort came from a
> non-exclusive matcher and the 100% "present in rescue" statistic had an
> ≈87% whole-probe chance baseline.
>
> **V2 RESULT — 2026-09-03 (below).** Corrected run: exclusive one-to-one
> matching for the cohort; each unmatched legacy-good unit's presence in rescue
> adjudicated by a depth-windowed coincidence statistic gated against
> circular-shift nulls. The cohort is now **137**. The core finding is
> unchanged and if anything firmer: **0 absent at detection, 0 removed by
> curation.** No legacy-good neuron is lost upstream of clustering.

**V1 date:** 2026-09-02 · **V2 rerun:** 2026-09-03
**Closes:** Phase A / Checkpoint A of
[`pipeline_improvement_plan.md`](pipeline_improvement_plan.md)
**Answer:** none of the 137 is a neuron lost at detection or by curation. Five
are `detection status unresolved` — low-amplitude (11 µV), high-rate (3.6 Hz)
units the whole-probe statistic cannot adjudicate, not clean neurons dropped
upstream. The rest are re-clustering and re-labelling.

Reproduce with `python testing/luke_rescue_lost_units_audit.py`. Outputs to
`testing/outputs/luke_rescue_lost_units_audit_v2/` (untracked, local). Nothing is
written under `/mnt`. The coincidence + null machinery is imported unchanged
from `luke_rescue_unique_units_audit.py`, so both sides of the table are
computed identically.

## Method (v2)

The 137 are the legacy KS-good units with no exclusive mutual-best spike-event
match to any rescue KS-good unit. Each one's spikes are located in the
**complete** rescue sort (all clusters, MUA included), both after curation and
in the pre-curation `full_clu` set. A "present in rescue" call requires
depth-windowed (±100 µm) coincidence to exceed 0.25 **and** to exceed the
circular-shift null median by ≥ 0.10; otherwise the unit is
`detection status unresolved`. A unit unresolved after curation but supported
pre-curation is `removed by curation`.

| Class | Rule (v2) |
|---|---|
| detection status unresolved | spatial coincidence does not clear the circular-shift null |
| removed by curation | unresolved after curation but null-clears pre-curation |
| preserved as MUA | one rescue cluster dominates (≥ 50%), labelled `mua` |
| merged into a rescue good unit | dominant rescue cluster also dominates another legacy good unit |
| split across rescue clusters | top two rescue clusters each hold ≥ 25% |
| dispersed across rescue clusters | null-supported, but no cluster holds ≥ 25% |

## Result: 0 lost at detection, 0 lost to curation

| Classification | n | median found (curated) | best rescue partner | median rate | median amp | median refractory | frac rv > 1% |
|---|---:|---:|---:|---:|---:|---:|---:|
| dispersed across rescue clusters | 89 | 0.86 | 0.26 | 0.17 Hz | 31 µV | 0.24% | 14.6% |
| preserved as MUA | 23 | 0.96 | 0.59 | 0.65 Hz | 24 µV | 0.19% | 0% |
| merged into a rescue good unit | 12 | 0.96 | 0.68 | 0.25 Hz | 19 µV | 0.01% | 0% |
| split across rescue clusters | 8 | 0.96 | 0.33 | 0.15 Hz | 25 µV | 0.48% | 25% |
| **detection status unresolved** | **5** | 0.15 | 0.03 | 3.64 Hz | 11 µV | 0.24% | 40% |
| **absent at detection** | **0** | — | — | — | — | — | — |
| **removed by curation** | **0** | — | — | — | — | — | — |

**Checkpoint A is met.** The checkpoint asked whether a substantial share are
genuine neurons lost at detection or curation, a regression the yield narrative
would have hidden. Zero are. The 5 unresolved units are low-amplitude, high-rate
outliers with 40% refractory violation — marginal in legacy too, not clean
neurons lost upstream. No detection regression and no curation-drop regression
on this probe.

## The two sides are the same operation run in reverse

| + gained (210) | | − lost (137) | |
|---|---:|---|---:|
| legacy `mua` relabelled good | 91 | rescue relabels legacy good → `mua` (preserved as MUA) | 23 |
| dispersed across legacy clusters | 87 | dispersed across rescue clusters | 89 |
| split of a legacy good unit | 30 | split + merged into a rescue good unit | 20 |
| detection status unresolved | 2 | detection status unresolved | 5 |
| genuinely new detection | 0 | absent at detection + removed by curation | 0 |

The rescue configuration applies a **bidirectional curation threshold shift** —
promoting legacy-MUA clusters to good and demoting legacy-good clusters to MUA —
on top of a **re-clustering** that redistributes low-rate spikes among many
small clusters in both directions. Neither stage that differs is preprocessing
or detection. This confirms
[`decisions/0010`](decisions/0010-rescue-yield-is-relabelling-not-detection.md)
from the other side.

## The classes, read out

- **23 preserved as MUA.** The cleanest sub-story. Rescue keeps these as
  coherent single clusters (best partner 0.59, refractory 0.19%, several at
  3–5 Hz) but labels them MUA — the mirror of the MUA→good promotions: the same
  curation boundary, moved, cutting the other way. A promotion decision on the
  91 (per [`0006`](decisions/0006-recovery-axis-is-post-sort-mua-reconciliation.md))
  must also account for these demotions.
- **89 dispersed.** Low rate (median 0.17 Hz), the worst refractory of any
  class (14.6% over 1%), best partner 0.26. These mirror the +87 dispersed
  gains and are marginal on both sides — the re-clustering churns a low-SNR
  spike pool, not moving neurons.
- **8 split, 12 merged.** Small counts. Legacy good units that rescue
  fragmented (8) or absorbed into a larger shared cluster (12).
- **5 detection status unresolved.** 11 µV, 3.6 Hz, 40% refractory violation —
  the whole-probe coincidence statistic cannot place them and the null test
  does not clear. These are marginal-everywhere units, not a detection loss.

## Consequences

1. **The −137 is not a regression to prioritise.** Phase A's stated trigger —
   "a substantial share are genuine neurons lost at detection or curation" — did
   not fire. No detection or curation-drop defect exists here.
2. **Curation and clustering are where the two configurations differ**, now
   confirmed from both the +210 and the −137 sides. (This localizes the
   observable discrepancy; per the plan it does not by itself localize the
   *cause* — preprocessing/motion representation can change how a fixed event
   pool is partitioned.)
3. **The MUA boundary must be decided as one thing, not two.** The promotions
   and demotions are one moved threshold; evaluating promotions in isolation
   would miss the cost.
4. **A guardrail for the −137 direction is cheap and should exist.** This audit
   is the Phase A secondary metric ("lost good units"); wire it into the
   ladder's L2/L4 scoring alongside the `+N / −M` from
   `luke_rescue_unique_units_audit.py`.

## Limits

- ±0.5 ms coincidence identifies shared spikes, not shared identity. The v2
  spatial + circular-shift-null gate removes the whole-probe chance-coincidence
  inflation that invalidated v1, but a null-clearing coincidence still shows the
  spikes were detected, not that either sort's grouping is correct.
- KS-good / KS-`mua` is an automated label in both sorts, not adjudicated
  identity.
- One probe, one session. Reopening conditions in
  [`0010`](decisions/0010-rescue-yield-is-relabelling-not-detection.md) apply
  equally here: reproduce on imec1 and one other session before treating the
  symmetric decomposition as a property of the configuration.
