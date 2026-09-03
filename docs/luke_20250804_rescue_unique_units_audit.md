# What the rescue pipeline's extra KS-good units actually are (Luke0804 imec0)

> **V1 RESULT RETRACTED — 2026-09-03.** The v1 matcher reused target events, and
> "found anywhere within 0.5 ms" had an ≈89% whole-probe chance baseline. The
> +200/−127 decomposition and 80/85/35 class counts are historical only.
>
> **V2 RESULT — 2026-09-03 (below).** Corrected run: maximum-cardinality
> one-to-one event matching, plus a depth-windowed coincidence statistic gated
> against fixed circular-shift nulls before any shared-detection call. Cohort
> counts shift (+210 / −137) but the conclusion is unchanged: **the rescue
> yield gain is relabelling and re-clustering of an already-detected spike
> population, not new detection.**

**V1 date:** 2026-09-02 · **V2 rerun:** 2026-09-03
**Question left open by** [`decisions/0009`](decisions/0009-cross-sort-comparisons-must-be-unit-matched.md):
rescue reports 301 KS-good units against legacy's 228. Are the extra units
genuine neurons, promoted MUA, fragments, or noise?
**Answer:** none of the difference is a supported new detection. Every
rescue-unique good unit is built from spikes the legacy sort already detected.

Reproduce with `python testing/luke_rescue_unique_units_audit.py`. Outputs to
`testing/outputs/luke_rescue_unique_units_audit_v2/` (untracked, local). Nothing
is written under `/mnt`.

## Method (v2)

Rescue and legacy good units are matched by **maximum-cardinality one-to-one**
spike-event pairing (±0.5 ms, interval-order two-pointer, symmetric under
swap), mutual best match, coincident fraction of the smaller unit ≥ 0.5. Each
rescue-unique good unit's spikes are then located in the **complete** legacy
sort (MUA clusters included) and classified. A "shared detection" call now
requires the observed depth-windowed (±100 µm) coincidence fraction both to
exceed 0.25 **and** to exceed the median of three circular-shift nulls by ≥ 0.10
— so a unit is never called "genuinely new" from a low whole-probe coincidence
fraction alone; it is called `detection status unresolved`.

## The +73 is really +210 / −137

| | v2 count | (v1, retracted) |
|---|---:|---:|
| rescue good units matched to a legacy good unit | **91** | 101 |
| rescue good units unique to rescue | **210** | 200 |
| legacy good units not matched by rescue | **137** | 127 |
| net | +73 | +73 |

**Rescue is not a superset of legacy.** It gains 210 good units and loses 137.
The headline "+73 KS-good units (+32%)" conceals a large two-way relabelling.
(Exclusive matching yields fewer confirmed matches than the v1 reuse-prone
matcher, so both the gained and lost cohorts grow while the net is unchanged.)
The −137 lost legacy-good units are examined in
[`luke_20250804_rescue_lost_units_audit.md`](luke_20250804_rescue_lost_units_audit.md).

## None of the 210 are supported new detections

| Classification | n | median found in legacy | best legacy partner | median rate | median refractory | frac rv > 1% |
|---|---:|---:|---:|---:|---:|---:|
| legacy **mua** relabelled good | 91 | 1.00 | 0.52 | 0.36 Hz | 0.11% | 7.7% |
| dispersed across legacy clusters | 87 | 1.00 | 0.17 | 0.12 Hz | 0.16% | 16.1% |
| legacy **good** relabelled good | 30 | 1.00 | 0.46 | 0.26 Hz | 0.05% | 0% |
| **detection status unresolved** | **2** | 0.94 | 0.08 | 0.29 Hz | 0.02% | 0% |

The 2 "unresolved" units do **not** clear the spatial + circular-shift-null bar
for a genuine new detection; they are simply cases the whole-probe statistic
cannot adjudicate. There is no unit in the +210 that the null-controlled test
supports as detected-by-rescue-only.

**The rescue pipeline detects no spikes the legacy pipeline did not detect that
this test can confirm.** Its entire yield difference is re-clustering and
re-labelling of an already-detected spike population — a statement about the
*curation and clustering* stages, not about preprocessing or detection.

### The classes

- **91 legacy-MUA promotions.** About half of each unit's spikes come from a
  single legacy cluster that legacy labelled `mua` (best partner 0.52). They are
  refractory-clean — median 0.11%, 7.7% exceeding 1%. The most defensible class,
  and exactly the reconciliation axis
  [`0006`](decisions/0006-recovery-axis-is-post-sort-mua-reconciliation.md)
  identified as the safest recovery route.
- **87 dispersed units.** Assembled from spikes spread across many legacy
  clusters with no dominant source (best partner 0.17). Very low rate
  (0.12 Hz). Refractory-clean in aggregate, but 16.1% exceed 1% — the worst
  class — consistent with a low-SNR spike pool being re-churned rather than real
  neurons being recovered.
- **30 splits of legacy good units.** Substantial but sub-threshold overlap
  with a unit legacy already called good (best partner 0.46).

## The similar-pair gate failure is not explained by any class

The prespecified gate that failed counts nearby similar good–good pairs:
template similarity ≥ 0.8 within 100 µm of depth. That definition reproduces
the reported **27** exactly on the curated rescue output.

| Class | n units | in a similar pair | % |
|---|---:|---:|---:|
| legacy good relabelled good | 30 | 6 | 20.0 |
| matched to a legacy good unit | 91 | 14 | 15.4 |
| legacy mua relabelled good | 91 | 14 | 15.4 |
| dispersed across legacy clusters | 87 | 8 | 9.2 |

Dispersed versus all others: odds ratio 0.54, Fisher p = 0.145.

The prediction that the dispersed units drive the similar-pair excess is **not
supported** (if anything, involvement is lowest in the dispersed class). The
association is descriptive, not causal. The 27 pairs are a property of the
curated population as a whole — including the units rescue shares with legacy —
not of the extra units.

## Consequences

1. **A preprocessing sweep would target the wrong stage.** Since rescue detects
   nothing legacy did not, differences between the configurations are produced
   by clustering and curation. Preprocessing variants can only change what is
   available to detect, and detection is not where these two differ.
2. **The MUA promotions deserve a decision.** 80 units are automatic promotions
   of clusters legacy called MUA. [`0006`](decisions/0006-recovery-axis-is-post-sort-mua-reconciliation.md)
   concluded that under a conservative screen *no* unit was safe to promote and
   that promotion should be reversible and evidence-backed. The rescue
   configuration is performing that promotion automatically, at scale, without
   the family-link evidence 0006 required. The refractory evidence here is
   reassuring but it is not the screen 0006 specified.
3. **The 137 lost legacy good units** — examined in
   [`luke_20250804_rescue_lost_units_audit.md`](luke_20250804_rescue_lost_units_audit.md):
   none is lost at detection or by curation (0 absent at detection, 0 removed by
   curation under the null-controlled test); the −137 is a symmetric mirror of
   the +210 — MUA demotions plus re-clustering.
4. **The similar-pair gate failure remains unexplained** and is not attributable
   to the extra units.

## Limits

- Spike-time coincidence at ±0.5 ms identifies shared spikes, not shared
  neuronal identity. A high coincident fraction with a legacy MUA cluster shows
  the spikes were detected, not that the rescue unit's grouping is correct.
- KS-good is an automated label, not adjudicated identity, in both sorts.
- Refractory violation fraction is a necessary, not sufficient, condition for a
  single unit.
- Everything here is one probe of one session.
