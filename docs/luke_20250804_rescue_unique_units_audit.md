# What the rescue pipeline's extra KS-good units actually are (Luke0804 imec0)

> **RETRACTED PENDING V2 RERUN — 2026-09-03.** The matcher reused target events,
> and “found anywhere within 0.5 ms” had an approximately 89% whole-probe chance
> baseline. The +200/-127 decomposition, 80/85/35 classes, and “none are new
> detections” conclusion below are historical only. Corrected code uses
> one-to-one matching plus spatial and circular-shift-null evidence.

**Date:** 2026-09-02
**Question left open by** [`decisions/0009`](decisions/0009-cross-sort-comparisons-must-be-unit-matched.md):
rescue reports 301 KS-good units against legacy's 228. Are the extra units
genuine neurons, promoted MUA, fragments, or noise?
**Answer:** none of the difference is new detection. Every rescue-unique good
unit is built from spikes the legacy sort already detected.

Reproduce with `python testing/luke_rescue_unique_units_audit.py`. Outputs to
`testing/outputs/luke_rescue_unique_units_audit/` (untracked, local). Nothing
is written under `/mnt`.

## Method

Rescue and legacy good units were matched by spike-time coincidence (±0.5 ms,
mutual best match, coincident fraction of the smaller unit ≥ 0.5). Each
rescue-unique good unit's spikes were then located in the **complete** legacy
sort, MUA clusters included, and classified by where they came from.

## The +73 is really +200 / −127

| | count |
|---|---:|
| rescue good units matched to a legacy good unit | 101 |
| rescue good units unique to rescue | **200** |
| legacy good units not matched by rescue | **127** |
| net | +73 |

**Rescue is not a superset of legacy.** It gains 200 good units and loses 127.
The headline "+73 KS-good units (+32%)" conceals a substantial two-way
relabelling. The 127 legacy good units that rescue does not reproduce as good
have never been examined.

## None of the 200 are new detections

| Classification | n | spikes found in legacy | best legacy partner | median rate | median refractory | frac > 1% |
|---|---:|---:|---:|---:|---:|---:|
| dispersed across legacy clusters | 85 | 100% | 16.3% | 0.10 Hz | 0.11% | 17.6% |
| legacy **mua** relabelled good | 80 | 100% | 63.2% | 0.48 Hz | 0.16% | 6.3% |
| legacy **good** relabelled good | 35 | 100% | 46.3% | 0.16 Hz | 0.06% | 0% |
| genuinely new detection | **0** | — | — | — | — | — |

Reference: the 101 units matched to a legacy good unit have median refractory
violation fraction 0.09% with 1.0% exceeding 1%.

**The rescue pipeline detects no spikes that the legacy pipeline did not
detect.** Its entire yield difference is re-clustering and re-labelling of an
already-detected spike population. This is a statement about the *curation and
clustering* stages, not about preprocessing or detection.

### The three classes

- **80 legacy-MUA promotions.** Roughly two thirds of each unit's spikes come
  from a single legacy cluster that legacy labelled `mua`. They are
  refractory-clean — median 0.16%, only 6.3% exceeding 1%, close to the matched
  reference. These are the most defensible of the three and are exactly the
  reconciliation axis [`0006`](decisions/0006-recovery-axis-is-post-sort-mua-reconciliation.md)
  identified as the safest recovery route.
- **85 dispersed units.** Assembled from spikes spread across many legacy
  clusters, with no dominant source (best partner 16%). Mostly very low rate
  (0.10 Hz). Refractory-clean in aggregate, but 17.6% exceed 1%, the highest of
  the three classes. Within the >1 Hz subset specifically, these are markedly
  worse: median refractory 0.76%, half exceeding 1%, 29% exceeding 2%.
- **35 splits of legacy good units.** Substantial but sub-threshold overlap
  with a unit legacy already called good.

## The similar-pair gate failure is not explained by any class

The prespecified gate that failed counts nearby similar good–good pairs:
template similarity ≥ 0.8 within 100 µm of depth. That definition reproduces
the reported **27** exactly on the curated rescue output.

| Class | n units | in a similar pair | % |
|---|---:|---:|---:|
| legacy good relabelled good | 35 | 7 | 20.0 |
| matched to a legacy good unit | 101 | 15 | 14.9 |
| legacy mua relabelled good | 80 | 11 | 13.8 |
| dispersed across legacy clusters | 85 | 9 | 10.6 |

Dispersed versus all others: odds ratio 0.66, Fisher p = 0.357.

The prediction that the dispersed units drive the similar-pair excess is **not
supported**. Involvement is spread roughly in proportion across all classes,
and is if anything lowest in the dispersed class. The 27 pairs are a property
of the curated population as a whole — including the units rescue shares with
legacy — not of the extra units.

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
3. **The 127 lost legacy good units** — now examined in
   [`luke_20250804_rescue_lost_units_audit.md`](luke_20250804_rescue_lost_units_audit.md):
   none is lost at detection or by curation; the −127 is 27 good→`mua`
   demotions plus 100 re-clustered, a symmetric mirror of the +200.
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
