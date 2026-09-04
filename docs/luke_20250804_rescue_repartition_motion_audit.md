# Phase A2: no coexisting-fragment signature; mechanism remains ambiguous

> **V1 RESULT RETRACTED — 2026-09-03.** The original report inherited the invalid
> cross-sort cohorts and measured "merged" refractory violations on the anchor
> train rather than the union of fragment trains. The 92–95% clean-merge result,
> the "these are one clean neuron" interpretation, and the stitching
> recommendation are withdrawn.
>
> **V2 RESULT — 2026-09-03 (below).** Corrected run: exclusive cross-sort
> identities, spatially-plausible fragment selection, and refractory violations
> measured on the actual union of the fragment clusters' full trains. The
> absence of the prespecified coexisting-fragment signature survives (0% on
> both probes). This does not exclude every form of over-splitting. The
> "one clean neuron / stitching would recover it" finding **does not** — only
> 6–13% of fragment unions are refractory-clean. Almost every family is now
> `ambiguous`: neither classic over-splitting nor clean motion fragmentation.

**V1 date:** 2026-09-02 · **V2 rerun:** 2026-09-03
**Closes:** Phase A2 / Checkpoint A2 of
[`pipeline_improvement_plan.md`](pipeline_improvement_plan.md)
**Prespecified** (frozen before looking) in `PRESPEC` inside
`testing/luke_rescue_repartition_motion_audit.py` and written to
`testing/outputs/luke_rescue_repartition_motion_audit_v2/prespec.json`. The script
refuses to run against a changed prespec.
**Runs on existing sorts only** — imec0 **and** imec1, no new sort.

## Question

Phase A ([`luke_20250804_rescue_lost_units_audit.md`](luke_20250804_rescue_lost_units_audit.md))
found the lost legacy-good units are dispersed across many rescue clusters, and
named a hypothesis: legacy stabilised moving neurons by resampling voltage;
rescue preserves the voltage and leaves KS4 to represent a moving footprint by
splitting it across templates. Phase A2 tests that against the plan's
discriminator, **fixed in advance**:

| Observation | Reading |
|---|---|
| Fragments occupy **successive** epochs, coherent depth trajectory (tracks estimated motion or monotonic), `S` merges **without** refractory violations | motion fragmentation |
| Fragments **coexist** at the same times and motion state | over-splitting / over-peeling |

## Method

For each strongly-dispersed legacy↔rescue family (a legacy-lost good unit whose
best rescue partner captures < 60% of its train, or a rescue-gained good unit
whose best legacy partner captures < 25%; ≥ 300 spikes; ≥ 2 fragments; every
qualifying family scored, no subsampling):

- **Temporal structure** — 30 s bins over the unit's span. Per bin, which
  fragment captures the unit's spikes. `temporal_overlap` = Σ min / Σ max of the
  top-two fragments' per-bin shares (0 = strictly one-at-a-time, 1 = always
  co-present). `ownership_switch_per_hr` = how often the dominant fragment
  changes.
- **Depth trajectory** — per-bin median spike depth vs time (Spearman), vs the
  DREDGE **rigid** motion trace (`motion_corr`), and the depth excursion.
- **Merge cleanliness (v2)** — refractory-violation fraction (ISI < 1.5 ms) of
  the **union of the fragment clusters' full spike trains** — every spike that
  would actually enter a merge, including each fragment's unmatched/extra
  events. (V1 measured the anchor train alone, which is clean by construction
  and cannot test a merge.)

Classes: `motion_fragmentation` (successive + trajectory + clean),
`over_splitting` (coexisting), `successive_clean_no_motion_signal` (successive +
clean but no trajectory — reported as its own bucket, forced into neither),
`ambiguous` (none of the above).

Reproduce: `python testing/luke_rescue_repartition_motion_audit.py --probe both`.
Outputs to `testing/outputs/luke_rescue_repartition_motion_audit_v2/<probe>/`.

## V2 result — no coexisting-fragment signature; the clean-merge finding fails

| | imec0 | imec1 | (v1, retracted) |
|---|---:|---:|---|
| families scored | 96 | 46 | 117 / 62 |
| **coexisting fragments** (overlap > 0.5) | **0.0%** | **0.0%** | 0.0% / 1.6% |
| successive fragments (overlap < 0.2) | 56% | 72% | 50% / 69% |
| **fragment-union merge is refractory-clean** (rv ≤ 1.5%) | **6%** | **13%** | ~92% / ~95% |
| median fragment-union refractory-violation fraction | **14.1%** | **8.1%** | (anchor: 0.2%) |
| median ownership switches / hour | 17.6 | 21.2 | 18 / 22 |
| **median \|depth ↔ rigid-motion correlation\|** | **0.13** | **0.17** | 0.11 / 0.13 |
| classed `motion_fragmentation` | **0 / 96** | 2 / 46 | 2 / 117 · 3 / 62 |
| classed `over_splitting` | 0 | 0 | 0 / 1 |
| classed `ambiguous` | **92 / 96** | **41 / 46** | — |

(imec1 rescue is the **uncurated** KS4 output — no `cur/` stage was run for that
probe. The temporal-structure discriminator does not depend on curation labels;
the KS-good filter for family selection does. Noted as an asymmetry.)

### What survives, what falls

1. **The prespecified over-peeling signature was not observed.** Coexisting fragments — the signature of ordinary
   over-peeling, where two templates both fire for one neuron at the same time —
   are **absent** on both probes (0.0%). That branch is not indicated by this
   test, but broader forms of over-splitting are not ruled out.
   *This finding is unchanged from v1.*

2. **The fragments are NOT demonstrably one clean neuron.** When the fragment
   clusters' full trains are unioned — the actual merge, not the anchor — only
   **6–13%** are refractory-clean; the median union carries an **8–14%**
   refractory-violation fraction. Fragment capture is partial (top-3 fragments
   typically cover 60–80% of the anchor train, medians ~3 fragments) and the
   fragment clusters are frequently large contaminated clusters that merely clip
   the anchor. **Post-sort family stitching of these would produce
   refractory-violating units, not recover clean ones.** This directly
   is qualitatively consistent with the historical Candidate 1 stitching run
   ([`luke_20250804_family_stitch_candidate.md`](luke_20250804_family_stitch_candidate.md):
   whose exact 2/127, 4-destroyed and 34-absorbed accounting was later retracted
   by [decision 0011](decisions/0011-cross-sort-event-matching-and-detection-evidence.md)).
   *This reverses the load-bearing v1 finding.*

3. **It is not tracked by the rigid motion estimate, and it is not slow.**
   Depth ↔ rigid-DREDGE correlation is ≈ 0.13–0.17 (no relationship). Ownership
   of a unit's spikes flips between templates ~18–21 times per hour — every
   ~3 minutes — with no depth trajectory. This is **rapid template flicker**,
   not slow drift-driven succession. 0/96 (imec0) and 2/46 (imec1) families
   clear the `motion_fragmentation` bar. *Unchanged from v1.*

## Reading, as a mix (not a verdict)

The dominant pattern on both probes (~90% of families, `ambiguous`) is
**temporally one-at-a-time but not cleanly reassemblable**, with **rapidly
flickering** template ownership and **no rigid-motion depth signature**. It is
neither classic over-splitting (fragments never coexist) nor clean motion
fragmentation (the pieces don't merge cleanly and don't track motion). The
re-partitioning churns a contaminated, low-rate spike pool.

A2 still cannot distinguish:

- **non-rigid / fast motion** that wobbles the footprint faster than the rigid
  estimate's resolution, from
- **template competition** in KS4's matching-pursuit step — several near-duplicate
  templates for one neuron, with an unstable choice between them over time.

Both produce successive, flickering fragments. **C2 (the paired
static-vs-moving injection) is what separates them** — and C2's geometry-aware
rerun is still pending.

## Consequence for Phase D's decision tree

| Plan's A2 row | Fits? |
|---|---|
| Fragments temporally complementary **and track estimated motion** | Complementary yes; track the rigid estimate no; **merge cleanly no** |
| Fragments coexist at the same time and motion state | **No** (0.0% both probes) |
| Mixed | Closest — but neither pure branch is supported |

**The v1 "first target: post-sort family stitching of refractory-clean
fragments" recommendation is withdrawn.** V2 shows the fragment unions are
*not* refractory-clean (6–13%), so stitching them reconstructs contaminated
units — which is exactly what the Candidate 1 full-session evaluation went on to
   suggest. It does not establish that fragmentation must be prevented upstream;
   the mechanism remains unresolved.

**C2 still decides the framing** — whether this is "the cost of the no-motion
strategy" (motion) or "KS4 template competition on preserved voltage" (not
motion). Both A2 and the corrected C2 are required before Phase D's tree
resolves; A2 v2 has reported, C2's geometry-aware rerun has not.

## Limits

- `temporal_overlap` uses the top-two fragments only; families with 3–5
  comparable fragments (median 3) are partially characterised by it. The
  ownership-switch rate covers all fragments and tells the same story.
- `motion_corr` uses the **rigid** DREDGE trace, which is QC-unqualified
  (`weights_thresh` non-finite). A non-rigid or better-QC'd estimate could show
  a signal this misses. That is precisely the open question.
- 30 s bins set the floor on "successive"; sub-30 s alternation is invisible to
  it (reads as one fragment owning the bin).
- imec1 rescue is uncurated; the family set there is legacy-good vs
  rescue-KS-good-raw.
- Spike-time coincidence (±0.5 ms) identifies shared spikes, not shared
  identity. Refractory cleanliness of a merge is necessary, not sufficient.
- **V2 fragment-union cleanliness is a conservative (pessimistic) merge
  estimate.** `_fragments` admits any spatially-plausible cluster capturing
  ≥ 5% of the anchor train, then unions the full trains. A real stitcher would
  merge mutual-best partners only, so the true "best-case stitch" cleanliness
  lies somewhere between the 6–13% here and the retracted 92–95% anchor figure.
  The Candidate 1 full-session result (net loss, contaminated merges) is the
  direct test and lands on the pessimistic side.
