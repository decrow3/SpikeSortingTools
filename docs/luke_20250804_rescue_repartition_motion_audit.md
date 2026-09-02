# Phase A2: rescue's re-partitioning is clean temporal flicker, not motion-tracked, not over-splitting

**Date:** 2026-09-02
**Closes:** Phase A2 / Checkpoint A2 of
[`pipeline_improvement_plan.md`](pipeline_improvement_plan.md)
**Prespecified** (frozen before looking) in `PRESPEC` inside
`testing/luke_rescue_repartition_motion_audit.py` and written to
`testing/outputs/luke_rescue_repartition_motion_audit/prespec.json`. The script
refuses to run against a changed prespec.
**Runs on existing sorts only** — imec0 **and** imec1, no new sort.

## Question

Phase A ([`luke_20250804_rescue_lost_units_audit.md`](luke_20250804_rescue_lost_units_audit.md))
split the −127 lost legacy-good units into 27 label-threshold demotions and
**100 re-clustered** units, and named a hypothesis: legacy stabilised moving
neurons by resampling voltage; rescue preserves the voltage and leaves KS4 to
represent a moving footprint by splitting it across templates. Phase A2 tests
that against the plan's discriminator, **fixed in advance**:

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
- **Merge cleanliness** — refractory-violation fraction (ISI < 1.5 ms) of the
  merged train `S`.

Classes: `motion_fragmentation` (successive + trajectory + clean),
`over_splitting` (coexisting), `successive_clean_no_motion_signal` (successive +
clean but no trajectory — reported as its own bucket, forced into neither),
`ambiguous` (the 0.2–0.5 partial-overlap zone).

Reproduce: `python testing/luke_rescue_repartition_motion_audit.py --probe both`.

## Result — consistent across both probes

| | imec0 | imec1 |
|---|---:|---:|
| families scored | 117 | 62 |
| **coexisting fragments** (overlap > 0.5) | **0.0%** | **1.6%** |
| successive fragments (overlap < 0.2) | 50% | 69% |
| **merge is refractory-clean** | **92%** | **95%** |
| median ownership switches / hour | 18 | 22 |
| **median \|depth ↔ rigid-motion correlation\|** | **0.11** | **0.13** |
| classed `motion_fragmentation` | 2 / 117 | 3 / 62 |
| classed `over_splitting` | 0 | 1 |

(imec1 rescue is the **uncurated** KS4 output — no `cur/` stage was run for that
probe. The temporal-structure discriminator does not depend on curation labels;
the KS-good filter for family selection does. Noted as an asymmetry.)

### Three things this rules in and out

1. **It is not over-splitting.** Coexisting fragments — the signature of ordinary
   over-peeling, where two templates both fire for one neuron at the same time —
   are essentially absent (0% / 1.6%). The Phase D branch "the repartitioning is
   ordinary over-splitting → fix clustering and curation" is **not indicated**.

2. **The fragments are one neuron.** Merging each dispersed legacy unit's own
   spike train is refractory-clean in 92–95% of families (median violation
   0.2%). Whatever splits them, **post-sort family stitching would recover the
   unit** — the spikes are already there and already clean.

3. **It is not tracked by the rigid motion estimate, and it is not slow.**
   Depth ↔ rigid-DREDGE correlation is ≈ 0.11–0.13 (no relationship). Ownership
   of a unit's spikes flips between templates ~18–22 times per hour — every
   ~3 minutes — with no depth trajectory. This is **rapid template flicker**,
   not slow drift-driven succession. Only 2–3 families per probe clear the
   `motion_fragmentation` bar, and those are borderline.

## Reading, as a mix (not a verdict)

The dominant pattern on both probes is **temporally one-at-a-time, refractory-clean,
rapidly-flickering** template ownership with **no rigid-motion depth signature**.

This is consistent with the plan's *tension*: on imec0 the rigid DREDGE estimate
is small and QC-unqualified, so "if this is motion-driven the cause must be
non-rigid, or that estimate must be wrong." A2 cannot distinguish:

- **non-rigid / fast motion** that wobbles the footprint faster than the rigid
  estimate's resolution, from
- **template competition** in KS4's matching-pursuit step — several near-duplicate
  templates for one neuron, with an unstable choice between them over time.

Both produce successive, clean, flickering fragments. **C2 (the paired
static-vs-moving injection) is what separates them**: if a neuron injected on a
known trajectory flickers where the identical static injection does not, motion
is the driver.

Weak consistency signal for a motion contribution: imec1 (motion known to be
real) has a higher successive fraction (69% vs 50%) and larger depth excursions
in its fragmented families (16–35 µm vs 11–23 µm) than imec0.

## Consequence for Phase D's decision tree

| Plan's A2 row | Fits? |
|---|---|
| Fragments temporally complementary **and track estimated motion** | Partly — complementary yes, track the *rigid* estimate no |
| Fragments coexist at the same time and motion state | **No** (0–2%) |
| Mixed | Closest, but the "clustering/curation" side is near-empty |

**First target: unwarped motion-aware identity handling — post-sort family
stitching of temporally-complementary, refractory-clean fragments.** It is
indicated regardless of whether the root cause turns out to be non-rigid motion
or template competition, because family stitching repairs flicker and slow drift
alike, whereas curation-threshold tuning addresses neither. Curation/clustering
tuning drops down the priority order.

**C2 still decides the framing** — whether this is "the cost of the no-motion
strategy" (motion) or "KS4 template competition on preserved voltage" (not
motion). The plan requires A2 **and** C2 before Phase D begins; A2 has reported.

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
