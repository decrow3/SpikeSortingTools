# 0005 — DARTsort and KIAsort are deferred, not rejected

**Status:** Deferred. Not part of production. Work continues in the research repository.

## Decision

Neither DARTsort nor KIAsort enters the production pipeline. The bake-off
(`pipeline/bakeoff.py`, `sorter_bakeoff.py`) stays in the research repository and
is **excluded** from the production extraction.

A challenger may only be promoted by passing the full production acceptance
process, not by winning a bounded window.

## The governing guardrail

> A challenger cannot advance on unit count alone.

Reviewed-event recovery, refractory violations, and near-coincident duplicate
burden are guardrails that must clear **before** longitudinal continuity is
interpreted. Every candidate also starts from the accepted rescue recording
manifest, and spatial resampling of voltage for motion correction stays
forbidden for challengers too.

## What the KIAsort comparison actually showed

Apparent extra yield was largely **waveform-family fragmentation**, not
recovered neurons:

- 51 eligible KS4 units and 90 eligible geometry-valid KIAsort units resolve into
  8 one-to-one families, 5 one-KS4-to-many-KIAsort candidates, 1 reverse
  candidate, 25 isolated KIAsort units, 15 isolated KS4 units, plus complex
  many-to-many neighbourhoods. Connected-component shape varies with the edge
  rule, so it is a hypothesis generator, not a unit count.
- The 5 simple one-to-many families have median cross-sort template cosines of
  0.86–0.91 with up to six KIAsort identity switches across 10 s bins —
  consistent with KIAsort splitting temporally complementary parts of one family.
- The single reverse candidate is weak (median cross-sort cosine 0.53).
- Of 25 isolated KIAsort families (81,979 spikes), a conservative waveform gate
  retains only units 46 and 82 — **2,788 spikes** — and those are *review
  targets, not accepted added yield*. Default auto-curation was tested and does
  not rescue the configuration.

Elsewhere, KIAsort-nominated candidates matched KS4 events scattered across
16–19 KS4 units, almost all MUA, with no single KS4 unit capturing more than
22.2%. Naively unioning the implicated labels gives 67.4% and 75.4% refractory
violations. These are explicitly **not safe merge sets**.

## Reopening conditions

If KIAsort is reopened, the evidence supports **at most one merge/sampling-focused
tuning pass** — not a full-probe runtime effort. That branch sits behind the
paired KS4-seeded motion-aware benchmark in priority.

## Operational notes

- DARTsort is imported lazily (`pipeline/bakeoff.py`), so it is not a hard
  dependency of the package.
- DARTsort reported 104 initially uncovered spikes in a neighbourhood-coverage
  check and expanded to zero uncovered.
- Numba disabled its TBB threading backend (installed TBB too old); runs still
  completed.

## Evidence pointers

- `docs/sorter_architecture_bakeoff.md`
- `testing/luke_sorter_unit_families.py`, `testing/luke_sorter_waveform_arbitration.py`
