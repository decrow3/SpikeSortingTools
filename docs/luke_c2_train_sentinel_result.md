# Threshold cliffs follow train composition and phase, not event count

**Date:** 2026-09-04
**Status:** diagnostic. Invalidates the single-train threshold ranking; does not
select a configuration.
**Code:** [`testing/luke_c2_train_sentinel.py`](../testing/luke_c2_train_sentinel.py)
(`luke-c2-train-sentinel-v1`)
**Output:** `testing/outputs/luke_c2_train_sentinel/` — 105 cells, no errors.

## Why

The 154-cell static sweep nominated 8/8 and 9/9 against production 12/9. The
staircase comparison then re-ran the same static configurations with 687 injected
events instead of 708, and the candidates' advantage vanished: total static FP
went 12 → 1244 for 8/8 and 97 → 896 for 9/9, with D02, D04 and D07 each falling
from ~0.99 to 0.45–0.55.

That invalidated the ranking, but could not say *why*. The 21 removed events are
structured — they cluster at the staircase's three step boundaries — so count,
which events, phase and general composition were all confounded.

## Design

Five sentinels (D02, D04, D07 collapsed; D10 the original failure; D14 moved the
other way) × seven frozen realisations × three configurations, static only,
correction off, every configuration seeing the identical realisation. **Every
realisation holds the count at 687 except the reference**, so count varies on
exactly one axis:

| realisation | n | isolates |
|---|---:|---|
| `full_708` | 708 | the only count difference |
| `boundary_687` | 687 | the structured deletion that broke the comparison |
| `random_687_s1/s2/s3` | 687 | which events, and spread across seeds |
| `uniform_687` | 687 | spread rather than clustered deletion |
| `phase_687` | 687 | the same events removed, every spike shifted half an ISI |

## Result: not count

| configuration | `boundary_687` fails | the other five 687 trains fail |
|---|---|---|
| 12/9 | 1/5 | **7/25** |
| 8/8 | 2/5 | 0/25 |
| 9/9 | 1/5 | 1/25 |

Collapses occur on random and phase-shifted trains too, so the 21 boundary events
are not special. Two cases separate the remaining axes:

* **D04 at 12/9** — `boundary_687` 0.991, `phase_687` **0.852**. The same events
  are removed; only absolute timing differs. **Phase alone triggers a collapse.**
* **D02 at 12/9** — fine on five realisations, **0.842** on `random_687_s2`, a
  deletion with no structure at all.

The trigger is therefore general sensitivity to train composition and phase. For
a given donor and configuration, a particular realisation is a lottery ticket.

## D10 is mechanistically different

D10 at 12/9 fails on **all seven** realisations (0.432–0.493, FP 271–283),
invariant to composition. That matches the
[stage trace](../testing/luke_c2_threshold_stage_trace.py): 99 of 708 events never
cross `Th_universal=12` and are absent from `full_st` before any clustering, so no
rearrangement of the train can rescue them. Every other failure observed here is
contamination-driven and realisation-dependent.

**Systematic and sporadic failures must be counted separately.** They have
different causes and different remedies — a detection-stage loss needs a lower
threshold, a contamination lottery needs a contamination guard.

## Distributions over all 35 donor × realisation cells

| configuration | median | p10 | failures <0.9 | excl. D10 | FP p90 | FP max |
|---|---:|---:|---:|---:|---:|---:|
| 12/9 (production) | 0.991 | **0.459** | 9/35 | 2/28 | **276** | 283 |
| 8/8 | 0.987 | 0.980 | 2/35 | 2/28 | 5 | 697 |
| 9/9 | **0.993** | **0.988** | 3/35 | 3/28 | **1** | 820 |

On lower tail and FP tail, production is the worst of the three — but that is
driven entirely by D10's systematic failure. Excluding it, all three sit at 2–3
sporadic failures in 28 cells and **the configurations are not separated**.

An earlier reading that production 12/9 was "the stable one" is withdrawn: it was
based on two realisations. Across seven it is not more stable; it simply did not
draw a losing ticket in those two.

## Consequences

1. **No threshold configuration is currently supported.** 12/9 remains the
   operational baseline by default, not on evidence.
2. **Single-realisation benchmarking cannot rank configurations here.** Any
   future candidate comparison on injected trains must run several realisations
   and be judged on distributions.
3. **Stage 2 needs more realisations than this.** 2–3 failure events per
   configuration cannot separate candidates; see the
   [stage-2 prespec](luke_c2_train_stability_stage2_prespec.md).

## Limits

Five donors, seven realisations, one background window, static arms only, one
donor amplitude scale. The failure criterion (accuracy < 0.9) is a threshold on a
continuum. Whether real Luke units show the same composition sensitivity is
untested — this is a property of the injected benchmark as built.
