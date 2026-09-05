# Stage 2 prespec — threshold configurations judged on distributions

**Date:** 2026-09-04
**Status:** **draft for review. Not run.**
**Gates:** L1C, held-out evaluation and fractional threshold cells stay paused
until this completes. 12/9 remains the operational baseline throughout.
**Motivated by:** [`luke_c2_train_sentinel_result.md`](luke_c2_train_sentinel_result.md)

## What stage 1 established, and what it left open

Threshold collapses follow train **composition and phase**, not event count, and
a given realisation is a lottery ticket for a given donor × configuration. D10 at
12/9 is the one systematic failure — a detection-stage loss invariant to the
train. Excluding it, all three configurations showed 2–3 sporadic failures in 28
cells, which **cannot separate them**. Stage 2 exists to give the failure rate
enough precision to be usable, or to establish that it cannot be had cheaply.

## Design

14 donors × 14 realisations × 3 configurations = **588 cells**, static only,
correction off, paired: every configuration sees the identical frozen realisation.
At the measured 0.62 min/cell this is ~6 h, so it is an overnight run.

**Realisations** — all 687 events, preregistered and frozen by seed:

* `random_s1..s6` at phase 0 — composition;
* `random_s1..s6` at phase +½ ISI — phase crossed with composition;
* `boundary_687` — the structured deletion, as a named reference;
* `uniform_687` — spread deletion, as a named reference.

Crossing six deletions with two phases is deliberate: stage 1 showed both axes
can trigger a collapse independently, so neither may be held fixed.

## Endpoints, prespecified

Primary is the **sporadic failure rate** per configuration; a cell fails if
accuracy < 0.9.

* **Systematic** — a donor × configuration failing on ≥ 12 of 14 realisations.
  Reported separately and excluded from the sporadic rate; it is a different
  mechanism with a different remedy.
* **Sporadic** — everything else, compared paired across configurations
  (McNemar on realisation-matched discordant cells).

Secondary, all with bootstrap CIs over donors: median accuracy, **p10 accuracy**,
**FP p90**, FP max, split rate, refractory-violation median.

Ranking uses median, lower tail, failure probability and FP tail together. Not
best accuracy, not mean accuracy, not KS-good yield.

## Decision rules, written before the data

1. A candidate **replaces** production only if its paired sporadic failure rate
   is lower with a CI excluding zero, **and** its FP p90 is not worse, **and**
   it introduces no new systematic failure.
2. A candidate is **dropped** on a material fragmentation or contamination
   regression: split rate or FP p90 worse than production with a CI excluding
   zero.
3. If no candidate satisfies (1), the honest outcome is **no threshold change**,
   and the benchmark's realisation sensitivity becomes the finding.

## Power, stated in advance

Paired McNemar on 14 × 14 = 196 cells per configuration detects a shift from a
~7 % to a ~2 % sporadic failure rate at roughly 80 % power. It **cannot** resolve
differences smaller than about 3 percentage points. If the observed rates land
within that band the result is "not separated", which rule 3 already covers —
this is not licence to add realisations until something separates.

## What this deliberately does not do

* No fractional threshold cells. The preregistered 8.5/8, 8.5/8.5 and 9/8.5
  interpolation stays unavailable until a stable integer winner exists.
* No held-out donors or snippets. Stage 2 is development evidence; the same 14
  donors that produced the candidates cannot also confirm them.
* No motion arms. Displacement is not the variable; the staircase comparison
  already showed threshold changes do not fix 40 µm fragmentation.
* No production change on stage-2 evidence alone. A winner here still owes L1C,
  held-out static, and matched real-data evaluation.

## Open question for review

Whether 14 realisations is the right cost. Halving to 7 makes it a ~3 h run but
drops the detectable difference to roughly 5 percentage points; doubling to 28
costs ~12 h for about 2 points. The proposal takes the middle, on the basis that
stage 1's observed rates (~7 % including D10, ~9 % excluding) sit near the top of
the detectable band.
