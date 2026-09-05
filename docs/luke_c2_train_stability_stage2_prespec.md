# Stage 2 prespec — threshold configurations judged on distributions

**Date:** 2026-09-04
**Status:** **draft for review. Not run.**
**Revision 2, 2026-09-05** — after an external code review found the decision
analysis statistically invalid. Changes are marked **[rev2]** and were made
*before* any stage-2 data existed.
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

Primary is the **sporadic failure rate**; a cell fails if accuracy < 0.9.

* **Systematic** — a donor × configuration failing on ≥ 12 of 14 realisations.
  A different mechanism with a different remedy (stage 1's D10 loses events
  before clustering; a lower threshold is the only fix), so it is reported
  separately.
* **[rev2] Eligibility is common to a contrast.** A donor is excluded from a
  baseline-versus-candidate contrast if it is systematic under **either** arm.
  Conditioning each configuration on its own systematic set is biased in both
  directions: at 11 failures all 14 cells stay in the denominator, at 12 the
  donor vanishes, so a configuration could improve its reported rate *by failing
  once more*. The union rule makes both arms describe the same donors, and the
  exclusion applies to the paired test as well as to the descriptive rate.
* **Sporadic** — everything else, on that common population.

**[rev2] Inference is a paired donor bootstrap, not McNemar.** The 14
realisations within a donor share that donor's waveform, amplitude and
placement, so the 196 donor-realisation pairs are clustered, and an exact
McNemar test would assume independence they do not have. Every decision endpoint
gets a **paired difference with a donor-bootstrap CI**; McNemar is still
reported for continuity, explicitly labelled unadjusted, and **no decision rests
on it**.

**[rev2] Inference is conditional on the 14 frozen realisations** and
generalises across donors only. Generalising to other train realisations would
need that axis resampled too, which this design cannot support.

**[rev2] Every endpoint is judged on its own paired CI.** An endpoint is never
compared through another endpoint's p-value — the earlier draft would have
dropped a candidate for a trivial FP increase whenever failure rates happened to
differ, and passed a large FP regression whenever they did not.

Secondary, all with donor-bootstrap CIs: median accuracy, **p10 accuracy**,
**FP p90**, FP max, split rate, **refractory-violation median**.

**[rev2] Multiplicity:** two candidates are tested, so the familywise α of 0.05
is split Bonferroni-wise; each CI is computed at 97.5 %.

Ranking uses median, lower tail, failure probability and FP tail together. Not
best accuracy, not mean accuracy, not KS-good yield.

## Decision rules, written before the data

1. A candidate **replaces** production only if its paired sporadic failure-rate
   difference is negative with a CI excluding zero, **and** it shows no
   regression under rule 2.
2. A candidate is **dropped** on a material regression: FP p90, split rate or
   failure rate worse than production with its own paired CI excluding zero, or
   a new systematic donor. **[rev2] A regression disqualifies even when the
   failure rate is better** — the earlier draft's branch order returned
   "replaces" for a candidate that was simultaneously better and regressed.
3. If no candidate satisfies (1), the outcome is **no threshold change**, and
   the benchmark's realisation sensitivity becomes the finding.
4. **[rev2] If both candidates qualify**, they are ranked by a prespecified
   ordering — failure-rate difference, then FP p90, then split rate. An exact
   tie is **escalated for human decision**, not resolved by the code.

## Power, stated in advance

**[rev2] The earlier power claim is withdrawn.** It assumed 196 independent
pairs per configuration. They are clustered within 14 donors, so the effective
sample size is smaller by a factor that depends on the intra-donor correlation —
unknown before the run. With donors as the unit, the design has **14 independent
clusters**, and a paired donor bootstrap over 14 clusters resolves only large
differences: expect roughly a 5–10 percentage-point resolution rather than 3.

No numeric power claim is made in advance. The width of the observed CI is
itself a reportable result: if it is too wide to separate the configurations,
that is the finding, and rule 3 applies. This is explicitly **not** licence to
add realisations until something separates — adding realisations within the same
14 donors does not add independent clusters.

## What this deliberately does not do

* No fractional threshold cells. The preregistered 8.5/8, 8.5/8.5 and 9/8.5
  interpolation stays unavailable until a stable integer winner exists.
* No held-out donors or snippets. Stage 2 is development evidence; the same 14
  donors that produced the candidates cannot also confirm them.
* No motion arms. Displacement is not the variable; the staircase comparison
  already showed threshold changes do not fix 40 µm fragmentation.
* No production change on stage-2 evidence alone. A winner here still owes L1C,
  held-out static, and matched real-data evaluation.

## [rev2] Validation the run and analysis enforce

* exactly 687 events in every realisation — equality alone would pass if
  upstream admission shifted every train to 686;
* exactly the 14 unique hash-frozen donors, no subsets or duplicates;
* the effective `Th_universal` and `Th_learned` KS4 applied, not the requested
  override, plus `effective_nblocks == 0`;
* a frozen run manifest, compared and refused if changed;
* fail-closed recording deletion and a per-realisation free-space recheck, since
  a silently failed delete would break the single start-up disk guard;
* the analysis refuses anything but 588 unique cells, 14 donors, 14
  realisations, 3 configurations and complete per-cell config triplets.

## Open question for review

**[rev2]** The cost question has changed. Because donors are the independent
unit, more realisations do not buy resolution — 14 donors is the binding
constraint either way. The real options are to accept ~5–10 point resolution on
this cohort, or to widen the donor cohort, which needs new donors and is a
larger piece of work. The proposal keeps 14 × 14 and reports the CI width
honestly rather than implying a precision the design cannot deliver.
