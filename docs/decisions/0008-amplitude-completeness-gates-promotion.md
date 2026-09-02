# 0008 — Amplitude completeness gates promotion, and yield alone never does

**Status:** Adopted 2026-09-02
**Supersedes the yield-centred reading of:** [0001](0001-ks4-unwarped-is-the-production-sorter.md)
**Evidence:** [`luke_20250804_imec0_postcuration_evaluation.md`](../luke_20250804_imec0_postcuration_evaluation.md)

## Decision

The rescue configuration is **not** promoted to a universal production default.
The formal `reject_universal_default` verdict remains in force, and it now rests
on two independent grounds rather than one:

1. the prespecified gates it already failed (similar good–good pairs, edge-spike
   fraction), and
2. **amplitude completeness**, a dimension the original gate set did not measure
   at all.

Amplitude completeness becomes a required acceptance dimension. No configuration
is promoted on yield and contamination/refractory metrics alone, because this
result demonstrates those can improve while completeness degrades.

## The finding

Post-curation, matched on the same 10,473.55 s imec0 recording, the rescue
produces the most nominally good units and the *worst* estimated amplitude
completeness of the three configurations:

| Good units >1 Hz | Rescue | Legacy | Claim-mask |
|---|---:|---:|---:|
| Median estimated missingness | 3.07% | 1.16% | **0.82%** |
| Fraction below 10% missingness | 68.8% | 77.9% | **91.8%** |

The rescue has greater typical estimated missingness in **every** reported
cohort. Claim-mask is strongest on completeness throughout, at a large yield
cost (191 KS-good units against the rescue's 301).

The yield increase is also narrower than the headline suggests. Against legacy:
+73 KS-good units (+32%), but only +29 stable good units (+16%), with the stable
fraction *falling* 79.8% → 70.1%, and no increase in >5 Hz or >10 Hz good units.
The gain is concentrated in low-rate units.

## Why this changes the reading of 0001

[0001](0001-ks4-unwarped-is-the-production-sorter.md) argued that yield gains
arriving *alongside fewer assigned spikes* were the load-bearing evidence that
the improvement was not detection inflation. That argument is now known to be
insufficient, not wrong. Fewer total spikes plus more good units is compatible
with a population whose individual units are each less completely detected —
which is what the truncation analysis measures and what it found.

0001's measurements stand. Its interpretation is narrowed by this record.

## What this establishes generally

- Increased good-unit yield can **conceal** poorer amplitude completeness.
- Contamination and refractory-violation improvements do **not** establish
  detection completeness. They are contamination measures, not recall measures.
- Firing-rate-bin occupancy ("stable" units) is **not** an adequate substitute
  for amplitude-based missing-spike estimation. Units can be counted stable by
  time-bin occupancy and still be poorly captured by amplitude fitting.
- Claim masking, previously rejected on yield grounds, remains the strongest
  configuration for completeness. The rejection was reasonable on the evidence
  then available; the trade-off it represents was never measured.

This is [0007](0007-stage-local-validation.md) recurring at the acceptance layer:
an observable the next stage cannot manufacture was missing from the gate set,
so a real regression passed unseen.

## The reframed production question

Not "does rescue increase good-unit yield?" but:

> Can the rescue configuration retain its yield advantage while matching legacy
> or claim-mask amplitude completeness and controlling similar-unit
> proliferation?

The current answer is **no**.

## Unresolved tension — do not paper over it

- The post-curation artifact-aware audit finds **0** strong and **0**
  partial-or-strong duplicate hypotheses.
- Yet the rescue retains **27** similar good–good pairs against 8 (legacy) and
  11 (claim-mask).
- And completeness is worse despite the higher nominal yield.

Subtler fragmentation or partial detection that does not satisfy the current
duplicate-pair criteria is one plausible explanation. Differences in amplitude
scaling, fit behaviour, or unit composition are others. **No mechanism is
established.** The analyses diagnose association and consistency, not cause.

## Required before promotion is reconsidered

1. Make amplitude completeness a **formal** gate — unit-balanced, preferably on
   the >1 Hz good-unit cohort; candidate criteria are median unit missingness
   and the fraction of units below a prespecified threshold.
2. **Recompute all three truncation analyses under one frozen implementation.**
   The stored results were produced at different times and the legacy runs lack
   identity-bound receipts. Until then, small differences are not definitive.
3. **Validate the truncation fitter** before setting any numeric threshold: many
   windows reach its hard 50% ceiling, and the ceiling behaviour, fallback fits,
   amplitude units, and parameter bounds are unaudited.
4. **Manually inspect discordant units** — rescue good units ≥10% missingness,
   >5 Hz units with ceiling-level windows, similar-pair units with poor
   completeness, units near the repaired-channel depth, and units counted stable
   but poorly fit.
5. **Test the fragmentation hypothesis directly** — whether low-amplitude tails
   are reassigned, discarded, or absorbed into noise. Template similarity alone
   is insufficient.
6. **Build a yield-versus-completeness trade-off curve** across intermediate
   detection/artifact/claim settings.
7. **Replicate on other probes and sessions** under the same frozen gates before
   any policy change. Luke0804 imec0 stays a diagnostic case, not the basis for
   production-wide policy.

Items 2 and 3 gate item 1: do not formalize a numeric threshold on top of an
unvalidated fitter and unmatched recomputations.

## Confidence

**High** that the rescue must not be promoted now — it fails prespecified gates
and carries an independent material concern.

**Moderate** on the exact ranking and magnitude of the truncation differences:
only units with supported 1,000-spike windows are estimated (110/301 rescue,
78/228 legacy, 72/191 claim-mask); the fitter caps at 50%; comparator results
were generated at different times without equivalent provenance; and KS-good
labels are automated, not manually adjudicated.

These caveats define the validation still owed. They do not remove the red flag.
