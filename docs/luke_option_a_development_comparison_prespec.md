# Option A development comparison — prespec

Date: 2026-09-06. Status: **frozen before execution.** Nothing here was chosen
after seeing a sorting outcome.

Contract: [`configs/option_a_development_comparison.v1.json`](../configs/option_a_development_comparison.v1.json)
(digest `fd4fca3c…`). Readiness: [assessment](luke_option_a_readiness_assessment.md).

## 1. The question, and what it is not

**Does applying one explicitly identified motion field improve sorting in one
frozen interval and one frozen depth band?**

That is an *operational* question about a specific artifact. It is not a
qualification of the field, not a measurement of motion, and not a ranking of
estimators.

**The absolute gain of this field is unmeasured.** No field on this session has
been compared against known motion; the calibration that would supply one (the
D2b-1 envelope) is retracted. So a positive result here means "applying this
array helped here", never "the motion was corrected" or "the motion was N µm".
A negative result means this array did not help here. Neither generalises to
another session, another window, or another estimator.

## 2. Why this proceeds when `qualify_field` still fails closed

The §5.3 requirements are not all the same kind of thing, and the readiness
assessment separates them. `FieldGate` and `qualify_field` are **unmodified** —
they remain the historical record of the retracted envelope and the production
gate — and this contract does not waive them. It records which requirements are
met and which are not.

### Indispensable implementation integrity — all four verified

These are load-bearing for *any* application of a field to voltage. If one
fails the operation is wrong however accurate the estimate is.

| requirement | status | evidence |
|---|---|---|
| acquisition ↔ recording clock mapping | ✅ verified | 3057.677050340359 s, from SpikeGLX `firstSample`/`imSampRate` and SI `t_starts`, agreeing to the digit |
| supported application domain | ✅ verified locally | frozen band draws only from depth bins with ≥ 0.95 local support |
| displacement polarity | ✅ verified | zero-motion identity holds; +60 µm moves a feature 300.0 → 240.0 µm, −60 µm → 360.0 µm, recovered 59.99999968 µm, symmetric |
| operator behaviour | ✅ verified | SI 0.102.1 `InterpolateMotionRecording` present and exact; KS4 internal correction disabled in **both** arms |

### Provisional scientific criteria — recorded, not waived

Inherited from the retracted D2b-1 envelope. They are acceptance standards, not
integrity requirements, and their replacement does not exist.

| criterion | status |
|---|---|
| `max_abs_displacement_um` ≤ 60 | passes locally (39.0 µm) |
| `min_support_fraction` ≥ 0.95 | **fails whole-probe (0.917)**; met inside the frozen domain |
| `min_split_half_correlation` ≥ 0.80 | **not measurable** — see §3 |
| `max_estimated_gain_error_fraction` ≤ 0.30 | **not evaluable** — gain unmeasured |

**Versioned departure.** Proceeding on the indispensable requirements inside a
frozen local domain, with the provisional criteria explicitly unmet or
unevaluable, is a departure from the old procedure. It is recorded here and in
the contract's `qualification_status.departure_statement`, and it applies to
this development-only comparison alone. It changes no production gate.

## 3. The split-half check, and what it found

Run as instructed, bounded to the frozen window: 555,661 peaks, split into
deterministic interleaved 2 s blocks so both halves span the same 120 s and see
the same slow motion, re-estimated with `dredge_ap` (rigid).

**Both halves returned identically zero displacement**, twice — once with
library defaults and once with the only DREDGE configuration recorded anywhere
on disk. The correlation is therefore undefined, not low.

That configuration is the *withdrawn* rescue sidecar's, and it carries
`"fallback": "identity"`; decision 0013 records that this sidecar degenerated
(`weights_thresh` entirely non-finite) and produced the retracted 1.28 µm
figure. So the zero result is consistent with reproducing a known degeneracy,
and is **not** a measurement of the accepted field.

The concrete finding is upstream of the number:

> **The accepted motion estimates' generating configurations are not recorded
> alongside them.** `dredge_pipeline_results/.../motion/<estimator>/` holds
> `motion.npy`, `time_bins.npy` and `depth_bins.npy` and no request or config.
> The accepted fields therefore cannot be re-estimated, so split-half
> reproducibility cannot be measured for the field that would actually be
> applied.

This does not block the operational question, because the field is identified by
**content digest** rather than by a recipe: the experiment applies a fixed array
that exists on disk. It does bound the conclusion — an unreproducible field
cannot support a claim that generalises past this artifact — and that limit is
stated in the contract.

## 4. What is frozen

**Field.** `ks-motion`, by sha256 of its three arrays. Chosen for the smallest
displacement magnitude of the four (max |d| 39.0 µm in the window, median 120 s
excursion 4.06 µm) — the most conservative choice while gain is unmeasured — and
because it yields the largest usable interior. **Not** chosen for accuracy: no
accuracy evidence exists for any of the four.

**Domain.** Interval **[7200, 7320] s**, depth band **[1810, 3710] µm**, 190 of
384 channels.

The band is the longest contiguous run of 100 µm depth bins with local support
≥ 0.95 inside the interval ([1710, 3810] µm), eroded at both ends by
max |d| + 3σ of the kriging kernel = 39.0 + 60 = 99 µm, so every retained channel
draws only from supported depths. Only two bins fail locally: 1660 µm (0.942)
and 3860 µm (0.125).

Correction is applied on the **full** channel geometry; the depth crop happens
after, per §5.4. `border_mode` is `remove_channels`, a versioned departure from
the recorded `force_extrapolate` starting policy — this contract refuses to
invent values off the probe. The frozen band already excludes every channel that
would need it, so the crop is not a response to anything observed.

**Policy.** kriging, σ = 20 µm, float32 in, int16 out, gain fixed at 1.0. Gain is
not a tuning knob; sweeping it would fit the field to the outcome.

**Control.** Uncorrected, same KS4 settings, same interval, same band. KS4
internal correction is off in **both** arms, so the only difference is the
external application.

**Endpoints.** Amplitude completeness is primary, on matched units, with a 50%
coverage floor below which it is `inconclusive` rather than `pass`. Identity
correspondence and contamination are required support: a completeness
improvement is reportable only when units are matched and refractory violation
has not increased by more than 0.01. Completeness alone is not a result.

**Stop rule.** One execution, 6 h cap. Abandon if the materialized manifest does
not bind source and motion digests, if either arm fails to record
`do_correction=False`/`nblocks=0`, if any channel inside the frozen band is
removed, or if fewer than half the eligible units match between arms.

## 5. Predicted outcomes, recorded before the run

- **Improvement.** External registration is worth one further bounded test at a
  larger scale. It is not promotion, and the gain remains unmeasured.
- **No improvement.** Closes external registration as a development direction
  for this session at this scale. It does **not** license loosening the domain,
  swapping the field, or sweeping the gain.
- **Inconclusive** (coverage below 50%, or units unmatchable). Reported as such.
  A 120 s window at 1000 spikes per window is a real risk here — the v2 healthy
  arm reached only 1.7% coverage on the same window length — and if it happens,
  the honest next step is a longer interval, frozen in advance, not a lower bar.

## 6. What is deliberately not being done

- No gain calibration. It is the binding constraint on qualification and it is a
  research programme, not a step.
- No re-estimation of the field, and no attempt to make it reproducible.
- No use of the closed v2 replay or its 18 unvalidated merges.
- No change to `FieldGate`, `qualify_field`, or any production gate.

## Related records

- [Readiness assessment](luke_option_a_readiness_assessment.md)
- [Build instructions §5](luke_two_motion_pipeline_build_instructions.md)
- [Decision 0013 — imec0 has appreciable rigid motion](decisions/0013-luke-imec0-has-appreciable-rigid-motion.md)
- [Decision 0002 — motion is estimated, never applied](decisions/0002-motion-is-estimated-never-applied.md)
- [v2 result — the closed replay branch](luke_first_pipeline_candidate_v2_result.md)
