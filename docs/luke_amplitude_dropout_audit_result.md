# Amplitude-completeness dropout audit — result

Date: 2026-09-06. Status: **executed on real data; the decision it produced is
not yet scientifically interpretable as motion evidence.** See §6 before acting
on the nomination.

Prescription: [amplitude_completeness_next_step_prescription.md](amplitude_completeness_next_step_prescription.md).
Implementation: `testing/luke_amplitude_dropout_audit.py` (+ tests), commits
`9a3a5bf` → `2186b3b`. Outputs: `testing/outputs/luke_amplitude_dropout_audit/`
(gitignored; `run1` and `run2` are two independent runs of the same inputs).

## 1. What ran

`inventory` → `select` → `inspect` on the frozen config
`testing/configs/luke_amplitude_dropout_audit_v1.json`, over the two
pre-existing Luke 2025-08-04 imec0 sorts (rescue and legacy). No sorting was
run, no cache was repaired, and no production QC was modified.

Case selection had never been run against real data before this. It was run
only after the delivery contract's interval gate landed (`9a3a5bf`), so the
first ranking anyone has seen was already confined to permitted development
windows and judged against constants frozen beforehand.

## 2. Inventory

| | |
|---|---|
| windows normalized from cached QC | 57,762 |
| boundary-pinned (censored at 50%) | 31,766 (55.0%) |
| finite-interior | 25,333 (43.9%) |
| `no_fit` clusters (no cached window) | 663 |

The boundary-pinned fraction matches the already-published figures for these
sorts (54.9% rescue / 56.3% legacy, decision 0009), which is the cross-check
that the loader reproduces production QC's own semantics rather than a
re-derivation of it.

## 3. Selection — and what the interval gate excluded

Six cases, exactly at the caps: two failure cases and one diagnostic control
per sort.

| case | ref → failing missing % | Δ pp | span s | dev window |
|---|---|---|---|---|
| `legacy…c423__failure1` | 0.51 → 39.85 | 39.34 | 479 | 5 |
| `legacy…c424__failure2` | 3.06 → 32.66 | 29.60 | 443 | 5 |
| `rescue…c37__failure1` | 0.67 → 43.47 | 42.81 | 220 | 5 |
| `rescue…c549__failure2` | 0.62 → 41.39 | 40.77 | 160 | 5 |

Plus `legacy…c32__control1` and `rescue…c666__control1`.

**The gate bit.** 19 failure runs and 1,869 control runs cleared every
threshold but were excluded as `outside_development_window` — they lay in the
sealed held-out panel, its 300 s buffer, or a reserved healthy evaluation
interval. Without the gate those 19 would have been ranked, and a stronger one
could have taken a capped slot. 66.8% of the recording (6,993.6 s of 10,473.6 s)
is permitted.

All four failure cases landed in development window 5 (`[5760, 8700] s`). That
is worth noting as a possible clustering in time, not interpreted here.

**Reproducibility:** two independent runs (`run1`, `run2`, different module
hashes) produced identical case IDs and identical exclusion counts.

## 4. Replay — acceptance test 5 on real data

All 24 case windows reproduced their cached historical missing-percentage:
max |difference| = **3.6e-15 pp**, against a tolerance of rtol/atol 1e-6. No
input/runtime mismatch; no case's interpretation was stopped.

The exact-1,000 sensitivity (fitting `amps[i0:i1+1]` rather than production's
`amps[i0:i1]`) moved estimates by a median of 0.006 pp and at most 0.31 pp, and
**flipped no case's eligibility**. On these cases the 999-vs-1,000 indexing
discrepancy is real but immaterial to the answer.

## 5. Evidence

| case | curation | identity | motion/amplitude | voltage | reading |
|---|---|---|---|---|---|
| `legacy…c423__failure1` | unresolved | not_attempted | unsupported | unavailable | unresolved |
| `legacy…c424__failure2` | unresolved | not_attempted | **supported** | unavailable | motion_amplitude_change |
| `legacy…c32__control1` | unavailable | not_attempted | unavailable | unavailable | unresolved |
| `rescue…c37__failure1` | unresolved | not_attempted | **supported** | unsupported | motion_amplitude_change |
| `rescue…c549__failure2` | unresolved | not_attempted | **supported** | unavailable | motion_amplitude_change |
| `rescue…c666__control1` | unavailable | not_attempted | unavailable | unsupported | unresolved |

**Which evidence was unavailable, and why:**

- **Curation** is `unresolved` on every failure case, not because nothing was
  measured but because the measurement licenses no stage claim. The retained-row
  lineage is striking and worth recording: **0 of 2,000** full-table rows
  carrying the case cluster's own full-sort label inside the failing span are
  absent from `kept_spikes`, on all four failure cases. Nothing was dropped
  there. Calling that "curation did not exclude anything" still requires
  establishing retained-array semantics against the installed KS4 source, which
  this checkpoint did not do.
- **Identity redistribution** is `not_attempted` on every case: there is no
  frozen shift-null protocol here, and the prescription forbids an identity
  claim from time coincidence alone. Not attempted is not evidence of absence.
- **Voltage** was reviewed for the two highest-ranked failure cases and their
  sorts' controls. The two rescue cases were read from the real 241 GB
  recording — 400 events each on 16 channels frozen from the reference windows,
  0 clipped frames, **0.000%** of read samples at or beyond the 500 µV blanking
  threshold. The two legacy cases are `unavailable`: that sort's
  `traces_cached_seg0.raw` was deleted after sorting. That is the expected,
  permitted outcome; it was not reconstructed and the rescue recording was not
  substituted.

## 6. What is actually supported — read this before acting

Three failure cases read `motion_amplitude_change`. **None of them has motion
evidence.** Every measured depth shift is far below the frozen 8 µm threshold:

| case | median depth shift | median amplitude drop | verdict rests on |
|---|---|---|---|
| `legacy…c423__failure1` | +3.01 µm | +1.4% | — (unsupported) |
| `legacy…c424__failure2` | +0.84 µm | +15.4% | amplitude only |
| `rescue…c37__failure1` | +2.81 µm | +30.2% | amplitude only |
| `rescue…c549__failure2` | +1.14 µm | +28.8% | amplitude only |

Every supported verdict rests entirely on the amplitude limb of the category.
The `rescue…c37` panel shows this plainly: the amplitude density collapses from
~15–20 to ~10–13 sorter-native units while the depth scatter stays at ~75–85 µm
throughout, and the bounded voltage review shows the same waveform shape with a
visibly shallower trough in the failing windows.

Two consequences:

1. **"motion_amplitude_change" here means amplitude change, not motion.** The
   prescription's row groups amplitude and depth together and routes both to
   "a bounded existing registration or identity experiment". With depth flat,
   the *registration* half of that next action has little to act on; the
   identity half does. A reader who takes the category label as motion evidence
   would be misreading it.
2. **The amplitude limb is not independent of what selected the case.** It is
   measured on the same amplitude distribution whose truncation defined the
   missingness. It is not a restatement — `c423` dissociates them, rising 39 pp
   in missingness with only a 1.4% median drop — but it cannot distinguish
   motion from any other cause of amplitude loss, and it should not be read as
   mechanism evidence.

## 7. Two defects in the implementation, found by this run

**(a) The nomination picks by frozen case order, not evidence strength.**
`_nominate` returns the first eligible case in the freeze's order, which is
alphabetical by sort ID. It therefore nominated `legacy…c424__failure2` —
depth shift 0.84 µm, amplitude drop 15.4% (barely over the 15% threshold),
rank 2, voltage unavailable — over `rescue…c37__failure1`, which has rank 1,
double the amplitude effect, the largest depth shift of the supported set, and
a completed voltage review. That is hard to defend as "the audit's nomination".

This has **not** been fixed and re-run: changing a decision rule after seeing
the first ranking is exactly what the prespec discipline forbids, and the
executed answer stands on record. The fix and the re-run are a decision for the
user, and the corrected rule should be written down before it is executed.

**(b) The voltage excerpts were rendered without event markers** (fixed in
`2186b3b`, before the `run2` figures). Found by looking at the rendered figure,
not by a test: the four excerpts were visually near-identical and nothing in
the output distinguished data from bug.

## 8. Decision

The audit executed successfully and produced a nomination
(`legacy…c424__failure2`, a bounded registration-or-identity experiment). Given
§6 and §7(a), **that nomination should not be carried into an experiment as
written.** Successful execution is not the same as an interpretable result.

What is supported: on four selected failure cases, a large rise in estimated
missingness is accompanied by an amplitude collapse, with depth essentially
stationary, no rows dropped by the retained-array mask, and no voltage
saturation on the two cases where voltage exists.

What remains ambiguous: whether that amplitude collapse is motion the depth
estimate does not capture, identity/assignment failure, or a change in the
unit itself. Nothing here separates them, and identity redistribution was never
attempted.

## 9. Next implementation action

1. Decide and record the corrected nomination rule (§7a), then re-run
   `select`/`inspect` into a fresh output root. The rule must be written before
   the re-run.
2. If the nomination then lands on `rescue…c37__failure1`, the better-targeted
   half of its next action is the identity experiment, not registration —
   depth is flat on every case. The experiment JSON must fix its margins from
   this case's baseline evidence, before any candidate result is seen.
3. Consider whether a depth-limb requirement (or a separate reading for
   amplitude-only support) belongs in the evidence constants. That is a
   prespec change, not an implementation fix, and it must not be made to obtain
   a different answer from the run already on record.

## Related records

- [Prescription](amplitude_completeness_next_step_prescription.md)
- [Improvement plan](pipeline_improvement_plan.md)
- [Truncation fitter audit / decision 0009](luke_20250804_truncation_fitter_audit.md)
- [Exclusive matching and detection evidence](decisions/0011-cross-sort-event-matching-and-detection-evidence.md)
