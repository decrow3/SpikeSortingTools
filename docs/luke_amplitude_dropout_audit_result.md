# Amplitude-completeness dropout audit — result

Date: 2026-09-06. Status: **executed on real data. The nomination rule was
corrected under a prespec written beforehand and re-run (§7a); the evidence
reading is unchanged and is still not motion evidence.** See §6 before acting
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

**(a) The nomination picked by frozen case order, not evidence strength —
corrected under a prespec, and re-run.**
`_nominate` returned the first eligible case in the freeze's order, which is
alphabetical by sort ID. It therefore nominated `legacy…c424__failure2` —
depth shift 0.84 µm, amplitude drop 15.4% (barely over the 15% threshold),
rank 2, voltage unavailable — over `rescue…c37__failure1`, which has rank 1,
double the amplitude effect, the largest depth shift of the supported set, and
a completed voltage review.

It was deliberately **not** fixed before reporting: changing a decision rule
once its answer is visible voids the point of freezing constants before a
ranking is read. The corrected rule was written down first
([prespec](luke_amplitude_dropout_audit_nomination_rule_prespec.md), `360eb1e`)
and executed only on the user's explicit go-ahead (`81b128d`).

Its basis predates every ranking: the legacy sort's raw voltage was deleted
after sorting, so a legacy case's voltage limb is uncollectable *in principle*.
That was recorded at 14:14:35 (`701d890`) and 14:21:01 (`bfce40c`, the commit
that froze case selection), while the first ranking was written at 19:35:11.
The rule prefers executable evidence, then the frozen rank, then the larger
effect on the supported limb, then ids — none of which references which case it
selects.

**Both answers are on record:**

| run | rule | nominated |
|---|---|---|
| `run1`, `run2` | first eligible in frozen order | `legacy…c424__failure2` |
| `run3_corrected_rule` | prespec `360eb1e` | `rescue…c37__failure1` |

The re-run is auditable because everything else held: **identical case IDs,
identical exclusion counts and identical evidence readings across all three
runs**, with only `decision.md` differing. That separates "the decision rule
changed" from "the selection changed" — had the case set moved, the re-run
would not have been clean.

**Known limitation — the nomination's basis is not in `decision.md`.** That
file names the nominated case but records neither the rule nor the value
criterion 1 consumed. Both are on disk, in `manifest.json`:
`inspect.evidence_extra_inputs.voltage.raw_voltage_available_by_sort` and
`inspect.voltage_review.raw_voltage_available_by_sort`, each reading
`{legacy: false, rescue: true}`. The map is evaluated once during `inspect`
and the nomination consumes the recorded value rather than re-querying the
filesystem, so the basis stays readable after the disk moves on. Anyone
quoting `decision.md` onward should carry that pointer with it.

This was left unfixed rather than patched. `decision.md` is generated output
whose `decision_sha256` is attested in `manifest.json`, so hand-editing it
would desync the hash and corrupt the run's attestation — the note belongs
here, not there. Regenerating it properly means a fourth output root, which
is not worth spending on a provenance line; the writer should record its own
rule when some later change touches it anyway.

**(b) The voltage excerpts were rendered without event markers** (fixed in
`2186b3b`, before the `run2` figures). Found by looking at the rendered figure,
not by a test: the four excerpts were visually near-identical and nothing in
the output distinguished data from bug.

## 8. Decision

The audit executed successfully and, under the corrected rule, nominates
**`rescue…c37__failure1`** for a bounded registration-or-identity experiment.
Successful execution is not the same as an interpretable result: §6 still
governs what that nomination may be read as. The nomination says *which case*
to take forward; it does not upgrade the evidence about *why* the case fails.

What is supported: on four selected failure cases, a large rise in estimated
missingness is accompanied by an amplitude collapse, with depth essentially
stationary, no rows dropped by the retained-array mask, and no voltage
saturation on the two cases where voltage exists.

What remains ambiguous: whether that amplitude collapse is motion the depth
estimate does not capture, identity/assignment failure, or a change in the
unit itself. Nothing here separates them, and identity redistribution was never
attempted.

## 9. Next implementation action

1. **Done** (§7a): the corrected rule is recorded in its prespec, implemented,
   and re-run into `run3_corrected_rule`, with the case set and exclusion
   counts verified unchanged.
2. Write the experiment contract for `rescue…c37__failure1`. Two limbs
   converge on **identity handling** as the better-targeted half of that
   category's next action, and neither depends on the category's name:
   registration has little to act on because depth is flat on every case
   (0.84–3.01 µm against a frozen 8 µm threshold), and curation repair is
   removed because 0 of 2,000 labelled rows in every failing span were dropped
   by `kept_spikes` — a negative that needs no KS4 semantics, unlike the
   positive claim. The contract's margins must come from this case's baseline
   evidence and be fixed before any candidate result is seen.
3. Open, and not to be settled by re-running: whether a depth-limb requirement
   (or a separate reading for amplitude-only support) belongs in the evidence
   constants. That is a prespec change to a frozen constant's semantics, it
   would invalidate comparison with the three runs on record, and it must not
   be made in order to obtain a different answer from them.

   **Step 2 does not wait on step 3, and does not depend on its answer.** The
   steer to identity handling is built from two *exclusions* — flat depth, and
   no rows dropped by `kept_spikes` — not from `motion_amplitude_change`
   reaching a supported verdict. Both exclusions hold whatever this category
   turns out to mean, or should have been called. A later reader should not
   infer that the steer rested on the verdict that step 3 questions.

## Related records

- [Prescription](amplitude_completeness_next_step_prescription.md)
- [Improvement plan](pipeline_improvement_plan.md)
- [Truncation fitter audit / decision 0009](luke_20250804_truncation_fitter_audit.md)
- [Exclusive matching and detection evidence](decisions/0011-cross-sort-event-matching-and-detection-evidence.md)
