# Plan: improve spike recovery throughout the recording, and reach decisions faster

## Current direction — 2026-09-05

The practical goal is reliable recovery of individual neurons' spike trains
throughout the recording. Progress means fixing a recognizable failure,
preserving healthy periods, and confirming the improvement on independently
chosen data. Unit yield, benchmark scores and explanations support that goal;
none alone establishes that a pipeline is better for the lab's recordings.

**The immediate milestone is one reproducible improvement in a real dropout
case, with preserved healthy intervals, followed by confirmation on an
independently chosen case.** This is a development milestone, not production
promotion or a replacement for the held-out and second-session gates below.

### Use the lab's diagnostic to choose the next question

Time-resolved amplitude completeness is a central diagnostic: fit the truncated
amplitude distribution in nominal 1,000-spike windows to estimate missing-spike
percentage. In the lab's experience this reveals loss of capture more usefully
than spike rate alone, which also changes with neural activity. Select real
failure periods with this signal, then distinguish amplitude loss, identity
redistribution, curation exclusion and voltage artifacts using supporting evidence.

The estimate is model-dependent. Show window time coverage, no-fit gaps and
boundary-pinned fits; completely missing periods cannot supply a fit. Compare
the same supported neuron identity and physical-time interval, and require
waveform and contamination evidence before interpreting lower missingness as
improvement. Preserve the distinction between historical QC's 999-value slice
and an explicitly versioned exact-1,000 calculation.

The bounded [amplitude-completeness implementation prescription](amplitude_completeness_next_step_prescription.md)
defines the next diagnostic checkpoint: at most four failure cases and two
controls, two working days of implementation/review effort, and one nominated
intervention or an explicit insufficient-evidence conclusion. It does not open
a new programme or block already-authorized motion implementation work.

### Separate three kinds of work

| Work | Required outcome | Stopping condition |
|---|---|---|
| Operational improvement | Better supported recovery with bounded costs on relevant data | Advance through the existing ladder, reject the candidate, or retain the baseline |
| Mechanism research | Evidence distinguishing specific causal explanations | Answer the stated question or record its unresolved limit; do not gate unrelated improvements |
| Evaluation implementation | A trustworthy measurement needed for the next decision | Validate the smallest necessary interfaces and use them; avoid general framework expansion |

A development screen and a production replacement require different amounts
of evidence. Cheap screens may nominate a candidate; promotion still requires
the frozen criteria. A complete mechanism is not an additional promotion gate.
Conversely, a synthetic benchmark must demonstrate relevance to an observed
failure before its ranking justifies substantial new development. Synthetic
truth remains necessary evidence for the claims it can actually test.

### Limit work in progress and make each experiment decide something

- External voltage registration (A) and unwarped motion-aware identity handling
  (B) remain the two development options. Give each one minimal demonstration
  and an explicit continuation/stop rule before expanding it. The next expensive
  experiment should follow the strongest actionable evidence, not an obligation
  to complete both architectures before learning anything useful.
- Every experiment states the decision it informs, the observation that would
  distinguish alternatives, its resource cap, and actions for success, failure
  and ambiguity. If all outcomes mean more investigation, narrow the question.
- Freeze settings, comparison intervals and numerical decision margins before
  candidate results. An ambiguous result does not automatically authorize a
  larger panel, another variant or a full-session run.
- Implement small, independently testable contracts for amplitude sourcing,
  window bounds, clocks, geometry, identity matching and provenance. Known-answer
  fixtures at these interfaces should precede an end-to-end run; repeated broad
  review is not a substitute for them.
- Protect task-relevant spikes: inspect condition-dependent loss, weak events,
  bursts, collisions and artifact-adjacent periods when applicable to the frozen
  comparison. Aggregate QC can improve while scientifically useful spikes worsen.

Production remains the frozen rescue graph at 12/9. Threshold Stage 2 ended in
`no_threshold_change`; its lower candidate failure-rate point estimates neither
establish equivalence nor reopen the closed 8/8 and 9/9 gates. Dense-donor motion
modelling remains a separate scientific task. Existing historical corrections,
experiment prespecifications and production promotion rules remain in force.

The delivery sequence below governs execution order; §9 inventories available
work rather than requiring every item before a release. The evidence chronology
explains how we arrived here; dated priorities within it are historical.

## Delivery sequence: a first updated pipeline we can actually use

**Build one credible candidate, establish a meaningful improvement, freeze it,
and use it.** The first version may leave optimizations and mechanisms unresolved.
“Step change” is an intended outcome, not a claim established by the existing
rescue results or by successful software integration. It must be demonstrated
against the pre-existing pipeline on recovery and downstream-relevant evidence.

This sequence supersedes older instructions to complete both motion options,
find the best conventional pipeline, or exhaust diagnostic avenues before a
first candidate can advance. It does not waive frozen experiment decisions,
operator integrity checks, or the production-promotion criteria in §6 Phase E.

### Step 1 — freeze a small first-version delivery contract

Create `configs/first_pipeline_candidate.v1.json` before new candidate runs.
Record the exact pre-existing legacy comparator and current rescue control,
recording/sort identities, development windows, healthy controls, output root,
candidate settings and runtime budget. Name the practical recovery failure and
the minimum improvement that would justify adoption, with numerical margins for
completeness, identity, contamination and healthy-interval preservation. Set
those margins from baseline evidence before inspecting candidate results;
“better looking” or a statistically detectable tiny change is insufficient.

Reuse the existing ladder, data, QC and frozen evaluation gates. State which
unresolved implementation dependencies are required for this particular candidate.
Do not expand the contract into a comprehensive framework or redefine the old
threshold experiment. Deliverable: one executable comparison contract, not a
list of potential experiments. Contract omissions block execution, not planning
or implementation of independent components.

### Step 2 — choose one candidate when the bounded audit ends

Use the amplitude-completeness audit within its two-working-day cap to choose
the intervention with the clearest supporting evidence. A targeted curation
repair can be the first delivery if it directly fixes an observed exclusion;
the first release need not contain a new motion architecture.

If the audit remains ambiguous, stop it and proceed with the minimum Option B
implementation: qualified motion coordinates plus conservative, reversible
identity links, retaining original voltage, spike times and cluster labels.
This is an engineering default because it can reuse an existing sort; it is not
a claim that B is scientifically superior. If B lacks a field or cannot pass
its identity/union-cleanliness controls, choose A only if its required field and
operator checks pass. Do not manufacture links or apply an unsupported field to
meet the schedule. If neither is admissible, record the single blocking contract
and repair that prerequisite; stop unrelated diagnostic work.

Only one candidate receives active integration priority. Preserve the other
option's existing work and schedule a return condition; its completion is not
a prerequisite for releasing the first useful candidate. This also means that
a B-versus-legacy result cannot be called superiority over all voltage correction.

### Step 3 — make that candidate run end to end on bounded data

Use one thin runner/configuration to execute accepted input → candidate change
→ sorting or retained-sort replay → curation → waveform/refractory/amplitude QC
→ standard analysis export. Reuse `pipeline/`, `testing/ladder_l1.py`,
`testing/ladder_sorter.py` and the relevant existing motion implementation.
For a motion candidate, use the runner contract in the
[motion build instructions](luke_two_motion_pipeline_build_instructions.md).
For a curation-only candidate, use retained-output replay through the same
downstream interfaces. Do not fork the whole production pipeline.

Deliver a named candidate version, resolved configuration, manifests, restartable
outputs, standard exports, and one command documented from input through QC.
For B, exports must explicitly distinguish original clusters and proposed
families. Evaluate the actual train used downstream; a coordinate plot or family
annotation alone is not a recovered spike train. Ambiguous links stay separate.
Keep all outputs separate from accepted production results.

Pass known-answer interface tests and one end-to-end smoke run before expansion.
This is the first legitimate runnable attempt; it is allowed to fail its scientific
screen. Its completion must not be reported as a proven pipeline improvement.

### Step 4 — screen the integrated candidate, then freeze its settings

Compare against both the specified legacy comparator and rescue control on the
selected failure and healthy intervals. Use supported same-neuron comparisons,
time-resolved completeness and identity evidence, plus the common truth scores
and contamination/refractory guardrails. Require the Step 1 practical margins.
Confirm a promising result on an independently chosen development case without
retuning, then apply L1C and L2 as required by the existing ladder. The new case
must not silently consume the sealed held-out panel.

Permit at most one mechanism-directed revision after the first scientific
screen, within the existing candidate-count caps. Record both attempts. A failed
hard integrity check must be fixed before any scientific interpretation; it is
not another parameter trial. If the candidate still fails, close its attempt
and make a written switch-or-stop decision. Do not turn failure into an automatic
sweep. If it passes, freeze the version and stop optimizing it during validation.

### Step 5 — validate the frozen first version and make the adoption decision

Run the surviving version through L2L longitudinal continuity, the sealed
held-out panel and second-session replication, with the condition-dependent
recovery checks and all Phase E promotion criteria. Only after those gates pass,
run L4 at full-session scale under the existing budget and review its outputs
before adoption. Retain failed and unresolved safeguards explicitly; they are
not evidence of preserved neurons. No new mechanism explanation or exhaustive
alternative-sorter comparison is added as a gate.

The first-version decision is adopt within the validated domain, reject, or
retain as a research candidate with a stated limitation. A successful version
gets a frozen configuration, environment receipt, run instructions, validation
summary, known limitations and a route to reproduce the legacy comparison.
It need not be the best achievable pipeline to be worth adopting.

### Step 6 — use the version to measure effects on downstream analyses

Once the frozen candidate has bounded validation and compatible exports, a
bounded downstream comparison may run on those existing outputs alongside
remaining validation. Full-session comparisons wait for Step 5's L4 gate;
routine replacement of the analysis input waits for adoption.

Freeze the downstream analysis code, trial selection, behavioral/stimulus inputs
and inclusion rules before comparing pipeline outputs. Change only the sorting
input. Report matched-neuron analyses separately from population analyses that
include newly admitted or lost units. Assess changes in the lab's selected
endpoints, their uncertainty, and whether changes concentrate in periods of
improved completeness. Choose those endpoints from actual analysis requirements
in the delivery contract; do not invent a new downstream suite here.

Changed tuning, decoding or effect sizes do not by themselves prove a better
sort. The purpose is to quantify scientific consequences of independently
supported recovery improvements. Deliver a paired analysis report with versioned
inputs, then decide whether any remaining issue is consequential enough to reopen.

### Deferred questions: explicitly allowed to remain open

Maintain a short deferred list in the candidate's delivery record, with question,
current evidence, why it is unnecessary for this release, and a return trigger.
These are initial defaults, not additional work packages:

| Deferred work | Return trigger |
|---|---|
| Dense spatial donors and Luke-scale fractional-motion mechanism | A later motion claim or optimization actually requires that causal calibration |
| Completing/optimizing the other motion architecture | The first candidate fails, or a validated residual deficit warrants a challenger |
| Broad preprocessing, interpolation or sorter searches | A reproducible remaining failure survives the released candidate's relevant controls |
| Exhaustive explanation of every dropout, similar pair or polarity pattern | It breaches a release guardrail or materially affects a downstream conclusion |
| Further threshold exploration | A separately justified future protocol; the closed Stage 2 gates remain closed |
| General tooling and dashboard improvements | Repeated use of the first pipeline exposes a concrete maintenance or decision bottleneck |

Questions that invalidate the candidate's measurements, identity claims, waveform
integrity or frozen acceptance criteria cannot be deferred as “optimization.”
Everything else may remain documented and unexplored while a useful first version
is built, validated and used. The release milestone is a demonstrated practical
advance, not exhaustion of the hypothesis list.

## Evidence chronology and qualifications

> **HISTORICAL EVIDENCE CORRECTION — 2026-09-03.** The original C2,
> Candidate 2, D2b-1, D2b-3, and Checkpoint C panel comparisons are
> **retracted**. Their motion
> injection used contiguous channel-index shifts on a four-column probe while
> oracle correction used continuous physical y-motion; the operators were not
> inverses. Ground-truth matching was non-exclusive, and injected-recording
> cache identity did not include voltage content. The donor cohort and ladder
> infrastructure remain useful, but no drift penalty, interpolation ceiling,
> field-tolerance envelope, kernel conclusion, or rescue-versus-legacy panel
> direction from those runs is established. Those historical results remain
> withdrawn. The separately frozen C2 v4 has since supplied a valid exact
> 40 µm control, described below; it does not rehabilitate the old runs or
> establish a Luke-scale result. D2a implementation is no longer waiting on
> C2, while quantitative Luke-scale simulation awaits a suitable dense donor.

> **EARLIER-EVIDENCE CORRECTION — 2026-09-03.** Phase A, Phase A2, the
> matched-unit truncation comparison, and full-session stitching were
> **retracted pending rerun** under [decision 0011](decisions/0011-cross-sort-event-matching-and-detection-evidence.md).
> Cross-sort matching reused target spikes; whole-probe ±0.5 ms coincidence had
> an 87–89% chance-coverage baseline; and A2 scored merge cleanliness on the
> anchor rather than the actual fragment union. The old +200/-127, 80/85/35 and
> 27/100 decompositions, and 92–95% clean merges are historical only.
>
> **V2 RERUN COMPLETE — 2026-09-03.** Phase A (both audits) and Phase A2 have
> been re-run with exclusive one-to-one matching and a depth-windowed,
> circular-shift-null detection gate. Results:
> - **Phase A gives a bounded negative result.** Cohort is now +210 / −137.
>   The audit supports cross-sort detection overlap for 208/210 rescue-side and
>   132/137 legacy-side units; two and five remain unresolved. No new or lost
>   detection is confirmed, but equivalence and superiority are not established.
> - **Phase A2's load-bearing finding is reversed.** The prespecified
>   coexisting-fragment signature is absent (0.0% both probes), but this does not
>   rule out every form of over-splitting. When refractory violations are
>   measured on the actual fragment-cluster union only **6–13%** of merges are
>   clean (was 92–95% on the anchor). The "fragments are one clean neuron /
>   stitching would recover them" reading is withdrawn. About 90% of families
>   remain mechanistically ambiguous.
>
> The historical D2b chain remains retracted. Historical C2 is superseded only
> by the separately frozen v4 result and only within v4's stated scope. No
> claim that the new pipeline beats legacy is currently supported.

> **C2 DONOR DECISION — 2026-09-03.** Geometry-correct C2 v2 is retired without
> rerun because it still froze the discredited T01/T04/T06 plateau donors. Per
> [decision 0012](decisions/0012-c2-uses-compact-donor-cohort.md), C2 v3 first
> introduced all 14 hash-frozen D2b-2 compact donors; C2 v4 retains them. A
> donor enters an inferential contrast only when its static arm reaches the
> prespecified accuracy floor under both configurations in that contrast.
> Consequently the primary correction comparison (`rescue` versus
> `rescue_rigid`) uses the common qualified cohort reported in the v4 result;
> the `legacy_style` comparison has its own qualified cohort.

> **CURRENT C2 STATUS — 2026-09-04.** C2 v4 is complete across 14 compact
> donors and is the first run to survive the operator, scorer, donor and cache
> controls ([result](luke_20250804_c2_v4_result.md)). The exact 40 µm staircase
> establishes that rigid displacement can fragment identity (13/14 donors) and
> that rigid correction can recover it. The 5/11/22 µm arms do **not** answer the
> Luke-scale question: fractional resampling dims the compact recorded
> footprints instead of translating them. A dense spatial donor model is the
> next requirement for that causal question, but not a pipeline-development
> gate. The truncation-vs-truth follow-up supports positional
> identity splitting at 40 µm and is explicitly diagnostic, not production QC.

> **TWO-OPTION DEVELOPMENT DECISION — 2026-09-03.** “No external voltage
> warp” is the shared control and current operational reference; it is **not**
> one of the two development bets. The candidate programme must build and test
> both (A) a physically validated external voltage-registration pipeline and
> (B) an unwarped, motion-aware identity pipeline. Historical external-warp
> failures reject those implementations, fields and operating points—not the
> standard approach as a class. Implementation and smoke testing may proceed in
> parallel. C2 v4 now supplies a validated 40 µm machinery control, but not a
> Luke-scale causal calibration; promotion still requires the shared ladder and
> real-data confirmation. Detailed build instructions:
> [`luke_two_motion_pipeline_build_instructions.md`](luke_two_motion_pipeline_build_instructions.md).

> **CURRENT FAST ITERATION — 2026-09-05.** The 154-cell static
> detection-threshold sweep is complete across the 14 compact donors, with
> correction off everywhere. It establishes that D10's 12/9 failure is a
> threshold effect: 9/8 changes accuracy from 0.475 to 0.976 and FP from 275 to
> 6. But 9/8 breaks D14 (accuracy 0.744; 229 FP), so it is not a replacement.
> The response surface has deterministic cliffs: 15 independent re-sorts of
> three collapse cells and two controls were bit-identical. **8/8** is the
> robustness candidate (12/14 recovered, no donor below 0.976, 12 total FP);
> **9/9** is the recovery candidate (13/14 recovered, worst donor 0.891, 97
> total FP). Production 12/9 recovered 11/14, with a 0.475 worst donor and 289
> total FP. These are development candidates, not an authorized pipeline
> change. The exact-staircase rerun then invalidated that single-train ranking:
> removing 21 events changed total static FP from 12 to 1244 for 8/8 and from
> 97 to 896 for 9/9. A 105-cell sentinel showed that the cliffs follow injected
> train composition and timing phase, not event count. Excluding D10's distinct
> systematic detection failure, the three configurations are not separated.
> Production 12/9 therefore remains the operational baseline by default, not
> because it won. The frozen 588-cell Stage 2 distributional comparison is now
> complete ([result](luke_c2_train_stability_stage2_result.md)). Both candidates
> improved the failure-rate point estimate (8/8: −0.112; 9/9: −0.102), but their
> paired 97.5% donor-bootstrap intervals crossed zero. Neither qualified, so the
> preregistered decision is **no threshold change**. Fractional refinement, L1C,
> held-out static and matched-real-data evaluation do not open for these
> candidates; the threshold branch closes and effort returns to Options A and B.
> Dense-donor modelling remains background science and does not gate this path.

**Status:** active, updated 2026-09-05
**Supersedes as a work plan:** the follow-up lists in
[`decisions/0008`](decisions/0008-amplitude-completeness-gates-promotion.md) and
[`decisions/0010`](decisions/0010-rescue-yield-is-relabelling-not-detection.md)
**Related:** [`0006`](decisions/0006-recovery-axis-is-post-sort-mua-reconciliation.md),
[`0007`](decisions/0007-stage-local-validation.md),
[`0009`](decisions/0009-cross-sort-comparisons-must-be-unit-matched.md),
[`0011`](decisions/0011-cross-sort-event-matching-and-detection-evidence.md),
[`0013`](decisions/0013-luke-imec0-has-appreciable-rigid-motion.md),
[`0014`](decisions/0014-injected-truth-scoring-is-per-cluster.md), and
[`0015`](decisions/0015-corrected-cross-sort-audits-do-not-establish-equivalence.md).
**Implementation companion:**
[`luke_two_motion_pipeline_build_instructions.md`](luke_two_motion_pipeline_build_instructions.md).

## 1. The goal, as a testable claim

> A candidate pipeline is **operationally promotable** when, on data it has
> never been tuned against, it recovers more known-identity neurons correctly
> than legacy does, has only prespecified and bounded consequential regressions,
> preserves its advantage across relevant sorting contexts and a second
> session, and does not cost materially more runtime.

Four clauses, each independently falsifiable. All four must hold. A complete
mechanistic explanation is desirable but is **not** a fifth operational gate;
mechanistic claims require their own evidence and may remain narrower than the
adoption claim. Yield is not
one of them — [`0010`](decisions/0010-rescue-yield-is-relabelling-not-detection.md)
previously described a +32% KS-good headline as entirely relabelling; that
empirical decomposition is now retracted by 0011.

## 2. Why the last attempt took days to fail

Not because the science was wrong. Because of the evaluation architecture:

| Problem | Consequence |
|---|---|
| Every question was asked at full-session scale | Days per answer |
| Endpoints were population counts (KS-good, spikes) | Confounded by composition; answered the wrong question |
| Per-stage audits and end-to-end runs were separate efforts | Stage evidence cost extra days |
| No known-truth reference | Could not distinguish "more units" from "better units" |
| Comparators re-run at different times | Provenance gaps forced re-litigation |

The fix is structural, not a matter of working faster: **make the cheap tier
answer the same question the expensive tier does**, and forbid the expensive
tier until the cheap tier passes.

## 3. The evaluation ladder

The core deliverable. Five tiers, each with a hard gate. Nothing runs at tier
N+1 until it passes tier N.

| Tier | Scope | Target wall clock | Runs when |
|---|---|---|---|
| **L0** | Unit + contract tests | **< 1 min** | Every commit; use the current test run, not a fixed historical count |
| **L1** | One snippet, full pipeline, scored | **< 5 min** | Every parameter/code change |
| **L1C** | One frozen spatial/temporal context-ranking check | **< 2 h** | Once for the surviving candidate pair, before trusting L2 rank |
| **L2** | 8-snippet development panel | **< 45 min** | Every candidate configuration |
| **L2L** | One full-duration narrow depth strip | **hours** | Only L2 winners — longitudinal identity only |
| **L3** | 8-snippet held-out panel + second session | **< 4 h** | Only L2 winners |
| **L4** | Full session, both probes | days | Only L3 winners; ≤ 4 per quarter |

Two rules make this work:

1. **Same endpoints at every tier.** L1 computes the *identical* score
   dictionary as L4. A change that helps at L1 is measured the same way at L4,
   so tiers are comparable and a regression is visible immediately.
2. **Stage observables are recorded on every run, free.** Each L1 run emits its
   per-stage diagnostics (detection counts, coincidence, motion estimate,
   amplitude distributions, residual). Per-stage auditing stops being a separate
   project — it is a by-product of every end-to-end iteration. This is
   [`0007`](decisions/0007-stage-local-validation.md) made cheap enough to
   actually follow.

### L2L — the longitudinal tier

**Added 2026-09-02, and it repairs a structural weakness.** The ladder as first
drafted assumed short snippets and the full session could share one endpoint.
For motion-driven identity fragmentation that assumption fails: four 30-second
identity bins inside a 120-second snippet are **not** equivalent to holding one
neuron together across a 2.9-hour recording. A candidate could win eight short
snippets and then fragment catastrophically over the full session.

L2L is **one full-duration narrow depth strip** (~96 channels), run only for L2
winners. A full-session sort at this width is already known to be surprisingly
cheap. Its job is **longitudinal identity continuity only** — label switches per
neuron over the full duration, family stitching rate, drift-tracked identity —
not general yield. It sits between L2 and L3 precisely so a longitudinal failure
is caught before anything expensive runs.

This also relieves Checkpoint B: the short tiers no longer have to carry a
phenomenon they are structurally incapable of measuring.

### Context-rank checkpoint — before a cheap winner is trusted

Cropping and duration change more than runtime: reference support, whitening,
neighbouring-spike competition, the population available for template learning,
and motion estimation all change. A candidate chosen on a narrow 120 s strip
therefore needs one bounded **ranking-preservation** check before L2 promotion:
this uses development data only and does not open the held-out panel.

1. Sort the same representative interval as the usual narrow strip, a wider
   strip with explicit margins, and full-probe context if affordable; score only
   the shared central neurons/region.
2. Sort one short interval alone and score the identical interval inside one
   longer sort.
3. Compare the rank of production 12/9, 8/8 and 9/9 on the primary score and
   consequential guardrails. Exact cluster identity is not required; a change
   in the preferred configuration is the failure condition.

If ranking changes, the cheap tier is a screening assay only. Candidate
selection moves to the smallest context that preserves the longer/wider rank;
do not average incompatible contexts into a winner.

### Artifact learning-context diagnostic

Artifact-proximity scoring tests local damage, but a transient-rich interval
may also contaminate covariance/whitening estimates or template learning and
degrade otherwise clean periods. Run one bounded paired diagnostic: score the
**same clean central interval** while the surrounding learning context either
includes versus excludes the pathological interval. Hold the scored samples,
channels and candidate settings fixed. If the clean-interval result changes,
follow once with a controlled replay using preprocessing statistics learned
from clean versus contaminated context, and locate the first divergence. This
distinguishes local voltage corruption from a recording-wide learning effect;
it is not permission for a broad artifact-preprocessing sweep.

### Caching contract

Three cache layers keyed by content hash, so an iteration only recomputes what
changed:

```
snippet   (raw int16 + probe geometry + injected truth)   ← built once, frozen
   └── preprocessed   (per preprocessing config hash)
          └── sorted  (per sorter config hash)
                 └── curated + scored
```

A sorter-parameter change reuses the preprocessing cache. A preprocessing change
reuses the snippet. Only a panel change invalidates everything.

## 4. The frozen snippet panel

The second core asset. Built once, hashed, versioned, and **split into
development and held-out halves before anyone looks at a result**.

### Axes

Each snippet is a **time window × depth strip**, not a time window alone.

| Axis | Levels | Source |
|---|---|---|
| Motion regime | quiet / rapid-estimate-change / anomalous-input / support-dropout | `testing/luke_motion_regime_windows.py` (exists) |
| SNR | high / medium / low tertile by local noise and template amplitude | **to build** |
| Depth strip | 96–128 contiguous channels; shallow / mid / deep | `luke_rigid025_depth_strip.py`, `luke_inward_crop_pair_sort.py` (exist) |
| Artifact proximity | near a >500 µV sidecar cluster vs clear | `pipeline/artifacts.py` (exists) |

### Size and budget

- **16 snippets**, 8 development + 8 held out.
- 120 s × 112 channels ≈ **0.8 GB** raw each; panel ≈ 13 GB, ~40 GB with
  preprocessing variants cached.
- Lives on local disk (`/` has 617 GB, `/media/huklab/Data` 1.1 TB).
  **Nothing on `/mnt`** — it has 556 GB left of 175 TB.

### Rules

- Panel selection uses **input-side and estimator-side signatures only** — never
  sorter labels, never challenger results. `luke_motion_regime_windows.py`
  already enforces this discipline; keep it.
- The held-out half is opened **once per promotion attempt**, at L3. If it is
  consulted during development it is burned and must be rebuilt.
- Cap L2 evaluations against the development panel and log every one. If a
  candidate is the 30th variant tried, its L2 result is a selection artifact,
  and only L3 can rescue it.

## 5. Metrics

### Primary — hybrid ground truth (the decisive layer)

Inject known spike trains with known waveforms into real Luke background, then
score the sort against truth. The injection, geometry-aware motion operator,
sorter connection, content-bound caching, and per-cluster exclusive scorer are
built. Their first end-to-end C2 runs exposed operator, donor, cache, and scorer
faults. C2 v4 is the first run to combine the corrected components and survive
those controls. Its exact 40 µm staircase establishes the fragmentation
mechanism; its fractional-offset ramps remain uninformative about Luke-scale
displacement because the donor forward model attenuates rather than translates
the compact footprints.

Per injected unit:

| Metric | Definition |
|---|---|
| **Accuracy** | `TP / (TP + FP + FN)` against the injected train (±0.5 ms) |
| **Split** | number of output units capturing > 5% of the injected train |
| **Merge** | best-matching unit also captures > 5% of a *different* injected train |
| **Identity continuity** | accuracy per 30 s bin; count of label switches across bins |

**Headline number:** *units recovered at accuracy ≥ 0.8 with no split and no
merge.* One integer, per snippet, comparable across every tier.

Injection must cover what we actually care about: rigid and non-rigid drift,
amplitude variation, overlapping spikes, both polarities (imec1 is ~59%
positive-dominant — a negative-only detector is disqualifying), and artifact
proximity.

**Two denominators, never conflated.** For a motion-mechanism contrast, require
the donor to be recovered in the relevant static arms: the question is what
motion changes in a neuron both configurations can sort while stationary. For
pipeline performance, freeze the neuronal cohort before scoring and retain
every donor in the denominator, including donors a candidate fails statically.
Candidate-specific static qualification would hide exactly the losses the
pipeline comparison is meant to detect. Reports must label every table
`mechanism-qualified` or `fixed-cohort`; neither may stand in for the other.

**Donor scope.** The 14 compact imec0 KS-derived donors remain a valuable
regression set, not a representative sample of all neurons Luke needs rescued.
Before promotion, add a small, independently reviewed extension containing
several imec1 waveform families and weaker-amplitude examples, including
waveforms Kilosort does not reliably learn where a defensible waveform source
exists. Freeze provenance, waveform review, amplitudes and inclusion before any
candidate result is inspected. This is a bounded cohort extension, not the
dense spatial donor project needed for fractional-motion mechanism work.

### Condition-dependent missingness — scientific distortion endpoint

Overall recall can conceal errors concentrated around bursts, collisions,
stimulus onset, eye movements or movement-associated artifacts. On a small
fixed injected cohort, use a prespecified rate-modulated truth train and score:

- recovered modulation magnitude and timing;
- recall stratified by firing-rate phase;
- recall by collision timing and artifact proximity.

On real data, ask whether matched pipeline disagreements concentrate around
the same experimental events. Never select a pipeline because its tuning curves
look more plausible; that would use the scientific answer as sorter ground
truth. The endpoint is whether sorting error or disagreement acquires a
systematic dependence on condition.

### Completeness — where the QC truncation estimator fits

The amplitude-truncation estimator measures something **nothing else in this
plan measures**: per-unit detection completeness — what fraction of a unit's
spikes fall below the detection floor.
[`0008`](decisions/0008-amplitude-completeness-gates-promotion.md) was right
that the frozen gate set had no completeness dimension, and that point survived
[`0009`](decisions/0009-cross-sort-comparisons-must-be-unit-matched.md)'s
retraction. 0009 retracted one *use* of the metric, not the instrument.

| Layer | Use | Required guard |
|---|---|---|
| **Primary** (injected truth) | On an injected unit **recall is the completeness ground truth** — TP/(TP+FN) says exactly what fraction of a known train was recovered. Truncation is **calibrated against it**: does the estimator track true missingness in this pipeline, on this background? | Report estimator *error*, not only the estimate. First chance to check it against truth rather than against itself. |
| **Secondary** (real data) | **Matched-unit paired** truncation difference — the same neuron, found by both pipelines, compared to itself. For real units with no truth, the only completeness measure available. | **Never a population median across sorts** (0009). Filter censored windows with `pipeline.truncation.is_saturated` before any aggregate. |
| **Mechanistic** (A2 / C2) | A fragmentation signature. *Temporal* fragmentation (A→B→C over epochs) leaves each fragment with the full amplitude range, so truncation should stay flat; *amplitude* fragmentation splits off the low tail and should elevate it. | A **prediction to test**, not an established result. Stated so it is falsifiable. |

**Estimator limits, established by the audit**
(`docs/luke_20250804_truncation_fitter_audit.md`):

- **Trustworthy where used:** unbiased to under 0.5 pp for true missing
  fractions of 0.5–40%. Every KS-good cohort measured sits at 0.6–3%.
- **Hard-censored at 50%.** The bound `x0 >= x_min` makes the statistic unable
  to exceed 50, so true 70% reports as 50.0. A window at exactly 50.0 is a
  boundary-pinned fit, not a measurement — filter it, never average it in.
- **Eligibility is rate-dependent.** A unit needs 1000 spikes in a continuous
  block, so only ~1/3 of KS-good units are ever estimated.
- **Amplitude-driven** (within-method Spearman −0.44 to −0.56). Match units or
  stratify on amplitude, or the comparison measures composition — exactly how it
  produced a false conclusion once already.

### Secondary — real-data symmetric agreement

Automated from `testing/luke_rescue_unique_units_audit.py`: gained and **lost**
good units, reported as `+N / −M`, never as a net. The implementation now
uses exclusive event identities; detection classifications additionally require
spatial agreement and excess above a time-shift null. Corrected v2 output is
complete: `+210 / -137`, with supported overlap for 208/210 and 132/137 and
seven total unresolved cases. Report “no confirmed difference,” never
equivalence; see decision 0015.

### Guardrails (prespecified consequential bounds)

- Similar good–good pairs per good unit (similarity ≥ 0.8 within 100 µm)
- Refractory violation distribution vs the matched-unit reference
- Edge-spike fraction (40 µm)
- **Matched-unit completeness** — paired truncation difference, censored windows
  filtered
- **Waveform preservation** — required from the moment a candidate deliberately
  modifies voltage:
  - waveform cosine against the known injected waveform, or the matched
    uncorrected real-unit waveform
  - peak-amplitude retention
  - spatial-footprint / depth preservation, where it can be measured reliably

  Evaluated **by donor / waveform stratum as well as in aggregate**. A candidate
  must not buy a headline identity improvement by systematically degrading the
  sharp, high-SNR stratum. Waveform-preservation acceptance bounds are frozen
  before L2/L3 promotion testing, never selected from candidate yield. Exact
  peak-amplitude equality is not the scientific objective: a modest bounded
  change may be acceptable when independently validated identity and recall
  improve and no waveform class is selectively damaged.
- **Runtime per unit data**, tracked at every tier — and measured during D2
  (estimation / interpolation / sorting / total), not deferred to Phase E

### Explicitly not endpoints

KS-good count. Total spikes. Stable-bin occupancy. Population medians of any
per-unit metric across sorts with different unit populations
([`0009`](decisions/0009-cross-sort-comparisons-must-be-unit-matched.md)).

Guardrails are not required to improve simultaneously. Before a promotion run,
classify each as **hard** (a safety/identity failure that blocks immediately) or
**bounded** (a small regression is tolerable within a frozen numerical margin).
Promotion requires a reproducible net primary benefit with no hard breach and
no bounded breach outside its margin. This prevents both destructive tradeoffs
and the impossible rule that every diagnostic must move in the favourable
direction.

## 6. Phases, checkpoints, go/no-go

### Phase A — the symmetric audit (no new sort)  ✅ v2 rerun complete 2026-09-03

Runs on existing outputs; can proceed in parallel with Phase B.

**V2 result:** [`luke_20250804_rescue_unique_units_audit.md`](luke_20250804_rescue_unique_units_audit.md)
+ [`luke_20250804_rescue_lost_units_audit.md`](luke_20250804_rescue_lost_units_audit.md).
Exclusive one-to-one matching, depth-windowed coincidence gated against
circular-shift nulls. Cohort **+210 / −137** (net +73 unchanged). The audit
supports overlap for 208/210 rescue-side units and 132/137 legacy-side units;
two and five respectively are `detection status unresolved`. No new or lost
detection is confirmed, but the unresolved cases cannot be counted as shared
and the result does not establish equivalence. Checkpoint A's substantial
confirmed-regression trigger did not fire.

**Historical v1 result — retracted by 0011:** classified 127; same qualitative
conclusion (0 lost at detection) but via a non-exclusive matcher and a
whole-probe coincidence statistic with an ≈87% chance baseline.

**Interpretation — corrected 2026-09-02.** An earlier draft of this section
concluded that "curation/clustering remain the only stages that differ." That
is a stage-*localization* claim smuggled in as a *cause* claim, and it is
exactly the error [`0007`](decisions/0007-stage-local-validation.md) exists to
prevent. The correct statement:

> For the 340 null-supported cases, the observable discrepancy occurs at
> assignment/clustering rather than confirmed gross detection. **This does not
> resolve the seven other cases or localize the cause.** Different preprocessing or
> motion representations can preserve the same event pool while changing how a
> moving neuron is partitioned into identities.

The two halves of the −137 are different phenomena and must not be pooled:

| | n (v2) | Nature |
|---|---:|---|
| legacy-good → `mua` demotions (preserved as MUA) | 23 | A **labelling-threshold** issue — the mirror of the MUA→good promotions. One moved threshold, to be decided as one. |
| **re-clustered** (dispersed + split + merged) | **109** | A **repartitioning** phenomenon. Non-rigid motion and KS4 template competition are both candidate explanations; A2 does not separate them. |

The leading mechanistic hypothesis for the re-clustered losses, and for the
corresponding dispersed/split gains among the +210:

> Legacy partially stabilized moving neurons by resampling voltage. Rescue
> preserves the voltage but leaves KS4 to represent a moving waveform footprint,
> which it can only do by splitting it across templates.

This is consistent with the earlier rescue work, which explicitly retained
motion-driven fragmentation and an unwarped alternative that links temporally
fragmented clusters or lets templates follow tissue. **Phase A2 tests it before
any candidate search begins.**

*Tension that A2 must confront — resolved 2026-09-03, see
[decision 0013](decisions/0013-luke-imec0-has-appreciable-rigid-motion.md).* An
earlier draft here cited a **1.28 µm** total-drift DREDGE sidecar for imec0 and
concluded that motion-driven re-clustering on this probe would have to be
non-rigid. That 1.28 µm value came from a rigid-only, QC-unqualified sidecar
(`weights_thresh` entirely non-finite) and is **not reproduced by any accepted
on-disk motion estimate**: `ks-motion` 6.6 µm, `dredge-motion` 15.3 µm,
`medicine` 21.9 µm, `decentralized-motion` 29.5 µm full-session rigid range, and
the motion-overlap gate found 0 / 87 imec0 120 s windows quiet enough to reach
the Yates 75th percentile
([`luke_yates_stable_window_overlap_result.md`](luke_yates_stable_window_overlap_result.md)).
**Rigid motion on imec0 is appreciable and is a live candidate mechanism for the
re-clustered losses.** At 120 s scale the dominant imec0-vs-Yates difference is
rigid translation magnitude/rate, not non-rigid deformation (Luke's
depth-normalised non-rigid gradient is at or below Yates). Estimators still
disagree ~5× on the magnitude — a quantification problem, not grounds to treat
imec0 as stationary. A2 continues to run on imec1 as well.

1. Classify all legacy-good units rescue does not reproduce: preserved as MUA,
   split, merged, dispersed, absent at detection, or rejected by curation.
2. Apply the same waveform/refractory/amplitude evidence used for the gains.

**Checkpoint A.** *Go:* a symmetric `+N / −M` table with both sides classified.
*Decision:* if a substantial share of the losses are confirmed genuine neurons
lost at detection or curation, that is a regression the yield narrative hid.
**Met in the bounded sense 2026-09-03 (v2): no substantial confirmed loss;
132/137 overlap-supported and 5 unresolved.**

### Phase A2 — is the repartitioning motion-structured? (no new sort)  ✅ v2 rerun complete 2026-09-03

Runs entirely on existing outputs. Hours, not days. **This gates Phase D's
priority order**, and could change what the whole candidate search is for.

**V2 result:** [`luke_20250804_rescue_repartition_motion_audit.md`](luke_20250804_rescue_repartition_motion_audit.md).
`testing/luke_rescue_repartition_motion_audit.py` (prespec frozen in `PRESPEC`).
96 imec0 + 46 imec1 dispersed families scored, exclusive identities, spatially
plausible fragments, refractory violations on the **fragment-cluster union**:

| | imec0 | imec1 | (v1, retracted) |
|---|---:|---:|---|
| coexisting fragments (over-splitting signature) | **0.0%** | **0.0%** | 0% / 1.6% |
| successive (one fragment at a time) | 56% | 72% | 50% / 69% |
| **fragment-union merge is refractory-clean** | **6%** | **13%** | ~92% / ~95% |
| median fragment-union RV fraction | 14% | 8% | (anchor: 0.2%) |
| depth ↔ rigid-DREDGE-motion correlation | 0.13 | 0.17 | 0.11 / 0.13 |
| ownership flips per hour | 18 | 21 | 18 / 22 |
| classed `ambiguous` | 92/96 | 41/46 | — |

**Reading (the mix, not a verdict):** the prespecified **coexisting-fragment
signature was not observed**. That does not exclude every form of
over-splitting. The fragments
are **not demonstrably one clean neuron** — unioning the fragment clusters'
trains is refractory-clean in only 6–13% of families. It is **not tracked by the
rigid motion estimate** and **not slow** — rapid template-ownership flicker
(~every 3 min) with no depth trajectory. ~90% of families are `ambiguous`:
neither classic over-splitting nor clean motion fragmentation. A2 cannot
separate *non-rigid / fast motion* from *KS4 template competition on preserved
voltage*. **The corrected C2 would separate them.**

**Prespecify before looking**, or this becomes story-fitting: freeze the sample
of legacy↔rescue families and the decision rule first, then run once.

1. Sample the strongly **dispersed** legacy↔rescue families — using the v2
   cohorts rather than the historical 100-loss/85-gain counts — stratified by depth and by
   estimated motion in the window. Run on **imec0 and imec1**.
2. For each family, ask whether the rescue fragments are **temporally
   complementary** and whether ownership switches as estimated tissue position
   changes.

**The discriminator, fixed in advance:**

| Observation | Reading |
|---|---|
| Fragments occupy **successive** epochs, follow a coherent depth/waveform trajectory, and merge **without** creating refractory violations | **Motion fragmentation** |
| Fragments **coexist** at the same times and the same motion state | Ordinary over-splitting / over-peeling |

Both patterns can be present; report the mix, not a verdict.

**Checkpoint A2.** *Go:* a classified sample with the mix quantified per probe.
*Decision:* this sets the Phase D priority order (see the decision tree there).

*Reached 2026-09-02 (v1); re-run 2026-09-03 (v2).* The v1 reading — "fragments
are one clean neuron → post-sort family stitching would recover them" — was the
load-bearing input to the first Phase D target. **V2 retracts it:** the
fragment-cluster unions are refractory-clean in only 6–13% of families.
The coexisting-fragment signature remains absent (0.0%), so that specific
ordinary-over-peeling branch is not indicated — but broader over-splitting is
not excluded, and neither is “stitch first” supported. The v2 mix is mostly
`ambiguous`.

*Historical stitching evaluation retracted 2026-09-03.* Its exact 2/127 and
net-loss accounting inherited the invalid v1 matcher and is not active evidence.
V2 A2 independently shows that naively unioning the selected fragment clusters
usually produces refractory-violating trains, so unqualified stitching is not
supported. It does not establish where the underlying problem must be fixed.

### Phase B — build the ladder

1. ~~**Snippet builder**~~ — **done 2026-09-02**, `testing/ladder_snippets.py`
   (+ `test_ladder_snippets.py`, 8 tests). `SnippetSpec` → `build_snippet` cuts
   a time × depth-strip block from the accepted recording via
   `resolve_bakeoff_window`, saves it as a sealed SI binary folder with a
   content hash, and refuses `/mnt`. `SnippetSpec` rejects any selection basis
   that names sorter labels. `Snippet.raw_domain_float32()` is the Phase C
   injection view. `verify_snippet` catches a mutated panel; `freeze_panel`
   enforces 8 + 8. Validated end-to-end against the accepted imec0 recording (a
   120 s × 112 ch `quiet`-window strip builds in ~26 s to 770 MB, hash verifies;
   validation snippets then deleted; the panel was subsequently frozen below). Snippet root
   configured in `configs/ladder.toml` (`configs/example.ladder.toml` tracked).
2. **SNR stratification** to complete the panel axes — the last missing axis
   before `freeze_panel` can seal the real 16-snippet panel. Motion regime
   comes from `luke_motion_regime_windows.py` (5 windows exist), depth strip and
   artifact proximity from the existing scripts.

   **`ladder_snr.py` done 2026-09-02** (+ `test_ladder_snr.py`, 3 tests):
   `snr_profile` derives a snippet's noise floor (per-channel MAD µV) and event
   amplitude distribution (`detect_peaks`, both polarities) from the voltage
   alone — no labels — and `stratify_by_snr` tertiles a pool. Validated on the
   quiet window's three depth strips: shallow SNR 8.1 → mid 11.4 → deep 13.8,
   tertiles low/medium/high. ~18 s/snippet (build + profile).

   **Panel design 2026-09-02** — `testing/luke_ladder_panel.py`
   (+ `test_…`, 6 tests). 16 cells on **imec0**: the 5
   `luke_motion_regime_windows` windows × 3 depth strips (112 ch: shallow ch
   8–120, mid 136–248, deep 260–372) + a second quiet window. **Split rule,
   written before any characterisation was read:** cells ordered regime-major
   (quiet, rapid_motion, sustained_noise, support_dropout, noise_plus_motion) ×
   strip-minor; odd positions → development, even → held_out. Deterministic,
   both halves span all 5 regimes and all 3 strips. SNR and artifact-proximity
   (>500 µV sidecar point density) are **measured, not chosen** — recorded in
   `axes` after the build (`SnippetSpec.digest` now hashes only the physical
   window × channel selection, so filling an emergent label does not change
   snippet identity). `--characterise` builds + profiles all 16 (no sorter);
   `--freeze` then calls `freeze_panel`.

   **Panel FROZEN 2026-09-02**, `panel_digest 07d5d808…` (approved after the
   balance table was reviewed; split rule unchanged). 16 imec0 snippets, 13 GB,
   all content-hash-verified. Both halves span all 5 regimes and all 3 strips;
   SNR tertiles 3H/3M/2L (dev) vs 2H/2M/4L (held); artifact-near 2 (dev) / 3
   (held). Table:
   `testing/outputs/luke_ladder_panel/panel_characterisation.csv`.
3. ~~**`score_sort(sorter_output, snippet) -> dict`**~~ — **done 2026-09-02**,
   `testing/ladder_score.py` (+ `test_ladder_score.py`, 8 tests). One function,
   three layers (primary hybrid-GT / secondary symmetric agreement / guardrails)
   + a provenance-only `context` block that flags the non-endpoints. Reuses the
   coincidence machinery from `luke_rescue_unique_units_audit.py` unchanged.
   Its original full-imec0 validation reproduced the now-retracted v1 Phase A
   counts (`matched 101, gained 200, lost 127`). The implementation has since
   moved to exclusive identity matching and scorer schema v3; only current v2
   or later outputs may be cited. The guardrail calculations still reproduce 27
   similar good-good pairs and a 0.10% median refractory fraction. Hybrid truth
   is now connected end to end; its first runs exposed the scorer defect fixed
   by decision 0014.
4. ~~**L1 runner**~~ — **done 2026-09-02**, `testing/ladder_l1.py`
   (+ `test_ladder_l1.py`, 6 tests). `l1_run(snippet_dir)` →
   `sort → curate → score`, one command, three-layer content cache
   (`<spec digest>/sort/` reused across `cur-<curation digest>/` leaves — a
   curation-param change reuses the sort). `build_snippet` now also writes a
   `rescue_recording_manifest.json`, so `pipeline.sorting.run_kilosort4` and
   `pipeline.downstream.run_curation_stage` run over a snippet unchanged. Every
   run records cheap per-stage observables (§3 rule 2): sort summary, amplitude
   spread, curated counts. `l1_root` in `configs/ladder.toml`, never `/mnt`.
5. ~~**Calibrate the tiers**~~ — **measured 2026-09-02** on the `quiet` mid
   strip (120 s × 112 ch): snippet build ≈ 26 s, KS4 ≈ 38 s, curation < 5 s,
   score ≈ 23 s → **L1 pipeline well under the 5-minute budget** at 120 s. The
   snippet does not need shrinking for L1. *Caveat surfaced:* on the **secondary
   KS-good metric** a 120 s snippet does not reproduce the full-session
   direction — the windowed legacy reference has 58 good units in the strip, a
   fresh snippet-scale rescue sort has 16. The historical
   `lost_absent_at_detection = 0` explanation used the invalid v1 matching path
   and is withdrawn; the discrepancy still demonstrates that short-window
   KS-good labels are not comparable to full-session labels. The
   **primary injected-truth metric is duration-robust where this is not**; the
   secondary metric on a snippet needs the *same-length legacy sort* as its
   comparator, not the full-session sort — which needs a config-parametrised
   sorter (Phase D infrastructure).

**Checkpoint B — representativeness clause revised 2026-09-02.** The original
clause required a snippet to reproduce the known full-session legacy-vs-rescue
direction. That is too strong, and B.5 demonstrated why empirically: if the
full-session difference is specifically a *longitudinal* fragmentation effect,
a short snippet may correctly show almost no difference, and failing it would
condemn a perfectly representative panel.

*Go:* L1 runs end-to-end in < 5 min and L2 in < 45 min, **and** L1/L2 reproduce
phenomena known to be **local**:

- reviewed-event recovery
- duplicate / near-coincidence burden
- motion-event behaviour within the window

*The longitudinal identity phenomenon is reproduced at **L2L**, not here.*

*No-go:* failure on the local phenomena means the panel is unrepresentative —
fix that before trusting any L1 result.

*Progress 2026-09-02:* the < 5 min L1 clause is **met** (see B.5). The
reproduction clause is **not yet testable** — it needs the legacy pipeline run
at snippet scale as the comparator (a config-parametrised sorter, Phase D),
because the full-session legacy sort is not a valid comparator for a 120 s
snippet sort on the KS-good metric. The primary injected-truth metric is the
one Checkpoint B should ultimately turn on; that is Phase C.

### Phase C — connect ground truth to a sorter

1. ~~Wire the sealed injection scaffold to the L1 runner~~ — **done 2026-09-02**,
   `testing/ladder_inject.py` (+ `test_ladder_inject.py`, 9 tests).
   `inject_trajectory` schedules per-spike `InjectionEvent`s along a known
   channel trajectory and calls the sealed `inject_float32_raw_domain`
   unchanged (float32 µV view only — never the stored int16).
   `write_injected_recording` quantises back to int16 and writes an
   accepted-recording folder + manifests, so `l1_run` sorts and
   `score_sort(truth=…)` scores it. `drift_penalty(static, moving)` is the
   decisive Δ.
2. **Historical static sanity result; rerun required** — the rescue pipeline
   was reported to recover the
   high-SNR (SNR 11) donor template T01, injected static into a quiet imec1
   strip, at **accuracy 0.94 ≥ 0.9**. This old result does not validate the
   moving-arm benchmark and will be regenerated with the corrected cache and
   scorer.
3. Establish the legacy baseline score on the development panel — pending the
   config-parametrised sorter and the frozen panel.

### C2 — the paired drift challenge (primary mechanistic experiment)

**Promoted 2026-09-02** from "one thing injection should cover" to the
experiment that decides Phase D. The sealed scaffold already anticipated a
gated known-drift second phase; this is it.

A **within-subject** contrast — the same waveform and the same spike train,
injected twice:

| Arm | Injection |
|---|---|
| **Static** | Neuron held at a fixed depth |
| **Moving** | The identical neuron translated along a known Luke-like trajectory |

The decisive quantity is the **drift penalty** — the change caused *solely* by
motion:

- Δ accuracy
- Δ number of output identities claiming the train
- Δ label switches

> If static KS4 recovers the stationary neuron cleanly but breaks the moving
> version into A→B→C, we have **directly demonstrated the cost of the no-motion
> strategy** — the mechanism Phase A2 can only infer from observational data.

Run the trajectories at several amplitudes, including rigid and non-rigid, and
at both polarities.

**C2 v4 result — mechanism established at 40 µm; Luke-scale arms
uninformative (2026-09-04; [full result](luke_20250804_c2_v4_result.md)).**
The motion-overlap analysis measured Luke imec0's actual regime: per 120 s
window, rigid excursion runs ~4–23 µm (median ~11 µm under MEDiCINe, ~4 µm under
`ks-motion`) with rigid speed ~0.2–0.8 µm/s, while the depth-normalised non-rigid
gradient is at or below Yates. C2 v4 therefore tested 5/11/22 µm rigid ramps
plus an exact lattice-commensurate 40 µm staircase. The staircase is the first
C2 result to survive every control: uncorrected rescue fragments 13/14 donors,
and rigid correction largely restores them. But the 5/11/22 µm recorded-template
operator attenuates compact footprints in place instead of faithfully
translating them, so those arms cannot answer whether Luke-scale motion is
harmful or whether correction helps there. The original Luke-scale question
remains open in both directions.

Because the injected train is known, the staircase also **calibrates the
truncation estimator against truth** for the first time. The completed
diagnostic finds the predicted temporal-fragmentation signature: phase-local
truncation stays low while the best full-duration identity loses about half the
train. This is mechanistic evidence at 40 µm, not a Luke-scale result.

*Remaining simulation work:* a dense spatial donor model could answer the
Luke-scale causal question by translating a compact waveform without the
attenuation/translation degeneracy. It is scientifically useful but is now a
background task, not a gate on threshold, staircase, or real-data pipeline
testing.

**Historical C2 runs — retracted/retired; retained for audit only.** The run
below did not apply mutually inverse forward and correction operators on the
real probe geometry. Geometry-correct v2 was not run because it retained the
discredited plateau donors. V3 used the compact donors but failed as a scorer
validation run. C2 v4 supersedes them only within the limits above.

**Historical first run 2026-09-02** — [`luke_20250804_c2_drift_challenge.md`](luke_20250804_c2_drift_challenge.md).
`testing/luke_rescue_c2_drift_challenge.py` (historical prespec; reused the
now-discredited pilot donor templates; static arm from the quiet imec1 window).
It reported static T01 (SNR 11) / T04 (SNR 6) recovered at
0.94–0.98 under both sorter configs.

**Retracted historical drift penalty (Δ accuracy = moving − static), both arms:**

The following table is retained for audit history only. Its motion/scoring
implementation was invalid for inference and the values must not guide Phase D
until reproduced by the corrected experiment.

| donor | trajectory | rescue | legacy_style (`nblocks=1`) |
|---|---|---:|---:|
| T01 | rigid 15 µm | −0.35 | −0.32 |
| T01 | rigid 40 µm | −0.54 | **−0.81** |
| T01 | osc 20 µm/40 s | −0.70 | −0.68 |
| T04 | rigid 15 µm | −0.30 | **−0.58** |
| T04 | rigid 40 µm | −0.53 | −0.31 |
| T04 | osc 20 µm/40 s | −0.38 | −0.49 |

The original run claimed the following two findings. **Both are withdrawn
pending rerun:**

1. **Motion alone costs 30–80 accuracy points, under both configs** — mostly
   missed spikes (T01 rigid-40: FN 24→314 rescue, 11→503 legacy) and identity
   proliferation (T04 rigid-40: +15/+12 output units on one train). This
   reproduces the A2 fragmentation signature **causally**: motion is a
   sufficient cause.
2. **KS4 rigid internal drift correction does not recover it** — `legacy_style`
   is worse on 3 of 6 clean conditions, comparable on 2, better on 1, and piles
   up false positives on the oscillation (FP 874 vs 473). Turning `nblocks` back
   on is **not the fix.**

**Current consequence — not a conclusion from the historical table:** the
completed static sweep confirms that detection/template-learning thresholds
are a real lever and that production 12/9 lies beside a poor region of the
response surface. Its provisional single-train ranking did not transfer to the
exact-staircase train. The subsequent sentinel localized that failure to train
composition and phase, and the completed 588-cell Stage 2 comparison did not
separate 8/8 or 9/9 from 12/9 at the frozen donor-level confidence threshold
([result](luke_c2_train_stability_stage2_result.md)). The threshold branch is
closed with no change. Unqualified family stitching remains unsupported by
A2's real-data refractory results.

V4 supplies both polarities and the full compact real-donor amplitude range.
The staircase truncation-vs-truth diagnostic is now complete
([report](luke_20250804_c2_v4_truncation_diagnostic.md)): uncorrected rescue's
phase-local estimate stays low while its best identity loses about half the
whole train, supporting temporal/positional fragmentation rather than uniform
low-amplitude truncation. It is a calibrated 250-spike post-hoc diagnostic, not
production QC. Genuinely non-rigid trajectories, a second window, and the dense
donor model are optional background science rather than gates on the active
threshold-to-real-data path. Static qualification is donor-wise and
prespecified; there is no
special T06 exception because all pilot T donors are forbidden.

**Checkpoint C status.** C2 v4 supplies valid known-truth scoring and an exact
40 µm drift/control result for rescue, `rescue_rigid`, and `legacy_style`. It
does not supply the intended Luke-scale causal calibration because the
fractional-offset donor perturbation is not a faithful translation. That open
scientific question no longer blocks cheap pipeline candidates: threshold
settings can be tested statically, then on the exact staircase, before matched
real-data confirmation.

*Historical checkpoint status, withdrawn 2026-09-03.* The v1 drift-penalty half
was initially marked done, but its forward injection and inverse correction did
not implement the same physical motion and its scoring was non-exclusive.

**Retracted pending rerun 2026-09-03.** The historical result below used a
content-unbound cache and non-exclusive scorer; Checkpoint C is not reached.

**Historical run recorded 2026-09-03** — `testing/luke_ladder_checkpoint_c.py`,
[`luke_20250804_checkpoint_c_panel_baseline.md`](luke_20250804_checkpoint_c_panel_baseline.md).
3 compact D2b-2 donors (73 / 149 / 274 µV, static) injected into all 8 dev
snippets, scored under `RESCUE` and `LEGACY_STYLE`. Sanity condition met (D02
recovered static at ≥ 0.94 everywhere). **Result: no clean winner — rescue wins
where there is motion (noise+motion: +0.6 to +0.8 accuracy; `legacy_style`'s
rigid correction *collapses* there), legacy wins on pure sustained noise
(−0.2 to −0.6), quiet is tied.** The 73 µV donor is below *both* pipelines'
floor (rescue median 0.02, legacy 0.26) — the low-SNR detection limit is where
panel yield is actually lost, and it is not a motion problem.

*(Caveat, now resolved: this historical run's `score_sort` used the v2 matcher,
which both over-counted splits (non-exclusive within a cluster) and — the
failure the C2 v3 run exposed — under-counted recovery (global exclusivity let
background clusters steal events). Both are fixed in `SCORE_SCHEMA` v3,
[decision 0014](decisions/0014-injected-truth-scoring-is-per-cluster.md): per
candidate cluster, exclusive 1:1 within the cluster, best cluster by accuracy.
Any Checkpoint C rerun uses the v3 scorer.)*

### Phase D — candidate search

Only now does pipeline variation begin. Each candidate: L1 on one snippet, then
L2 on the panel, then stop. Log every candidate and its score.

The search has **two required development options**, evaluated against one
shared no-correction control:

1. **External voltage registration (Option A).** A full-duration,
   independently qualified motion field is applied on the full supported probe
   geometry with a validated interpolation/boundary policy before any spatial
   crop. KS4 internal motion correction is disabled so the intervention is
   identifiable.
2. **Unwarped motion-aware identity (Option B).** Original accepted voltages
   are never spatially resampled. Motion enters only through coordinates,
   template/identity tracking or evidence-gated longitudinal family links.

Both options may be implemented and smoke-tested now. Neither may be tuned on
L2, held-out data or full-session yield before its rule is frozen.
The uncorrected rescue pipeline remains the comparator in every experiment; it
does not consume a development-option slot.

**Immediate cheap branch — threshold branch closed with no change
(2026-09-05).** The static 14-donor grid held correction off and varied only
`Th_universal` and `Th_learned`. D10's 12/9 failure reproduced exactly and is
removed by lower `Th_universal`, proving that this specific failure is caused
by thresholds rather than motion correction. The grid also exposed genuine,
deterministic cliffs rather than a smooth monotone optimum: D14 fails at 9/8
between much better neighbouring cells. Three repeats each of D10 12/9, D10
9/9, D14 9/8, D14 12/7 and D14 10/8 produced bit-identical spike times,
clusters, amplitudes and scores.

That single-train grid provisionally nominated two integer candidates:

- **8/8 — robustness candidate:** 12/14 recovered, median accuracy 0.987,
  12 total FP, and no donor below 0.976.
- **9/9 — recovery candidate:** 13/14 recovered, median accuracy 0.994,
  97 total FP, and one donor below 0.9 (worst 0.891).

Production 12/9 recovered 11/14, with median accuracy 0.992, 289 total FP and a
0.475 worst donor. This establishes that 12/9 is brittle on the development
cohort, but did **not** license changing production: D10 motivated the
experiment and the same 14 donors selected the candidates. On the
exact-staircase train the candidate FP totals rose sharply, and the 105-cell
sentinel showed that threshold collapses depend on train composition and timing
phase rather than event count. Stage 2 therefore compared distributions over
14 frozen realisations. Its point estimates favoured both candidates, but both
97.5% donor-bootstrap intervals crossed zero (8/8: −0.112 [−0.311, +0.026];
9/9: −0.102 [−0.306, +0.036]). The frozen result is **no threshold change**.
Do not open the fractional cells, L1C, held-out static or matched-real-data
gates for these candidates.

**D10/D14 stage trace — complete diagnostic.**
The threshold cliffs are more informative than a parameter ranking alone. For
the same hashed injected recordings, compare each reproducible success/failure
pair at four boundaries:

1. universal-template detections and their waveform/amplitude composition;
2. learned templates and when each first appears;
3. learned-template assignments, residuals and event ownership;
4. final clustering and merging.

The trace found D10's first divergence at detection: 12/9 missed 99 of 708
events before clustering, while 9/9 found all 708. D14 was different: injected
events survived every recorded stage in both the 9/8-versus-10/8 and
12/7-versus-10/8 contrasts; the failed cells were contaminated at the winning
cluster (0.247 and 0.363 versus 0.0), not missing donor spikes. This separates a
systematic detection loss from the contamination lottery, but Stage 2 found no
globally supported threshold remedy.

**The priority order is a decision tree, not a fixed list** (revised
2026-09-02). The earlier fixed ordering — curation first, motion last — rested
on reading [`0010`](decisions/0010-rescue-yield-is-relabelling-not-detection.md)
as localizing the *cause* to clustering. It does not. A2 and C2 decide the
order:

| A2 finding | C2 drift penalty | → First target |
|---|---|---|
| Fragments are temporally complementary and track estimated motion | Moving injections fragment where static ones do not | **Unwarped motion-aware identity handling** — post-sort family stitching, or a sorter whose templates track tissue. Curation tuning comes after. |
| Fragments coexist at the same time and motion state | Moving injections stay one identity | **Clustering and curation.** The repartitioning is ordinary over-splitting. |
| Mixed | Mixed | Split the effort by the measured proportions, and say what the split was. |

**Correction 2026-09-04:** C2 v4 survived the operator, donor, cache, and scorer
controls. Its 40 µm staircase follows the first row of the tree — motion creates
positional identity fragments and correction restores continuity — but that
displacement is about twice Luke's largest and discontinuous. The 5/11/22 µm
ramps cannot resolve the Luke-scale branch because their recorded-template
forward model dims instead of translating the donor. A2 therefore remains
observational evidence at Luke scale. A dense spatial donor model is the next
causal test of that scientific question, but it does not block the threshold,
staircase, or matched-real-data pipeline path.

**Historical reading (2026-09-02).** A2: fragments are temporally
complementary, refractory-clean, ~0 % coexisting — **not** over-splitting. C2:
moving injections fragment where static ones do not (−0.3 to −0.8 accuracy), and
**KS4 rigid drift correction does not recover it**. So the first target is the
top row, narrowed:

> **Post-sort family stitching of temporally-complementary, refractory-clean
> fragments** is the first Phase D candidate — it is indicated by both audits
> and repairs slow drift and fast flicker alike. A *non-rigid* motion
> representation is the second, tested only against injected-truth drift penalty
> and L2L identity continuity, never unit counts. `nblocks=1` rigid correction
> is **not** a candidate — C2 showed it comparable-or-worse. Curation-threshold
> tuning is deprioritised — C2 showed the lower-threshold config fragments more.

**Historical priority after Candidate 2 (2026-09-02; withdrawn 2026-09-03).** Candidate 1 was rejected
and Candidate 2's oracle arm turned voltage interpolation from a closed question
into an open, optimizable one. The order is now:

1. **Optimize conventional voltage-based motion correction with KS4** — the
   full-session field and a bounded stabilization-versus-interpolation tradeoff
   study (**D2** below).
2. **Validate the winner at L2 and especially L2L.** Longitudinal identity is
   the endpoint a short snippet cannot measure.
3. **Only if a significant identity deficit remains**, compare alternative
   motion-aware architectures — behind the architecture gate.
4. **Curation / MUA-threshold work stays secondary and orthogonal.**

**Candidates 1 and 2 — historical results retracted.** The Candidate 1
full-session accounting inherited the invalid v1 cross-sort matcher; its exact
2/127 recovered, 4 destroyed, and 34 absorbed values are not evidence. The
Candidate 2 injection/oracle evaluation used non-inverse motion operators,
discredited plateau donors, non-exclusive scoring, and content-unbound caches.
Its claimed oracle recovery and SNR/interpolation tradeoff are likewise not
evidence. The implementations remain available as scaffolds, and the historical
reports retain the audit trail:
[`family stitch`](luke_20250804_family_stitch_candidate.md) and
[`nonrigid motion`](luke_20250804_nonrigid_motion_candidate.md).

The current stitching evidence is Phase A2 v2: naive unions of the selected
fragment clusters are usually refractory-violating. That argues against applying
an unqualified stitcher now, but it neither proves that all principled family
tracking will fail nor identifies an upstream cause. The only current
interpolation evidence also includes C2 v4's exact staircase: rigid correction
recovers a true 40 µm positional split, but `nblocks=1` breaks two stationary
donors and the fractional ramp operator cannot adjudicate Luke-scale benefit.
Whether a known accurate field helps compact neurons at Luke-calibrated motion
therefore requires a dense spatial donor model, not another reading of those
ramps.

**Stopping rule.** After C2 v4, any interpolation-policy search must be small,
prespecified, scored on known truth and waveform preservation, and stopped if it
does not beat the no-correction baseline without guardrail regressions.

To stop this becoming indefinite, **preregister the candidate budget**: at most
**6 field/application configurations** across D2a–D2c, logged, before the
branch is either adopted or closed.

### D2 / Option A — conventional external voltage-registration optimization

**A required primary development branch**, tested fairly rather than inferred
from the rejected historical warp. It is developed alongside the lightweight
Option B implementation. Expensive alternative-sorter work remains behind the
architecture gate.

The question is no longer *"does voltage interpolation work?"* It is:

> **Is the motion field we can estimate accurate enough, and can it be applied in
> a regime where stabilization benefit exceeds interpolation damage across the
> neuronal population we care about?**

That is the conventional motion-correction question, and it should be answered
before changing architectures.

#### Motion-correction research sequence — no longer the pipeline-critical path

C2 v4 is complete, but it exposed a narrower simulation requirement: a compact
donor must be translated at fractional offsets without being dimmed into an
in-place spatial low-pass image. Numeric Luke-scale motion-correction claims
wait for that dense-field donor control. This sequence continues in the
background; it does **not** block the threshold sweep, exact-staircase candidate
check, or matched real-data evaluation.

| # | Step | Status |
|---|---|---|
| 0 | **C2 v4** — corrected-scorer static qualification, then paired static-vs-moving compact-donor drift penalty | ✅ complete; 40 µm mechanism established, Luke-scale ramps uninformative ([result](luke_20250804_c2_v4_result.md)) |
| 1 | **D2b-2 donor cohort** — 14 compact imec0 donors, both polarities, 73–295 µV, hash-frozen | ✅ complete ([`d2b2_donor_cohort`](luke_20250804_d2b2_donor_cohort.md)) |
| 2 | **Oracle positive control** — attainable stabilization benefit and interpolation cost, on the compact cohort | ✅ exact 40 µm staircase positive control; fractional-offset dense-donor control is background work |
| 3 | **D2b-1 field-error tolerance** — how accurate an estimated field must be | background/paused — needs the dense spatial donor model exposed by C2 v4 |
| 4 | **D2b-3 per-stratum interpolation tradeoff** — waveform/SNR-class sensitivity | background/paused — rerun after the dense donor forward model is qualified |
| 5 | **D2a full-session external estimation** — best-supported real field | paused — infra built (`ladder_motion_estimate.py`), not run |
| 6 | **Pre-sort field qualification** — independent support, reproducibility, error evidence | paused — fails closed; numeric envelope to be set by the reruns |
| 7 | **D2c bounded policy comparison** — none / best full / one simple selective policy | not started |
| 8 | **L2 → L2L → L3** — injected truth, longitudinal identity, waveform preservation, guardrails | not started |
| 9 | **Alternative-sorter diagnostic / heavyweight architecture gate** — one frozen diagnostic may run early; pipeline development opens only if the two-option bakeoff retains a meaningful deficit | diagnostic optional; development gate closed |

Steps 2–4 come **before** the expensive full-session estimation in step 5. That
ordering is the point: step 6 can reject a field on cheap evidence before it
ever reaches a sorter.

#### D2a — Best-supported motion field

If D2a is reopened after the corrected reruns, estimate motion from the
**full-duration recording**, not from isolated 120 s snippets, so the estimator
has the temporal and population support it will have in the real sorting
problem. The old C2 comparison cannot currently be used to quantify the cost of
snippet-based estimation.

Prefer the best-supported external field available from the existing DREDGE work
initially. Preserve, and record in the manifest:

- exact time reference / clock handling
- probe geometry
- support/confidence diagnostics
- real voltage support at spatial boundaries
- float interpolation prior to final quantization
- explicit provenance and caching

**Do not assume the previously successful 0.25 gain is a physical calibration.**
The direct scale audit did not support a simple fourfold error. Treat it as one
point in the D2b sweep, not as a correction constant.

#### D2b — Field-error tolerance and the interpolation tradeoff

The old oracle arm purported to bound stabilization benefit and interpolation
damage. That bound is **withdrawn** until the geometry-correct rerun.

> **The practical unknown is how accurate an estimated displacement field must be
> for stabilization benefit to exceed both residual-motion error and
> interpolation damage.**

After rerun, a valid oracle may serve as a positive-control ceiling. No
field-error tolerance is currently established.

##### D2b-1 — Oracle degradation (run before any expensive full-session candidate)

Uses the **cached C2 injected recordings**, so it is cheap. Start from the exact
injected trajectory and introduce controlled field errors, measuring where
correction stops outperforming no correction.

| Perturbation | What it emulates |
|---|---|
| **Gain error** | under- and over-estimation of displacement amplitude |
| **Temporal smoothing / bandwidth loss** | progressive removal of fast components |
| **Temporal lag / offset** | clock or alignment error — if computationally cheap |
| **Spatial mismatch** (non-rigid) | wrong depth dependence, or spatial over-smoothing |
| **Constant displacement bias** | residual offset after centering, if relevant |

**Coarse levels only.** This is not a factorial sweep; it needs just enough
resolution to locate a crossover region.

Report per perturbation: injected-train **accuracy and recall**, number of
**claiming output identities / split burden**, **waveform cosine**,
**peak-amplitude retention**, **localization/depth error**.

The deliverable is a **field-error tolerance envelope**:

> How much error in displacement amplitude, timing, and spatial structure can be
> tolerated before corrected voltage performs no better than the uncorrected
> rescue baseline?

**This becomes the standard against which any estimated DREDGE or other external
field is judged. A candidate field demonstrably outside the envelope does not
get an expensive full-session sorting run.** That is a pre-sort gate (step 5 of
the sequence above), and it is the main defence against another days-long
failure.

**Historical run 2026-09-02; retracted pending rerun** — `testing/luke_rescue_d2b1_field_tolerance.py`,
[`luke_20250804_d2b1_field_tolerance.md`](luke_20250804_d2b1_field_tolerance.md).
Perturbed the exact field (gain ×0.5–1.5, temporal lag 2–6 s, smoothing σ 3–10 s,
bias 8–20 µm, spurious depth gradient 0.3–0.7) on T04/T06/T01 rigid-40 µm and T04
oscillation. Findings:

- **Severe rigid drift is forgiving.** Every perturbation still beat
  no-correction. ±25 % amplitude error keeps ≳ 50 % of the benefit;
  over-estimation (≥ ×1.25) starts generating false positives. Timing errors are
  nearly free on a slow ramp.
- **Oscillation is fragile** — and it is the A2-relevant regime. Over-smoothing
  to σ = ¼ period drops recovery to **0** (correction = no correction); a 20 µm
  constant bias goes **negative** (worse than nothing). A fast-motion field must
  keep temporal bandwidth ≳ 3× the motion frequency and near-zero bias.
- **Exact ≠ optimal for oscillation:** a *deliberately over-scaled* field
  (gain ×1.25–1.5) sorted the T04 oscillation better than the true trajectory
  (0.99 vs 0.78) by suppressing residual FPs. Chase in D2b-3.
- **The pilot donor templates are not spatially compact** — T04/T06 are flat
  ±160 µm plateaus, T01 is at noise level. The waveform guardrail (criterion 4)
  cannot be frozen on them and the penalty magnitudes may not transfer to
  compact real neurons. **This makes D2b-2 a prerequisite.**

**Withdrawn gate:** the ~30 % amplitude and ~⅓-bandwidth thresholds must not
be applied. The implementation now fails closed unless independent error,
support, and split-half evidence are supplied; corrected reruns must define any
numeric envelope.

##### D2b-2 — Rebuild the donor cohort (prerequisite; frozen before any gate is fitted)

**D2b-1 promoted this from a follow-up to a blocker.** The T01–T10 pilot donors
are **not spatially compact** — inspected raw, T04/T06 are ~flat high-amplitude
plateaus across ±160 µm and T01 sits at noise level. They are common-mode-
contaminated or were never spatially localised. Consequences: the waveform
guardrail (criterion 4) cannot be frozen on them, and the D2b-1 penalty
magnitudes may not transfer to compact real neurons (which could be more
sensitive — sharper spatial gradient, more interpolation blur — or less — higher
local SNR).

T01 is the warning: exact-trajectory interpolation made the **lowest-amplitude**
donor (−68 µV, near noise) *worse* (0.40 → 0.29). Treat it as evidence that
low-SNR / poorly-localised units occupy a different tradeoff regime — not that
"sharp high-SNR" units do (T01 is neither).

Rebuild the cohort from **verified spatially-compact reviewed waveforms** with a
real amplitude-decay footprint (< 10 % of peak within ~100 µm). It must span:

- waveform sharpness / spatial compactness
- SNR (measured on the compact footprint, not the plateau)
- both polarities where feasible

**At least 8 independent donor waveforms before any correction gate is
defined**, preferably ~12 given how cheap L1 has proven. Span the observed
sharpness and SNR range. **Freeze the cohort before fitting any
selective/partial policy.**

> Do not infer a selective-correction crossover from T01 and T04 alone —
> especially not from the flat pilot donors.

**Real half done 2026-09-03** — `testing/ladder_donors.py`,
[`luke_20250804_d2b2_donor_cohort.md`](luke_20250804_d2b2_donor_cohort.md).
14 spatially-compact imec0 donors: **de-whitened Kilosort template shape**
(the STA KS computed after its internal high-pass + CAR + whitening, mapped back
to sensor space with `whitening_mat_inv`) **scaled to µV by the unit's bandpass
spike-triggered-average peak**. Median energy-within-±3-channels 0.73 vs the
pilot's 0.22; half-energy width 1 channel vs 33. Both polarities (7 neg / 7 pos),
73–295 µV (median 135). No manual review, no /mnt writes.

**Scaling bug caught by the sanity check:** `cluster_Amplitude` is *not* µV — it
runs ~4–7× small, so the first cohort was injected at 22–70 µV and 5 of 6 donors
failed to sort. Fixed to the bandpass-STA anchor. The C2 "−270 µV" donors were
plateau artefacts — their nominal amplitude was noise energy, not a real
footprint.

**The compact clean good units span 73–295 µV.** A genuinely sharp,
very-high-SNR stratum is best rounded out with **synthetic** parametric
templates on top of the 14 real anchors — D2b-3's first task. Until the
synthetic extension exists and D2b-1 + the C2 drift penalty are re-run per
stratum on the full cohort, criterion 4's waveform guardrail stays unfrozen.

##### D2b-3 — Tradeoff axes

| Axis | Levels | D2b-3 first-pass result (2026-09-03) |
|---|---|---|
| Displacement magnitude | sub-channel to ≥ 40 µm | tested at 40 µm rigid; residual large there |
| Estimator support / confidence | high vs low | deferred to D2a |
| Rigid vs depth-varying displacement | where justified | deferred to D2a |
| Interpolation spatial scale / kernel | kriging σ 20/40, IDW | **retracted; unresolved pending corrected rerun** |
| Correction strength | full vs reduced/partial | not yet tested; a *selective* (per-unit-SNR) policy is what D2c should carry |
| **Waveform / SNR class** | sharpness, compactness, SNR, polarity | **amplitude/SNR dominates; the synthetic sharpness knob did not predict cost.** Only D02 (274 µV) fully recovers; D08 (73 µV) is harmed. The 120–255 µV middle is not cleanly ordered |

The key comparison is always:

> **uncorrected motion error** versus **interpolation-induced waveform error**

**Do not optimize any of these against KS-good count or total unit yield.** The
purpose is to locate a **Pareto region**, not to reopen an unconstrained
preprocessing sweep. Stay inside the preregistered budget.

#### D2c — Test correction policies

At minimum, three arms:

| # | Policy |
|---|---|
| 1 | **No voltage correction** — the present rescue baseline |
| 2 | **Best full correction** — best-supported field and interpolation configuration from D2a/D2b |
| 3 | **Best selective/partial correction** — only if D2b shows a clear crossover where small corrections cost more than they help |

##### Discovery and validation must be separated

D2b is a **development experiment**. It *estimates* the correction/interpolation
crossover on the frozen donor cohort. It does not validate a policy.

If D2b motivates a displacement / support / waveform-dependent correction rule:

1. **Freeze the rule** — before it touches the L2 development panel, and long
   before held-out evaluation.
2. It must be **simple enough to state prospectively**, e.g.:

   > Correct only when estimated displacement and field confidence place the
   > epoch inside the region where D2b showed stabilization benefit reliably
   > exceeds interpolation cost.

3. **No unit-specific exceptions constructed after inspecting sorter results.**
   That is fitting to the answer.

Selective correction is gated on **prespecified physical/estimator criteria** —
displacement magnitude, support/confidence, waveform class — **never on
downstream sorter yield**.

The physical tradeoff motivates this directly, but its magnitude on Luke is
unresolved. C2 v4 measured the large-displacement mechanism, but its compact
recorded-template operator could not preserve Luke-scale translation. The next
calibration must generate displaced donors from a dense spatial model before
asking whether correction benefit exceeds interpolation cost at Luke scale.

#### Runtime accounting — measured during D2, not at Phase E

The promotion ceiling is **≤ 1.25× legacy per unit data**. A correction
architecture already outside that envelope should be identified *before* it
consumes L2L, held-out, or replication tiers.

Measure and report separately at D2a–D2c:

| # | Stage |
|---|---|
| 1 | Full-session motion-field estimation |
| 2 | Voltage interpolation / materialization |
| 3 | KS4 sorting |
| 4 | Total end-to-end |

Report **absolute runtime and the multiplier relative to the legacy/current
reference per unit data**.

Runtime does not determine scientific correctness — but an obviously
non-promotable implementation should not consume later ladder tiers.

#### The role of the 137 legacy-side unmatched units

Resolving an inconsistency between Candidate 2's evaluation and Phase E
criterion 3, which had been left standing side by side.

The corrected cohort contains **137** legacy-good units without an exclusive
good-good match. Of these, 132 have null-controlled evidence of overlap with
the complete rescue sort and five remain unresolved. It is a useful real-data
diagnostic and should continue to be reported for every candidate, but the
historical stitching accounting on the old 127-unit cohort is retracted.

**But it is not a promotion endpoint and it is not ground truth.**

| Use | Status |
|---|---|
| Reconstitution / accounting of the 137, preserving the five unresolved labels | ✅ keep, report always |
| *Maximizing* recovery of legacy cluster identities | ❌ not an objective |
| Injected ground truth, matched high-confidence family safeguards, L2L longitudinal identity | ✅ the actual promotion endpoints |

> A candidate may fail to recreate a particular legacy cluster and still be
> better, if injected truth and validated family evidence show the legacy
> partition was itself wrong.

Phase A v2 found no **confirmed** detection loss in this cohort, but five cases
remain unresolved. A legacy-good label is not ground truth, and neither is an
unresolved result evidence that the unit was preserved.

### Option B — unwarped motion-aware identity, and the architecture gate

**Executed 2026-09-06 as the first candidate; verdict FAIL.** See
[first pipeline candidate v1 result](luke_first_pipeline_candidate_v1_result.md).
The bounded replay ran on the nominated case (`rescue…c37__failure1`) and on
healthy interval H1 under the frozen contract. It made no cross-cluster identity
links at all and fragmented 659 clusters into 2,038 units, because the linker's
refractory gate was specified as an absolute 0.01 against units whose own
baseline is ~0.40. Identity and contamination both failed; completeness was
`unevaluable` — and would otherwise have read as a 20.8 pp "improvement" produced
by discarding 82% of the interval. The gate is not to be retuned and this case
re-run; the next action is one prespec change for a v2 contract (result §8).

When B is the selected first candidate, build the minimum viable unwarped option:
retain the accepted
voltage exactly, transform only spike/template coordinates into a tissue frame,
and link identities through time only when waveform, trajectory, temporal
complementarity and union-refractory evidence all pass frozen gates. This
lightweight candidate is one of the two development options. The other option
may remain deferred under the first-version delivery sequence.

The **heavyweight architecture gate** still controls developing DARTsort,
KIASORT, TDC motion-aware template matching or another sorter as a new pipeline.
One bounded comparison with one established alternative sorter is permitted
earlier as a **diagnostic** on frozen D10/D14 and one representative context:
does the same deficit survive a materially different implementation? Freeze
the sorter, defaults, cells and endpoints in advance; do not tune it or turn
the result into an open bakeoff. Full alternative-sorter development remains
behind the gate and opens only if the lightweight unwarped option or the best
conventional motion-corrected KS4 candidate retains a meaningful, reproducible
identity deficit on injected truth and L2L longitudinal continuity.

For a broad claim of architecture superiority, the unwarped option and any later
architecture must be compared against a credible conventional motion-corrected
KS4 pipeline and the shared uncorrected control. First-version adoption instead
uses the specified legacy comparator, rescue control and frozen promotion gates;
it need not wait for an optimized conventional challenger. An architecture-wide
claim must not be judged solely against:

- the pathological historical DREDGE warp;
- the intentionally uncorrected rescue baseline; or
- an obviously under-supported short-window `nblocks` estimate.

Anything else is an unfair architecture comparison, and would let a challenger
win against a strawman.

### Why the field's conventional approach is still worth testing

Voltage interpolation is widely used because in ordinary regimes the benefit of
stabilizing spike waveforms can exceed the interpolation error. **That does not
establish that it is optimal** — adoption is not evidence of correctness. But
the favorable regime has **not yet been demonstrated at Luke's measured motion
scale**. C2 v4 demonstrates successful correction only at the exact 40 µm
staircase; its 5/11/22 µm donor arms are uninformative. The working hypothesis
is therefore not that voltage interpolation is fundamentally unsuitable, but
that benefit depends on the motion field, spatial support and
correction/interpolation operating point. That remains worth testing, without
blocking the cheaper threshold and real-data comparisons.

**Orthogonal, and deliberately later:** the post-sort MUA-labelling threshold
question (not the active detection-threshold sweep) — the v2
cohorts contain 91 MUA-to-good promotions and 23 good-to-MUA demotions among
the overlap-supported cases. This is a bidirectional threshold issue. Changing
`good` versus `mua` labels **will not fix identity fragmentation**, so it should
not compete for priority with the repartitioning question. It still needs
[`0006`](decisions/0006-recovery-axis-is-post-sort-mua-reconciliation.md)'s
reversible family-link evidence, scored on a blinded stratified sample rather
than by inspecting only one direction.

Any motion handling is judged on **identity continuity at L2L**, never on unit
counts, and requires a faithful deterministic static arm first.

**Checkpoint D.** *Go:* at least one candidate beats legacy on the fixed-cohort
primary metric on the development panel, preserves its rank in the bounded
context check, has no hard guardrail breach or bounded breach outside its
frozen margin, and remains within runtime budget. Mechanism-qualified analyses
may explain the result but cannot substitute for the fixed-cohort score.

### Phase E — held-out, replication, promotion

Before Phase E, freeze the fixed performance cohort and the separate
mechanism-qualified cohort. The performance denominator includes every donor
regardless of which candidate recovers it statically, including the bounded
imec1/weak-waveform extension.

1. Run the L2 winner through **L2L** — one full-duration narrow depth strip, for
   longitudinal identity continuity. A candidate that fragments over the full
   duration stops here, whatever its snippet scores.
2. Run on the **held-out panel** (opened once).
3. Run on a **second session** — the panel construction must be re-run there
   from scratch.
4. Check condition-dependent missingness on the prespecified modulated trains
   and matched real-data event strata.
5. Only then L4 full session.

**On criterion 3 — revised 2026-09-02.** The original wording, "preserves ≥ 95%
of legacy good units", quietly made *reproduce the old clustering* an objective.
That is wrong here. If legacy motion correction sometimes held one neuron
together by damaging the voltage, while rescue preserves the voltage but
fragments it, **neither set of cluster IDs deserves privileged status**. Phase
A v2 found no confirmed detection loss among 137 legacy-side unmatched units,
but five remain unresolved. Legacy cluster identity is a safeguard, not ground
truth; unresolved cases cannot be dismissed as partitioning artifacts.

A **high-confidence legacy-supported neuron family** must be defined
operationally and **frozen before any candidate is scored against it**, or the
criterion is gameable. The proposed definition, to be fixed at Checkpoint C:
convincing waveform evidence (stable template, adequate SNR), refractory
cleanliness at or better than the matched-unit reference, and temporal support
across a stated fraction of the window. A family that fails those is not a
regression when a candidate does not reproduce it.

The escape clause matters as much as the threshold: a "loss" that is actually a
validated merge or split — the same spikes, reorganised, with evidence — is not
a regression. **Injected ground truth remains the decisive layer**; this
criterion is a regression safeguard, not the definition of correctness.

**Promotion criteria, all required:**

| # | Criterion |
|---|---|
| 1 | Fixed-cohort ground truth: ≥ legacy on units-recovered-at-accuracy-0.8, on held-out **and** second session; no candidate-specific qualification exclusions |
| 2 | Strictly better on at least one of {high-motion, low-SNR} subsets |
| 3 | Preserves ≥ 95% of **high-confidence legacy-supported neuron families**, *or* accounts for each apparent loss by a validated merge/split relationship |
| 4 | No hard guardrail breach and no bounded guardrail regression beyond its prospectively frozen margin, including waveform preservation by stratum |
| 5 | Runtime ≤ 1.25× legacy per unit data |
| 6 | Held-out and second-session results consistent in direction |
| 7 | Candidate ranking does not reverse in the prespecified wider/longer context check |
| 8 | No material condition-dependent recovery distortion relative to legacy on the prespecified injected/event strata |

Failure at any point returns to Phase D. It does **not** justify a full-session
run to "check anyway" — that is exactly the days-long failure mode being
eliminated.

Passing these criteria licenses an **operational** claim: the pipeline improves
validated recovery with bounded costs in the tested domain. A causal statement
about *why* it improves—threshold learning, whitening context, motion
correction, or identity tracking—requires the corresponding stage trace or
intervention and must be stated at that evidence's narrower scope. Operational
adoption need not wait for the dense-donor motion mechanism to be complete.

## 7. Stop doing

- Broad, undirected preprocessing sweeps. Not because preprocessing cannot
  matter — the motion-representation hypothesis says it can — but because a
  sweep scored on yield is the wrong instrument. Preprocessing changes are
  tested through A2 and the C2 drift penalty, hypothesis-first.
- Truncation as a **population median across sorts**. The instrument is sound and
  is now a metric in its own right (§5); the fitter audit is complete. The one
  use that must not return is the unmatched population comparison that produced
  a false conclusion in 0008.
- **Tuning motion gain, interpolation scale, or correction gates against unit
  yield.** These parameters *may* be tested in a small prespecified
  operator-level tradeoff study (**D2b**) using injected truth, waveform
  preservation, estimator support, and identity metrics. The prohibition is on
  the *endpoint*, not on the parameters.
- Motion-aware TDC arms until the static replay is deterministic and exceeds a
  prespecified fidelity threshold. The current static control reproduced 7.8% of
  KS4 events and used a negative-only detector.
- Treating the claim mask as the solution — it suppresses real reviewed events
  as well as duplicate-like ones.
- Chasing the 27 similar pairs with threshold variants before defining an
  endpoint separating duplicated neurons from harmless waveform similarity.
- Yates parity as validation. Confounded by anatomy, depth, preprocessing and
  duration.
- Any full-session challenger run before a bounded known-identity test passes.

## 8. Risks

| Risk | Mitigation |
|---|---|
| **Overfitting to the panel** | Held-out half opened once; second-session replication required; candidate count logged |
| **Snippets change candidate ranking** | Score the shared central region across narrow/wide/full context and the same short interval alone/inside a longer sort; escalate selection context if rank reverses |
| **Injected benchmark omits Luke's failures** | Keep the 14 compact imec0 donors as regression tests; add a small frozen, independently reviewed imec1/weak-waveform extension; use a fixed performance denominator distinct from mechanism qualification |
| **Artifacts poison learning outside their local window** | Score one clean interval with versus without pathological surrounding context; if it changes, replay clean- versus contaminated-context preprocessing statistics once |
| **Ground truth ≠ biological truth** | Hybrid GT scores detection/assignment, not biology. Pair with the real-data symmetric audit; never let GT alone promote |
| **Benchmark improves while task-relevant spikes worsen** | Measure rate-modulation recovery and stratify recall/disagreement by collisions, artifacts and experimental events without selecting on plausible tuning |
| **Panel becomes stale** | Version and hash it; a panel change invalidates all cached scores by construction |
| **Speed encourages more variants than the statistics support** | Pre-register candidate count per checkpoint; treat late candidates as selection artifacts |

## 9. What lands first

### Complete

1. **Phase A — symmetric audit (v2).** `+210 / −137` KS-good, net +73. **No
   confirmed gross detection difference:** overlap supported for 208/210
   rescue-side and 132/137 legacy-side units; two and five unresolved. Exclusive
   one-to-one matching + circular-shift-null detection gate. This does not
   establish equivalence or superiority.
   Docs: [`rescue_unique_units_audit`](luke_20250804_rescue_unique_units_audit.md),
   [`rescue_lost_units_audit`](luke_20250804_rescue_lost_units_audit.md).
2. **Phase A2 — repartition audit (v2).** **0% coexisting fragments** on both
   probes (the prespecified ordinary-over-peeling signature is absent), but only
   **6–13% of fragment-cluster unions are
   refractory-clean** (median union RV 8–14%); ~90% of families `ambiguous`. No
   rigid-motion tracking (|r| ≈ 0.13–0.17). **Mechanism is ambiguous** —
   non-rigid/fast motion vs KS4 template competition are not separated here.
   Doc: [`rescue_repartition_motion_audit`](luke_20250804_rescue_repartition_motion_audit.md).
3. **Ladder infrastructure.** `ladder_score.py` (L1==L4 scoring dict),
   `ladder_snippets.py` / `ladder_snr.py` / `ladder_l1.py`, `ladder_sorter.py`
   (`RESCUE` / `LEGACY_STYLE`), `ladder_inject.py` + geometry-aware
   `ladder_motion.paired_geometry_motion_injection`. 16-snippet imec0 panel
   **frozen** (`panel_digest 07d5d808…`), pre-registered position-parity split.
   L1 measured well under the 5-minute budget at 120 s.
4. **Compact donor cohort (D2b-2).** 14 spatially-compact imec0 donors, both
   polarities, 73–295 µV, de-whitened KS shape scaled to bandpass-STA µV.
   Cohort hash-frozen. Corrected static rescoring removes the artificial ~0.78
   floor: 12/14 donors score 0.97–0.99 under both configurations and satisfy the
   accuracy qualification; 10/14 are also cleanly recovered under both. D01
   fails only under legacy-style, D10 only under rescue, and D05/D14 retain a
   real chance-null-supported second cluster. The pilot `T*` plateau donors are forbidden
   ([decision 0012](decisions/0012-c2-uses-compact-donor-cohort.md)).
   Doc: [`d2b2_donor_cohort`](luke_20250804_d2b2_donor_cohort.md).

### Completed C2 gate — useful control, not another pipeline blocker

5. **C2 v4 — paired static-vs-moving injected identity challenge.** All 14
   compact donors, geometry-aware forward motion, exclusive truth scoring,
   content-bound caches; a donor enters a mechanism-qualified drift contrast
   only if its static arm reaches accuracy ≥ 0.8 under both configurations in
   that contrast. Pipeline-performance comparisons retain the fixed cohort.
   V4 used a new frozen prespec/output namespace because v3 is an immutable void
   run with stale trajectories. The 392-cell run is complete
   ([result](luke_20250804_c2_v4_result.md)).

   *First run 2026-09-03 is VOID* — a scorer bug, not a C2 result. All 14
   donors failed static qualification at a near-constant ~0.78 accuracy because
   `ground_truth_scores` v2 matched injected truth against the pooled spike
   river and let background clusters steal ~10 % of events
   ([decision 0014](decisions/0014-injected-truth-scoring-is-per-cluster.md),
   [`c2_v3_scorer_validation_failure`](luke_20250804_c2_v3_scorer_validation_failure.md)).
   Fixed in v4: per-cluster exclusive matching plus chance-null split/merge
   gates, `SCORE_SCHEMA` → v3, with regression coverage. The exact 40 µm arm
   demonstrates both positional fragmentation and correction. The
   Luke-calibrated 5/11/22 µm arms are not interpretable as displacement tests,
   because fractional resampling attenuates the compact recorded donors while
   leaving them largely in place. A dense spatial donor model is needed to
   finish that causal question, but is not required to test threshold or
   identity-handling candidates against the exact staircase and real data.

### Completed threshold branch — no production change

6. **Stage 2 train-stability comparison.** The provenance-bound 588-cell run
   completed across 14 donors, 14 frozen train realisations and production 12/9
   versus candidates 8/8 and 9/9. Both candidates reduced the failure-rate point
   estimate, but neither paired 97.5% donor-bootstrap interval excluded zero.
   No candidate qualified; the frozen outcome is `no_threshold_change`.
   Fractional thresholds, L1C, held-out static and matched-real-data gates remain
   closed for these candidates. Doc:
   [`train_stability_stage2_result`](luke_c2_train_stability_stage2_result.md).

### Build and test now

Execute the delivery sequence at the top of this document. The options below
are an inventory, not simultaneous prerequisites for the first candidate.

- **Immediate decision checkpoint: real amplitude-completeness failures.** Follow
  the [bounded prescription](amplitude_completeness_next_step_prescription.md).
  Use existing QC to nominate one intervention, or close with insufficient
  evidence at the effort cap. A positive local result must preserve healthy
  intervals and be confirmed on an independently chosen case before broader
  escalation. This checkpoint does not replace the shared ladder.
- **Option A external voltage registration** — build/operator smoke tests may
  run now; its specific Luke-scale causal interpretation waits for the dense
  donor, but real-data pipeline performance does not. Start with one minimal
  demonstration and a written continuation/stop rule.
- **Option B unwarped motion-aware identity** — build/identity-preservation
  smoke tests may run now and may use the exact staircase mechanism; promotion
  still requires the same frozen panels and matched real-data checks.
- **Detection-threshold branch** — the 154-cell static grid and 15-sort
  determinism check are complete, but the exact-staircase and sentinel results
  invalidated the single-realisation ranking of 8/8 and 9/9. The frozen 588-cell
  Stage 2 comparison is complete and did not separate either candidate from
  12/9 at the preregistered donor-level confidence threshold
  ([result](luke_c2_train_stability_stage2_result.md)). **Close this branch with
  no threshold change:** do not run fractional refinement, L1C, held-out static
  or matched real data for 8/8 or 9/9.
- **Stage/context diagnostics** — use retained evidence when it discriminates
  the nominated failure. The completed threshold branch does not trigger another
  D10/D14 investigation. A surviving candidate still requires the prespecified
  spatial/temporal context check before trusting the cheap benchmark's ranking.
- **Bounded donor extension** — freeze a few independently reviewed imec1
  waveform families and weaker examples for fixed-denominator confirmation;
  do not replace the 14-donor regression cohort.
- **D2b** (field-error tolerance, interpolation tradeoff) — `d2b1` fails closed
  until the compact donor can be translated at fractional offsets without
  being dimmed in place; this is background motion-method research.
- **D2a** (best-supported full-session field) — infra built
  (`ladder_motion_estimate.py`), not run. Field estimation can proceed, but
  the causal injected-donor benefit at Luke scale cannot be quantified before
  the dense-donor control.
- **Alternative-sorter diagnostic / heavyweight gate** — one frozen,
  no-tuning alternative-sorter comparison may run early on D10/D14 and one
  context to localise implementation dependence. Developing an alternative
  sorter as a pipeline opens only if the threshold branch and two-option
  bakeoff leave a meaningful deficit after L2/L2L and matched real-data
  evaluation. D2 supports Option A; it is not a blanket prerequisite for B.

### The defensible claim now

> **Existing observational audits are consistent with identity instability;
> Phase A2 shows that naive unions of selected fragments are usually
> refractory-violating. C2 v4 establishes that exact 40 µm displacement can
> fragment identity and that rigid correction can repair it, but it does not
> establish a Luke-scale effect because the fractional-offset donor model dims
> rather than translates compact footprints. No current result shows that the
> rescue pipeline beats legacy. The threshold programme confirms that 12/9
> causes a reproducible D10 detection-stage failure, while D14-like collapses
> are contamination-driven and train-sensitive. Nevertheless the completed
> 588-cell Stage 2 comparison did not separate 8/8 or 9/9 from production at the
> frozen donor-level confidence threshold, so the operational decision is no
> threshold change and that branch is closed. The current development options
> are physically validated external voltage registration and unwarped
> motion-aware identity handling. The bounded amplitude-completeness checkpoint
> connects the next intervention to a recognizable real failure; no result from
> that checkpoint is yet established. Both options use the shared ladder and
> matched real-data confirmation. Operational improvement can be established
> before its complete mechanism; dense-donor modelling remains a parallel
> scientific task.**

### Historical (retracted 2026-09-03 — not active work)

The 2026-09-02 C2 diagnostic (pilot plateau donors, discrete-index motion,
non-exclusive scoring) reported a −0.30 to −0.81 drift penalty and that
`nblocks=1` did not recover it; Candidate 2's oracle was reported to close a
40 µm penalty to 0.99; D2b-1/2/3 reported a field-error envelope and an
interpolation ceiling. **All retracted** under
[decision 0011](decisions/0011-cross-sort-event-matching-and-detection-evidence.md)
and [decision 0012](decisions/0012-c2-uses-compact-donor-cohort.md). These
numbers must not appear in active checkpoint status or decision logic; the
detail is preserved in the labelled historical blocks of §C2 and §D2.
