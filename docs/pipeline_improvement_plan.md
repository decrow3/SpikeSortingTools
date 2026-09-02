# Plan: beat the legacy pipeline, and find out fast

**Status:** proposed 2026-09-02
**Supersedes as a work plan:** the follow-up lists in
[`decisions/0008`](decisions/0008-amplitude-completeness-gates-promotion.md) and
[`decisions/0010`](decisions/0010-rescue-yield-is-relabelling-not-detection.md)
**Related:** [`0006`](decisions/0006-recovery-axis-is-post-sort-mua-reconciliation.md),
[`0007`](decisions/0007-stage-local-validation.md),
[`0009`](decisions/0009-cross-sort-comparisons-must-be-unit-matched.md)

## 1. The goal, as a testable claim

> A candidate pipeline is **promotable** when, on data it has never been tuned
> against, it recovers more known-identity neurons correctly than legacy does,
> loses no well-supported neuron legacy found, and does not cost materially more
> runtime.

Four clauses, each independently falsifiable. All four must hold. Yield is not
one of them — [`0010`](decisions/0010-rescue-yield-is-relabelling-not-detection.md)
showed a +32% KS-good headline that was entirely relabelling.

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
| **L0** | Unit + contract tests | **< 1 min** | Every commit (exists: 385 tests, ~20 s) |
| **L1** | One snippet, full pipeline, scored | **< 5 min** | Every parameter/code change |
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
score the sort against truth. **This is ~70% built already**:
`testing/luke_injected_ground_truth_benchmark.py` holds the sealed manifest and
the injection/scoring primitives, validated on synthetic arrays; the pilot
traces injected deltas through conditioning. Neither has ever been connected to
a sorter. Connecting them is the single highest-leverage piece of work in this
plan.

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

### Secondary — real-data symmetric agreement

Automated from `testing/luke_rescue_unique_units_audit.py` (exists, works):
gained good units, **lost good units**, and their classification. Reported as
`+N / −M`, never as a net. The 127 losses were invisible for months precisely
because only the net was reported.

### Guardrails (any breach blocks promotion)

- Similar good–good pairs per good unit (similarity ≥ 0.8 within 100 µm)
- Refractory violation distribution vs the matched-unit reference
- Edge-spike fraction (40 µm)
- **Runtime per unit data**, tracked at every tier

### Explicitly not endpoints

KS-good count. Total spikes. Stable-bin occupancy. Population medians of any
per-unit metric across sorts with different unit populations
([`0009`](decisions/0009-cross-sort-comparisons-must-be-unit-matched.md)).

## 6. Phases, checkpoints, go/no-go

### Phase A — the symmetric audit (no new compute)  ✅ complete 2026-09-02

Runs on existing outputs; can proceed in parallel with Phase B.

**Result:** [`luke_20250804_rescue_lost_units_audit.md`](luke_20250804_rescue_lost_units_audit.md).
`testing/luke_rescue_lost_units_audit.py` classifies all 127 (~35 s on existing
outputs). **0 lost at detection, 0 removed by curation** — every one has 100% of
its spikes in the rescue sort. The −127 is 27 legacy-good→`mua` demotions (the
mirror of the 80 MUA→good promotions) plus **100 re-clustered**. Checkpoint A's
regression trigger did not fire.

**Interpretation — corrected 2026-09-02.** An earlier draft of this section
concluded that "curation/clustering remain the only stages that differ." That
is a stage-*localization* claim smuggled in as a *cause* claim, and it is
exactly the error [`0007`](decisions/0007-stage-local-validation.md) exists to
prevent. The correct statement:

> The observable discrepancy occurs at assignment/clustering rather than gross
> detection. **This does not localize its cause.** Different preprocessing or
> motion representations can preserve the same event pool while changing how a
> moving neuron is partitioned into identities.

The two halves of the −127 are different phenomena and must not be pooled:

| | n | Nature |
|---|---:|---|
| legacy-good → `mua` demotions | 27 | A **labelling-threshold** issue — the mirror of the 80 MUA→good promotions. One moved threshold, to be decided as one. |
| **re-clustered** | **100** | A **repartitioning** phenomenon. Motion-driven fragmentation is now a leading candidate explanation, alongside ordinary over-splitting. |

The leading mechanistic hypothesis for the 100, and for the corresponding
dispersed/split gains among the +200:

> Legacy partially stabilized moving neurons by resampling voltage. Rescue
> preserves the voltage but leaves KS4 to represent a moving waveform footprint,
> which it can only do by splitting it across templates.

This is consistent with the earlier rescue work, which explicitly retained
motion-driven fragmentation and an unwarped alternative that links temporally
fragmented clusters or lets templates follow tissue. **Phase A2 tests it before
any candidate search begins.**

*Tension that A2 must confront:* on imec0 the DREDGE rigid sidecar reports only
**1.28 µm** of total drift (6.4% of one site pitch). If the 100 are
motion-fragmented on *this* probe, the cause must be non-rigid, or that estimate
must be wrong — it is rigid-only and QC-unqualified, with `weights_thresh`
entirely non-finite. A2 therefore runs on **imec1 as well**, where the motion is
known to be real.

1. Classify all **127 legacy-good units rescue does not reproduce**: preserved
   as MUA, split, merged, dispersed, absent at detection, or rejected by
   curation. Reuse the coincidence machinery already written.
2. Apply the same waveform/refractory/amplitude evidence used for the 200 gains.

**Checkpoint A.** *Go:* a symmetric `+200 / −127` table with both sides
classified. *Decision:* if a substantial share of the 127 are genuine neurons
lost at detection or curation, that is a regression the yield narrative hid, and
it becomes the top-priority defect.

### Phase A2 — is the repartitioning motion-structured? (no new sort)

Runs entirely on existing outputs. Hours, not days. **This gates Phase D's
priority order**, and could change what the whole candidate search is for.

**Prespecify before looking**, or this becomes story-fitting: freeze the sample
of legacy↔rescue families and the decision rule first, then run once.

1. Sample the strongly **dispersed** legacy↔rescue families — the 100
   re-clustered losses and the 85 dispersed gains — stratified by depth and by
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
   validation snippets then deleted — the panel is not yet frozen). Snippet root
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
   tertiles low/medium/high. ~18 s/snippet (build + profile). What remains for
   this item: choose the 16 concrete regime×strip×SNR×artifact combinations and
   fix the dev/held-out split *before* any result is seen, then `freeze_panel`.
3. ~~**`score_sort(sorter_output, snippet) -> dict`**~~ — **done 2026-09-02**,
   `testing/ladder_score.py` (+ `test_ladder_score.py`, 8 tests). One function,
   three layers (primary hybrid-GT / secondary symmetric agreement / guardrails)
   + a provenance-only `context` block that flags the non-endpoints. Reuses the
   coincidence machinery from `luke_rescue_unique_units_audit.py` unchanged.
   Validated on the full imec0 pair: reproduces Phase A exactly (`matched 101,
   gained 200, lost 127, lost_absent 0`) and the known guardrail numbers (27
   similar good–good pairs, refractory median 0.10%). ~90 s at full-session
   scale; sub-second on a snippet. `window_reference_sort` re-bases and
   depth-cuts a full-session reference so a snippet sort can be compared to it.
   Still needs: injected-truth wired through (Phase C — the `truth=` path is
   built and tested, just not fed real injections yet).
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
   fresh snippet-scale rescue sort has 16 (`lost_absent_at_detection = 0`, so
   the spikes are all detected — it is the Phase A relabelling story, amplified
   because 120 s is too short for many real units to clear a KS-good bar). The
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

1. Wire the sealed injection scaffold to the L1 runner: inject into raw-domain
   `float32` (the existing contract already forbids injecting into stored
   `int16`), then run the real pipeline over the injected snippet.
2. Validate the benchmark itself: legacy and rescue must both recover the
   easy high-SNR, no-drift injections at accuracy ≥ 0.9. **If they do not, the
   benchmark is wrong, not the pipelines.**
3. Establish the legacy baseline score on the development panel.

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

*Confound to control:* the background contains real tissue motion of its own, so
an injected trajectory interacts with it. Either define the trajectory relative
to the estimated tissue position, or draw the static arm from quiet windows and
say so. Record which was done.

**Checkpoint C.** *Go:* a known-truth score exists for legacy on all 8
development snippets with the sanity condition met, **and** a measured drift
penalty for both legacy and rescue. This is the first point at which "better"
becomes measurable, and together with A2 it sets Phase D's target.

### Phase D — candidate search

Only now does pipeline variation begin. Each candidate: L1 on one snippet, then
L2 on the panel, then stop. Log every candidate and its score.

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

**Orthogonal, and deliberately later:** the MUA threshold question — the 80
promotions and their 27 mirrored demotions. It is one moved threshold. Changing
`good` versus `mua` labels **will not fix identity fragmentation**, so it should
not compete for priority with the repartitioning question. It still needs
[`0006`](decisions/0006-recovery-axis-is-post-sort-mua-reconciliation.md)'s
reversible family-link evidence, scored on a blinded stratified sample rather
than by inspecting all 80.

Any motion handling is judged on **identity continuity at L2L**, never on unit
counts, and requires a faithful deterministic static arm first.

**Checkpoint D.** *Go:* at least one candidate beats legacy on the primary
metric on the development panel, with all guardrails intact and runtime within
budget.

### Phase E — held-out, replication, promotion

1. Run the L2 winner through **L2L** — one full-duration narrow depth strip, for
   longitudinal identity continuity. A candidate that fragments over the full
   duration stops here, whatever its snippet scores.
2. Run on the **held-out panel** (opened once).
3. Run on a **second session** — the panel construction must be re-run there
   from scratch.
4. Only then L4 full session.

**On criterion 3 — revised 2026-09-02.** The original wording, "preserves ≥ 95%
of legacy good units", quietly made *reproduce the old clustering* an objective.
That is wrong here. If legacy motion correction sometimes held one neuron
together by damaging the voltage, while rescue preserves the voltage but
fragments it, **neither set of cluster IDs deserves privileged status** — and
Phase A already showed the −127 contains zero detection losses, so "legacy good
unit" is a labelling and partitioning artifact as much as a fact about neurons.

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
| 1 | Ground truth: ≥ legacy on units-recovered-at-accuracy-0.8, on held-out **and** second session |
| 2 | Strictly better on at least one of {high-motion, low-SNR} subsets |
| 3 | Preserves ≥ 95% of **high-confidence legacy-supported neuron families**, *or* accounts for each apparent loss by a validated merge/split relationship |
| 4 | All guardrails ≤ legacy |
| 5 | Runtime ≤ 1.25× legacy per unit data |
| 6 | Held-out and second-session results consistent in direction |

Failure at any point returns to Phase D. It does **not** justify a full-session
run to "check anyway" — that is exactly the days-long failure mode being
eliminated.

## 7. Stop doing

- Broad, undirected preprocessing sweeps. Not because preprocessing cannot
  matter — the motion-representation hypothesis says it can — but because a
  sweep scored on yield is the wrong instrument. Preprocessing changes are
  tested through A2 and the C2 drift penalty, hypothesis-first.
- Further truncation-fitter work unless it serves a matched-unit question.
  Audit complete; estimator sound in its working range.
- Tuning motion amplitudes or kernels against unit yield.
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
| **Snippets unrepresentative of full session** | Checkpoint B explicitly tests reproduction of a known full-session result |
| **Injected truth unrealistic** | Injected waveforms drawn from reviewed real events; benchmark validated by requiring both pipelines to solve easy cases |
| **Ground truth ≠ biological truth** | Hybrid GT scores detection/assignment, not biology. Pair with the real-data symmetric audit; never let GT alone promote |
| **Panel becomes stale** | Version and hash it; a panel change invalidates all cached scores by construction |
| **Speed encourages more variants than the statistics support** | Pre-register candidate count per checkpoint; treat late candidates as selection artifacts |

## 9. What lands first

1. ~~Phase A classification of the 127 losses~~ — done (2026-09-02): no
   detection or curation-drop regression; the −127 mirrors the +200.
2. ~~Snippet builder + `score_sort` + L1 runner~~ — the ladder's load-bearing
   piece, done 2026-09-02: `testing/ladder_score.py`, `ladder_snippets.py`,
   `ladder_snr.py`, `ladder_l1.py` (30 tests).
3. ~~Tier calibration against the 5-minute constraint~~ — L1 measured well
   under budget at 120 s (build 26 s + KS4 38 s + curation <5 s + score 23 s).
4. Injection wired to the sorter (Phase C) — `score_sort`'s `truth=` path is
   built and tested; still needs real injections fed through `l1_run`.
5. Define + freeze the real 16-snippet panel (regime × strip × SNR × artifact),
   dev/held-out split fixed before any result is seen.
6. Config-parametrised sorter so the legacy pipeline can be run at snippet scale
   as the secondary-metric comparator (also unblocks Phase D candidate search).

**Then, before any curation parameter search** — these two results decide what
the next block of time is spent on, and neither needs a new sorter:

7. **Phase A2**: the temporal-fragment analysis of existing sorts. Are the 100
   re-clustered losses and the dispersed gains temporally complementary and
   motion-tracking, or coexisting? Prespecify the sample and the discriminator,
   then run once. imec0 **and** imec1.
8. **Phase C2**: the paired stationary-vs-moving injected identity challenge.
   Same waveform, same train, one held still and one translated along a known
   trajectory. Measures the drift penalty directly.

Phase D does not begin until 7 and 8 report. The decision tree in Phase D
consumes their output.

The defensible claim until then:

> **The current pipeline avoids a proven damaging motion-warp path and preserves
> the unwarped voltage, but this may trade interpolation damage for
> motion-driven identity fragmentation. Whether rescue, legacy, or a
> motion-aware unwarped approach best preserves neuron identity remains
> undetermined.**
