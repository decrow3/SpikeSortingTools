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

Automated from `testing/luke_rescue_unique_units_audit.py` (exists, works):
gained good units, **lost good units**, and their classification. Reported as
`+N / −M`, never as a net. The 127 losses were invisible for months precisely
because only the net was reported.

### Guardrails (any breach blocks promotion)

- Similar good–good pairs per good unit (similarity ≥ 0.8 within 100 µm)
- Refractory violation distribution vs the matched-unit reference
- Edge-spike fraction (40 µm)
- **Matched-unit completeness** — paired truncation difference, censored windows
  filtered
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

### Phase A2 — is the repartitioning motion-structured? (no new sort)  ✅ complete 2026-09-02

Runs entirely on existing outputs. Hours, not days. **This gates Phase D's
priority order**, and could change what the whole candidate search is for.

**Result:** [`luke_20250804_rescue_repartition_motion_audit.md`](luke_20250804_rescue_repartition_motion_audit.md).
`testing/luke_rescue_repartition_motion_audit.py` (prespec frozen in `PRESPEC`;
+ `test_…`, 6 tests). 117 imec0 + 62 imec1 dispersed families scored. Consistent
across both probes:

| | imec0 | imec1 |
|---|---:|---:|
| coexisting fragments (over-splitting signature) | **0%** | 1.6% |
| successive (one fragment at a time) | 50% | 69% |
| merge is refractory-clean | 92% | 95% |
| depth ↔ rigid-DREDGE-motion correlation | 0.11 | 0.13 |
| ownership flips per hour | 18 | 22 |

**Reading (the mix, not a verdict):** the re-partitioning is **not over-splitting**
(fragments never coexist), the fragments **are one clean neuron** (merge is
refractory-clean → family stitching would recover them), and it is **not tracked
by the rigid motion estimate** and **not slow** — it is rapid template-ownership
flicker (~every 3 min) with no depth trajectory. A2 cannot separate *non-rigid /
fast motion* from *KS4 template competition on preserved voltage*; both produce
this. **C2 separates them.**

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

*Reached 2026-09-02.* The mix points away from the "clustering/curation first"
branch (coexisting fragments ~0%) and toward **post-sort family stitching of
temporally-complementary, refractory-clean fragments** as the first Phase D
target — indicated whether the root cause is non-rigid motion or template
competition, because stitching repairs both flicker and slow drift while
curation tuning addresses neither. **C2 is still required** to frame the cause
before the search starts.

*Superseded 2026-09-02.* Stitching was built and rejected — it reconstitutes
only 2 of the 127 on the full session and loses matches on net (see Phase D
Candidate 1). The A2 "family stitching would recover them" reading held only at
snippet scale; full-session the dispersed fragments are too finely smeared.
Phase D target is now Candidate 2, a non-rigid motion representation.

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

1. ~~Wire the sealed injection scaffold to the L1 runner~~ — **done 2026-09-02**,
   `testing/ladder_inject.py` (+ `test_ladder_inject.py`, 9 tests).
   `inject_trajectory` schedules per-spike `InjectionEvent`s along a known
   channel trajectory and calls the sealed `inject_float32_raw_domain`
   unchanged (float32 µV view only — never the stored int16).
   `write_injected_recording` quantises back to int16 and writes an
   accepted-recording folder + manifests, so `l1_run` sorts and
   `score_sort(truth=…)` scores it. `drift_penalty(static, moving)` is the
   decisive Δ.
2. **Benchmark validated 2026-09-02** — the rescue pipeline recovers the
   high-SNR (SNR 11) donor template T01, injected static into a quiet imec1
   strip, at **accuracy 0.94 ≥ 0.9**. The benchmark is sound (legacy arm still
   pending the config-parametrised sorter).
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

Because the injected train is known, C2 also **calibrates the truncation
estimator against truth** for the first time, and separates the two
fragmentation modes: temporal splitting should leave per-fragment truncation
flat, amplitude splitting should elevate it.

*Confound to control:* the background contains real tissue motion of its own, so
an injected trajectory interacts with it. Either define the trajectory relative
to the estimated tissue position, or draw the static arm from quiet windows and
say so. Record which was done.

**First run 2026-09-02** — [`luke_20250804_c2_drift_challenge.md`](luke_20250804_c2_drift_challenge.md).
`testing/luke_rescue_c2_drift_challenge.py` (prespec frozen; diagnostic — reuses
the pilot's discovery-cohort donor templates; static arm from the quiet imec1
window). Benchmark **sane**: static T01 (SNR 11) / T04 (SNR 6) recovered at
0.94–0.98 under both sorter configs.

**Drift penalty (Δ accuracy = moving − static), both arms:**

| donor | trajectory | rescue | legacy_style (`nblocks=1`) |
|---|---|---:|---:|
| T01 | rigid 15 µm | −0.35 | −0.32 |
| T01 | rigid 40 µm | −0.54 | **−0.81** |
| T01 | osc 20 µm/40 s | −0.70 | −0.68 |
| T04 | rigid 15 µm | −0.30 | **−0.58** |
| T04 | rigid 40 µm | −0.53 | −0.31 |
| T04 | osc 20 µm/40 s | −0.38 | −0.49 |

Two findings, both decisive for Phase D:

1. **Motion alone costs 30–80 accuracy points, under both configs** — mostly
   missed spikes (T01 rigid-40: FN 24→314 rescue, 11→503 legacy) and identity
   proliferation (T04 rigid-40: +15/+12 output units on one train). This
   reproduces the A2 fragmentation signature **causally**: motion is a
   sufficient cause.
2. **KS4 rigid internal drift correction does not recover it** — `legacy_style`
   is worse on 3 of 6 clean conditions, comparable on 2, better on 1, and piles
   up false positives on the oscillation (FP 874 vs 473). Turning `nblocks` back
   on is **not the fix.**

→ Phase D's motion target is **non-rigid handling, a better estimate, or
post-sort family stitching** — not `nblocks=1`. Curation-threshold tuning stays
low: the lower-threshold `legacy_style` fragments *more* at baseline, not less.

Still to add: non-rigid trajectories, both polarities, truncation-vs-truth,
a second window. T06 (SNR 4.6) excluded — static baseline below sanity.

**Checkpoint C.** *Go:* a known-truth score exists for legacy on all 8
development snippets with the sanity condition met, **and** a measured drift
penalty for both legacy and rescue. This is the first point at which "better"
becomes measurable, and together with A2 it sets Phase D's target.

*Partially reached 2026-09-02.* The **drift-penalty half is done** (diagnostic):
measured for both the rescue config and KS4-with-rigid-correction, on injected
imec1 snippets. Motion is a sufficient cause of A2's fragmentation, and rigid
correction does not fix it. The **panel-baseline half** (legacy score on all 8
dev snippets) waits on the frozen panel. Phase D's *direction* is set; its
*promotion baseline* is not yet.

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

**The tree has resolved (2026-09-02).** A2: fragments are temporally
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

**Candidate 1 built, evaluated, and rejected 2026-09-02** —
[`luke_20250804_family_stitch_candidate.md`](luke_20250804_family_stitch_candidate.md),
`testing/ladder_stitch.py` (+ tests), `testing/luke_rescue_stitch_c2_eval.py`,
`testing/luke_rescue_stitch_fullsession_eval.py`.
`stitch_families` merges mutual-best-partner units that are spatially plausible,
temporally complementary, not simultaneous, and refractory-clean on merge. On
the C2 injected-truth pairs it is safe at snippet scale (0 families on every
static arm) and helps two mild-drift arms (+0.16, +0.23 accuracy). **But the
decisive full-session test fails.** Run on the whole imec0 rescue sort against
the 127 legacy-lost units: it reconstitutes **2** of the 127, destroys **4**
existing legacy matches by over-merging, absorbs **34** genuine rescue good
units, and takes the legacy-match count from 101 to **99** — a net loss. The 82
"dispersed" units (the bulk of the 127) are smeared across 5–15
contamination-dominated clusters each; there is no clean 2–4-member family to
rejoin. **Verdict: not adopted.** The motion fragmentation must be *prevented*,
not repaired → Candidate 2 (non-rigid representation). `ladder_stitch.py` stays
as a tested negative result and family-detection primitive, not a curation
stage.

**Candidate 2 built and evaluated 2026-09-02** —
[`luke_20250804_nonrigid_motion_candidate.md`](luke_20250804_nonrigid_motion_candidate.md),
`testing/ladder_motion.py` (oracle correction), `ladder_sorter.NONRIGID`
(`do_correction=True, nblocks=6`), `testing/luke_rescue_c2_nonrigid_eval.py`.
Three arms on the cached C2 injected recordings — `rescue` (baseline),
`nonrigid` (KS4 datashift, the estimated case), `oracle` (correct with the
**exact** injected trajectory, then rescue sort — the ceiling).
**A non-rigid representation is a real lever** — the first candidate that is.
Oracle correction closes the severe rigid-drift penalty on both clean donors:
T04 and T06 at 40 µm go from accuracy ≈ 0.40 back to ≈ 0.99 (static baseline),
median penalty −0.35 → −0.17. Static arms untouched (0.941→0.941, 0.948→0.948).
**But** (a) KS4's own `nblocks=6` is *worse than no correction* on a snippet
(median −0.48) — too few units, too short a window to estimate drift; and (b)
interpolation blurs the waveform, so it does nothing for 15 µm (sub-channel)
drift and *hurts* T01, the sharpest/highest-SNR donor (0.40→0.29 at 40 µm).
→ Next: an **external non-rigid estimate computed on the full session** (where a
drift estimate has 10 000 s and hundreds of units), then interpolation, then the
rescue sort — scored against the 127 (full-session reconstitution, the test that
killed stitching) and against C2 injected truth per SNR tertile. If it
reconstitutes the 127 with no similar-pair/edge-spike regression it is the Phase
D winner → L2. If not, rescue vs legacy resolves as "trade the same errors,
neither clearly better on this recording; lever is elsewhere".

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
- Truncation as a **population median across sorts**. The instrument is sound and
  is now a metric in its own right (§5); the fitter audit is complete. The one
  use that must not return is the unmatched population comparison that produced
  a false conclusion in 0008.
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
4. ~~Injection wired to the sorter (Phase C)~~ — done 2026-09-02,
   `testing/ladder_inject.py` + `luke_rescue_c2_drift_challenge.py`. Benchmark
   validated (static T01 recovered at accuracy 0.94); first drift penalty
   measured (−0.54 accuracy for a 40 µm rigid ramp, rescue arm).
5. ~~Define + freeze the real 16-snippet panel~~ — **frozen 2026-09-02**,
   `panel_digest 07d5d808…`, `testing/luke_ladder_panel.py`. imec0, 16 snippets,
   pre-registered position-parity split, SNR/artifact measured not chosen.
6. ~~Config-parametrised sorter~~ — done 2026-09-02, `testing/ladder_sorter.py`
   (`SorterConfig`, `RESCUE`, `LEGACY_STYLE` = `nblocks=1, Th 9/8` — the two
   `ops.npy` diffs); `l1_run(sorter=…)` caches each config at its own leaf.
   Unblocks C2's legacy arm and the Phase D candidate search.

**Then, before any curation parameter search** — these two results decide what
the next block of time is spent on, and neither needs a new sorter:

7. ~~**Phase A2**: the temporal-fragment analysis of existing sorts.~~ **done
   2026-09-02** — `testing/luke_rescue_repartition_motion_audit.py`,
   [`luke_20250804_rescue_repartition_motion_audit.md`](luke_20250804_rescue_repartition_motion_audit.md).
   Not over-splitting (0% coexisting), fragments are one clean neuron (92–95%
   refractory-clean merges), not rigid-motion-tracked (|r| ≈ 0.12), rapid
   flicker not slow succession. imec0 and imec1 agree.
8. **Phase C2**: the paired stationary-vs-moving injected identity challenge.
   Same waveform, same train, one held still and one translated along a known
   trajectory. Measures the drift penalty directly. **Now the single gating
   item** before Phase D — A2 has reported; C2 has not.

Phase D does not begin until 8 reports. The decision tree in Phase D consumes
A2's output (above) and C2's.

The defensible claim until then:

> **The current pipeline avoids a proven damaging motion-warp path and preserves
> the unwarped voltage, but this may trade interpolation damage for
> motion-driven identity fragmentation. Whether rescue, legacy, or a
> motion-aware unwarped approach best preserves neuron identity remains
> undetermined.**
