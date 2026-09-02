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
> preserves the units legacy already found, and does not cost materially more
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

### Phase A — the symmetric audit (no new compute)

Runs on existing outputs; can proceed in parallel with Phase B.

1. Classify all **127 legacy-good units rescue does not reproduce**: preserved
   as MUA, split, merged, dispersed, absent at detection, or rejected by
   curation. Reuse the coincidence machinery already written.
2. Apply the same waveform/refractory/amplitude evidence used for the 200 gains.

**Checkpoint A.** *Go:* a symmetric `+200 / −127` table with both sides
classified. *Decision:* if a substantial share of the 127 are genuine neurons
lost at detection or curation, that is a regression the yield narrative hid, and
it becomes the top-priority defect.

### Phase B — build the ladder

1. **Snippet builder**: extract time × depth-strip snippets with receipts, from
   the accepted recording, reusing `resolve_bakeoff_window`'s fingerprinting.
2. **SNR stratification** to complete the panel axes.
3. **`score_sort(sorter_output, snippet) -> dict`**: one function, all endpoints,
   used identically at L1 and L4.
4. **L1 runner**: snippet → preprocess → sort → curate → score, cached, one
   command.
5. **Calibrate the tiers.** Measure actual wall clock. If L1 exceeds 5 minutes,
   shrink the snippet until it does not. The 5-minute number is the design
   constraint, not an estimate to be discovered.

**Checkpoint B.** *Go:* L1 runs end-to-end in < 5 min and L2 in < 45 min; the
harness reproduces the known legacy-vs-rescue difference on a snippet where the
full-session answer is already known. *No-go:* if the snippet result does not
reproduce the known full-session direction, the panel is unrepresentative — fix
that before trusting any L1 result.

### Phase C — connect ground truth to a sorter

1. Wire the sealed injection scaffold to the L1 runner: inject into raw-domain
   `float32` (the existing contract already forbids injecting into stored
   `int16`), then run the real pipeline over the injected snippet.
2. Validate the benchmark itself: legacy and rescue must both recover the
   easy high-SNR, no-drift injections at accuracy ≥ 0.9. **If they do not, the
   benchmark is wrong, not the pipelines.**
3. Establish the legacy baseline score on the development panel.

**Checkpoint C.** *Go:* a known-truth score exists for legacy on all 8
development snippets, with the sanity condition above met. This is the first
point at which "better" becomes measurable.

### Phase D — candidate search

Only now does pipeline variation begin. Each candidate: L1 on one snippet, then
L2 on the panel, then stop. Log every candidate and its score.

Priority order, from what the evidence actually implicates:

1. **Curation and clustering** — [`0010`](decisions/0010-rescue-yield-is-relabelling-not-detection.md)
   showed the rescue-vs-legacy difference lives entirely here. This is where the
   available gain is.
2. **The 80 MUA promotions** — provisional until they pass
   [`0006`](decisions/0006-recovery-axis-is-post-sort-mua-reconciliation.md)'s
   reversible family-link evidence. Score a blinded stratified sample rather
   than inspecting all 80.
3. **Motion handling** — only with a faithful, deterministic static arm first,
   and only judged on identity continuity through high-motion snippets. Not on
   unit counts.

**Checkpoint D.** *Go:* at least one candidate beats legacy on the primary
metric on the development panel, with all guardrails intact and runtime within
budget.

### Phase E — held-out, replication, promotion

1. Run the L2 winner on the **held-out panel** (opened once).
2. Run on a **second session** — the panel construction must be re-run there
   from scratch.
3. Only then L4 full session.

**Promotion criteria, all required:**

| # | Criterion |
|---|---|
| 1 | Ground truth: ≥ legacy on units-recovered-at-accuracy-0.8, on held-out **and** second session |
| 2 | Strictly better on at least one of {high-motion, low-SNR} subsets |
| 3 | Preserves ≥ 95% of legacy good units (the 127-loss criterion) |
| 4 | All guardrails ≤ legacy |
| 5 | Runtime ≤ 1.25× legacy per unit data |
| 6 | Held-out and second-session results consistent in direction |

Failure at any point returns to Phase D. It does **not** justify a full-session
run to "check anyway" — that is exactly the days-long failure mode being
eliminated.

## 7. Stop doing

- Broad preprocessing sweeps. [`0010`](decisions/0010-rescue-yield-is-relabelling-not-detection.md):
  rescue detects nothing legacy did not, so the difference is not in
  preprocessing.
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

1. Phase A classification of the 127 losses — existing tools, no new compute.
2. Snippet builder + `score_sort` + L1 runner — the ladder's load-bearing piece.
3. Tier calibration against the 5-minute constraint.
4. Injection wired to the sorter.

The defensible claim until then remains: *the current pipeline avoids a proven
damaging motion-warp path and improves conventional QC, but whether it sorts
neurons more correctly is undetermined.*
