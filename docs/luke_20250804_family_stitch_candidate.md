# Phase D candidate 1: post-sort family stitching — helps mild drift at snippet scale, fails the full session

> **RETRACTED PENDING V2 RERUN — 2026-09-03.** Both supports were invalidated:
> C2 used non-inverse motion operators and non-exclusive truth scoring, while
> the full-session test inherited the invalid 127-unit cohort and non-exclusive
> matcher. The +0.16–0.23, 2 recovered, and 4 destroyed results are historical
> only. Family stitching's status is unresolved until both corrected evaluations
> are regenerated.

**Date:** 2026-09-02
**Advances:** Phase D of [`pipeline_improvement_plan.md`](pipeline_improvement_plan.md)
**Module:** `testing/ladder_stitch.py` (+ `test_ladder_stitch.py`, 9 tests)
**Evaluation:** `testing/luke_rescue_stitch_c2_eval.py` (C2 injected truth) and
`testing/luke_rescue_stitch_fullsession_eval.py` (the 127 legacy-lost units) —
both pure post-processing on cached sorts, no sorter run.

## Verdict

Family stitching is **not adopted**. At snippet scale it is safe and adds
+0.16–0.23 accuracy on the two mild-rigid-drift arms. But the decisive test —
does it put back the 127 legacy-good units rescue drops on the *full* imec0
session, the regime where A2 says the fragments are clean and substantial — it
fails: it reconstitutes **2** of the 127, destroys **4** existing legacy
matches by over-merging, and absorbs **34** genuine rescue good units, taking
the legacy-match count from 101 to 99. The A2 "dispersed across many clusters"
structure is not a set of clean 2–4-member families that a pairwise-chain
stitch can rejoin. The motion fragmentation has to be prevented, not repaired
(Phase D candidate 2 — a non-rigid motion representation).

## Why this candidate

A2 + C2 established: rescue's KS-good deficit against legacy is one clean neuron
whose spikes are partitioned across templates as it drifts, and KS4 rigid
correction does not fix it. A2 showed the full-session dispersed families are
**temporally complementary** (≈ 0 % co-fire) and **refractory-clean when
merged** (92–95 %). Production curation
(`pipeline.curation`) will not merge them — its CCG gate *requires* the two units
to co-fire cleanly, the opposite of a temporally-complementary pair.

`stitch_families` merges a group of units when all of:

* **spatial** — peak-channel depths within `depth_window_um` (150 µm, wide
  enough for a drift trajectory), template cosine ≥ 0.30 after depth-alignment
  (a weak floor only — low-count fragment templates are unreliable);
* **temporal** — pairwise `temporal_overlap` ≤ 0.25 (the A2 metric: Σ min / Σ max
  of per-bin spike shares);
* **not simultaneous** — < 3 % coincident spikes (±0.5 ms);
* **refractory** — merged ISI-violation fraction ≤ 1.0 % (**the primary gate** —
  merging a genuine fragment stays clean; merging in a contaminated cluster does
  not).

Units → nodes. An edge is drawn only between **mutual best partners** — each
node's highest-cosine stitchable partner is the other. This matters at
full-session scale: the pairwise gates alone qualify ~3 000 pairs (any two
low-rate units fire in disjoint 30 s bins and merge refractory-clean), and a
transitive closure over them collapses the probe into four mega-components
(266/141/62/10 nodes). The mutual-best rule cuts that to 83 pair-edges.
Connected components of those (≤ 4 units, must contain a KS-good member) →
families, each relabelled to its largest good member. `include_mua` is on: KS4
demotes drift fragments to `mua`.

## Result on the C2 injected-truth pairs

| donor | trajectory | acc before | acc after | Δ | families |
|---|---|---:|---:|---:|---:|
| T01 | static | 0.94 | 0.94 | 0.00 | 0 |
| T01 | rigid 15 µm | 0.59 | **0.75** | **+0.16** | 1 |
| T01 | rigid 40 µm | 0.40 | 0.40 | 0.00 | 0 |
| T01 | osc 20 µm/40 s | 0.24 | 0.24 | 0.00 | 0 |
| T04 | static | 0.95 | 0.95 | 0.00 | 0 |
| T04 | rigid 15 µm | 0.65 | 0.65 | 0.00 | 0 |
| T04 | rigid 40 µm | 0.42 | 0.42 | 0.00 | 1 (not the limiting one) |
| T04 | osc 20 µm/40 s | 0.57 | 0.57 | 0.00 | 0 |
| T06 | rigid 40 µm | 0.38 | **0.62** | **+0.23** | 1 |

(T06 static baseline is 0.50 — below the sanity bar — so its Δ is noisy.)

**Three things this shows:**

1. **At snippet scale it is safe.** Zero families on every static arm; never
   hurts any arm.
2. **It sometimes helps at snippet scale** — +0.16 to +0.23 accuracy, and it
   removes label switches, on the mild-drift arms where KS4 produced a few
   clean, temporally-complementary fragments.
3. **It is not a complete fix.** Severe drift (40 µm rigid on the higher-SNR
   donors, every oscillation) is not recovered. Inspection of T04 rigid-40 µm:
   the injected train is smeared across ~15 clusters, and most are
   **contamination-dominated** — a background unit that caught 100–200 drifting
   spikes among its own 1000–3000. Merging those accumulates contamination
   (merge-all: accuracy 0.10, 41 % refractory violations).

## The decisive test: the full imec0 session and the 127

A2 predicted that stitching would work *better* on the full session than on a
120 s snippet — the dispersed families there are longer and cleaner. The
opposite is true. `luke_rescue_stitch_fullsession_eval.py` runs `stitch_families`
on the whole imec0 rescue curated sort (301 good + 409 mua, 29 M spikes) and
measures reconstitution with the same `mutual_best_matches` machinery that
defined the ±200/−127 relabelling.

| | before stitch | after stitch |
|---|---:|---:|
| rescue KS-good units | 301 | 267 |
| legacy-good units matched to a rescue good | 101 | **99** |
| of the 127 lost: newly matched to a rescue good | — | **2** (clusters 1, 89) |
| existing legacy matches destroyed by over-merge | — | **4** (425, 451, 489, 526) |
| rescue good units absorbed into another good unit | — | **34** |
| the 127, classified: dispersed / MUA / split / merged | 82 / 29 / 10 / 6 | 81 / 28 / 10 / 8 |

Stitching **loses the pipeline two legacy matches on net** and folds 34 genuine
good units into their neighbours. Of the 82 "dispersed" units — the bulk of the
127 — it moves exactly one. The dispersed structure A2 found is not a clean
2–4-member family: each lost unit's spikes are smeared across 5–15 rescue
clusters, most of them contamination-dominated background units, exactly as at
snippet scale. There is no pairwise chain to walk.

## Consequence for Phase D

Family stitching is **not adopted**. It is a snippet-scale curiosity that helps
two mild-drift arms and actively harms the full-session yield. The motion
fragmentation has to be **prevented, not repaired**: **a non-rigid motion
representation** (Phase D candidate 2), tested the same way — injected-truth
drift penalty, L2L identity continuity, and the full-session reconstitution of
the 127.

`testing/ladder_stitch.py` stays in the tree as a tested, documented negative
result and as the mutual-best-partner family-detection primitive (a trajectory
diagnostic), not as a curation stage.

## Limits

- `apply_stitch` keeps one member's template row for the merged unit; the
  similar-pair and edge-spike guardrails are approximate post-merge until the
  templates are recomputed.
- Evaluated on the C2 diagnostic donors (discovery-cohort reuse), one background
  window, negative-compact polarity only.
- The +0.23 on T06 is against a static baseline that itself failed the sanity
  bar.
