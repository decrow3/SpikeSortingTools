# Phase C2: motion shatters a clean neuron, and KS4 rigid drift correction does not fix it

> **RETRACTED PENDING RERUN — 2026-09-03.** The moving injection shifted
> contiguous channel indices, which is not physical depth motion on Luke's
> four-column geometry, while the oracle correction applied a continuous
> y-displacement. Ground-truth event matching was also non-exclusive. The
> numerical drift penalties below are retained as history but must not be cited
> as evidence that motion causes the reported loss until the geometry-aware,
> exclusively scored experiment is rerun.

> **DONOR CORRECTION — 2026-09-03.** C2 v2 is retired without rerun. Although
> its operator was fixed, it still specified T01/T04/T06, which D2b-2 showed are
> common-mode plateaus or noise rather than compact neuron footprints. C2 v3
> introduced all 14 hash-frozen compact donors but its run was void because of
> the scorer defect in decision 0014. C2 v4 retains the cohort and qualifies each under
> both rescue and `legacy_style`. See [decision 0012](decisions/0012-c2-uses-compact-donor-cohort.md).

**Date:** 2026-09-02
**Advances:** Phase C (steps 1–2) and Phase C2 of
[`pipeline_improvement_plan.md`](pipeline_improvement_plan.md)
**Prespecified** in `PRESPEC` inside `testing/luke_rescue_c2_drift_challenge.py`
(written to `.../prespec.json`; run-once, refuses a changed prespec).
**Status: diagnostic.** Reuses the discovery-cohort donor templates from
`luke_injected_ground_truth_pilot` (real reviewed imec1 neural events, each
qualified against an independent event). Per that scaffold's contract the
numbers are diagnostic — they set Phase D's *direction*, they do not promote
anything.

## What was done

A **within-subject** contrast, wiring the sealed injection scaffold to the L1
runner for the first time. The same donor waveform and the same 6 Hz spike train
are injected into a **quiet imec1 depth strip** (start 4080 s, 112 channels)
twice:

- **static** — the neuron held at a fixed channel;
- **moving** — the identical neuron translated along a known trajectory
  (rigid ramp 15 µm or 40 µm over 120 s; or a 20 µm / 40 s oscillation).

Then `l1_run` sorts each injected snippet with the frozen rescue KS4 config and
`score_sort(truth=…)` scores it against the known train. The **drift penalty** is
the within-subject Δ (moving − static): the change caused *solely* by motion.

Confound control (plan C2): the static arm is drawn from a quiet window, so the
background's own tissue motion is minimal; the trajectory is imposed on top and
reported in µm and channels.

Historical command (do not rerun): `python testing/luke_rescue_c2_drift_challenge.py --templates T01 T04 T06`.

Two sorter configs were run on the *identical* injected snippets:

- **rescue** — the frozen no-motion config (`nblocks=0`, `Th 12/9`).
- **legacy_style** — KS4 with rigid internal drift correction (`nblocks=1`) and
  the legacy detection thresholds (`Th 9/8`), `ladder_sorter.LEGACY_STYLE`.

## Historical benchmark sanity — not sufficient to validate the experiment

| donor | SNR | rescue static acc | legacy_style static acc |
|---|---:|---:|---:|
| T01 | 11.0 | **0.94** | **0.98** |
| T04 | 6.1 | **0.95** | **0.95** |
| T06 | 4.6 | 0.50 | 0.65 |

Both configs recovered the easy high-SNR static injections at ≥ 0.9. This
does **not** make the old moving-arm benchmark sound because the forward motion
operator was invalid. T06 (lowest SNR) was not cleanly recovered
even static under either config, so its drift penalties are measured against a
broken baseline and are **not interpretable** — excluded below.

## Retracted historical result — the drift penalty, both arms

For the two donors with a clean static baseline (Δ accuracy = moving − static):

| donor | trajectory | **rescue Δacc** | **legacy_style Δacc** | rescue Δidentities | legacy Δidentities |
|---|---|---:|---:|---:|---:|
| T01 | rigid 15 µm | −0.35 | −0.32 | +1 | +1 |
| T01 | rigid 40 µm | **−0.54** | **−0.81** | 0 | +5 |
| T01 | osc 20 µm / 40 s | −0.70 | −0.68 | +3 | +4 |
| T04 | rigid 15 µm | **−0.30** | **−0.58** | +3 | +3 |
| T04 | rigid 40 µm | −0.53 | −0.31 | +15 | +12 |
| T04 | osc 20 µm / 40 s | −0.38 | −0.49 | +13 | +10 |

### Original interpretation — withdrawn pending rerun

1. **Motion alone costs 30–80 accuracy points — under *both* configs.** The
   waveform and train are identical between arms; the only difference is a
   translation the size of Luke's real session motion. Representing a moving
   footprint with static templates has a large, direct cost, and it is not
   specific to the rescue config.

2. **KS4's rigid internal drift correction does not recover the penalty.** On
   the six clean conditions `legacy_style` is worse than `rescue` on three
   (T01-40 µm: −0.81 vs −0.54; T04-15 µm: −0.58 vs −0.30; T04-osc: −0.49 vs
   −0.38), comparable on two, and better on only one (T04-40 µm). Turning
   `nblocks` back on is **not the fix.**

3. **Fast motion defeats rigid correction and generates false positives.** The
   20 µm / 40 s oscillation (three cycles in the window) is as damaging as a
   40 µm ramp under both configs; `legacy_style` piles up false positives doing
   it (T06 osc: FP 2041; T01 osc: FP 874 vs rescue 473). A slow rigid estimate
   cannot track sub-minute motion — consistent with A2's "rapid flicker".

4. **The loss is mostly missed spikes.** T01 rigid-40 µm: false negatives
   24 → 314 (rescue) and 11 → 503 (legacy_style). The moving footprint drifts
   off its template and falls below detection.

5. **The neuron shatters across templates.** T04 rigid-40 µm: **+15** (rescue) /
   **+12** (legacy_style) output units each capturing > 5 % of the injected
   train. The Phase A2 fragmentation signature — identity proliferation, rapid
   ownership change — produced **causally** by imposed motion under both configs.

6. **Even 1.5 channels of drift costs −0.30 to −0.58.** Motion does not have to
   be large to matter.

## Historical consequence for Phase D — withdrawn

Phase A2 could not separate *non-rigid / fast motion* from *KS4 template
competition*. The old C2 run does not resolve that ambiguity. It cannot show
that motion is a sufficient cause because its imposed motion was not physically
consistent with the correction operator.

The old C2 run also cannot rule out `nblocks=1` or establish a preferred Phase D
target. The original candidate list was:

- **non-rigid** motion handling (the rigid arm's failure on the oscillation is
  the direct evidence);
- a **better motion estimate** — the DREDGE rigid sidecar is QC-unqualified,
  and KS4's own rigid estimate clearly is not tracking this;
- **post-sort family stitching** of the fragments — which A2 independently
  favours (the fragments are refractory-clean and mergeable) and which repairs
  both slow drift and fast flicker.

Curation-threshold tuning stays low priority: A2 found ~0 % coexisting
fragments, and C2 shows the lower-threshold `legacy_style` config fragments
*more* at baseline (T04 static: 12 identities vs rescue's 7), not less.

## Still to add

- **non-rigid trajectories** (depth-varying field), **both polarities** (imec1
  is positive-dominant; these donors are negative-compact);
- truncation-estimator calibration against the known train;
- a second background window and a second donor train shape.

## Limits

- Diagnostic: discovery-cohort template reuse; not a promotion result.
- One background window, one probe (imec1), one train shape (regular 6 Hz),
  amplitude scale 1.0, two donors with a clean baseline (T01, T04).
- `legacy_style` differs from the rescue config in `nblocks`, `Th_universal`
  and `Th_learned` together — it is *KS4 with rigid drift correction*, not a
  faithful replay of the historical legacy pipeline (different preprocessing).
  It isolates "what does KS4's own rigid drift correction buy on this input",
  which is the relevant question for Phase D.
- The static arms score 0.94–0.98, not 1.0. The penalty is a within-subject Δ,
  so this is controlled.
- Trajectory expressed in channel-index steps (exact for injection), converted
  to µm via the strip geometry (~0.10 ch/µm); the µm figures are approximate.
- T06 excluded — static baseline below the sanity threshold under both configs.
