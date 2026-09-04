# C2 v4 machinery positive control: the discrete lattice-commensurate staircase

**Date:** 2026-09-04
**Status:** implemented and verified interpolation-free over the full 120 s.
**Not** a Luke-scale result; reported separately from the 5/11/22 µm dose–response.
**Code:** [`testing/luke_c2_staircase_control.py`](../testing/luke_c2_staircase_control.py)
(schema `luke-c2-staircase-control-v1`)
**Output:** `testing/outputs/luke_c2_staircase_control/summary.json`
**Companion:** [`luke_20250804_c2_operator_calibration.md`](luke_20250804_c2_operator_calibration.md)
**Tests:** `environments/rescue-production/.venv/bin/python -m pytest testing/test_luke_c2_staircase_control.py -q` → 16 pass.

## What it is for, and what it cannot do

The operator calibration showed the injected-motion forward model attenuates
compact donors by 20–50 % at Luke-scale displacements. The one family it
reproduces exactly is the lattice-commensurate one: 40 µm is two 20 µm rows, and
with two sites per row that maps every site onto another site four channel
indices away, so the correct answer is a pure channel roll.

A trajectory that **only ever sits at a multiple of 40 µm** therefore delivers
real displacement and real template mismatch with no interpolation anywhere in
the recording. Its question is narrow:

> Can this experiment and this sorter demonstrate a correction benefit at all,
> when interpolation error is absent?

It is **not** an anchor for the Luke-calibrated arms. 40 µm is roughly twice the
largest proposed Luke displacement and the motion is discontinuous rather than
Luke-like, so a positive result here **cannot** establish that correction helps
at 5–22 µm, and a negative result would indict the machinery rather than
correction.

## Two design faults this arm had, and how they were fixed

The first implementation used 4 s ramped transitions and warped exactly the
strip the sorter sees. Both leaked interpolation into the sorter input:

| fault | why filtering truth did not fix it | fix |
|---|---|---|
| 24 fractional-offset bins (12 s of the recording) at 4–29 % relative error | those degraded waveforms and background still reach KS4's detection and template formation, scored or not | **hard, bin-aligned steps**: `transition_s = 0`, so no fractional bin exists |
| 4 edge channels with no source site, at 0.17–0.44 relative error | they are in the sorter input wherever donors are placed | **warp wide, crop after**: warp a 128-channel strip, crop the central 112 |

Excluded spikes are now removed **before injection**, not merely before scoring,
and the same filtered train goes into every arm.

## The trajectory

Four 30 s plateaus alternating 0 → 40 → 0 → 40 µm. Steps at 30, 60 and 90 s land
exactly on interpolation-bin edges (`bin_s = 0.5 s`), so every bin is a pure
plateau bin.

Displacement is specified in **µm directly**. `sampled_displacement` and
`warp_array_with_known_motion` gained `trajectory_units="um"` (default unchanged
at `"channels"`, so every pre-v4 caller is untouched), because routing 40 µm
through the channel round trip is only exact to floating point.

## The three arms

All cropped identically from the same wide strip, so they have byte-identical
spatial support and channel geometry, and all carry the same admitted train:

| arm | construction |
|---|---|
| `static` | unwarped crop of the wide strip |
| `staircase` | wide strip warped by the staircase, then cropped |
| `staircase_corrected` | the staircase arm warped by the exact inverse, then cropped |

## Verification: real background, 112 channels, full 120 s

| check | result |
|---|---|
| fractional-offset bins | **0** of 240 (levels present: exactly 0.0 and 40.0 µm) |
| bins exact vs the expected channel shift | **240 / 240**, max rel err **3.16e-07** |
| first / last bin | 3.12e-07 / 3.12e-07 — time boundaries clean |
| **sorter channels exact** | **112 / 112**, max rel err **8.39e-07** |
| channel shifts seen | 0 and 4, as the geometry requires |
| exact inverse restores the static arm | **yes**, rel err **6.26e-07** |
| spatial support identical across arms | yes |
| truth admitted | **687 / 708 (97.0 %)**, balanced 344 at 0 µm / 343 at 40 µm |

Admission rose from 615/708 (86.9 %) under the ramped design to 687/708: with
hard steps only spikes whose template window straddles one of the three steps,
plus a one-bin guard, need removing.

That the exact inverse returns the static arm to 6.26e-07 is the property that
makes this a positive control: at commensurate displacements the round trip is
lossless, unlike the fractional case where the calibration measured 30–50 % of
peak amplitude lost. Any sorting difference between `staircase_corrected` and
`static` is therefore sorter behaviour, not forward-model damage.

## Two bugs the verification found

1. **In the verification.** It sliced frames by `round(edge × fs)` while the
   operator assigns them by `searchsorted(edges, times, side="right") − 1`,
   which differs by one sample wherever an edge does not land on a sample —
   contaminating exactly the bins adjacent to a displacement change (3.7e-04
   instead of 3e-07). `frame_bin_assignment` now lives in
   [`ladder_motion.py`](../testing/ladder_motion.py) alongside the operator whose
   semantics it encodes; anything reasoning about which frames sit in which
   motion bin must use it.
2. **In the design**, twice — the transitions and the edge channels above.

## Prespec block

```json
"staircase_positive_control": {
  "schema": "luke-c2-staircase-control-v1",
  "role": "machinery positive control, reported separately",
  "cannot_answer": "whether correction helps at 5-22 um",
  "levels_um": [0.0, 40.0],
  "plateau_s": 30.0,
  "transition_s": 0.0,
  "n_plateaus": 4,
  "bin_s": 0.5,
  "trajectory_units": "um",
  "spatial_margin_channels": 8,
  "arms": ["static", "staircase", "staircase_corrected"],
  "truth_admission": {
    "rule": "every bin the template window touches, widened by guard_bins, must carry the same commensurate displacement",
    "applied": "before injection, identically in every arm",
    "template_pre_samples": 30,
    "template_post_samples": 30,
    "guard_bins": 1,
    "measured_fraction_admitted": 0.9703
  },
  "verified": {
    "fractional_bins": 0,
    "bins_exact": "240/240",
    "channels_exact": "112/112",
    "max_channel_rel_err": 8.39e-07,
    "corrected_vs_static_rel_err": 6.26e-07,
    "on": "real imec1 quiet strip, 128 wide -> 112 cropped, 120 s, real noise"
  }
}
```

## The truth contract (scorer threading)

`ladder_score.build_truth_contract` binds the admitted train to its provenance
and its spatial support, and `score_sort(..., truth_contract=...)` validates it
and fails closed. The contract records the exact admitted array's hash,
`n_expected`, per-unit hashes, the admission schema and every parameter that
produced the filter, the counts by displacement level, the cropped channel-id
and geometry hashes, and an attestation that filtering happened **before**
injection — `build_truth_contract` refuses to issue one otherwise.

The scorer scores exactly the train it is handed; it never reconstructs the
708-event regular train from a prespec, and a train that differs by a single
sample fails the hash. `assert_paired_truth` requires every paired arm and
sorter config to share one truth hash, one `n_expected`, one channel-id hash and
one geometry hash, so a within-subject Δ cannot be taken across different
denominators or different channels.

Threaded through `l1_run(..., truth_contract=...)`. Regression tests in
`testing/test_luke_c2_truth_contract.py` (12) cover: only admitted events enter
recall and miss counts; all three arms share one denominator; and truth-hash,
sample-shift, channel-id, geometry, denominator, miscounted-admission and
filtered-after-injection mismatches each fail closed.

## Still to do before this arm can be scored

- **Sort it.** Everything above verifies the recording handed to KS4, not KS4's
  response; that is the smoke matrix's job
  (`testing/luke_c2_staircase_smoke.py`).

## Limits

Verified on background voltage, so it establishes that the *recording* handed to
the sorter is exact. That an injected donor is likewise exact follows from the
calibration's 40 µm result (1.16e-06 on donor fields) but has not been
re-measured inside the staircase. The 8-channel margin covers a 40 µm excursion
in both warp directions; an 80 µm arm would need 16.
