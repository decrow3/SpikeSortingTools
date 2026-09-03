# Checkpoint C panel baseline: rescue wins where there is motion, legacy wins on pure noise

> **RETRACTED PENDING RERUN — 2026-09-03.** Injected-recording cache identity did
> not include voltage content, and the cached rescue/legacy sorts currently
> predate the corresponding materialized recordings. Ground-truth matching was
> also non-exclusive. The direction and magnitudes below are unauditable and
> must not be used as Checkpoint C evidence. Checkpoint C returns to **pending**
> until both configurations are rerun with content-bound cache receipts and the
> corrected scorer.

**Date:** 2026-09-03
**Advances:** Checkpoint C of [`pipeline_improvement_plan.md`](pipeline_improvement_plan.md) §5
**Module:** `testing/luke_ladder_checkpoint_c.py` (+ `test_luke_ladder_checkpoint_c.py`)
**Output:** `testing/outputs/luke_ladder_checkpoint_c/`

## What this is

The first head-to-head on the **primary promotion metric** — injected hybrid
ground truth on the frozen development panel. Three D2b-2 compact donors
(73 / 149 / 274 µV, non-overlapping footprints, static) are injected into each of
the 8 development snippets and both pipelines (`RESCUE`, `LEGACY_STYLE` =
KS4 + `nblocks=1` rigid correction + legacy thresholds) are scored against the
injected trains.

Checkpoint C's two halves: the drift-penalty half came from C2; this is the
panel half. **Sanity condition met** — the highest-SNR donor (D02) is recovered
static at accuracy ≥ 0.94 on every snippet the rescue pipeline sorts.

## Retracted historical result — accuracy on the best matching cluster, rescue − legacy

| snippet | regime | D08 (73 µV) | D06 (149 µV) | D02 (274 µV) |
|---|---|---:|---:|---:|
| noise_plus_motion_deep | noise + motion | **+0.84** | **+0.61** | **+0.64** |
| noise_plus_motion_shallow | noise + motion | −0.01 | **+0.57** | **+0.56** |
| rapid_motion_mid | motion | −0.16 | +0.11 | **+0.34** |
| quiet_deep | quiet | 0.00 | 0.00 | 0.00 |
| quiet_shallow | quiet | −0.03 | −0.01 | −0.05 |
| support_dropout_mid | dropout | −0.33 | 0.00 | 0.00 |
| sustained_noise_shallow | noise | **−0.58** | **−0.19** | +0.01 |
| sustained_noise_deep | noise | **−0.38** | **−0.61** | +0.01 |

Per-donor median accuracy across the panel:

| donor | rescue | legacy_style |
|---|---:|---:|
| D02 (274 µV) | 0.985 | 0.980 |
| D06 (149 µV) | 0.98 | 0.99 |
| D08 (73 µV) | **0.02** | **0.26** |

## Original reading — withdrawn pending rerun

1. **Where motion is present, rescue wins — decisively when noise and motion
   combine.** On the two noise+motion snippets, `legacy_style` drops D06/D02
   from 0.97–0.99 (rescue) to 0.33–0.43: the `nblocks=1` rigid correction is
   *actively destructive* in that regime. This is the C2 finding, now on the
   panel and on the primary metric.
2. **Where there is noise but no motion, legacy wins** (sustained_noise:
   −0.2 to −0.6). The rigid correction / legacy detection thresholds help on
   pure noise.
3. **Quiet: a dead heat.**
4. **The 73 µV donor is below both pipelines' reliable floor.** Neither
   recovers it (rescue median 0.02, legacy 0.26). This is the low-SNR detection
   limit, not a pipeline difference — and it is where panel yield is actually
   lost.

## Current consequence for the plan

- **Checkpoint C is pending.** Neither its motion half nor its panel half is
  currently valid promotion evidence.
- No rescue-versus-legacy direction is established by this run.
- The low-SNR floor and its relationship to motion or detection remain
  hypotheses to retest, not conclusions.

## Limits

- `score_sort`'s `headline_units_recovered` (accuracy ≥ 0.8 **and**
  `n_output_units_capturing ≤ 1`) is unreliable for a multi-unit injection into
  dense real background: the ±0.5 ms coincidence test is non-exclusive, so many
  real output clusters incidentally "capture" > 5 % of an injected train and
  trip the split gate even when one cluster cleanly owns 95 % of it. Accuracy on
  the best cluster is the robust readout and is what is reported above. The
  headline count needs an exclusive-assignment fix before it is trustworthy
  outside the C2 single-injection case.
- 3 donors, one static amplitude each, one injection position per snippet, one
  background realisation. Diagnostic until the held-out panel is opened.
- `LEGACY_STYLE` is KS4-with-rigid-correction, not the historical legacy
  pipeline byte-for-byte (same `ops.npy` diffs, same conditioned input).
