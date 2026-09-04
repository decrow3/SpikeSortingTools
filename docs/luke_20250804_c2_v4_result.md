# C2 v4 result: Luke-scale rigid motion does not cost neuron recovery

**Date:** 2026-09-04
**Status:** first C2 run to survive all controls. Supersedes the void v3 run.
**Runner:** [`testing/luke_rescue_c2_drift_challenge_v4.py`](../testing/luke_rescue_c2_drift_challenge_v4.py)
(`luke-rescue-c2-drift-challenge-v4`, prespec `ff993d2b1b79…`)
**Analysis:** [`testing/luke_c2_v4_analysis.py`](../testing/luke_c2_v4_analysis.py)
(`luke-c2-v4-analysis-v1`)
**Output:** `testing/outputs/luke_rescue_c2_drift_challenge_v4/`
**Run:** 14 donors × 28 cells = **392 cells**, 168 sorts, 02:29–07:51 (5 h 21 m).
**Code:** commit `1ca7458`.

## Headline

| condition | median motion penalty (`rescue`) | donors fragmenting |
|---|---|---|
| ramp 5 µm | **−0.002** | 1/14 |
| ramp 11 µm | **−0.006** | 2/14 |
| ramp 22 µm | −0.175 | 7/14 |
| staircase 40 µm | −0.534 | 13/14 |

Motion penalty is the within-condition, within-config Δ (`moved − static`), on
the 10-donor common primary cohort.

**At the rigid motion Luke imec0 actually experiences — 4–23 µm per 120 s window,
median ~11 µm ([decision 0013](decisions/0013-luke-imec0-has-appreciable-rigid-motion.md))
— imposed rigid motion costs essentially nothing.** The penalty is a threshold
effect, not a gradient: it appears only at 22 µm and is severe at 40 µm.

This converges with the
[within-Luke rigid-motion dose–response](luke_within_rigid_motion_dose_response_result.md),
whose primary endpoint was also null. Two independent designs — one observational
across real windows, one causal with injected ground truth — now agree that
Luke-scale *rigid* motion is not the mechanism behind the neuron loss.

## Rigid correction

The correction effect is the **interaction**, never a single difference, because
`nblocks=1` re-registers a recording and changes clustering with no motion
present:

> `(moved_rigid − static_rigid) − (moved_rescue − static_rescue)`

| condition | median interaction | positive |
|---|---|---|
| ramp 5 µm | +0.001 | 5/10 |
| ramp 11 µm | −0.007 | 3/10 |
| ramp 22 µm | +0.071 | 8/10 |
| staircase 40 µm | **+0.508** | 10/10 |

Rigid correction helps only where there is a penalty to fix. At Luke scale there
is none, so it buys nothing.

**And it is not free.** Two donors fail static qualification under
`rescue_rigid` with no motion present at all: **D01 0.992 → 0.569** and
**D02 0.990 → 0.538**. Qualification is 13/14 under `rescue` and 11/14 under
`rescue_rigid`. Without the stationary `rescue_rigid` control, those donors'
moving-arm gains would have been read as motion recovery; they are partly a
different sort of the same stationary neuron.

## The machinery validated itself

`moved_corrected − static`, both under `rescue` — the exact-inverse registration
reference (**not** a performance ceiling: the exact inverse minimises positional
error, not amplitude or accuracy):

| condition | within 0.01 | median Δ |
|---|---|---|
| staircase 40 µm | **14/14** | 0.0000 |
| ramp 5 µm | 12/14 | 0.0000 |
| ramp 11 µm | 6/14 | −0.0137 |
| ramp 22 µm | **2/14** | −0.1925 |

Exactly what the
[operator calibration](luke_20250804_c2_operator_calibration.md) predicted: the
lattice-commensurate staircase round-trips losslessly, while the ramps degrade
with excursion because they resample twice at fractional offsets. The
calibration's independent measurement therefore reproduces itself in the sorted
result, which is the strongest available check that the forward model is
understood.

## Cohorts

Contrast-specific and per condition. Each condition sorts its own static
baseline — recordings differ by name, so they take separate cache leaves, and KS4
is not bit-deterministic (observed spread ≤ 0.005 across the three ramps'
statics). Qualification is therefore measured per condition, not read off a
representative.

| cohort | n | donors |
|---|---|---|
| common primary (qualified in every condition) | **10** | D03–D06, D08, D09, D11–D14 |
| excluded | 4 | D01, D02 (fail under `rescue_rigid`), D07, D10 |
| all-donor | 14 | reported alongside, never replaced |
| operator-qualified sensitivity (ramp-mean SNR ≥ 3) | 11 / 10 / 9 at 5 / 11 / 22 µm | from the calibration |

Per-arm exclusion is forbidden and not implemented: it would let each magnitude
run on a different, progressively easier cohort. Cohorts are built from static
arms only, so a poor moving arm never removes a donor from its own contrast.

## What this does and does not establish

**Does.** Rigid displacement *can* shatter a cleanly-recovered neuron — the
staircase shows −0.534 with 13/14 donors fragmenting, with no interpolation
artifact anywhere in the recording. The mechanism is real. Rigid correction
recovers it almost completely there (+0.508). So both the injury and the repair
are demonstrable when the displacement is large enough.

**Does not.** That this matters at Luke scale. The 5 and 11 µm arms are flat, and
those bracket Luke's measured range. The 22 µm penalty cannot be cleanly
attributed: the operator attenuates compact donors to ~0.68 mean peak retention
there, so it mixes motion with resampling loss. The staircase is 40 µm — roughly
twice Luke's largest — and discontinuous, so it bounds the mechanism, not the
Luke-scale effect.

## Consequences for the plan

1. **Rigid motion is not the target.** Two independent lines agree it costs
   nothing at Luke scale, and `nblocks=1` buys nothing there while actively
   breaking two donors when stationary. Phase D should not pursue rigid
   correction on the strength of C2.
2. **`legacy_style` is not equivalent to `rescue_rigid`.** On D12's staircase arm
   `legacy_style` recovered the neuron (0.991) where `rescue_rigid` did not
   (0.529), despite both running `nblocks=1`. The only remaining difference is
   the detection thresholds (9/8 vs 12/9), so that recovery is a threshold
   effect, not correction. Any claim about "what motion correction buys" from
   `legacy_style` is confounded.
3. **D10 needs a look.** Static `rescue` 0.475 against `legacy_style` 0.992 — a
   donor the rescue configuration cannot sort even stationary.
4. **The remaining candidates are non-rigid motion, a better estimate, or
   post-sort family stitching** — unchanged by this result, but no longer
   competing with rigid correction.

## Limits

One background window, one probe (imec1), one train shape (6 Hz regular),
amplitude scale 1.0. Ramp arms carry the forward-model confound quantified above.
The staircase admits 687 of 708 events (boundary straddlers removed before
injection); the ramps carry all 708 because no exact subset exists at fractional
offsets. Truth trains therefore differ between the staircase and the ramps by
design — every arm *within* a condition shares one train, verified by contract on
all 56 condition groups.
