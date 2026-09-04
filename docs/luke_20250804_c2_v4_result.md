# C2 v4 result: the mechanism is real at 40 µm; the Luke-scale arms are uninformative

> **CORRECTED 2026-09-04.** The first version of this document was headed
> "Luke-scale rigid motion does not cost neuron recovery" and concluded that
> rigid motion is not Phase D's target. **Both claims are withdrawn.** The flat
> 5 and 11 µm arms do not show robustness to Luke-scale motion; they show that
> at fractional offsets the forward model degenerates into *attenuation* rather
> than *displacement*, which is the perturbation KS4 is least sensitive to. See
> "Why the Luke-scale arms cannot answer the question". The staircase result and
> the stationary-control finding are unaffected.

**Date:** 2026-09-04
**Status:** first C2 run to survive all controls. Supersedes the void v3 run.
**Runner:** [`testing/luke_rescue_c2_drift_challenge_v4.py`](../testing/luke_rescue_c2_drift_challenge_v4.py)
(`luke-rescue-c2-drift-challenge-v4`, prespec `ff993d2b1b79…`)
**Analysis:** [`testing/luke_c2_v4_analysis.py`](../testing/luke_c2_v4_analysis.py)
(`luke-c2-v4-analysis-v1`)
**Truncation diagnostic:** [`luke_20250804_c2_v4_truncation_diagnostic.md`](luke_20250804_c2_v4_truncation_diagnostic.md)
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

A large penalty appears at 22 µm and above. **The 5 and 11 µm arms — which
bracket Luke's measured range — are flat, but that is not evidence of
robustness**, for the reason set out in the next section.

It does **not** converge with the
[within-Luke rigid-motion dose–response](luke_within_rigid_motion_dose_response_result.md)
in the way first claimed here. That study's *primary* endpoint (E3, qualified
units/mm) was null, but 4 of its 5 endpoints degrade with motion — E6 waveform
stability ρ = −0.66, E7 qualified firing rate ρ = −0.59, and **E8 fragmentation
ρ = +0.42**. Fragmentation is precisely what C2 measures, so the observational
line points *toward* a Luke-scale motion effect on within-unit quality, not away
from one.

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

## Why the Luke-scale arms cannot answer the question

The forward model produces a qualitatively different perturbation at
lattice-commensurate and at fractional offsets:

| offset | peak retention | in-place cosine | what it did |
|---|---:|---:|---|
| 5 µm | 0.823 | 0.985 | dimmed in place |
| 11 µm | **0.638** | **0.897** | dimmed in place |
| 22 µm | 0.517 | 0.580 | mixed |
| **40 µm (exact)** | **1.000** | **0.193** | **translated** |

At 40 µm, where the operator is validated exact to 1.1e-6, displacement behaves
as displacement: amplitude is preserved and the footprint moves off its original
sites. At 11 µm the simulation does the opposite — it destroys 36 % of the
amplitude while barely changing the in-place shape. That is the signature of a
spatial low-pass filter, not of translation, and it follows from the geometry:
the footprint is compact relative to a 20 µm row pitch, so reconstructing it at a
half-row offset means interpolating across sites that do not sample it.

This matters because the two perturbations are not interchangeable for a sorter.
KS4 tolerates dimming — the spike is still detected and still matches its
template, merely smaller — but fragments on positional change, which is what
creates a new template. **At Luke scale the forward model supplies the
perturbation KS4 is least sensitive to and withholds the one it is most
sensitive to.** A flat penalty is the expected outcome whether or not real
Luke-scale motion is harmful.

The bias also runs in both directions at once, which is why it cannot be signed
away: attenuation should make recovery *harder* (conservative), while
loss of positional change makes it *easier* (anti-conservative). The
anti-conservative term acts on the mechanism C2 exists to measure.

## What this does and does not establish

**Does.** Rigid displacement *can* shatter a cleanly-recovered neuron — the
staircase shows −0.534 with 13/14 donors fragmenting, with no interpolation
artifact anywhere in the recording, and rigid correction recovers it almost
completely (+0.508). Both the injury and the repair are demonstrable when the
displacement is real. It also establishes that `nblocks=1` is not free: it breaks
two donors outright with no motion present.

**Does not.** Anything about Luke scale, in either direction. The 5 and 11 µm
arms are uninformative for the reason above; the 22 µm arm mixes motion with
resampling loss (0.517 peak retention); and the staircase is ~2× Luke's largest
displacement and discontinuous. C2 v4 gives **no evidence that Luke-scale rigid
motion is harmless, and none that it is harmful.**

The phase-specific amplitude-truncation diagnostic independently sharpens the
staircase mechanism. In the 13/14 uncorrected-rescue donors that split, the best
single identity misses a median 50.51% of the full train, while within-position
truth loss is 1.02% and estimated amplitude truncation is 0.62%. Thus the 40 µm
failure is predominantly positional identity fragmentation, not a uniform loss
of low-amplitude spikes. This does not repair the uninformative Luke-scale ramp
arms.

## What would answer it

The forward model must move a donor without dimming it — i.e. generate the
displaced footprint from a *model* of the donor's spatial field (a monopole or
dipole fit, or a dense biophysical template) evaluated at shifted positions,
rather than by resampling an under-sampled recorded field. This is the
dense-template comparison already listed as an open limit in
[the operator calibration](luke_20250804_c2_operator_calibration.md). The
background can continue to be warped by the existing operator, since its errors
do not touch the injected unit's score — only the motion estimate.

Until then the honest position is that C2 v4 has validated the machinery and
established the mechanism at large displacement, and the Luke-scale question is
still open.

## Consequences for the plan

1. **Rigid motion is not ruled out as a target, and C2 v4 does not settle it.**
   The earlier version of this document concluded the opposite; that conclusion
   is withdrawn. What stands is narrower: `nblocks=1` buys nothing in the arms
   where the forward model is trustworthy at Luke scale — but those arms cannot
   detect a Luke-scale effect in the first place. Separately, `nblocks=1`
   actively breaks two donors when stationary, which is a real cost independent
   of any benefit.
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
