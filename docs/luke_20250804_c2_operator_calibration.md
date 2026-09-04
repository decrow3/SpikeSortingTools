# C2 v4 pre-freeze: the injected-motion forward model attenuates compact donors

**Date:** 2026-09-03
**Status:** blocking input to the C2 v4 prespec freeze. Not a drift result.
**Code:** [`testing/luke_c2_operator_calibration.py`](../testing/luke_c2_operator_calibration.py)
(schema `luke-c2-operator-calibration-v1`)
**Output:** `testing/outputs/luke_c2_operator_calibration/`
**Runtime:** ~100 s, no GPU, no sorting.
**Tests:** 16 pass —
`environments/rescue-production/.venv/bin/python -m pytest testing/test_luke_c2_operator_calibration.py testing/test_luke_rescue_c2_drift_challenge.py -q`
(9 calibration + 7 existing C2; interpreter Python 3.12.4, pytest 8.3.5, the
locked uv runtime — not system Python and not a Conda env).

## Why

C2's moving arm is produced by warping the injected voltage through
`InterpolateMotionRecording`; the static arm is never warped. `moving − static`
therefore mixes the sorter's response to temporal motion, the ordinary change in
a footprint sampled from a different electrode position, and numerical error
contributed by the forward model. Only the first two are wanted. The third had
never been measured on the frozen cohort.

## Method

All 14 compact D2b-2 donors (hash-verified), placed on the real imec1 target
strip by C2's own `prepare_template` / `donor_base_channel`, injected noise-free,
then warped by constant offsets on a 0.5 µm grid to 25 µm plus 40 and 80 µm.
Because Luke's rigid speed is 0.2–0.8 µm/s, displacement changes by < 0.002 µm
across a 2.03 ms template: **every spike in the moving arm is a constant-offset
resampling**, so the offset sweep characterises the per-spike cost exactly and a
ramp's expected cost is the sweep averaged over the offsets it visits.

Metric discipline matters. After a forward warp the footprint has *moved*, so
element-wise comparison against the unmoved original measures displacement, not
error. Only `peak_retention` (translation-invariant) and the round-trip measures
(taken after the footprint is returned to its original position) are valid
fidelity measures. `cosine` and `rel_rms` on the forward warp are retained as
displacement diagnostics only.

## The implementation is validated; the forward model is not

The strip has a 20 µm row pitch with a two-column x-stagger, so a 40 µm shift maps
every site exactly onto another site and the correct warped field is a pure
channel roll. The operator reproduces it:

| check | result |
|---|---|
| max relative error vs exact channel roll at 40 / 80 µm | **1.16e-06** |
| min peak retention at 40 / 80 µm | **0.999999** |
| peak retention, cosine, rel RMS at 0 µm | 1.000 / 1.000 / 7e-07 |
| µm ↔ channel conversion product | exactly 1.0 |

This validates the geometry and sign implementation. **It does not establish that
the operator's fractional-offset behaviour reproduces the unknown continuous
biological voltage field.** The loss measured below is therefore
*phase-dependent attenuation of the discrete injection/interpolation model* — an
artifact of how motion is simulated, whose relation to what a real neuron
displaced by that amount would produce is untested.

## Result: the attenuation is large at exactly Luke's magnitudes

Donor-mean peak retention against constant offset — smooth decay to a floor at
the 20 µm anti-phase point of the 40 µm super-lattice, then recovery to 1.000:

| offset µm | 0 | 5 | 10 | 11 | 15 | 20 | 22 | 25 | 40 |
|---|---|---|---|---|---|---|---|---|---|
| peak retention | 1.00 | 0.82 | 0.67 | 0.64 | 0.55 | 0.51 | 0.52 | 0.55 | **1.00** |

Averaged over the offsets a linear ramp visits (the moving-arm expectation):

| ramp excursion | mean peak retention | worst donor | after exact inverse | round-trip rel RMS |
|---|---|---|---|---|
| 5 µm | 0.912 | 0.725 (D02) | 0.847 | 0.162 |
| 11 µm | 0.812 | 0.403 (D04) | 0.713 | 0.302 |
| 22 µm | 0.680 | 0.287 (D04) | 0.586 | 0.433 |

## The exact inverse is a registration reference, not a ceiling

The exact inverse removes the displacement but adds a second interpolation, so it
returns a smaller waveform than leaving the motion uncorrected — on 14/14 donors
at every magnitude (5 µm: 0.823 → 0.711; 11 µm: 0.638 → 0.513; 22 µm: 0.517 →
0.439).

This must **not** be read as a performance ceiling. The exact inverse minimises
*positional* error; it does not maximise amplitude or sorting accuracy. A partial
or estimated correction that registers less displacement can retain more
amplitude and could plausibly sort better. What the numbers do establish is that
the correction arms carry a forward-model amplitude handicap that scales with
excursion and does not exist in a real recording — so a null or negative result
for `rescue_rigid` at Luke scale is not by itself evidence against rigid
correction.

## SNR: three rules, three different answers

Background noise on the quiet imec1 strip is 27.8 µV MAD; static donor SNR runs
2.6–10.6. The count of donors below SNR 3 depends entirely on which rule is used,
and the earlier draft of this document conflated them:

| ramp | below 3 statically | ramp-**min** < 3 | ramp-**mean** < 3 | cross above→below (min) | cross above→below (mean) |
|---|---|---|---|---|---|
| 5 µm | 3 | 4 | 3 | D10 | — |
| 11 µm | 3 | 5 | 4 | D04, D10 | D10 |
| 22 µm | 3 | 11 | 5 | 8 donors | D04, D10 |

At 11 µm, three of the five donors below SNR 3 (D03 2.67, D08 2.62, D11 2.82)
were **already below 3 statically** — the ramp did not push them there. Only D04
(4.38 → min 1.78, mean 3.11) and D10 (3.43 → min 2.05, mean 2.74) cross. Under a
ramp-**mean** rule D04 stays eligible at 3.107.

## Cohort policy: do not exclude per-arm

A per-arm operator-qualification rule would make each magnitude run on a
different and progressively easier cohort — at 22 µm a ramp-min rule drops **11
of 14** donors, so the 22 µm result would describe only the three strongest. v4
must instead report:

1. a **common statically qualified primary cohort**, identical across magnitudes;
2. **all-donor** results;
3. a **separately labelled operator-qualified sensitivity analysis**, with the
   rule (min vs mean) and threshold stated explicitly.

## Freeze the operator

Alternatives were measured; the spread is small and does not change the
conclusion (mean round-trip rel RMS: kriging σ=10 0.447, σ=40 0.454, σ=20 0.466,
idw σ=20 0.505; `border_mode` made no difference). Freeze the v3 implicit
default. Optimising interpolation parameters does not address the underlying
problem.

## The 40 µm staircase is a machinery positive control

A piecewise-constant trajectory dwelling only at multiples of 40 µm has exact
footprints at every spike (1e-6), real displacement, and real template mismatch,
with the forward-model attenuation essentially absent. Its question is narrow:

> Can this experiment and this sorter demonstrate a correction benefit at all,
> when interpolation error is essentially absent?

It is twice the largest proposed Luke displacement and is discontinuous rather
than Luke-like, so **it cannot establish that correction helps at 5–22 µm** and
is not an anchor for the Luke-calibrated arms. Now implemented and verified —
see [`luke_20250804_c2_staircase_positive_control.md`](luke_20250804_c2_staircase_positive_control.md).
Implementation safeguards:

- long plateaus at exactly 0 and 40 µm;
- no injected truth spikes during interpolated transition bins;
- explicit verification of boundary behaviour on the full recording — the
  exactness check above used a centred, noise-free donor field;
- results reported separately from the Luke-calibrated dose–response.

## Revised v4 design

- static condition;
- Luke-calibrated 5 / 11 / 22 µm ramps, explicitly labelled **forward-model
  confounded**;
- exact-inverse round-trip arms as registration references, not ceilings;
- a 0↔40 µm commensurate staircase positive control, reported separately;
- `rescue`, `rescue_rigid`, and contextual `legacy_style`, with contrast-specific
  static qualification;
- common-cohort primary analyses plus operator-qualified sensitivity analyses;
- a small sorting smoke matrix before the full run.

## Proposed prespec block

```json
"motion_operator": {
  "spatial_interpolation_method": "kriging",
  "sigma_um": 20.0,
  "border_mode": "force_extrapolate",
  "bin_s": 0.5,
  "sign_convention": "forward warp sign=-1, exact inverse sign=+1",
  "validation": {
    "lattice_commensurate_offsets_um": [40.0, 80.0],
    "max_rel_err_vs_channel_roll": 1e-05,
    "zero_offset_max_rel_rms": 1e-05
  },
  "measured_forward_model_attenuation": {
    "calibration": "luke-c2-operator-calibration-v1",
    "ramp_mean_peak_retention": {"5": 0.912, "11": 0.812, "22": 0.680},
    "exact_inverse_round_trip_rel_rms": {"5": 0.162, "11": 0.302, "22": 0.433},
    "exact_inverse_amplitude_cost": {"5": -0.064, "11": -0.098, "22": -0.094},
    "note": "a registration reference, not a performance ceiling"
  },
  "cohort_policy": {
    "primary": "common statically qualified cohort, identical across magnitudes",
    "also_report": "all-donor results",
    "sensitivity": "operator-qualified subset, rule and threshold stated",
    "forbidden": "per-arm exclusion that varies the cohort across magnitudes",
    "static_noise_uv_mad": 27.798
  }
}
```

## Limits

Noise-free single-spike fields, so this bounds the forward model's contribution,
not the sorter's response to it. Retained SNR uses a MAD noise estimate on the
unwarped strip. The staircase arm is now implemented and verified on the real
background, but not yet sorted or scored. Whether the measured attenuation resembles real neuronal displacement is
untested and would need a biophysical or dense-template comparison.
