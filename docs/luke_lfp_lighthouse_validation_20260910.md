# Lighthouse assessment of the imec0 LFP motion traces

## Technical summary

The highlighted `band_0p5_8` first-spatial-derivative LFP trace is **not
validated for correction**. It agrees with shared depth-aware lighthouse
movement in 930--1030 s, including after predeclared plausibility and lattice
controls, but it produces substantial false motion in the quieter 1150--1200 s
window. That failure to generalize is decision-limiting.

Only 930--1030 s and 1150--1200 s overlap the frozen waveform-only lighthouse
panel. The 5000--5100 s and 6000--6100 s LFP traces remain untested by this
evidence.

## Strongest controlled comparison

The primary robustness cut removes units already flagged before the LFP audit
as implausible/lookalike-prone (`125`, `161`, `353`, `557`, `698`, `705`) and
retains strict events within 5 um of a 40 um template-lattice node. Unit 557 is
removed only from this sensitivity; its family remains represented by pooled
labels 673/675.

| Window | Families | Adjacent 5 s increments | LFP RMSE (um) | Zero-motion RMSE (um) | Skill vs zero | Median family r | Median slope |
|---|---:|---:|---:|---:|---:|---:|---:|
| 930--1030 s | 6 | 53 | 42.35 | 109.17 | 0.612 | 0.955 | 0.814 |
| 1150--1200 s | 4 | 28 | 21.86 | 1.43 | -14.329 | -0.112 | -2.881 |

Thus the early trace captures real shared structure, though with attenuated
amplitude. In 1150--1200 s, the supported lighthouse families are essentially
stationary on this timescale while the LFP trace moves by tens of microns.

The all-strict comparison is less trustworthy because known waveform lookalike
tracks contain hundreds-of-microns jumps. Those rows are preserved in the
outputs; they can make large erroneous LFP excursions look marginally useful.

## Method

- Freeze identities, event gates, family assignments, plausibility flags, and
  lattice thresholds before inspecting the LFP trace.
- Sample each LFP output at the actual lighthouse event time, using only
  `supported=True` and `invalid=False` values.
- Pool the 17 labels into the 15 pre-existing waveform identity families;
  family 11 contains 557/673/675 and receives one vote.
- Within each family, calculate 5 s median depth observations and consecutive
  5 s displacement increments. Increment scoring avoids arbitrary motion-trace
  offsets.
- Calculate residuals as lighthouse minus LFP. Score each family separately,
  then aggregate family mean-square errors with equal family weight.
- Require at least three adjacent supported increments for a scored family and
  compare every arm with predicting zero motion.
- Preserve strict, lower-score, identity-ambiguous, lattice phase, event counts,
  temporal span, LFP support, and per-family residual evidence separately.

No LFP, AP, DREDGE, absolute-depth prior, temporal continuity, lag fit, sign fit,
gain fit, smoothing, or consensus field entered lighthouse identity matching.
This script is a validator, not a motion estimator.

## Support and limitations

In 930--1030 s the frozen table contains 897 strict, 1,741 lower-score, and
2,116 identity-ambiguous observations. In 1150--1200 s it contains 987 strict,
2,379 lower-score, and 2,862 identity-ambiguous observations. Fifteen and ten
families, respectively, have at least one strict observation, but only ten and
six have at least three adjacent supported increments in the all-strict score.

Fully unmatched detections remain preserved in the original chunk/event
artifacts but cannot receive an identity-family residual. Missing bins and
non-scored families are dropout evidence, not zero motion. The lighthouse panel
is provisional waveform evidence rather than certified biological ground truth.

## Recommendation

Keep the 0.5--8 Hz derivative trace as a diagnostic candidate only. The
cheapest next check is to extend the frozen waveform-only panel to one of the
late LFP windows and repeat the same event-time, equal-family comparison. A
quiet-window false-motion gate should be mandatory before any new LFP estimator
is considered for correction.

Reproduce with:

```bash
MPLCONFIGDIR=/tmp/luke-lfp-lighthouse-mpl \
  python -m testing.luke_lfp_lighthouse_validation_v1
```

Outputs are in
`testing/outputs/luke_lfp_lighthouse_validation_v1/`, including the complete
arm and sensitivity tables, per-family metrics, event-time evaluations,
population/family tracks, plots, and the validated report artifact.

The companion peak-raster overlay with the prior DREDGE and MEDiCINe fields is
in `testing/outputs/luke_early_motion_raster_with_lfp_v1/` and is reproduced by
`testing/luke_early_motion_raster_with_lfp_v1.py`.
