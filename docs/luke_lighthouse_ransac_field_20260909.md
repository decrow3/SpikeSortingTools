# Lighthouse RANSAC depth/time field pilot

> **Disposition:** negative control only. Lighthouse cells are now reserved for
> independent verification; neither this prototype nor a refined lighthouse-
> derived field should enter motion estimation without a new explicit decision.

## Result

A continuous depth-by-time field can be constructed mechanically from the 16
deduplicated waveform-traced lighthouse families, but the cheapest affine RANSAC
version fails held-out validation and should not be used as a motion estimate.

Each 5-second bin gives every identity family one observation at its reference
depth. For bins with at least six families, RANSAC fits

`displacement(z,t) = intercept(t) + slope(t) * (z - 2000 um) / 1000 um`.

The independently fitted coefficients are linearly interpolated in time. Three
of 20 bins lack six families and are interpolation-only.

![Unvalidated RANSAC field](../testing/outputs/luke_lighthouse_ransac_field_v1/01_ransac_motion_field.png)

## Why it fails

At the predeclared 25-um RANSAC residual threshold, leave-one-family-out median
absolute error is 73.4 um versus 61.7 um for the much simpler rigid median. Mean
absolute error is 133.0 versus 100.5 um, and RANSAC wins only 45.1% of held-out
predictions. Thresholds of 15, 25, and 40 um all lose to the rigid baseline.

Median RANSAC inlier support is only 50%. At 1007.5 s, six available families
include an implausible +1720-um observation from unit 161. RANSAC selects a
spatially aligned four-family subset and infers a slope of -587 um displacement
per millimetre of probe, producing values above 2000 um when extrapolated. This
is a structured-outlier failure: RANSAC can prefer an accidental line, not just
reject a bad point.

## Interpretation

The available lighthouse evidence supports a robust shared time trace more than
it supports depth dependence. A constant-in-depth median field is saved as a
descriptive baseline, but it is not validated motion ground truth because the
same template-lattice and identity caveats remain.

Do not repair this pilot by silently clipping displacement or slope. A bounded
field would look plausible because of the imposed bound, not because the current
data identify the spatial gradient.

## Cheapest next step

Use the 16-family median as the provisional rigid trace and validate it on
near-lattice-node events. Revisit a nonrigid field only after more simultaneous
lighthouse support is available within multiple depth bands. At that point a
temporally coupled, slope-bounded robust model with family-level cross-validation
is preferable to independent per-bin RANSAC.

## Outputs

- `testing/luke_lighthouse_ransac_field_v1.py`
- `testing/outputs/luke_lighthouse_ransac_field_v1/time_bin_ransac_fits.csv`
- `testing/outputs/luke_lighthouse_ransac_field_v1/leave_one_family_out_predictions.csv`
- `testing/outputs/luke_lighthouse_ransac_field_v1/ransac_threshold_sensitivity.csv`
- `testing/outputs/luke_lighthouse_ransac_field_v1/motion_field.npz` (explicitly unvalidated)
- `testing/outputs/luke_lighthouse_ransac_field_v1/rigid_median_baseline_field.npz` (descriptive control)
