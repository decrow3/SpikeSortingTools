# Luke direct motion-scale audit

**Evidence snapshot:** 2026-08-29  
**Scope:** imec1 full session, discovery-only pre-registration peaks  
**Decision:** a simple fourfold DREDGE scale error is not supported; moderate
scale inflation remains plausible.

## Technical summary

An explicit amplitude-by-depth fingerprint tracker now remeasures physical
peak-raster translation without applying a motion field or using sorted-unit
labels. It agrees with DREDGE on direction in 95.7--100% of qualified pairs,
but generally measures smaller shifts. Across the three viable
amplitude-resolved raster definitions, robust observed-versus-DREDGE slopes
are 0.51, 0.64 and 0.61. The depth-only control gives 0.81.

The primary 2 micrometer / 6 micrometer-smoothed tracker retained 23
non-overlapping large-shift pairs. Its slope was 0.505 (bootstrap 95% interval
0.307--0.903), correlation was 0.744, and median observed/DREDGE ratio was
0.482. A separate 120-second pair-separation sensitivity gave slope 0.640
(0.335--1.107) and correlation 0.886 across 14 qualified pairs.

This supports either moderate DREDGE scale inflation or a systematic
difference between DREDGE's globally regularized field and direct local raster
matching. It does not support treating the successful 0.25 voltage-resampling
gain as a calibrated estimate of true tissue motion.

## What was measured

The audit uses the 40,462,735 localized peaks saved before registration. Peaks
are aggregated into ten-second amplitude-by-depth fingerprints. Time blocks
60 seconds apart are matched by an independently implemented explicit spatial
cross-correlation in physical micrometers. Only pairs for which DREDGE predicts
at least 6 micrometers of rigid displacement are considered. Pair starts are
120 seconds apart so primary comparisons do not reuse either endpoint.

A pair qualifies only when the optimum stays inside the search range, the
fingerprint correlation and peak margin pass fixed thresholds, and two
deterministic peak halves agree in direction and within 10 micrometers. The
primary run qualified 23 of 24 candidate pairs.

## Scale sensitivity

| Raster definition | Qualified pairs | Robust slope | 95% interval | Correlation | Median observed/DREDGE |
|---|---:|---:|---:|---:|---:|
| Amplitude-depth, 2 µm bins, 6 µm smoothing | 23 | 0.505 | 0.307--0.903 | 0.744 | 0.482 |
| Amplitude-depth, 2 µm bins, 10 µm smoothing | 22 | 0.644 | 0.468--0.999 | 0.825 | 0.602 |
| Amplitude-depth, 4 µm bins, 10 µm smoothing | 21 | 0.609 | 0.409--1.092 | 0.812 | 0.575 |
| Depth-only, 2 µm bins, 10 µm smoothing | 23 | 0.810 | 0.596--1.222 | 0.886 | 0.792 |

The 20 micrometer-smoothed amplitude raster retained only two pairs and is not
interpretable. The useful sensitivity range therefore centers near 0.5--0.8,
with uncertainty that can include unity for several definitions.

The discrete-gain comparison is deliberately reported separately. On the
primary tracker's median absolute error, gains 0.25 and 0.50 are nearly tied
(2.51 versus 2.61 micrometers), while mean absolute error favors 0.50 (4.47
versus 5.30 micrometers). The two other viable amplitude-resolved definitions
favor 0.50, and the depth-only control favors 0.75. Thus one loss function on
one raster can select 0.25, but the broader scale evidence does not identify a
uniform fourfold error.

## What this does and does not establish

The geometry path is already independently exact: acquisition metadata,
SpikeInterface coordinates and Kilosort positions agree in micrometers. This
makes a channel-pitch or coordinate-unit factor of four unlikely.

However, this audit and DREDGE share the same detected and localized peaks.
The matching code and loss are independent, but the biological observations
are not. Activity changes, common artifacts, localization bias and DREDGE's
global regularization can all create differences between the two estimates.
The result is therefore a direct implementation-scale remeasurement, not an
independent tissue-motion ground truth.

## Recommendation

Do not rescale the motion field globally to 0.25 on the strength of the sorter
or synthetic residual results. Retain 0.25 as an engineering candidate for
voltage resampling, and add 0.50 and 0.75 as physical-scale controls.

The next decisive test should use a genuinely separate observable: stable LFP
depth landmarks, raw multichannel waveform families qualified independently
of Kilosort labels, or an external mechanical/behavioral reference. It should
estimate displacement before any voltage warp and compare DREDGE, direct peak
raster, and the independent observable on the same prespecified episodes.

## Reproducibility

The implementation is `testing/luke_direct_motion_scale_audit.py`; focused
tests are in `testing/test_luke_direct_motion_scale_audit.py`. Primary outputs
are under `testing/outputs/luke_direct_motion_scale_audit/`; the 120-second
pair-separation sensitivity is under
`testing/outputs/luke_direct_motion_scale_audit_sep120/`. Neither prospective
holdout labels nor sorter labels were accessed.
