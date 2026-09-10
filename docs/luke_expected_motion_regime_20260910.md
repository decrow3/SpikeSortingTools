# Expected motion regime for Luke0804 imec0

**Window:** 930--1230 s  
**Primary estimate:** screened 5-sigma-relaxed MEDiCINe, 10,000 steps  
**Purpose:** numerical reference for motion magnitude, speed, estimator
bandwidth, probe-scale tracking limits, and correction design

The MEDiCINe field is an estimate, not physical ground truth. The speeds below
are absolute first differences of an already regularized field. They may
understate under-resolved physical speed, while isolated depth-knot changes may
overstate physical nonrigidity because of local support or optimizer variability.

## Headline regime

- Temporal grid: 0.25 s, or 4 Hz.
- Spatial field: four knots separated by 1,279.74 um across the 3.84 mm probe.
- Peak-to-peak displacement by knot: 282.24, 339.81, 265.82, and 252.23 um.
- Peak-to-peak common component, the mean of the four knots: 200.23 um.
- Largest common 0.25 s step: 82.67 um, equivalent to 330.67 um/s.
- Largest step at any individual knot: 136.06 um, equivalent to 544.23 um/s.
- Large coherent excursions typically accumulate about 150--200 um over
  1--5 s.

Luke is substantially larger and faster than ordinary mouse AP examples, far
smaller in total displacement than millimetre-scale human or long NHP
recordings, and dynamically closer to fast-human motion during its short bursts.

## Common-motion speed distribution

The common component is the most defensible summary of shared physical motion.

| Interval | Median | P90 | P99 | Maximum |
| --- | ---: | ---: | ---: | ---: |
| Full 930--1230 s | 13.92 um/s | 60.13 um/s | 225.62 um/s | 330.67 um/s |
| Motion-rich 935--1150 s | 14.74 um/s | 90.79 um/s | 239.56 um/s | 330.67 um/s |
| After 1150 s | 12.84 um/s | 26.31 um/s | 39.33 um/s | 48.02 um/s |

| Interval | Below 80 um/s | Below 160 um/s | Above 200 um/s | Above 320 um/s |
| --- | ---: | ---: | ---: | ---: |
| Full window | 91.49% | 96.50% | 1.92% | 0.083% |
| 935--1150 s | 88.14% | 95.12% | 2.67% | 0.116% |
| After 1150 s | 100% | 100% | 0% | 0% |

The practical description is **mostly ordinary motion with a few fast,
potentially under-resolved bursts**, rather than a uniformly extreme five-minute
recording. Useful targeted episodes are approximately 935--941, 974--986,
998--1002, 1014--1025, 1071--1075, and 1097--1112 s.

## Individual depth-knot tail

Across the full window, at least one knot exceeds 80 um/s in 37.36% of
intervals, 160 um/s in 11.68%, and 200 um/s in 7.92%. After 1150 s the common
component never exceeds 80 um/s, yet an individual knot still exceeds 80 um/s
in 31.66% of intervals and reaches 173.06 um/s.

That divergence warns against treating the 544.23 um/s knot maximum as expected
whole-probe velocity. The local tail may combine genuine differential motion
with uneven peak support, population changes, localization bias, or optimizer
variability.

## Temporal bandwidth and peak support

The nominal 1 s triangular kernel becomes approximately

```text
0.25 * previous + 0.50 * current + 0.25 * next
```

on the 250 ms grid. It retains about 85% of a 2 s-period oscillation, 50% at a
1 s period, 15% at a 0.67 s period, and removes a 0.5 s alternation. The useful
bandwidth is therefore closer to 0.5--1 Hz than 4 Hz. Interpret the present
field as a broad 1--2 s trajectory, not four independent measurements per
second. Evaluating it at 100 ms can smooth template-position steps but cannot
create new 100 ms motion information.

The fit uses 296,684 screened peaks, approximately 989 peaks/s. Counts across
four equal observed-depth regions show strongly uneven support:

| Bin width | Median peaks/bin | Median by depth region | Empty depth-time cells |
| --- | ---: | --- | ---: |
| 250 ms | 215 | 60, 7, 97, 22 | 4.19% |
| 100 ms | 80 | 21, 2, 36, 9 | 8.65% |
| 50 ms | 38 | 9, 1, 17, 4 | 14.09% |
| 25 ms | 18 | 4, 0, 8, 2 | 23.47% |

Practical ceilings are therefore:

- Shared/rigid motion: 100 ms is plausible; 50 ms is exploratory and still
  needs roughly 250--500 ms regularization.
- Four-knot nonrigid motion: 250 ms is defensible, 100--125 ms is aggressive,
  and 50 ms is underconstrained in sparse depth regions.
- Below 25--50 ms: these AP-peak summaries are insufficient, regardless of the
  original 30 kHz voltage sampling rate.

## Probe-scale velocity heuristic

Luke imec0 has 192 rows spanning 0--3820 um, a 20 um row pitch, and 25.61 um
nearest-contact spacing. Adjacent rows alternate x columns, so exact same-column
contacts recur every 40 um.

| Estimate interval | One 20 um row/bin | Two rows/bin |
| --- | ---: | ---: |
| 250 ms | 80 um/s | 160 um/s |
| 125 ms | 160 um/s | 320 um/s |
| 100 ms | 200 um/s | 400 um/s |
| 50 ms | 400 um/s | 800 um/s |

This one/two-row rule is a conservative spatial-ambiguity heuristic, not a hard
limit. At 250 ms, Luke's maximum common step spans 4.13 rows and the maximum
individual-knot step spans 6.80 rows. Sampling the 330.67 um/s common burst at
100--125 ms would reduce it to about 1.7--2.1 rows. A real 544.23 um/s local
trajectory would need roughly 50 ms sampling to approach 1.4 rows, where the
available nonrigid peak support is inadequate.

## Numerical stability

Extending the same fit from 10,000 to 30,000 optimizer steps changes the field
by 11.24 um RMS and as much as 99.72 um at one time/depth. Per-knot correlations
are 0.903, 0.985, 0.990, and 0.969. Small rapid excursions and isolated local
deviations therefore require split-half or optimizer replication.

MEDiCINe uses peak time, localized depth, and amplitude; it discards waveform
identity and full multichannel geometry. It cannot by itself distinguish motion
from firing-rate changes, motion-dependent detectability, localization bias,
coherent artifacts, or waveform-family turnover.

## Correction implications

The goal is not stationary-looking voltage. It is motion-invariant localization,
template support, matching, clustering, and longitudinal identity on the
original unwarped AP signal.

- Move template support and search position during supported ordinary motion.
- During the rare fastest periods, permit reduced confidence, wider search, or
  an explicit track gap instead of forcing smooth identity.
- Reconnect across gaps using waveform evidence selected independently of the
  motion estimate.
- Treat corrected depth as routing or evaluation information, not sufficient
  identity evidence.
- Keep voltage interpolation as a mechanistic control. Prior rigid and nonrigid
  voltage warps degraded sorting despite essentially unchanged event timing.

For DARTsort, the first comparison should use identical unwarped voltage in
static and native-motion arms. The present MEDiCINe field is an external
reference and possible later handoff, but its bandwidth and uneven depth support
must not be mistaken for a fully observed fast trajectory.

## DREDge comparison scale

- Ordinary mouse drift: about 100 um, mildly nonrigid, commonly treated with
  1 s AP bins.
- Imposed mouse triangle: about 50 um amplitude with a 100 s period, only about
  1--2 um/s.
- NHP insertion: about 26 mm at 10 um/s, primarily a long-range turnover problem.
- Fast human intraoperative motion: roughly 500 um oscillations on about 1 mm of
  drift, supported by much higher-bandwidth LFP evidence.

Luke is intermediate in displacement but closer to the fast-human examples in
short-term dynamics. The closest precedent argues for higher-bandwidth evidence
through the bursts, not for assuming discontinuous physical motion. See the
[DREDge paper](https://doi.org/10.1038/s41592-025-02614-5).

## Audited sources

- `testing/outputs/luke_screened_medicine_300s_v1/fit_5sigma_relaxed_10000/field.npz`
  — SHA-256 `33d372eafb3dd845c80e425d5f00886d1b7421ea1873869fcb2d5c67342d9cd2`.
- `testing/outputs/luke_screened_medicine_300s_v1/fit_5sigma_relaxed_30000/field.npz`
  — SHA-256 `2637d851f105d84ca459d8f52f429344accdb1a91ceedd7cb1a5776036536179`.
- `testing/outputs/luke_screened_medicine_300s_v1/input_5sigma_relaxed/peaks.npy`
  — SHA-256 `42721d0779c04043f7e6f3e0b0c92b7cbaffd8e843f16be78cc110e070d4554b`.
- `testing/outputs/luke_screened_medicine_300s_v1/input_5sigma_relaxed/locations.npy`
  — SHA-256 `8a90522447e3631ec47bd0fa626ed3192d4a1e309932708bd3bfef1a0b9fdc3e`.
- Luke imec0 `channel_positions.npy`
  — SHA-256 `0469ca92fb739a0cfd2f1613262d3a2d75af1098385385d7462ee6e3fd038d75`.

