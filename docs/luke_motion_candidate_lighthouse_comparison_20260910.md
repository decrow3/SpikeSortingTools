# Frozen motion candidates against depth-aware lighthouse observations

## Result

This cached comparison treats every field as a candidate and the lighthouse panel as independent validation. It does not authorize correction. The AP candidates are the completed full-session MEDiCINe nonrigid field and its frozen equal-depth rigid projection; the LFP candidate is the frozen `band_0p5_8` rigid trace.

The result is regime-dependent. In 930–1030 s, LFP rigid has the lowest all-increment RMSE (42.35 µm), versus 83.76 µm for AP rigid, 84.62 µm for AP nonrigid, and 109.17 µm for no motion. But on the 24 lighthouse-quiet increments in that same window, no motion is best (1.44 µm), while LFP rigid rises to 48.86 µm. In the entirely quiet 1150–1200 s window, no motion is again best (1.43 µm); AP rigid, AP nonrigid, and LFP rigid produce 4.92, 9.03, and 21.86 µm RMSE, respectively. Thus LFP rigid captures the large shared excursions best but also introduces the most quiet-period false motion.

## All-increment comparison

| Window | Candidate | Families | Increments | RMSE (µm) | Skill vs zero | Median family r | Median slope |
|---|---|---:|---:|---:|---:|---:|---:|
| 930–1030 | No correction | 6 | 53 | 109.17 | 0.000 | nan | 0.000 |
| 930–1030 | LFP rigid | 6 | 53 | 42.35 | 0.612 | 0.955 | 0.814 |
| 930–1030 | AP rigid | 6 | 53 | 83.76 | 0.233 | 0.937 | 0.238 |
| 930–1030 | AP nonrigid | 6 | 53 | 84.62 | 0.225 | 0.941 | 0.222 |
| 1150–1200 | No correction | 4 | 28 | 1.43 | 0.000 | nan | 0.000 |
| 1150–1200 | LFP rigid | 4 | 28 | 21.86 | -14.329 | -0.112 | -2.881 |
| 1150–1200 | AP rigid | 4 | 28 | 4.92 | -2.449 | -0.065 | -0.130 |
| 1150–1200 | AP nonrigid | 4 | 28 | 9.03 | -5.335 | 0.183 | 2.197 |

## Quiet and movement checks

| Window | Regime | Candidate | Families | Increments | RMSE (µm) | Skill vs zero |
|---|---|---|---:|---:|---:|---:|
| 930–1030 | quiet | No correction | 6 | 24 | 1.44 | 0.000 |
| 930–1030 | quiet | LFP rigid | 6 | 24 | 48.86 | -33.001 |
| 930–1030 | quiet | AP rigid | 6 | 24 | 8.28 | -4.759 |
| 930–1030 | quiet | AP nonrigid | 6 | 24 | 10.38 | -6.225 |
| 930–1030 | movement | No correction | 7 | 33 | 133.19 | 0.000 |
| 930–1030 | movement | LFP rigid | 7 | 33 | 39.75 | 0.702 |
| 930–1030 | movement | AP rigid | 7 | 33 | 103.32 | 0.224 |
| 930–1030 | movement | AP nonrigid | 7 | 33 | 105.38 | 0.209 |
| 1150–1200 | quiet | No correction | 4 | 28 | 1.43 | 0.000 |
| 1150–1200 | quiet | LFP rigid | 4 | 28 | 21.86 | -14.329 |
| 1150–1200 | quiet | AP rigid | 4 | 28 | 4.92 | -2.449 |
| 1150–1200 | quiet | AP nonrigid | 4 | 28 | 9.03 | -5.335 |

There were no ≥20 µm lighthouse increments in 1150–1200 s, so that window contributes only to the quiet stratum.

## Fair-comparison contract

- Lighthouse identities, family mapping, plausibility exclusions, and the 5 µm lattice-node sensitivity were frozen before sampling candidates.
- Each candidate is sampled at accepted events' actual times. AP nonrigid is also interpolated at each event's frozen template reference depth.
- Strict events are reduced to consecutive 5 s family increments. Every family receives one vote; labels 557/673/675 remain one family.
- The primary table uses common support across all candidates and requires at least three increments per family.
- Quiet and movement strata are defined independently from the candidates using absolute lighthouse increments below or above 20 µm.
- No sign, gain, lag, offset, smoothing, interpolation across LFP support gaps, or candidate-informed lighthouse selection is fitted.

## Evidence retained

Strict, lower-score, identity-ambiguous, unsupported, lattice-phase, per-family residual, correlation, slope, and support evidence are saved separately. The AP package remains marked `requires_review` and `correction_ready: false`.

## Interpretation boundary

Agreement can reject obvious candidate failures but cannot certify biological identity, physical displacement calibration, or correction safety. The quiet/movement split is descriptive because it uses the observed lighthouse increment magnitude. Only the two windows covered by the frozen lighthouse panel are evaluated here.

## Reproduce

```bash
MPLCONFIGDIR=/tmp/luke-motion-candidate-lighthouse-mpl python -m testing.luke_motion_candidate_lighthouse_comparison_v1
```
