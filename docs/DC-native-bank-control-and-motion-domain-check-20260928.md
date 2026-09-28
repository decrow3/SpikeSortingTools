# DC native-bank control and motion-domain check

## Frozen field-domain analysis

Analyze saved W2 finals for static S, original accepted D2L, REMATCH0, and
CD1_FULL. Use the adopted two-layer v1 field (SHA-256 `85062a37...`), canonical
censor mask (`86425e8a...`), catalogue (`46463cda...`), and cropped W2 field
(`ee34e8a...`).

Define the full-session scalar field as the first depth column minus its
centered 480-sample rolling median (`min_periods=1`). Interpolate it linearly at
exact AP event times. On the exact half-open W2 source-frame interval, define
three mutually exclusive domains in this order:

1. `negative_excursion`: deviation `< -120 µm`;
2. `outside_mask_flat`: outside the canonical half-open mask and absolute
   deviation `< 20 µm`;
3. `catalogue_outside_remainder`: everything else.

The third domain is not called true rest: it may contain displaced time outside
the catalogue. Compute exposure from exact piecewise-linear threshold crossings
and mask/window boundaries, not 0.25-second grid-bin counts or event extrema.
Report all/assigned/noise counts and segment-safe ISIs per arm/domain.

Across-unit rank correlations are descriptive state/rate-heterogeneity checks,
not identity evidence. Restrict to units with at least 100 assigned flat-domain
events. Use `log((count + 0.5) / exposure)` and report negative-vs-flat Spearman
rho. A same-row control may average other units' flat log rates whose median
saved template-channel depth is within 10 µm; saved channel cohorts differ by
arm and are not cross-arm identity matches.

Do not repeat or promote the earlier T16 ceiling statistic: its block loop used
clamped last-bin indexing for out-of-range blocks and counted requested rather
than in-bounds exposure. No threshold search, ceiling calibration, null, raw
voltage read, sort, or holdout access is authorized.

Limits: 1,200 all-work CPU seconds, two numerical threads, one reader, 20 GB
RAM, 1 GB scratch, 500 MB final, at least 30 GB free, no raw voltage and no GPU.
H1 cumulative accounting begins at 17,513.22 seconds.

The native-bank CD0 arm, once completed externally, receives only the unchanged
frozen DA RF evaluator followed by an independent saved-output review. It is not
part of this four-arm field-domain pass.
