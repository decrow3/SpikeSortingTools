# CJ v2 pre-outcome method freeze

CJ v1 is preserved as exploratory but is not the intended force gate. This v2
repair is frozen before reading any v2 outcome.

Two corrections are mandatory:

1. The central refractory annulus is exactly `9 <= abs(lag) <= 29` samples: 42
   lag values. Inherited upstream deduplication censors sample lags 0 through 8,
   so those lags cannot be counted as evidence. The shoulder remains
   `45 <= abs(lag) <= 89`: 90 values. Expected central exposure is therefore
   shoulder count times `42/90`.
2. Complete 5-s rest blocks are tiled from the exact complement of the canonical
   AE censor mask within W2's exact half-open frame interval and declared
   900-s time origin. No boundary is inferred from observed spike extrema. Every
   retained source row is asserted to have saved rest state.

The canonical input is
`/mnt/NPX/Luke/DARTsort_motion_experiments/ck_input_intervals_20260927/`.
The mask hash is
`86425e8a6627a10ac8932ba01b799cb7be7141f731ebb4ecc01e303326b9da55`;
sampling frequency is 29,999.759166666667 Hz; the exact source-frame interval is
`[26999783,37199701)` and local interval is `[0,10199918)`.

Everything else remains frozen: at least 100 eligible events per unit, at least
20 common 5-s blocks, expected central count at least 20, 1,000 block-bootstrap
draws with upper 95% bound at most 0.50, and observed ratio at or below the fifth
percentile of 1,000 within-segment nonwrapping block derangements. No evaluated-
pair collision filter is added. The null is only a same-genuine-segment,
nonwrapping, shared-support control; nonstationary rates and inherited
missingness remain limitations, so it is not biological identity proof.

The gate acts only on direct force edges, then recomputes connectivity and
unions unchanged QDA. All-pass and zero-pass controls remain mandatory. No
actual aggregation/deduplication, ISI-benefit claim, W3 execution or threshold
search is authorized. The output is versioned under the preserved CJ v1 packet
as `v2/`; a zero-support or zero-edge result is finite and must not trigger rule
relaxation.
