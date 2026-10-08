# Refractory interval defect: independent H1 impact review

## Verdict

The confirmed defect is a clock mismatch inside the R1 sliding-refractory calculation. Observed positive-lag counts are accumulated in forty bins of seven samples, while the Poisson confidence model assigns those same cumulative counts to forty nominal 0.25-ms bins. At the frozen clocks, a bin is about 0.233335 ms and forty bins cover about 9.3334 ms, not 10 ms. The modeled refractory upper edges are therefore longer than the observed count support by about 7.14%.

This is load-bearing for sliding-RP minimum-contamination values and pass flags, and consequently for strict `P`, through-displacement `K`, their bootstrap intervals, and R1/R1-T verdicts on both probes. It does not affect the separately implemented inclusive 9-29-sample short-interval guardrail, Allen-like metrics, or D0-D2 spike-correspondence mechanics.

No corrected scientific outcome is claimed here. The direction of the model change alone is not enough to infer which unit flags or arm contrasts change.

## What actually ran

The directly available earlier evaluator snapshot, SHA-256 `a31f1349854eff0e4c732063993be37e8cf53de5d206628f66d68a65ff7f341e`, shows:

- `sliding_rp_from_acg` cumulatively sums the first 40 observed bins, but defines `rp_upper` as `(1..40) * 0.00025` seconds (`full_session_r1_evaluation.py:98-111`).
- `_lag_counts` bins integer sample differences with `k = d // 7` and stops at 280 samples (`full_session_r1_evaluation.py:114-133`).
- all, q0, and displaced calculations feed those counts into the nominal-time confidence model (`full_session_r1_evaluation.py:288-318`).
- bootstrap recomputation uses the same cached ACG tensor and function (`full_session_r1_evaluation.py:323-334`).
- `P` and `K`, their intervals, and the overall verdict consume the resulting composite flags (`full_session_r1_evaluation.py:473-490`).

The corrected R1c packets are not shared locally, and their exact executed source SHA-256 `415f29d7f26102606143d0bd4fbb8ed9a136199a6642e664d50c2a6716b7c423` is absent from the shared packet tree. However, the frozen post-sort census contract binds both R1c packets to that evaluator and explicitly freezes the same 7-sample/0.25-ms semantics. Its producer audit reports actual bin widths of `0.0002333346090260354` s (imec0) and `0.00023333520649651882` s (imec1), with forty-bin support of `0.009333384361041416` and `0.009333408259860752` s. This confirms impact while leaving exact R1c source inspection as a provenance gap.

## Saved sufficient statistics

The published R1 tables save derived minimum-contamination values and pass flags, spike counts, exposures, and downstream decisions. They do not save the 40-bin observed lag counts or confidence matrices. `AVAILABILITY_LEDGER.csv` explicitly marks `sliding_rp_observed_lag_counts` and `sliding_rp_confidence_matrix` unavailable for every probe/arm.

Therefore:

- exact corrected per-unit results cannot be derived by trivial arithmetic from the published summaries;
- exact corrected bootstrap P/K intervals cannot be reconstructed from saved `per_draw_deltas`, because those are already aggregated under the defective metric;
- zero-observed-count feasibility calculations in `SRP_AUDIT.csv` are valid power diagnostics, not replacements for each unit's unknown observed lag vector.

For point estimates, the minimally sufficient missing artifact is the per-unit, per-scope 40-bin positive-lag count vector plus spike count, exposure, and exact sampling frequency. For the frozen block bootstrap, the missing artifact is the per-unit by 300-s-block by scope 40-bin ACG tensor, per-block spike counts, and field-state exposure matrix, with the original block membership and state/segment rules bound by hash.

## Scope and provenance

Files inspected include the earlier frozen evaluator source/config and the immutable post-sort census, selection, panel, and D0 packets. No raw voltage, waveform arrays, RF/holdout, new sorting, or scientific outcome recomputation was performed. A bounded repository/shared-packet search found no other consumer bound to the R1 evaluator hash; similarly named direct refractory metrics remain `unknown` until their executed-source provenance is traced.

## Implementation checks

- Done: parameter-to-consumer trace -> 7-sample observed bins flow into nominal 0.25-ms confidence edges, then into per-unit flags, bootstrap P/K, and verdict mapping (earlier frozen evaluator source lines cited above).
- Done: clocks/support -> actual 7/fs bin widths and forty-bin supports independently read from the immutable `SRP_AUDIT.csv`; both are shorter than the modeled interval.
- Done: saved-statistic availability -> observed lag counts and confidence matrices are explicitly absent in all four probe/arm ledgers.
- Done: consumer classification -> each known R1, census, selection, panel, and divergence consumer is listed in `IMPACT_MAP.csv` with a path and action.
- Not done: exact R1c executed-source read -> hash `415f29d7...` is bound but the source snapshot is not present on shared storage.
- Not done: corrected outcomes -> unavailable sufficient statistics prevent summary-only recalculation; no spike-train traversal was performed under this audit.
- Can establish: the metric implementation is temporally inconsistent and every decision that consumes its values must be treated as affected pending correction.
- Cannot establish: which units flip, corrected P/K intervals, corrected R1/R1-T verdicts, biological purity/identity, or a causal arm advantage.
