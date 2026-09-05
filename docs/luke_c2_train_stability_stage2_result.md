# Stage 2 result — threshold configurations are not separated

**Date:** 2026-09-05  
**Status:** complete; preregistered decision is **no threshold change**  
**Prespec:** [`luke_c2_train_stability_stage2_prespec.md`](luke_c2_train_stability_stage2_prespec.md)  
**Code:** [`testing/luke_c2_stability_stage2.py`](../testing/luke_c2_stability_stage2.py),
[`testing/luke_c2_stability_stage2_analysis.py`](../testing/luke_c2_stability_stage2_analysis.py)  
**Outputs:** `testing/outputs/luke_c2_stability_stage2/`

## Decision

Neither 8/8 nor 9/9 replaces production 12/9. Both candidates have lower point
estimates of the failure rate, but neither paired 97.5% donor-bootstrap interval
excludes zero. Under frozen rule 3 the outcome is therefore
`no_threshold_change`; 12/9 remains the operational baseline by default, not
because Stage 2 showed it superior.

| candidate vs 12/9 | failure-rate difference | 97.5% CI | verdict |
|---|---:|---:|---|
| 8/8 | −0.112 | [−0.311, +0.026] | not separated |
| 9/9 | −0.102 | [−0.306, +0.036] | not separated |

The corresponding marginal failure rates were 0.179 for 12/9, 0.066 for 8/8,
and 0.077 for 9/9. Those marginals are descriptive; the paired donor-level
interval is the decision endpoint.

## Run and provenance

The run completed all **588 cells**: 14 frozen donors × 14 frozen train
realisations × three configurations. Analysis verified all 14 donor IDs, all 14
realisation IDs and truth hashes, and all three applied-setting maps. Every cell
contained 687 events, correction was off (`effective_nblocks=0`), and no
refractory endpoint was undefined. The frozen Python, NumPy, pandas, SciPy and
`uv.lock` identities matched at analysis time.

## Guardrails and secondary findings

Neither candidate acquired a new systematic donor. D10 remained systematic
under 12/9 (14/14 failures) and was not systematic under either candidate. The
worst donor-level deterioration remained below the frozen cap: 8/8 added three
failures on D14, while 9/9 added two on D05.

| endpoint difference vs 12/9 | 8/8 | 97.5% CI | 9/9 | 97.5% CI |
|---|---:|---:|---:|---:|
| FP p90 | −257 | [−272, 0] | −226 | [−276, +60] |
| split rate | +0.046 | [−0.015, +0.122] | +0.041 | [−0.020, +0.122] |
| FP maximum | +484 | [+188, +722] | +342 | [−256, +697] |
| refractory median | −0.00152 | [−0.00179, −0.00126] | −0.00141 | [−0.00164, −0.00093] |

FP maximum and refractory median are reported secondary endpoints, not frozen
decision endpoints. In particular, the large 8/8 FP-maximum regression does not
change the formal `not_separated` verdict, but it is an additional reason not to
advance 8/8 casually.

The unadjusted cell-level McNemar tests appear significant (8/8: p=0.0021;
9/9: p=0.0045), but they treat 196 donor-realisation cells as independent. The
prespec explicitly excludes them from decisions because the 14 realisations
within a donor share waveform, amplitude and placement. The donor-bootstrap and
the prespecified donor-level t sensitivity both fail to separate either
candidate.

## Consequence

The integer detection-threshold branch closes with **no production change**.
Do not open fractional threshold refinement, L1C context ranking, held-out
static evaluation, or matched-real-data evaluation for 8/8 or 9/9. Those gates
were conditional on a stable Stage 2 winner, and there is none. Pipeline work
returns to the two required development branches: physically validated external
voltage registration and unwarped motion-aware identity handling.
