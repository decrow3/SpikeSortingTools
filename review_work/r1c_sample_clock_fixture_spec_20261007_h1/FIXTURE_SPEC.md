# R1c sample-clock known-answer fixture specification

Status: author-side draft, no real outcomes. This is input to a future frozen correction contract, not production authorization.

## Decision tested

The corrected confidence calculation must use time support derived from the same integer-sample bins that construct the observed cumulative ACG counts. A nominal 0.25-ms model must fail the fixture at the two frozen probe clocks.

## Frozen counter semantics

- Sort spike samples ascending as signed integer AP-frame-zero samples.
- For each ordered event pair with `j > i`, let `d = sample[j] - sample[i]`.
- Count only `0 <= d < 280` samples.
- Assign `k = d // 7`, giving forty bins `k = 0..39`.
- Bin `k` contains integer lags `7k..7k+6`; its right edge is exclusive at `7(k+1)` samples.
- Cumulative count `obs[k]` is the sum through bin `k`.
- Model upper time for `obs[k]` is exactly `7(k+1)/fs` seconds.
- Model center is exactly `7(k+0.5)/fs` seconds.
- Eligibility is the physical rule `center > 0.0005` seconds. At both frozen clocks, the first eligible index is 2. This rule must be retained explicitly rather than replaced by a hard-coded index.
- Confidence is `1 - PoissonCDF(obs, expected)`, with `expected = (N/T) * contamination * upper * 2N`.
- Contamination grid is 0.005 through 0.345 in steps of 0.005; a finite minimum at most 0.10 passes.

## Boundary fixture

Construct lags at `0, 1, 6, 7, 13, 14, 20, 21, 278, 279, 280` samples using a direct independent reference counter. Expected bin membership:

- bin 0: `0, 1, 6`
- bin 1: `7, 13`
- bin 2: `14, 20`
- bin 3: `21`
- bin 39: `278, 279`
- excluded: `280`

The production counter and independent reference must agree on the full 40-vector and cumulative vector. A helper compared against itself is not acceptable.

## Discriminating confidence fixture

Use forty zero observed counts, `N = 3500`, and each probe's full-session exposure. This case lies between the nominal and sample-derived power thresholds:

| Probe | fs (Hz) | exposure (s) | nominal min contamination | sample-derived min contamination | nominal pass | sample pass |
|---|---:|---:|---:|---:|---|---|
| imec0 | 29999.835983263598 | 10473.5537279367 | 0.10 | 0.11 | true | false |
| imec1 | 29999.759166666667 | 10473.553879363088 | 0.10 | 0.11 | true | false |

At contamination 0.10 and the final cumulative bin, expected confidence is `0.9035976907026476` nominal versus `0.8873296203604679` sample-derived for imec0, and `0.9035976874422771` nominal versus `0.8873302466827586` sample-derived for imec1.

Acceptance requires the corrected implementation to reproduce the sample-derived values and fail a deliberately nominal implementation. This fixture tests interval consistency only; it does not validate field-state assignment, block bootstrap composition, unit purity, or any scientific verdict.

## Integration controls

- Positive control: a zero-count high-N case well above both thresholds must pass both implementations.
- Negative control: a zero-count low-N case below both thresholds must be undefined/fail in both.
- Regression control: the separate inclusive 9-29-sample short-interval guardrail output must be bit-for-bit unchanged.
- Bootstrap fixture: combine two synthetic blocks with distinct ACG vectors and deterministic multiplicities; independently sum counts/exposures and verify the corrected helper consumes the composed counts with sample-derived edges.
- Provenance: save the independent reference code, expected JSON, exact clock inputs, production source hash, and test receipt in the correction packet.

## Implementation checks

- Done: sample-bin boundaries and right-edge convention derived from the executed historical counter.
- Done: independent closed-form zero-count confidence arithmetic identifies a nominal-pass/sample-fail case at both exact clocks.
- Not done: production implementation or real spike-time evaluation; those belong to the separately frozen correction contract.
- Can establish: a cheap deterministic test that detects the confirmed nominal-versus-sample timing defect.
- Cannot establish: corrected unit classifications or R1/R1-T decisions.
