# Luke KS4 native voltage-registration operator audit result

**Decision date:** 2026-08-31

**Decision:** Do not advance the tested KS4 4.0.27 native voltage-registration
operator to a bounded externally supplied-`dshift` sort.

## What was tested

The completed audit implemented the six preregistered arms from
[`Luke KS4 native operator audit plan.md`](../testing/Luke%20KS4%20native%20operator%20audit%20plan.md):

- stationary without correction;
- stationary with native KS4 `dshift=0`;
- forward-moved without correction;
- forward-moved with native-order KS4 inverse;
- forward-moved with the identical KS4 matrix applied before preprocessing;
- forward-moved with a SpikeInterface inverse.

The discovery factorial used six sealed donor identities, three previously
opened Luke backgrounds, three forward generators, twelve nonzero signed
displacements from ±1 to ±20 µm, and exact Kilosort 4.0.27 spatial matrices.
Whitening was frozen from the accepted full-probe rescue sort. No sorter, GPU,
prospective holdout, or new int16-warped recording was used.

The source-domain adapter re-extracted each sealed donor identity from the
accepted rescue recording at the same sample and channel, then cast that
existing KS4 input to float32. All ten donors passed the frozen adapter gate:
raw-to-rescue local cosine ranged from 0.858 to 0.995, with median 0.944.

## Validation evidence

- Synthetic and unit contracts passed.
- Guarded 4,096-sample versus full 2-second KS high-pass output agreed at the
  event window with relative L2 error `1.46e-7` and maximum absolute error
  `5.96e-7`.
- The one-template dry run validated matrix orientation, signed displacement,
  hashes, and runtime.
- The full run produced exactly 2,628 waveform rows, 2,610 fixed-pair
  separability rows, and 61 finite operator matrices with no missing values.
- The final validator hashed the complete artifact set before issuing the
  decision.

## Preregistered gate result

| Metric | Required | Observed | Pass |
| --- | ---: | ---: | --- |
| Worst-generator median Δ residual | ≤ −0.005 | +0.102625 | No |
| Worst-generator median Δ absolute amplitude error | ≤ +0.005 | +0.028315 | No |
| Worst-generator median Δ cosine | ≥ 0 | −0.005995 | No |
| Worst-generator tax-adjusted residual improvement | ≥ +0.005 | −0.119482 | No |

The stationary zero-shift KS4 tax was itself measurable:

- median residual fraction: 0.016857;
- median absolute amplitude error: 0.017769;
- median cosine loss: 0.000142.

Native correction did not merely fail to exceed that tax. Under the least
favorable preregistered generator, it increased residual error substantially.

## Generator and displacement dependence

The result was heterogeneous in a scientifically informative way:

| Forward generator | Median native Δ residual vs uncorrected | Tax-adjusted residual improvement |
| --- | ---: | ---: |
| SI IDW4 | +0.014687 | −0.031545 |
| SI kriging, p=2, σ=10 | +0.102625 | −0.119482 |
| SI kriging, p=2, σ=20 | −0.058409 | +0.041552 |

KS4 helped the broad σ=20 generator and the largest displacements, but harmed
the sharp σ=10 generator and generally harmed small displacements. That is
exactly why the plan froze the worst-generator aggregation: pooling would have
hidden strong dependence on the unknown continuous waveform model.

Even the favorable σ=20 residual result did not clear the complete gate because
absolute amplitude error worsened by 0.028315, above the allowed 0.005.

## Mechanism findings

Median residual across all moved cases was:

- uncorrected: 0.1243;
- native KS4 inverse: 0.1744;
- external-order KS4 inverse: 0.1751;
- SI inverse: 0.1530.

Native and external KS4 ordering were therefore very similar in this bounded
screen. The hypothesized order effect did not explain the historical failure.
Nor was the native operator generally less destructive than SI.

Pairwise template-distance retention remained relatively high, but matched
filter SNR often increased because interpolation reduced projected noise. That
did not compensate for the waveform residual and amplitude failures, and it
reinforces why apparent SNR alone was not used as the gate.

![Operator recovery curves](../testing/outputs/luke_ks4_native_operator_audit/operator_recovery_curves.png)

![Template separability](../testing/outputs/luke_ks4_native_operator_audit/template_separability.png)

![Zero-shift tax](../testing/outputs/luke_ks4_native_operator_audit/zero_shift_tax.png)

## Empirical smoothness companion

The pulled-forward smoothness check was **unvalidated**, not failed. None of the
three provisional recurrent families met the frozen eligibility rule:

- family 265 had five states but only 19.10 µm energy-weighted depth span,
  below the 20 µm threshold;
- family 294 spanned 21.90 µm but had only two states;
- family 338 had three states but only 5.80 µm span and unstable polarity.

There was therefore no defensible empirical basis for claiming that the three
continuous forward generators describe Luke's true waveform-versus-depth
manifold. This uncertainty cannot reverse the operator failure; it further
limits generalization.

## Stopping decision and scope

The edge challenge was not run after the necessary native operator gate failed.
Edge success could not rescue an operator that already failed on 648 interior
native cases. The supplied-trajectory KS4 sort was likewise not run because it
was explicitly conditional on passing this audit.

This result shows that the exact KS4 4.0.27 `sig_interp=20` native operator did
not have a favorable cost-benefit ratio across the tested displacement
amplitudes and three preregistered continuous forward models. It does **not**
reject:

- motion-estimator development;
- coordinate-only motion correction;
- KS2 or another non-warping sorter strategy;
- every conceivable voltage-registration operator.

It also does not establish that every displacement should be treated with the
same application policy. A post hoc displacement-stratified analysis found
that residual and cosine first favored native correction in every signed
generator stratum at 20 µm. Absolute amplitude preservation still failed at
that magnitude, so no complete crossover exists on the tested grid and no
selective policy is currently authorized. This observation motivates a new
confirmatory hypothesis rather than changing the failed preregistered decision:
use an exact identity branch during benign motion and pay interpolation cost
only beyond a separately validated crossover. See
[`Luke KS4 selective correction crossover plan.md`](../testing/Luke%20KS4%20selective%20correction%20crossover%20plan.md).

The active stage-local ladder should therefore continue with the motion-
estimator bakeoff and coordinate-only application. Native KS4 voltage warping
is not on the advancement path under the current evidence.

## Artifacts

- Machine decision: [`final_decision.json`](../testing/outputs/luke_ks4_native_operator_audit/final_decision.json)
- Operator result: [`result.json`](../testing/outputs/luke_ks4_native_operator_audit/result.json)
- Generator gate summary: [`generator_gate_summary.csv`](../testing/outputs/luke_ks4_native_operator_audit/generator_gate_summary.csv)
- Per-case metrics: [`case_metrics.csv`](../testing/outputs/luke_ks4_native_operator_audit/case_metrics.csv)
- Pairwise metrics: [`pair_separability_metrics.csv`](../testing/outputs/luke_ks4_native_operator_audit/pair_separability_metrics.csv)
- Smoothness result: [`result.json`](../testing/outputs/luke_ks4_native_operator_audit/waveform_depth_smoothness/result.json)
- Implementation: [`luke_ks4_native_operator_audit.py`](../testing/luke_ks4_native_operator_audit.py)
- Selective-correction discovery summary: [`result.json`](../testing/outputs/luke_ks4_native_operator_audit/selective_correction/result.json)
- Crossover implementation: [`luke_ks4_selective_correction_crossover.py`](../testing/luke_ks4_selective_correction_crossover.py)
