# CM anchor-noise calibration: pre-outcome method

This cached-only calibration is frozen before outputs. It uses CK units 23 and
120, their unchanged physical supports, and each unit's half-1 noise estimate.
It does not fit temporal lag, gain, support or a motion correction.

For each unit and each half, deterministic block-balanced samples of 25, 50 and
100 events are drawn 100 times (the 100-event case is the full saved half).
Cosine is reported separately for median and mean templates. These are labeled
sample-size/estimator diagnostics and cannot reclassify the frozen CK result.

The known-stable surrogate uses the fixed half-1 median waveform. This template
already contains finite-sample waveform noise. Independent synthetic events add
noise sampled only from half-1 spike-free edge residuals: 20-sample residual
segments preserve physical-channel covariance and within-segment temporal
structure, but tiling them to 121 samples cannot preserve covariance across tile
boundaries or longer than 20 samples. Ten equal blocks of ten events are used.
A zero-signal control uses the same generated noise without the waveform.

There are 100 Monte Carlo repetitions per unit/sample-size/estimator/control,
giving empirical tail-probability resolution about 0.01. No parameter is chosen
to reproduce CK failure and no held-out-half output selects a setting. A stable
surrogate result diagnoses the measurement pipeline only; it does not establish
true-neuron reliability, biological identity or an anchor pass.
