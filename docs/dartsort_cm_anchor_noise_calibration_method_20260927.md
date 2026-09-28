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

One stable and one zero-signal 100-event bank pair are generated per unit. The
100 draws at 25 and 50 events are overlapping, dependent block-balanced
subsamples of those fixed banks; their empirical quantile grid is 0.01 but this
is not independent Monte Carlo or inferential tail resolution. At 100 events
the entire bank is used, so all 100 repeated values are identical and provide no
simulation-frequency estimate. No parameter is chosen to reproduce CK failure
and no held-out-half output selects a setting. A stable surrogate result
diagnoses one realized measurement pipeline only; it does not estimate a false-
rejection frequency, establish true-neuron reliability, biological identity or
an anchor pass.
