# CM: cached anchor-noise calibration

## Verdict

The present 100-event, high-dimensional CK statistic can reject a known-stable
waveform at the observed training-noise scale. For unit 23, the stable surrogate
scores only 0.867 with median templates at 100 events per half; unit 120 scores
0.910. Observed scores are lower, 0.767 and 0.839. Reliability rises strongly
from 25 to 100 events, and mean templates score above medians in every observed
comparison.

This does not rescue or reclassify CK: **0/8 anchors remain qualified** under the
frozen bootstrap-lower >=0.90 rule. The surrogate's zero-signal control is
positively biased, so synthetic scores are not calibrated biological
probabilities or true-neuron reliability estimates.

## Frozen comparison

The method was committed as `36da5eb` before outcomes. It uses only units 23 and
120, their unchanged physical supports, unchanged half-1 noise, fixed zero lag
and unit gain. One hundred deterministic block-balanced repetitions were run at
25, 50 and 100 events per half; empirical tail resolution is about 0.01.

| Unit/source | Estimator | 25 events | 50 events | 100 events |
| --- | --- | ---: | ---: | ---: |
| 23 observed | median | 0.453 | 0.625 | 0.767 |
| 23 observed | mean | 0.558 | 0.713 | 0.835 |
| 23 stable surrogate | median | 0.584 | 0.750 | 0.867 |
| 23 stable surrogate | mean | 0.641 | 0.778 | 0.872 |
| 120 observed | median | 0.570 | 0.732 | 0.839 |
| 120 observed | mean | 0.657 | 0.793 | 0.882 |
| 120 stable surrogate | median | 0.684 | 0.824 | 0.910 |
| 120 stable surrogate | mean | 0.735 | 0.846 | 0.919 |

Values are medians across the 100 repetitions. At 100 events the full saved
half is deterministic, so the repeated value is not an uncertainty interval.

## Controls and limitations

The surrogate fixes the half-1 median waveform, which itself contains
finite-sample waveform noise. It adds independent samples from half-1
spike-free edge residuals. This preserves physical-channel covariance and up to
20 samples of temporal covariance, but tiling to 121 samples breaks longer-range
and tile-boundary covariance.

The zero-signal median-template control rises from about 0.087--0.089 at 25
events to 0.269--0.297 at 100; mean-template values rise from 0.098--0.105 to
0.285--0.319. Shared structure in the finite training residual pool therefore
creates a positive baseline. That makes the stable surrogate unsuitable as a
calibrated null and likely optimistic, but it does not erase the central finding
that a fixed waveform can score below 0.90 under this measurement pipeline.

The mean-versus-median advantage is descriptive, not a post-hoc replacement of
the frozen CK estimator. Independent spatial waveform change across halves also
remains possible. No support, lag, gain, noise scale or surrogate parameter was
fit to the held-out half.

## Next informative test

Do not start per-event observed-channel recentering first: the channel is an
output-dependent, quantized proxy and does not validate residual motion. The
smallest informative extension is a deliberately larger cache for these same
two predetermined units (for example 400 events per half), with a disjoint
training-only waveform/noise bank, longer spike-free residual snippets and
precommitted median/mean estimators. That would distinguish sample-size failure
from stable spatial variation without adding anchors or fitting a field.

Outputs:
`/mnt/NPX/Luke/DARTsort_motion_experiments/cm_anchor_noise_calibration_20260927/`.
No raw voltage read, GPU work, field fit or anchor promotion occurred.
