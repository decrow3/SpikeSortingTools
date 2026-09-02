"""Validation of the amplitude-truncation estimator.

`pipeline/truncation.py` had no direct test coverage. It feeds the QC
missing-spike estimates that the post-curation evaluation used to argue the
rescue pipeline detects units less completely, so its behaviour needs to be
pinned down before any acceptance gate is built on it
(docs/decisions/0008, follow-up item 3).

These tests establish, against synthetic ground truth, both where the estimator
is trustworthy and where it silently is not.
"""

import numpy as np
import pytest

from pipeline.truncation import (
    analyze_amplitude_truncation,
    fit_amp_cdf,
    truncated_sigmoid_missing_pct,
)

X0 = 20.0
K = 0.5
CEILING = 50.0


def truncated_sample(rng, true_missing, n=1000):
    """Draw `n` amplitudes from a logistic distribution truncated so that
    `true_missing` of the full distribution lies below the cut."""
    p = true_missing
    threshold = X0 + np.log(p / (1 - p)) / K
    draw = rng.logistic(loc=X0, scale=1 / K, size=int(n / (1 - p)) + 4000)
    kept = draw[draw > threshold]
    assert len(kept) >= n, "not enough retained samples"
    return kept[:n]


@pytest.mark.parametrize("true_pct", [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0])
def test_estimator_is_unbiased_in_its_working_range(true_pct):
    """Below ~40% the estimate tracks truth to well under a percentage point.

    This is the range every KS-good cohort actually occupies (0.6-3%), so the
    reported cohort numbers are computed correctly.
    """
    rng = np.random.default_rng(4242)
    estimates = [fit_amp_cdf(truncated_sample(rng, true_pct / 100))[1] for _ in range(40)]
    assert np.median(estimates) == pytest.approx(true_pct, abs=1.0)


@pytest.mark.parametrize("true_pct", [60.0, 70.0, 80.0])
def test_estimator_is_hard_censored_at_fifty_percent(true_pct):
    """Above 50% the estimate saturates and understates truth without warning.

    `fit_truncated_sigmoid` bounds x0 below by x_min, so
    `truncated_sigmoid_missing_pct` = 100*sigmoid(x_min; x0, k) can never exceed
    50. A window reported at exactly 50.0 is a boundary-pinned fit, not a
    measurement of "50% missing", and must be treated as censored.
    """
    rng = np.random.default_rng(99)
    estimates = np.array(
        [fit_amp_cdf(truncated_sample(rng, true_pct / 100))[1] for _ in range(20)]
    )
    assert estimates.max() <= CEILING + 1e-9
    assert np.median(estimates) == pytest.approx(CEILING, abs=0.5)
    assert np.median(estimates) < true_pct - 5


def test_reported_value_can_never_exceed_the_ceiling():
    """The bound x0 >= x_min caps the statistic at 50% by construction."""
    for x0, k, x_min in [(5.0, 1.0, 5.0), (5.0, 0.1, 5.0), (5.0, 9.0, 5.0)]:
        assert truncated_sigmoid_missing_pct([x0, k, 1.0], x_min) == pytest.approx(50.0)
    # x0 above x_min is the only regime the optimiser can reach, and it is < 50.
    assert truncated_sigmoid_missing_pct([10.0, 1.0, 5.0], 5.0) < 50.0


def test_exactly_fifty_identifies_censored_windows():
    """Saturated windows are identifiable post hoc, which is how stored QC
    output must be filtered before any median is taken."""
    rng = np.random.default_rng(7)
    severe = [fit_amp_cdf(truncated_sample(rng, 0.70))[1] for _ in range(15)]
    mild = [fit_amp_cdf(truncated_sample(rng, 0.02))[1] for _ in range(15)]
    assert np.isclose(severe, CEILING).mean() > 0.5
    assert not np.isclose(mild, CEILING).any()


def test_window_construction_drops_one_spike_per_window():
    """analyze_amplitude_truncation slices [i0:i1] where i1 is inclusive-by-
    intent, so each window fits 999 of its 1000 spikes. Harmless at this size
    but pinned here so a future change is deliberate."""
    rng = np.random.default_rng(3)
    n = 2000
    times = np.sort(rng.uniform(0, 500, n))
    amps = rng.logistic(20, 2, n)
    window_blocks, _, _, mpcts = analyze_amplitude_truncation(times, amps)
    assert len(window_blocks) == 2
    i0, i1 = window_blocks[0]
    assert i1 - i0 == 999
    assert len(amps[i0:i1]) == 999
    assert np.all(np.isfinite(mpcts))


def test_low_rate_units_are_silently_ineligible():
    """A unit with fewer than 1000 spikes in any continuous block yields no
    windows at all. Eligibility is therefore correlated with firing rate, and
    cohort statistics are conditioned on it."""
    rng = np.random.default_rng(11)
    times = np.sort(rng.uniform(0, 900, 400))
    amps = rng.logistic(20, 2, 400)
    window_blocks, _, _, mpcts = analyze_amplitude_truncation(times, amps)
    assert len(window_blocks) == 0
    assert len(mpcts) == 0


def test_estimate_is_invariant_to_amplitude_scale():
    """Missingness depends on k*(x0 - x_min), so a pure rescaling of amplitudes
    must not change it. This is what makes cross-sort comparison arithmetically
    valid even when whitening differs -- the confound is unit composition, not
    amplitude units."""
    rng = np.random.default_rng(5)
    amps = truncated_sample(rng, 0.05)
    _, base = fit_amp_cdf(amps)
    _, scaled = fit_amp_cdf(amps * 7.5)
    assert scaled == pytest.approx(base, abs=0.5)


# ---------------------------------------------------------------------------
# Interpretation helpers
# ---------------------------------------------------------------------------

from pipeline.truncation import (  # noqa: E402
    SATURATION_PCT,
    is_saturated,
    missing_pct_from_normalisation,
)


def test_is_saturated_flags_only_boundary_pinned_windows():
    mpcts = np.array([0.0, 1.5, 12.0, 49.99, 50.0, 50.0])
    assert is_saturated(mpcts).tolist() == [False, False, False, False, True, True]


def test_missing_pct_from_normalisation_inverts_the_renormalisation():
    # A = 1/(1-F) so F = 1 - 1/A. A=2 -> 50%, A=1 -> 0%, A=4 -> 75%.
    got = missing_pct_from_normalisation([[10.0, 1.0, 2.0], [10.0, 1.0, 1.0], [10.0, 1.0, 4.0]])
    assert got == pytest.approx([50.0, 0.0, 75.0])


def test_the_two_estimates_agree_when_the_model_holds():
    """On well-behaved synthetic data the discarded A-estimate reproduces the
    reported statistic, which is what makes it usable as a fit-quality check."""
    rng = np.random.default_rng(17)
    for true_pct in (2.0, 10.0, 25.0):
        amps = truncated_sample(rng, true_pct / 100)
        popt, reported = fit_amp_cdf(amps)
        from_a = float(missing_pct_from_normalisation([popt])[0])
        assert from_a == pytest.approx(reported, abs=3.0)


def test_censored_windows_bias_a_median_when_not_filtered():
    """Averaging saturated windows in as if they were estimates understates
    severity; filtering them changes the answer. This is why is_saturated exists."""
    mpcts = np.array([1.0, 2.0, 3.0, SATURATION_PCT, SATURATION_PCT, SATURATION_PCT])
    assert np.median(mpcts) == pytest.approx(26.5)
    assert np.median(mpcts[~is_saturated(mpcts)]) == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# The methodological guard: population comparison is confounded by composition
# ---------------------------------------------------------------------------

def test_population_medians_differ_even_when_per_unit_behaviour_is_identical():
    """Encodes the error found on Luke0804 imec0.

    Two configurations whose shared units behave *identically* still produce
    very different population medians when one of them additionally recovers
    small-amplitude units. Estimated missingness depends strongly on unit
    amplitude, so comparing whole-population medians across sorts measures
    which units each sort admits, not how completely it detects them.

    On the real data: rescue and legacy scored 0.63% on all 43 shared neurons,
    yet the reported population medians were 3.07% and 1.16%, purely because
    rescue additionally found 50 smaller units at 9.45%.

    A cross-sort comparison must therefore be matched on units before any
    conclusion about detection quality is drawn.
    """
    shared = np.full(43, 0.63)
    rescue_only = np.full(50, 9.45)
    legacy_only = np.full(25, 3.01)

    rescue_pop = np.concatenate([shared, rescue_only])
    legacy_pop = np.concatenate([shared, legacy_only])

    # Population comparison: rescue looks far worse.
    assert np.median(rescue_pop) > 3 * np.median(legacy_pop)

    # Matched comparison on the shared units: no difference whatsoever.
    assert np.median(shared) == pytest.approx(np.median(shared))
    paired_difference = shared - shared
    assert np.abs(paired_difference).max() == 0.0
