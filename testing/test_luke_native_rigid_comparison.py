import numpy as np

from testing.luke_native_rigid_comparison import amplitude_cv_range, assess_efficacy


def test_amplitude_cv_range_constant_amplitudes():
    times = np.arange(25, 1000, 10, dtype=np.int64)
    amplitudes = np.full(len(times), 4.0)
    median, spread = amplitude_cv_range(
        times, amplitudes, duration_frames=1000,
        average_num_spikes_per_bin=5, min_num_bins=10,
    )
    assert median == 0.0
    assert spread == 0.0


def test_amplitude_cv_range_refuses_empty_bins_like_si():
    times = np.r_[np.arange(10, 110), np.arange(900, 1000)]
    amplitudes = np.ones(len(times))
    median, spread = amplitude_cv_range(
        times, amplitudes, duration_frames=1000,
        average_num_spikes_per_bin=10, min_num_bins=10,
    )
    assert np.isnan(median)
    assert np.isnan(spread)


def test_pure_dropout_and_perfect_rescue_cannot_close_or_promote():
    for baseline, candidate in [(100, 80), (80, 100)]:
        shared = min(baseline, candidate)
        summary = {
            "median_baseline_retention": shared / baseline,
            "median_candidate_retention": shared / candidate,
            "median_missingness_improvement_pp": 20.0,
        }
        decision = assess_efficacy(summary, {"endpoint_status": "infeasible_insufficient_coverage"})
        assert decision["status"] == "motion_diagnostics_required"
        assert decision["efficacy_pass"] is None
        assert decision["amplitude_efficacy_pass"] is None


def test_measured_amplitude_is_reported_without_automatic_promotion():
    for value, expected in [(5.0, True), (4.9, False), (float("nan"), None)]:
        decision = assess_efficacy(
            {"median_missingness_improvement_pp": value}, {"endpoint_status": "measured"}
        )
        assert decision["amplitude_efficacy_pass"] is expected
        assert decision["efficacy_pass"] is None
