import numpy as np

from testing.luke_truncation_window_feasibility import exact_windows, fit_exact_window


def test_exact_windows_are_half_open_and_consume_exact_counts():
    windows = exact_windows(1001, 250)
    assert windows == [(0, 250), (250, 500), (500, 750), (750, 1000)]
    assert all(stop - start == 250 for start, stop in windows)


def test_fit_exact_window_uses_all_supplied_samples():
    rng = np.random.default_rng(123)
    amplitudes = rng.lognormal(mean=3.0, sigma=0.4, size=250)
    result = fit_exact_window(amplitudes)
    assert set(result) == {"estimate_pct", "saturated", "fallback"}
    assert np.isfinite(result["estimate_pct"])