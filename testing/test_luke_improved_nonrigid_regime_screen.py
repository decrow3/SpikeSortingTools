import numpy as np

from testing.luke_improved_nonrigid_regime_screen import _regime_metrics, selected_trains


def test_selected_trains_preserves_chronological_order():
    times = np.array([1, 2, 3, 4, 5, 6])
    clusters = np.array([4, 2, 4, 3, 2, 4])
    trains = selected_trains(times, clusters, np.array([2, 4]))
    assert trains[2].tolist() == [2, 5]
    assert trains[4].tolist() == [1, 3, 6]


def test_regime_metrics_uses_exclusive_matches_and_within_interval_isis():
    baseline = np.array([10, 20, 30, 110, 120])
    candidate = np.array([10, 21, 40, 110, 121, 122])
    result = _regime_metrics(
        baseline, candidate, np.array([[0, 100], [100, 200]]),
        tolerance=1, refractory=5,
    )
    assert result["baseline_events"] == 5
    assert result["candidate_events"] == 6
    assert result["matched_events"] == 4
    assert result["baseline_retention"] == 0.8
    assert result["candidate_retention"] == 4 / 6
    assert result["candidate_refractory_fraction"] > result["baseline_refractory_fraction"]
