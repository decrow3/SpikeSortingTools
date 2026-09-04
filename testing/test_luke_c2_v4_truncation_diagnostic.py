import numpy as np
import pytest

from testing.luke_c2_v4_truncation_diagnostic import (
    exact_count_windows,
    select_primary_phase_clusters,
    staircase_phase,
)


def test_exact_count_windows_are_exact_nonoverlapping_and_centred():
    assert exact_count_windows(249) == []
    assert exact_count_windows(250) == [(0, 250)]
    assert exact_count_windows(687) == [(93, 343), (343, 593)]
    windows = exact_count_windows(1001)
    assert all(stop - start == 250 for start, stop in windows)
    assert all(a_stop == b_start for (_, a_stop), (b_start, _) in zip(windows, windows[1:]))
    assert windows[0][0] == (1001 - 1000) // 2


def test_staircase_phase_uses_four_hard_plateaus():
    fs = 1000.0
    samples = np.array([0, 29999, 30000, 59999, 60000, 89999, 90000, 119999])
    assert staircase_phase(samples, fs).tolist() == [0, 0, 40, 40, 0, 0, 40, 40]


def test_staircase_phase_clips_the_recording_endpoint():
    assert staircase_phase(np.array([120000]), 1000.0).tolist() == [40.0]


def test_small_window_calibration_is_deterministic(monkeypatch):
    # Exercise the orchestration cheaply; the full run uses 100 replicates.
    from testing.luke_c2_v4_truncation_diagnostic import calibrate_small_window

    first = calibrate_small_window(window_size=250, replicates=3, seed=12)
    second = calibrate_small_window(window_size=250, replicates=3, seed=12)
    assert first.equals(second)
    assert first.true_missing_pct.tolist() == pytest.approx(
        [0.5, 1, 2, 5, 10, 20, 30, 40]
    )
    assert np.isfinite(first.median_estimated_pct).all()


def test_primary_phase_selection_does_not_overweight_extra_fragments():
    import pandas as pd

    frame = pd.DataFrame([
        {"template": "D01", "arm": "moved", "sorter": "rescue", "phase_um": 0.0,
         "output_cluster": 8, "tp_phase": 200, "precision_pct": 100.0,
         "n_output_spikes_phase": 200, "median_estimated_missing_pct": np.nan},
        {"template": "D01", "arm": "moved", "sorter": "rescue", "phase_um": 0.0,
         "output_cluster": 9, "tp_phase": 300, "precision_pct": 90.0,
         "n_output_spikes_phase": 333, "median_estimated_missing_pct": 12.0},
    ])
    selected = select_primary_phase_clusters(frame)
    assert len(selected) == 1
    assert selected.iloc[0].output_cluster == 9


def test_primary_phase_selection_does_not_splice_non_null_columns():
    import pandas as pd

    frame = pd.DataFrame([
        {"template": "D01", "arm": "moved", "sorter": "rescue", "phase_um": 40.0,
         "output_cluster": 8, "tp_phase": 300, "precision_pct": 100.0,
         "n_output_spikes_phase": 200, "median_estimated_missing_pct": np.nan},
        {"template": "D01", "arm": "moved", "sorter": "rescue", "phase_um": 40.0,
         "output_cluster": 9, "tp_phase": 200, "precision_pct": 100.0,
         "n_output_spikes_phase": 300, "median_estimated_missing_pct": 12.0},
    ])
    selected = select_primary_phase_clusters(frame)
    assert selected.iloc[0].output_cluster == 8
    assert np.isnan(selected.iloc[0].median_estimated_missing_pct)
