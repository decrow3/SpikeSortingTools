import numpy as np

from testing.luke_trace_reviewed_events import local_match_details


def test_local_match_requires_both_time_and_depth_proximity():
    result = local_match_details(
        event_samples=np.array([100, 200, 300]),
        event_depths=np.array([1000.0, 1000.0, 1000.0]),
        spike_times=np.array([99, 202, 400]),
        spike_depths=np.array([1020.0, 1300.0, 1000.0]),
        tolerance_samples=3,
        depth_tolerance_um=100.0,
    )
    assert result["present"].tolist() == [True, False, False]
    assert result["n_temporal_candidates"].tolist() == [1, 1, 0]
    assert result.loc[1, "nearest_depth_error_um"] == 300.0


def test_local_match_selects_nearest_time_among_local_spikes():
    result = local_match_details(
        event_samples=np.array([100]),
        event_depths=np.array([500.0]),
        spike_times=np.array([97, 101, 102]),
        spike_depths=np.array([500.0, 520.0, 900.0]),
        tolerance_samples=4,
        depth_tolerance_um=100.0,
    )
    assert bool(result.loc[0, "present"])
    assert result.loc[0, "nearest_time_samples"] == 1
