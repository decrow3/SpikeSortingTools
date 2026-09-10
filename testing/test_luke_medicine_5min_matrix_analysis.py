import numpy as np

from testing.luke_medicine_5min_matrix_analysis import match_events


def test_match_events_is_one_to_one_within_identity_and_reports_duplicates():
    result = match_events(
        np.array([100, 102]), np.array([200.0, 200.0]),
        np.array([101, 103]), np.array([200.0, 205.0]), np.array([4, 5]),
        tolerance_samples=2, depth_tolerance_um=20.0,
    )
    assert result.recovered.tolist() == [True, True]
    assert result.cluster.tolist() == [4, 5]
    assert result.nearby_clusters.tolist() == [1, 2]


def test_match_events_enforces_depth_gate():
    result = match_events(
        np.array([100]), np.array([200.0]), np.array([100]), np.array([400.0]), np.array([4]),
        tolerance_samples=2, depth_tolerance_um=100.0,
    )
    assert not bool(result.loc[0, "recovered"])
