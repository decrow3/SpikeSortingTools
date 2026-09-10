import numpy as np

from testing import luke_premerge_prior_lighthouse_reconciliation_v1 as audit


def test_map_events_prefers_time_then_depth():
    result = audit.map_events_to_premerge(
        target_frames=np.array([100]),
        target_depth_um=np.array([205.0]),
        pre_frames=np.array([99, 100, 100, 101]),
        pre_depth_um=np.array([205.0, 100.0, 210.0, 205.0]),
        pre_clu=np.array([1, 2, 3, 4]),
    ).iloc[0]
    assert result["matched"]
    assert result["sample_error"] == 0
    assert result["premerge_cluster"] == 3
    assert result["candidate_detections"] == 2


def test_categorize_keeps_singletons_visible_as_a_distinct_outcome():
    lookup = {7: ("c097_F001", "depth_time_coherent"), 8: ("c097_F002", "implausible")}
    assert "singleton" in audit.categorize(6, lookup)
    assert "coherent" in audit.categorize(7, lookup)
    assert "implausible" in audit.categorize(8, lookup)

