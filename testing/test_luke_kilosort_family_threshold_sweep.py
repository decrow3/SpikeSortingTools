import numpy as np
import pandas as pd

from testing import luke_kilosort_family_threshold_sweep_v4 as sweep


def test_threshold_tag_is_unambiguous():
    assert sweep.threshold_tag(0.95) == "c095"
    assert sweep.threshold_tag(0.97) == "c097"
    assert sweep.threshold_tag(0.98) == "c098"


def test_parent_mapping_tracks_split_components():
    table = pd.DataFrame([
        {"cosine_threshold": 0.95, "threshold_family_id": "c095_F001", "member_signature": "1 2 3"},
        {"cosine_threshold": 0.95, "threshold_family_id": "c095_F002", "member_signature": "7 8"},
        {"cosine_threshold": 0.97, "threshold_family_id": "c097_F001", "member_signature": "1 2"},
        {"cosine_threshold": 0.97, "threshold_family_id": "c097_F002", "member_signature": "7 8"},
    ])
    mapped = sweep.attach_baseline_parent(table)
    split = mapped[mapped["threshold_family_id"] == "c097_F001"].iloc[0]
    intact = mapped[mapped["threshold_family_id"] == "c097_F002"].iloc[0]
    assert split["baseline_0p95_parent"] == "c095_F001"
    assert split["baseline_member_retention_fraction"] == 2 / 3
    assert intact["baseline_0p95_parent"] == "c095_F002"
    assert intact["baseline_member_retention_fraction"] == 1


def test_cross_cid_join_metrics_ignores_same_cid_and_long_gaps():
    members = pd.DataFrame({"family_id": ["F001", "F001"], "cid": [1, 2]})
    frames = np.array([0, 10, 20, 1000])
    cids = np.array([1, 1, 2, 1])
    depths = np.array([100, 101, 150, 100], float)
    result = sweep.cross_cid_join_metrics(members, frames, cids, depths, fs=100.0).iloc[0]
    assert result["cross_cid_joined_intervals"] == 1
    assert result["cross_cid_depth_jump_median_um"] == 49
    assert result["cross_cid_depth_jump_max_um"] == 49
    assert result["cross_cid_depth_jumps_ge_100_um"] == 0
    assert result["cross_cid_depth_jumps_ge_160_um"] == 0
