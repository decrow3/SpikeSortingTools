import numpy as np

from testing import luke_kilosort_premerge_lighthouse_v5 as premerge


def test_wall_reconstruction_matches_expected_axes():
    wall = np.zeros((2, 3, 2), dtype=np.float32)
    wall[0, 1] = [2, 3]
    basis = np.array([[1, 0, -1], [0, 2, 0]], dtype=np.float32)
    templates = premerge.reconstruct_temporal_templates(wall, basis)
    assert templates.shape == (2, 3, 3)
    np.testing.assert_array_equal(templates[0, :, 1], [2, 6, -2])


def test_maximum_spanning_tree_bottleneck_allows_coherent_chain():
    similarity = np.array([
        [1.00, 0.99, 0.20, 0.10],
        [0.99, 1.00, 0.98, 0.30],
        [0.20, 0.98, 1.00, 0.97],
        [0.10, 0.30, 0.97, 1.00],
    ])
    assert premerge.maximum_spanning_tree_bottleneck(similarity) == 0.97


def test_static_merge_fate_detects_collapsed_and_preserved_members():
    import pandas as pd

    audits = pd.DataFrame([
        {"cosine_threshold": 0.97, "threshold_family_id": "c097_F001", "member_cids": "0 1",
         "depth_time_class": "depth_time_coherent"},
        {"cosine_threshold": 0.97, "threshold_family_id": "c097_F002", "member_cids": "2 3",
         "depth_time_class": "depth_time_coherent"},
    ])
    fate = premerge.static_merge_fate(
        audits, np.array([0, 0, 1, 1, 2, 3]), np.array([5, 5, 5, 5, 6, 7])
    ).set_index("threshold_family_id")
    assert fate.loc["c097_F001", "member_pairs_collapsed_by_static_merge"] == 1
    assert not bool(fate.loc["c097_F001", "all_members_remain_separate_after_static_merge"])
    assert bool(fate.loc["c097_F002", "all_members_remain_separate_after_static_merge"])
