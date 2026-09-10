import numpy as np

from pipeline.motion_merge_gates import (
    dominant_final_cluster_mapping,
    endpoint_rival_score,
    family_support,
)


def test_dominant_mapping_uses_aligned_events():
    result = dominant_final_cluster_mapping(
        np.array([2, 2, 2, 3]), np.array([8, 8, 9, 7])
    )
    assert result[2].final_cluster == 8
    assert result[2].matched_spikes == 2
    assert np.isclose(result[2].purity, 2 / 3)


def test_rival_excludes_all_states_of_intended_cluster():
    rows = [
        {"cluster_a": 1, "cluster_b": 2, "state_a": 0, "state_b": 0, "score": .9},
        {"cluster_a": 1, "cluster_b": 2, "state_a": 0, "state_b": 1, "score": .95},
        {"cluster_a": 1, "cluster_b": 3, "state_a": 0, "state_b": -1, "score": .8},
    ]
    assert endpoint_rival_score(rows, 1, 0, 2, 0) == (.8, 3, -1)


def test_family_support_is_conservative():
    membership = {4: "F1", 5: "F1", 6: "F2"}
    classes = {"F1": "depth_time_coherent", "F2": "implausible"}
    assert family_support(4, 5, membership, classes)[0]
    assert not family_support(4, 6, membership, classes)[0]
    assert not family_support(4, 99, membership, classes)[0]
