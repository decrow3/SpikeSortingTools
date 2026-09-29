import numpy as np

from testing.luke_cluster553_artifact_proximity import (
    artifact_supported,
    nearest_distance_frames,
)


def test_nearest_distance_handles_edges():
    assert nearest_distance_frames(np.array([0, 5, 20]), np.array([3, 8, 30])).tolist() == [3, 2, 10]


def test_artifact_support_requires_absolute_or_enriched_fraction():
    rule = {"minimum_absolute_fraction": 0.5, "minimum_increase_over_failing_matched": 0.02,
            "minimum_ratio_over_failing_matched": 2.0}
    assert artifact_supported(0.6, 0.4, rule)
    assert artifact_supported(0.05, 0.01, rule)
    assert not artifact_supported(0.03, 0.02, rule)
