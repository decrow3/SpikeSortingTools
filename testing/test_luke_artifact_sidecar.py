import numpy as np

from testing.luke_artifact_sidecar import threshold_points


def test_threshold_points_uses_strict_bilateral_threshold_and_excludes_191_from_claim_samples():
    traces = np.array(
        [
            [0, 214, 0],
            [-214, 0, 0],
            [0, -213, 214],
        ],
        dtype=np.int16,
    )
    samples, channels, values, claim_samples = threshold_points(
        traces,
        start_frame=100,
        channel_ids=np.array([190, 191, 192]),
        threshold_counts=213.33333333333334,
    )
    assert samples.tolist() == [100, 101, 102]
    assert channels.tolist() == [191, 190, 192]
    assert values.tolist() == [214, -214, 214]
    assert claim_samples.tolist() == [101, 102]


def test_threshold_points_rejects_incompatible_shapes():
    try:
        threshold_points(np.zeros((3, 2)), 0, np.array([1, 2, 3]))
    except ValueError as error:
        assert "incompatible" in str(error)
    else:
        raise AssertionError("Expected incompatible channel ids to fail")
