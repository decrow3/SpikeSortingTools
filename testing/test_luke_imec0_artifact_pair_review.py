import numpy as np

from testing.luke_imec0_artifact_pair_review import (
    evenly_sample,
    percentile_rank,
    threshold_footprint,
)


def test_threshold_footprint_counts_channels_and_simultaneous_points():
    samples = np.array([98, 100, 100, 101, 110])
    channels = np.array([2, 2, 3, 4, 5])
    values = np.array([-220, -230, 250, 240, 300])
    assert threshold_footprint(100, samples, channels, values, 2) == (3, 2, 250, 4)


def test_even_sampling_and_percentile_rank_are_deterministic():
    np.testing.assert_array_equal(evenly_sample(np.arange(10), 3), [0, 4, 9])
    assert percentile_rank(np.array([1, 2, 3, 4]), 3) == 75
