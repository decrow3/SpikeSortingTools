import numpy as np
import pandas as pd

from testing.luke_imec1_dots_lighthouse_direct_check import (
    bin_events,
    closest_rival_cosines,
    exact_np1_patch,
    unit_interval_counts,
)


def test_exact_np1_patch_preserves_relative_geometry():
    geom = np.column_stack(
        [np.tile([16.0, 48.0, 0.0, 32.0], 12), np.repeat(np.arange(0.0, 480.0, 20.0), 2)]
    )
    first = exact_np1_patch(geom, 180.0)
    second = exact_np1_patch(geom, 260.0)
    assert first is not None and second is not None
    base_a, channels_a = first
    base_b, channels_b = second
    np.testing.assert_array_equal(
        geom[channels_a] - [0.0, base_a], geom[channels_b] - [0.0, base_b]
    )


def test_closest_rival_cosines_excludes_self():
    waveforms = np.eye(3, dtype=np.float32)
    waveforms[1] = np.array([0.8, 0.6, 0.0], dtype=np.float32)
    cosine, rival = closest_rival_cosines(waveforms, block=2)
    assert rival[0] == 1
    assert np.isclose(cosine[0], 0.8)
    assert np.all(rival != np.arange(3))


def test_counts_and_binning_use_half_open_intervals():
    labels = np.array([1, 1, 2, 2])
    times = np.array([0.0, 1.0, 1.999, 2.0])
    counts = unit_interval_counts(labels, times, (1.0, 2.0))
    assert counts.to_dict() == {1: 1, 2: 1}
    events = pd.DataFrame(
        {
            "unit_id": [1, 1, 1],
            "time_bin_s": [1.25, 1.25, 1.75],
            "relative_depth_um": [0.0, 2.0, 4.0],
            "seed_depth_um": [100.0, 100.0, 100.0],
        }
    )
    binned = bin_events(events)
    assert binned.event_count.tolist() == [2, 1]
    assert binned.observed_relative_um.tolist() == [1.0, 4.0]
