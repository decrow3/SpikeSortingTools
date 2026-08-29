import numpy as np
import pandas as pd

from testing.luke_claimmask_window_sweep import (
    CLAIM_GRID,
    Window,
    cross_unit_near_coincident_fraction,
    event_local_samples,
    events_in_window,
    local_match_mask,
    window_frames,
)


def test_claim_grid_is_the_prespecified_six_point_sweep():
    assert [(x.claim_ms, x.claim_um) for x in CLAIM_GRID] == [
        (0.0, 0.0),
        (0.10, 25.0),
        (0.10, 50.0),
        (0.25, 25.0),
        (0.25, 50.0),
        (0.25, 75.0),
    ]


def test_event_mapping_uses_original_window_start():
    window = Window("test", 10.0, 2.0)
    assert window_frames(window, 30_000.0) == (300_000, 360_000)
    np.testing.assert_array_equal(
        event_local_samples(np.array([300_000, 300_123]), window, 30_000.0),
        [0, 123],
    )


def test_window_selection_is_left_closed_right_open():
    events = pd.DataFrame({"time_seconds": [9.9, 10.0, 11.999, 12.0]})
    selected = events_in_window(events, Window("test", 10.0, 2.0))
    assert selected["time_seconds"].tolist() == [10.0, 11.999]


def test_local_match_requires_time_and_depth():
    result = local_match_mask(
        np.array([100, 200, 300]),
        np.array([1000.0, 2000.0, 3000.0]),
        np.array([103, 202, 400]),
        np.array([1050.0, 2500.0, 3000.0]),
        time_tolerance=5,
        depth_tolerance_um=100.0,
    )
    np.testing.assert_array_equal(result, [True, False, False])


def test_cross_unit_near_coincident_fraction_marks_both_spikes():
    fraction = cross_unit_near_coincident_fraction(
        times=np.array([100, 103, 200, 202, 400]),
        clusters=np.array([1, 2, 1, 1, 3]),
        depths=np.array([1000.0, 1050.0, 1000.0, 1020.0, 1000.0]),
        time_tolerance=5,
    )
    assert fraction == 2 / 5
