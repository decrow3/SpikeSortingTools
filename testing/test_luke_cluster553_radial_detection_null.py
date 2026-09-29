import numpy as np

from testing.luke_cluster553_radial_detection_null import circular_shift, radius_rows


def test_circular_shift_stays_inside_window():
    shifted = circular_shift(np.array([101, 108]), 100, 110, 5)
    assert shifted.tolist() == [106, 103]


def test_radius_rows_reports_observed_and_null_gains():
    rows = radius_rows(
        anchor=np.array([100, 200]),
        times=np.array([100, 200, 350]),
        depths=np.array([1000.0, 1150.0, 1000.0]),
        target_depth=1000.0,
        radii=[100.0, 200.0], tolerance=2, shifts=[50], start=0, stop=400,
    )
    assert rows[0]["observed_matched_fraction"] == 0.5
    assert rows[1]["observed_matched_fraction"] == 1.0
    assert rows[1]["observed_gain_over_100um"] == 0.5
    assert rows[1]["gain_above_max_null_gain"] == 0.5
