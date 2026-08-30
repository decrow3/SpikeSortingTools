from testing.luke_depth_strip_runtime_calibration import calibration_frame_range


def test_calibration_frame_range():
    assert calibration_frame_range(10.0, 20.0, 1000.0, 40_000) == (
        10_000,
        30_000,
    )


def test_calibration_frame_range_rejects_out_of_bounds():
    for start, duration in [(-1, 1), (0, 0), (30, 20)]:
        try:
            calibration_frame_range(start, duration, 1000.0, 40_000)
        except ValueError:
            pass
        else:
            raise AssertionError("Expected invalid calibration range to fail")
