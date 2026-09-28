import numpy as np

from testing.co_exposure_qualification import expected_full_support


def test_segment_requires_second_common_positive_block():
    fs = 100.0
    one = [{"segment_id": 0, "start_sample": 0, "end_sample": 500}]
    assert expected_full_support(one, 1.0, 1.0, 1.0, fs)[
        "expected_common_positive_blocks"] == 0.0

    two = one + [{"segment_id": 0, "start_sample": 500, "end_sample": 1000}]
    q = (1.0 - np.exp(-5.0)) ** 2
    got = expected_full_support(two, 1.0, 1.0, 1.0, fs)
    assert np.isclose(got["expected_common_positive_blocks"], 2 * q * q)
    assert got["expected_scoring_events_i"] > 0


def test_separate_single_block_segments_contribute_nothing():
    blocks = [
        {"segment_id": 0, "start_sample": 0, "end_sample": 500},
        {"segment_id": 1, "start_sample": 500, "end_sample": 1000},
    ]
    got = expected_full_support(blocks, 2.0, 3.0, 1.0, 100.0)
    assert got["expected_common_positive_blocks"] == 0.0
    assert got["expected_scoring_events_i"] == 0.0
    assert got["expected_independent_central"] == 0.0
