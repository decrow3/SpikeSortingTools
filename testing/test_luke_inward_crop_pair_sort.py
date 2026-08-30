import numpy as np

from testing import luke_inward_crop_pair_sort as pair
from testing.luke_two_axis_pilot import pilot_channel_ids


def test_inward_crop_is_80_channels_with_eight_channel_margin():
    np.testing.assert_array_equal(pair.RETAINED_IDS, np.arange(184, 264))
    np.testing.assert_array_equal(pilot_channel_ids(pair.PILOT), pair.RETAINED_IDS)


def test_condition_paths_are_distinct_and_claim_off():
    assert pair.sort_path("no_motion") != pair.sort_path("rigid025_p2")
    assert pair.CLAIM_OFF.claim_ms == 0.0
    assert pair.CLAIM_OFF.claim_um == 0.0
    assert pair.cropped_recording_path("no_motion") != pair.cropped_recording_path("rigid025_p2")
