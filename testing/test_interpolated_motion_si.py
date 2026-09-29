import numpy as np
from spikeinterface.core import NumpyRecording
from spikeinterface.preprocessing import get_spatial_interpolation_kernel

from npx_preprocessing.motion.interpolated_motion_si import (
    InterpolatedMotionRecording,
    bad_channel_insertion,
)


def fixture():
    full_ids = ["AP190", "AP191", "AP192"]
    parent_ids = ["AP190", "AP192"]
    locations = np.asarray([[0.0, 0.0], [0.0, 20.0], [0.0, 40.0]])
    traces = np.asarray([[1, 5], [2, 6], [3, 7], [4, 8]], dtype=np.float32)
    recording = NumpyRecording([traces], sampling_frequency=4.0, channel_ids=parent_ids)
    recording.set_dummy_probe_from_locations(locations[[0, 2]])
    return recording, full_ids, locations


def test_bad_channel_insertion_preserves_good_channels_and_interpolates_hole():
    recording, full_ids, locations = fixture()
    insertion, weights = bad_channel_insertion(
        recording.get_channel_ids(),
        full_ids,
        recording.get_channel_locations(),
        locations,
        bad_channel_id="AP191",
        sigma_um=20.0,
    )

    np.testing.assert_array_equal(insertion[:, [0, 2]], np.eye(2, dtype=np.float32))
    np.testing.assert_allclose(insertion[:, 1], weights)
    np.testing.assert_allclose(weights.sum(), 1.0, rtol=0, atol=1e-6)


def test_recording_matches_composed_si_kernel_and_stepwise_boundary():
    recording, full_ids, locations = fixture()
    corrected = InterpolatedMotionRecording(
        recording,
        np.asarray([0.0, 0.5, 1.0]),
        np.asarray([0.0, 40.0, 40.0]),
        full_channel_ids=full_ids,
        full_channel_locations=locations,
        target_channel_ids=full_ids,
        bad_channel_id="AP191",
        cell_width_s=0.5,
        method="kriging",
        sigma_um=20.0,
        p=1.0,
        time_origins_s=[0.0],
    )
    insertion, _ = bad_channel_insertion(
        recording.get_channel_ids(),
        full_ids,
        recording.get_channel_locations(),
        locations,
        bad_channel_id="AP191",
        sigma_um=20.0,
    )
    expected = np.empty((4, 3), dtype=np.float32)
    # With fs=4 and width=.5, t=.25 is the exact boundary and uses state +40.
    states = [0.0, 40.0, 40.0, 40.0]
    source = recording.get_traces()
    for frame, state in enumerate(states):
        moved = locations.copy()
        moved[:, 1] += state
        kernel = get_spatial_interpolation_kernel(
            locations,
            moved,
            method="kriging",
            sigma_um=20.0,
            p=1.0,
            sparse_thresh=None,
            dtype="float32",
            force_extrapolate=False,
        )
        expected[frame] = source[frame] @ insertion @ kernel

    np.testing.assert_allclose(corrected.get_traces(), expected, rtol=0, atol=1e-6)
    np.testing.assert_array_equal(corrected.get_channel_ids(), full_ids)
    assert corrected.get_traces().dtype == np.float32

    assert corrected._kwargs["method"] == "kriging"
    assert corrected._kwargs["recording"] is recording


def test_nearest_outer_border_is_zero_filled():
    recording, full_ids, locations = fixture()
    corrected = InterpolatedMotionRecording(
        recording,
        [0.0, 0.5, 1.0],
        [0.0, 40.0, 40.0],
        full_channel_ids=full_ids,
        full_channel_locations=locations,
        target_channel_ids=full_ids,
        bad_channel_id="AP191",
        cell_width_s=0.5,
        method="nearest",
        time_origins_s=[0.0],
    )

    # The last target at +40 requests y=80 after the boundary, beyond support.
    assert corrected.get_traces(start_frame=1, end_frame=4)[:, -1].tolist() == [0.0, 0.0, 0.0]


def test_supported_target_crop_preserves_attached_probe_metadata():
    recording, full_ids, locations = fixture()
    corrected = InterpolatedMotionRecording(
        recording,
        [0.0, 0.5, 1.0],
        [0.0, 0.0, 0.0],
        full_channel_ids=full_ids,
        full_channel_locations=locations,
        target_channel_ids=["AP192"],
        bad_channel_id="AP191",
        cell_width_s=0.5,
        time_origins_s=[0.0],
    )

    np.testing.assert_array_equal(corrected.get_channel_ids(), ["AP192"])
    np.testing.assert_array_equal(corrected.get_channel_locations(), locations[[2]])
    assert corrected.get_property("contact_vector") is not None
