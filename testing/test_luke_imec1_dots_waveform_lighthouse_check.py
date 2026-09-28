import numpy as np

from testing.luke_imec1_dots_waveform_lighthouse_check import (
    channel_signature_ids,
    track_plausibility,
    waveform_centroids,
)


def test_channel_signatures_ignore_absolute_vertical_position():
    geom = np.array([[0, 0], [0, 40], [0, 80], [0, 120]], dtype=float)
    channel_index = np.array([[0, 1], [1, 2], [2, 3], [3, 4]])
    signatures = channel_signature_ids(geom, channel_index)
    assert signatures[0] == signatures[1] == signatures[2]


def test_waveform_centroid_moves_with_translated_support():
    geom = np.array([[0, 0], [0, 40], [0, 80], [0, 120]], dtype=float)
    channel_index = np.array([[0, 1], [1, 2], [2, 3], [3, 4]])
    gram = np.eye(1)
    features = np.array([[[1.0, 2.0]], [[1.0, 2.0]]])
    centroids = waveform_centroids(
        features, np.array([0, 1]), channel_index, geom, gram
    )
    assert np.isclose(centroids[1] - centroids[0], 40.0)


def test_track_plausibility_is_post_match_depth_sanity_check():
    import pandas as pd

    recovery = pd.DataFrame({"unit_id": [1, 2], "seed_qualified": [True, True]})
    events = pd.DataFrame(
        {
            "unit_id": [1] * 20 + [2] * 20,
            "status": ["strict"] * 40,
            "seed": [False] * 40,
            "waveform_centroid_um": list(np.linspace(100, 200, 20))
            + list(np.linspace(100, 800, 20)),
        }
    )
    result = track_plausibility(events, recovery).set_index("unit_id")
    assert bool(result.loc[1, "provisional_track_plausible"])
    assert bool(result.loc[2, "implausible_gt500um_span"])
    assert not bool(result.loc[2, "provisional_track_plausible"])
