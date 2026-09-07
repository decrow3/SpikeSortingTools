import numpy as np
import pytest

from pipeline.config import fingerprint
from testing.development_strip import classify_unit_depths, select_depth_channels


class FakeRecording:
    def __init__(self):
        self.ids = np.array(["c7", "c2", "c9", "c1", "c4"], dtype=object)
        self.locations = np.array(
            [[0, 0], [32, 100], [0, 200], [32, 300], [0, 400]], dtype=float
        )

    def get_channel_ids(self):
        return self.ids

    def get_channel_locations(self):
        return self.locations


def test_physical_selection_preserves_ids_geometry_and_halo():
    selected = select_depth_channels(
        FakeRecording(), processing_depth_um=[50, 350], scoring_depth_um=[150, 250]
    )
    assert selected["processing_channel_ids"] == ["c2", "c9", "c1"]
    assert selected["interior_channel_ids"] == ["c9"]
    assert selected["halo_channel_ids"] == ["c2", "c1"]
    assert selected["processing_channel_locations_um"] == [[32.0, 100.0], [0.0, 200.0], [32.0, 300.0]]


@pytest.mark.parametrize(
    "processing, scoring, match",
    [([500, 600], [525, 575], "no channels"), ([0, 400], [410, 420], "strictly inside")],
)
def test_invalid_or_out_of_range_selection_is_refused(processing, scoring, match):
    with pytest.raises(ValueError, match=match):
        select_depth_channels(
            FakeRecording(), processing_depth_um=processing, scoring_depth_um=scoring
        )


def test_interior_scoring_keeps_edges_as_diagnostics():
    labels = classify_unit_depths(
        np.array([0, 25, 100, 200, 300, 375, 500]),
        processing_depth_um=[0, 400],
        scoring_depth_um=[100, 300],
        minimum_edge_exclusion_um=50,
    )
    assert labels.tolist() == ["edge", "edge", "interior", "interior", "interior", "edge", "outside"]


def test_strip_request_digest_changes_with_source_or_geometry():
    base = {"source_request": "a", "geometry": [[0, 100]], "depth": [0, 200]}
    assert fingerprint(base) != fingerprint({**base, "source_request": "b"})
    assert fingerprint(base) != fingerprint({**base, "geometry": [[0, 101]]})

