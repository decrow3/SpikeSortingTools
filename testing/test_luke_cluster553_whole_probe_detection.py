import numpy as np

from testing.luke_cluster553_whole_probe_detection import spatial_escape_summary


def test_spatial_escape_summary_separates_depth_gate_from_temporal_detection():
    result = spatial_escape_summary(
        anchor=np.array([100, 200, 300]),
        all_times=np.array([100, 201, 500]),
        all_depths=np.array([1000.0, 1300.0, 1000.0]),
        target_depth=1000.0,
        radius_um=100.0,
        tolerance=2,
    )
    assert result["spatial_matched_events"] == 1
    assert result["whole_probe_matched_events"] == 2
    assert result["spatial_escape_events"] == 1
    assert result["temporal_match_fraction_among_spatially_unmatched"] == 0.5
    assert result["escaped_depth_offset_um"]["median"] == 300.0


def test_spatial_escape_summary_handles_no_spatially_unmatched_events():
    result = spatial_escape_summary(
        anchor=np.array([100]), all_times=np.array([100]), all_depths=np.array([1000.0]),
        target_depth=1000.0, radius_um=100.0, tolerance=1,
    )
    assert result["spatially_unmatched_events"] == 0
    assert result["temporal_match_fraction_among_spatially_unmatched"] == 0.0
