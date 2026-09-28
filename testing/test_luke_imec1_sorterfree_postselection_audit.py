import numpy as np
import pandas as pd

from testing.luke_imec1_sorterfree_postselection_audit import family_audit, pairwise_replication


def test_rapid_depth_changes_flag_but_preserve_candidate():
    families = pd.DataFrame({
        "family_id": ["a"], "selected_depth_blind": [True],
        "seed_depth_median_um": [1000.0], "postmatch_seed_spatially_plausible": [True],
    })
    times = np.repeat([935.0, 974.0, 998.0], 8) + np.tile(np.arange(8) / 4, 3)
    events = pd.DataFrame({
        "family_id": "a", "status": "strict", "time_s": times,
        "waveform_centroid_um": np.where(np.arange(24) % 2, 1150.0, 1000.0),
    })
    got = family_audit(events, families).iloc[0]
    assert got.trace_status == "unreplicated_cross_depth_candidate"
    assert got.jumps_ge_100um > 0


def test_replication_requires_enough_temporal_pairs():
    families = pd.DataFrame({
        "family_id": ["a", "b"], "selected_depth_blind": [True, True],
        "seed_depth_median_um": [1000.0, 2000.0],
        "postmatch_seed_spatially_plausible": [True, True],
    })
    events = pd.DataFrame({
        "family_id": ["a"] * 5 + ["b"] * 5, "status": "strict",
        "time_s": [1, 2, 3, 4, 5, 1.1, 2.1, 3.1, 4.1, 5.1],
        "waveform_centroid_um": [1000, 1010, 1020, 1030, 1040, 2000, 2010, 2020, 2030, 2040],
    })
    got = pairwise_replication(events, families).iloc[0]
    assert bool(got.replication_evaluable)
    assert got.paired_events_le_250ms == 5
    assert got.displacement_correlation > 0.99


def test_diffuse_waveform_families_are_not_independent_replication():
    families = pd.DataFrame({
        "family_id": ["a", "b"], "selected_depth_blind": [True, True],
        "seed_depth_median_um": [1000.0, 2000.0],
        "postmatch_seed_spatially_plausible": [True, False],
    })
    events = pd.DataFrame({
        "family_id": ["a"] * 5 + ["b"] * 5, "status": "strict",
        "time_s": [1, 2, 3, 4, 5, 1.1, 2.1, 3.1, 4.1, 5.1],
        "waveform_centroid_um": [1000, 1010, 1020, 1030, 1040, 2000, 2010, 2020, 2030, 2040],
    })
    got = pairwise_replication(events, families).iloc[0]
    assert got.paired_events_le_250ms == 5
    assert not bool(got.replication_evaluable)
    assert not bool(got.replication_supported)
