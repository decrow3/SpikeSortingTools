import numpy as np
import pandas as pd

from testing.luke_imec1_medicine_stage2_o_v1 import (
    CONFIGS, HELDOUT_WINDOWS, corrected_quiet_values, heldout_overlap_check, selection,
)


def test_heldout_windows_are_120s_and_do_not_overlap_stage1():
    audit = heldout_overlap_check()
    assert all(abs(stop - start - 120) < 1e-9 for start, stop in HELDOUT_WINDOWS.values())
    assert all(not row["overlapping_stage1_windows"] and row["adjustment_s"] == 0
               for row in audit.values())


def test_corrected_quiet_increment_excludes_expanded_episode_endpoints():
    field = {"session_time_s": np.arange(0, 21, 1.0), "depth_um": np.array([100.0]),
             "displacement_um": np.arange(0, 21, dtype=float)[:, None]}
    candidates = pd.DataFrame({"start_s": [9.0], "stop_s": [11.0]})
    values, pairs = corrected_quiet_values(field, candidates, (0.0, 20.0))
    # Endpoints are 0,5,10,15,20; 10 is excluded by episode +/-2s.
    assert pairs == 2
    assert np.allclose(values, [5.0, 5.0])


def test_selection_uses_block_gate_then_simplicity_tie_break():
    table = pd.DataFrame([
        {"config": "amp50", "episode_err": 10.0, "episode_ratio": 1.0,
         "block_err": 10.0, "quiet_abs": 1.0, "quiet_inc": 1.0},
        {"config": "amp50_d2", "episode_err": 12.0, "episode_ratio": 1.0,
         "block_err": 10.5, "quiet_abs": 1.0, "quiet_inc": 1.0},
        {"config": "amp50_d1", "episode_err": 13.0, "episode_ratio": 1.0,
         "block_err": 11.0, "quiet_abs": 1.0, "quiet_inc": 1.0},
        {"config": "amp25", "episode_err": 9.0, "episode_ratio": 1.0,
         "block_err": 20.0, "quiet_abs": 1.0, "quiet_inc": 1.0},
    ])
    got = selection(table)
    assert got["config"] == "amp50_d1"
    assert set(got["eligible"]) == {"amp50", "amp50_d2", "amp50_d1"}
    assert len(CONFIGS) == 8
