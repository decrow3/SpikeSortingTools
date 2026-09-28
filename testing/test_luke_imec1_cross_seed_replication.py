import numpy as np
import pandas as pd

from testing.luke_imec1_cross_seed_replication_v1 import reciprocal_pairs, shift_within_windows


def test_reciprocal_pairs_use_each_event_at_most_once():
    a = pd.DataFrame({"time_s": [1.0, 1.1, 2.0], "relative_um": [1, 2, 3]})
    b = pd.DataFrame({"time_s": [1.04, 2.04], "relative_um": [4, 5]})
    pairs = reciprocal_pairs(a, b, .1)
    assert len(pairs) == 2
    assert pairs.time_a.is_unique and pairs.time_b.is_unique


def test_window_shift_wraps_and_preserves_other_times(monkeypatch):
    monkeypatch.setattr("testing.luke_imec1_cross_seed_replication_v1.MOTION_WINDOWS", ((10.0, 12.0),))
    table = pd.DataFrame({"time_s": [10.5, 11.8, 20.0], "relative_um": [1, 2, 3]})
    shifted = shift_within_windows(table, np.array([.5]))
    assert np.allclose(np.sort(shifted.time_s), [10.3, 11.0, 20.0])
