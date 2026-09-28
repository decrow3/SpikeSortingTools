import numpy as np
import pandas as pd

from testing.luke_imec1_medicine_m_reference_v1 import (
    SHIFTS, intervals, join_short_gaps, null_validation, shifted_correlation,
)


def test_join_short_gaps_and_intervals():
    mask = np.array([0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0], bool)
    joined = join_short_gaps(mask, 4)
    assert joined.tolist() == [False, True, True, True, True, True, False,
                               False, False, False, False, True, False]
    got = intervals(joined, np.arange(len(joined)) * 0.25 + 0.125)
    assert got == [(1, 6, 0.25, 1.5), (11, 12, 2.75, 3.0)]


def test_shift_sign_is_applied_to_episode_map():
    rng = np.random.default_rng(0)
    rest = rng.normal(size=(80, 4))
    episode = np.zeros_like(rest)
    episode[:-20] = rest[20:]
    shift, _, correlations = shifted_correlation(episode, rest)
    assert shift == -200
    assert correlations[np.flatnonzero(SHIFTS == -200)[0]] > 0.99


def test_null_validation_requires_each_window_at_zero(monkeypatch):
    import testing.luke_imec1_medicine_m_reference_v1 as module

    monkeypatch.setattr(module, "WINDOWS", {"a": (0, 1), "b": (0, 1)})
    table = pd.DataFrame({"window": ["a"] * 3 + ["b"] * 3,
                          "resolved": True,
                          "best_shift_um": [0, 0, 10, 0, 0, 0]})
    assert null_validation(table)["status"] == "pass"
    table.loc[table.window == "b", "best_shift_um"] = -10
    assert null_validation(table)["status"] == "fail"
