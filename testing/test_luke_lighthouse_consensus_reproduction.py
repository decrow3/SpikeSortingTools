import numpy as np
import pandas as pd

from testing import luke_lighthouse_consensus_reproduction_v1 as audit


def test_binned_track_uses_first_ten_seconds_as_baseline():
    events = pd.DataFrame({"time_s": [931, 936, 941, 946], "relative_um": [10, 20, 30, 40]})
    track = audit.binned_track(events)
    np.testing.assert_allclose(track.to_numpy(), [-5, 5, 15, 25])


def test_consensus_is_equal_vote_median_with_minimum_support():
    wide = pd.DataFrame({"a": [0, 0], "b": [2, np.nan], "c": [100, 4]})
    value, support = audit.consensus(wide)
    assert value.iloc[0] == 2
    assert np.isnan(value.iloc[1])
    assert support.tolist() == [3, 2]
