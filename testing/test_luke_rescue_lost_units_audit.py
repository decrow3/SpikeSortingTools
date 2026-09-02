import numpy as np
import pandas as pd

from testing.luke_rescue_lost_units_audit import (
    classify,
    spike_distribution,
    _symmetric_table,
)


def test_classify_covers_every_branch():
    assert classify(0.0, [], set()) == "absent at detection"
    assert classify(0.1, [(1, 0.1, "mua")], set()) == "absent at detection"
    assert classify(1.0, [(5, 0.9, "mua")], set()) == "preserved as MUA"
    assert classify(1.0, [(5, 0.9, "mua")], {5}) == "merged into a rescue good unit"
    assert classify(1.0, [(5, 0.9, "good")], set()) == "merged into a rescue good unit"
    assert (
        classify(1.0, [(5, 0.3, "mua"), (6, 0.3, "mua")], set())
        == "split across rescue clusters"
    )
    assert (
        classify(1.0, [(5, 0.2, "mua"), (6, 0.1, "mua")], set())
        == "dispersed across rescue clusters"
    )


def test_spike_distribution_ranks_by_share_of_the_source_train():
    sort = {
        "st": np.array([101, 201, 202, 5000], dtype=np.int64),
        "cl": np.array([7, 7, 9, 9], dtype=np.int64),
        "label": {7: "good", 9: "mua"},
    }
    a_st = np.array([100, 200, 300], dtype=np.int64)  # third has no partner
    frac_found, ranked = spike_distribution(a_st, sort)
    assert frac_found == 2 / 3
    assert ranked[0][0] == 7
    assert ranked[0][1] == 2 / 3
    assert ranked[0][2] == "good"


def test_symmetric_table_reports_both_sides_without_netting():
    plus = pd.DataFrame({"classification": ["a", "a", "b"]})
    minus = pd.DataFrame({"classification": ["c", "c", "c", "d"]})
    table = _symmetric_table(plus, minus)
    assert set(table["side"]) == {"+ gained", "- lost"}
    assert table.loc[table.classification == "a", "n"].item() == 2
    assert table.loc[table.classification == "c", "n"].item() == 3
    assert table["n"].sum() == 7  # 3 gained + 4 lost, never a net
