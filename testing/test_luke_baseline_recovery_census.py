import json

import pandas as pd

from testing.luke_baseline_recovery_census import SOURCE, census, parse_selection_constants


def run(rows, intervals=((0, 900),)):
    constants = parse_selection_constants(json.loads((SOURCE / "selection.json").read_text())["selection_constants"])
    return census(pd.DataFrame(rows), constants, intervals)


def rows(cid=1, values=(2, 3, 20, 25)):
    return [dict(sort_id="rescue_test", cluster_id=cid, source_row=i,
                 i0=i*1000, i1=i*1000+999, nominal_count=1000,
                 start_s=i*100, end_s=i*100+99, missing_pct=v,
                 status="finite_interior") for i,v in enumerate(values)]


def test_uncapped_and_closed_case_excluded():
    table, abrupt = run(sum((rows(cid) for cid in [1, 2, 3, 37]), []))
    assert len(table) == 4
    assert set(abrupt.cluster_id) == {1, 2, 3}
    assert table.loc[table.cluster_id == 37, "status"].item() == "excluded_closed_cluster_37"


def test_cannot_bridge_development_intervals():
    table, abrupt = run(rows(), ((0, 200), (200, 500)))
    assert table.status.item() == "no_ordered_pair_in_same_dev_interval"
    assert abrupt.empty


def test_invalid_intervening_window_only_allowed_in_exploratory_rule():
    data = rows(values=(2, 3, 10, 20, 25))
    data[2]["status"] = "invalid_input"
    table, abrupt = run(data)
    assert table.status.item() == "qualifying_transition"
    assert table.intervening_windows.item() == 1
    assert abrupt.empty


def test_invalid_or_nonfinite_pair_is_rejected():
    for field, value in [("status", "boundary_pinned"), ("missing_pct", float("nan"))]:
        data = rows()
        data[1][field] = value
        table, abrupt = run(data)
        assert table.status.item() != "qualifying_transition"
        assert abrupt.empty
