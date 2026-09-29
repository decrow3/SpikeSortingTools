import pandas as pd
import pytest

from testing.em2b_cached_qc_summary import summarize


def table():
    return pd.DataFrame(
        {
            "unit_id": [0, 1, 2],
            "spike_count": [10, 20, 30],
            "presence_ratio": [1.0, 5 / 6, 0.5],
            "raw_adjacent_isi_lt_refractory_fraction": [0.0, 0.01, 0.02],
            "exact_duplicate_sample_count": [0, 0, 2],
        }
    )


def test_summary_uses_declared_boundaries():
    result = summarize(table())
    assert result["accepted_events"] == 60
    assert result["full_presence_units"] == 1
    assert result["presence_ge_five_sixths_units"] == 2
    assert result["units_raw_refractory_fraction_gt_0p01"] == 1
    assert result["exact_duplicate_sample_count"] == 2


def test_summary_rejects_duplicate_units_and_invalid_ratios():
    value = table()
    value.loc[2, "unit_id"] = 1
    with pytest.raises(RuntimeError, match="not unique"):
        summarize(value)
    value = table()
    value.loc[0, "presence_ratio"] = 1.1
    with pytest.raises(RuntimeError, match="invalid QC"):
        summarize(value)
