import numpy as np
import pandas as pd

from testing.luke_kilosort_dartsort_detailed_qc import comparison_summary, components, partition_summary


def test_components_preserve_split_merge_families():
    edges = pd.DataFrame({
        "ks_unit": [1, 1, 2, 3],
        "dartsort_unit": [10, 11, 11, 12],
    })
    assert components(edges) == [([1, 2], [10, 11]), ([3], [12])]


def test_comparison_summary_is_paired_and_deterministic():
    rows = []
    for arm in ("unwarped", "ap_rigid", "dartsort_native", "lfp_native", "lfp_savgol"):
        for value in (1.0, 2.0, 3.0):
            rows.append({"arm": arm, "ks_x": value, "dartsort_x": value - 0.5})
    table = pd.DataFrame(rows)
    first = comparison_summary(table, ["x"], seed=4)
    second = comparison_summary(table, ["x"], seed=4)
    pd.testing.assert_frame_equal(first, second)
    assert np.allclose(first.dartsort_minus_ks_median, -0.5)
    assert np.allclose(first.dartsort_lower_fraction, 1.0)


def test_partition_summary_reports_only_finite_support():
    rows = []
    for arm in ("unwarped", "ap_rigid", "dartsort_native", "lfp_native", "lfp_savgol"):
        rows.append({
            "arm": arm,
            "ks_only_to_shared_amplitude_ratio": 0.8,
            "dartsort_only_to_shared_amplitude_ratio": 0.7,
            "ks_only_minus_shared_depth_um": 2.0,
            "dartsort_only_minus_shared_depth_um": -1.0,
            "ks_only_shared_temporal_tv": 0.1,
            "dartsort_only_shared_temporal_tv": np.nan,
        })
    result = partition_summary(pd.DataFrame(rows))
    assert set(result.n) == {0, 1}
    assert result.loc[result.metric.eq("ks_only_to_shared_amplitude_ratio"), "median"].eq(0.8).all()
