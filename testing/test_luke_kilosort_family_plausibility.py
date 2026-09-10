import numpy as np
import pandas as pd

from testing import luke_kilosort_family_plausibility_v3 as audit


def test_coactive_depth_separation_detects_two_simultaneous_bands():
    time_s = np.array([930.1, 930.2, 930.3, 930.4, 931.1, 931.2, 931.3, 931.4])
    cids = np.array([1, 1, 2, 2, 1, 1, 2, 2])
    depths = np.array([100, 102, 200, 202, 110, 112, 140, 142], float)
    count, fraction, median_span = audit.coactive_depth_separation(
        time_s, depths, cids, np.array([1, 2])
    )
    assert count == 2
    assert fraction == 0.5
    assert median_span == 65.0


def test_classification_keeps_failure_separate_from_low_support():
    coherent = pd.Series({
        "cross_cid_joined_intervals": 10,
        "rapid_jump_fraction_ge_100um": 0.0,
        "coactive_one_second_bins": 5,
        "separated_coactive_fraction_ge_80um": 0.0,
        "merged_to_poisson_short_isi_ratio": 1.0,
    })
    assert audit.classify_family(coherent) == "depth_time_coherent"
    failing = coherent.copy()
    failing["merged_to_poisson_short_isi_ratio"] = 4.0
    assert audit.classify_family(failing) == "implausible"
    underpowered = coherent.copy()
    underpowered["cross_cid_joined_intervals"] = 2
    underpowered["coactive_one_second_bins"] = 1
    assert audit.classify_family(underpowered) == "underpowered"


def test_auc_ties_and_direction():
    labels = np.array([False, False, True, True])
    assert audit.auc_as_listed(np.array([0, 1, 2, 3]), labels) == 1.0
    assert audit.auc_as_listed(np.array([3, 2, 1, 0]), labels) == 0.0
    assert audit.auc_as_listed(np.ones(4), labels) == 0.5
