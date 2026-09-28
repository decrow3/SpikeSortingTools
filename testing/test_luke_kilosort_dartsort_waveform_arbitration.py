import numpy as np
import pandas as pd

from testing.luke_kilosort_dartsort_waveform_arbitration import (
    choose_candidates,
    cosine_at_best_lag,
)


def test_cosine_recovers_small_sample_shift():
    reference = np.zeros((21, 2)); reference[10, 0] = -2; reference[11, 1] = 1
    shifted = np.zeros_like(reference); shifted[12, 0] = -2; shifted[13, 1] = 1
    cosine, lag = cosine_at_best_lag(reference, shifted)
    assert np.isclose(cosine, 1)
    assert lag == 2


def test_candidates_require_good_support_and_are_bounded_per_arm():
    rows = []
    for arm in ("unwarped", "ap_rigid", "dartsort_native", "lfp_native", "lfp_savgol"):
        for index in range(8):
            rows.append({"arm": arm, "family_id": index, "ks_good_unit_count": index != 0,
                         "matched_events": 100, "jaccard": 0.1 * index,
                         "ks_spikes": 1000, "dartsort_spikes": 500})
    chosen = choose_candidates(pd.DataFrame(rows))
    assert chosen.groupby("arm").size().eq(5).all()
    assert chosen.ks_good_unit_count.gt(0).all()
