import numpy as np
import pandas as pd

from testing.luke_imec1_event_localization_audit import (
    channels_for_fraction,
    effective_channel_count,
    summarize_channel_counts,
)


def test_distribution_helpers():
    assert effective_channel_count(np.ones(4)) == 4.0
    assert channels_for_fraction(np.array([5, 3, 1, 1]), 0.5) == 1
    assert channels_for_fraction(np.array([4, 3, 2, 1]), 0.5) == 2


def test_localization_summary_rejects_row216_as_distributed_driver():
    rows = []
    for window in ("pathological", "shared", "session-wide"):
        for stage in (
            "common_bandpass_equal_5_reference",
            "common_bandpass_local_reference",
            "common_bandpass_shank_median",
        ):
            for polarity in ("negative", "positive"):
                for channel in range(384):
                    rows.append(
                        {
                            "dataset": f"Luke imec1 {window}",
                            "window_kind": window,
                            "stage": stage,
                            "polarity": polarity,
                            "batch_index": 0,
                            "duration_s": 2.0,
                            "channel": channel,
                            "y_um": channel * 10.0,
                            "event_count": 2 + (channel % 5),
                        }
                    )
    _, localization, correlations, decision = summarize_channel_counts(
        pd.DataFrame(rows)
    )
    assert len(localization) == 18
    assert len(correlations) == 9
    assert decision["decision"] == "row216_not_dominant_positive_excess_is_distributed"
