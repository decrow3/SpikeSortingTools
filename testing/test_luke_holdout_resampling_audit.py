import numpy as np

from testing.luke_holdout_resampling_audit import polarity_anchored_metrics


def test_polarity_anchor_finds_positive_and_negative_events():
    traces = np.zeros((301, 5), dtype=np.float32)
    traces[:, :] += np.linspace(-0.2, 0.2, 301)[:, None]
    traces[150, 1] = -10
    traces[150, 3] = 12
    depths = np.arange(5) * 20.0
    negative, _ = polarity_anchored_metrics(traces, depths, 30000, 20, "negative")
    positive, _ = polarity_anchored_metrics(traces, depths, 30000, 60, "positive")
    assert negative["anchor_peak_channel"] == 1
    assert positive["anchor_peak_channel"] == 3
    assert negative["anchor_peak_amplitude_counts"] == 10
    assert positive["anchor_peak_amplitude_counts"] == 12
