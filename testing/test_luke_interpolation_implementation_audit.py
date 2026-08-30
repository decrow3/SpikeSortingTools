import numpy as np
import pandas as pd

from testing.luke_interpolation_implementation_audit import (
    add_paired_changes,
    anchored_event_metrics,
    summarize,
)


def test_add_paired_changes_uses_within_event_baseline():
    rows = []
    for review_id, baseline_amp, variant_amp in (("a", 10.0, 8.0), ("b", 20.0, 10.0)):
        for variant, amplitude in (
            ("conditioned_baseline", baseline_amp),
            ("candidate", variant_amp),
        ):
            rows.append(
                {
                    "review_id": review_id,
                    "window": "w",
                    "review_label": "neural",
                    "variant": variant,
                    "anchor_peak_amplitude_counts": amplitude,
                    "anchor_peak_snr": amplitude,
                    "snippet_rms_counts": amplitude / 2,
                    "local_snippet_rms_counts": amplitude / 2,
                    "anchor_peak_depth_error_um": 0.0,
                    "zero_fraction": 0.0,
                    "local_zero_fraction": 0.0,
                    "correlation_to_conditioned_baseline": 1.0,
                }
            )
    paired = add_paired_changes(pd.DataFrame(rows))
    candidate = paired.loc[paired["variant"] == "candidate"]
    np.testing.assert_allclose(candidate["ratio_anchor_peak_amplitude_counts"], [0.8, 0.5])
    result = summarize(paired)
    row = result.loc[result["variant"] == "candidate"].iloc[0]
    assert row["n_events"] == 2
    assert row["median_ratio_anchor_peak_amplitude_counts"] == 0.65


def test_anchored_event_metrics_ignores_flat_edge_channel():
    fs = 30_000.0
    rng = np.random.default_rng(3)
    traces = rng.normal(0, 0.3, size=(301, 8)).astype("float32")
    traces[:, 7] = 0
    traces[150, 3] = -8
    metrics, core = anchored_event_metrics(traces, np.arange(8) * 20.0, fs, 60.0)
    assert metrics["anchor_peak_channel"] == 3
    assert metrics["anchor_peak_amplitude_counts"] == 8
    assert core.shape == (91, 8)
