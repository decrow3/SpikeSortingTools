import pandas as pd

from testing.luke_channel_artifact_audit import REFERENCE_STAGES, run


def test_channel_audit_separates_quiet_bad_channel_from_probe_imbalance(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    datasets = [
        ("Luke imec0 pathological", "pathological", 1.2),
        ("Luke imec1 pathological", "pathological", 4.0),
        ("Luke imec1 shared", "shared", 4.5),
        ("Luke imec1 session", "session-wide", 4.2),
        ("Yates raw session", "session-wide", 0.25),
    ]
    event_rows = []
    footprint_rows = []
    for dataset, window, ratio in datasets:
        for stage in REFERENCE_STAGES:
            for polarity, rate in (("negative", 100.0), ("positive", 100.0 * ratio)):
                event_rows.append(
                    {
                        "dataset": dataset,
                        "window_kind": window,
                        "stage": stage,
                        "polarity": polarity,
                        "threshold_kind": "absolute_uv",
                        "threshold": 75.0,
                        "median_event_rate_per_mm_s": rate,
                    }
                )
                footprint_rows.append(
                    {
                        "dataset": dataset,
                        "window_kind": window,
                        "stage": stage,
                        "polarity": polarity,
                        "sampled_events": 10,
                        "compact_fraction": 0.3 if "imec1" in dataset else 0.5,
                        "median_local_energy_fraction": 0.4,
                        "median_footprint_depth_sd_um": 200.0,
                        "median_active_channels_4sigma": 5.0,
                    }
                )
    pd.DataFrame(event_rows).to_csv(source / "raw_event_summary.csv", index=False)
    pd.DataFrame(footprint_rows).to_csv(
        source / "raw_footprint_summary.csv", index=False
    )
    channel_rows = []
    for dataset, window, _ in datasets:
        if "imec1" not in dataset:
            continue
        for channel, sigma in ((0, 30.0), (191, 8.0), (216, 31.0)):
            channel_rows.append(
                {
                    "dataset": dataset,
                    "window_kind": window,
                    "channel": channel,
                    "y_um": float(channel * 10),
                    "median_bandpass_sigma_uv": sigma,
                    "median_fraction_abs_raw_over_500uv": 0.0,
                }
            )
    pd.DataFrame(channel_rows).to_csv(source / "raw_channel_summary.csv", index=False)

    result = run(source, tmp_path / "out")
    assert result["decision"].startswith("distributed_imec1")
    assert result["reference_controls"]["imec1_positive_to_negative_ratio_range"] == [4.0, 4.5]
    assert result["channel_191"]["maximum_sigma_percentile"] <= 1 / 3
