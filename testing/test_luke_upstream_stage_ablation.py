import numpy as np

from testing.luke_upstream_stage_ablation import (
    Window,
    event_metrics,
    max_channel_shift_correlation,
    motion_window_metrics,
)


def test_motion_window_metrics_applies_recording_time_origin():
    times = np.arange(100.5, 106.5)
    displacement = np.column_stack((np.arange(6.0), np.arange(6.0) * 2))
    result = motion_window_metrics(
        displacement,
        times,
        np.array([0.0, 100.0]),
        Window("test", 2.0, 2.0),
        recording_t_start_s=100.0,
    )
    assert result["n_motion_bins"] == 2
    assert result["absolute_start_s"] == 102.0
    assert result["max_spread_relative_time_s"] in (2.5, 3.5)


def test_max_channel_shift_correlation_recovers_shifted_waveform():
    reference = np.zeros((11, 8))
    reference[4:7, 2] = [-1.0, -4.0, -1.0]
    shifted = np.zeros_like(reference)
    shifted[4:7, 5] = [-1.0, -4.0, -1.0]
    assert max_channel_shift_correlation(reference, shifted, max_shift_channels=4) > 0.999


def test_event_metrics_counts_extra_temporal_extrema_and_spatial_peaks():
    fs = 30_000.0
    traces = np.zeros((301, 8), dtype=np.float32)
    rng = np.random.default_rng(4)
    traces += rng.normal(scale=0.2, size=traces.shape)
    center = len(traces) // 2
    traces[center, 2] = -8.0
    traces[center + 12, 2] = 5.0
    traces[center, 6] = -7.0
    metrics, core = event_metrics(traces, np.arange(8) * 40.0, fs)
    assert metrics["extra_temporal_extrema"] >= 1
    assert metrics["spatial_peak_count_4sigma"] >= 2
    assert metrics["peak_snr"] > 4
    assert core.shape[0] == 91
