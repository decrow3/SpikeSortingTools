import numpy as np
import pandas as pd

from testing.luke_motion_estimator_band_ablation import (
    array_digest,
    deterministic_half,
    paired_band_voltage,
    select_separated_indices,
    summarize_voltage,
)


def test_deterministic_half_is_stable_and_balanced():
    samples = np.arange(10_000, dtype=np.int64)
    channels = samples % 384
    first = deterministic_half(samples, channels)
    second = deterministic_half(samples, channels)
    np.testing.assert_array_equal(first, second)
    assert set(first) == {0, 1}
    assert 0.45 < np.mean(first) < 0.55


def test_deterministic_half_keeps_simultaneous_channels_together():
    samples = np.repeat(np.arange(100), 4)
    channels = np.tile(np.arange(4), 100)
    halves = deterministic_half(samples, channels).reshape(100, 4)
    assert np.all(halves == halves[:, :1])


def test_event_selection_enforces_temporal_separation():
    samples = np.array([100, 100, 110, 200, 400])
    amplitudes = np.array([-5, -10, -9, -8, -7])
    selected = select_separated_indices(samples, amplitudes, 3, 30)
    np.testing.assert_array_equal(selected, [1, 3, 4])


def test_array_digest_changes_with_peak_content():
    first = np.arange(10, dtype=np.int64)
    second = first.copy()
    assert array_digest(first) == array_digest(second)
    second[-1] += 1
    assert array_digest(first) != array_digest(second)


def test_voltage_summary_preserves_variant_pairing():
    frame = pd.DataFrame(
        {
            "variant": ["no_motion", "no_motion", "band_g0.25", "band_g0.25"],
            "event_index": [0, 1, 0, 1],
            "peak_amplitude_ratio_to_no_motion": [1.0, 1.0, 0.8, 1.0],
            "waveform_correlation_to_no_motion": [1.0, 1.0, 0.7, 0.9],
            "anchor_peak_depth_error_um": [0.0, 0.0, -20.0, 20.0],
            "local_zero_fraction": [0.0, 0.0, 0.1, 0.2],
        }
    )
    summary = summarize_voltage(frame).set_index("variant")
    assert np.isclose(summary.loc["band_g0.25", "median_peak_amplitude_ratio"], 0.9)
    assert np.isclose(summary.loc["band_g0.25", "median_waveform_correlation"], 0.8)


def test_paired_band_voltage_uses_directional_metrics():
    frame = pd.DataFrame(
        {
            "variant": ["ap_300_3000_g0.25"] * 3 + ["ap_300_6000_g0.25"] * 3,
            "event_index": [0, 1, 2, 0, 1, 2],
            "waveform_correlation_to_no_motion": [0.8, 0.8, 0.8, 0.9, 0.9, 0.9],
            "peak_amplitude_ratio_to_no_motion": [0.8, 0.8, 0.8, 0.9, 0.9, 0.9],
        }
    )
    result = paired_band_voltage(frame).set_index("metric")
    assert np.isclose(
        result.loc["waveform_correlation_wide_minus_narrow", "median_paired_difference"],
        0.1,
    )
    assert np.isclose(
        result.loc["absolute_peak_error_wide_minus_narrow", "median_paired_difference"],
        -0.1,
    )
