import numpy as np

from testing.luke_sorter_band_comparison import (
    SpikeSet,
    _same_time_label_multiset,
    cross_unit_coincidence_fraction,
    event_overlap,
    unit_metrics,
)


def test_time_label_multiset_ignores_equal_time_order_only():
    first_times = np.array([10, 10, 20])
    first_labels = np.array([1, 2, 3])
    second_times = np.array([10, 10, 20])
    second_labels = np.array([2, 1, 3])
    assert _same_time_label_multiset(
        first_times, first_labels, second_times, second_labels
    )
    second_labels[-1] = 4
    assert not _same_time_label_multiset(
        first_times, first_labels, second_times, second_labels
    )


def test_cross_unit_coincidence_respects_unit_and_depth():
    times = np.array([10, 12, 13, 30])
    labels = np.array([1, 2, 1, 3])
    depths = np.array([100.0, 110.0, 100.0, 300.0])
    assert cross_unit_coincidence_fraction(times, labels, depths, 3, 75.0) == 0.75


def test_event_overlap_is_symmetric_and_depth_gated():
    first = SpikeSet(
        "first", np.array([10, 100]), np.array([1, 1]), np.array([100.0, 100.0])
    )
    second = SpikeSet(
        "second", np.array([12, 101, 200]), np.array([2, 2, 2]), np.array([110.0, 300.0, 100.0])
    )
    result = event_overlap(first, second, time_radius=3, depth_radius=60.0)
    assert result["first_matched_fraction"] == 0.5
    assert result["second_matched_fraction"] == 1 / 3


def test_unit_metrics_uses_fixed_ten_second_bins_and_endpoint_windows():
    fs = 100.0
    times = np.array([0, 1001, 2001, 3001, 4001, 5001, 6001, 7001, 8001, 9001, 10001, 11001])
    spikes = SpikeSet("sorter", times, np.ones(times.size, dtype=int), np.full(times.size, 50.0))
    metrics = unit_metrics(spikes, fs=fs, duration_s=120.0).iloc[0]
    assert metrics.presence_fraction_10s == 1.0
    assert bool(metrics.present_first_20s)
    assert bool(metrics.present_last_20s)
