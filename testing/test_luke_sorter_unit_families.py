import numpy as np

from testing.luke_sorter_band_comparison import SpikeSet
from testing.luke_sorter_unit_families import (
    best_target_labels,
    component_table,
    directional_unit_support,
    requalify_edges,
    unit_pair_edges,
)


def spikes(name, times, labels, depths=None):
    times = np.asarray(times, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    if depths is None:
        depths = np.full(times.size, 100.0)
    return SpikeSet(name, times, labels, np.asarray(depths, dtype=float))


def test_directional_support_counts_source_once_per_target_unit():
    source = spikes("a", [10], [1])
    target = spikes("b", [9, 11], [2, 2])
    pairs, bins = directional_unit_support(source, target, 2, 60.0, 100)
    assert pairs.iloc[0].matched_source_events == 1
    assert bins.iloc[0].matched_source_events == 1


def test_best_target_labels_uses_nearest_time_then_depth():
    source = spikes("a", [10], [1], [100])
    target = spikes("b", [9, 11], [2, 3], [130, 110])
    np.testing.assert_array_equal(best_target_labels(source, target, {2, 3}, 2, 60.0), [3])


def test_edges_and_components_detect_one_to_many_family():
    first = spikes("ks", [10, 20, 30, 40], [1, 1, 1, 1])
    second = spikes("kia", [10, 20, 30, 40], [2, 2, 3, 3])
    edges, _ = unit_pair_edges(
        first,
        second,
        time_radius=0,
        depth_radius_um=1.0,
        bin_samples=100,
        minimum_unit_spikes=2,
        minimum_pair_events=2,
    )
    assert edges.qualified_family_edge.sum() == 2
    # component_table deliberately uses the production eligibility threshold.
    repeated_first = spikes("ks", np.arange(40), np.ones(40, dtype=int))
    repeated_second = spikes(
        "kia", np.arange(40), np.repeat([2, 3], 20)
    )
    repeated_edges, _ = unit_pair_edges(
        repeated_first,
        repeated_second,
        time_radius=0,
        depth_radius_um=1.0,
        bin_samples=100,
    )
    families = component_table([repeated_first, repeated_second], repeated_edges, "test")
    assert families.iloc[0].mapping_shape == "1x2"


def test_requalification_applies_event_and_coverage_thresholds():
    first = spikes("a", np.arange(40), np.ones(40, dtype=int))
    second = spikes("b", np.arange(40), np.ones(40, dtype=int) * 2)
    edges, _ = unit_pair_edges(first, second, 0, 1.0, 100)
    assert requalify_edges(edges, 20, 0.7, 0.3).qualified_family_edge.iloc[0]
    assert not requalify_edges(edges, 50, 0.7, 0.3).qualified_family_edge.iloc[0]
