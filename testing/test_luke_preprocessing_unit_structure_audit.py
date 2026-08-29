import numpy as np

import pandas as pd

from testing.luke_preprocessing_unit_structure_audit import (
    graph_component_count,
    template_peak_depths,
    unit_bin_counts,
)


def test_unit_bin_counts_preserves_unit_and_bin_grain():
    times = np.array([0, 9, 10, 19])
    clusters = np.array([2, 2, 5, 5])
    result = unit_bin_counts(times, clusters, np.array([2, 5]), fs=1.0)
    assert result[0, :2].tolist() == [2, 0]
    assert result[1, :2].tolist() == [0, 2]


def test_template_peak_depths_uses_largest_absolute_contact():
    templates = np.zeros((1, 3, 2))
    templates[0, 1, 1] = -5
    positions = np.array([[0, 10], [0, 30]])
    assert template_peak_depths(templates, positions).tolist() == [30]


def test_graph_component_count_collapses_connected_pairs():
    pairs = pd.DataFrame({"unit_first": [0, 1], "unit_second": [1, 2]})
    assert graph_component_count(np.array([0, 1, 2, 3]), pairs) == 2
