import numpy as np

from testing.luke_imec1_sorterfree_phase_audit_v1 import (
    complete_link_labels,
    lagged_overlap_cosine,
    overlap_indices,
)


RELATIVE_GEOMETRY = np.array([
    [0, -60], [32, -60], [16, -40], [48, -40],
    [0, -20], [32, -20], [16, 0], [48, 0],
    [0, 20], [32, 20], [16, 40], [48, 40],
    [0, 60], [32, 60], [16, 80], [48, 80],
], dtype=float)


def test_cross_phase_overlap_uses_exact_peak_centered_coordinates():
    a, b = overlap_indices(RELATIVE_GEOMETRY, 6, 8)
    assert len(a) == 10
    assert np.array_equal(RELATIVE_GEOMETRY[a] - RELATIVE_GEOMETRY[6],
                          RELATIVE_GEOMETRY[b] - RELATIVE_GEOMETRY[8])


def test_lagged_overlap_recovers_equal_relative_waveform():
    rng = np.random.default_rng(0)
    first = rng.normal(size=(49, 16))
    second = np.zeros_like(first)
    ia, ib = overlap_indices(RELATIVE_GEOMETRY, 6, 8)
    second[:, ib] = first[:, ia]
    score, coverage, common = lagged_overlap_cosine(first, 6, second, 8, RELATIVE_GEOMETRY)
    assert score > 0.999999
    assert common == 10
    assert 0 < coverage <= 1


def test_complete_link_does_not_single_link_chain():
    similarity = np.array([[1, .96, .91], [.96, 1, .96], [.91, .96, 1.]])
    labels = complete_link_labels(similarity, .95)
    assert len(np.unique(labels)) == 2
