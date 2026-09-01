import numpy as np

from testing.luke_sorter_waveform_arbitration import (
    best_shift_cosine,
    explained_fraction,
    shifted_control_times,
    stratified_times,
)


def test_best_shift_cosine_recovers_alignment():
    first = np.zeros((15, 2))
    second = np.zeros((15, 2))
    first[5, 0] = -3
    second[7, 0] = -3
    cosine, shift = best_shift_cosine(first, second, 3)
    assert np.isclose(cosine, 1.0)
    assert shift == 2


def test_explained_fraction_separates_matching_and_opposite_waveforms():
    template = np.zeros((20, 2), dtype=float)
    template[10, 0] = -2
    snippets = np.stack([template, -template, np.zeros_like(template)])
    np.testing.assert_allclose(explained_fraction(snippets, template), [1, 0, 0])


def test_stratified_times_is_bounded_and_deterministic():
    times = np.arange(10, 1210, 10)
    first = stratified_times(times, 24, 1200, 5)
    second = stratified_times(times, 24, 1200, 5)
    assert first.size <= 24
    np.testing.assert_array_equal(first, second)


def test_shifted_controls_reverse_near_end():
    values = shifted_control_times(np.array([20, 90]), 100, margin=5, offset=10)
    np.testing.assert_array_equal(values, [30, 80])
