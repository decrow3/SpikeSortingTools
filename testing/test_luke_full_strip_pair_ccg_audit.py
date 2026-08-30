import numpy as np

from testing.luke_full_strip_pair_ccg_audit import (
    deduplicate_times,
    pair_count_within,
    template_cosine_best_shift,
)


def test_pair_count_within_counts_all_pairs():
    assert pair_count_within(np.array([10, 20]), np.array([9, 10, 11, 30]), 1) == 3


def test_deduplicate_times():
    assert deduplicate_times(np.array([10, 11, 20, 21, 40]), 1).tolist() == [10, 20, 40]


def test_template_cosine_finds_shift():
    first = np.zeros((7, 2))
    second = np.zeros((7, 2))
    first[2, 0] = 1
    second[3, 0] = 1
    value, shift = template_cosine_best_shift(first, second, maximum_shift=2)
    assert np.isclose(value, 1.0)
    assert shift == 1
