import numpy as np

from testing.cj_force_gate_w2 import lag_counts, random_derangement


def test_exact_lag_boundaries_and_nonwrapping_derangement():
    # Zero and +/-29 are central; +/-30 are neither; +/-45 and +/-89 are
    # shoulders; +/-90 are outside. No coincidence event is filtered.
    b = np.array([-90, -89, -45, -30, -29, 0, 29, 30, 45, 89, 90])
    central, shoulder = lag_counts(np.array([0]), b)
    assert central == 3
    assert shoulder == 4
    permutation = random_derangement(7, np.random.default_rng(4))
    assert np.array_equal(np.sort(permutation), np.arange(7))
    assert np.all(permutation != np.arange(7))
