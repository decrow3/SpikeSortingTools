import numpy as np

from testing.luke_truncation_matched_units import coincidence_matrix


def test_coincidence_matrix_does_not_reuse_target_events():
    a_st = np.array([100, 105], dtype=np.int64)
    a_cl = np.array([0, 0], dtype=np.int64)
    b_st = np.array([103], dtype=np.int64)
    b_cl = np.array([7], dtype=np.int64)

    _, _, counts = coincidence_matrix(a_st, a_cl, b_st, b_cl)
    assert counts.sum() == 1
