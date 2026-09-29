import numpy as np

from testing.luke_cluster452_curation_lineage import reconstruct_final_labels


def test_reconstruction_applies_duplicate_merge_remove_and_renumber_order():
    raw = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    original, final, rows = reconstruct_final_labels(
        raw, np.array([1]), [[1, 2]], {3}
    )
    assert rows.tolist() == [0, 2, 3, 4, 5]
    assert original.tolist() == [0, 1, 1, 2, 2]
    assert final.tolist() == [0, 1, 1, 1, 1]
