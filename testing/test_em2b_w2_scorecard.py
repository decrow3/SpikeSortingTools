import numpy as np

from testing.em2b_w2_scorecard import (
    DOMAINS,
    block_domain_exposure,
    bootstrap_rhos,
    safe_spearman,
)


def test_block_domain_exposure_closes_blocks_and_splits_boundaries():
    intervals = [(0.0, 1.5, DOMAINS[0]), (1.5, 3.0, DOMAINS[1])]
    result = block_domain_exposure(intervals, np.asarray([0.0, 1.0, 2.0, 3.0]))

    np.testing.assert_allclose(result.sum(axis=1), 1.0)
    np.testing.assert_allclose(result[:, :2], [[1, 0], [0.5, 0.5], [0, 1]])


def test_safe_spearman_leaves_undefined_cases_unresolved():
    assert np.isnan(safe_spearman([1, 1, 1], [1, 2, 3]))
    assert np.isnan(safe_spearman([1, 2], [1, 2]))
    assert safe_spearman([1, 2, 3], [3, 2, 1]) == -1.0


def test_bootstrap_uses_identical_block_draws_for_every_arm():
    # Identical per-block counts must remain identical in every common draw.
    block_counts = np.asarray(
        [
            [[3, 4, 0], [2, 1, 0], [1, 3, 0]],
            [[1, 2, 0], [4, 3, 0], [2, 5, 0]],
            [[5, 1, 0], [1, 5, 0], [3, 2, 0]],
        ]
    )
    arms = {
        name: {"eligible": np.ones(3, dtype=bool), "block_counts": block_counts}
        for name in ("a", "b")
    }
    exposure = np.asarray([[1.0, 1.0, 0.0]] * 3)

    result, multiplicities = bootstrap_rhos(arms, exposure, draws=20, seed=4)

    np.testing.assert_array_equal(result["a"], result["b"])
    np.testing.assert_array_equal(multiplicities.sum(axis=1), 3)
