import numpy as np

from testing.bj_matching_objective_fixtures import (
    DEFAULT_INV_LAMBDA,
    algebra_rows,
    direct_residual_decrease,
    physical_sampling_rows,
    scaled_objective,
)


def test_scaled_objective_matches_residual_decrease_plus_prior():
    x = np.array([2.0, -1.0, 0.5])
    t = np.array([1.0, -0.25, 0.75])
    conv, normsq = x @ t, t @ t
    scale, objective = scaled_objective(conv, normsq, DEFAULT_INV_LAMBDA)
    assert np.isclose(objective, direct_residual_decrease(x, t, scale, DEFAULT_INV_LAMBDA))


def test_free_scaling_cancels_only_declared_scalar_case():
    rows = algebra_rows()
    assert np.allclose([x["free_unbounded_objective"] for x in rows], 50.0)
    assert not np.allclose([x["free_bounded_objective"] for x in rows], 50.0)
    assert not np.allclose([x["regularized_default_objective"] for x in rows], 50.0)
    assert not np.allclose([x["coscaled_signal_objective"] for x in rows], 50.0)


def test_physical_sampling_changes_max_ptp_without_continuous_peak_change():
    _, rows, _, _ = physical_sampling_rows()
    assert {x["continuous_peak_ptp"] for x in rows} == {1.0}
    sampled = [x["exact_sampled_max_channel_ptp"] for x in rows]
    assert max(sampled) == 1.0
    assert min(sampled) < 0.85
