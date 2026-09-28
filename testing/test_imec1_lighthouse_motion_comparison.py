from __future__ import annotations

import numpy as np
import pandas as pd

from testing.luke_imec1_lighthouse_motion_adapters_v1 import MotionField, zero_field


def test_rigid_linear_sampling_and_sign_contract():
    field = MotionField("rigid", np.array([0.0, 1.0, 2.0]), np.array([0.0, 10.0, 20.0]))
    value, support = field.sample(np.array([0.5, 1.5]), np.array([100.0, 100.0]))
    np.testing.assert_allclose(value, [5.0, 15.0])
    assert support.all()
    observed = np.array([105.0, 115.0])
    np.testing.assert_allclose(observed - value, [100.0, 100.0])


def test_nonrigid_bilinear_reference_depth():
    field = MotionField(
        "nonrigid",
        np.array([0.0, 1.0]),
        np.array([[0.0, 10.0], [20.0, 30.0]]),
        depth_um=np.array([100.0, 200.0]),
    )
    value, support = field.sample(np.array([0.5]), np.array([150.0]))
    np.testing.assert_allclose(value, [15.0])
    assert support.item()


def test_no_endpoint_or_depth_clamping():
    field = MotionField(
        "nonrigid", np.array([0.0, 1.0]), np.zeros((2, 2)), depth_um=np.array([0.0, 10.0])
    )
    value, support = field.sample(np.array([-0.1, 0.5, 1.1]), np.array([5.0, 20.0, 5.0]))
    assert not support.any()
    assert np.isnan(value).all()


def test_nearest_rejects_gap_and_unsupported_sample():
    field = MotionField(
        "lfp", np.array([0.0, 0.004, 0.008, 0.1]), np.arange(4.0),
        supported=np.array([True, False, True, True]), sampling="nearest", time_tolerance_s=0.01,
    )
    value, support = field.sample(np.array([0.004, 0.050]), np.ones(2))
    assert support.tolist() == [False, False]
    assert np.isnan(value).all()


def test_zero_field_supports_finite_queries():
    value, support = zero_field().sample(np.array([1.0, np.nan]), np.array([2.0, 2.0]))
    assert value[0] == 0
    assert support.tolist() == [True, False]


def test_common_support_is_identical_before_reduction():
    table = pd.DataFrame({"a_supported": [True, True, False], "b_supported": [True, False, True]})
    common = table[["a_supported", "b_supported"]].all(axis=1)
    assert common.tolist() == [True, False, False]


def test_one_to_one_assignment_does_not_reuse_spike():
    from testing.luke_imec1_lighthouse_sort_concentration_v1 import one_to_one_time_assignment

    pairs = one_to_one_time_assignment(
        np.array([100, 102]), np.array([101]), tolerance_samples=2
    )
    assert len(pairs) == 1
    assert pairs[0][1] == 0


def test_family_balanced_rmse_differs_from_event_weighting():
    from testing.luke_imec1_lighthouse_motion_comparison_v1 import family_balanced_metrics

    increments = pd.DataFrame({
        "family_uid": ["a", "a", "a", "b", "b", "b", "b", "b", "b"],
        "residual_um": [10.0, 10.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "lighthouse_delta_um": [10.0, 10.0, 10.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    })
    metrics = family_balanced_metrics(increments)
    assert np.isclose(metrics["rmse_um"], np.sqrt(50.0))
    assert not np.isclose(metrics["rmse_um"], np.sqrt(100.0 / 3.0))


def test_historical_imec0_regression_fixture():
    path = "testing/outputs/luke_motion_candidate_lighthouse_comparison_v1/candidate_summary.csv"
    table = pd.read_csv(path)
    row = table.loc[
        table.sensitivity.eq("plausibility_lattice_node_5um")
        & table.regime.eq("all") & table.window.eq("930_1030")
        & table.candidate.eq("lfp_rigid")
    ].iloc[0]
    assert row.families == 6 and row.increments == 53
    assert np.isclose(row.family_balanced_rmse_um, 42.35266631201773)
    assert np.isclose(row.skill_vs_zero, 0.6120515044604633)


def test_concentration_is_invariant_to_label_renumbering():
    labels = pd.Series([4, 4, 4, 9, 10])
    renamed = labels.map({4: 400, 9: 2, 10: 8})
    assert labels.value_counts().sort_values().tolist() == renamed.value_counts().sort_values().tolist()
