import numpy as np
import pandas as pd

from testing.luke_ks4_native_operator_audit import (
    ARMS,
    DISPLACEMENTS_UM,
    SpatialKernel,
    apply_spatial,
    event_is_interior,
    fixed_nearest_pairs,
    frozen_plan,
    matched_filter_snr,
    pair_separability,
    robust_gate_summary,
    si_spatial_matrix,
    validate_frozen_plan,
)


def test_frozen_plan_preserves_six_arms_grid_and_guards():
    plan = frozen_plan()
    validate_frozen_plan(plan)
    assert tuple(plan["arms"]) == ARMS
    assert tuple(plan["displacements_um"]) == DISPLACEMENTS_UM
    assert plan["prospective_holdout_accessed"] is False
    assert plan["source_domain_policy"]["status"] == "must_be_qualified_before_discovery_dry_run"


def test_si_matrix_orientation_matches_time_by_channel_application():
    positions = np.column_stack((np.zeros(5), np.arange(5) * 20.0))
    traces = np.arange(35, dtype=np.float32).reshape(7, 5)
    kernel = SpatialKernel("nearest", "nearest")
    matrix = si_spatial_matrix(positions, 0.0, kernel)
    np.testing.assert_allclose(matrix, np.eye(5), atol=0, rtol=0)
    np.testing.assert_array_equal(apply_spatial(traces, matrix), traces)


def test_matched_filter_uses_fixed_reference_direction():
    reference = np.zeros((5, 2), dtype=float)
    reference[2, 0] = -2.0
    event = reference * 1.5
    rng = np.random.default_rng(4)
    noise = rng.normal(scale=0.1, size=(101, 2))
    snr, score, sigma = matched_filter_snr(event, reference, noise, n_noise_windows=10)
    assert score > 0
    assert sigma > 0
    assert snr > 5


def test_nearest_pairs_are_frozen_from_reference_only():
    refs = {
        "a": np.array([[0.0, -1.0, 0.0]]),
        "b": np.array([[0.0, -0.9, 0.1]]),
        "c": np.array([[1.0, 0.0, 0.0]]),
    }
    pairs = fixed_nearest_pairs(refs)
    assert pairs.set_index("template_id").loc["a", "neighbor_template_id"] == "b"
    candidates = {name: value.copy() for name, value in refs.items()}
    candidates["b"] = np.array([[1.0, 0.0, 0.0]])
    noise = np.random.default_rng(2).normal(size=(31, 3))
    measured = pair_separability(pairs, refs, candidates, noise)
    assert set(measured.template_id) == set(refs)
    assert pairs.set_index("template_id").loc["a", "neighbor_template_id"] == "b"


def test_gate_uses_worst_generator_not_pooled_success():
    rows = []
    for generator, residual in [("good", 0.08), ("bad", 0.099)]:
        for case in range(4):
            rows.append(
                {
                    "generator": generator,
                    "case": case,
                    "residual_fraction": residual,
                    "amplitude_retention": 0.90,
                    "template_cosine": 0.90,
                    "baseline_residual_fraction": 0.10,
                    "baseline_amplitude_retention": 0.90,
                    "baseline_template_cosine": 0.90,
                }
            )
    by_generator, decision = robust_gate_summary(pd.DataFrame(rows))
    assert len(by_generator) == 2
    assert np.isclose(decision["worst_generator_median_delta_residual"], -0.001)
    assert decision["primary_screen_pass"] is False


def test_event_edge_status_uses_only_nonzero_template_support():
    positions = np.column_stack((np.zeros(6), np.arange(6) * 20.0))
    waveform = np.zeros((5, 6), dtype=float)
    waveform[2, 2:4] = -1.0
    assert event_is_interior(waveform, positions, 20.0)
    waveform[:] = 0
    waveform[2, 5] = -1.0
    assert not event_is_interior(waveform, positions, 20.0)
