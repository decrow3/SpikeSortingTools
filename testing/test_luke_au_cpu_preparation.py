import numpy as np

from testing.luke_au_cpu_preparation import (
    DONOR_RULE,
    array_sha256,
    exact_remap,
    exact_same_column_map,
    immutable_regular_train,
    observed_interior_energy,
    qualify_exact_state,
    sample_and_quantize_trajectory,
    score_one_cluster,
    seeded_independent_population,
    seeded_independent_train,
    select_donors,
)


def staggered_geometry(n_rows=12):
    return np.asarray(
        [(0.0 if row % 2 == 0 else 32.0, row * 20.0) for row in range(n_rows)],
        dtype=float,
    )


def test_frozen_train_is_5hz_local_and_immutable():
    train = immutable_regular_train()
    assert train.size == 1690
    assert np.all(np.diff(train) == 6000)
    assert train.flags.writeable is False


def test_seeded_population_is_independent_refractory_and_order_stable():
    forward = seeded_independent_population([11, 22, 33])
    reverse = seeded_independent_population([33, 22, 11])
    for unit_id in forward:
        assert np.array_equal(forward[unit_id], reverse[unit_id])
        assert forward[unit_id].flags.writeable is False
        assert np.all(np.diff(forward[unit_id]) >= 90)
    assert not np.array_equal(forward[11], forward[22])
    assert np.array_equal(forward[11], seeded_independent_train(11))


def test_donor_rule_balances_depth_then_fills_without_outcomes():
    rows = []
    for unit in range(42):
        rows.append(
            {
                "unit_id": unit,
                "quality": "good",
                "depth_um": unit * 80.0,
                "ptp_uv": 100.0 + unit,
                "refractory_violation_fraction": 0.001,
                "rest_spike_fraction": 0.9,
                "isolation_score": 0.9,
            }
        )
    selected = select_donors(rows)
    assert len(selected) == DONOR_RULE.target_count
    assert set(row["depth_stratum"] for row in selected) == set(range(6))


def test_exact_40um_map_is_same_column_and_roundtrips_interior_energy():
    geom = staggered_geometry()
    mapping = exact_same_column_map(geom, 40.0)
    for source, target in enumerate(mapping):
        if target >= 0:
            assert geom[target, 0] == geom[source, 0]
            assert geom[target, 1] == geom[source, 1] + 40.0
    template = np.zeros((9, geom.shape[0]), dtype=np.float32)
    template[3:6, 4:7] = np.array([[-1.0], [-4.0], [-1.0]], dtype=np.float32)
    moved = exact_remap(template, mapping)
    assert np.sum(moved * moved) == np.sum(template * template)
    result = qualify_exact_state(template, geom, 40.0)
    assert result["passed"] is True


def test_qualification_fails_when_energy_leaves_probe():
    geom = staggered_geometry()
    template = np.zeros((9, geom.shape[0]), dtype=np.float32)
    template[3:6, -2:] = 1.0
    assert qualify_exact_state(template, geom, 40.0)["passed"] is False


def test_crop_energy_check_is_explicitly_observed_support_only():
    geom = staggered_geometry(80)
    template = np.zeros((9, len(geom)), dtype=np.float32)
    template[3:6, 38:42] = 1.0
    result = observed_interior_energy(
        template,
        geom,
        max_abs_shift_um=280.0,
        interpolation_radius_um=200.0,
    )
    assert result["observed_support_only"] is True
    assert result["full_probe_energy_known"] is False
    assert result["margin_um_each_edge"] == 480.0
    assert result["passes_observed_99pct"] is True


def test_crop_energy_check_rejects_edge_supported_donor():
    geom = staggered_geometry(80)
    template = np.zeros((9, len(geom)), dtype=np.float32)
    template[3:6, :4] = 1.0
    result = observed_interior_energy(
        template,
        geom,
        max_abs_shift_um=280.0,
        interpolation_radius_um=200.0,
    )
    assert result["passes_observed_99pct"] is False


def test_corrected_exclusive_scorer_is_inclusive_at_point4ms_only():
    # 0.4 ms at 30 kHz is 12 samples. One competing output cannot be reused.
    result = score_one_cluster([100, 124, 300], [88, 112, 137, 500])
    assert result == {
        "tolerance_samples": 12,
        "tp": 2,
        "fp": 2,
        "fn": 1,
        "accuracy": 0.4,
    }


def test_matching_chunk_centres_and_half_away_quantization():
    result = sample_and_quantize_trajectory(
        np.array([0.0, 1.0, 2.0]),
        np.array([0.0, 20.0, -60.0]),
        np.array([0, 30_000]),
        30_000,
    )
    assert np.array_equal(result["chunk_center_samples"], [15_000, 45_000])
    assert np.allclose(result["sampled_displacement_um"], [10.0, -20.0])
    assert np.array_equal(result["quantized_displacement_um"], [0.0, -40.0])


def test_array_hash_binds_shape_dtype_and_content():
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    assert array_sha256(a) == array_sha256(a.copy())
    assert array_sha256(a) != array_sha256(a.astype(np.float64))
    assert array_sha256(a) != array_sha256(a.reshape(3, 2))
