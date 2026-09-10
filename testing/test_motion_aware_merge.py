import numpy as np
import pytest

from pipeline.motion_aware_merge import (
    StatePair,
    aggregate_reciprocal_state_pairs,
    aligned_template_similarity,
    build_motion_state_templates,
    exact_same_column_pairs,
    feature_channel_indices,
    medicine_basis_weights,
    sample_eligible_field,
    squared_pc_feature_depths,
)


def imec0_geometry(rows=6):
    points = []
    for row in range(rows):
        xs = (16.0, 48.0) if row % 2 == 0 else (0.0, 32.0)
        points.extend((x, 20.0 * row) for x in xs)
    return np.asarray(points)


def test_staggered_geometry_has_no_20um_same_column_mapping():
    pos = imec0_geometry()
    target20, source20 = exact_same_column_pairs(pos, 20.0)
    target40, source40 = exact_same_column_pairs(pos, 40.0)
    assert target20.size == source20.size == 0
    assert target40.size == source40.size == 8
    np.testing.assert_allclose(pos[target40, 0], pos[source40, 0])
    np.testing.assert_allclose(pos[target40, 1], pos[source40, 1] + 40.0)


def test_exact_translation_recovers_template_and_rejects_wrong_shift():
    pos = imec0_geometry()
    source = np.zeros((len(pos), 3))
    source[0] = [1.0, -2.0, 0.5]
    target = np.zeros_like(source)
    target[4] = source[0]
    right = aligned_template_similarity(target, source, pos, 40.0)
    wrong = aligned_template_similarity(target, source, pos, 0.0)
    assert right["score"] == pytest.approx(1.0)
    assert wrong["score"] == pytest.approx(0.0)


def test_feature_depth_uses_squared_pc_energy_and_template_channel_map():
    t_f = np.zeros((2, 2, 2), dtype=float)
    t_f[0, 0, 0] = 1
    t_f[0, 1, 0] = 3
    i_cc = np.array([[0, 2], [1, 3]])
    i_u = np.array([0, 1])
    channels = feature_channel_indices(np.array([0, 1]), i_cc, i_u)
    np.testing.assert_array_equal(channels, [[0, 1], [2, 3]])
    depth, eligible = squared_pc_feature_depths(
        t_f, np.array([0, 1]), i_cc, i_u, np.array([0, 20, 40, 60])
    )
    assert depth[0] == pytest.approx(18.0)
    assert eligible.tolist() == [True, False]
    assert np.isnan(depth[1])


def test_support_is_effective_count_not_constant_confidence():
    raw, total, effective = medicine_basis_weights(
        np.array([0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 100.0]),
        np.array([0.0, 1.0]),
        np.array([0.0, 100.0]),
        time_kernel_width_s=2.0,
    )
    # MEDiCINe adds epsilon to the depth-basis width, so the opposite endpoint
    # contributes a tiny but auditable nonzero weight.
    assert raw[0, 0] == 3
    assert 2.0 < effective[0, 0] < 2.01
    assert raw[1].sum() == 0
    assert np.all(total[1] == 0)


def test_field_sampling_rejects_one_active_ineligible_corner():
    displacement = np.array([[0.0, 10.0], [20.0, 30.0]])
    eligible = np.ones((2, 2), dtype=bool)
    eligible[1, 1] = False
    value, good = sample_eligible_field(
        np.array([0.0, 1.0]), np.array([0.0, 100.0]), displacement, eligible,
        np.array([0.5, 0.0]), np.array([50.0, 0.0]),
    )
    assert not good[0] and np.isnan(value[0])
    assert good[1] and value[1] == pytest.approx(0.0)


def test_state_aggregation_requires_reciprocity_coverage_and_no_veto():
    pairs = [
        StatePair(0, 0, 0.95, 50, 50),
        StatePair(0, 1, 0.80, 50, 50),
        StatePair(1, 1, 0.93, 50, 50),
    ]
    result = aggregate_reciprocal_state_pairs(
        pairs, eligible_spikes_a=100, eligible_spikes_b=100, min_pairs=2,
        min_coverage=0.8, score_threshold=0.9, incompatibility_threshold=0.5,
    )
    assert result["reciprocal_pairs"] == [(0, 0), (1, 1)]
    assert result["accepted"]
    vetoed = list(pairs)
    vetoed[2] = StatePair(1, 1, 0.93, 50, 50, rival_veto=True)
    assert not aggregate_reciprocal_state_pairs(
        vetoed, eligible_spikes_a=100, eligible_spikes_b=100, min_pairs=2,
        min_coverage=0.8, score_threshold=0.9, incompatibility_threshold=0.5,
    )["accepted"]


def test_state_templates_use_global_zero_embedding_and_track_ineligible():
    t_f = np.array([
        [[2.0], [4.0]],
        [[6.0], [8.0]],
        [[99.0], [99.0]],
    ])
    templates, ineligible = build_motion_state_templates(
        t_f,
        cluster_ids=np.array([7, 7, 7]),
        feature_channels=np.array([[0, 2], [0, 2], [0, 2]]),
        displacement_um=np.array([39.0, 41.0, np.nan]),
        eligible=np.array([True, True, False]),
        n_channels=4,
        state_step_um=40,
        min_spikes=2,
    )
    assert len(templates) == 1
    assert templates[0].cluster == 7 and templates[0].state == 1
    np.testing.assert_allclose(templates[0].template[:, 0], [4, 0, 6, 0])
    assert ineligible == {7: 1}
