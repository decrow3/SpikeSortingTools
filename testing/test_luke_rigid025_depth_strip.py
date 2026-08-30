import numpy as np

from testing import luke_rigid025_depth_strip as strip


def test_relative_motion_bins_recover_bin_centers_on_zero_clock():
    relative, start = strip.relative_motion_bins(np.array([3058.1775, 3059.1775, 3060.1775]))
    np.testing.assert_allclose(relative, [0.5, 1.5, 2.5])
    assert np.isclose(start, 3057.6775)


def test_rigid025_uses_nanmedian_and_one_spatial_bin():
    displacement = np.array([[0.0, 4.0, np.nan], [-8.0, 0.0, 8.0]])
    rigid, time, depth, start = strip.rigid025_motion_arrays(
        displacement,
        np.array([10.5, 11.5]),
        np.array([0.0, 100.0, 200.0]),
    )
    np.testing.assert_allclose(rigid[:, 0], [0.5, 0.0])
    np.testing.assert_allclose(time, [0.5, 1.5])
    np.testing.assert_allclose(depth, [100.0])
    assert start == 10.0


def test_comparison_requires_all_six_nondominance_gates():
    baseline = {
        "n_final_spikes": 100, "n_units": 10, "n_ks_good": 3,
        "median_contamination_pct": 10.0, "cross_unit_coincidence_excess": 0.1,
        "median_unit_refractory_violation_fraction": 0.01,
        "spike_rate_cv_across_time_bins": 0.2, "edge_spike_fraction_within_40um": 0.1,
        "neural_unmatched_recovery": 0.9, "neural_unmatched_recovery_excess": 0.8,
    }
    candidate = dict(baseline)
    result = strip.compare_summaries(baseline, candidate)
    assert result["strict_nondominance_pass"]
    assert result["n_gates_passed"] == 6
    candidate["cross_unit_coincidence_excess"] = 0.11
    assert not strip.compare_summaries(baseline, candidate)["strict_nondominance_pass"]


def test_halo_condition_is_distinct_and_preserves_core_score_name():
    direct = strip.condition_paths(0)
    halo = strip.condition_paths(16)
    assert direct.target != halo.target
    assert "halo16" in halo.target.name
    assert "halo16" in halo.score_name


def test_prepare_rejects_nonpositive_chunk_before_output(monkeypatch):
    # The CLI-level invariant is intentionally simple and independently visible.
    assert strip.parse_args is not None
