import numpy as np

from testing.luke_threshold_family_diagnostic import (
    amplitude_features,
    classify_signature,
    distribution_js_divergence,
    exclusive_candidate_core_mask,
    near_coincident_marks,
    queries_near_other_baseline,
    template_features,
)


def test_exclusive_candidate_core_mask_does_not_reuse_events():
    baseline = np.array([10, 20, 30], dtype=np.int64)
    candidate = np.array([9, 10, 20, 40], dtype=np.int64)
    assert exclusive_candidate_core_mask(baseline, candidate, 1).tolist() == [True, False, True, False]


def test_near_coincident_marks_respects_unit_and_depth():
    times = np.array([10, 11, 12, 20], dtype=np.int64)
    clusters = np.array([1, 2, 1, 3], dtype=np.int64)
    depths = np.array([100.0, 110.0, 105.0, 300.0])
    assert near_coincident_marks(times, clusters, depths, 1, 75.0).tolist() == [True, True, True, False]


def test_queries_near_other_baseline_excludes_own_cluster():
    query_times = np.array([10, 20], dtype=np.int64)
    query_depths = np.array([100.0, 100.0])
    baseline_times = np.array([10, 20, 21], dtype=np.int64)
    baseline_clusters = np.array([1, 1, 2], dtype=np.int64)
    baseline_depths = np.array([100.0, 100.0, 110.0])
    assert queries_near_other_baseline(
        query_times, query_depths, baseline_times, baseline_clusters, baseline_depths, 1, 1, 75.0
    ).tolist() == [False, True]


def test_amplitude_features_are_baseline_percentiles():
    result = amplitude_features(np.array([1.0, 2.0, 3.0, 4.0]), np.array([1.5, 3.5]))
    assert result["added_amplitude_median_baseline_percentile"] == 0.5
    assert result["added_amplitude_below_baseline_median_fraction"] == 0.5


def test_template_features_identify_same_composition():
    result = template_features(np.array([1, 1, 2, 2]), np.array([1, 2, 1, 2]))
    assert result["added_template_seen_in_core_fraction"] == 1.0
    assert result["added_vs_core_template_js_divergence"] == 0.0


def test_distribution_js_divergence_is_zero_for_proportional_counts():
    assert distribution_js_divergence(np.array([1, 2, 3]), np.array([2, 4, 6])) == 0.0


def test_signature_classification_is_explicit():
    base = {
        "baseline_retention": 0.9,
        "refractory_change": 0.0,
        "unit_coincidence_excess_change": 0.0,
        "amplitude_cv_change": 0.0,
        "presence_change": 0.1,
        "active_lifetime_change": 0.0,
        "added_in_baseline_empty_bins_fraction": 0.1,
        "added_amplitude_median_baseline_percentile": 0.2,
        "added_template_seen_in_core_fraction": 0.95,
        "added_vs_core_template_js_divergence": 0.05,
    }
    assert classify_signature(base) == "recovery_compatible"
    broad = dict(
        base,
        refractory_change=0.02,
        presence_change=0.0,
        added_in_baseline_empty_bins_fraction=0.0,
        added_amplitude_median_baseline_percentile=0.6,
    )
    assert classify_signature(broad) == "broadening_compatible"
    assert classify_signature(dict(base, baseline_retention=0.7)) == "unstable_correspondence"
