import json

import numpy as np
import pandas as pd
import pytest

from testing.sort_comparison import compare_sorts, correspondence, load_comparison_inputs
from pipeline.completeness_timeline import classify_fit_trust


def pop(times, clusters, *, labels=None, depths=None, amplitudes=None):
    result = {
        "st": np.asarray(times, dtype=np.int64),
        "cl": np.asarray(clusters, dtype=np.int64),
        "labels": labels or {},
        "identity_digest": "identity",
    }
    if depths is not None:
        result["depth"] = np.asarray(depths, dtype=float)
    if amplitudes is not None:
        result["amp"] = np.asarray(amplitudes, dtype=float)
    return result


def windows(cluster, values, *, starts=None, status="finite_interior"):
    starts = np.arange(len(values), dtype=float) if starts is None else np.asarray(starts, dtype=float)
    return pd.DataFrame({
        "cluster_id": cluster,
        "status": status,
        "start_s": starts,
        "end_s": starts + 1,
        "missing_pct": values,
        "fit_x0": 1.0,
        "fit_k": 1.0,
        "fit_A": 1.0 / (1.0 - np.asarray(values, dtype=float) / 100.0),
    })


def config():
    return {
        "sampling_frequency_hz": 1000.0,
        "duration_s": 4.0,
        "correspondence_tolerance_ms": 1.0,
        "minimum_correspondence_overlap": 0.1,
        "primary_retention": 0.5,
        "minimum_valid_amplitude_windows": 2,
        "minimum_common_time_fraction": 0.5,
        "minimum_measurable_unit_fraction": 0.5,
        "coincidence_tolerance_ms": 1.0,
        "coincidence_depth_um": 75.0,
        "coincidence_seed": 4,
        "longitudinal_bin_s": 1.0,
    }


def test_lost_units_stay_in_coverage_denominator():
    baseline = pop([100,200,300,1100,1200,1300,2100,2200,2300,3100,3200,3300], [1,2,3]*4)
    candidate = pop([100,1100,2100,3100], [11]*4)
    bq = {"amplitude_windows": pd.concat([windows(c,[20]*4) for c in [1,2,3]], ignore_index=True)}
    cq = {"amplitude_windows": windows(11,[1]*4)}
    report = compare_sorts(baseline,candidate,bq,cq,config())
    coverage = report["coverage_summary"]
    assert coverage["baseline_eligible_units"] == 3
    assert coverage["amplitude_measurable_fraction"] == pytest.approx(1/3)
    assert coverage["matched_pair_conditional_measurable_fraction"] == 1
    assert coverage["baseline_cohort_status_counts"] == {"unmatched":2,"measurable":1}
    assert report["decision"]["status"] == "endpoint_infeasible"


def test_no_correspondence_reports_zero_coverage():
    baseline=pop([100,1100,2100,3100],[1]*4)
    candidate=pop([500,1500,2500,3500],[2]*4)
    report=compare_sorts(baseline,candidate,{"amplitude_windows":windows(1,[20]*4)},
                        {"amplitude_windows":windows(2,[1]*4)},config())
    assert report["coverage_summary"]["amplitude_measurable_fraction"]==0
    assert report["baseline_eligibility"].status.tolist()==["unmatched"]


def test_clear_split_merge_unmatched_and_good_to_mua_are_retained():
    baseline = pop(
        [100, 200, 300, 400, 500, 600], [1, 1, 1, 1, 9, 9], labels={1: "good", 9: "mua"}
    )
    candidate = pop(
        [100, 200, 300, 400, 700, 800], [2, 2, 3, 3, 8, 8], labels={2: "mua", 3: "mua", 8: "good"}
    )
    edges = correspondence(baseline, candidate, tolerance=0)
    assert set(zip(edges.baseline_cluster, edges.candidate_cluster)) == {(1, 2), (1, 3)}
    assert not edges.primary_match.any()  # tied one-to-many split remains ambiguous
    assert 9 not in set(edges.baseline_cluster) and 8 not in set(edges.candidate_cluster)
    # Labels never gate graph construction: a non-tied good -> MUA train is primary.
    gm = correspondence(
        pop([10, 20, 30], [1, 1, 1], labels={1: "good"}),
        pop([10, 20, 30], [2, 2, 2], labels={2: "mua"}),
        0,
    )
    assert gm.primary_match.item()


def test_many_to_one_merge_is_visible_in_full_edge_table():
    baseline = pop([100, 200, 300, 400], [1, 1, 2, 2])
    candidate = pop([100, 200, 300, 400], [3, 3, 3, 3])
    edges = correspondence(baseline, candidate, 0)
    assert set(edges.baseline_cluster) == {1, 2}
    assert not edges.primary_match.any()


def test_dense_background_does_not_steal_primary_events():
    baseline = pop([100, 200, 300], [1, 1, 1])
    candidate = pop([99, 100, 101, 199, 200, 201, 299, 300, 301], [8, 2, 8, 8, 2, 8, 8, 2, 8])
    edges = correspondence(baseline, candidate, 0)
    primary = edges[edges.primary_match]
    assert list(primary.candidate_cluster) == [2]
    assert primary.matched_events.item() == 3


def test_comparator_propagates_coverage_and_applies_interior_rule(tmp_path):
    baseline = pop(
        [100, 200, 1100, 1200], [1, 1, 2, 2],
        depths=[200, 200, 20, 20], amplitudes=[10, 11, 12, 13],
    )
    candidate = pop(
        [100, 200, 1100, 1200], [11, 11, 12, 12],
        depths=[200, 200, 20, 20], amplitudes=[10, 11, 12, 13],
    )
    bw = pd.concat([windows(1, [20, 10], starts=[0, 2]), windows(2, [5, 5], starts=[0, 2])])
    cw = pd.concat([windows(11, [10, 5], starts=[0, 2]), windows(12, [5, 5], starts=[0, 2])])
    report = compare_sorts(
        baseline, candidate,
        {"amplitude_windows": bw, "request_digest": "bqc"},
        {"amplitude_windows": cw, "request_digest": "cqc"},
        config(),
        spatial_region={
            "processing_depth_um": [0, 400], "scoring_depth_um": [100, 300],
            "minimum_edge_exclusion_um": 50,
        },
        output_dir=tmp_path,
    )
    assert report["coverage_summary"]["interior_primary_matches"] == 1
    assert report["coverage_summary"]["amplitude_measurable_both_common_time"] == 1
    assert report["summary"]["median_missingness_improvement_pp"] == 7.5
    assert set(report["unit_metrics_baseline"].spatial_class) == {"interior", "edge"}
    expected = {
        "summary.json", "candidate_manifest.json", "correspondence_edges.csv",
        "primary_matches.csv", "split_merge_summary.json", "amplitude_windows.csv",
        "amplitude_completeness_pairs.csv", "unit_metrics_baseline.csv",
        "unit_metrics_candidate.csv", "guardrail_summary.csv", "coverage_summary.json",
        "decision.json",
    }
    assert expected <= {path.name for path in tmp_path.iterdir()}
    assert json.loads((tmp_path / "decision.json").read_text())["automatic_rank"] is None


def test_fit_failure_and_insufficient_nominal_support_remain_infeasible():
    baseline = pop([100, 200], [1, 1])
    candidate = pop([100, 200], [2, 2])
    failed = windows(1, [np.nan], status="nonfinite_fit")
    candidate_windows = windows(2, [5])
    report = compare_sorts(
        baseline, candidate,
        {"amplitude_windows": failed}, {"amplitude_windows": candidate_windows}, config()
    )
    assert report["coverage_summary"]["endpoint_status"] == "infeasible_insufficient_coverage"
    assert not report["amplitude_completeness_pairs"].measurable.item()


def test_swapping_amplitude_arms_reverses_signed_difference():
    a, b = pop([100, 200], [1, 1]), pop([100, 200], [2, 2])
    aw, bw = windows(1, [20, 10], starts=[0, 2]), windows(2, [10, 5], starts=[0, 2])
    ab = compare_sorts(a, b, {"amplitude_windows": aw}, {"amplitude_windows": bw}, config())
    ba = compare_sorts(b, a, {"amplitude_windows": bw}, {"amplitude_windows": aw}, config())
    x = ab["amplitude_completeness_pairs"].baseline_minus_candidate_missingness_pp.item()
    y = ba["amplitude_completeness_pairs"].baseline_minus_candidate_missingness_pp.item()
    assert x == -y


def test_wrong_recording_clock_is_rejected():
    baseline = pop([100, 5000], [1, 1])
    candidate = pop([100, 200], [2, 2])
    with pytest.raises(ValueError, match="recording clock"):
        compare_sorts(
            baseline, candidate,
            {"amplitude_windows": windows(1, [1, 1])},
            {"amplitude_windows": windows(2, [1, 1])},
            config(),
        )


SPATIAL = {
    "processing_depth_um": [0, 400], "scoring_depth_um": [100, 300],
    "minimum_edge_exclusion_um": 50,
}


def test_outside_units_cannot_change_population_guardrails():
    # The outside unit is highly refractory and coincides with an edge unit.
    # Previously it changed all three contamination/boundary summaries.
    inside = pop([100, 101, 1100, 2100, 2101, 3100], [1, 1, 2, 1, 1, 2],
                 depths=[20, 20, 200, 20, 20, 200])
    extended = pop([100, 100, 101, 101, 102, 1100, 2100, 2100, 2101, 2101, 2102, 3100],
                   [1, 9, 1, 9, 9, 2, 1, 9, 1, 9, 9, 2],
                   depths=[20, -10, 20, -10, -10, 200, 20, -10, 20, -10, -10, 200])
    qc = {"amplitude_windows": pd.concat([windows(c, [10]*4) for c in (1, 2, 9)])}
    clean = compare_sorts(inside, inside, qc, qc, config(), SPATIAL)
    expanded = compare_sorts(extended, inside, qc, qc, config(), SPATIAL)
    pd.testing.assert_frame_equal(clean["guardrail_summary"], expanded["guardrail_summary"])
    assert expanded["summary"]["baseline_units"] == 3  # full inventory retained
    assert expanded["coverage_summary"]["baseline_eligible_units"] == 1
    edge = expanded["guardrail_summary"].set_index("metric").loc["edge_unit_fraction"]
    assert edge.baseline == edge.candidate == 0.5


def test_spike_guardrails_exclude_excursions_outside_processing_range():
    # Median depth is interior but one event is outside; it must not dilute
    # the edge-spike denominator (two edge events / five supported events).
    a = pop([100, 200, 300, 1100, 1200, 1300], [1, 1, 1, 2, 2, 2],
            depths=[200, 200, -10, 20, 20, 200])
    qc = {"amplitude_windows": pd.concat([windows(c, [10]*4) for c in (1, 2)])}
    report = compare_sorts(a, a, qc, qc, config(), SPATIAL)
    guard = report["guardrail_summary"].set_index("metric")
    assert guard.loc["edge_spike_fraction", "baseline"] == pytest.approx(2/5)


def test_empty_common_domain_marks_guardrails_unavailable():
    a = pop([100, 200], [1, 1], depths=[-20, -20])
    qc = {"amplitude_windows": windows(1, [10]*4)}
    report = compare_sorts(a, a, qc, qc, config(), SPATIAL)
    assert not report["guardrail_summary"].available.any()
    assert report["coverage_summary"]["endpoint_status"] == "infeasible_insufficient_coverage"


def test_spatial_contract_requires_depths():
    a = pop([100, 200], [1, 1])
    qc = {"amplitude_windows": windows(1, [10]*4)}
    with pytest.raises(ValueError, match="finite spike depths"):
        compare_sorts(a, a, qc, qc, config(), SPATIAL)


def test_disagreeing_fits_do_not_count_as_completeness_support():
    a = pop([100, 1100, 2100, 3100], [1]*4)
    trusted = windows(1, [20]*4)
    untrusted = trusted.copy()
    untrusted["fit_A"] = 1.0  # normalization estimate 0%, shape estimate 20%
    report = compare_sorts(a, a, {"amplitude_windows": trusted},
                           {"amplitude_windows": untrusted}, config())
    assert report["coverage_summary"]["amplitude_measurable_both_common_time"] == 0
    assert report["amplitude_completeness_pairs"].candidate_valid_windows.item() == 0
    saved = report["amplitude_windows"]
    assert (saved.loc[saved['sort']=='candidate', 'measurement_status'] == 'poor_fit_disagreement').all()
    assert report["candidate_manifest"]["fit_trust_policy"]["fit_disagreement_warning_pp"] == 5.0


def test_shared_fit_trust_boundaries_and_status_precedence():
    _, _, status = classify_fit_trust(
        [5.0, 5.001, 50.0, np.nan, 10.0],
        [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, np.nan]],
    )
    assert status.tolist() == ['measured', 'poor_fit_disagreement',
                               'censored_at_least_50pct', 'nonfinite', 'nonfinite']


def test_missing_fit_trust_inputs_fail_closed():
    a = pop([100, 200], [1, 1])
    qc = {"amplitude_windows": windows(1, [10]*4).drop(columns="fit_A")}
    with pytest.raises(ValueError, match="fit-trust inputs"):
        compare_sorts(a, a, qc, qc, config())


def test_depth_bytes_change_identity_and_loaded_arrays_are_stable(tmp_path):
    curated = tmp_path / 'cur'
    qc = tmp_path / 'qc'
    curated.mkdir()
    (qc / 'amp_truncation').mkdir(parents=True)
    np.save(curated / 'spike_times.npy', np.array([100, 200], dtype=np.int64))
    np.save(curated / 'spike_clusters.npy', np.array([1, 1], dtype=np.int64))
    np.save(curated / 'full_st.npy', np.array([[100, 0, 10], [200, 0, 11]], dtype=float))
    np.save(curated / 'kept_spikes.npy', np.array([True, True]))
    (curated / 'cluster_KSLabel.tsv').write_text('cluster_id\tKSLabel\n1\tgood\n')
    np.savez(qc / 'amp_truncation/truncation_qc.npz', cid=np.empty(0, dtype=int),
             window_blocks=np.empty((0, 2), dtype=int), popts=np.empty((0, 3)), mpcts=np.empty(0))
    np.save(curated / 'spike_positions.npy', np.array([[0, 200], [0, 200]], dtype=float))
    a, _ = load_comparison_inputs('test', curated, qc, sampling_frequency_hz=1000)
    np.save(curated / 'spike_positions.npy', np.array([[0, 20], [0, 20]], dtype=float))
    b, _ = load_comparison_inputs('test', curated, qc, sampling_frequency_hz=1000)
    assert a['identity_digest'] != b['identity_digest']
    np.testing.assert_array_equal(a['depth'], [200, 200])
    np.testing.assert_array_equal(b['depth'], [20, 20])
    np.save(curated / 'spike_positions.npy', np.array([[0, np.nan], [0, 20]]))
    with pytest.raises(ValueError, match='nonfinite spike depths'):
        load_comparison_inputs('test', curated, qc, sampling_frequency_hz=1000)


def test_v3_refuses_to_overwrite_prior_comparison(tmp_path):
    prior = '{"request_digest": "historical-v2"}\n'
    (tmp_path / 'candidate_manifest.json').write_text(prior)
    (tmp_path / 'summary.json').write_text('historical result\n')
    a = pop([100, 200], [1, 1])
    qc = {"amplitude_windows": windows(1, [10]*4)}
    with pytest.raises(RuntimeError, match='another request'):
        compare_sorts(a, a, qc, qc, config(), output_dir=tmp_path)
    assert (tmp_path / 'candidate_manifest.json').read_text() == prior
    assert (tmp_path / 'summary.json').read_text() == 'historical result\n'
