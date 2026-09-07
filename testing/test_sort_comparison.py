import json

import numpy as np
import pandas as pd
import pytest

from testing.sort_comparison import compare_sorts, correspondence


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
