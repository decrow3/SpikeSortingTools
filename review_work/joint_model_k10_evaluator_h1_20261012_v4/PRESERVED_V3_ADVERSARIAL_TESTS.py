"""Independent synthetic review fixtures for frozen K10 evaluator v3."""

from __future__ import annotations

import importlib.util
import itertools
from pathlib import Path

import numpy as np
import pytest


PACKET = Path(__file__).resolve().parents[1] / "joint_model_k10_evaluator_h1_20261012_v3"
SPEC = importlib.util.spec_from_file_location("frozen_k10_evaluator_v3", PACKET / "EVALUATOR.py")
assert SPEC is not None and SPEC.loader is not None
evaluator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluator)


def config(**changes):
    value = {
        "time_tolerance_samples": 15,
        "spatial_tolerance_um": 40.0,
        "session_bounds_samples": [0, 10_000],
        "null_shifts_samples": [2_000, 4_000],
        "clock_frame": "session_acquisition_samples",
        "clock_identity": "v3-independent-review-clock",
    }
    value.update(changes)
    return value


def truth(samples, units, *, placements=None, frames=None, amplitudes=None,
          overlaps=None, decoys=None, event_ids=None):
    n = len(samples)
    return {
        "event_id": np.asarray(event_ids if event_ids is not None else [f"e{i}" for i in range(n)]),
        "unit_id": np.asarray(units),
        "sample_index": np.asarray(samples),
        "placement_um": np.asarray(placements if placements is not None else np.zeros(n), dtype=float),
        "frame": np.asarray(frames if frames is not None else ["tissue"] * n),
        "amplitude": np.asarray(amplitudes if amplitudes is not None else np.ones(n), dtype=float),
        "is_overlap": np.asarray(overlaps if overlaps is not None else np.zeros(n), dtype=bool),
        "decoy_group": np.asarray(decoys if decoys is not None else [""] * n),
    }


def output(samples, clusters, *, placements=None, frames=None, amplitudes=None):
    n = len(samples)
    return {
        "sample_index": np.asarray(samples),
        "cluster_id": np.asarray(clusters),
        "placement_um": np.asarray(placements if placements is not None else np.zeros(n), dtype=float),
        "assigned_frame": np.asarray(frames if frames is not None else ["tissue"] * n),
        "amplitude": np.asarray(amplitudes if amplitudes is not None else np.ones(n), dtype=float),
    }


def brute_cardinality(ts, tz, os, oz, time_gate=15, spatial_gate=40.0):
    edges = [
        [
            j for j in range(len(os))
            if abs(int(os[j]) - int(ts[i])) <= time_gate
            and abs(float(oz[j]) - float(tz[i])) <= spatial_gate
        ]
        for i in range(len(ts))
    ]

    def visit(i, used):
        if i == len(edges):
            return 0
        return max(
            [visit(i + 1, used)]
            + [1 + visit(i + 1, used | {j}) for j in edges[i] if j not in used]
        )

    return visit(0, set())


def test_global_and_per_unit_pooled_credit_match_independent_oracle():
    rng = np.random.default_rng(20261012)
    for _ in range(80):
        nt = int(rng.integers(1, 5))
        no = int(rng.integers(1, 6))
        ts = rng.integers(100, 300, nt, dtype=np.int64)
        tz = rng.integers(-2, 3, nt).astype(float) * 25.0
        units = rng.integers(1, 3, nt, dtype=np.int64)
        os = rng.integers(100, 300, no, dtype=np.int64)
        oz = rng.integers(-2, 3, no).astype(float) * 25.0
        clusters = rng.integers(10, 13, no, dtype=np.int64)
        result = evaluator.evaluate_saved_outputs(
            truth(ts, units, placements=tz),
            output(os, clusters, placements=oz),
            config(),
        )
        credit = result["global_event_credit"]
        expected_global = brute_cardinality(ts, tz, os, oz)
        assert credit["unique_tp"] == expected_global
        assert len(set(credit["matched_truth_row_indices"])) == expected_global
        assert len(set(credit["matched_output_row_indices"])) == expected_global
        for ti, oi in zip(
            credit["matched_truth_row_indices"], credit["matched_output_row_indices"]
        ):
            assert abs(int(os[oi]) - int(ts[ti])) <= 15
            assert abs(float(oz[oi]) - float(tz[ti])) <= 40.0

        expected_pooled = 0
        expected_gross = 0
        for row in result["units"]:
            mask = units == row["truth_unit_id"]
            pooled = brute_cardinality(ts[mask], tz[mask], os, oz)
            expected_pooled += pooled
            assert row["pooled_output_unique_tp"] == pooled
            assert len(set(row["pooled_output_matched_truth_row_indices"])) == pooled
            assert len(set(row["pooled_output_matched_output_row_indices"])) == pooled
            gross = 0
            for cluster in np.unique(clusters):
                cmask = clusters == cluster
                gross += brute_cardinality(ts[mask], tz[mask], os[cmask], oz[cmask])
            expected_gross += gross
            assert row["gross_per_cluster_tp"] == gross
            assert row["cross_cluster_duplicate_credit_count"] == gross - pooled

        assert credit["per_unit_pooled_tp"] == expected_pooled
        assert credit["gross_pairwise_tp_not_recovery"] == expected_gross
        assert credit["cross_cluster_duplicate_credit_count"] == expected_gross - expected_pooled
        assert credit["cross_unit_competing_credit_count"] == expected_pooled - expected_global
        assert sum(credit["credited_by_truth_unit"].values()) == expected_global
        assert sum(credit["credited_by_output_cluster"].values()) == expected_global


def test_cross_cluster_and_cross_unit_credits_are_separate_known_answers():
    duplicated = evaluator.evaluate_saved_outputs(
        truth([100], [1]), output([100, 100], [10, 11]), config()
    )["global_event_credit"]
    assert (
        duplicated["gross_pairwise_tp_not_recovery"],
        duplicated["per_unit_pooled_tp"],
        duplicated["unique_tp"],
        duplicated["cross_cluster_duplicate_credit_count"],
        duplicated["cross_unit_competing_credit_count"],
    ) == (2, 1, 1, 1, 0)

    competing = evaluator.evaluate_saved_outputs(
        truth([100, 100], [1, 2]), output([100], [10]), config()
    )["global_event_credit"]
    assert (
        competing["gross_pairwise_tp_not_recovery"],
        competing["per_unit_pooled_tp"],
        competing["unique_tp"],
        competing["cross_cluster_duplicate_credit_count"],
        competing["cross_unit_competing_credit_count"],
    ) == (2, 2, 1, 0, 1)


def test_split_merge_many_to_one_null_and_overlap_endpoints_are_preserved():
    samples_a = np.arange(100, 1100, 100)
    samples_b = np.arange(1200, 2200, 100)
    ts = np.concatenate([samples_a, samples_b])
    units = np.asarray([1] * 10 + [2] * 10)
    overlaps = np.asarray([True] + [False] * 19)
    frames = np.asarray(["probe"] * 10 + ["tissue"] * 10)
    os = np.concatenate([samples_a, samples_b, samples_a[:2]])
    clusters = np.asarray([7] * 20 + [8] * 2)
    result = evaluator.evaluate_saved_outputs(
        truth(ts, units, frames=frames, overlaps=overlaps, decoys=["pair"] * 20),
        output(os, clusters, frames=["probe"] * 20 + ["tissue"] * 2),
        # Keep this preservation fixture's nulls away from its compact event span.
        config(null_shifts_samples=[3_000, 5_000]),
        overlap_residuals={"before": 3.0, "after": 2.0},
    )
    unit1, unit2 = result["units"]
    assert unit1["split"] is True and unit1["capturing_clusters"] == [7, 8]
    assert unit2["merge"] is True and unit2["merged_with_truth_units"] == [1]
    assert result["many_to_one"] == [{"cluster_id": 7, "truth_unit_ids": [1, 2]}]
    assert unit1["best_score"]["null_match_counts"] == [0, 0]
    assert result["overlap"]["truth_event_count"] == 1
    assert result["overlap"]["detection_recall"] == 1.0
    assert result["overlap"]["residual_energy"] == {"before": 3.0, "after": 2.0}


def arm_config(rules=None):
    return {
        "baseline_arm": "N0",
        "candidate_arms": ["N2", "N1"],
        "selection_primary_endpoint": "recovery",
        "selection_direction": "higher",
        "endpoint_rules": rules or {
            "recovery": {"direction": "higher", "practical_loss_margin": 0.10},
            "duplicates": {"direction": "lower", "practical_loss_margin": 0.20},
        },
    }


def holdout(candidate="N1", endpoints=None, **extra):
    value = {
        "candidate_arm": candidate,
        "baseline_arm": "N0",
        "endpoints": endpoints or {
            "recovery": {"estimate": 0.0, "ci_low": -0.10, "ci_high": 0.10, "n_units": 1},
            "duplicates": {"estimate": 0.0, "ci_low": -0.10, "ci_high": 0.20, "n_units": 1},
        },
    }
    value.update(extra)
    return value


def test_development_only_selection_is_deterministic_and_holdout_values_do_not_select():
    development = {"N1": {"recovery": 0.8}, "N2": {"recovery": 0.8}}
    a = evaluator.selection_aware_arm_report(development, holdout(), arm_config())
    extreme = holdout()
    extreme["endpoints"]["recovery"] = {
        "estimate": -999.0, "ci_low": -1000.0, "ci_high": -998.0, "n_units": 1
    }
    b = evaluator.selection_aware_arm_report(development, extreme, arm_config())
    assert a["selected_arm"] == b["selected_arm"] == "N1"
    assert a["selection_source"] == b["selection_source"] == "development_only"


def test_wrong_arm_is_rejected():
    development = {"N1": {"recovery": 0.9}, "N2": {"recovery": 0.8}}
    with pytest.raises(ValueError, match="selected arm"):
        evaluator.selection_aware_arm_report(development, holdout(candidate="N2"), arm_config())


@pytest.mark.parametrize(
    "direction,margin,low,estimate,high,expected",
    [
        ("higher", 0.10, -0.10, 0.0, 0.2, "PASS_NONINFERIOR"),
        ("higher", 0.10, -0.20, -0.15, -0.10, "INCONCLUSIVE"),
        ("higher", 0.10, -0.20, -0.15, -0.100001, "FAIL_PRACTICAL_LOSS"),
        ("lower", 0.20, -0.1, 0.0, 0.20, "PASS_NONINFERIOR"),
        ("lower", 0.20, 0.20, 0.25, 0.30, "INCONCLUSIVE"),
        ("lower", 0.20, 0.200001, 0.25, 0.30, "FAIL_PRACTICAL_LOSS"),
    ],
)
def test_higher_and_lower_margin_boundaries(direction, margin, low, estimate, high, expected):
    rules = {"recovery": {"direction": direction, "practical_loss_margin": margin}}
    endpoints = {
        "recovery": {"estimate": estimate, "ci_low": low, "ci_high": high, "n_units": 1}
    }
    report = evaluator.selection_aware_arm_report(
        {"N1": {"recovery": 1.0}, "N2": {"recovery": 0.0}},
        holdout(endpoints=endpoints),
        arm_config(rules),
    )
    assert report["endpoints"]["recovery"]["decision"] == expected
    assert report["overall_decision"] == expected
    assert report["equivalence_claim"] is False
    assert report["superiority_claim"] is False
    assert report["inconclusive_is_equivalence"] is False


@pytest.mark.parametrize(
    "decisions,expected",
    [
        (("pass", "pass"), "PASS_NONINFERIOR"),
        (("pass", "inc"), "INCONCLUSIVE"),
        (("inc", "inc"), "INCONCLUSIVE"),
        (("fail", "pass"), "FAIL_PRACTICAL_LOSS"),
        (("fail", "inc"), "FAIL_PRACTICAL_LOSS"),
    ],
)
def test_overall_aggregation(decisions, expected):
    ci = {
        "pass": {"estimate": 0.0, "ci_low": -0.1, "ci_high": 0.1, "n_units": 1},
        "inc": {"estimate": -0.15, "ci_low": -0.2, "ci_high": -0.1, "n_units": 1},
        "fail": {"estimate": -0.2, "ci_low": -0.3, "ci_high": -0.100001, "n_units": 1},
    }
    rules = {
        "a": {"direction": "higher", "practical_loss_margin": 0.1},
        "b": {"direction": "higher", "practical_loss_margin": 0.1},
    }
    report = evaluator.selection_aware_arm_report(
        {"N1": {"recovery": 1.0}, "N2": {"recovery": 0.0}},
        holdout(endpoints={"a": ci[decisions[0]], "b": ci[decisions[1]]}),
        arm_config(rules),
    )
    assert report["overall_decision"] == expected


def test_one_unit_is_reported_with_no_minimum_floor_and_no_broader_claims():
    report = evaluator.selection_aware_arm_report(
        {"N1": {"recovery": 1.0}, "N2": {"recovery": 0.0}}, holdout(), arm_config()
    )
    assert {row["n_units"] for row in report["endpoints"].values()} == {1}
    assert report["minimum_unit_floor"] is None
    assert report["equivalence_claim"] is False
    assert report["superiority_claim"] is False


@pytest.mark.parametrize(
    "bad",
    [0, -1, 1.5, True, 1 + 0j, "1", np.array(1, dtype=object), 2**63],
)
def test_invalid_n_units_fail_closed(bad):
    bad_holdout = holdout()
    bad_holdout["endpoints"]["recovery"]["n_units"] = bad
    with pytest.raises(ValueError):
        evaluator.selection_aware_arm_report(
            {"N1": {"recovery": 1.0}, "N2": {"recovery": 0.0}},
            bad_holdout,
            arm_config(),
        )


@pytest.mark.parametrize("bad", [True, "1", 1 + 0j, np.inf, np.nan, np.array([1.0])])
def test_invalid_development_numeric_types_fail_closed(bad):
    with pytest.raises(ValueError, match="finite real scalar"):
        evaluator.selection_aware_arm_report(
            {"N1": {"recovery": bad}, "N2": {"recovery": 0.0}}, holdout(), arm_config()
        )


@pytest.mark.parametrize("bad", [True, "0.1", 0.1 + 0j, np.inf, np.nan, -0.1])
def test_invalid_margin_numeric_types_or_ranges_fail_closed(bad):
    rules = {"recovery": {"direction": "higher", "practical_loss_margin": bad}}
    with pytest.raises(ValueError, match="practical loss margin"):
        evaluator.selection_aware_arm_report(
            {"N1": {"recovery": 1.0}, "N2": {"recovery": 0.0}},
            holdout(endpoints={"recovery": {"estimate": 0.0, "ci_low": -0.1, "ci_high": 0.1, "n_units": 1}}),
            arm_config(rules),
        )


@pytest.mark.parametrize(
    "bad_endpoint",
    [
        {"estimate": True, "ci_low": -0.1, "ci_high": 0.1, "n_units": 1},
        {"estimate": 0.0, "ci_low": np.nan, "ci_high": 0.1, "n_units": 1},
        {"estimate": 0.0, "ci_low": -0.1, "ci_high": np.inf, "n_units": 1},
        {"estimate": 0.2, "ci_low": -0.1, "ci_high": 0.1, "n_units": 1},
        {"estimate": 0.0, "ci_low": 0.1, "ci_high": -0.1, "n_units": 1},
    ],
)
def test_invalid_ci_numeric_types_or_ranges_fail_closed(bad_endpoint):
    rules = {"recovery": {"direction": "higher", "practical_loss_margin": 0.1}}
    with pytest.raises(ValueError):
        evaluator.selection_aware_arm_report(
            {"N1": {"recovery": 1.0}, "N2": {"recovery": 0.0}},
            holdout(endpoints={"recovery": bad_endpoint}),
            arm_config(rules),
        )


def test_nonselected_holdout_summary_is_rejected_not_silently_ignored():
    """Contract requires exactly one selected-arm-versus-baseline summary."""
    payload = holdout(non_selected_candidate_holdout={
        "candidate_arm": "N2",
        "baseline_arm": "N0",
        "endpoints": {"recovery": {"estimate": 999.0}},
    })
    with pytest.raises(ValueError, match="exactly"):
        evaluator.selection_aware_arm_report(
            {"N1": {"recovery": 1.0}, "N2": {"recovery": 0.0}}, payload, arm_config()
        )


def test_exact_clocks_and_half_open_bounds_remain_enforced():
    with pytest.raises(ValueError, match="integer"):
        evaluator.evaluate_saved_outputs(
            truth([100], [1]), output([100], [1]),
            config(session_bounds_samples=[0.5, 10_000.5]),
        )
    with pytest.raises(ValueError, match="outside"):
        evaluator.evaluate_saved_outputs(truth([10_000], [1]), output([], []), config())
    with pytest.raises(ValueError, match="integer"):
        evaluator.evaluate_saved_outputs(
            truth(np.asarray([True]), [1]), output([], []), config()
        )


def test_create_only_writer_preserves_first_result(tmp_path):
    path = tmp_path / "nested" / "RESULT.json"
    evaluator.write_json_create_only(path, {"result": "first"})
    first = path.read_bytes()
    with pytest.raises(FileExistsError):
        evaluator.write_json_create_only(path, {"result": "second"})
    assert path.read_bytes() == first
    assert not list(path.parent.glob(".k10-result.*"))
