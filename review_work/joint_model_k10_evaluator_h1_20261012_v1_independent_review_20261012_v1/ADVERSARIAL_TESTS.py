"""Independent adversarial fixtures for the frozen K10 evaluator packet.

These fixtures import only the packet's frozen EVALUATOR.py.  They do not read
any joint-model arm output, recording, voltage, or DARTsort code.
"""

from __future__ import annotations

import importlib.util
import itertools
from pathlib import Path

import numpy as np
import pytest


PACKET = Path(__file__).resolve().parents[1] / "joint_model_k10_evaluator_h1_20261012_v1"
SPEC = importlib.util.spec_from_file_location("frozen_k10_evaluator", PACKET / "EVALUATOR.py")
assert SPEC is not None and SPEC.loader is not None
evaluator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluator)


def config(**changes):
    result = {
        "time_tolerance_samples": 15,
        "spatial_tolerance_um": 40.0,
        "session_bounds_samples": [0, 10_000],
        "null_shifts_samples": [2_000, 4_000],
        "clock_frame": "session_acquisition_samples",
        "clock_identity": "independent-review-session-clock",
    }
    result.update(changes)
    return result


def truth(samples, units, *, placements=None, frames=None, amplitudes=None,
          overlaps=None, decoys=None, event_ids=None):
    n = len(samples)
    return {
        "event_id": np.asarray(event_ids if event_ids is not None else [f"e{i}" for i in range(n)]),
        "unit_id": np.asarray(units, dtype=np.int64),
        "sample_index": np.asarray(samples, dtype=np.int64),
        "placement_um": np.asarray(placements if placements is not None else np.zeros(n), dtype=float),
        "frame": np.asarray(frames if frames is not None else ["tissue"] * n),
        "amplitude": np.asarray(amplitudes if amplitudes is not None else np.ones(n), dtype=float),
        "is_overlap": np.asarray(overlaps if overlaps is not None else np.zeros(n), dtype=bool),
        "decoy_group": np.asarray(decoys if decoys is not None else [""] * n),
    }


def output(samples, clusters, *, placements=None, frames=None, amplitudes=None):
    n = len(samples)
    return {
        "sample_index": np.asarray(samples, dtype=np.int64),
        "cluster_id": np.asarray(clusters, dtype=np.int64),
        "placement_um": np.asarray(placements if placements is not None else np.zeros(n), dtype=float),
        "assigned_frame": np.asarray(frames if frames is not None else ["tissue"] * n),
        "amplitude": np.asarray(amplitudes if amplitudes is not None else np.ones(n), dtype=float),
    }


def brute_cardinality(edges, n_output):
    """Independent exhaustive oracle; no evaluator matching helper is reused."""
    best = 0
    for choices in itertools.product(range(-1, n_output), repeat=len(edges)):
        selected = [value for value in choices if value >= 0]
        if len(selected) != len(set(selected)):
            continue
        if all(choice == -1 or choice in edges[i] for i, choice in enumerate(choices)):
            best = max(best, len(selected))
    return best


def test_matcher_cardinality_exclusivity_and_gate_receipts_against_exhaustive_oracle():
    rng = np.random.default_rng(20261012)
    for n_truth in range(1, 5):
        for n_output in range(1, 5):
            for _ in range(30):
                ts = rng.integers(0, 80, n_truth, dtype=np.int64)
                tz = rng.integers(-3, 4, n_truth).astype(float) * 20.0
                os = rng.integers(0, 80, n_output, dtype=np.int64)
                oz = rng.integers(-3, 4, n_output).astype(float) * 20.0
                ti, oi = evaluator.maximum_cardinality_pairs(
                    ts, tz, os, oz,
                    time_tolerance_samples=15,
                    spatial_tolerance_um=40.0,
                )
                edges = [
                    [
                        j for j in range(n_output)
                        if abs(int(os[j]) - int(ts[i])) <= 15
                        and abs(float(oz[j]) - float(tz[i])) <= 40.0
                    ]
                    for i in range(n_truth)
                ]
                assert len(ti) == brute_cardinality(edges, n_output)
                assert len(set(ti.tolist())) == len(ti)
                assert len(set(oi.tolist())) == len(oi)
                assert all(int(o) in edges[int(t)] for t, o in zip(ti, oi))


def test_both_gates_are_inclusive_and_spatial_gate_is_per_event():
    ti, oi = evaluator.maximum_cardinality_pairs(
        np.asarray([100, 200]), np.asarray([0.0, 100.0]),
        np.asarray([115, 215, 200]), np.asarray([40.0, 141.0, 0.0]),
        time_tolerance_samples=15,
        spatial_tolerance_um=40.0,
    )
    assert list(zip(ti.tolist(), oi.tolist())) == [(0, 0)]


def test_one_output_event_is_exclusive_within_truth_cluster_comparison():
    result = evaluator.evaluate_saved_outputs(
        truth([100, 101], [1, 1]), output([100], [9]), config()
    )
    score = result["units"][0]["best_score"]
    assert (score["tp"], score["fp"], score["fn"]) == (1, 0, 1)


def test_clusters_are_scored_independently_not_as_a_pooled_river():
    result = evaluator.evaluate_saved_outputs(
        truth([100, 200], [1, 1]), output([100, 200], [8, 9]), config()
    )
    row = result["units"][0]
    assert row["best_output_cluster"] == 8
    assert row["best_score"]["tp"] == 1
    assert row["capturing_clusters"] == [8, 9]
    assert row["split"] is True


def test_shifted_null_uses_same_exclusive_statistic_and_event_spatial_support():
    t = truth([100, 200], [1, 1], placements=[0.0, 100.0])
    aligned = output([100, 1100, 1200], [7, 7, 7], placements=[0.0, 0.0, 100.0])
    result = evaluator.evaluate_saved_outputs(
        t, aligned, config(null_shifts_samples=[1000])
    )
    score = result["units"][0]["best_score"]
    assert score["tp"] == 1 and score["null_match_counts"] == [2]
    assert score["above_chance"] is False

    spatially_swapped = output(
        [100, 1100, 1200], [7, 7, 7], placements=[0.0, 100.0, 0.0]
    )
    swapped_score = evaluator.evaluate_saved_outputs(
        t, spatially_swapped, config(null_shifts_samples=[1000])
    )["units"][0]["best_score"]
    assert swapped_score["tp"] == 1 and swapped_score["null_match_counts"] == [0]


def test_event_id_values_do_not_enter_matching_or_endpoint_results():
    common_output = output([101, 199], [4, 4], placements=[5.0, 5.0])
    a = evaluator.evaluate_saved_outputs(
        truth([100, 200], [1, 1], placements=[5.0, 5.0], event_ids=["a", "b"]),
        common_output,
        config(),
    )
    b = evaluator.evaluate_saved_outputs(
        truth([100, 200], [1, 1], placements=[5.0, 5.0], event_ids=["unrelated", "keys"]),
        common_output,
        config(),
    )
    assert a == b


def test_many_to_one_merge_and_decoy_endpoint_are_separate_and_explicit():
    t = truth(
        [100, 200, 300, 400], [1, 1, 2, 2],
        decoys=["pair"] * 4,
    )
    result = evaluator.evaluate_saved_outputs(t, output([100, 200, 300, 400], [6] * 4), config())
    assert result["many_to_one"] == [{"cluster_id": 6, "truth_unit_ids": [1, 2]}]
    assert [row["merged_with_truth_units"] for row in result["units"]] == [[2], [1]]
    assert result["frames"]["false_merge_decoy_groups"] == ["pair"]


def test_overlap_detection_assignment_frame_amplitude_and_residual_are_separate():
    t = truth(
        [100, 101], [1, 1], placements=[0.0, 0.0], frames=["probe", "probe"],
        amplitudes=[2.0, 4.0], overlaps=[True, True],
    )
    result = evaluator.evaluate_saved_outputs(
        t,
        output([115], [3], placements=[40.0], frames=["tissue"], amplitudes=[3.0]),
        config(),
        overlap_residuals={"before": 5.0, "after": 4.0},
    )["overlap"]
    assert result["detection_recall"] == 1.0
    assert result["best_cluster_assignment_recall"] == 0.5
    assert result["frame_assignment_recall"] == 0.0
    assert result["median_abs_relative_amplitude_error"] == 0.5
    assert result["residual_energy"] == {"before": 5.0, "after": 4.0}


def test_clock_identity_and_half_open_domain_fail_closed():
    with pytest.raises(ValueError, match="clock_identity"):
        evaluator.evaluate_saved_outputs(
            truth([100], [1]), output([100], [1]), config(clock_identity="")
        )
    with pytest.raises(ValueError, match="outside"):
        evaluator.evaluate_saved_outputs(
            truth([10_000], [1]), output([], []), config()
        )


def test_noninteger_session_bounds_fail_closed_instead_of_being_truncated():
    """Contract requires integer acquisition-sample clocks and fail-closed times."""
    with pytest.raises(ValueError, match="integer"):
        evaluator.evaluate_saved_outputs(
            truth([100], [1]),
            output([100], [1]),
            config(session_bounds_samples=[0.75, 10_000.75]),
        )


def test_create_only_writer_preserves_original_bytes_and_cleans_temporary(tmp_path):
    path = tmp_path / "nested" / "RESULT.json"
    evaluator.write_json_create_only(path, {"receipt": "first"})
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        evaluator.write_json_create_only(path, {"receipt": "replacement"})
    assert path.read_bytes() == original
    assert not list(path.parent.glob(".k10-result.*"))
