import json

import numpy as np
import pytest

from testing.joint_model_k10_evaluator import (
    evaluate_saved_outputs,
    maximum_cardinality_pairs,
    write_json_create_only,
)


def config(**changes):
    value = {
        "time_tolerance_samples": 15,
        "spatial_tolerance_um": 40.0,
        "session_bounds_samples": [0, 100_000],
        "null_shifts_samples": [20_000, 40_000, 60_000],
        "clock_frame": "session_acquisition_samples",
        "clock_identity": "fixture-session-clock-v1",
    }
    value.update(changes)
    return value


def truth(
    samples,
    units,
    *,
    placements=None,
    frames=None,
    amplitudes=None,
    overlaps=None,
    decoys=None,
    event_ids=None,
):
    n = len(samples)
    return {
        "event_id": np.asarray(event_ids if event_ids is not None else [f"truth-{i}" for i in range(n)]),
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


def test_maximum_cardinality_reassigns_early_preferred_output():
    # t0 first prefers o0 but can use o1; t1 can only use o0 spatially.
    ti, oi = maximum_cardinality_pairs(
        np.array([100, 101]), np.array([0.0, 80.0]),
        np.array([100, 110]), np.array([40.0, 0.0]),
        time_tolerance_samples=15, spatial_tolerance_um=40.0,
    )
    assert len(ti) == len(oi) == 2
    assert set(zip(ti.tolist(), oi.tolist())) == {(0, 1), (1, 0)}


def test_maximum_cardinality_matches_bruteforce_small_graphs():
    rng = np.random.default_rng(4)
    for _ in range(100):
        nt, no = 4, 5
        ts = rng.integers(0, 50, nt)
        tz = rng.integers(0, 5, nt) * 20.0
        os = rng.integers(0, 50, no)
        oz = rng.integers(0, 5, no) * 20.0
        ti, _ = maximum_cardinality_pairs(
            ts, tz, os, oz, time_tolerance_samples=8, spatial_tolerance_um=20.0
        )
        edges = [
            [j for j in range(no) if abs(int(os[j] - ts[i])) <= 8 and abs(float(oz[j] - tz[i])) <= 20]
            for i in range(nt)
        ]

        def brute(i, used):
            if i == nt:
                return 0
            return max(
                [brute(i + 1, used)]
                + [1 + brute(i + 1, used | {j}) for j in edges[i] if j not in used]
            )

        assert len(ti) == brute(0, set())


def test_perfect_recovery_uses_time_and_placement_not_event_keys():
    t = truth([100, 200, 300], [1, 1, 1], event_ids=["not", "shared", "keys"])
    o = output([101, 199, 300], [10, 10, 10], placements=[1, -1, 0])
    result = evaluate_saved_outputs(t, o, config())
    row = result["units"][0]
    assert row["best_score"]["tp"] == 3
    assert row["best_score"]["accuracy"] == 1.0
    assert row["split"] is False and row["merge"] is False


def test_spatial_gate_rejects_same_time_wrong_placement():
    result = evaluate_saved_outputs(
        truth([100], [1], placements=[0]),
        output([100], [10], placements=[80]),
        config(spatial_tolerance_um=40.0),
    )
    assert result["units"][0]["best_output_cluster"] is None
    assert result["units"][0]["best_score"] is None


def test_split_requires_capture_and_shifted_null_control():
    samples = np.arange(100, 2100, 100)
    t = truth(samples, np.ones(20, dtype=int))
    o = output(samples, [10] * 10 + [11] * 10)
    result = evaluate_saved_outputs(t, o, config())
    row = result["units"][0]
    assert row["capturing_clusters"] == [10, 11]
    assert row["split"] is True


def test_shifted_null_suppresses_periodic_chance_capture():
    samples = np.arange(100, 2100, 100)
    periodic = np.arange(0, 100_000, 100)
    result = evaluate_saved_outputs(
        truth(samples, np.ones(samples.size, dtype=int)),
        output(periodic, np.full(periodic.size, 88)),
        config(null_shifts_samples=[20_000, 40_000]),
    )
    row = result["units"][0]
    assert row["best_score"]["tp"] == samples.size
    assert row["best_score"]["null_max"] == samples.size
    assert row["best_score"]["above_chance"] is False
    assert row["capturing_clusters"] == []


def test_many_to_one_merge_and_decoy_false_merge_are_separate_endpoints():
    a = np.arange(100, 1100, 100)
    b = np.arange(1150, 2150, 100)
    t = truth(
        np.concatenate([a, b]),
        [1] * len(a) + [2] * len(b),
        decoys=["pair-a"] * (len(a) + len(b)),
    )
    o = output(np.concatenate([a, b]), [7] * (len(a) + len(b)))
    result = evaluate_saved_outputs(t, o, config())
    assert result["many_to_one"] == [{"cluster_id": 7, "truth_unit_ids": [1, 2]}]
    assert all(row["merge"] for row in result["units"])
    assert result["frames"]["decoy_false_merge_rate"] == 1.0


def test_overlap_scores_detection_assignment_frame_amplitude_and_residual_separately():
    t = truth(
        [100, 200], [1, 1], frames=["probe", "probe"],
        amplitudes=[2.0, 4.0], overlaps=[True, True],
    )
    o = output(
        [100, 200], [9, 9], frames=["probe", "probe"], amplitudes=[2.2, 3.6]
    )
    result = evaluate_saved_outputs(
        t, o, config(), overlap_residuals={"before": 10.0, "after": 4.0}
    )
    overlap = result["overlap"]
    assert overlap["detection_recall"] == 1.0
    assert overlap["best_cluster_assignment_recall"] == 1.0
    assert overlap["frame_assignment_recall"] == 1.0
    assert overlap["median_abs_relative_amplitude_error"] == pytest.approx(0.1)
    assert overlap["residual_energy"] == {"before": 10.0, "after": 4.0}


def test_overlap_detection_is_not_conflated_with_exclusive_assignment():
    t = truth([100, 101], [1, 1], overlaps=[True, True])
    o = output([100], [9])
    overlap = evaluate_saved_outputs(t, o, config())["overlap"]
    assert overlap["detection_recall"] == 1.0
    assert overlap["best_cluster_assignment_recall"] == 0.5


def test_frame_recall_retention_and_false_probe_rate():
    t = truth([100, 200, 300], [1, 2, 3], frames=["probe", "tissue", "tissue"])
    o = output([100, 200, 300], [11, 12, 13], frames=["probe", "tissue", "probe"])
    result = evaluate_saved_outputs(t, o, config())
    assert result["frames"]["probe_fixed_recall"] == 1.0
    assert result["frames"]["tissue_fixed_retention"] == 0.5
    assert result["frames"]["false_probe_fixed_rate"] == 0.5


@pytest.mark.parametrize(
    "bad_truth,bad_output,bad_config,match",
    [
        (truth([100, 101], [1, 1], event_ids=["x", "x"]), output([], []), config(), "unique"),
        ({**truth([100], [1]), "sample_index": np.array([100.5])}, output([], []), config(), "integer"),
        (truth([100], [1]), output([], []), config(time_tolerance_samples=14), "frozen"),
        (truth([100], [1]), output([], []), config(null_shifts_samples=[]), "required"),
        (truth([100], [1]), output([], []), config(session_bounds_samples=[0.75, 100_000.75]), "integer"),
        (truth([100_000], [1]), output([], []), config(), "outside"),
        (truth([100], [1]), output([], []), config(clock_frame="crop_local_samples"), "clock_frame"),
    ],
)
def test_fail_closed_input_guards(bad_truth, bad_output, bad_config, match):
    with pytest.raises(ValueError, match=match):
        evaluate_saved_outputs(bad_truth, bad_output, bad_config)


def test_unsorted_inputs_are_normalized_without_changing_original_row_receipts():
    t = truth([300, 100, 200], [1, 1, 1])
    o = output([199, 301, 99], [8, 8, 8])
    result = evaluate_saved_outputs(t, o, config())
    score = result["units"][0]["best_score"]
    assert score["tp"] == 3
    assert sorted(score["matched_truth_row_indices"]) == [0, 1, 2]
    assert sorted(score["matched_output_row_indices"]) == [0, 1, 2]


def test_background_and_cost_are_passed_through_without_imputation():
    result = evaluate_saved_outputs(
        truth([100], [1]), output([100], [1]), config(),
        background={"baseline_real_units": 12, "candidate_real_units": 11},
        cost={"runtime_seconds": 3.2, "peak_memory_bytes": 1000},
    )
    assert result["background"]["candidate_real_units"] == 11
    assert result["cost"]["peak_memory_bytes"] == 1000


def test_result_write_is_complete_and_never_overwrites(tmp_path):
    path = tmp_path / "result" / "RESULT.json"
    write_json_create_only(path, {"status": "first"})
    before = path.read_bytes()
    assert json.loads(before) == {"status": "first"}
    with pytest.raises(FileExistsError):
        write_json_create_only(path, {"status": "second"})
    assert path.read_bytes() == before
