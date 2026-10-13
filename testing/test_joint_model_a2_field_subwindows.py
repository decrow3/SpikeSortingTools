import numpy as np

from testing.joint_model_a2_field_subwindows import (
    nearest_boundaries_samples,
    round_half_away_lattice,
    score_candidate,
    select_subwindow,
)


def test_round_half_away_lattice():
    values = np.array([-60.0, -20.0, -19.9, 19.9, 20.0, 60.0])
    assert np.array_equal(
        round_half_away_lattice(values),
        np.array([-80.0, -40.0, -0.0, 0.0, 40.0, 80.0]),
    )


def test_nearest_boundary_and_scores():
    times = np.array([0.0, 1.0, 2.0, 3.0])
    boundaries = nearest_boundaries_samples(times, 10.0)
    assert np.array_equal(boundaries, np.array([5, 15, 25]))
    q = np.array([0.0, 80.0, 80.0, -80.0])
    assert score_candidate(0, 20, q, boundaries) == (1, 15)
    assert score_candidate(10, 30, q, boundaries) == (1, 20)


def test_primary_then_secondary_then_earliest_tie_break():
    q = np.array([0.0, 80.0, 0.0, 80.0, 0.0])
    boundaries = np.array([10, 20, 30, 40], dtype=np.int64)
    selected = select_subwindow(
        (0, 60), q, boundaries, chunk_length_samples=10, chunks_per_subwindow=2
    )
    assert selected["selected_half_open_samples"] == [10, 30]
    assert selected["state_switch_count"] == 2
    assert selected["abs_q_at_least_80um_samples"] == 10


def test_parent_containment_and_grid():
    q = np.array([0.0, 40.0])
    boundaries = np.array([100], dtype=np.int64)
    selected = select_subwindow(
        (7, 58), q, boundaries, chunk_length_samples=10, chunks_per_subwindow=3
    )
    assert selected["selected_half_open_samples"] == [10, 40]
    assert selected["chunk_starts_samples"] == [10, 20, 30]
    assert selected["candidate_count"] == 2
