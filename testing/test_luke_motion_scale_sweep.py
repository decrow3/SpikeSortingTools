import numpy as np
from pathlib import Path

from testing.luke_motion_scale_sweep import (
    CANDIDATES,
    _load_run_field,
    extra_quality,
    run_spec,
    select_split,
    spec_hash,
    stable_peak_split,
)


def test_load_run_field_broadcasts_rigid_trace_across_depth(tmp_path: Path):
    np.save(tmp_path / "motion.npy", np.array([[0.0], [2.0], [4.0]]))
    np.save(tmp_path / "time_bins.npy", np.array([0.0, 1.0, 2.0]))
    np.save(tmp_path / "depth_bins.npy", np.array([1000.0]))
    field = _load_run_field(tmp_path, np.array([0.5, 1.5]), np.array([300.0, 500.0, 700.0]))
    assert np.array_equal(field, np.array([[1.0, 1.0, 1.0], [3.0, 3.0, 3.0]]))


def test_peak_halves_are_deterministic_disjoint_and_complete():
    peaks = np.zeros(100, dtype=[("sample_index", "i8")])
    locations = np.zeros(100, dtype=[("y", "f8")])
    a_peaks, _ = select_split(peaks, locations, "half_a", seed=7)
    b_peaks, _ = select_split(peaks, locations, "half_b", seed=7)
    assert len(a_peaks) + len(b_peaks) == 100
    assert np.array_equal(stable_peak_split(100, 7), stable_peak_split(100, 7))


def test_spec_hash_changes_with_probe_split_or_parameters():
    candidate = CANDIDATES[0]
    hashes = {
        spec_hash(run_spec("imec0", candidate, "full", 1)),
        spec_hash(run_spec("imec1", candidate, "full", 1)),
        spec_hash(run_spec("imec0", candidate, "half_a", 1)),
        spec_hash(run_spec("imec0", candidate, "full", 2)),
    }
    assert len(hashes) == 4


def test_extra_quality_summarizes_off_diagonal_correlations_and_weights():
    correlations = np.array([[[1.0, 0.2, 0.05], [0.2, 1.0, 0.3], [0.05, 0.3, 1.0]]])
    weights = np.array([[[0.0, 1.0, 0.0], [1.0, 0.0, 2.0], [0.0, 2.0, 0.0]]])
    result = extra_quality({"C": correlations, "U": weights}, mincorr=0.1)
    assert result["n_pair_correlations"] == 3
    assert np.isclose(result["median_pair_correlation"], 0.2)
    assert np.isclose(result["pair_correlation_fraction_ge_mincorr"], 2 / 3)
    assert np.isclose(result["positive_weight_fraction"], 4 / 9)
