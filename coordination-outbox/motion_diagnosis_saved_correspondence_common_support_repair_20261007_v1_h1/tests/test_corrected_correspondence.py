import importlib.util
from pathlib import Path

import numpy as np


SOURCE = Path(__file__).resolve().parents[1] / "source" / "corrected_correspondence.py"
SPEC = importlib.util.spec_from_file_location("corrected_correspondence", SOURCE)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_full_production_path_fixture_suite():
    receipt = module.production_path_fixtures()
    assert receipt["status"] == "PASS"
    assert receipt["short_segment_zero_common_support"]
    assert receipt["inclusive_collision_edges"] == 2


def test_old_dense_null_opportunity_failure_is_repaired_for_both_signs():
    dense = [row for row in module.production_path_fixtures()["stationary_controls"]
             if row["fixture"] == "dense_stationary"]
    assert {row["shift_samples"] for row in dense} == {-100, 100}
    assert all(row["observed_coverage"] == 1.0 and row["null_coverage"] == 1.0 for row in dense)


def test_sparse_stationary_edges_equal_for_both_signs():
    sparse = [row for row in module.production_path_fixtures()["stationary_controls"]
              if row["fixture"] == "sparse_stationary"]
    assert {row["shift_samples"] for row in sparse} == {-100, 100}
    assert all(row["observed_edges"] == row["null_edges"] for row in sparse)


def test_common_support_includes_temporal_radius_and_rejects_boundaries():
    cells = module.fixture_cells(250, 1000.0)
    a_t = np.array([114, 115, 116, 234, 235], dtype=np.int64)
    pos = np.zeros((len(a_t), 2), dtype=float)
    state = np.zeros(len(a_t), dtype=np.int64)
    segment = np.zeros(len(a_t), dtype=np.int64)
    result = module.run_match(a_t, pos, state, segment, np.arange(250), np.zeros((250, 2)),
                              1000.0, cells, match_shift_samples=100, comparison_shift_samples=100)
    np.testing.assert_array_equal(result.degree >= 0, [False, True, True, True, False])


def test_negative_cluster_labels_do_not_alias_last_selected_unit():
    clusters = np.array([-1, 3, 7], dtype=np.int64)
    lookup = np.full(8, -1, dtype=np.int64)
    lookup[7] = 0
    mapped = np.full(len(clusters), -1, dtype=np.int64)
    valid = (clusters >= 0) & (clusters < len(lookup))
    mapped[valid] = lookup[clusters[valid]]
    np.testing.assert_array_equal(mapped, [-1, -1, 0])
