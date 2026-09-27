import numpy as np

from testing.ci_force_gate import gated_force_union


def _sym(n, edges):
    result = np.eye(n, dtype=bool)
    for i, j in edges:
        result[i, j] = result[j, i] = True
    return result


def test_zero_force_gate_is_exact_no_force_union():
    direct = _sym(4, [(0, 1), (1, 2)])
    qda = _sym(4, [(2, 3)])
    result = gated_force_union(direct_force_mask=direct,
        secondary_pass_mask=np.zeros((4, 4), bool), qda_accept_mask=qda)
    assert np.array_equal(result["final_union_mask"], qda)
    assert result["accepted_direct_edges"] == 0


def test_gating_precedes_connectivity_and_reports_indirect_reconnection():
    direct = _sym(4, [(0, 1), (1, 2), (0, 2), (2, 3)])
    passed = np.zeros((4, 4), dtype=bool)
    passed[0, 1] = passed[1, 2] = passed[2, 3] = True
    qda = np.eye(4, dtype=bool)
    result = gated_force_union(direct_force_mask=direct,
        secondary_pass_mask=passed, qda_accept_mask=qda)
    assert result["accepted_direct_edges"] == 3
    assert result["gated_force_relations"] == 6
    assert result["rejected_direct_edges_reconnected"] == 1
    assert result["indirect_relations"] == 3
