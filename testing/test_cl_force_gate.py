import numpy as np

from testing.ci_force_gate import gated_force_union


def test_optional_si_is_preserved_after_force_gate():
    direct = np.eye(4, dtype=bool)
    direct[0, 1] = direct[1, 0] = True
    secondary = np.zeros_like(direct)
    qda = np.eye(4, dtype=bool)
    si = np.eye(4, dtype=bool)
    si[2, 3] = si[3, 2] = True
    result = gated_force_union(
        direct_force_mask=direct, secondary_pass_mask=secondary,
        qda_accept_mask=qda, si_accept_mask=si)
    assert result["final_union_mask"][2, 3]
    assert not result["final_union_mask"][0, 1]
