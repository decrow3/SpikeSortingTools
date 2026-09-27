"""Apply a frozen secondary gate to direct force edges before route union."""
from __future__ import annotations

import numpy as np
from scipy.sparse.csgraph import connected_components


def gated_force_union(*, direct_force_mask, secondary_pass_mask,
                      qda_accept_mask, si_accept_mask=None):
    """Gate direct edges, reconnect components, then union unchanged routes.

    Unknown secondary tests must be represented as False in
    ``secondary_pass_mask``. This withholds force but is not evidence that the
    pair is biologically different. QDA and optional SI routes are never gated.
    """
    direct = np.asarray(direct_force_mask, bool)
    passed = np.asarray(secondary_pass_mask, bool)
    qda = np.asarray(qda_accept_mask, bool)
    if direct.ndim != 2 or direct.shape[0] != direct.shape[1]:
        raise ValueError("direct force mask must be square")
    if passed.shape != direct.shape or qda.shape != direct.shape:
        raise ValueError("all route masks must share shape")
    if not np.array_equal(direct, direct.T) or not np.array_equal(qda, qda.T):
        raise ValueError("route masks must be symmetric")
    if si_accept_mask is None:
        si = np.zeros_like(direct)
    else:
        si = np.asarray(si_accept_mask, bool)
        if si.shape != direct.shape or not np.array_equal(si, si.T):
            raise ValueError("SI mask must be same-shape symmetric")

    n = len(direct)
    direct_no_diag = direct.copy()
    np.fill_diagonal(direct_no_diag, False)
    # The frozen edge table is naturally strict-upper-triangle; accept either
    # that representation or an already symmetrized pass mask.
    pass_no_diag = passed | passed.T
    np.fill_diagonal(pass_no_diag, False)
    accepted_direct = direct_no_diag & pass_no_diag
    _, component_ids = connected_components(accepted_direct, directed=False)
    gated_force = component_ids[:, None] == component_ids[None, :]
    final_union = gated_force | qda | si
    np.fill_diagonal(final_union, True)

    upper = np.triu_indices(n, 1)
    rejected_reconnected = direct_no_diag & ~accepted_direct & gated_force
    transitive_only = gated_force & ~accepted_direct
    return {
        "accepted_direct_mask": accepted_direct,
        "gated_force_mask": gated_force,
        "final_union_mask": final_union,
        "component_ids": component_ids,
        "accepted_direct_edges": int(accepted_direct[upper].sum()),
        "gated_force_relations": int(gated_force[upper].sum()),
        "indirect_relations": int(transitive_only[upper].sum()),
        "rejected_direct_edges_reconnected": int(rejected_reconnected[upper].sum()),
    }
