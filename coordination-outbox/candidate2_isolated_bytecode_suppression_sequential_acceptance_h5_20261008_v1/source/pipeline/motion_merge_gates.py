"""Final, identity-independent guardrails for motion-aware merge candidates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FinalClusterMapping:
    premerge_cluster: int
    final_cluster: int
    spike_count: int
    matched_spikes: int
    purity: float


def dominant_final_cluster_mapping(
    premerge_labels: np.ndarray,
    final_labels: np.ndarray,
) -> dict[int, FinalClusterMapping]:
    """Map bounded premerge labels to existing full-session labels by event overlap."""
    pre = np.asarray(premerge_labels, dtype=np.int64).reshape(-1)
    final = np.asarray(final_labels, dtype=np.int64).reshape(-1)
    if pre.shape != final.shape:
        raise ValueError("premerge and final labels must align spike-for-spike")
    result = {}
    for cluster in np.unique(pre):
        values, counts = np.unique(final[pre == cluster], return_counts=True)
        best = int(np.argmax(counts))
        n = int(np.sum(counts))
        result[int(cluster)] = FinalClusterMapping(
            premerge_cluster=int(cluster),
            final_cluster=int(values[best]),
            spike_count=n,
            matched_spikes=int(counts[best]),
            purity=float(counts[best] / n),
        )
    return result


def endpoint_rival_score(
    pair_rows: list[dict[str, object]],
    cluster: int,
    state: int,
    intended_cluster: int,
    intended_state: int,
) -> tuple[float, int | None, int | None]:
    """Return the best outside-cluster rival for one motion-state endpoint.

    The intended partner and every state from its cluster are excluded. This
    prevents another state of the proposed merge partner from being mislabeled
    as an independent rival.
    """
    del intended_state  # retained in the API so receipts fully identify the link
    candidates: list[tuple[float, int, int]] = []
    for row in pair_rows:
        ca, cb = int(row["cluster_a"]), int(row["cluster_b"])
        sa, sb = int(row["state_a"]), int(row["state_b"])
        if ca == cluster and sa == state:
            other_cluster, other_state = cb, sb
        elif cb == cluster and sb == state:
            other_cluster, other_state = ca, sa
        else:
            continue
        score = float(row["score"])
        if other_cluster != intended_cluster and np.isfinite(score):
            candidates.append((score, other_cluster, other_state))
    if not candidates:
        return np.nan, None, None
    return max(candidates, key=lambda item: item[0])


def family_support(
    final_a: int,
    final_b: int,
    family_by_cluster: dict[int, str],
    family_class: dict[str, str],
) -> tuple[bool, str]:
    """Require both identities in the same independently coherent family."""
    fa, fb = family_by_cluster.get(final_a), family_by_cluster.get(final_b)
    if fa is None or fb is None:
        return False, "one_or_both_final_identities_unmatched"
    if fa != fb:
        return False, "different_waveform_only_families"
    if family_class.get(fa) != "depth_time_coherent":
        return False, "shared_family_not_depth_time_coherent"
    return True, "same_independently_coherent_waveform_only_family"
