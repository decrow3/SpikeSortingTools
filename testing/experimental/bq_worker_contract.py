"""Source-only BR checks for a future BQ worker repair.

This module does not import DARTsort, read recordings, or run clustering.  It
validates orchestration facts which can be checked using source text and tiny
arrays before another service is authorized.
"""

from __future__ import annotations

import ast
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from testing.experimental.agglomeration_route_provenance import (
    source_consumed_linkage_distance,
    source_linkage_components,
)


class WorkerContractError(ValueError):
    pass


@dataclass(frozen=True)
class PartitionDelta:
    event_count: int
    changed_event_count: int
    unchanged_group_count: int
    changed_before_group_count: int
    changed_after_group_count: int


def keyword_only_parameters(source: Path, function: str, *, class_name: str | None = None) -> tuple[str, ...]:
    """Read a function's keyword-only parameters without importing its package."""

    tree = ast.parse(Path(source).read_text())
    body = tree.body
    if class_name is not None:
        classes = [node for node in body if isinstance(node, ast.ClassDef) and node.name == class_name]
        if len(classes) != 1:
            raise WorkerContractError(f"expected one class {class_name}")
        body = classes[0].body
    funcs = [
        node for node in body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function
    ]
    if len(funcs) != 1:
        raise WorkerContractError(f"expected one function {function}")
    return tuple(arg.arg for arg in funcs[0].args.kwonlyargs)


def validate_prefix_contract(contract: Mapping[str, object]) -> None:
    """Reject an implicit or incomplete pcmerge/TMM/agglomerate prefix."""

    required = {
        "stages", "recluster_after_matching", "waveform_config_sha256",
        "template_state_sha256", "rng_state_sha256", "merge_cutoff",
        "force_cutoff", "cross_merge_cutoff", "cross_merge_cutoff_used",
    }
    if set(contract) != required:
        raise WorkerContractError("effective prefix fields must be exact")
    if tuple(contract["stages"]) != ("pcmerge", "tmm", "agglomerate"):
        raise WorkerContractError("effective prefix must preserve pcmerge -> TMM -> agglomerate")
    if contract["recluster_after_matching"] is not False:
        raise WorkerContractError("recluster_after_matching must be explicitly false")
    for name in ("waveform_config_sha256", "template_state_sha256", "rng_state_sha256"):
        value = contract[name]
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise WorkerContractError(f"{name} must be an explicit SHA-256")
    if contract["merge_cutoff"] != 0.6 or contract["force_cutoff"] != 0.3:
        raise WorkerContractError("executed merge/force cutoffs must be 0.6/0.3")
    if contract["cross_merge_cutoff"] != 0.5 or contract["cross_merge_cutoff_used"] is not False:
        raise WorkerContractError("cross-merge 0.5 must be recorded as inactive for agglomerate")


def flatten_with_lineage(labels: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    """Flatten sparse positive IDs and return flat->original lineage."""

    labels = np.asarray(labels)
    if labels.ndim != 1 or labels.dtype.kind not in "iu" or np.any(labels < -1):
        raise WorkerContractError("labels must be a one-dimensional integer vector >= -1")
    original_ids = np.unique(labels[labels >= 0])
    flat = np.full(labels.shape, -1, dtype=np.int64)
    if original_ids.size:
        flat[labels >= 0] = np.searchsorted(original_ids, labels[labels >= 0])
    return flat, original_ids.astype(np.int64, copy=False)


def compose_actual_lineage(
    original_ids: Sequence[int], recluster_map: Sequence[int], reorder: Sequence[int]
) -> list[dict[str, int]]:
    """Compose the persisted recluster map with the actual depth reorder."""

    original_ids = np.asarray(original_ids)
    merge = np.asarray(recluster_map)
    reorder = np.asarray(reorder)
    if (original_ids.ndim != 1 or merge.shape != original_ids.shape
            or original_ids.dtype.kind not in "iu" or merge.dtype.kind not in "iu"
            or reorder.ndim != 1 or reorder.dtype.kind not in "iu"):
        raise WorkerContractError("lineage arrays must be compatible integer vectors")
    if np.unique(original_ids).size != original_ids.size or np.any(original_ids < 0) or np.any(merge < 0):
        raise WorkerContractError("original IDs must be unique and mappings nonnegative")
    component_ids, inverse = np.unique(merge, return_inverse=True)
    if reorder.size != component_ids.size or not np.array_equal(np.sort(reorder), np.arange(reorder.size)):
        raise WorkerContractError("reorder must permute the actual recluster components")
    final = reorder[inverse]
    return [
        {
            "original_unit_id": int(original_id),
            "flat_unit_id": flat_id,
            "actual_pre_reorder_component": int(pre),
            "actual_final_component": int(post),
        }
        for flat_id, (original_id, pre, post) in enumerate(zip(original_ids, merge, final, strict=True))
    ]


def partition_delta(before: Sequence[int], after: Sequence[int]) -> PartitionDelta:
    """Permutation-invariant O(N) partition comparison, including the -1 state."""

    before = np.asarray(before)
    after = np.asarray(after)
    if (before.shape != after.shape or before.ndim != 1
            or before.dtype.kind not in "iu" or after.dtype.kind not in "iu"):
        raise WorkerContractError("partitions must be same-length integer vectors")
    pairs = Counter(zip(before.tolist(), after.tolist(), strict=True))
    before_counts = Counter(before.tolist())
    after_counts = Counter(after.tolist())
    exact = {
        (b, a) for (b, a), n in pairs.items()
        if n == before_counts[b] == after_counts[a]
    }
    unchanged_events = sum(pairs[pair] for pair in exact)
    unchanged_before = {b for b, _ in exact if b >= 0}
    unchanged_after = {a for _, a in exact if a >= 0}
    before_groups = {int(v) for v in before if v >= 0}
    after_groups = {int(v) for v in after if v >= 0}
    return PartitionDelta(
        event_count=int(before.size),
        changed_event_count=int(before.size - unchanged_events),
        unchanged_group_count=len(unchanged_before),
        changed_before_group_count=len(before_groups - unchanged_before),
        changed_after_group_count=len(after_groups - unchanged_after),
    )


def stratified_isi_summary(
    times_samples: Sequence[int],
    labels: Sequence[int],
    domains: Mapping[str, Iterable[tuple[int, int]]],
    *,
    refractory_samples: int,
) -> dict[str, dict[str, int]]:
    """Count within-segment adjacent ISIs on the original sample clock."""

    times = np.asarray(times_samples)
    labels = np.asarray(labels)
    if (times.shape != labels.shape or times.ndim != 1 or times.dtype.kind not in "iu"
            or labels.dtype.kind not in "iu" or refractory_samples < 0):
        raise WorkerContractError("invalid ISI inputs")
    used = np.zeros(times.size, dtype=bool)
    result: dict[str, dict[str, int]] = {}
    for state, raw_intervals in domains.items():
        intervals = list(raw_intervals)
        if any(lo >= hi for lo, hi in intervals):
            raise WorkerContractError("state intervals must be nonempty and half-open")
        spike_count = pairs = violations = 0
        for segment_id, (lo, hi) in enumerate(intervals):
            in_segment = (times >= lo) & (times < hi) & (labels >= 0)
            if np.any(used & in_segment):
                raise WorkerContractError("state intervals overlap")
            used |= in_segment
            spike_count += int(in_segment.sum())
            for unit in np.unique(labels[in_segment]):
                unit_times = np.sort(times[in_segment & (labels == unit)], kind="stable")
                diffs = np.diff(unit_times)
                pairs += int(diffs.size)
                violations += int(np.count_nonzero(diffs <= refractory_samples))
        result[state] = {
            "spike_count": spike_count,
            "adjacent_pair_denominator": pairs,
            "violation_count": violations,
            "refractory_samples_inclusive": int(refractory_samples),
        }
    return result


def require_shared_state(
    disabled: Mapping[str, object], enabled: Mapping[str, object], required: Sequence[str]
) -> None:
    """Missing-both is failure; required frozen state must be exactly equal."""

    for key in required:
        if key not in disabled or key not in enabled:
            raise WorkerContractError(f"required equivalence field {key} is missing")
        left, right = disabled[key], enabled[key]
        if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
            if not isinstance(left, np.ndarray) or not isinstance(right, np.ndarray) or not np.array_equal(left, right):
                raise WorkerContractError(f"required equivalence field {key} differs")
        elif left != right:
            raise WorkerContractError(f"required equivalence field {key} differs")


def qda_accounting(
    scheduled_upper: Sequence[Sequence[bool]],
    statuses: Sequence[Sequence[str]],
    accepted: Sequence[Sequence[bool]],
) -> dict[str, int]:
    """Count explicit QDA lifecycle states without interpreting score sentinels."""

    scheduled = np.asarray(scheduled_upper)
    accepted = np.asarray(accepted)
    statuses = np.asarray(statuses, dtype=object)
    if (scheduled.dtype != np.bool_ or accepted.dtype != np.bool_
            or scheduled.shape != accepted.shape or scheduled.shape != statuses.shape
            or scheduled.ndim != 2 or scheduled.shape[0] != scheduled.shape[1]):
        raise WorkerContractError("QDA lifecycle arrays must be compatible square arrays")
    complete = statuses == "completed"
    unknown = statuses == "scheduled_unknown"
    if np.any(accepted & ~complete):
        raise WorkerContractError("accepted QDA pairs require explicit completed status")
    return {
        "scheduled_pairs": int(scheduled.sum()),
        "completed_pairs": int(np.count_nonzero(np.triu(complete, 1))),
        "unknown_pairs": int(np.count_nonzero(np.triu(unknown, 1))),
        "completed_rejections": int(np.count_nonzero(np.triu(complete & ~accepted, 1))),
    }


def linkage_accounting(
    raw_distance: Sequence[Sequence[float]], *, threshold: float, eps: float = 1e-5,
    method: str = "single",
) -> dict[str, object]:
    """Keep direct threshold edges separate from expanded co-membership."""

    consumed = source_consumed_linkage_distance(
        np.asarray(raw_distance, dtype=float), threshold=threshold, eps=eps
    )
    direct = consumed <= threshold
    np.fill_diagonal(direct, True)
    components = source_linkage_components(consumed, threshold=threshold, method=method)
    expanded = components[:, None] == components[None, :]
    return {
        "source_consumed_distance": consumed,
        "direct_pair_count": int(np.count_nonzero(np.triu(direct, 1))),
        "expanded_pair_count": int(np.count_nonzero(np.triu(expanded, 1))),
        "component_count": int(np.unique(components).size),
    }
