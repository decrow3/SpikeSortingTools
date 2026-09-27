"""Standalone agglomeration-route provenance supplement for BO.

This adapter reuses the default-disabled atomic diagnostic provenance writer.
It records observations only and performs no QDA, merging, or sorting.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from testing.experimental.ccg_linkage_contract import single_link_components
from testing.experimental.diagnostic_provenance import (
    DiagnosticProvenanceWriter,
    ProvenanceValidationError,
    validate_bundle,
)


ROUTE_SCHEMA = "dartsort-agglomeration-route-observation-v2"
BOOL_MATRICES = (
    "direct_distance_adjacency",
    "distance_linkage_mask",
    "direct_force_adjacency",
    "force_linkage_mask",
    "firing_eligible",
    "qda_requested",
    "qda_scheduled_upper",
    "qda_accepted",
    "union_mask",
)
NUMERIC_OBSERVATIONS = (
    "raw_template_distance",
    "distance_linkage_consumed_distance",
    "force_linkage_consumed_distance",
    "template_shift_samples",
    "firing_correlation",
    "qda_score",
    "qda_coverage",
)


def _matrix(value, *, size: int, dtype, name: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=dtype)
    except (TypeError, ValueError) as exc:
        raise ProvenanceValidationError(f"{name} must have shape {(size, size)}") from exc
    if array.shape != (size, size):
        raise ProvenanceValidationError(f"{name} must have shape {(size, size)}")
    return array


def _bool_matrix(value, *, size: int, name: str) -> np.ndarray:
    if (not isinstance(value, list) or len(value) != size
            or any(not isinstance(row, list) or len(row) != size for row in value)
            or any(type(item) is not bool for row in value for item in row)):
        raise ProvenanceValidationError(f"{name} must be a strict boolean {(size, size)} matrix")
    return np.asarray(value, dtype=bool)


def _numeric_observation(
    value: dict, *, size: int, name: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if value.get("status") != "available":
        raise ProvenanceValidationError(f"{name} must be an available observation")
    rows = value.get("values")
    states = value.get("value_states")
    if not isinstance(rows, list) or len(rows) != size or any(
        not isinstance(row, list) or len(row) != size for row in rows
    ):
        raise ProvenanceValidationError(f"{name}.values must be a square JSON matrix")
    if not isinstance(states, list) or len(states) != size or any(
        not isinstance(row, list) or len(row) != size for row in states
    ):
        raise ProvenanceValidationError(f"{name}.value_states must be a square JSON matrix")
    numeric = np.empty((size, size), dtype=float)
    available = np.empty((size, size), dtype=bool)
    valid_states = {"finite", "positive_infinity", "negative_infinity", "missing"}
    for i, (row, state_row) in enumerate(zip(rows, states, strict=True)):
        for j, (item, state) in enumerate(zip(row, state_row, strict=True)):
            if state not in valid_states:
                raise ProvenanceValidationError(f"invalid {name} value state")
            available[i, j] = state != "missing"
            if state == "finite":
                if not isinstance(item, (int, float)) or isinstance(item, bool) or not math.isfinite(item):
                    raise ProvenanceValidationError(f"finite {name} entry must be finite")
                numeric[i, j] = item
            elif state == "positive_infinity":
                if item is not None:
                    raise ProvenanceValidationError(f"infinite {name} entry must use null JSON value")
                numeric[i, j] = np.inf
            elif state == "negative_infinity":
                if item is not None:
                    raise ProvenanceValidationError(f"infinite {name} entry must use null JSON value")
                numeric[i, j] = -np.inf
            else:
                if item is not None:
                    raise ProvenanceValidationError(f"missing {name} entry must be null")
                numeric[i, j] = np.nan
    return numeric, np.asarray(states, dtype=object), available


def _optional_route(value: dict, *, size: int) -> tuple[np.ndarray, np.ndarray]:
    status = value.get("status")
    if status == "unavailable":
        if not value.get("reason") or value.get("tested_mask") is not None or value.get("accepted_mask") is not None:
            raise ProvenanceValidationError("unavailable optional route needs reason and null masks")
        empty = np.zeros((size, size), dtype=bool)
        return empty, empty
    if status != "available":
        raise ProvenanceValidationError("optional route status must be available or unavailable")
    tested = _bool_matrix(value.get("tested_mask"), size=size, name="optional_route.tested_mask")
    accepted = _bool_matrix(value.get("accepted_mask"), size=size, name="optional_route.accepted_mask")
    if not np.array_equal(tested, tested.T) or not np.array_equal(accepted, accepted.T):
        raise ProvenanceValidationError("optional route masks must be symmetric")
    if np.any(accepted & ~tested):
        raise ProvenanceValidationError("optional accepted pairs must have been tested")
    return tested, accepted


def source_consumed_linkage_distance(raw: np.ndarray, *, threshold: float, eps: float) -> np.ndarray:
    """Mirror the upper-triangle numeric preprocessing in hierarchical_cluster."""

    raw = np.asarray(raw, dtype=float)
    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ProvenanceValidationError("raw distance must be square")
    size = raw.shape[0]
    upper = raw[np.triu_indices(size, 1)].copy()
    if np.isnan(upper).any() or np.isneginf(upper).any():
        raise ProvenanceValidationError("source-consumed upper triangle cannot be missing or -inf")
    upper[(upper > -eps) & (upper < 0)] = 0.0
    finite = np.isfinite(upper)
    if finite.any() and not finite.all():
        upper[~finite] = max(0.0, float(upper[finite].max())) + threshold + 1.0
    consumed = np.zeros_like(raw)
    consumed[np.triu_indices(size, 1)] = upper
    consumed += consumed.T
    return consumed


def source_linkage_components(consumed: np.ndarray, *, threshold: float, method: str) -> np.ndarray:
    consumed = np.asarray(consumed, dtype=float)
    size = consumed.shape[0]
    if size <= 1:
        return np.arange(size)
    upper = squareform(consumed, checks=False)
    finite = np.isfinite(upper)
    if not finite.any() or upper.min() > threshold:
        return np.arange(size)
    labels = fcluster(linkage(upper, method=method), threshold, criterion="distance") - 1
    return labels


def validate_route_observation(route: dict) -> dict:
    required = {
        "schema", "stage_id", "source_git_commit", "unit_ids", "clock",
        "namespaces", "label_semantics", "event_rows", "matrices", "optional_route",
        "linkage_contract", "qda_evaluation_status",
        "final_mapping_pre_reorder", "depth_reorder_pre_to_final",
        "final_mapping_depth_reordered",
    }
    missing = required.difference(route)
    if missing:
        raise ProvenanceValidationError(f"route observation missing {sorted(missing)}")
    if route["schema"] != ROUTE_SCHEMA:
        raise ProvenanceValidationError("unexpected route schema")
    source_commit = route["source_git_commit"]
    if (not isinstance(source_commit, str) or len(source_commit) != 40
            or any(c not in "0123456789abcdef" for c in source_commit)):
        raise ProvenanceValidationError("source_git_commit must be a 40-hex Git object ID")
    unit_ids = route["unit_ids"]
    if (not isinstance(unit_ids, list) or not unit_ids
            or any(not isinstance(v, int) or isinstance(v, bool) or v < 0 for v in unit_ids)
            or len(set(unit_ids)) != len(unit_ids)):
        raise ProvenanceValidationError("unit_ids must be unique nonnegative integers")
    size = len(unit_ids)
    namespaces = route["namespaces"]
    expected_namespaces = {
        "pre_stage_events", "direct_pair_observations", "linkage_expanded_masks",
        "final_pre_reorder", "final_depth_reordered",
    }
    if (set(namespaces) != expected_namespaces
            or any(not isinstance(v, str) or not v for v in namespaces.values())
            or len(set(namespaces.values())) != len(namespaces)):
        raise ProvenanceValidationError("stage namespaces must be exact, nonempty, and unique")

    clock = route["clock"]
    if (clock.get("bounds") != "half-open"
            or not isinstance(clock.get("origin_sample"), int)
            or isinstance(clock.get("origin_sample"), bool)
            or not isinstance(clock.get("sampling_frequency_hz"), (int, float))
            or not math.isfinite(clock["sampling_frequency_hz"])
            or clock["sampling_frequency_hz"] <= 0):
        raise ProvenanceValidationError("invalid route clock")
    semantics = route["label_semantics"]
    if semantics.get("noise_label") != -1 or semantics.get("padding_id") != -1:
        raise ProvenanceValidationError("noise and padding sentinel must remain explicit as -1")

    events = route["event_rows"]
    if events.get("status") == "unavailable":
        if not events.get("reason"):
            raise ProvenanceValidationError("unavailable events need a reason")
    elif events.get("status") == "available":
        rows = events.get("rows")
        if not isinstance(rows, list):
            raise ProvenanceValidationError("available events need rows")
        row_ids = set()
        for row in rows:
            if set(row) != {"event_row_id", "label", "time_sample", "namespace_id"}:
                raise ProvenanceValidationError("event row fields must be exact")
            if (not isinstance(row["event_row_id"], int)
                    or isinstance(row["event_row_id"], bool) or row["event_row_id"] < 0):
                raise ProvenanceValidationError("event row ID must be a nonnegative integer")
            if row["event_row_id"] in row_ids:
                raise ProvenanceValidationError("duplicate event row ID")
            row_ids.add(row["event_row_id"])
            if not isinstance(row["time_sample"], int) or isinstance(row["time_sample"], bool):
                raise ProvenanceValidationError("event time must be an integer sample")
            if (not isinstance(row["label"], int) or isinstance(row["label"], bool)
                    or row["label"] < -1):
                raise ProvenanceValidationError("event label must be an integer >= -1")
            if row["label"] >= 0 and row["label"] not in unit_ids:
                raise ProvenanceValidationError("event label is outside the pre-stage unit namespace")
            if row["namespace_id"] != namespaces["pre_stage_events"]:
                raise ProvenanceValidationError("event row namespace must match pre-stage namespace")
    else:
        raise ProvenanceValidationError("event row status must be available or unavailable")

    matrices = route["matrices"]
    if set(matrices) != set(BOOL_MATRICES).union(NUMERIC_OBSERVATIONS):
        raise ProvenanceValidationError("route matrix fields must be exact")
    bools = {
        name: _bool_matrix(matrices[name], size=size, name=name)
        for name in BOOL_MATRICES
    }
    for name, value in bools.items():
        if name != "qda_scheduled_upper" and not np.array_equal(value, value.T):
            raise ProvenanceValidationError(f"{name} must be symmetric")
    numeric = {
        name: _numeric_observation(matrices[name], size=size, name=name)
        for name in NUMERIC_OBSERVATIONS
    }
    contract = route["linkage_contract"]
    contract_fields = {"method", "distance_threshold", "force_threshold", "eps", "tie", "diagonal"}
    if set(contract) != contract_fields:
        raise ProvenanceValidationError("linkage contract fields must be exact")
    if contract["method"] not in ("single", "complete"):
        raise ProvenanceValidationError("unsupported linkage method")
    for name in ("distance_threshold", "force_threshold", "eps"):
        if (not isinstance(contract[name], (int, float)) or isinstance(contract[name], bool)
                or not math.isfinite(contract[name]) or contract[name] <= 0):
            raise ProvenanceValidationError(f"invalid linkage {name}")
    if contract["eps"] >= min(contract["distance_threshold"], contract["force_threshold"]):
        raise ProvenanceValidationError("linkage eps must be smaller than both thresholds")
    if contract["tie"] != "inclusive" or contract["diagonal"] != "included":
        raise ProvenanceValidationError("linkage tie/diagonal semantics mismatch")

    raw_distance = numeric["raw_template_distance"][0]
    distance_consumed_expected = source_consumed_linkage_distance(
        raw_distance, threshold=contract["distance_threshold"], eps=contract["eps"]
    )
    force_consumed_expected = source_consumed_linkage_distance(
        raw_distance, threshold=contract["force_threshold"], eps=contract["eps"]
    )
    distance_consumed_saved = numeric["distance_linkage_consumed_distance"][0]
    force_consumed_saved = numeric["force_linkage_consumed_distance"][0]
    if not np.array_equal(distance_consumed_saved, distance_consumed_expected):
        raise ProvenanceValidationError("stored distance-linkage consumed distance mismatch")
    if not np.array_equal(force_consumed_saved, force_consumed_expected):
        raise ProvenanceValidationError("stored force-linkage consumed distance mismatch")
    direct_distance_expected = distance_consumed_expected <= contract["distance_threshold"]
    direct_force_expected = force_consumed_expected <= contract["force_threshold"]
    np.fill_diagonal(direct_distance_expected, True)
    np.fill_diagonal(direct_force_expected, True)
    if not np.array_equal(bools["direct_distance_adjacency"], direct_distance_expected):
        raise ProvenanceValidationError("direct distance adjacency violates upper-triangle source semantics")
    if not np.array_equal(bools["direct_force_adjacency"], direct_force_expected):
        raise ProvenanceValidationError("direct force adjacency violates upper-triangle source semantics")
    if np.any(bools["direct_distance_adjacency"] & ~bools["distance_linkage_mask"]):
        raise ProvenanceValidationError("direct distance edges must survive linkage expansion")
    if np.any(bools["direct_force_adjacency"] & ~bools["force_linkage_mask"]):
        raise ProvenanceValidationError("direct force edges must survive linkage expansion")
    for threshold_name, consumed_name, expanded_name in (
        ("distance_threshold", "distance_linkage_consumed_distance", "distance_linkage_mask"),
        ("force_threshold", "force_linkage_consumed_distance", "force_linkage_mask"),
    ):
        component_ids = source_linkage_components(
            numeric[consumed_name][0],
            threshold=contract[threshold_name],
            method=contract["method"],
        )
        expected_expanded = component_ids[:, None] == component_ids[None, :]
        if not np.array_equal(expected_expanded, bools[expanded_name]):
            raise ProvenanceValidationError(f"{expanded_name} does not match direct-edge connectivity")
    expected_scheduled = np.triu(
        bools["qda_requested"] & bools["firing_eligible"], k=1
    )
    if not np.array_equal(expected_scheduled, bools["qda_scheduled_upper"]):
        raise ProvenanceValidationError("QDA scheduled mask must be eligible requested upper triangle")

    statuses = route["qda_evaluation_status"]
    if (not isinstance(statuses, list) or len(statuses) != size
            or any(not isinstance(row, list) or len(row) != size for row in statuses)):
        raise ProvenanceValidationError("QDA evaluation status must be square")
    allowed_status = {
        "not_applicable_diagonal", "not_applicable_lower_triangle", "not_scheduled",
        "scheduled_unknown", "early_absent", "early_overlap", "early_count",
        "kde_failed", "completed",
    }
    completed = np.zeros((size, size), dtype=bool)
    unknown = np.zeros((size, size), dtype=bool)
    for i, row in enumerate(statuses):
        for j, status in enumerate(row):
            if status not in allowed_status:
                raise ProvenanceValidationError("invalid QDA evaluation status")
            scheduled = bools["qda_scheduled_upper"][i, j]
            if i == j:
                expected = "not_applicable_diagonal"
            elif i > j:
                expected = "not_applicable_lower_triangle"
            elif not scheduled:
                expected = "not_scheduled"
            else:
                expected = None
            if expected is not None and status != expected:
                raise ProvenanceValidationError("QDA evaluation status disagrees with scheduling")
            if expected is None and status in {
                "not_applicable_diagonal", "not_applicable_lower_triangle", "not_scheduled"
            }:
                raise ProvenanceValidationError("scheduled QDA pair lacks evaluation status")
            if status == "completed":
                completed[i, j] = completed[j, i] = True
            if status == "scheduled_unknown":
                unknown[i, j] = unknown[j, i] = True
    np.fill_diagonal(completed, True)
    if np.any(bools["qda_accepted"] & ~completed):
        raise ProvenanceValidationError("QDA accepted pairs must have completed statistics")
    optional_tested, optional_accepted = _optional_route(route["optional_route"], size=size)
    expected_union = bools["qda_accepted"] | bools["force_linkage_mask"] | optional_accepted
    np.fill_diagonal(expected_union, True)
    if not np.array_equal(expected_union, bools["union_mask"]):
        raise ProvenanceValidationError("union mask does not reproduce route union")

    components = single_link_components(bools["union_mask"])
    pre_values = route["final_mapping_pre_reorder"]
    if (not isinstance(pre_values, list) or len(pre_values) != size
            or any(not isinstance(v, int) or isinstance(v, bool) or v < 0 for v in pre_values)):
        raise ProvenanceValidationError("pre-reorder mapping must be an integer vector")
    pre_mapping = np.asarray(pre_values, dtype=np.int64)
    same_component = pre_mapping[:, None] == pre_mapping[None, :]
    if not np.array_equal(same_component, components[:, None] == components[None, :]):
        raise ProvenanceValidationError("pre-reorder mapping disagrees with union connectivity")
    unique_pre = np.unique(pre_mapping)
    reorder_values = route["depth_reorder_pre_to_final"]
    if (not isinstance(reorder_values, list) or len(reorder_values) != unique_pre.size
            or any(not isinstance(v, int) or isinstance(v, bool) or v < 0 for v in reorder_values)):
        raise ProvenanceValidationError("depth reorder must be a permutation")
    reorder = np.asarray(reorder_values, dtype=np.int64)
    if not np.array_equal(np.sort(reorder), np.arange(unique_pre.size)):
        raise ProvenanceValidationError("depth reorder must be a permutation")
    expected_final = reorder[np.searchsorted(unique_pre, pre_mapping)]
    final_values = route["final_mapping_depth_reordered"]
    if (not isinstance(final_values, list) or len(final_values) != size
            or any(not isinstance(v, int) or isinstance(v, bool) or v < 0 for v in final_values)):
        raise ProvenanceValidationError("depth-reordered mapping must be an integer vector")
    final = np.asarray(final_values, dtype=np.int64)
    if not np.array_equal(final, expected_final):
        raise ProvenanceValidationError("depth-reordered mapping mismatch")

    route_count = (
        bools["qda_accepted"].astype(np.int8)
        + bools["force_linkage_mask"].astype(np.int8)
        + optional_accepted.astype(np.int8)
    )
    return {
        "unit_count": size,
        "qda_requested_pairs": int(np.triu(bools["qda_requested"], 1).sum()),
        "qda_eligible_pairs": int(np.triu(bools["firing_eligible"], 1).sum()),
        "qda_scheduled_pairs": int(bools["qda_scheduled_upper"].sum()),
        "qda_not_requested_pairs": int(np.triu(~bools["qda_requested"], 1).sum()),
        "qda_scheduled_unknown_pairs": int(np.triu(unknown, 1).sum()),
        "qda_rejected_completed_pairs": int(np.triu(completed & ~bools["qda_accepted"], 1).sum()),
        "optional_untested_pairs": int(np.triu(~optional_tested, 1).sum()),
        "component_count": int(np.unique(components).size),
        "multi_route_pairs": int(np.triu(route_count > 1, 1).sum()),
        "numeric_available_counts": {
            name: int(np.triu(available).sum())
            for name, (_values, _states, available) in numeric.items()
        },
    }


class AgglomerationRouteCapture:
    def __init__(self, *, enabled: bool = False, max_output_bytes: int = 100_000_000):
        self.enabled = bool(enabled)
        self.max_output_bytes = int(max_output_bytes)

    def capture(self, output: Path, base_manifest: dict, route: dict) -> dict:
        if not self.enabled:
            return {"status": "disabled", "written": False, "path": str(output)}
        summary = validate_route_observation(route)
        manifest = copy.deepcopy(base_manifest)
        manifest["agglomeration_route_observation"] = copy.deepcopy(route)
        result = DiagnosticProvenanceWriter(
            enabled=True, max_output_bytes=self.max_output_bytes
        ).write(output, manifest)
        validate_route_bundle(output)
        return {**result, "route_summary": summary}


def validate_route_bundle(output: Path) -> dict:
    base = validate_bundle(output)
    manifest = json.loads((Path(output) / "manifest.json").read_text())
    route = manifest.get("agglomeration_route_observation")
    if route is None:
        raise ProvenanceValidationError("bundle is missing route observation")
    return {**base, "route_summary": validate_route_observation(route)}
