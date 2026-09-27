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

from testing.experimental.ccg_linkage_contract import single_link_components
from testing.experimental.diagnostic_provenance import (
    DiagnosticProvenanceWriter,
    ProvenanceValidationError,
    validate_bundle,
)


ROUTE_SCHEMA = "dartsort-agglomeration-route-observation-v1"
BOOL_MATRICES = (
    "direct_distance_adjacency",
    "distance_linkage_mask",
    "direct_force_adjacency",
    "force_linkage_mask",
    "firing_eligible",
    "qda_tested",
    "qda_accepted",
    "union_mask",
)
NUMERIC_OBSERVATIONS = (
    "direct_template_distance",
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


def _numeric_observation(value: dict, *, size: int, name: str) -> tuple[np.ndarray, np.ndarray]:
    if value.get("status") != "available":
        raise ProvenanceValidationError(f"{name} must be an available observation")
    available = _matrix(value.get("available_mask"), size=size, dtype=bool, name=f"{name}.available_mask")
    rows = value.get("values")
    if not isinstance(rows, list) or len(rows) != size or any(
        not isinstance(row, list) or len(row) != size for row in rows
    ):
        raise ProvenanceValidationError(f"{name}.values must be a square JSON matrix")
    numeric = np.empty((size, size), dtype=float)
    for i, row in enumerate(rows):
        for j, item in enumerate(row):
            if available[i, j]:
                if not isinstance(item, (int, float)) or isinstance(item, bool) or not math.isfinite(item):
                    raise ProvenanceValidationError(f"available {name} entry must be finite")
                numeric[i, j] = item
            else:
                if item is not None:
                    raise ProvenanceValidationError(f"unavailable {name} entry must be null")
                numeric[i, j] = np.nan
    if not np.array_equal(available, available.T):
        raise ProvenanceValidationError(f"{name} availability must be symmetric")
    return numeric, available


def _optional_route(value: dict, *, size: int) -> tuple[np.ndarray, np.ndarray]:
    status = value.get("status")
    if status == "unavailable":
        if not value.get("reason") or value.get("tested_mask") is not None or value.get("accepted_mask") is not None:
            raise ProvenanceValidationError("unavailable optional route needs reason and null masks")
        empty = np.zeros((size, size), dtype=bool)
        return empty, empty
    if status != "available":
        raise ProvenanceValidationError("optional route status must be available or unavailable")
    tested = _matrix(value.get("tested_mask"), size=size, dtype=bool, name="optional_route.tested_mask")
    accepted = _matrix(value.get("accepted_mask"), size=size, dtype=bool, name="optional_route.accepted_mask")
    if np.any(accepted & ~tested):
        raise ProvenanceValidationError("optional accepted pairs must have been tested")
    return tested, accepted


def validate_route_observation(route: dict) -> dict:
    required = {
        "schema", "stage_id", "source_commit", "unit_ids", "clock",
        "namespaces", "label_semantics", "event_rows", "matrices", "optional_route",
        "final_mapping_pre_reorder", "depth_reorder_pre_to_final",
        "final_mapping_depth_reordered",
    }
    missing = required.difference(route)
    if missing:
        raise ProvenanceValidationError(f"route observation missing {sorted(missing)}")
    if route["schema"] != ROUTE_SCHEMA:
        raise ProvenanceValidationError("unexpected route schema")
    source_commit = route["source_commit"]
    if (not isinstance(source_commit, str) or len(source_commit) != 64
            or any(c not in "0123456789abcdef" for c in source_commit)):
        raise ProvenanceValidationError("source_commit must be a SHA-256-style lowercase hash")
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
            if row["event_row_id"] in row_ids:
                raise ProvenanceValidationError("duplicate event row ID")
            row_ids.add(row["event_row_id"])
            if not isinstance(row["time_sample"], int) or isinstance(row["time_sample"], bool):
                raise ProvenanceValidationError("event time must be an integer sample")
            if not isinstance(row["label"], int) or isinstance(row["label"], bool):
                raise ProvenanceValidationError("event label must be integer")
            if row["label"] >= 0 and row["label"] not in unit_ids:
                raise ProvenanceValidationError("event label is outside the pre-stage unit namespace")
    else:
        raise ProvenanceValidationError("event row status must be available or unavailable")

    matrices = route["matrices"]
    if set(matrices) != set(BOOL_MATRICES).union(NUMERIC_OBSERVATIONS):
        raise ProvenanceValidationError("route matrix fields must be exact")
    bools = {
        name: _matrix(matrices[name], size=size, dtype=bool, name=name)
        for name in BOOL_MATRICES
    }
    for name, value in bools.items():
        if not np.array_equal(value, value.T):
            raise ProvenanceValidationError(f"{name} must be symmetric")
    numeric = {
        name: _numeric_observation(matrices[name], size=size, name=name)
        for name in NUMERIC_OBSERVATIONS
    }
    if np.any(bools["direct_distance_adjacency"] & ~bools["distance_linkage_mask"]):
        raise ProvenanceValidationError("direct distance edges must survive linkage expansion")
    if np.any(bools["direct_force_adjacency"] & ~bools["force_linkage_mask"]):
        raise ProvenanceValidationError("direct force edges must survive linkage expansion")
    for direct_name, expanded_name in (
        ("direct_distance_adjacency", "distance_linkage_mask"),
        ("direct_force_adjacency", "force_linkage_mask"),
    ):
        direct_components = single_link_components(bools[direct_name])
        expected_expanded = direct_components[:, None] == direct_components[None, :]
        if not np.array_equal(expected_expanded, bools[expanded_name]):
            raise ProvenanceValidationError(f"{expanded_name} does not match direct-edge connectivity")
    if np.any(bools["qda_accepted"] & ~bools["qda_tested"]):
        raise ProvenanceValidationError("QDA accepted pairs must have been tested")
    qda_eligible = bools["distance_linkage_mask"] & bools["firing_eligible"]
    if np.any(bools["qda_tested"] & ~qda_eligible):
        raise ProvenanceValidationError("QDA tested pairs must be distance/firing eligible")
    for name in ("qda_score", "qda_coverage"):
        if not np.array_equal(numeric[name][1], bools["qda_tested"]):
            raise ProvenanceValidationError(f"{name} availability must match QDA tested mask")
    optional_tested, optional_accepted = _optional_route(route["optional_route"], size=size)
    expected_union = bools["qda_accepted"] | bools["force_linkage_mask"] | optional_accepted
    np.fill_diagonal(expected_union, True)
    if not np.array_equal(expected_union, bools["union_mask"]):
        raise ProvenanceValidationError("union mask does not reproduce route union")

    components = single_link_components(bools["union_mask"])
    pre_mapping = np.asarray(route["final_mapping_pre_reorder"])
    if pre_mapping.shape != (size,) or not np.issubdtype(pre_mapping.dtype, np.integer):
        raise ProvenanceValidationError("pre-reorder mapping must be an integer vector")
    same_component = pre_mapping[:, None] == pre_mapping[None, :]
    if not np.array_equal(same_component, components[:, None] == components[None, :]):
        raise ProvenanceValidationError("pre-reorder mapping disagrees with union connectivity")
    unique_pre = np.unique(pre_mapping)
    reorder = np.asarray(route["depth_reorder_pre_to_final"])
    if (reorder.shape != (unique_pre.size,) or not np.issubdtype(reorder.dtype, np.integer)
            or not np.array_equal(np.sort(reorder), np.arange(unique_pre.size))):
        raise ProvenanceValidationError("depth reorder must be a permutation")
    expected_final = reorder[np.searchsorted(unique_pre, pre_mapping)]
    final = np.asarray(route["final_mapping_depth_reordered"])
    if not np.array_equal(final, expected_final):
        raise ProvenanceValidationError("depth-reordered mapping mismatch")

    route_count = (
        bools["qda_accepted"].astype(np.int8)
        + bools["force_linkage_mask"].astype(np.int8)
        + optional_accepted.astype(np.int8)
    )
    return {
        "unit_count": size,
        "qda_untested_pairs": int(np.triu(~bools["qda_tested"], 1).sum()),
        "qda_rejected_pairs": int(np.triu(bools["qda_tested"] & ~bools["qda_accepted"], 1).sum()),
        "optional_untested_pairs": int(np.triu(~optional_tested, 1).sum()),
        "component_count": int(np.unique(components).size),
        "multi_route_pairs": int(np.triu(route_count > 1, 1).sum()),
        "numeric_available_counts": {
            name: int(np.triu(available).sum()) for name, (_values, available) in numeric.items()
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
