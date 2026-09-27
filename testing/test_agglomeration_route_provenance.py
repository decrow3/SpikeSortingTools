import json

import numpy as np
import pytest

from testing.experimental.agglomeration_route_provenance import (
    AgglomerationRouteCapture,
    ProvenanceValidationError,
    ROUTE_SCHEMA,
    validate_route_bundle,
    validate_route_observation,
)
from testing.experimental.diagnostic_provenance import SCHEMA


H = "a" * 64
G = "edcfe1b51d672b4136eb13cc78c0875da804b851"


def base_manifest():
    return {
        "schema": SCHEMA,
        "artifact_id": "route-fixture",
        "parent_stage": "agglomeration",
        "clock": {
            "kind": "AP-frame-zero", "origin_sample": 0, "origin_time_s": 0.0,
            "sampling_frequency_hz": 30000.0, "start_sample": 0,
            "stop_sample": 1000, "bounds": "half-open",
        },
        "sources": {
            "source": {"sha256": H}, "config": {"sha256": H},
            "geometry": {"sha256": H},
            "coordinate_semantics": {
                "time": "AP samples", "channel": "physical", "depth": "registered",
                "units": "samples and um",
            },
        },
        "representations": {
            "temporal_components": {"status": "unavailable", "reason": "route-only fixture"},
            "spatial_components": {"status": "unavailable", "reason": "route-only fixture"},
            "norm_discount": {"status": "unavailable", "reason": "route-only fixture"},
        },
        "template_construction": [], "events": [], "candidates": [],
        "score_conventions": {"route_capture": "observation only"},
    }


def numeric(values):
    encoded = []
    states = []
    for row in values:
        encoded_row = []
        state_row = []
        for item in row:
            if item is None:
                encoded_row.append(None)
                state_row.append("missing")
            elif np.isposinf(item):
                encoded_row.append(None)
                state_row.append("positive_infinity")
            elif np.isneginf(item):
                encoded_row.append(None)
                state_row.append("negative_infinity")
            else:
                encoded_row.append(item)
                state_row.append("finite")
        encoded.append(encoded_row)
        states.append(state_row)
    return {"status": "available", "values": encoded, "value_states": states}


def route_fixture(*, alternate=False):
    if not alternate:
        qda = [[True, True, False], [True, True, False], [False, False, True]]
        optional = {
            "status": "available",
            "tested_mask": [[True, True, False], [True, True, False], [False, False, True]],
            "accepted_mask": [[True, True, False], [True, True, False], [False, False, True]],
        }
    else:
        qda = [[True, False, False], [False, True, True], [False, True, True]]
        optional = {
            "status": "unavailable", "reason": "preset disabled",
            "tested_mask": None, "accepted_mask": None,
        }
    force_direct = (
        [[True, True, False], [True, True, False], [False, False, True]]
        if alternate else
        [[True, True, False], [True, True, True], [False, True, True]]
    )
    force_linkage = (
        [[True, True, False], [True, True, False], [False, False, True]]
        if alternate else [[True] * 3 for _ in range(3)]
    )
    distance_direct = [[True, True, False], [True, True, True], [False, True, True]]
    distance_linkage = [[True] * 3 for _ in range(3)]
    requested = distance_linkage
    eligible = [[True, True, False], [True, True, True], [False, True, True]]
    scheduled = [[False, True, False], [False, False, True], [False, False, False]]
    statuses = [
        ["not_applicable_diagonal", "scheduled_unknown" if alternate else "completed", "not_scheduled"],
        ["not_applicable_lower_triangle", "not_applicable_diagonal", "completed"],
        ["not_applicable_lower_triangle", "not_applicable_lower_triangle", "not_applicable_diagonal"],
    ]
    raw_distance = (
        [[0.0, 0.2, np.inf], [9.0, 0.0, 0.5], [7.0, 8.0, 0.0]]
        if alternate else
        [[0.0, 0.2, np.inf], [9.0, 0.0, 0.2], [7.0, 8.0, 0.0]]
    )
    distance_replacement = 2.1 if alternate else 1.8
    force_replacement = 1.8 if alternate else 1.5
    distance_consumed = (
        [[0.0, 0.2, distance_replacement], [0.2, 0.0, 0.5], [distance_replacement, 0.5, 0.0]]
        if alternate else
        [[0.0, 0.2, distance_replacement], [0.2, 0.0, 0.2], [distance_replacement, 0.2, 0.0]]
    )
    force_consumed = (
        [[0.0, 0.2, force_replacement], [0.2, 0.0, 0.5], [force_replacement, 0.5, 0.0]]
        if alternate else
        [[0.0, 0.2, force_replacement], [0.2, 0.0, 0.2], [force_replacement, 0.2, 0.0]]
    )
    optional_accepted = (
        optional["accepted_mask"] if optional["status"] == "available"
        else [[False] * 3 for _ in range(3)]
    )
    union = (
        np.asarray(qda) | np.asarray(force_linkage) | np.asarray(optional_accepted)
    ).tolist()
    return {
        "schema": ROUTE_SCHEMA,
        "stage_id": "agglomeration:synthetic",
        "source_git_commit": G,
        "unit_ids": [10, 30, 90],
        "namespaces": {
            "pre_stage_events": "ns:pre",
            "direct_pair_observations": "ns:direct",
            "linkage_expanded_masks": "ns:expanded",
            "final_pre_reorder": "ns:final-pre-depth",
            "final_depth_reordered": "ns:final-depth",
        },
        "clock": {"origin_sample": 0, "sampling_frequency_hz": 30000.0, "bounds": "half-open"},
        "label_semantics": {"noise_label": -1, "padding_id": -1},
        "linkage_contract": {
            "method": "single", "distance_threshold": 0.6, "force_threshold": 0.3,
            "eps": 1e-5, "tie": "inclusive", "diagonal": "included",
        },
        "event_rows": {
            "status": "available",
            "rows": [
                {"event_row_id": 0, "label": 10, "time_sample": 100, "namespace_id": "ns:pre"},
                {"event_row_id": 1, "label": -1, "time_sample": 200, "namespace_id": "ns:pre"},
            ],
        },
        "matrices": {
            "raw_template_distance": numeric(raw_distance),
            "distance_linkage_consumed_distance": numeric(distance_consumed),
            "force_linkage_consumed_distance": numeric(force_consumed),
            "template_shift_samples": numeric([[0.0, 1.0, None], [-1.0, 0.0, 2.0], [3.0, -2.0, 0.0]]),
            "firing_correlation": numeric([[1.0, -0.2, None], [-0.2, 1.0, -0.3], [None, -0.3, 1.0]]),
            "qda_score": numeric([[0.0, 0.8, 0.0], [0.8, 0.0, 0.1], [0.0, 0.1, 0.0]]),
            "qda_coverage": numeric([[0.0, 0.9, 0.0], [0.9, 0.0, 0.8], [0.0, 0.8, 0.0]]),
            "direct_distance_adjacency": distance_direct,
            "distance_linkage_mask": distance_linkage,
            "direct_force_adjacency": force_direct,
            "force_linkage_mask": force_linkage,
            "firing_eligible": eligible,
            "qda_requested": requested,
            "qda_scheduled_upper": scheduled,
            "qda_accepted": qda,
            "union_mask": union,
        },
        "qda_evaluation_status": statuses,
        "optional_route": optional,
        "final_mapping_pre_reorder": [7, 7, 7],
        "depth_reorder_pre_to_final": [0],
        "final_mapping_depth_reordered": [0, 0, 0],
    }


def test_disabled_capture_does_not_invoke_writer(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("writer invoked")

    monkeypatch.setattr(
        "testing.experimental.diagnostic_provenance.DiagnosticProvenanceWriter.write",
        forbidden,
    )
    out = tmp_path / "disabled"
    result = AgglomerationRouteCapture().capture(out, base_manifest(), route_fixture())
    assert result["status"] == "disabled"
    assert not out.exists()


def test_enabled_capture_is_atomic_valid_and_resists_later_mutation(tmp_path):
    out = tmp_path / "enabled"
    route = route_fixture()
    result = AgglomerationRouteCapture(enabled=True).capture(out, base_manifest(), route)
    assert result["status"] == "complete"
    assert validate_route_bundle(out)["route_summary"]["component_count"] == 1
    route["unit_ids"][0] = 999
    saved = json.loads((out / "manifest.json").read_text())
    assert saved["agglomeration_route_observation"]["unit_ids"] == [10, 30, 90]


def test_same_component_can_have_different_direct_route_histories():
    first = route_fixture()
    second = route_fixture(alternate=True)
    assert first["final_mapping_pre_reorder"] == second["final_mapping_pre_reorder"]
    assert first["matrices"]["qda_accepted"] != second["matrices"]["qda_accepted"]
    assert first["matrices"]["direct_force_adjacency"] != second["matrices"]["direct_force_adjacency"]
    assert validate_route_observation(first)["component_count"] == 1
    assert validate_route_observation(second)["component_count"] == 1


def test_untested_rejected_and_overlapping_routes_remain_distinct():
    summary = validate_route_observation(route_fixture())
    assert summary["qda_scheduled_unknown_pairs"] == 0
    assert summary["qda_rejected_completed_pairs"] == 1
    route = route_fixture()
    assert route["matrices"]["qda_accepted"][0][1]
    assert route["matrices"]["direct_force_adjacency"][0][1]
    assert route["optional_route"]["accepted_mask"][0][1]


def test_nontrivial_depth_reorder_is_explicit():
    route = route_fixture(alternate=True)
    diagonal = [[True, False, False], [False, True, False], [False, False, True]]
    route["matrices"]["qda_accepted"] = diagonal
    route["matrices"]["union_mask"] = route["matrices"]["force_linkage_mask"]
    route["final_mapping_pre_reorder"] = [7, 7, 9]
    route["depth_reorder_pre_to_final"] = [1, 0]
    route["final_mapping_depth_reordered"] = [1, 1, 0]
    assert validate_route_observation(route)["component_count"] == 2


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda r: r.pop("unit_ids"), "missing"),
        (lambda r: r["matrices"]["qda_scheduled_upper"].__setitem__(0, [True]), "strict boolean"),
        (lambda r: r["matrices"]["qda_score"]["values"][0].__setitem__(1, float("nan")), "finite"),
        (lambda r: r["matrices"]["template_shift_samples"]["values"][0].__setitem__(2, 0.0), "must be null"),
        (lambda r: r.__setitem__("final_mapping_depth_reordered", [0, 1, 0]), "reordered"),
    ],
)
def test_invalid_route_fails_before_any_complete_bundle(tmp_path, mutate, match):
    route = route_fixture()
    mutate(route)
    out = tmp_path / "invalid"
    with pytest.raises(ProvenanceValidationError, match=match):
        AgglomerationRouteCapture(enabled=True).capture(out, base_manifest(), route)
    assert not out.exists()
