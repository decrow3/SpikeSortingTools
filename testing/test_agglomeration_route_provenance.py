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
    available = [[item is not None for item in row] for row in values]
    return {"status": "available", "values": values, "available_mask": available}


def route_fixture(*, alternate=False):
    if not alternate:
        qda = [[True, True, False], [True, True, False], [False, False, True]]
        force_direct = [[True, True, False], [True, True, True], [False, True, True]]
        optional = {
            "status": "available",
            "tested_mask": [[True, True, False], [True, True, False], [False, False, True]],
            "accepted_mask": [[True, True, False], [True, True, False], [False, False, True]],
        }
    else:
        qda = [[True, False, False], [False, True, True], [False, True, True]]
        force_direct = [[True, True, False], [True, True, False], [False, False, True]]
        optional = {
            "status": "unavailable", "reason": "preset disabled",
            "tested_mask": None, "accepted_mask": None,
        }
    if alternate:
        force_linkage = [[True, True, False], [True, True, False], [False, False, True]]
    else:
        force_linkage = [[True, True, True], [True, True, True], [True, True, True]]
    distance_direct = [[True, True, False], [True, True, True], [False, True, True]]
    distance_linkage = [[True, True, True], [True, True, True], [True, True, True]]
    tested = [[True, True, False], [True, True, True], [False, True, True]]
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
        "source_commit": H,
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
        "event_rows": {
            "status": "available",
            "rows": [
                {"event_row_id": "e0", "label": 10, "time_sample": 100, "namespace_id": "pre"},
                {"event_row_id": "e1", "label": -1, "time_sample": 200, "namespace_id": "pre"},
            ],
        },
        "matrices": {
            "direct_template_distance": numeric([[0.0, 0.2, None], [0.2, 0.0, 0.2], [None, 0.2, 0.0]]),
            "template_shift_samples": numeric([[0.0, 1.0, None], [-1.0, 0.0, 2.0], [None, -2.0, 0.0]]),
            "firing_correlation": numeric([[1.0, -0.2, None], [-0.2, 1.0, -0.3], [None, -0.3, 1.0]]),
            "qda_score": numeric([[1.0, 0.8, None], [0.8, 1.0, 0.1], [None, 0.1, 1.0]]),
            "qda_coverage": numeric([[1.0, 0.9, None], [0.9, 1.0, 0.8], [None, 0.8, 1.0]]),
            "direct_distance_adjacency": distance_direct,
            "distance_linkage_mask": distance_linkage,
            "direct_force_adjacency": force_direct,
            "force_linkage_mask": force_linkage,
            "firing_eligible": tested,
            "qda_tested": tested,
            "qda_accepted": qda,
            "union_mask": union,
        },
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
    assert summary["qda_untested_pairs"] == 1
    assert summary["qda_rejected_pairs"] == 1
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
        (lambda r: r["matrices"]["qda_tested"].__setitem__(0, [True]), "shape"),
        (lambda r: r["matrices"]["qda_score"]["values"][0].__setitem__(1, float("nan")), "finite"),
        (lambda r: r["matrices"]["qda_score"]["values"][0].__setitem__(2, 0.0), "must be null"),
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
