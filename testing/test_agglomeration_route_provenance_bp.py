"""Targeted BP semantic repairs for the BO route-observation adapter."""

import copy

import numpy as np
import pytest

from testing.experimental.agglomeration_route_provenance import (
    ProvenanceValidationError,
    source_consumed_linkage_distance,
    validate_route_observation,
)
from testing.test_agglomeration_route_provenance import H, route_fixture


def test_git_object_id_is_distinct_from_sha256_blob_identity():
    route = route_fixture()
    assert len(route["source_git_commit"]) == 40
    assert validate_route_observation(route)["unit_count"] == 3
    route["source_git_commit"] = H
    with pytest.raises(ProvenanceValidationError, match="40-hex"):
        validate_route_observation(route)


def test_boolean_masks_are_not_coerced_from_integer_values():
    route = route_fixture()
    route["matrices"]["qda_requested"][0][1] = 1
    with pytest.raises(ProvenanceValidationError, match="strict boolean"):
        validate_route_observation(route)


def test_directed_observations_and_source_upper_triangle_are_preserved():
    route = route_fixture()
    validate_route_observation(route)
    shifts = route["matrices"]["template_shift_samples"]
    assert shifts["values"][0][1] == 1.0
    assert shifts["values"][1][0] == -1.0
    raw = route["matrices"]["raw_template_distance"]
    assert raw["values"][1][0] == 9.0
    assert raw["value_states"][0][2] == "positive_infinity"
    distance_consumed = route["matrices"]["distance_linkage_consumed_distance"]["values"]
    force_consumed = route["matrices"]["force_linkage_consumed_distance"]["values"]
    assert distance_consumed[0][2] == distance_consumed[2][0] == 1.8
    assert force_consumed[0][2] == force_consumed[2][0] == 1.5


def test_source_tolerance_infinity_substitution_and_threshold_tie():
    raw = np.array([[0.0, -1e-6, np.inf], [91.0, 0.0, 0.3], [92.0, 93.0, 0.0]])
    consumed = source_consumed_linkage_distance(raw, threshold=0.3, eps=1e-5)
    assert consumed[0, 1] == 0.0
    assert consumed[1, 2] == 0.3
    assert consumed[0, 2] == pytest.approx(1.6)


def test_unknown_scheduled_qda_is_not_counted_as_rejected_or_completed():
    route = route_fixture(alternate=True)
    summary = validate_route_observation(route)
    assert summary["qda_scheduled_pairs"] == 2
    assert summary["qda_scheduled_unknown_pairs"] == 1
    assert summary["qda_rejected_completed_pairs"] == 0
    route["matrices"]["qda_accepted"][0][1] = True
    route["matrices"]["qda_accepted"][1][0] = True
    with pytest.raises(ProvenanceValidationError, match="completed statistics"):
        validate_route_observation(route)


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda r: r["event_rows"]["rows"][0].__setitem__("event_row_id", "0"), "row ID"),
        (lambda r: r["event_rows"]["rows"][0].__setitem__("label", -2), "label"),
        (lambda r: r["event_rows"]["rows"][0].__setitem__("namespace_id", "ns:direct"), "namespace"),
    ],
)
def test_event_identity_and_namespace_are_strict(mutation, match):
    route = route_fixture()
    mutation(route)
    with pytest.raises(ProvenanceValidationError, match=match):
        validate_route_observation(route)


def test_same_final_component_does_not_identify_pairwise_route_history():
    first = route_fixture()
    second = route_fixture(alternate=True)
    first_summary = validate_route_observation(first)
    second_summary = validate_route_observation(second)
    assert first["final_mapping_depth_reordered"] == second["final_mapping_depth_reordered"]
    assert first_summary["component_count"] == second_summary["component_count"] == 1
    assert first["matrices"]["qda_accepted"] != second["matrices"]["qda_accepted"]
    assert first["matrices"]["direct_force_adjacency"] != second["matrices"]["direct_force_adjacency"]


def test_positive_infinity_and_missingness_are_distinct_states():
    route = route_fixture()
    assert validate_route_observation(route)["unit_count"] == 3
    changed = copy.deepcopy(route)
    changed["matrices"]["raw_template_distance"]["value_states"][0][2] = "missing"
    with pytest.raises(ProvenanceValidationError, match="cannot be missing"):
        validate_route_observation(changed)
