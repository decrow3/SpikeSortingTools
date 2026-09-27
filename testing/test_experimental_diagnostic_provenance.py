import copy
import json
from pathlib import Path

import numpy as np
import pytest

from testing.experimental.diagnostic_provenance import (
    DiagnosticProvenanceWriter,
    ProvenanceValidationError,
    SCHEMA,
    validate_bundle,
)


H = "a" * 64


def synthetic_manifest():
    scale = {"value": 1.0, "convention": "multiplicative template scale"}
    threshold = {"value": 36.0, "convention": "squared matching objective; inclusive"}
    return {
        "schema": SCHEMA,
        "artifact_id": "synthetic-fixture",
        "parent_stage": "matching:iteration0",
        "clock": {
            "kind": "AP-frame-zero", "origin_sample": 0, "origin_time_s": 0.0,
            "sampling_frequency_hz": 30000.0, "start_sample": 100, "stop_sample": 300,
            "bounds": "half-open",
        },
        "sources": {
            "source": {"sha256": H}, "config": {"sha256": H}, "geometry": {"sha256": H},
            "coordinate_semantics": {
                "time": "integer AP samples from frame zero", "channel": "physical AP channel ID",
                "depth": "observed depth; corrected = observed - displacement", "units": "samples and um",
            },
        },
        "representations": {
            "norm_discount": {"status": "unavailable", "reason": "whitener absent in fixture"}
        },
        "template_construction": [
            {"template_id": "t0", "membership_status": "exact", "member_event_ids": ["train:0"],
             "bank_unit_ids": [7], "parent_stage": "template-bank"},
            {"template_id": "t1", "membership_status": "ambiguous", "member_event_ids": ["train:1"],
             "bank_unit_ids": [8, 9], "parent_stage": "template-bank"},
            {"template_id": "t2", "membership_status": "unknown", "member_event_ids": None,
             "bank_unit_ids": None, "parent_stage": "legacy-import"},
        ],
        "events": [
            {"event_id": "e0", "candidate_origin_id": "row:10", "parent_stage": "matching:coarse",
             "sample_index": 120, "availability": "available"},
            {"event_id": "e1", "candidate_origin_id": "row:11", "parent_stage": "matching:coarse",
             "sample_index": 220, "availability": "unavailable"},
        ],
        "score_conventions": {
            "objective": "2*s*(conv+inv_lambda)-s^2*(normsq+inv_lambda)-inv_lambda",
            "runner_up": "coarse-local and global-final are distinct",
        },
        "candidates": [
            {"candidate_id": "c0", "event_id": "e0", "template_id": "t0",
             "parent_stage": "matching:coarse", "stage": "coarse", "iteration": 0,
             "evaluation_status": "evaluated", "role": "coarse_runner_up",
             "ranking_scope": "coarse_local", "score": 40.0, "scale": scale, "threshold": threshold},
            {"candidate_id": "c1", "event_id": "e0", "template_id": "t1",
             "parent_stage": "matching:fine", "stage": "fine", "iteration": 0,
             "evaluation_status": "evaluated", "role": "global_final_runner_up",
             "ranking_scope": "global_final", "score": 42.0, "scale": scale, "threshold": threshold},
            {"candidate_id": "c2", "event_id": "e0", "template_id": "t0",
             "parent_stage": "matching:fine", "stage": "fine", "iteration": 0,
             "evaluation_status": "evaluated", "role": "selected",
             "ranking_scope": "global_final", "score": 45.0, "scale": scale, "threshold": threshold},
            {"candidate_id": "c3", "event_id": "e1", "template_id": None,
             "parent_stage": "matching:fine", "stage": "fine", "iteration": 0,
             "evaluation_status": "not_evaluated", "role": "not_evaluated",
             "ranking_scope": "none", "score": None, "scale": scale, "threshold": threshold},
            {"candidate_id": "c4", "event_id": "e0", "template_id": None,
             "parent_stage": "matching:final", "stage": "final", "iteration": 1,
             "evaluation_status": "evaluated_unmatched", "role": "unmatched",
             "ranking_scope": "global_final", "score": None, "scale": scale, "threshold": threshold},
        ],
    }


def arrays():
    return {
        "temporal_components": np.arange(15, dtype=np.float32).reshape(3, 5),
        "spatial_components": np.arange(24, dtype=np.float32).reshape(2, 3, 4),
    }


def test_default_disabled_writes_nothing(tmp_path):
    out = tmp_path / "disabled"
    result = DiagnosticProvenanceWriter().write(out, synthetic_manifest(), arrays=arrays())
    assert result["status"] == "disabled"
    assert not out.exists()


def test_deterministic_round_trip_ambiguous_and_unknown_membership(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    writer = DiagnosticProvenanceWriter(enabled=True)
    writer.write(first, synthetic_manifest(), arrays=arrays())
    writer.write(second, synthetic_manifest(), arrays=arrays())
    assert validate_bundle(first)["status"] == "valid"
    assert (first / "manifest.json").read_bytes() == (second / "manifest.json").read_bytes()
    assert json.loads((first / "manifest.json").read_text())["template_construction"][2]["membership_status"] == "unknown"


def test_incomplete_write_is_rejected(tmp_path):
    out = tmp_path / "incomplete"
    out.mkdir()
    (out / "manifest.json").write_text("{}")
    with pytest.raises(ProvenanceValidationError, match="incomplete bundle"):
        validate_bundle(out)


def test_corrupted_content_addressed_reference_is_rejected(tmp_path):
    out = tmp_path / "corrupt"
    DiagnosticProvenanceWriter(enabled=True).write(out, synthetic_manifest(), arrays=arrays())
    manifest = json.loads((out / "manifest.json").read_text())
    path = out / manifest["representations"]["temporal_components"]["path"]
    path.write_bytes(path.read_bytes() + b"corruption")
    with pytest.raises(ProvenanceValidationError, match="corrupt or missing"):
        validate_bundle(out)


def test_missing_content_addressed_reference_is_rejected(tmp_path):
    out = tmp_path / "missing"
    DiagnosticProvenanceWriter(enabled=True).write(out, synthetic_manifest(), arrays=arrays())
    manifest = json.loads((out / "manifest.json").read_text())
    path = out / manifest["representations"]["spatial_components"]["path"]
    path.unlink()
    with pytest.raises(ProvenanceValidationError, match="corrupt or missing"):
        validate_bundle(out)


def test_missing_reference_and_conflated_runner_up_are_rejected(tmp_path):
    bad = synthetic_manifest()
    bad["candidates"][0]["ranking_scope"] = "global_final"
    with pytest.raises(ProvenanceValidationError, match="coarse runner-up"):
        DiagnosticProvenanceWriter(enabled=True).write(tmp_path / "bad", bad, arrays=arrays())


def test_output_bound_is_enforced_before_write(tmp_path):
    with pytest.raises(ProvenanceValidationError, match="bounded output"):
        DiagnosticProvenanceWriter(enabled=True, max_output_bytes=100).write(
            tmp_path / "too_big", synthetic_manifest(), arrays=arrays()
        )
