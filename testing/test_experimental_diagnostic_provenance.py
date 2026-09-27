import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from testing.experimental.diagnostic_provenance import (
    DiagnosticProvenanceWriter,
    ProvenanceValidationError,
    SCHEMA,
    validate_bundle,
    validate_manifest_semantics,
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


def semantic_manifest():
    manifest = synthetic_manifest()
    manifest["representations"].update({
        "temporal_components": {"status": "unavailable", "reason": "semantic fixture"},
        "spatial_components": {"status": "unavailable", "reason": "semantic fixture"},
    })
    return manifest


def available_ref(*, sha256="b" * 64, shape=None):
    return {
        "status": "available", "logical_name": "temporal_components",
        "path": f"objects/{sha256}.npy", "sha256": sha256,
        "shape": [3, 5] if shape is None else shape, "dtype": "<f4",
        "encoding": "npy-v1-content-addressed", "bytes": 188,
    }


def rewrite_bundle_manifest(out, mutate):
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    mutate(manifest)
    payload = (json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode()
    manifest_path.write_bytes(payload)
    complete_path = out / "COMPLETE.json"
    complete = json.loads(complete_path.read_text())
    complete["manifest_sha256"] = hashlib.sha256(payload).hexdigest()
    complete_path.write_text(json.dumps(complete))


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


def test_writer_rejects_preexisting_available_reference_without_publishing(tmp_path):
    out = tmp_path / "missing-payload"
    manifest = synthetic_manifest()
    manifest["representations"].update({
        "temporal_components": available_ref(),
        "spatial_components": {"status": "unavailable", "reason": "fixture"},
    })
    with pytest.raises(ProvenanceValidationError, match="supply array payload"):
        DiagnosticProvenanceWriter(enabled=True).write(out, manifest)
    assert not out.exists()


@pytest.mark.parametrize(
    "ref, match",
    [
        (available_ref(sha256="bad"), "invalid representation"),
        (available_ref(shape=[3, 1.5]), "invalid representation"),
    ],
)
def test_writer_rejects_invalid_preexisting_reference_without_publishing(tmp_path, ref, match):
    out = tmp_path / "invalid-reference"
    manifest = synthetic_manifest()
    manifest["representations"].update({
        "temporal_components": ref,
        "spatial_components": {"status": "unavailable", "reason": "fixture"},
    })
    with pytest.raises(ProvenanceValidationError, match=match):
        DiagnosticProvenanceWriter(enabled=True).write(out, manifest)
    assert not out.exists()


def test_validation_happens_before_publish_and_preserves_partial(tmp_path, monkeypatch):
    from testing.experimental import diagnostic_provenance as provenance

    out = tmp_path / "publish"

    def reject_partial(path):
        assert Path(path).name.endswith(".partial")
        raise ProvenanceValidationError("synthetic validation failure")

    monkeypatch.setattr(provenance, "validate_bundle", reject_partial)
    with pytest.raises(ProvenanceValidationError, match="synthetic validation failure"):
        DiagnosticProvenanceWriter(enabled=True).write(out, synthetic_manifest(), arrays=arrays())
    assert not out.exists()
    assert out.with_name(out.name + ".partial").is_dir()


def test_missing_reference_and_conflated_runner_up_are_rejected(tmp_path):
    bad = synthetic_manifest()
    bad["candidates"][0]["ranking_scope"] = "global_final"
    with pytest.raises(ProvenanceValidationError, match="invalid evaluated role"):
        DiagnosticProvenanceWriter(enabled=True).write(tmp_path / "bad", bad, arrays=arrays())


@pytest.mark.parametrize(
    "stage, role, scope",
    [
        ("coarse", "selected", "global_final"),
        ("fine", "coarse_selected", "coarse_local"),
        ("fine", "runner_up", "global_final"),
    ],
)
def test_contradictory_evaluated_role_stage_scope_is_rejected(stage, role, scope):
    bad = semantic_manifest()
    bad["candidates"][2].update(stage=stage, role=role, ranking_scope=scope)
    with pytest.raises(ProvenanceValidationError):
        validate_manifest_semantics(bad)


def test_selection_is_unique_per_event_stage_iteration_snapshot():
    manifest = semantic_manifest()
    manifest["candidates"][0]["role"] = "coarse_selected"
    validate_manifest_semantics(manifest)

    duplicate = copy.deepcopy(manifest["candidates"][2])
    duplicate["candidate_id"] = "c5"
    duplicate["template_id"] = "t1"
    manifest["candidates"].append(duplicate)
    with pytest.raises(ProvenanceValidationError, match="event/stage/iteration snapshot"):
        validate_manifest_semantics(manifest)

    duplicate["iteration"] = 1
    validate_manifest_semantics(manifest)


def test_evaluated_unmatched_cannot_claim_global_final_from_coarse_stage():
    manifest = semantic_manifest()
    manifest["candidates"][4]["stage"] = "coarse"
    with pytest.raises(ProvenanceValidationError, match="evaluated-unmatched"):
        validate_manifest_semantics(manifest)


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda m: m["clock"].__setitem__("sampling_frequency_hz", float("nan")), "positive rate"),
        (lambda m: m["clock"].__setitem__("origin_time_s", float("inf")), "must be finite"),
        (lambda m: m["clock"].__setitem__("start_sample", 100.5), "integer sample"),
        (lambda m: m["events"][0].__setitem__("sample_index", 120.5), "integer sample"),
        (lambda m: m["candidates"][0].__setitem__("iteration", 0.5), "nonnegative integer"),
    ],
)
def test_public_bundle_validator_rejects_nonfinite_or_fractional_fields(tmp_path, mutate, match):
    out = tmp_path / "numeric"
    DiagnosticProvenanceWriter(enabled=True).write(out, synthetic_manifest(), arrays=arrays())
    rewrite_bundle_manifest(out, mutate)
    with pytest.raises(ProvenanceValidationError, match=match):
        validate_bundle(out)


def test_output_bound_is_enforced_before_write(tmp_path):
    with pytest.raises(ProvenanceValidationError, match="bounded output"):
        DiagnosticProvenanceWriter(enabled=True, max_output_bytes=100).write(
            tmp_path / "too_big", synthetic_manifest(), arrays=arrays()
        )
