import hashlib
import json
from pathlib import Path

import pytest

from testing.validate_operational_pipeline import validate


IDENTITY = "a" * 64
RECORDING_REQUEST = "b" * 64
RECORDING_CONTENT = "c" * 64
SORT_REQUEST = "d" * 64


def _write(path: Path, value: dict) -> dict[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _profile(tmp_path: Path) -> Path:
    required = tmp_path / "required.out"
    required.write_text("complete")
    receipt = lambda stage, summary={}: {
        "stage": stage, "complete": True, "sort_identity_digest": IDENTITY,
        "required_files": [str(required)], "summary": summary,
    }
    sidecars = tmp_path / "sidecars"
    sidecars.mkdir()
    packaged = sidecars / "quality_summary.json"
    packaged.write_text("{}")
    package_manifest = {
        "schema": "luke-operational-pipeline-sidecars-v1", "status": "complete",
        "sort_identity_digest": IDENTITY,
        "files": [{"path": packaged.name, "bytes": packaged.stat().st_size,
                   "sha256": hashlib.sha256(packaged.read_bytes()).hexdigest()}],
        "total_bytes": packaged.stat().st_size,
    }
    artifacts = {
        "recording_manifest": _write(tmp_path / "recording.json", {
            "complete": True, "request_digest": RECORDING_REQUEST,
            "recording_content_sha256": RECORDING_CONTENT,
            "graph": ["phase", "blank", "interpolate"],
            "external_filter": None, "external_reference": None,
            "external_voltage_motion_correction": False,
        }),
        "sort_manifest": _write(tmp_path / "sort.json", {
            "complete": True, "request_digest": SORT_REQUEST,
            "recording_request_digest": RECORDING_REQUEST,
            "sorter_params": {"Th_universal": 12, "Th_learned": 9,
                              "do_correction": False, "do_CAR": True,
                              "artifact_threshold": "Infinity"},
            "summary": {"critical_saved_settings": {"effective_nblocks": 0}},
        }),
        "sort_identity": _write(tmp_path / "identity.json", {
            "identity_digest": IDENTITY, "request_digest": SORT_REQUEST,
        }),
        "curation_receipt": _write(tmp_path / "curation.json", receipt(
            "legacy_compatible_curation", {"spike_count": 200}
        )),
        "legacy_qc_receipt": _write(tmp_path / "qc.json", receipt("legacy_compatible_qc")),
        "standard_qc_receipt": _write(tmp_path / "standard.json", receipt(
            "standard_unit_quality", {"automatic_curation_changes": 0, "unit_count": 10,
                                      "spike_count": 200, "units_with_any_lab_warning": 2}
        )),
        "completeness_timeline_receipt": _write(tmp_path / "timeline.json", receipt(
            "amplitude_completeness_timeline", {"screening_only": True, "unit_count": 10,
                "window_count": 30, "measured_window_count": 20}
        )),
        "sidecar_manifest": _write(sidecars / "MANIFEST.json", package_manifest),
    }
    profile = {
        "schema_version": "luke-operational-pipeline-profile-v1",
        "identity": {"recording_request_digest": RECORDING_REQUEST,
                     "recording_content_sha256": RECORDING_CONTENT,
                     "sort_request_digest": SORT_REQUEST, "sort_identity_digest": IDENTITY},
        "processing": {"preprocessing_graph": ["phase", "blank", "interpolate"],
            "external_filter": None, "external_reference": None,
            "external_voltage_motion_correction": False,
            "critical_sorter_settings": {"Th_universal": 12, "Th_learned": 9,
                "do_correction": False, "effective_nblocks": 0, "do_CAR": True,
                "artifact_threshold": "Infinity"}},
        "artifacts": artifacts,
        "downstream_policy": {"standard_qc_automatic_curation_changes": 0,
                              "completeness_timeline_screening_only": True},
        "candidate_status": {"external_rigid": "comparison_pending"},
    }
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile))
    return path


def test_validates_identity_bound_operational_profile(tmp_path):
    result = validate(_profile(tmp_path))
    assert result["status"] == "pass"
    assert result["curated_units"] == 10
    assert result["measured_completeness_windows"] == 20


def test_rejects_changed_attested_artifact(tmp_path):
    profile = _profile(tmp_path)
    (tmp_path / "timeline.json").write_text("{}")
    with pytest.raises(RuntimeError, match="operational artifact changed"):
        validate(profile)
