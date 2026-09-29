import json
from pathlib import Path

import pytest

from pipeline.downstream import run_operational_pipeline_validation_stage


IDENTITY = "f" * 64


def _receipt(path: Path, stage: str, summary: dict, identity: str = IDENTITY) -> None:
    output = path.with_suffix(".out")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("done")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "schema_version": "rescue-downstream-stage-receipt-v1",
        "stage": stage,
        "request_digest": stage,
        "sort_identity_digest": identity,
        "complete": True,
        "required_files": [str(output)],
        "summary": summary,
    }))


def _tree(root: Path, *, screening_only: bool = True) -> None:
    _receipt(root / "cur/curation_receipt.json", "legacy_compatible_curation",
             {"unit_count": 10, "spike_count": 100})
    _receipt(root / "qc/qc_receipt.json", "legacy_compatible_qc", {"unit_count": 10})
    _receipt(root / "qc/standard/standard_qc_receipt.json", "standard_unit_quality",
             {"unit_count": 10, "spike_count": 100, "automatic_curation_changes": 0})
    _receipt(root / "qc/completeness_timeline/completeness_timeline_receipt.json",
             "amplitude_completeness_timeline",
             {"unit_count": 10, "screening_only": screening_only})


def test_seals_complete_identity_consistent_pipeline(tmp_path):
    _tree(tmp_path)
    identity = {"identity_digest": IDENTITY}
    first = run_operational_pipeline_validation_stage(tmp_path, identity)
    second = run_operational_pipeline_validation_stage(tmp_path, identity)
    assert first == second
    assert first["summary"]["status"] == "pass"
    assert first["summary"]["unit_count"] == 10


def test_rejects_non_screening_completeness_policy(tmp_path):
    _tree(tmp_path, screening_only=False)
    with pytest.raises(RuntimeError, match="not screening-only"):
        run_operational_pipeline_validation_stage(tmp_path, {"identity_digest": IDENTITY})


def test_rejects_mixed_sort_identities(tmp_path):
    _tree(tmp_path)
    path = tmp_path / "qc/standard/standard_qc_receipt.json"
    value = json.loads(path.read_text())
    value["sort_identity_digest"] = "0" * 64
    path.write_text(json.dumps(value))
    with pytest.raises(RuntimeError, match="another sort identity"):
        run_operational_pipeline_validation_stage(tmp_path, {"identity_digest": IDENTITY})
