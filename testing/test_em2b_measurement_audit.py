import json
from pathlib import Path

from testing.em2b_measurement_audit import compare_artifact


def test_compare_artifact_distinguishes_mtime_from_content(tmp_path: Path):
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"fixed")
    recorded = {
        "path": "artifact.bin",
        "resolved_path": str(artifact),
        "bytes": artifact.stat().st_size,
        "mtime_ns": artifact.stat().st_mtime_ns - 1,
    }
    row = compare_artifact(recorded)
    assert row["differences"] == ["mtime_ns"]


def test_compare_artifact_checks_recorded_digest(tmp_path: Path):
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"changed")
    recorded = {
        "path": "artifact.bin",
        "resolved_path": str(artifact),
        "bytes": artifact.stat().st_size,
        "sha256": "0" * 64,
    }
    row = compare_artifact(recorded)
    assert row["differences"] == ["sha256"]
    assert len(row["current"]["sha256"]) == 64
