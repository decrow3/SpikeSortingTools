import json
from pathlib import Path

import pytest

from testing.rigid_comparison_handoff import REQUIRED_FILES, publish, verify


IDENTITY = "06ea4b23a45335ef8a8428047cc54e64356905b6cdae6715e36f061f3cfdf555"


def fixture_source(root: Path) -> Path:
    source = root / "source"
    for relative in REQUIRED_FILES:
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".json":
            value = {"complete": True, "sort_identity_digest": IDENTITY}
            if path.name == "summary.json":
                value["status"] = "complete"
            if path.name == "sort_identity.json":
                value["identity_digest"] = IDENTITY
            path.write_text(json.dumps(value))
        else:
            path.write_bytes((relative + "\n").encode())
    return source


def test_publish_and_verify_minimal_package(tmp_path):
    source = fixture_source(tmp_path)
    destination = tmp_path / "published"
    manifest = publish(source, destination, IDENTITY)
    assert manifest["sort_identity_digest"] == IDENTITY
    assert [row["path"] for row in manifest["files"]] == list(REQUIRED_FILES)
    assert not destination.with_name("published.partial").exists()
    assert verify(destination, IDENTITY)["total_bytes"] > 0


def test_verify_rejects_changed_artifact(tmp_path):
    destination = tmp_path / "published"
    publish(fixture_source(tmp_path), destination, IDENTITY)
    (destination / "cur/cur_output/spike_times.npy").write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="artifact changed"):
        verify(destination, IDENTITY)


def test_publish_preserves_incomplete_partial(tmp_path):
    source = fixture_source(tmp_path)
    (source / "qc/qc_receipt.json").unlink()
    destination = tmp_path / "published"
    with pytest.raises(FileNotFoundError, match="missing required source"):
        publish(source, destination, IDENTITY)
    assert not destination.exists()
