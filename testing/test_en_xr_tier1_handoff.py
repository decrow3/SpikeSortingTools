from __future__ import annotations

import hashlib
import json
from pathlib import Path

from testing.en_xr_tier1_handoff import REQUIRED_FILES, publish, verify


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _source(root: Path, identity: str = "abc123") -> Path:
    payloads = {
        "spike_times.npy": b"times",
        "spike_clusters.npy": b"clusters",
        "spike_positions.npy": b"positions",
        "cluster_KSLabel.tsv": b"cluster_id\tKSLabel\n0\tgood\n",
    }
    output = root / "kilosort4/sorter_output"
    output.mkdir(parents=True)
    for name, data in payloads.items():
        (output / name).write_bytes(data)
    bound = {
        name: {"size_bytes": len(payloads[name]), "sha256": _sha(payloads[name])}
        for name in ("spike_times.npy", "spike_clusters.npy", "cluster_KSLabel.tsv")
    }
    (root / "sort_identity.json").write_text(
        json.dumps({"identity_digest": identity, "files": bound}) + "\n"
    )
    (root / "summary.json").write_text(
        json.dumps({"status": "complete", "sort_identity_digest": identity}) + "\n"
    )
    return root


def test_publish_and_verify_round_trip(tmp_path: Path) -> None:
    source = _source(tmp_path / "source")
    destination = tmp_path / "packet"
    manifest = publish(source, destination, "abc123")
    assert manifest["sort_identity_digest"] == "abc123"
    assert [row["path"] for row in manifest["files"]] == list(REQUIRED_FILES)
    assert verify(destination, "abc123")["total_bytes"] > 0
    complete = json.loads((destination / "COMPLETE.json").read_text())
    assert complete["manifest_sha256"] == hashlib.sha256(
        (destination / "MANIFEST.json").read_bytes()
    ).hexdigest()


def test_verify_rejects_modified_payload(tmp_path: Path) -> None:
    destination = tmp_path / "packet"
    publish(_source(tmp_path / "source"), destination, "abc123")
    (destination / "kilosort4/sorter_output/spike_positions.npy").write_bytes(b"changed")
    try:
        verify(destination, "abc123")
    except RuntimeError as error:
        assert "changed" in str(error)
    else:
        raise AssertionError("modified payload was accepted")
