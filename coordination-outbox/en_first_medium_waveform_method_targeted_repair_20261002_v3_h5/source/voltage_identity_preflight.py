"""No-rehash voltage identity preflight using an immutable reviewed receipt."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_voltage_identity(binding: Mapping[str, Any]) -> dict[str, Any]:
    """Verify compact provenance and stat the binary without opening its bytes."""
    policy = binding["immutable_full_hash_receipt"]
    packet = Path(policy["packet_path"])
    manifest_path = packet / "MANIFEST.json"
    complete_path = packet / "COMPLETE.json"
    receipt_path = packet / "Q0_FULL_HASH_RECEIPT.json"
    expected_files = {
        manifest_path: policy["packet_manifest_sha256"],
        complete_path: policy["packet_complete_sha256"],
        receipt_path: policy["receipt_sha256"],
    }
    observed_files = {str(path): _sha256(path) for path in expected_files}
    for path, expected in expected_files.items():
        if observed_files[str(path)] != expected:
            raise RuntimeError(f"immutable voltage receipt member mismatch: {path.name}")
    manifest = json.loads(manifest_path.read_text())
    complete = json.loads(complete_path.read_text())
    receipt = json.loads(receipt_path.read_text())
    member = manifest.get("files", {}).get("Q0_FULL_HASH_RECEIPT.json", {})
    if (
        complete.get("status") != "complete"
        or complete.get("manifest_sha256") != policy["packet_manifest_sha256"]
        or member.get("sha256") != policy["receipt_sha256"]
        or int(member.get("bytes", -1)) != receipt_path.stat().st_size
    ):
        raise RuntimeError("immutable voltage receipt packet linkage failed")
    binary_path = Path(binding["binary_path"])
    size = int(binding["binary_size_bytes"])
    digest = binding["binary_sha256"]
    if (
        receipt.get("status") != "pass"
        or receipt.get("binary_path") != str(binary_path)
        or int(receipt.get("bytes_read", -1)) != size
        or receipt.get("expected_sha256") != digest
        or receipt.get("actual_sha256") != digest
    ):
        raise RuntimeError("reviewed full-hash receipt does not bind the frozen voltage identity")
    stat = os.stat(binary_path, follow_symlinks=True)
    if not os.path.isfile(binary_path) or stat.st_size != size:
        raise RuntimeError("voltage binary path/type/size mismatch")
    recording_manifest = Path(binding["recording_manifest_path"])
    if _sha256(recording_manifest) != binding["recording_manifest_sha256"]:
        raise RuntimeError("recording manifest byte mismatch")
    manifest_values = json.loads(recording_manifest.read_text())
    expected_manifest = {
        "schema_version": "rescue-recording-manifest-v2",
        "num_samples": int(binding["num_samples"]),
        "num_channels": int(binding["num_channels"]),
        "sampling_frequency_hz": float(binding["sampling_frequency_hz"]),
        "dtype": "int16",
        "selected_start_frame": 0,
        "selected_end_frame": int(binding["num_samples"]),
        "external_voltage_motion_correction": False,
    }
    observed_manifest = {key: manifest_values.get(key) for key in expected_manifest}
    if observed_manifest != expected_manifest:
        raise RuntimeError("recording manifest scalar/frame semantics mismatch")
    expected_channels = [f"imec0.ap#AP{index}" for index in range(int(binding["num_channels"]))]
    if manifest_values.get("physical_channel_ids") != expected_channels:
        raise RuntimeError("recording manifest physical channel order mismatch")
    return {
        "schema": "waveform-voltage-identity-no-rehash-receipt-v1",
        "status": "PASS",
        "binary_path": str(binary_path),
        "binary_size_bytes": size,
        "binary_sha256_from_reviewed_receipt": digest,
        "reviewed_full_hash_receipt_sha256": policy["receipt_sha256"],
        "reviewed_receipt_packet_manifest_sha256": policy["packet_manifest_sha256"],
        "recording_manifest_sha256": binding["recording_manifest_sha256"],
        "stat": {
            "st_dev": int(stat.st_dev),
            "st_ino": int(stat.st_ino),
            "st_size": int(stat.st_size),
            "st_mtime_ns": int(stat.st_mtime_ns),
        },
        "full_binary_hash_performed_this_run": False,
        "voltage_bytes_read_by_identity_preflight": 0,
        "rule": "Recheck this exact stat tuple after the last bounded segment read; any mutation fails the run.",
    }
