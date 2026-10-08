"""Bind historic and current full hashes, then lstat the exact recording path."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat as stat_module
from typing import Any, Mapping


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _stat_tuple(value: os.stat_result) -> dict[str, int]:
    return {
        "st_dev": int(value.st_dev),
        "st_ino": int(value.st_ino),
        "st_mode": int(value.st_mode),
        "st_size": int(value.st_size),
        "st_mtime_ns": int(value.st_mtime_ns),
        "st_ctime_ns": int(value.st_ctime_ns),
    }


def _verify_historic_receipt(binding: Mapping[str, Any]) -> dict[str, Any]:
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
    for path, expected in expected_files.items():
        if _sha256(path) != expected:
            raise RuntimeError(f"historic voltage receipt member mismatch: {path.name}")
    manifest = json.loads(manifest_path.read_text())
    complete = json.loads(complete_path.read_text())
    receipt = json.loads(receipt_path.read_text())
    member = manifest.get("files", {}).get("Q0_FULL_HASH_RECEIPT.json", {})
    if (
        complete.get("status") != "complete"
        or complete.get("manifest_sha256") != policy["packet_manifest_sha256"]
        or member.get("sha256") != policy["receipt_sha256"]
        or int(member.get("bytes", -1)) != receipt_path.stat().st_size
        or receipt.get("status") != "pass"
        or receipt.get("binary_path") != binding["binary_path"]
        or int(receipt.get("bytes_read", -1)) != int(binding["binary_size_bytes"])
        or receipt.get("expected_sha256") != binding["binary_sha256"]
        or receipt.get("actual_sha256") != binding["binary_sha256"]
    ):
        raise RuntimeError("historic full-hash receipt linkage or values failed")
    return receipt


def _manifest_members(manifest_path: Path) -> dict[str, str]:
    result = {}
    for line in manifest_path.read_text().splitlines():
        digest, separator, name = line.partition("  ")
        if not separator or not name or Path(name).name != name:
            raise RuntimeError("current full-hash manifest contains an invalid member")
        result[name] = digest
    return result


def _verify_current_receipt(binding: Mapping[str, Any]) -> dict[str, Any]:
    policy = binding["current_full_hash_receipt"]
    packet = Path(policy["packet_path"])
    manifest_path = packet / "MANIFEST.sha256"
    complete_path = packet / "COMPLETE.json"
    receipt_path = packet / "CURRENT_VOLTAGE_FULL_HASH_RECEIPT.json"
    for path, expected in (
        (manifest_path, policy["packet_manifest_sha256"]),
        (complete_path, policy["packet_complete_sha256"]),
        (receipt_path, policy["receipt_sha256"]),
    ):
        if _sha256(path) != expected:
            raise RuntimeError(f"current voltage receipt member mismatch: {path.name}")
    members = _manifest_members(manifest_path)
    for name, digest in members.items():
        member = packet / name
        if not member.is_file() or _sha256(member) != digest:
            raise RuntimeError(f"current full-hash manifest member mismatch: {name}")
    complete = json.loads(complete_path.read_text())
    receipt = json.loads(receipt_path.read_text())
    if (
        complete.get("status") != "PASS_EXACT_MATCH"
        or complete.get("manifest_sha256") != policy["packet_manifest_sha256"]
        or complete.get("actual_sha256") != binding["binary_sha256"]
        or int(complete.get("bytes_read", -1)) != int(binding["binary_size_bytes"])
        or receipt.get("status") != "PASS_EXACT_MATCH"
        or receipt.get("binary_path") != binding["binary_path"]
        or receipt.get("binary_is_symlink") is not False
        or int(receipt.get("bytes_read", -1)) != int(binding["binary_size_bytes"])
        or receipt.get("expected_sha256") != binding["binary_sha256"]
        or receipt.get("actual_sha256") != binding["binary_sha256"]
        or receipt.get("stat_unchanged") is not True
        or receipt.get("source_sha256") != policy["verification_source_sha256"]
        or receipt.get("final_lstat") != policy["final_lstat"]
    ):
        raise RuntimeError("current full-hash receipt linkage or values failed")
    return receipt


def verify_voltage_identity(binding: Mapping[str, Any]) -> dict[str, Any]:
    """Verify compact receipts and exact no-follow stat without reading voltage."""
    historic = _verify_historic_receipt(binding)
    current = _verify_current_receipt(binding)
    binary_path = Path(binding["binary_path"])
    observed = os.lstat(binary_path)
    if not stat_module.S_ISREG(observed.st_mode) or binary_path.is_symlink():
        raise RuntimeError("voltage path must be the exact non-symlink regular file")
    observed_tuple = _stat_tuple(observed)
    if observed_tuple != binding["current_full_hash_receipt"]["final_lstat"]:
        raise RuntimeError("voltage binary stat identity changed after the current full hash")
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
    if {key: manifest_values.get(key) for key in expected_manifest} != expected_manifest:
        raise RuntimeError("recording manifest scalar/frame semantics mismatch")
    expected_channels = [f"imec0.ap#AP{index}" for index in range(int(binding["num_channels"]))]
    if manifest_values.get("physical_channel_ids") != expected_channels:
        raise RuntimeError("recording manifest physical channel order mismatch")
    return {
        "schema": "waveform-voltage-current-full-hash-bound-receipt-v1",
        "status": "PASS",
        "binary_path": str(binary_path),
        "binary_size_bytes": int(binding["binary_size_bytes"]),
        "binary_sha256_historic": historic["actual_sha256"],
        "binary_sha256_current": current["actual_sha256"],
        "historic_and_current_match": historic["actual_sha256"] == current["actual_sha256"],
        "current_full_hash_receipt_sha256": binding["current_full_hash_receipt"]["receipt_sha256"],
        "lstat": observed_tuple,
        "full_binary_hash_performed_by_this_compact_preflight": False,
        "voltage_bytes_read_by_this_compact_preflight": 0,
    }
