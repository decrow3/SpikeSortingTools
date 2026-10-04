import hashlib
import json
import os
from pathlib import Path

import pytest

from voltage_identity_preflight import verify_voltage_identity


PACKET = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_packet(tmp_path):
    binary = tmp_path / "voltage.raw"
    binary.write_bytes(b"0123456789abcdef")
    recording = tmp_path / "recording.json"
    recording.write_text(json.dumps({
        "schema_version": "rescue-recording-manifest-v2",
        "num_samples": 8,
        "num_channels": 1,
        "sampling_frequency_hz": 30000.0,
        "dtype": "int16",
        "selected_start_frame": 0,
        "selected_end_frame": 8,
        "external_voltage_motion_correction": False,
        "physical_channel_ids": ["imec0.ap#AP0"],
    }))
    digest = sha(binary)
    os.chmod(binary, 0)

    historic = tmp_path / "historic"
    historic.mkdir()
    old_receipt = {
        "status": "pass", "binary_path": str(binary), "bytes_read": binary.stat().st_size,
        "expected_sha256": digest, "actual_sha256": digest,
    }
    old_receipt_path = historic / "Q0_FULL_HASH_RECEIPT.json"
    old_receipt_path.write_text(json.dumps(old_receipt))
    old_receipt_sha = sha(old_receipt_path)
    old_manifest_path = historic / "MANIFEST.json"
    old_manifest_path.write_text(json.dumps({"files": {"Q0_FULL_HASH_RECEIPT.json": {
        "bytes": old_receipt_path.stat().st_size, "sha256": old_receipt_sha,
    }}}))
    old_manifest_sha = sha(old_manifest_path)
    old_complete_path = historic / "COMPLETE.json"
    old_complete_path.write_text(json.dumps({"status": "complete", "manifest_sha256": old_manifest_sha}))

    current = tmp_path / "current"
    current.mkdir()
    stat = os.lstat(binary)
    stat_value = {
        "st_dev": stat.st_dev, "st_ino": stat.st_ino, "st_mode": stat.st_mode,
        "st_size": stat.st_size, "st_mtime_ns": stat.st_mtime_ns, "st_ctime_ns": stat.st_ctime_ns,
    }
    source_sha = "7" * 64
    current_receipt = {
        "status": "PASS_EXACT_MATCH", "binary_path": str(binary), "binary_is_symlink": False,
        "bytes_read": binary.stat().st_size, "expected_sha256": digest, "actual_sha256": digest,
        "stat_unchanged": True, "final_lstat": stat_value, "source_sha256": source_sha,
    }
    current_receipt_path = current / "CURRENT_VOLTAGE_FULL_HASH_RECEIPT.json"
    current_receipt_path.write_text(json.dumps(current_receipt))
    started = current / "STARTED.json"
    started.write_text("{}")
    current_manifest_path = current / "MANIFEST.sha256"
    current_manifest_path.write_text(
        f"{sha(current_receipt_path)}  {current_receipt_path.name}\n"
        f"{sha(started)}  {started.name}\n"
    )
    current_manifest_sha = sha(current_manifest_path)
    current_complete_path = current / "COMPLETE.json"
    current_complete_path.write_text(json.dumps({
        "status": "PASS_EXACT_MATCH", "manifest_sha256": current_manifest_sha,
        "actual_sha256": digest, "bytes_read": binary.stat().st_size,
    }))
    binding = {
        "binary_path": str(binary), "binary_size_bytes": binary.stat().st_size,
        "binary_sha256": digest, "recording_manifest_path": str(recording),
        "recording_manifest_sha256": sha(recording), "num_samples": 8,
        "num_channels": 1, "sampling_frequency_hz": 30000.0,
        "immutable_full_hash_receipt": {
            "packet_path": str(historic), "packet_manifest_sha256": old_manifest_sha,
            "packet_complete_sha256": sha(old_complete_path), "receipt_sha256": old_receipt_sha,
        },
        "current_full_hash_receipt": {
            "packet_path": str(current), "packet_manifest_sha256": current_manifest_sha,
            "packet_complete_sha256": sha(current_complete_path),
            "receipt_sha256": sha(current_receipt_path),
            "verification_source_sha256": source_sha, "final_lstat": stat_value,
        },
    }
    return binding, binary, current_receipt_path


def test_current_and_historic_full_hashes_bind_without_compact_preflight_voltage_read(tmp_path):
    binding, binary, _ = make_packet(tmp_path)
    result = verify_voltage_identity(binding)
    assert result["status"] == "PASS"
    assert result["historic_and_current_match"] is True
    assert result["voltage_bytes_read_by_this_compact_preflight"] == 0


def test_same_size_replacement_fails_even_if_mtime_is_restored(tmp_path):
    binding, binary, _ = make_packet(tmp_path)
    original = os.lstat(binary)
    replacement = tmp_path / "replacement.raw"
    replacement.write_bytes(b"fedcba9876543210")
    os.utime(replacement, ns=(original.st_atime_ns, original.st_mtime_ns))
    os.chmod(replacement, 0)
    os.replace(replacement, binary)
    with pytest.raises(RuntimeError, match="stat identity changed"):
        verify_voltage_identity(binding)


def test_symlink_substitution_fails_closed(tmp_path):
    binding, binary, _ = make_packet(tmp_path)
    target = tmp_path / "target.raw"
    binary.rename(target)
    binary.symlink_to(target)
    with pytest.raises(RuntimeError, match="non-symlink regular file"):
        verify_voltage_identity(binding)


def test_current_receipt_corruption_fails_closed(tmp_path):
    binding, _, receipt = make_packet(tmp_path)
    receipt.write_text("{}")
    with pytest.raises(RuntimeError, match="current voltage receipt member mismatch"):
        verify_voltage_identity(binding)


def test_independent_runtime_receipt_binds_all_actual_h5_objects():
    receipt = json.loads((PACKET / "receipts/RUNTIME_VERIFICATION_RECEIPT.json").read_text())
    verifier = PACKET / "source/independent_h5_runtime_verifier.py"
    assert receipt["status"] == "PASS"
    assert receipt["scope"].startswith("actual H5 deployment")
    assert receipt["verifier_sha256"] == sha(verifier)
    assert receipt["member_count"] == len(receipt["members"]) == 89
    assert receipt["member_bytes_hashed"] == sum(row["bytes"] for row in receipt["members"]) == 3333088832
    assert receipt["aggregate_sha256"] == "6658089a5e123b1962d2a61b7b89c230e484fdde0f52a0479e52c102f10cfe48"
    assert receipt["backend"]["probe_output_sha256"] == "e904ebc218e24c983580f0f72ea012f9483a7ff72f64bea7b8cd4812aa7b55df"
    assert receipt["differences"] == []


def test_v5_changes_only_runtime_identity_and_voltage_continuity_boundaries():
    old = json.loads((PACKET / "snapshots/physical_waveform_evidence.execution_disabled.v4.json").read_text())
    new = json.loads((PACKET / "contracts/physical_waveform_evidence.execution_disabled.v5.json").read_text())
    assert new["execution_enabled"] is False
    assert new["voltage_access_enabled"] is False
    for key in (
        "comparisons", "input_identity", "epochs", "classification_and_matching",
        "sampling", "technical_support", "denominators_and_intervals",
        "quality_and_decision",
    ):
        assert new[key] == old[key]
    old_wave = dict(old["waveform_extraction"])
    new_wave = dict(new["waveform_extraction"])
    old_pre = dict(old_wave.pop("preprocessing"))
    new_pre = dict(new_wave.pop("preprocessing"))
    old_pre.pop("executable_runtime_closure")
    new_pre.pop("executable_runtime_closure")
    assert new_wave == old_wave
    assert new_pre == old_pre
    for key in (
        "event_reads_max", "logical_read_bytes_max", "read_accounting",
        "cpu_threads_max", "gpu_count", "ram_bytes_max",
        "compact_persistent_bytes_max", "local_transient_cache_bytes_max",
        "wall_seconds_max", "segment_read_receipt_rule", "stop_before_read",
        "stop_during_run", "on_stop",
    ):
        assert new["resources_and_stop_conditions"][key] == old["resources_and_stop_conditions"][key]
    assert new["voltage_identity"]["binary_path"] == old["voltage_identity"]["binary_path"]
    assert new["voltage_identity"]["binary_sha256"] == old["voltage_identity"]["binary_sha256"]
    assert new["voltage_identity"]["binary_size_bytes"] == old["voltage_identity"]["binary_size_bytes"]
    assert new["voltage_identity"]["current_full_hash_receipt"]["verified_sha256"] == old["voltage_identity"]["binary_sha256"]
