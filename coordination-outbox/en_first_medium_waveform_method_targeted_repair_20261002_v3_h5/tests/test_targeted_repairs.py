import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from retained_pair_correlation import REASON_PRIORITY, retained_pair_correlation
from runtime_closure import compare_closures
from voltage_identity_preflight import verify_voltage_identity


PACKET = Path(__file__).resolve().parents[1]
PARENT = Path("/mnt/NPX/Luke/DARTsort_motion_experiments/en_first_medium_waveform_contract_targeted_repair_20261002_v2_h5")
CONTRACT = PACKET / "contracts/physical_waveform_evidence.execution_disabled.v4.json"
RUNTIME = PACKET / "dependencies/EXECUTABLE_RUNTIME_CLOSURE.json"
IMPULSE = PARENT / "dependencies/kilosort/constants/highpass_filter_float32.npy"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def signal(channels, center, sign=1.0):
    absolute = np.arange(center - 20, center + 21, dtype=np.float64)
    return np.stack([sign * (absolute + 0.01 * int(channel)) for channel in channels])


def correlate(**overrides):
    values = {
        "reference_values": signal([3, 1, 2], 100),
        "candidate_values": signal([2, 3, 1], 100),
        "reference_channels": np.array([3, 1, 2]),
        "candidate_channels": np.array([2, 3, 1]),
        "reference_global_frame": 100,
        "candidate_global_frame": 100,
    }
    values.update(overrides)
    return retained_pair_correlation(**values)


def test_numeric_positive_negative_channel_order_and_absolute_alignment():
    positive = correlate()
    assert positive == {
        "status": "MEASURED", "reason": None, "common_channel_n": 3,
        "common_frame_n": 41, "correlation": pytest.approx(1.0),
    }
    negative = correlate(candidate_values=signal([2, 3, 1], 100, sign=-1.0))
    assert negative["status"] == "MEASURED"
    assert negative["correlation"] == pytest.approx(-1.0)
    shifted = correlate(
        candidate_values=signal([2, 3, 1], 115), candidate_global_frame=115
    )
    assert shifted["status"] == "MEASURED"
    assert shifted["common_frame_n"] == 26
    assert shifted["correlation"] == pytest.approx(1.0)


@pytest.mark.parametrize("constant_side", ["both", "reference", "candidate"])
def test_constant_vectors_and_one_sided_zero_variance(constant_side):
    changes = {}
    if constant_side in {"both", "reference"}:
        changes["reference_values"] = np.ones((3, 41))
    if constant_side in {"both", "candidate"}:
        changes["candidate_values"] = np.ones((3, 41))
    result = correlate(**changes)
    assert result["status"] == "UNMEASURED"
    assert result["reason"] == "ZERO_VARIANCE"
    assert result["correlation"] is None


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nan_and_inf_are_nonfinite_without_sample_dropping(bad):
    values = signal([3, 1, 2], 100)
    values[0, 0] = bad
    result = correlate(reference_values=values)
    assert result["reason"] == "NONFINITE_VALUES"
    assert result["common_frame_n"] == 41


def test_too_few_valid_frames_alignment_failure_and_insufficient_channels():
    too_few = correlate(
        candidate_values=signal([2, 3, 1], 116), candidate_global_frame=116
    )
    assert too_few["reason"] == "INSUFFICIENT_COMMON_FRAMES"
    assert too_few["common_frame_n"] == 25
    invalid_layout = correlate(reference_values=np.zeros((3, 40)))
    assert invalid_layout["reason"] == "ALIGNMENT_FAILURE"
    duplicate_channels = correlate(reference_channels=np.array([1, 1, 2]))
    assert duplicate_channels["reason"] == "ALIGNMENT_FAILURE"
    one_channel = correlate(
        reference_channels=np.array([7]), candidate_channels=np.array([7]),
        reference_values=signal([7], 100), candidate_values=signal([7], 100),
    )
    assert one_channel["reason"] == "INSUFFICIENT_COMMON_CHANNELS"


@pytest.mark.parametrize(
    "changes,reason",
    [
        ({"reference_values": None, "reference_selected": False}, "MISSING_ENDPOINT"),
        ({"reference_selected": False, "reference_technically_evaluable": False}, "ENDPOINT_NOT_SELECTED"),
        ({"reference_technically_evaluable": False, "reference_values": np.zeros((3, 40))}, "ENDPOINT_NOT_TECHNICALLY_EVALUABLE"),
        ({"reference_values": np.zeros((3, 40)), "reference_channels": np.array([7])}, "ALIGNMENT_FAILURE"),
        ({"reference_channels": np.array([7]), "candidate_channels": np.array([7]), "reference_values": signal([7], 100), "candidate_values": signal([7], 116), "candidate_global_frame": 116}, "INSUFFICIENT_COMMON_CHANNELS"),
        ({"candidate_values": np.full((3, 41), np.nan), "candidate_global_frame": 116}, "INSUFFICIENT_COMMON_FRAMES"),
        ({"candidate_values": np.full((3, 41), np.nan)}, "NONFINITE_VALUES"),
    ],
)
def test_overlapping_failure_conditions_follow_exact_priority(changes, reason):
    assert correlate(**changes)["reason"] == reason


def test_reason_priority_is_identical_in_contract_and_executable_helper():
    contract = json.loads(CONTRACT.read_text())
    frozen = contract["technical_support"]["retained_pair_correlation"]["invalid_reason_priority"]
    assert tuple(frozen) == REASON_PRIORITY
    assert sha(PACKET / contract["technical_support"]["retained_pair_correlation"]["implementation"]) == contract["technical_support"]["retained_pair_correlation"]["implementation_sha256"]
    assert sha(PACKET / contract["voltage_identity"]["preflight"]["implementation"]) == contract["voltage_identity"]["preflight"]["implementation_sha256"]
    runtime = contract["waveform_extraction"]["preprocessing"]["executable_runtime_closure"]
    assert sha(PACKET / runtime["path"]) == runtime["sha256"]
    assert sha(PACKET / runtime["verification_implementation"]) == runtime["verification_implementation_sha256"]


def test_accepted_preprocessing_matching_sampling_and_read_arithmetic_are_unchanged():
    repaired = json.loads(CONTRACT.read_text())
    accepted = json.loads((PARENT / "contracts/physical_waveform_evidence.execution_disabled.v3.json").read_text())
    assert repaired["classification_and_matching"] == accepted["classification_and_matching"]
    assert repaired["sampling"] == accepted["sampling"]
    assert repaired["waveform_extraction"]["core_offsets_half_open_samples"] == accepted["waveform_extraction"]["core_offsets_half_open_samples"]
    assert repaired["waveform_extraction"]["raw_read_offsets_half_open_samples"] == accepted["waveform_extraction"]["raw_read_offsets_half_open_samples"]
    assert repaired["waveform_extraction"]["raw_logical_bytes_per_record"] == accepted["waveform_extraction"]["raw_logical_bytes_per_record"]
    old_preprocessing = copy.deepcopy(accepted["waveform_extraction"]["preprocessing"])
    new_preprocessing = copy.deepcopy(repaired["waveform_extraction"]["preprocessing"])
    new_preprocessing.pop("executable_runtime_closure")
    new_preprocessing.pop("accepted_dependency_resolution")
    assert new_preprocessing == old_preprocessing
    old_resources = accepted["resources_and_stop_conditions"]
    new_resources = repaired["resources_and_stop_conditions"]
    assert new_resources["event_reads_max"] == old_resources["event_reads_max"] == 4608
    assert new_resources["logical_read_bytes_max"] == old_resources["logical_read_bytes_max"] == 4052090880
    assert new_resources["read_accounting"] == old_resources["read_accounting"]


def test_runtime_closure_binds_executable_loaded_elf_and_portable_byte_identity():
    closure = json.loads(RUNTIME.read_text())
    assert closure["member_count"] == len(closure["members"]) == 89
    assert closure["member_bytes"] == sum(row["bytes"] for row in closure["members"])
    interpreters = [row for row in closure["members"] if row["role"] == "python_executable"]
    assert len(interpreters) == 1 and interpreters[0]["basename"] == "python3.12"
    assert {"_C.cpython-312-x86_64-linux-gnu.so", "libtorch_cpu.so", "libtorch_python.so"} <= {
        row["basename"] for row in closure["members"]
    }
    relocated = copy.deepcopy(closure)
    for row in relocated["members"]:
        row["observed_path"] = "/relocated/" + row["basename"]
    assert compare_closures(closure, relocated) == []
    corrupted = copy.deepcopy(relocated)
    corrupted["members"][0]["sha256"] = "0" * 64
    assert compare_closures(closure, corrupted) == ["member_identity_multiset"]


def test_actual_runtime_verifier_passes_clean_cpu_probe():
    env = os.environ.copy()
    env.update({
        "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": "-1",
        "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
        "OPENBLAS_NUM_THREADS": "4", "NUMBA_NUM_THREADS": "4",
        "PYTHONPATH": os.pathsep.join([
            str(PACKET / "source"),
            "/home/huklaban5/Documents/SpikeSortingTools/SpikeSortingTools/environments/rescue-production/.venv/lib/python3.12/site-packages",
        ]),
    })
    result = subprocess.run(
        [sys.executable, str(PACKET / "source/runtime_closure.py"),
         "--impulse", str(IMPULSE), "--verify", str(RUNTIME)],
        env=env, cwd=PACKET, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    receipt = json.loads(result.stdout)
    assert receipt["status"] == "PASS"
    assert receipt["member_bytes_hashed"] == 3333088832


def make_fake_voltage_packet(tmp_path):
    packet = tmp_path / "prior"
    packet.mkdir()
    binary = tmp_path / "voltage.raw"
    binary.write_bytes(b"not-real-voltage")
    os.chmod(binary, 0)
    receipt = {
        "status": "pass", "binary_path": str(binary), "bytes_read": binary.stat().st_size,
        "expected_sha256": "a" * 64, "actual_sha256": "a" * 64,
    }
    receipt_path = packet / "Q0_FULL_HASH_RECEIPT.json"
    receipt_path.write_text(json.dumps(receipt))
    receipt_sha = sha(receipt_path)
    manifest = {"files": {"Q0_FULL_HASH_RECEIPT.json": {
        "bytes": receipt_path.stat().st_size, "sha256": receipt_sha,
    }}}
    manifest_path = packet / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest))
    manifest_sha = sha(manifest_path)
    complete_path = packet / "COMPLETE.json"
    complete_path.write_text(json.dumps({"status": "complete", "manifest_sha256": manifest_sha}))
    recording_manifest = tmp_path / "recording.json"
    recording_manifest.write_text(json.dumps({
        "schema_version": "rescue-recording-manifest-v2",
        "num_samples": 1, "num_channels": 1, "sampling_frequency_hz": 30000.0,
        "dtype": "int16", "selected_start_frame": 0, "selected_end_frame": 1,
        "external_voltage_motion_correction": False,
        "physical_channel_ids": ["imec0.ap#AP0"],
    }))
    binding = {
        "binary_path": str(binary), "binary_size_bytes": binary.stat().st_size,
        "binary_sha256": "a" * 64, "recording_manifest_path": str(recording_manifest),
        "recording_manifest_sha256": sha(recording_manifest),
        "num_samples": 1, "num_channels": 1, "sampling_frequency_hz": 30000.0,
        "immutable_full_hash_receipt": {
            "packet_path": str(packet), "packet_manifest_sha256": manifest_sha,
            "packet_complete_sha256": sha(complete_path), "receipt_sha256": receipt_sha,
        },
    }
    return binding, binary


def test_voltage_identity_uses_prior_receipt_and_zero_binary_bytes(tmp_path):
    binding, binary = make_fake_voltage_packet(tmp_path)
    result = verify_voltage_identity(binding)
    assert result["status"] == "PASS"
    assert result["full_binary_hash_performed_this_run"] is False
    assert result["voltage_bytes_read_by_identity_preflight"] == 0
    assert result["stat"]["st_size"] == binary.stat().st_size


def test_voltage_identity_receipt_corruption_fails_closed(tmp_path):
    binding, _ = make_fake_voltage_packet(tmp_path)
    receipt = Path(binding["immutable_full_hash_receipt"]["packet_path"]) / "Q0_FULL_HASH_RECEIPT.json"
    os.chmod(receipt, 0o600)
    receipt.write_text("{}")
    with pytest.raises(RuntimeError, match="member mismatch"):
        verify_voltage_identity(binding)


def test_real_prior_receipt_bytes_are_bound_without_opening_voltage():
    contract = json.loads(CONTRACT.read_text())
    voltage = contract["voltage_identity"]
    prior = voltage["immutable_full_hash_receipt"]
    packet = Path(prior["packet_path"])
    assert sha(packet / "MANIFEST.json") == prior["packet_manifest_sha256"]
    assert sha(packet / "COMPLETE.json") == prior["packet_complete_sha256"]
    assert sha(packet / "Q0_FULL_HASH_RECEIPT.json") == prior["receipt_sha256"]
    assert voltage["preflight"]["fresh_full_binary_hash"] == "FORBIDDEN"
    assert voltage["preflight"]["voltage_bytes_read"] == 0
