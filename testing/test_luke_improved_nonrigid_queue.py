import hashlib
import json
import os
from types import SimpleNamespace

import numpy as np
import pytest

from pipeline.sorting import _json_safe_params, build_kilosort4_params
from testing.luke_improved_nonrigid_queue import (
    CONTRACT_SCHEMA,
    EXPECTED_RECORDING_SHA256,
    _corrected_recording,
    _gpu_busy,
    ready_package,
)


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _package(tmp_path, *, ready=True, threshold=12):
    shared = tmp_path / "shared"
    package = shared / "estimation_huklaban1_v1"
    package.mkdir(parents=True)
    np.savez(
        package / "candidate_fields.npz",
        time_s=np.array([0.0, 1.0]),
        depth_um=np.array([0.0, 20.0]),
        nonrigid_displacement_um=np.zeros((2, 2)),
    )
    manifest = {
        "schema": "luke-shared-medicine-field-v1",
        "status": "estimation_artifacts_complete",
        "correction_ready": ready,
        "recording_content_sha256": EXPECTED_RECORDING_SHA256,
        "field_package": package.name,
        "files": {"candidate_fields.npz": _sha(package / "candidate_fields.npz")},
    }
    (shared / "field_manifest.json").write_text(json.dumps(manifest))
    contract = {
        "schema": CONTRACT_SCHEMA,
        "status": "frozen",
        "correction_ready": True,
        "field_manifest_sha256": _sha(shared / "field_manifest.json"),
        "recording_content_sha256": EXPECTED_RECORDING_SHA256,
        "nonrigid": {
            "field_file": "candidate_fields.npz",
            "time_array": "time_s",
            "depth_array": "depth_um",
            "displacement_array": "nonrigid_displacement_um",
            "polarity_convention": "output_depth = source_depth - displacement",
            "border_mode": "remove_channels",
            "spatial_interpolation_method": "kriging",
            "sigma_um": 20.0,
            "p": 1,
            "output_channel_ids": ["0", "1"],
            "leading_nearest_margin_max_s": 0.2,
            "trailing_nearest_margin_max_s": 0.2,
            "dtype": "int16",
        },
        "sorter": {
            **_json_safe_params(build_kilosort4_params()),
            "Th_universal": threshold,
        },
    }
    (shared / "run_contract.json").write_text(json.dumps(contract))
    return shared


def test_queue_refuses_estimator_artifact_until_separately_promoted(tmp_path):
    shared = _package(tmp_path, ready=False)
    with pytest.raises(ValueError, match="promoted"):
        ready_package(shared)


def test_queue_accepts_only_hash_bound_frozen_contract(tmp_path):
    shared = _package(tmp_path)
    manifest, contract, digest = ready_package(shared)
    assert manifest["correction_ready"] is True
    assert contract["sorter"]["do_correction"] is False
    assert digest == _sha(shared / "field_manifest.json")


def test_queue_refuses_sorter_drift_or_threshold_changes(tmp_path):
    shared = _package(tmp_path, threshold=11)
    with pytest.raises(ValueError, match="12/9"):
        ready_package(shared)


def test_gpu_gate_ignores_its_own_cuda_context(monkeypatch):
    monkeypatch.setattr(
        "testing.luke_improved_nonrigid_queue.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0, stdout=f"{os.getpid()}, python, 100\n", stderr=""
        ),
    )
    assert _gpu_busy() is False


def test_gpu_gate_waits_for_another_compute_process(monkeypatch):
    monkeypatch.setattr(
        "testing.luke_improved_nonrigid_queue.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0, stdout="999999, python, 100\n", stderr=""
        ),
    )
    assert _gpu_busy() is True


def test_gpu_gate_ignores_desktop_integration_context(monkeypatch):
    monkeypatch.setattr(
        "testing.luke_improved_nonrigid_queue.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout="127826, /snap/example/snapd-desktop-integration, 7\n",
            stderr="",
        ),
    )
    assert _gpu_busy() is False


def test_zero_field_is_identity_and_endpoint_margin_is_nearest(tmp_path):
    pytest.importorskip("spikeinterface.sortingcomponents.motion")
    from probeinterface import Probe
    from spikeinterface.core import NumpyRecording

    positions = np.c_[np.zeros(4), np.arange(4, dtype=float) * 20.0]
    probe = Probe(ndim=2, si_units="um")
    probe.set_contacts(positions=positions, shapes="circle", shape_params={"radius": 5})
    probe.set_device_channel_indices(np.arange(4))
    traces = np.arange(400, dtype=np.float32).reshape(100, 4)
    recording = NumpyRecording([traces], sampling_frequency=100.0).set_probe(probe)
    field = {
        "time_s": np.array([0.05, 0.95]),
        "depth_um": positions[:, 1],
        "displacement_um": np.zeros((2, 4)),
    }
    arm = {
        "leading_nearest_margin_max_s": 0.05,
        "trailing_nearest_margin_max_s": 0.05,
        "border_mode": "remove_channels",
        "spatial_interpolation_method": "kriging",
        "sigma_um": 20.0,
        "p": 1,
        "output_channel_ids": [0, 1, 2, 3],
    }
    corrected = _corrected_recording(recording, field, arm)
    assert corrected.dtype == np.dtype("int16")
    assert np.array_equal(corrected.get_traces(), traces.astype("int16"))


def test_nonzero_acquisition_origin_and_contract_channel_order_are_preserved():
    pytest.importorskip("spikeinterface.sortingcomponents.motion")
    from probeinterface import Probe
    from spikeinterface.core import NumpyRecording

    positions = np.c_[np.zeros(4), np.arange(4, dtype=float) * 20.0]
    probe = Probe(ndim=2, si_units="um")
    probe.set_contacts(positions=positions, shapes="circle", shape_params={"radius": 5})
    probe.set_device_channel_indices(np.arange(4))
    recording = NumpyRecording(
        [np.zeros((100, 4), dtype="float32")],
        sampling_frequency=100.0,
        t_starts=[3057.677050340359],
        channel_ids=["a", "b", "c", "d"],
    ).set_probe(probe)
    field = {
        "time_s": np.array([0.05, 0.95]),
        "depth_um": positions[:, 1],
        "displacement_um": np.zeros((2, 4)),
    }
    arm = {
        "leading_nearest_margin_max_s": 0.05,
        "trailing_nearest_margin_max_s": 0.05,
        "border_mode": "remove_channels",
        "spatial_interpolation_method": "kriging",
        "sigma_um": 20.0,
        "p": 1,
        "output_channel_ids": [2, 1],
    }
    corrected = _corrected_recording(recording, field, arm)
    assert list(corrected.get_channel_ids()) == [2, 1]
    interpolated = corrected._kwargs["recording"]._kwargs["parent_recording"]
    expected = field["time_s"] + recording.sample_index_to_time(0)
    assert np.array_equal(interpolated._kwargs["motion"].temporal_bins_s[0], expected)
