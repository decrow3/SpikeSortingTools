#!/usr/bin/env python3
"""Frozen synthetic falsifier for the v13 post-filter support boundary.

This opens no recording. It invokes the hash-bound production wrapper and
native Kilosort BinaryFiltered path on deterministic synthetic int16 data.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
import torch


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require_hash(path: str | Path, expected: str, label: str) -> None:
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(f"{label} hash mismatch: {observed} != {expected}")


def load_production_modules(contract: dict):
    root = Path(contract["production_source_root"])
    sys.path.insert(0, str(root))
    candidate = importlib.import_module(
        "npx_preprocessing.motion.kilosort_support_mask_candidate"
    )
    native_io = importlib.import_module("kilosort.io")
    preprocessing = importlib.import_module("kilosort.preprocessing")
    expected_candidate = Path(contract["bindings"]["production_candidate"]["path"])
    if Path(candidate.__file__).resolve() != expected_candidate.resolve():
        raise ValueError("wrong production candidate module imported")
    return candidate, native_io, preprocessing


def production_geometry(contract: dict) -> np.ndarray:
    source = json.loads(Path(contract["bindings"]["geometry_receipt"]["path"]).read_text())
    geometry = np.asarray(source["recording"]["channel_locations_um"], dtype=np.float64)
    expected = contract["fixture"]["geometry"]
    if geometry.shape != tuple(expected["shape"]):
        raise ValueError("geometry shape mismatch")
    if geometry[:, 0].min() != expected["x_min_um"] or geometry[:, 0].max() != expected["x_max_um"]:
        raise ValueError("geometry x bounds mismatch")
    if geometry[:, 1].min() != expected["y_min_um"] or geometry[:, 1].max() != expected["y_max_um"]:
        raise ValueError("geometry y bounds mismatch")
    if not np.array_equal(np.unique(geometry[:, 1]), np.arange(0.0, 3821.0, 20.0)):
        raise ValueError("geometry y lattice mismatch")
    return geometry


def synthetic_raw(contract: dict, *, perturb: bool, waveform: bool) -> np.ndarray:
    fixture = contract["fixture"]
    fs = float(fixture["fs_hz"])
    n_samples = int(fixture["file_samples"])
    n_channels = int(fixture["n_channels"])
    t = np.arange(n_samples, dtype=np.float64) / fs
    carrier = np.rint(
        8.0 * np.sin(2.0 * np.pi * 900.0 * t)
        + 4.0 * np.sin(2.0 * np.pi * 1800.0 * t)
    ).astype(np.int32)
    coefficients = np.arange(n_channels, dtype=np.int32) - 192
    raw = carrier[:, None] * coefficients[None, :]
    if waveform:
        spec = fixture["supported_waveform"]
        shape = np.asarray(spec["shape_counts"], dtype=np.int32)
        start = int(spec["start_frame"])
        channel = int(spec["channel"])
        raw[start:start + shape.size, channel] += shape
    if perturb:
        spec = fixture["unsupported_perturbation"]
        rows = np.asarray(spec["rows"], dtype=np.int64)
        raw[int(spec["start_frame"]):int(spec["stop_frame_exclusive"]), rows] += int(spec["amplitude_counts"])
    if raw.min() < np.iinfo(np.int16).min or raw.max() > np.iinfo(np.int16).max:
        raise ValueError("synthetic data exceeds int16")
    return raw.astype(np.int16)


def policy(candidate, geometry: np.ndarray, contract: dict, transition: bool):
    fixture = contract["fixture"]
    fs = float(fixture["fs_hz"])
    transition_frame = int(fixture["transition_frame"])
    centers = np.asarray(
        [transition_frame / (2.0 * fs), 3.0 * transition_frame / (2.0 * fs)],
        dtype=np.float64,
    )
    shifts = np.asarray([0.0, 40.0 if transition else 0.0], dtype=np.float64)
    validated = candidate.ValidatedSupportMaskContract(
        config_sha256="synthetic-fixture",
        contract_digest="synthetic-fixture",
        execution_enabled=True,
        geometry=geometry,
        temporal_centers_s=centers,
        shifts_um=shifts,
        fs=fs,
        imin=0,
        imax=int(fixture["file_samples"]),
        n_channels=int(fixture["n_channels"]),
        recording_manifest_sha256="synthetic-no-recording",
        recording_content_sha256="synthetic-no-recording",
        field_sha256="synthetic-two-state",
    )
    return candidate.policy_from_contract(validated)


def run_operator(candidate, native_io, hp_filter, geometry, contract, raw, *, transition, do_car):
    fixture = contract["fixture"]
    bound_policy = policy(candidate, geometry, contract, transition)
    binary_class = candidate.make_support_masked_binary_class(native_io, bound_policy)
    with tempfile.TemporaryDirectory(prefix="v13_support_boundary_fixture_") as temporary:
        raw_path = Path(temporary) / "synthetic.raw"
        raw.tofile(raw_path)
        bfile = binary_class(
            filename=str(raw_path),
            n_chan_bin=int(fixture["n_channels"]),
            fs=float(fixture["fs_hz"]),
            NT=int(fixture["batch_size"]),
            nt=int(fixture["padding_samples"]),
            nt0min=20,
            chan_map=np.arange(int(fixture["n_channels"])),
            hp_filter=hp_filter,
            whiten_mat=torch.eye(int(fixture["n_channels"]), dtype=torch.float32),
            dshift=None,
            device=torch.device("cpu"),
            do_CAR=bool(do_car),
            artifact_threshold=np.inf,
            invert_sign=False,
            dtype="int16",
            tmin=0.0,
            tmax=np.inf,
        )
        try:
            output, inds = bfile.padded_batch_to_torch(0, return_inds=True)
        finally:
            bfile.close()
    return output.detach().cpu().numpy(), [int(inds[0]), int(inds[1])], list(bound_policy.audit.batches)


def difference_metrics(left: np.ndarray, right: np.ndarray, rows: np.ndarray, columns: np.ndarray) -> dict:
    delta = np.asarray(left[np.ix_(rows, columns)] - right[np.ix_(rows, columns)], dtype=np.float64)
    return {
        "max_abs": float(np.max(np.abs(delta))),
        "rms": float(np.sqrt(np.mean(delta * delta))),
        "nonzero_values": int(np.count_nonzero(delta)),
        "values": int(delta.size),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    contract_path = Path(args.contract)
    output = Path(args.output)
    if output.exists():
        raise RuntimeError("fresh output required")
    contract = json.loads(contract_path.read_text())
    if contract.get("execution_enabled") is not True or contract.get("synthetic_only") is not True:
        raise PermissionError("contract is not enabled for synthetic-only execution")
    if Path(args.output).resolve() != Path(contract["output"]).resolve():
        raise ValueError("output path differs from frozen contract")
    if sha256_file(__file__) != contract["bindings"]["fixture_source"]["sha256"]:
        raise ValueError("fixture source hash mismatch")
    for binding in contract["bindings"].values():
        if binding["path"] != str(Path(__file__).resolve()):
            require_hash(binding["path"], binding["sha256"], binding.get("label", "binding"))
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["NVIDIA_VISIBLE_DEVICES"] = "void"
    torch.set_num_threads(int(contract["resources"]["cpu_threads_max"]))
    started = time.monotonic()
    candidate, native_io, preprocessing = load_production_modules(contract)
    geometry = production_geometry(contract)
    fixture = contract["fixture"]
    hp_filter = preprocessing.get_highpass_filter(
        fs=float(fixture["fs_hz"]), cutoff=float(fixture["highpass_cutoff_hz"]),
        device=torch.device("cpu"),
    )
    zero = synthetic_raw(contract, perturb=False, waveform=True)
    perturbed = synthetic_raw(contract, perturb=True, waveform=True)
    no_waveform = synthetic_raw(contract, perturb=False, waveform=False)

    transition_zero, inds, audit_zero = run_operator(
        candidate, native_io, hp_filter, geometry, contract, zero,
        transition=True, do_car=True,
    )
    transition_zero_repeat, repeat_inds, audit_repeat = run_operator(
        candidate, native_io, hp_filter, geometry, contract, zero,
        transition=True, do_car=True,
    )
    transition_perturbed, perturb_inds, audit_perturb = run_operator(
        candidate, native_io, hp_filter, geometry, contract, perturbed,
        transition=True, do_car=True,
    )
    all_supported_zero, all_zero_inds, audit_all_zero = run_operator(
        candidate, native_io, hp_filter, geometry, contract, zero,
        transition=False, do_car=True,
    )
    all_supported_perturbed, all_perturb_inds, audit_all_perturb = run_operator(
        candidate, native_io, hp_filter, geometry, contract, perturbed,
        transition=False, do_car=True,
    )
    all_supported_no_waveform, no_waveform_inds, audit_no_waveform = run_operator(
        candidate, native_io, hp_filter, geometry, contract, no_waveform,
        transition=False, do_car=True,
    )
    car_off_zero, car_off_zero_inds, audit_car_off_zero = run_operator(
        candidate, native_io, hp_filter, geometry, contract, zero,
        transition=True, do_car=False,
    )
    car_off_perturbed, car_off_perturb_inds, audit_car_off_perturb = run_operator(
        candidate, native_io, hp_filter, geometry, contract, perturbed,
        transition=True, do_car=False,
    )
    all_inds = [inds, repeat_inds, perturb_inds, all_zero_inds, all_perturb_inds,
                no_waveform_inds, car_off_zero_inds, car_off_perturb_inds]
    if any(value != inds for value in all_inds):
        raise ValueError("operator index ranges differ")

    output_start = inds[0]
    transition_frame = int(fixture["transition_frame"])
    radius = int(fixture["measurement"]["transition_radius_samples"])
    columns = np.arange(transition_frame - radius - output_start,
                        transition_frame + radius + 1 - output_start, dtype=np.int64)
    supported_rows = np.arange(0, int(fixture["unsupported_perturbation"]["rows"][0]), dtype=np.int64)
    perturbed_rows = np.asarray(fixture["unsupported_perturbation"]["rows"], dtype=np.int64)
    transition_metrics = difference_metrics(
        transition_perturbed, transition_zero, supported_rows, columns
    )
    repeat_metrics = difference_metrics(
        transition_zero_repeat, transition_zero, supported_rows, columns
    )
    all_supported_metrics = difference_metrics(
        all_supported_perturbed, all_supported_zero, perturbed_rows, columns
    )
    car_off_metrics = difference_metrics(
        car_off_perturbed, car_off_zero, supported_rows, columns
    )
    wave = fixture["supported_waveform"]
    waveform_columns = np.arange(
        int(wave["measurement_start_frame"]) - output_start,
        int(wave["measurement_stop_frame_exclusive"]) - output_start,
        dtype=np.int64,
    )
    waveform_delta = (
        all_supported_zero[int(wave["channel"]), waveform_columns]
        - all_supported_no_waveform[int(wave["channel"]), waveform_columns]
    ).astype(np.float64)
    waveform_peak_to_peak = float(np.ptp(waveform_delta))

    expected_unsupported = len(fixture["unsupported_perturbation"]["rows"]) * (
        (int(fixture["batch_size"]) + 2 * int(fixture["padding_samples"]))
        - (transition_frame - output_start)
    )
    audits = [audit_zero, audit_repeat, audit_perturb]
    audit_ok = all(
        len(item) == 1
        and item[0]["transition_boundaries"] == 1
        and item[0]["unsupported_values"] == expected_unsupported
        for item in audits
    )
    controls = contract["acceptance"]["controls"]
    decisions = contract["acceptance"]["premise"]
    control_pass = {
        "zero_perturbation_repeat": repeat_metrics["max_abs"] <= controls["repeat_max_abs"],
        "all_supported_stimulus_active": all_supported_metrics["max_abs"] >= controls["all_supported_min_max_abs"],
        "car_off_cross_row_negative_control": car_off_metrics["max_abs"] <= controls["car_off_max_abs"],
        "supported_waveform_preserved": waveform_peak_to_peak >= controls["supported_waveform_min_peak_to_peak"],
        "actual_support_transition": audit_ok,
    }
    all_controls_pass = all(control_pass.values())
    if not all_controls_pass:
        verdict = "INVALID_FIXTURE_CONTROL_FAILURE"
    elif (transition_metrics["max_abs"] >= decisions["survives_min_max_abs"]
          and transition_metrics["rms"] >= decisions["survives_min_rms"]):
        verdict = "PREMISE_SURVIVES_CURRENT_BOUNDARY_CROSS_ROW_EFFECT"
    elif transition_metrics["max_abs"] <= decisions["falsified_max_abs"]:
        verdict = "PREMISE_FALSIFIED_NO_SUPPORTED_OUTPUT_EFFECT"
    else:
        verdict = "INDETERMINATE_NO_ADVANCE"

    elapsed = time.monotonic() - started
    if elapsed > float(contract["resources"]["wall_seconds_max"]):
        raise RuntimeError("wall limit exceeded")
    result = {
        "schema": "v13-support-boundary-synthetic-falsifier-result-v1",
        "status": verdict,
        "contract_sha256": sha256_file(contract_path),
        "source_sha256": sha256_file(__file__),
        "production_candidate_sha256": sha256_file(contract["bindings"]["production_candidate"]["path"]),
        "native_io_sha256": sha256_file(contract["bindings"]["native_io"]["path"]),
        "native_preprocessing_sha256": sha256_file(contract["bindings"]["native_preprocessing"]["path"]),
        "output_global_frames": [output_start, output_start + transition_zero.shape[1]],
        "transition_frame": transition_frame,
        "transition_supported_rows": [0, int(supported_rows[-1])],
        "transition_unsupported_rows": fixture["unsupported_perturbation"]["rows"],
        "metrics": {
            "transition_car_on_supported_rows": transition_metrics,
            "zero_perturbation_repeat": repeat_metrics,
            "all_supported_stimulus_active": all_supported_metrics,
            "transition_car_off_supported_rows": car_off_metrics,
            "supported_waveform_peak_to_peak": waveform_peak_to_peak,
        },
        "control_pass": control_pass,
        "all_controls_pass": all_controls_pass,
        "transition_audits": [item[0] for item in audits],
        "interpretation": {
            "can_establish": "whether the controlled signal on rows unsupported after the +40-um transition changes supported-row output through the reviewed current production preprocessing boundary",
            "cannot_establish": "real-voltage exposure, biological effect, sort benefit, causal relevance beyond this synthetic construction, or repair efficacy",
            "real_waveform_contract_gate": verdict == "PREMISE_SURVIVES_CURRENT_BOUNDARY_CROSS_ROW_EFFECT",
            "automatic_advancement": False,
        },
        "resources": {
            "device": "cpu",
            "cpu_threads": torch.get_num_threads(),
            "elapsed_seconds": elapsed,
            "real_voltage_bytes_read": 0,
            "gpu_work_started": False,
        },
    }
    output.mkdir(parents=False)
    (output / "RESULT.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
