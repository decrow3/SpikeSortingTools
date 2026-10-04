#!/usr/bin/env python3
"""Bounded CPU replay of Kilosort preprocessing and isolated seed extraction.

The input is an already-materialized EN recording.  This is a diagnostic of the
Kilosort 4.0.27 basis-learning seed pathway, not a sort, template fit, or purity
test.  Recording bytes are read only for batches frozen in the protocol.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import time

import numpy as np
import torch
from torch.fft import fft, fftshift, ifft

from kilosort.io import BinaryFiltered, BinaryRWFile
from kilosort.preprocessing import fft_highpass
from kilosort.spikedetect import my_max2d, my_sum2d
from npx_preprocessing.motion.lattice_remap_si import exact_coordinate_mapping
from testing.en_rounded_field import rounded_rigid_field


def sha256(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def atomic_json(path: Path, value) -> None:
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(clean(value), indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"empty table: {path}")
    tmp = path.with_suffix(path.suffix + ".partial")
    with tmp.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def load_field(spec: dict, num_samples: int, fs: float) -> dict:
    path = Path(spec["path"])
    if sha256(path) != spec["sha256"]:
        raise RuntimeError(f"field hash mismatch: {path}")
    with np.load(path, allow_pickle=False) as saved:
        times = np.asarray(saved[spec["time_key"]], dtype=np.float64)
        displacement = np.asarray(saved[spec["displacement_key"]], dtype=np.float64)
    if displacement.ndim == 2:
        displacement = np.median(displacement, axis=1)
    rounded, reference = rounded_rigid_field(displacement)
    edges = np.empty(len(times) + 1, dtype=np.float64)
    edges[1:-1] = (times[:-1] + times[1:]) / 2
    edges[0], edges[-1] = 0.0, num_samples / fs
    edges = np.clip(edges, 0.0, num_samples / fs)
    states = np.unique(rounded)
    change_boundaries_s = edges[1:-1][rounded[1:] != rounded[:-1]]
    return {"edges": edges, "rounded": rounded, "states": states,
            "change_boundaries_s": change_boundaries_s,
            "reference_median_um": float(reference)}


def assigned_states(samples: np.ndarray, field: dict, fs: float) -> np.ndarray:
    knot = np.searchsorted(field["edges"], samples.astype(np.float64) / fs, side="right") - 1
    knot = np.clip(knot, 0, len(field["rounded"]) - 1)
    return field["rounded"][knot]


def switch_distance_bins(samples: np.ndarray, field: dict, fs: float) -> np.ndarray:
    seconds = samples.astype(np.float64) / fs
    boundaries = field["change_boundaries_s"]
    insertion = np.searchsorted(boundaries, seconds, side="left")
    left = boundaries[np.clip(insertion - 1, 0, len(boundaries) - 1)]
    right = boundaries[np.clip(insertion, 0, len(boundaries) - 1)]
    distance = np.minimum(np.abs(seconds - left), np.abs(seconds - right))
    return np.select([distance < 0.25, distance < 1.0, distance < 5.0],
                     [0, 1, 2], default=3).astype(np.uint8)


def channel_support(geometry: np.ndarray, states: np.ndarray) -> np.ndarray:
    return np.stack([
        exact_coordinate_mapping(geometry, float(state), direction="y") >= 0
        for state in states
    ])


def manual_stages(raw: torch.Tensor, ops: dict) -> dict[str, torch.Tensor]:
    chan_map = torch.as_tensor(np.asarray(ops["chanMap"]), dtype=torch.long)
    mapped = raw[chan_map]
    mean_removed = mapped - mapped.mean(1).unsqueeze(1)
    car = mean_removed - torch.median(mean_removed, 0)[0]
    hp_filter = torch.as_tensor(np.asarray(ops["preprocessing"]["hp_filter"]), dtype=torch.float32)
    fwav = fft_highpass(hp_filter, NT=car.shape[1])
    highpass = torch.real(ifft(fft(car) * torch.conj(fwav)))
    highpass = fftshift(highpass, dim=-1)
    whitening = torch.as_tensor(np.asarray(ops["preprocessing"]["whiten_mat"]), dtype=torch.float32)
    whitened = whitening @ highpass
    return {"input": mapped, "post_mean": mean_removed, "post_car": car,
            "post_highpass": highpass, "post_whitening": whitened}


def isolated_seed_locations(X: torch.Tensor, threshold: float, nt: int) -> torch.Tensor:
    Xabs = X.abs()
    Xmax = my_max2d(Xabs, [4, 5])
    ispeak = torch.logical_and(Xmax == Xabs, Xabs > threshold).float()
    ispeak_sum = my_sum2d(ispeak, [6, 30])
    isolated = ((ispeak_sum == 1) * (ispeak == 1))
    isolated[:, :nt] = 0
    isolated[:, -nt:] = 0
    return isolated.nonzero()


def stage_summary(values: torch.Tensor, mask: torch.Tensor | None) -> dict:
    if mask is not None:
        values = values[mask]
    flat = values.reshape(-1).float()
    if not len(flat):
        return {"values": 0, "rms": None, "abs_q50": None, "abs_q90": None,
                "abs_q99": None, "abs_q999": None, "abs_max": None,
                "exact_zero_fraction": None}
    absolute = flat.abs()
    # Exact CPU NumPy quantiles avoid torch.quantile's 2^24-element limit.
    quantiles = np.quantile(absolute.numpy(), [0.5, 0.9, 0.99, 0.999])
    return {
        "values": int(len(flat)), "rms": float(torch.sqrt(torch.mean(flat * flat))),
        "abs_q50": float(quantiles[0]), "abs_q90": float(quantiles[1]),
        "abs_q99": float(quantiles[2]), "abs_q999": float(quantiles[3]),
        "abs_max": float(absolute.max()),
        "exact_zero_fraction": float(torch.mean((flat == 0).float())),
    }


def basic_summary(values: torch.Tensor) -> dict:
    flat = values.reshape(-1).float()
    if not len(flat):
        return {"values": 0, "rms": None, "abs_max": None}
    return {"values": int(len(flat)), "rms": float(torch.sqrt(torch.mean(flat * flat))),
            "abs_max": float(flat.abs().max())}


def controls(ops: dict) -> dict:
    n_channels = int(ops["n_chan_bin"])
    width = int(ops["batch_size"] + 2 * ops["nt"])
    zero = torch.zeros((n_channels, width), dtype=torch.float32)
    zero_stages = manual_stages(zero, ops)
    zero_max = {name: float(value.abs().max()) for name, value in zero_stages.items()}
    if any(value != 0.0 for value in zero_max.values()):
        raise RuntimeError(f"all-zero control failed: {zero_max}")
    wave = torch.sin(torch.linspace(0, 31.0, width, dtype=torch.float32))
    common = wave.repeat(n_channels, 1)
    common_stages = manual_stages(common, ops)
    common_post_car_max = float(common_stages["post_car"].abs().max())
    common_final_max = float(common_stages["post_whitening"].abs().max())
    if common_post_car_max != 0.0 or common_final_max != 0.0:
        raise RuntimeError("common-mode control failed")
    isolated = zero.clone(); isolated[100, width // 2] = 7.0
    isolated_count = int(len(isolated_seed_locations(isolated, 6.0, int(ops["nt"]))))
    nearby = isolated.clone(); nearby[100, width // 2 + 10] = 7.0
    nearby_count = int(len(isolated_seed_locations(nearby, 6.0, int(ops["nt"]))))
    if isolated_count != 1 or nearby_count != 0:
        raise RuntimeError((isolated_count, nearby_count))
    return {
        "all_zero_stage_max_abs": zero_max,
        "common_mode_post_car_max_abs": common_post_car_max,
        "common_mode_final_max_abs": common_final_max,
        "isolated_peak_expected_observed": [1, isolated_count],
        "two_nearby_peaks_expected_observed": [0, nearby_count],
        "pass": True,
    }


def validate_arm(spec: dict) -> tuple[dict, dict]:
    root = Path(spec["sorter_output"])
    binary = Path(spec["binary_path"])
    if binary.stat().st_size != spec["binary_bytes"]:
        raise RuntimeError(f"binary size differs: {binary}")
    for metadata in spec["metadata_hashes"]:
        path = Path(metadata["path"])
        if sha256(path) != metadata["sha256"]:
            raise RuntimeError(f"metadata hash differs: {path}")
    ops_path = root / "ops.npy"
    if sha256(ops_path) != spec["ops_sha256"]:
        raise RuntimeError(f"ops hash differs: {ops_path}")
    ops = np.load(ops_path, allow_pickle=True).item()
    required = spec["effective_settings"]
    observed = {
        "fs": float(ops["fs"]), "n_chan_bin": int(ops["n_chan_bin"]),
        "batch_size": int(ops["batch_size"]), "nt": int(ops["nt"]),
        "nt0min": int(ops["nt0min"]), "do_CAR": bool(ops["do_CAR"]),
        "invert_sign": bool(ops["invert_sign"]),
        "artifact_threshold": str(ops["artifact_threshold"]),
        "Th_single_ch": float(ops["settings"]["Th_single_ch"]),
        "nskip": int(ops["settings"]["nskip"]),
        "effective_nblocks": int(ops["nblocks"]),
        "dshift_is_none": ops["dshift"] is None,
        "data_dtype": str(ops["data_dtype"]),
    }
    if observed != required:
        raise RuntimeError(f"effective settings differ: {observed} != {required}")
    if not np.array_equal(np.asarray(ops["chanMap"]), np.arange(384)):
        raise RuntimeError("channel map is not the frozen identity map")
    return ops, observed


def run(config_path: Path, output: Path) -> dict:
    started = time.perf_counter()
    selected_config = json.loads(config_path.read_text())
    if selected_config.get("schema_version") == "en-materialized-seed-replay-v2":
        base_path = Path(selected_config["base_protocol"]["path"])
        if sha256(base_path) != selected_config["base_protocol"]["sha256"]:
            raise RuntimeError("base replay protocol hash mismatch")
        config = json.loads(base_path.read_text())
        config["schema_version"] = selected_config["schema_version"]
        config["implementation_source_sha256"] = selected_config["implementation_source_sha256"]
        config["measurement_amendment"] = selected_config["measurement_amendment"]
    else:
        raise ValueError("corrected v2 replay protocol is required")
    source = Path(__file__).resolve()
    if sha256(source) != config["implementation_source_sha256"]:
        raise RuntimeError("implementation source differs from frozen protocol")
    for dependency in config["dependency_sources"]:
        path = Path(dependency["path"])
        if sha256(path) != dependency["sha256"]:
            raise RuntimeError(f"dependency source differs: {path}")
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    atomic_json(output / "STATUS.json", {"stage": "controls"})
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    batch_rows, stage_rows, state_seed_rows, state_stage_rows = [], [], [], []
    control_results, arm_receipts = {}, {}
    total_read_bytes = 0
    for arm, spec in config["arms"].items():
        ops, observed_settings = validate_arm(spec)
        field = load_field(spec["field"], spec["num_samples"], float(ops["fs"]))
        geometry = np.load(Path(spec["sorter_output"]) / "channel_positions.npy")
        if sha256(Path(spec["sorter_output"]) / "channel_positions.npy") != spec["geometry_sha256"]:
            raise RuntimeError("geometry hash differs")
        states = field["states"]
        state_support = channel_support(geometry, states)
        support_index = {float(state): index for index, state in enumerate(states)}
        if len(spec["selected_batches"]) > int(config["limits"]["max_batches_per_arm"]):
            raise RuntimeError(f"{arm}: too many frozen batches")
        planned_ranges = []
        for selection in spec["selected_batches"]:
            ibatch = int(selection["batch_index"])
            core_start = ibatch * int(ops["batch_size"])
            core_stop = core_start + int(ops["batch_size"])
            padded_start = core_start - int(ops["nt"])
            padded_stop = core_stop + int(ops["nt"])
            if padded_start < 0 or padded_stop > int(spec["num_samples"]):
                raise RuntimeError(f"{arm}: selection outside recording bounds")
            core_states_pre = assigned_states(
                np.arange(core_start, core_stop, dtype=np.int64), field, float(ops["fs"])
            )
            padded_states_pre = assigned_states(
                np.arange(padded_start, padded_stop, dtype=np.int64), field, float(ops["fs"])
            )
            unique_core = sorted(np.unique(core_states_pre).tolist())
            intended = float(selection["selection_state_um"])
            if selection["selection_kind"] == "stable":
                if unique_core != [intended] or sorted(np.unique(padded_states_pre).tolist()) != [intended]:
                    raise RuntimeError(f"frozen stable selection is not padded-stable: {arm} {ibatch}")
            elif selection["selection_kind"] == "transition":
                if len(unique_core) < 2 or intended not in unique_core:
                    raise RuntimeError(f"frozen transition selection is invalid: {arm} {ibatch}")
            else:
                raise ValueError("unknown selection kind")
            if unique_core != selection["expected_core_states_um"]:
                raise RuntimeError(f"core state set differs from protocol: {arm} {ibatch}")
            planned_ranges.append((padded_start, padded_stop))
        if any(b0 < a1 and a0 < b1 for i, (a0, a1) in enumerate(planned_ranges)
               for b0, b1 in planned_ranges[i + 1:]):
            raise RuntimeError(f"{arm}: frozen padded reads overlap")
        planned_read_bytes = sum(
            2 * (stop - start) * int(ops["n_chan_bin"]) * np.dtype(ops["data_dtype"]).itemsize
            for start, stop in planned_ranges
        )
        read_limit = int(config["limits"]["max_recording_seconds_per_arm"] * float(ops["fs"]) *
                         int(ops["n_chan_bin"]) * np.dtype(ops["data_dtype"]).itemsize)
        if planned_read_bytes > read_limit:
            raise RuntimeError(f"{arm}: frozen plan exceeds read limit before access")
        control_results[arm] = controls(ops)
        binary = Path(spec["binary_path"])
        raw_reader = BinaryRWFile(
            str(binary), int(ops["n_chan_bin"]), fs=float(ops["fs"]),
            NT=int(ops["batch_size"]), nt=int(ops["nt"]), nt0min=int(ops["nt0min"]),
            device=torch.device("cpu"), dtype=ops["data_dtype"],
        )
        exact_reader = BinaryFiltered(
            str(binary), int(ops["n_chan_bin"]), fs=float(ops["fs"]),
            NT=int(ops["batch_size"]), nt=int(ops["nt"]), nt0min=int(ops["nt0min"]),
            chan_map=np.asarray(ops["chanMap"]),
            hp_filter=torch.as_tensor(np.asarray(ops["preprocessing"]["hp_filter"])),
            whiten_mat=torch.as_tensor(np.asarray(ops["preprocessing"]["whiten_mat"])),
            dshift=None, device=torch.device("cpu"), do_CAR=bool(ops["do_CAR"]),
            invert_sign=bool(ops["invert_sign"]), dtype=ops["data_dtype"],
            artifact_threshold=float(ops["artifact_threshold"]),
            file_object=raw_reader.file,
        )
        arm_read_bytes = 0
        for selection in spec["selected_batches"]:
            ibatch = int(selection["batch_index"])
            atomic_json(output / "STATUS.json", {"stage": "replay", "arm": arm, "batch_index": ibatch})
            bstart, bend = raw_reader.get_batch_edges(ibatch)
            read_bytes = int((bend - bstart) * int(ops["n_chan_bin"]) * np.dtype(ops["data_dtype"]).itemsize)
            arm_read_bytes += 2 * read_bytes
            total_read_bytes += 2 * read_bytes
            raw, raw_inds = raw_reader.padded_batch_to_torch(ibatch, return_inds=True)
            exact, exact_inds = exact_reader.padded_batch_to_torch(ibatch, ops, return_inds=True)
            if raw_inds != exact_inds:
                raise RuntimeError("independent readers returned different bounds")
            expected_inds = [
                ibatch * int(ops["batch_size"]) - int(ops["nt"]),
                ibatch * int(ops["batch_size"]) + int(ops["batch_size"]) + int(ops["nt"]),
            ]
            if raw_inds != expected_inds or raw.shape[1] != expected_inds[1] - expected_inds[0]:
                raise RuntimeError(f"padded origin/bounds differ: {arm} {ibatch} {raw_inds}")
            stages = manual_stages(raw, ops)
            difference = (stages["post_whitening"] - exact).abs()
            max_difference = float(difference.max())
            if max_difference != 0.0:
                raise RuntimeError(f"stage reconstruction differs: {arm} {ibatch} {max_difference}")
            core_start_column = ibatch * int(ops["batch_size"]) - int(raw_inds[0])
            core_stop_column = core_start_column + int(ops["batch_size"])
            core = slice(core_start_column, core_stop_column)
            if core_start_column != int(ops["nt"]) or core_stop_column > raw.shape[1] or \
                    raw[:, core].shape[1] != int(ops["batch_size"]):
                raise RuntimeError(f"core origin/length differs: {arm} {ibatch}")
            core_samples = np.arange(ibatch * int(ops["batch_size"]),
                                     ibatch * int(ops["batch_size"]) + int(ops["batch_size"]),
                                     dtype=np.int64)
            core_states = assigned_states(core_samples, field, float(ops["fs"]))
            core_switch_bins = switch_distance_bins(core_samples, field, float(ops["fs"]))
            expected_supported = np.stack([
                state_support[support_index[float(state)]] for state in core_states
            ], axis=1)
            supported_mask = torch.as_tensor(expected_supported, dtype=torch.bool)
            unsupported_mask = ~supported_mask
            core_input = stages["input"][:, core]
            expected_zero_values = int(unsupported_mask.sum())
            expected_zero_nonzero = int(torch.count_nonzero(core_input[unsupported_mask])) if expected_zero_values else 0
            if expected_zero_nonzero:
                raise RuntimeError(
                    f"materialized unsupported targets are nonzero: {arm} {ibatch} {expected_zero_nonzero}"
                )
            all_zero_rows = torch.all(core_input == 0, dim=1).cpu().numpy()
            stable = len(np.unique(core_states)) == 1
            stable_state = float(core_states[0]) if stable else None
            expected_unsupported_channels = (
                int((~state_support[support_index[stable_state]]).sum()) if stable else None
            )
            peaks = isolated_seed_locations(
                exact, float(ops["settings"]["Th_single_ch"]), int(ops["nt"])
            )
            padded_samples = np.arange(int(raw_inds[0]), int(raw_inds[0]) + exact.shape[1], dtype=np.int64)
            padded_states = assigned_states(padded_samples, field, float(ops["fs"]))
            peak_unsupported = 0
            if len(peaks):
                pchan = peaks[:, 0].cpu().numpy(); ptime = peaks[:, 1].cpu().numpy()
                peak_unsupported = int(sum(
                    not state_support[support_index[float(padded_states[t])], c]
                    for c, t in zip(pchan, ptime)
                ))
            else:
                pchan = np.empty(0, dtype=np.int64); ptime = np.empty(0, dtype=np.int64)
            peak_samples = padded_samples[ptime]
            peak_states = assigned_states(peak_samples, field, float(ops["fs"]))
            peak_switch_bins = switch_distance_bins(peak_samples, field, float(ops["fs"]))
            temporal_names = ("lt0p25s", "0p25_to_lt1s", "1_to_lt5s", "ge5s")
            for state in sorted(np.unique(core_states)):
                for switch_bin, switch_name in enumerate(temporal_names):
                    sample_mask = (core_states == state) & (core_switch_bins == switch_bin)
                    peak_mask = (peak_states == state) & (peak_switch_bins == switch_bin)
                    if not sample_mask.any() and not peak_mask.any():
                        continue
                    unsupported_peaks = 0
                    if peak_mask.any():
                        unsupported_peaks = int(sum(
                            not state_support[support_index[float(state)], channel]
                            for channel in pchan[peak_mask]
                        ))
                    state_seed_rows.append({
                        "arm": arm, "batch_index": ibatch,
                        "selection_kind": selection["selection_kind"],
                        "state_um": float(state), "temporal_state_change_bin_s": switch_name,
                        "exposure_samples": int(sample_mask.sum()),
                        "exposure_seconds": float(sample_mask.sum() / float(ops["fs"])),
                        "supported_channel_sample_values": int(
                            sample_mask.sum() * state_support[support_index[float(state)]].sum()
                        ),
                        "unsupported_channel_sample_values": int(
                            sample_mask.sum() * (~state_support[support_index[float(state)]]).sum()
                        ),
                        "isolated_seed_peaks": int(peak_mask.sum()),
                        "isolated_seed_peaks_on_raw_unsupported_targets": unsupported_peaks,
                    })
                    for stage_name, values in stages.items():
                        state_values = values[:, core][:, sample_mask]
                        channel_supported = torch.as_tensor(
                            state_support[support_index[float(state)]], dtype=torch.bool
                        )
                        for region, region_values in (
                            ("all", state_values),
                            ("supported", state_values[channel_supported]),
                            ("unsupported", state_values[~channel_supported]),
                        ):
                            state_stage_rows.append({
                                "arm": arm, "batch_index": ibatch,
                                "selection_kind": selection["selection_kind"],
                                "state_um": float(state),
                                "temporal_state_change_bin_s": switch_name,
                                "stage": stage_name, "region": region,
                                **basic_summary(region_values),
                            })
            threshold_sites = int(torch.count_nonzero(
                exact[:, core].abs() > float(ops["settings"]["Th_single_ch"])
            ))
            batch_rows.append({
                "arm": arm, "batch_index": ibatch, "selection_state_um": selection["selection_state_um"],
                "selection_kind": selection["selection_kind"],
                "core_state_values_um": json.dumps(sorted(np.unique(core_states).tolist())),
                "core_stable": stable, "original_nskip25_schedule": ibatch % int(ops["settings"]["nskip"]) == 0,
                "original_seed_contribution_confirmed": False,
                "core_start_sample": int(core_samples[0]), "core_end_sample_exclusive": int(core_samples[-1] + 1),
                "read_start_sample": int(raw_inds[0]), "read_end_sample_exclusive": int(raw_inds[1]),
                "read_bytes_two_passes": 2 * read_bytes,
                "expected_unsupported_channels_if_stable": expected_unsupported_channels,
                "observed_all_zero_input_rows": int(all_zero_rows.sum()),
                "supported_all_zero_input_rows": int(np.sum(all_zero_rows & expected_supported.all(axis=1))),
                "expected_unsupported_input_values": expected_zero_values,
                "expected_unsupported_nonzero_input_values": expected_zero_nonzero,
                "isolated_seed_peaks": int(len(peaks)),
                "isolated_seed_peaks_on_raw_unsupported_targets": peak_unsupported,
                "core_abs_gt_Th_single_ch_sites": threshold_sites,
                "manual_vs_binaryfiltered_max_abs_difference": max_difference,
            })
            for stage_name, values in stages.items():
                core_values = values[:, core]
                for region, mask in (("all", None), ("supported", supported_mask),
                                     ("unsupported", unsupported_mask)):
                    summary = stage_summary(core_values, mask)
                    stage_rows.append({"arm": arm, "batch_index": ibatch,
                                       "selection_kind": selection["selection_kind"],
                                       "stage": stage_name, "region": region, **summary})
        raw_reader.close(); exact_reader.close()
        limit = int(config["limits"]["max_recording_seconds_per_arm"] * float(ops["fs"]) *
                    int(ops["n_chan_bin"]) * np.dtype(ops["data_dtype"]).itemsize)
        if arm_read_bytes > limit:
            raise RuntimeError(f"{arm} read budget exceeded")
        arm_receipts[arm] = {
            "observed_settings": observed_settings, "recording_bytes_read": arm_read_bytes,
            "equivalent_recording_seconds_read": arm_read_bytes /
                (float(ops["fs"]) * int(ops["n_chan_bin"]) * np.dtype(ops["data_dtype"]).itemsize),
            "planned_recording_bytes": planned_read_bytes, "read_limit_bytes": limit,
        }
    write_csv(output / "BATCH_METRICS.csv", batch_rows)
    write_csv(output / "STAGE_METRICS.csv", stage_rows)
    write_csv(output / "STATE_SEED_METRICS.csv", state_seed_rows)
    write_csv(output / "STATE_STAGE_METRICS.csv", state_stage_rows)
    result = {
        "schema": "en-materialized-seed-replay-result-v1", "status": "complete",
        "protocol": {"path": str(config_path.resolve()), "sha256": sha256(config_path)},
        "implementation": {"path": str(source), "sha256": sha256(source)},
        "controls": control_results, "arms": arm_receipts,
        "batches": batch_rows,
        "resources": {
            "wall_seconds": time.perf_counter() - started,
            "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "recording_bytes_read": total_read_bytes, "sorts_launched": 0,
            "templates_fit": 0, "gpu_used": False, "threads": 2,
            "rf_accesses": 0, "holdout_accesses": 0,
        },
        "interpretation_limits": [
            "Th_single_ch=6 is the basis-learning isolated-seed consumer, not final learned-template detection at Th_learned=9.",
            "A null isolated-seed result cannot rule out later template-learning or matching effects.",
            "Schedule eligibility does not prove a batch contributed original seeds because extraction stops after its clip buffer fills.",
            "This replay measures one executed preprocessing pathway and is not a purity or production test.",
        ],
    }
    atomic_json(output / "RESULT.json", result)
    products = []
    for path in sorted(output.iterdir()):
        if path.name in {"COMPLETE.json", "STATUS.json"} or not path.is_file():
            continue
        products.append({"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)})
    atomic_json(output / "MANIFEST.json", {"schema": "en-materialized-seed-replay-manifest-v1",
                                           "products": products})
    atomic_json(output / "STATUS.json", {"stage": "complete", "status": "complete"})
    atomic_json(output / "COMPLETE.json", {"schema": "en-materialized-seed-replay-complete-v1",
                                           "status": "complete",
                                           "manifest_sha256": sha256(output / "MANIFEST.json")})
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.config, args.output)


if __name__ == "__main__":
    main()
