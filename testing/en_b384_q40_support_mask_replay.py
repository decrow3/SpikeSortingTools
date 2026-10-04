#!/usr/bin/env python3
"""Review-gated paired baseline/candidate B384 matcher replay.

The execution path is intentionally unavailable until its config records human
scope approval and source review. Preflight performs no recording reads.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kilosort.io import BinaryRWFile, bfile_from_ops, load_ops
from kilosort.template_matching import prepare_matching, run_matching
from npx_preprocessing.motion.prewhitening_support_mask import (
    filter_then_mask_before_whitening,
    si_stepwise_states,
)
from testing.en_rounded_field import rounded_rigid_field


SCHEMA = "en-b384-q40-support-mask-replay-v4"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def manifest_owned_binary_path(manifest_path: Path, manifest: dict) -> Path:
    """Resolve the sole binary exactly within its owning manifest directory."""
    files = manifest.get("recording_binary_files", [])
    if len(files) != 1 or not isinstance(files[0].get("name"), str):
        raise RuntimeError("manifest does not declare exactly one named binary")
    recorded_name = files[0]["name"]
    if Path(recorded_name).name != recorded_name:
        raise RuntimeError("manifest binary name must be a basename")
    return (manifest_path.resolve(strict=True).parent / recorded_name).resolve(strict=True)


def verify(config: dict, *, execution: bool) -> None:
    if config.get("schema") != SCHEMA or config.get("device") != "cpu":
        raise RuntimeError("schema or device differs")
    if execution and not (
        config.get("human_scope_approved", False)
        and config.get("source_reviewed", False)
        and config.get("allow_recording_read_and_matcher_replay", False)
    ):
        raise RuntimeError("paired recording replay remains review-gated")
    for item in config["frozen_inputs"]:
        path = Path(item["path"])
        observed = sha256(path)
        if observed != item["sha256"]:
            raise RuntimeError(f"frozen input differs: {path}: {observed}")
    recording = Path(config["recording_path"])
    resolved_recording = recording.resolve(strict=True)
    if str(resolved_recording) != config["recording_path_resolved"]:
        raise RuntimeError("resolved recording path differs")
    if recording.stat().st_size != int(config["recording_bytes"]):
        raise RuntimeError("recording byte size differs")
    manifest_path = Path(config["recording_manifest_path"])
    if sha256(manifest_path) != config["recording_manifest_sha256"]:
        raise RuntimeError("recording manifest digest differs")
    manifest = json.loads(manifest_path.read_text())
    files = manifest.get("recording_binary_files", [])
    if resolved_recording != manifest_owned_binary_path(manifest_path, manifest):
        raise RuntimeError("recording path is not the binary owned by the accepted manifest")
    if (
        len(files) != 1
        or files[0].get("name") != resolved_recording.name
        or int(files[0].get("size_bytes", -1)) != recording.stat().st_size
        or files[0].get("sha256") != config["accepted_recording_sha256"]
    ):
        raise RuntimeError("accepted recording receipt differs")
    if manifest.get("recording_content_sha256") != config["recording_content_sha256"]:
        raise RuntimeError("recording content digest differs")
    if float(manifest.get("sampling_frequency_hz", -1)) != float(config["sampling_frequency_hz"]):
        raise RuntimeError("recording manifest sampling rate differs")
    if int(manifest.get("num_channels", -1)) != 384 or manifest.get("dtype") != "int16":
        raise RuntimeError("recording manifest channel count or dtype differs")
    complete_path = Path(config["materialization_complete_path"])
    if sha256(complete_path) != config["materialization_complete_sha256"]:
        raise RuntimeError("materialization COMPLETE digest differs")
    complete_recording = json.loads(complete_path.read_text())["recording"]
    if complete_recording != manifest:
        raise RuntimeError("materialization COMPLETE recording receipt differs from manifest")


def verified_ops(config: dict) -> dict:
    ops = load_ops(config["ops_path"], device=torch.device("cpu"))
    chan_map = np.asarray(
        ops["probe"]["chanMap"].cpu() if hasattr(ops["probe"]["chanMap"], "cpu") else ops["probe"]["chanMap"],
        dtype=np.int64,
    )
    if int(ops["Nchan"]) != 384 or int(ops["n_chan_bin"]) != 384 or not np.array_equal(chan_map, np.arange(384)):
        raise RuntimeError("candidate is limited to identity-mapped full B384 geometry")
    if len(ops["xc"]) != 384 or len(ops["yc"]) != 384 or ops["dshift"] is not None:
        raise RuntimeError("saved geometry or dshift is incompatible with bounded candidate")
    if int(ops["nt"]) != 61 or int(ops["batch_size"]) != 60000 or float(ops["Th_learned"]) != 9.0:
        raise RuntimeError("saved matcher settings differ")
    observed = {
        "Nchan": int(ops["Nchan"]), "n_chan_bin": int(ops["n_chan_bin"]),
        "batch_size": int(ops["batch_size"]), "nt": int(ops["nt"]), "nt0min": int(ops["nt0min"]),
        "data_dtype": str(ops["data_dtype"]), "Th_learned": float(ops["Th_learned"]),
        "do_CAR": bool(ops["do_CAR"]), "fs": float(ops["fs"]),
        "imin": int(np.int64(ops["tmin"] * ops["fs"])),
        "imax": int(np.int64(ops["tmax"] * ops["fs"])),
    }
    if observed != config["expected_ops"]:
        raise RuntimeError(f"measured ops differ: {observed}")
    manifest = json.loads(Path(config["recording_manifest_path"]).read_text())
    original_geometry = np.asarray(manifest["channel_locations_um"], dtype=np.float64)[:, :2]
    ops_geometry = np.column_stack((np.asarray(ops["xc"], dtype=float), np.asarray(ops["yc"], dtype=float)))
    if not np.array_equal(ops_geometry, original_geometry):
        raise RuntimeError("B384 ops geometry differs from full original materialization geometry")
    return ops


def validate_endpoint_coverage(
    config: dict, ops: dict, batches: pd.DataFrame, endpoints: pd.DataFrame, bank_templates: int
) -> dict:
    """Fail closed unless every frozen endpoint belongs to its selected batch."""
    expected_count = int(config["expected_selected_endpoints"])
    if len(endpoints) != expected_count:
        raise RuntimeError("selected endpoint count differs")
    selected = set(batches.batch_index.astype(int))
    endpoint_batches = set(endpoints.batch_index.astype(int))
    if not endpoint_batches.issubset(selected):
        raise RuntimeError("endpoint references an unselected batch")
    if config.get("require_every_selected_batch_has_endpoints", False) and endpoint_batches != selected:
        raise RuntimeError("endpoint batch coverage differs from selected batches")
    by_batch = {int(row.batch_index): row for row in batches.itertuples(index=False)}
    imin = int(np.int64(ops["tmin"] * ops["fs"]))
    imax = int(np.int64(ops["tmax"] * ops["fs"]))
    for endpoint in endpoints.itertuples(index=False):
        batch = by_batch[int(endpoint.batch_index)]
        event = int(endpoint.event_global)
        if not (
            imin <= event < imax
            and int(batch.batch_core_start_global) <= event < int(batch.batch_core_stop_global_exclusive)
        ):
            raise RuntimeError("endpoint global time is outside its selected batch/core recording support")
        if not 0 <= int(endpoint.template_id) < int(bank_templates):
            raise RuntimeError("endpoint template ID is outside the fixed bank")
    return {
        "endpoint_rows_validated": len(endpoints),
        "endpoint_batch_indices": sorted(endpoint_batches),
        "every_selected_batch_has_endpoints": endpoint_batches == selected,
        "template_id_half_open_bound": [0, int(bank_templates)],
    }


def batch_state_classifications(
    batches: pd.DataFrame,
    temporal_centers_s: np.ndarray,
    shifts_um: np.ndarray,
    sampling_frequency_hz: float,
    expected: dict[str, str],
) -> pd.DataFrame:
    """Bind each frozen padded batch to stable/transition field metadata."""
    rows = []
    for batch in batches.itertuples(index=False):
        frames = np.arange(
            int(batch.padded_read_start_global),
            int(batch.padded_read_stop_global_exclusive),
            dtype=np.int64,
        )
        states = si_stepwise_states(
            temporal_centers_s, shifts_um, frames, sampling_frequency_hz
        )
        changes = int(np.count_nonzero(np.diff(states)))
        classification = "stable" if changes == 0 else "transition"
        ibatch = int(batch.batch_index)
        if expected.get(str(ibatch)) != classification:
            raise RuntimeError(f"batch stable/transition classification differs: {ibatch}")
        rows.append({
            "batch_index": ibatch,
            "batch_classification": classification,
            "rounded_states_um": ";".join(map(str, np.unique(states).tolist())),
            "state_changes_in_padded_batch": changes,
        })
    return pd.DataFrame(rows)


def endpoint_neighbor_report(
    event_global: int,
    template_id: int,
    emission_times: np.ndarray,
    emission_templates: np.ndarray,
    amplitudes: np.ndarray,
    threshold_scores: np.ndarray,
) -> tuple[dict, list[dict]]:
    """Return all, nonexclusive matcher neighbors within the frozen nt//2 window."""
    times = np.asarray(emission_times, dtype=np.int64)
    templates = np.asarray(emission_templates, dtype=np.int64)
    amps = np.asarray(amplitudes, dtype=np.float64).reshape(-1)
    scores = np.asarray(threshold_scores, dtype=np.float64).reshape(-1)
    if not (len(times) == len(templates) == len(amps) == len(scores)):
        raise ValueError("matcher emission arrays differ in length")
    offsets = times - int(event_global)
    same_template = templates == int(template_id)
    within_1 = np.abs(offsets) <= 1
    within_30 = np.abs(offsets) <= 30
    exact = (offsets == 0) & same_template
    summary = {
        "exact_same_time_template_count": int(np.count_nonzero(exact)),
        "exact_same_time_template_retained": bool(np.any(exact)),
        "same_template_within_1_count": int(np.count_nonzero(same_template & within_1)),
        "same_template_within_1_retained": bool(np.any(same_template & within_1)),
        "any_template_within_1_count": int(np.count_nonzero(within_1)),
        "any_template_within_30_count": int(np.count_nonzero(within_30)),
    }
    neighbors = []
    for index in np.flatnonzero(within_30):
        neighbors.append({
            "neighbor_global_sample": int(times[index]),
            "neighbor_template_id": int(templates[index]),
            "sample_offset": int(offsets[index]),
            "amplitude": float(amps[index]),
            "threshold_score": float(scores[index]),
            "same_template": bool(same_template[index]),
            "within_1_sample": bool(within_1[index]),
            "exact_same_time_template": bool(exact[index]),
        })
    return summary, neighbors


def validate_static_plan(config: dict, ops: dict, batches: pd.DataFrame, endpoints: pd.DataFrame) -> dict:
    derived_per_batch = (
        (int(ops["batch_size"]) + 2 * int(ops["nt"]))
        * int(ops["n_chan_bin"]) * np.dtype(ops["data_dtype"]).itemsize
    )
    expected_batches = [int(value) for value in config["expected_batch_indices"]]
    measured_batches = batches.batch_index.astype(int).tolist()
    if measured_batches != expected_batches or len(batches) != int(config["raw_read_calls_max"]):
        raise RuntimeError("selected batch identity/count differs")
    if not np.all(batches.logical_recording_bytes.astype(np.int64) == derived_per_batch):
        raise RuntimeError("selected batch byte receipts differ from derived ops/dtype bytes")
    derived_total = int(derived_per_batch * len(batches))
    if derived_total != int(config["expected_logical_read_bytes"]):
        raise RuntimeError("derived total read bytes differ")
    if derived_total > int(config["logical_read_bytes_max"]):
        raise RuntimeError("planned reads exceed cap")
    bank = np.load(config["bank_path"], mmap_mode="r", allow_pickle=False)
    if list(bank.shape) != config["expected_bank_shape"]:
        raise RuntimeError("fixed bank shape differs")
    endpoint_validation = validate_endpoint_coverage(config, ops, batches, endpoints, bank.shape[0])
    with np.load(config["field_path"], allow_pickle=False) as field:
        centers = np.asarray(field[config["field_time_key"]], dtype=np.float64)
        displacement = np.asarray(field[config["field_displacement_key"]], dtype=np.float64)
    rounded, reference_um = rounded_rigid_field(displacement, 40.0)
    states = np.unique(rounded).tolist()
    if states != config["expected_rounded_states_um"]:
        raise RuntimeError("frozen field rounded states differ")
    dt = float(np.median(np.diff(centers)))
    if not np.allclose(np.diff(centers), dt, rtol=0, atol=1e-6):
        raise RuntimeError("frozen field time grid is not uniform")
    batch_states = batch_state_classifications(
        batches, centers, rounded, float(ops["fs"]),
        config["expected_batch_classification"],
    )
    return {
        "batch_indices": measured_batches,
        "derived_logical_bytes_per_batch": int(derived_per_batch),
        "derived_total_logical_bytes": derived_total,
        "selected_endpoints": len(endpoints),
        "endpoint_validation": endpoint_validation,
        "bank_shape": list(bank.shape),
        "rounded_states_um": states,
        "field_reference_um": reference_um,
        "field_cell_width_s": dt,
        "batch_state_classification": batch_states.to_dict(orient="records"),
        "recording_bytes": Path(config["recording_path"]).stat().st_size,
        "ops": config["expected_ops"],
    }


def matcher_global(center: np.ndarray, ibatch: int, imin: int, batch_size: int, nt: int, nt0min: int) -> np.ndarray:
    return ((center - nt) + ibatch * batch_size - nt // 2 + nt0min + imin).astype(np.int64)


def endpoint_table(config: dict) -> pd.DataFrame:
    pairs = pd.read_csv(config["selected_pairs_path"])
    singles = pd.read_csv(config["isolated_controls_path"])
    rows = []
    for pair in pairs.itertuples(index=False):
        common = {"unit_id": int(pair.unit_id), "selection_role": pair.selection_role,
                  "batch_index": int(pair.batch_index), "sample_rank": int(pair.sample_rank), "kind": "pair"}
        rows.append({**common, "endpoint": "first", "event_global": int(pair.first_time_global),
                     "template_id": int(pair.first_template)})
        rows.append({**common, "endpoint": "second", "event_global": int(pair.second_time_global),
                     "template_id": int(pair.second_template)})
    for single in singles.itertuples(index=False):
        rows.append({"unit_id": int(single.unit_id), "selection_role": single.selection_role,
                     "batch_index": int(single.batch_index), "sample_rank": 0, "kind": "single",
                     "endpoint": "control", "event_global": int(single.time_global),
                     "template_id": int(single.detection_template)})
    return pd.DataFrame(rows)


def preflight(config: dict, config_path: Path) -> None:
    verify(config, execution=False)
    output = Path(config["preflight_output"])
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    batches = pd.read_csv(config["selected_batches_path"])
    endpoints = endpoint_table(config)
    ops = verified_ops(config)
    measured = validate_static_plan(config, ops, batches, endpoints)
    expected_bytes = measured["derived_total_logical_bytes"]
    plan = {
        "schema": SCHEMA, "status": "ready_for_review", "created_at": datetime.now(timezone.utc).isoformat(),
        "candidate": "zero geometry/field-derived unsupported entries after native CAR/high-pass and before unchanged Wrot",
        "production_default_changed": False,
        "config_path": str(config_path.resolve(strict=True)), "config_sha256": sha256(config_path),
        "recording_reads": 0, "matcher_replays": 0,
        "planned_batches": len(batches), "planned_endpoints": len(endpoints),
        "planned_logical_recording_bytes": expected_bytes,
        "read_limit_bytes": int(config["logical_read_bytes_max"]),
        "cpu_threads_max": int(config["threads_max"]), "memory_max_bytes": int(config["memory_max_bytes"]),
        "wall_seconds_max": float(config["wall_seconds_max"]),
        "measured_validation": measured,
        "matching_rule": (
            "Exact: identical saved global time and template ID, count must be one in baseline. Tolerant: same "
            "template ID within +/-1 global sample, reported as an all-neighbor descriptive count."
        ),
        "interpretation": (
            "Paired fixed-bank causal replay only. It does not validate sorting quality, learning-stage implications, "
            "transition repair, biological identity, purity, or production readiness."
        ),
        "permanent_support_policy": config["permanent_support_policy"],
        "covariance_rank_policy": config["covariance_rank_policy"],
        "binary_replacement_risk": config["binary_replacement_risk"],
    }
    atomic_json(output / "PREFLIGHT.json", plan)
    products = ["PREFLIGHT.json"]
    manifest = {name: {"sha256": sha256(output / name), "bytes": (output / name).stat().st_size} for name in products}
    atomic_json(output / "MANIFEST.json", manifest)
    atomic_json(output / "COMPLETE.json", {"schema": SCHEMA, "status": "ready_for_review",
        "manifest_sha256": sha256(output / "MANIFEST.json")})


def execute(config: dict) -> None:
    verify(config, execution=True)
    output = Path(config["output"])
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    endpoints = endpoint_table(config)
    batches = pd.read_csv(config["selected_batches_path"])
    ops = verified_ops(config)
    validate_static_plan(config, ops, batches, endpoints)
    torch.set_num_threads(int(config["threads_max"]))
    torch.manual_seed(int(config["seed"]))
    bank = torch.from_numpy(np.load(config["bank_path"], allow_pickle=False)).to(torch.float32)
    with np.load(config["field_path"], allow_pickle=False) as field:
        temporal_centers_s = np.asarray(field[config["field_time_key"]], dtype=np.float64)
        shifts_um, _ = rounded_rigid_field(field[config["field_displacement_key"]], 40.0)
    geometry = np.column_stack((np.asarray(ops["xc"], dtype=float), np.asarray(ops["yc"], dtype=float)))
    ctc = prepare_matching(ops, bank)
    rows, neighbor_rows, emission_rows = [], [], []
    batch_rows, template_count_rows, changed_template_rows = [], [], []
    bytes_read = 0
    started = time.perf_counter()
    with bfile_from_ops(ops=ops, filename=config["recording_path"], device=torch.device("cpu")) as bfile:
        batch_state_frame = batch_state_classifications(
            batches, temporal_centers_s, shifts_um, float(ops["fs"]),
            config["expected_batch_classification"],
        )
        batch_state_lookup = batch_state_frame.set_index("batch_index").to_dict(orient="index")
        for batch in batches.itertuples(index=False):
            ibatch = int(batch.batch_index)
            batch_state = batch_state_lookup[ibatch]
            if len(batch_rows) >= int(config["raw_read_calls_max"]):
                raise RuntimeError("next read would exceed raw-read call limit")
            if bytes_read + int(batch.logical_recording_bytes) > int(config["logical_read_bytes_max"]):
                raise RuntimeError("next read would exceed logical byte limit")
            if time.perf_counter() - started > float(config["wall_seconds_max"]):
                raise RuntimeError("wall limit reached before next read")
            raw, inds = BinaryRWFile.padded_batch_to_torch(bfile, ibatch, return_inds=True)
            if [int(inds[0]), int(inds[1])] != [int(batch.padded_read_start_global), int(batch.padded_read_stop_global_exclusive)]:
                raise RuntimeError("padded batch/global axis differs")
            bytes_read += int(batch.logical_recording_bytes)
            paired = filter_then_mask_before_whitening(
                bfile, raw, ops=ops, ibatch=ibatch, global_start_frame=int(inds[0]),
                channel_locations_um=geometry, temporal_centers_s=temporal_centers_s,
                shifts_um=shifts_um,
            )
            native_baseline = bfile.filter(raw.clone(), ops=ops, ibatch=ibatch)
            baseline_binding_exact = torch.equal(native_baseline, paired.baseline_whitened)
            if not baseline_binding_exact:
                raise RuntimeError("candidate wrapper baseline differs from native bfile.filter")
            baseline_st, baseline_amplitudes, baseline_threshold, _ = run_matching(
                ops, paired.baseline_whitened, bank, ctc, device=torch.device("cpu")
            )
            baseline_global = matcher_global(baseline_st[:, 0].cpu().numpy(), ibatch, int(bfile.imin),
                                             int(ops["batch_size"]), int(ops["nt"]), int(ops["nt0min"]))
            baseline_templates = baseline_st[:, 1].cpu().numpy().astype(np.int64)
            batch_endpoints = endpoints[endpoints.batch_index.eq(ibatch)]
            for endpoint in batch_endpoints.itertuples(index=False):
                count = int(np.count_nonzero((baseline_global == endpoint.event_global) &
                                             (baseline_templates == endpoint.template_id)))
                if count != 1:
                    raise RuntimeError(
                        f"baseline selected endpoint reproduction failed: batch={ibatch} "
                        f"time={endpoint.event_global} template={endpoint.template_id} count={count}"
                    )
            candidate_st, candidate_amplitudes, candidate_threshold, _ = run_matching(
                ops, paired.candidate_whitened, bank, ctc, device=torch.device("cpu")
            )
            candidate_global = matcher_global(candidate_st[:, 0].cpu().numpy(), ibatch, int(bfile.imin),
                                              int(ops["batch_size"]), int(ops["nt"]), int(ops["nt0min"]))
            candidate_templates = candidate_st[:, 1].cpu().numpy().astype(np.int64)
            baseline_amplitudes_np = baseline_amplitudes.cpu().numpy().reshape(-1)
            baseline_threshold_np = baseline_threshold.cpu().numpy().reshape(-1)
            candidate_amplitudes_np = candidate_amplitudes.cpu().numpy().reshape(-1)
            candidate_threshold_np = candidate_threshold.cpu().numpy().reshape(-1)
            for arm, global_times, template_ids, amplitudes, scores in (
                ("baseline", baseline_global, baseline_templates, baseline_amplitudes_np, baseline_threshold_np),
                ("candidate", candidate_global, candidate_templates, candidate_amplitudes_np, candidate_threshold_np),
            ):
                for global_sample, template_id, amplitude, score in zip(
                    global_times, template_ids, amplitudes, scores, strict=True
                ):
                    emission_rows.append({
                        "arm": arm, "batch_index": ibatch,
                        "batch_classification": batch_state["batch_classification"],
                        "global_sample": int(global_sample), "template_id": int(template_id),
                        "amplitude": float(amplitude), "threshold_score": float(score),
                    })
            template_energy = torch.sum(bank**2, dim=1).cpu().numpy()
            peak_rows = np.argmax(template_energy, axis=1)
            for template_id in range(bank.shape[0]):
                baseline_count = int(np.count_nonzero(baseline_templates == template_id))
                candidate_count = int(np.count_nonzero(candidate_templates == template_id))
                count_row = {"batch_index": ibatch, "template_id": int(template_id),
                    "baseline_count": baseline_count, "candidate_count": candidate_count,
                    "count_change": candidate_count - baseline_count,
                    "template_peak_row": int(peak_rows[int(template_id)]),
                    "template_edge_distance_rows": int(min(peak_rows[int(template_id)], 383 - peak_rows[int(template_id)]))}
                template_count_rows.append(count_row)
                if baseline_count != candidate_count:
                    changed_template_rows.append(count_row)
            for endpoint in batch_endpoints.itertuples(index=False):
                baseline_match = np.flatnonzero((baseline_global == endpoint.event_global) &
                                                (baseline_templates == endpoint.template_id))
                candidate_match = np.flatnonzero((candidate_global == endpoint.event_global) &
                                                 (candidate_templates == endpoint.template_id))
                baseline_tolerant = np.flatnonzero((np.abs(baseline_global - endpoint.event_global) <= 1) &
                                                   (baseline_templates == endpoint.template_id))
                candidate_tolerant = np.flatnonzero((np.abs(candidate_global - endpoint.event_global) <= 1) &
                                                    (candidate_templates == endpoint.template_id))
                baseline_summary, baseline_neighbors = endpoint_neighbor_report(
                    endpoint.event_global, endpoint.template_id, baseline_global,
                    baseline_templates, baseline_amplitudes_np, baseline_threshold_np,
                )
                candidate_summary, candidate_neighbors = endpoint_neighbor_report(
                    endpoint.event_global, endpoint.template_id, candidate_global,
                    candidate_templates, candidate_amplitudes_np, candidate_threshold_np,
                )
                endpoint_identity = endpoint._asdict()
                for arm, local_neighbors in (("baseline", baseline_neighbors), ("candidate", candidate_neighbors)):
                    for neighbor in local_neighbors:
                        neighbor_rows.append({
                            **endpoint_identity, "arm": arm,
                            "batch_classification": batch_state["batch_classification"],
                            **neighbor,
                        })
                rows.append({**endpoint_identity,
                    "batch_classification": batch_state["batch_classification"],
                    "baseline_exact_count": len(baseline_match), "candidate_exact_count": len(candidate_match),
                    "baseline_same_template_within_1_sample_count": len(baseline_tolerant),
                    "candidate_same_template_within_1_sample_count": len(candidate_tolerant),
                    "baseline_any_template_within_1_sample_count": baseline_summary["any_template_within_1_count"],
                    "candidate_any_template_within_1_sample_count": candidate_summary["any_template_within_1_count"],
                    "baseline_any_template_within_30_samples_count": baseline_summary["any_template_within_30_count"],
                    "candidate_any_template_within_30_samples_count": candidate_summary["any_template_within_30_count"],
                    "candidate_exact_retained": candidate_summary["exact_same_time_template_retained"],
                    "candidate_same_template_within_1_retained": candidate_summary["same_template_within_1_retained"],
                    "baseline_threshold_score": float(baseline_threshold[baseline_match[0]]) if len(baseline_match) == 1 else None,
                    "candidate_threshold_score": float(candidate_threshold[candidate_match[0]]) if len(candidate_match) == 1 else None})
            batch_rows.append({"batch_index": ibatch, "raw_read_calls": 1,
                **batch_state,
                "logical_recording_bytes": int(batch.logical_recording_bytes),
                "baseline_detections": len(baseline_st), "candidate_detections": len(candidate_st),
                "unsupported_entries": int(torch.count_nonzero(~paired.support)),
                "baseline_native_filter_binding_exact": baseline_binding_exact})
    wall = time.perf_counter() - started
    if bytes_read > int(config["logical_read_bytes_max"]) or wall > float(config["wall_seconds_max"]):
        raise RuntimeError("execution budget exceeded")
    if len(rows) != int(config["expected_selected_endpoints"]):
        raise RuntimeError("processed endpoint count differs before output publication")
    pd.DataFrame(rows).to_csv(output / "ENDPOINT_COMPARISON.csv", index=False)
    pd.DataFrame(neighbor_rows).to_csv(output / "ENDPOINT_LOCAL_NEIGHBORS.csv", index=False)
    pd.DataFrame(emission_rows).to_csv(output / "MATCHER_EMISSIONS.csv", index=False)
    pd.DataFrame(batch_rows).to_csv(output / "BATCH_RECEIPTS.csv", index=False)
    pd.DataFrame(template_count_rows).to_csv(output / "BATCH_TEMPLATE_COUNTS.csv", index=False)
    pd.DataFrame(changed_template_rows).to_csv(output / "CHANGED_TEMPLATE_COUNTS.csv", index=False)
    result = {"schema": SCHEMA, "status": "complete", "wall_seconds": wall,
        "logical_recording_bytes_read": bytes_read, "raw_read_calls": len(batch_rows),
        "baseline_matcher_replays": len(batch_rows), "candidate_matcher_replays": len(batch_rows),
        "baseline_selected_endpoints_exactly_reproduced": len(rows),
        "matching_rule": "Exact same global time/template; descriptive same-template +/-1 sample all-neighbor count.",
        "transition_limit": "Post-filter pointwise masking does not repair high-pass transients already created on supported rows.",
        "training_limit": "Saved fixed bank only; candidate effects on template learning are untested.",
        "permanent_support_policy": config["permanent_support_policy"],
        "covariance_rank_policy": config["covariance_rank_policy"],
        "interpretation": "Fixed-bank paired replay only; no sorting-quality or biological verdict."}
    atomic_json(output / "RESULT.json", result)
    products = ["RESULT.json", "ENDPOINT_COMPARISON.csv", "ENDPOINT_LOCAL_NEIGHBORS.csv",
                "MATCHER_EMISSIONS.csv", "BATCH_RECEIPTS.csv",
                "BATCH_TEMPLATE_COUNTS.csv", "CHANGED_TEMPLATE_COUNTS.csv"]
    manifest = {name: {"sha256": sha256(output / name), "bytes": (output / name).stat().st_size} for name in products}
    atomic_json(output / "MANIFEST.json", manifest)
    atomic_json(output / "COMPLETE.json", {"schema": SCHEMA, "status": "complete",
        "manifest_sha256": sha256(output / "MANIFEST.json")})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=("preflight", "execute"), required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    preflight(config, args.config) if args.mode == "preflight" else execute(config)


if __name__ == "__main__":
    main()
