#!/usr/bin/env python3
"""Correction-only metadata gate for the bounded B384 q+40 snippet pilot."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from kilosort.template_matching import prepare_matching, run_matching
from testing.en_b384_q40_snippet_pilot_v2 import build_waveform, fit_pair
from testing.en_common_support_crop_panel import crop_field_cells_from_global_midpoints
from testing.en_rounded_field import rounded_rigid_field


SCHEMA = "en-b384-q40-snippet-preparation-v4"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(partial, path)


def verify(config: dict[str, Any]) -> None:
    if config["voltage_reads_permitted"]:
        raise RuntimeError("metadata correction must prohibit voltage reads")
    for item in config["frozen_inputs"]:
        if sha256(Path(item["path"])) != item["sha256"]:
            raise RuntimeError(f"frozen input differs: {item['path']}")


def load_cells(config: dict[str, Any], times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    field = config["field"]
    with np.load(field["path"], allow_pickle=False) as saved:
        displacement = np.asarray(saved[field["displacement_key"]], dtype=float)
        if displacement.ndim == 2:
            displacement = np.median(displacement, axis=1)
        rounded, _ = rounded_rigid_field(displacement)
        cells = crop_field_cells_from_global_midpoints(
            np.asarray(saved[field["time_key"]], dtype=float), rounded,
            float(config["recording_duration_s"]),
            int(config["crop_start_frame"]) / float(config["sampling_frequency_hz"]),
            int(config["crop_stop_frame"]) / float(config["sampling_frequency_hz"]),
            float(config["accounting_block_seconds"]),
        )
    state_index, segment = cells.assign(times, float(config["sampling_frequency_hz"]))
    return np.asarray(cells.states_um)[state_index], segment


def matcher_center(global_sample: int, ibatch: int, imin: int, batch_size: int, nt: int, nt0min: int) -> int:
    return int(global_sample - imin - ibatch * batch_size + nt + nt // 2 - nt0min)


def matcher_global(center: int, ibatch: int, imin: int, batch_size: int, nt: int, nt0min: int) -> int:
    return int((center - nt) + ibatch * batch_size - nt // 2 + nt0min + imin)


def fixtures() -> dict[str, Any]:
    nt, nt0min, batch_size, imin = 61, 20, 60000, 208498882
    for ibatch, sample, expected_center in ((0, imin + 120, 191), (69, imin + 69 * batch_size + 32123, 32194)):
        center = matcher_center(sample, ibatch, imin, batch_size, nt, nt0min)
        if center != expected_center or matcher_global(center, ibatch, imin, batch_size, nt, nt0min) != sample:
            raise RuntimeError("event/matcher-center known-answer fixture failed")
    wpca = torch.zeros((1, nt), dtype=torch.float32)
    wpca[0, nt // 2] = 1.0
    bank = torch.zeros((2, 1, 4), dtype=torch.float32)
    bank[0, 0, 0] = 1.0
    bank[1, 0, 1] = 1.0
    x = torch.zeros((4, 128), dtype=torch.float32)
    c1, c2 = 48, 59
    # Independent known answer for these delta-basis templates: only two scalar
    # samples are nonzero. Do not generate the fixture with build_waveform.
    x[0, c1] = 0.8
    x[1, c2] = 0.4
    fit = fit_pair(x, bank, wpca, 0, 1, c1, c2, nt // 2)
    if abs(fit["two_component_first_coefficient"] - 0.8) > 1e-6 or abs(fit["two_component_second_coefficient"] - 0.4) > 1e-6 or fit["two_component_residual_fraction"] > 1e-12:
        raise RuntimeError("two-component known-answer fixture failed")
    ops = {"nt": nt, "wPCA": wpca, "Th_learned": 9, "max_peels": 100}
    ctc = prepare_matching(ops, bank)
    zero_st, _, _, _ = run_matching(ops, torch.zeros((4, 1024)), bank, ctc, device=torch.device("cpu"))
    if len(zero_st):
        raise RuntimeError("zero matcher fixture failed")
    return {"event_center_roundtrip": "pass", "two_component_fit": "pass", "zero_matcher": "pass"}


def protocol() -> str:
    return """# V3 correction and future execution contract

- Preserve V2. Its same-unit isolated control was selected after restricting the unit train to q+40,
  contradicting the complete-train protocol. V3 recomputes only that control using neighbors from the
  complete saved unit train; the candidate event itself must be q+40 and in the already selected batch.
- Keep the V2 units, batches, and pair sample unchanged and verify their hashes.
- Before future recording access, bind saved ops, binary size/path/dtype/stride, crop imin/imax, exact
  bank shape, captured wPCA equality, selected batch core bounds, and expected padded read bounds.
- Engineering acceptance is mandatory: zero matcher gives zero events; every selected pair endpoint
  and every same-unit-isolated control is reproduced at its exact saved time/template. Any failure
  stops before scientific readout.
- Future capability is limited to exact matcher reproduction and saved-bank model fit. Full-channel
  residuals cannot identify biological components or decide spike versus non-spike truth.
- Execution must use the reviewed systemd unit with RuntimeMaxSec=295 and TimeoutStopSec=5,
  MemoryMax=4G, CPUQuota=400%, four or fewer padded batches, and no more than 512 MiB logical
  recording reads. The recording-processing lifetime, including stop grace, is at most 300 seconds.
"""


def run(config: dict[str, Any], output: Path) -> None:
    verify(config)
    v2 = config["v2"]
    units = pd.read_csv(v2["selected_units_path"])
    batches = pd.read_csv(v2["selected_batches_path"])
    pairs = pd.read_csv(v2["selected_pairs_path"])
    old_isolated = pd.read_csv(v2["isolated_controls_path"])
    sorter = Path(config["sorter_output"])
    times_abs = np.asarray(np.load(sorter / "spike_times.npy", mmap_mode="r"), dtype=np.int64).reshape(-1)
    clusters = np.asarray(np.load(sorter / "spike_clusters.npy", mmap_mode="r"), dtype=np.int64).reshape(-1)
    templates = np.asarray(np.load(sorter / "spike_detection_templates.npy", mmap_mode="r"), dtype=np.int64).reshape(-1)
    start = int(config["crop_start_frame"])
    times = times_abs - start
    state_um, _ = load_cells(config, times)
    nt, batch_size = int(config["expected_ops"]["nt"]), int(config["expected_ops"]["batch_size"])
    corrected = []
    for row in batches.itertuples(index=False):
        unit_indices = np.flatnonzero(clusters == row.unit_id)
        unit_times = times[unit_indices]
        candidate_positions = np.flatnonzero(
            (state_um[unit_indices] == 40.0)
            & (unit_times >= row.batch_core_start_global - start + nt)
            & (unit_times < row.batch_core_stop_global_exclusive - start - nt)
        )
        left = np.maximum(candidate_positions - 1, 0)
        right = np.minimum(candidate_positions + 1, len(unit_times) - 1)
        left_gap = np.where(candidate_positions > 0, unit_times[candidate_positions] - unit_times[left], np.iinfo(np.int64).max)
        right_gap = np.where(candidate_positions + 1 < len(unit_times), unit_times[right] - unit_times[candidate_positions], np.iinfo(np.int64).max)
        eligible = candidate_positions[(left_gap > nt) & (right_gap > nt)]
        if not len(eligible):
            raise RuntimeError(f"unit {row.unit_id} has no complete-train isolated control")
        event_index = int(unit_indices[eligible[0]])
        corrected.append({"unit_id": int(row.unit_id), "selection_role": row.selection_role,
            "batch_index": int(row.batch_index), "time_global": int(times_abs[event_index]),
            "detection_template": int(templates[event_index]),
            "complete_unit_train_exclusion_radius_samples": nt})
    corrected_frame = pd.DataFrame(corrected)

    ops = np.load(config["ops_path"], allow_pickle=True).item()
    observed_ops = {"n_chan_bin": int(ops["n_chan_bin"]), "Nchan": int(ops["Nchan"]),
        "batch_size": int(ops["batch_size"]), "nt": int(ops["nt"]), "nt0min": int(ops["nt0min"]),
        "data_dtype": str(ops["data_dtype"]), "imin": int(np.int64(ops["tmin"] * ops["fs"])),
        "imax": int(np.int64(ops["tmax"] * ops["fs"]))}
    if observed_ops != config["expected_ops"]:
        raise RuntimeError(f"saved ops differ: {observed_ops}")
    if str(ops["filename"]) != config["recording_path"] or Path(config["recording_path"]).stat().st_size != config["recording_bytes"]:
        raise RuntimeError("recording path/size metadata differs")
    bank = np.load(config["bank_path"], mmap_mode="r")
    wpca = np.load(config["wpca_path"], mmap_mode="r")
    ops_wpca = np.asarray(ops["wPCA"].cpu() if hasattr(ops["wPCA"], "cpu") else ops["wPCA"])
    if list(bank.shape) != config["expected_bank_shape"] or not np.array_equal(wpca, ops_wpca):
        raise RuntimeError("bank shape or wPCA binding differs")
    padded = []
    for row in batches.itertuples(index=False):
        ibatch = int(row.batch_index)
        read_start = observed_ops["imin"] + ibatch * batch_size - nt
        read_stop = read_start + batch_size + 2 * nt
        if not (observed_ops["imin"] <= row.batch_core_start_global < row.batch_core_stop_global_exclusive <= observed_ops["imax"]):
            raise RuntimeError("selected batch core outside effective crop")
        padded.append({**row._asdict(), "padded_read_start_global": read_start,
            "padded_read_stop_global_exclusive": read_stop,
            "logical_recording_bytes": (batch_size + 2 * nt) * observed_ops["n_chan_bin"] * np.dtype(observed_ops["data_dtype"]).itemsize})
    padded_frame = pd.DataFrame(padded)
    logical_bytes = int(padded_frame.logical_recording_bytes.sum())
    if len(padded_frame) > 4 or logical_bytes > 512 * 2**20:
        raise RuntimeError("future read budget exceeded")
    fixture = fixtures()
    acceptance = {
        "pre_science_abort_rules": [
            "zero matcher must return zero detections",
            "all selected saved pair endpoints must replay at exact time and detection-template ID",
            "all complete-unit-train isolated controls must replay at exact time and detection-template ID",
            "returned padded batch inds must equal SELECTED_BATCHES padded bounds",
            "any timeout, OOM, hash, ops, bank, wPCA, stride, dtype, crop, or budget mismatch is failure"
        ],
        "allowed_readout": "Exact matcher reproduction and saved-bank one/two-component model-fit metrics only",
        "forbidden_inference": "No residual result identifies biological spike count, identity, purity, or spike/non-spike truth",
        "decision_to_intervention": "Only a repeated excluded-target model-fit pattern beyond both interior-pair and isolated-event controls may motivate a separately frozen bounded matcher intervention; otherwise return to preprocessing/alignment diagnostics. No production change follows directly."
    }
    units.to_csv(output / "SELECTED_UNITS.csv", index=False)
    pairs.to_csv(output / "SELECTED_PAIRS.csv", index=False)
    padded_frame.to_csv(output / "SELECTED_BATCHES.csv", index=False)
    corrected_frame.to_csv(output / "SELECTED_ISOLATED_CONTROLS.csv", index=False)
    atomic_json(output / "FIXTURE.json", fixture)
    atomic_json(output / "ACCEPTANCE.json", acceptance)
    old_key = old_isolated[["unit_id", "batch_index", "time_global", "detection_template"]]
    new_key = corrected_frame[["unit_id", "batch_index", "time_global", "detection_template"]]
    result = {"schema": SCHEMA, "status": "prepared_review_pending",
        "completed_at": datetime.now(timezone.utc).isoformat(), "raw_voltage_bytes_read": 0,
        "sorts_launched": 0, "rf_accesses": 0, "selected_units_changed_from_v2": False,
        "selected_batches_changed_from_v2": False, "selected_pairs_changed_from_v2": False,
        "isolated_controls_changed_from_v2": not old_key.equals(new_key),
        "future_padded_batch_count": len(padded_frame), "future_logical_recording_read_bytes": logical_bytes,
        "future_logical_recording_read_mib": logical_bytes / 2**20,
        "bank_shape": list(bank.shape), "wpca_shape": list(wpca.shape),
        "rough_live_memory_mib": config["rough_live_memory_mib"],
        "runtime_under_300s_verified": False,
        "capability": "Preparation and fixtures only; no waveform result exists."}
    atomic_json(output / "RESULT.json", result)
    products = ["PROTOCOL.md", "FREEZE.json", "SELECTED_UNITS.csv", "SELECTED_PAIRS.csv", "SELECTED_BATCHES.csv",
        "SELECTED_ISOLATED_CONTROLS.csv", "FIXTURE.json", "ACCEPTANCE.json", "RESULT.json"]
    atomic_json(output / "MANIFEST.json", {"schema": SCHEMA, "products": {name: {
        "bytes": (output / name).stat().st_size, "sha256": sha256(output / name)} for name in products}})
    atomic_json(output / "COMPLETE.json", {"schema": SCHEMA, "status": "prepared_review_pending",
        "manifest_sha256": sha256(output / "MANIFEST.json"), "result_sha256": sha256(output / "RESULT.json")})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--phase", choices=("freeze", "prepare"), required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    verify(config)
    if args.phase == "freeze":
        if any((output / name).exists() for name in ("RESULT.json", "MANIFEST.json", "COMPLETE.json")):
            raise FileExistsError("V3 output exists")
        (output / "PROTOCOL.md").write_text(protocol())
        atomic_json(output / "FREEZE.json", {"schema": SCHEMA, "status": "frozen_preparation",
            "frozen_at": datetime.now(timezone.utc).isoformat(), "config_sha256": sha256(args.config),
            "source_sha256": sha256(Path(__file__)), "protocol_sha256": sha256(output / "PROTOCOL.md")})
        return
    freeze = json.loads((output / "FREEZE.json").read_text())
    if freeze["config_sha256"] != sha256(args.config) or freeze["source_sha256"] != sha256(Path(__file__)):
        raise RuntimeError("V3 source/config differs from freeze")
    run(config, output)


if __name__ == "__main__":
    main()
