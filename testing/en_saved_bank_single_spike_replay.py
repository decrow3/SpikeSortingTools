#!/usr/bin/env python3
"""Finite CPU-only single-spike replay through frozen EN matching banks."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd
import torch

from kilosort.template_matching import prepare_matching, run_matching
from npx_preprocessing.motion.lattice_remap_si import exact_coordinate_mapping
from testing.en_q0_anchored_diagnostic import (
    atomic_json,
    clean,
    load_events,
    load_field,
    sha256,
    state_assignments,
    verify_frozen_inputs,
)


SCHEMA = "en-saved-bank-single-spike-replay-v1"
NT = 61
N_SAMPLES = 1024
CENTER = 512
MAX_THREADS = 4


def verify_manifest(root: Path, manifest_name: str, expected_hash: str) -> None:
    manifest_path = root / manifest_name
    if sha256(manifest_path) != expected_hash:
        raise RuntimeError(f"manifest differs: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    for name, spec in manifest["products"].items():
        path = root / name
        if path.stat().st_size != spec["bytes"] or sha256(path) != spec["sha256"]:
            raise RuntimeError(f"manifest product differs: {path}")


def upstream(config: dict[str, Any]) -> tuple[dict[str, Any], Path, Path]:
    for name, spec in config["code_inputs"].items():
        if sha256(Path(spec["path"])) != spec["sha256"]:
            raise RuntimeError(f"pinned code differs: {name}")
    parent_path = Path(config["parent_config_path"])
    if sha256(parent_path) != config["parent_config_sha256"]:
        raise RuntimeError("parent config differs")
    parent = json.loads(parent_path.read_text())
    verify_frozen_inputs(parent)
    q40_root = Path(config["q40_root"])
    verify_manifest(q40_root, "MANIFEST.json", config["q40_manifest_sha256"])
    sort_root = Path(config["sort_root"])
    for arm, files in config["bank_inputs"].items():
        for name, expected in files.items():
            if sha256(sort_root / arm / name) != expected:
                raise RuntimeError(f"saved bank input differs: {arm}/{name}")
    return parent, q40_root, sort_root


def modal_lower(values: np.ndarray) -> tuple[int, int, int]:
    ids, counts = np.unique(np.asarray(values, dtype=np.int64), return_counts=True)
    if len(ids) == 0:
        raise RuntimeError("cannot select a modal template from zero events")
    maximum = int(counts.max())
    selected = int(ids[counts == maximum].min())
    return selected, maximum, int(len(values))


def select(config: dict[str, Any], output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    parent, q40_root, sort_root = upstream(config)
    relations = pd.read_csv(q40_root / "Q40_Q0_PAIR_RELATIONS_REF384_vs_B384.csv")
    selected = relations[relations.reference_endpoint_switched].copy()
    selected = selected.sort_values(
        ["reference_state_events", "reference_unit"], ascending=[False, True], kind="stable"
    ).head(int(config["selection_count"]))
    if len(selected) != int(config["selection_count"]) or selected.reference_unit.duplicated().any():
        raise RuntimeError("insufficient distinct switched REF units")
    fs = float(parent["sampling_frequency_hz"])
    start, stop = parent["crop"]["start_frame"], parent["crop"]["stop_frame"]
    cells = load_field(parent)
    q40 = np.flatnonzero(cells.states_um == 40.0)
    if len(q40) != 1:
        raise RuntimeError("expected exactly one +40 state")
    loaded: dict[str, tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]] = {}
    for arm in ("REF384", "B384"):
        events = load_events(Path(parent["arms"][arm]["sorter_output"]), start, stop)
        state = state_assignments(events, cells, fs)[0]
        detection = np.load(sort_root / arm / "sorter_output/spike_detection_templates.npy", mmap_mode="r").reshape(-1)
        if len(detection) != len(events["times"]):
            raise RuntimeError(f"{arm}: detection template/event length mismatch")
        loaded[arm] = events, state, detection
    rows = []
    for rank, row in enumerate(selected.itertuples(index=False), start=1):
        ref_id = int(row.reference_unit)
        b_id = int(row.reference_state_primary_candidate)
        values = {}
        for arm, unit in (("REF384", ref_id), ("B384", b_id)):
            events, state, detection = loaded[arm]
            keep = (events["clusters"] == unit) & (state == int(q40[0]))
            values[arm] = modal_lower(np.asarray(detection[keep]))
        rows.append({
            "selection_rank": rank,
            "reference_unit": ref_id,
            "q0_candidate_unit": int(row.candidate_unit),
            "q40_primary_candidate_unit": b_id,
            "reference_state_events_from_relation": int(row.reference_state_events),
            "reference_detection_template": values["REF384"][0],
            "reference_modal_template_events": values["REF384"][1],
            "reference_unit_q40_events": values["REF384"][2],
            "b_detection_template": values["B384"][0],
            "b_modal_template_events": values["B384"][1],
            "b_unit_q40_events": values["B384"][2],
        })
    payload = {
        "schema": SCHEMA, "phase": "selection", "status": "complete",
        "rule": "four distinct REF units with +40 switched primary; descending reference_state_events then unit ID; modal +40 detection template with lower-ID tie break",
        "waveform_outcomes_read": False, "rows": rows,
        "q40_manifest_sha256": config["q40_manifest_sha256"],
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, payload)
    return payload


def reconstruct_white(bank: np.ndarray, wpca: np.ndarray, template_id: int) -> np.ndarray:
    return np.einsum("pc,pt->tc", np.asarray(bank[template_id], dtype=np.float64),
                     np.asarray(wpca, dtype=np.float64))


def dewhiten(white_time_channel: np.ndarray, whitening: np.ndarray) -> np.ndarray:
    return np.linalg.solve(np.asarray(whitening, dtype=np.float64), white_time_channel.T).T


def rewhiten(physical_time_channel: np.ndarray, whitening: np.ndarray) -> np.ndarray:
    return (np.asarray(whitening, dtype=np.float64) @ physical_time_channel.T).T


def remap_time_channel(waveform: np.ndarray, mapping: np.ndarray) -> np.ndarray:
    result = np.zeros_like(waveform)
    valid = mapping >= 0
    result[:, valid] = waveform[:, mapping[valid]]
    return result


def inject(waveform: np.ndarray) -> np.ndarray:
    if waveform.shape[0] != NT:
        raise RuntimeError("unexpected waveform length")
    result = np.zeros((waveform.shape[1], N_SAMPLES), dtype=np.float32)
    start = CENTER - NT // 2
    result[:, start:start + NT] = waveform.T.astype(np.float32)
    return result


def tensor_ops(raw: dict[str, Any]) -> dict[str, Any]:
    return {"wPCA": torch.from_numpy(np.asarray(raw["wPCA"], dtype=np.float32)),
            "nt": int(raw["nt"]), "Th_learned": float(raw["Th_learned"]),
            "max_peels": int(raw["max_peels"])}


def trial(condition: str, selection_rank: int | None, expected_template: int | None,
          waveform: np.ndarray, ops: dict[str, Any], bank: torch.Tensor,
          ctc: torch.Tensor) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    x = torch.from_numpy(inject(waveform))
    initial_energy = float(torch.sum(x * x))
    st, amps, threshold_amps, residual = run_matching(
        ops, x, bank, ctc, device=torch.device("cpu")
    )
    rows = []
    for order in range(len(st)):
        rows.append({
            "condition": condition, "selection_rank": selection_rank,
            "returned_order": order, "sample": int(st[order, 0]),
            "temporal_offset": int(st[order, 0]) - CENTER,
            "template_id": int(st[order, 1]), "amplitude": float(amps[order, 0]),
            "threshold_amplitude": float(threshold_amps[order, 0]),
        })
    summary = {
        "condition": condition, "selection_rank": selection_rank,
        "expected_template": expected_template, "detections": int(len(st)),
        "centered_detections": int(sum(row["temporal_offset"] == 0 for row in rows)),
        "expected_centered_detections": int(sum(
            row["temporal_offset"] == 0 and row["template_id"] == expected_template for row in rows
        )) if expected_template is not None else None,
        "initial_energy": initial_energy, "residual_energy": float(torch.sum(residual * residual)),
        "residual_energy_fraction": float(torch.sum(residual * residual) / initial_energy) if initial_energy else None,
    }
    return summary, rows


def fixture(output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    w = np.array([[2.0, 1.0, 0.0], [0.0, 3.0, 1.0], [1.0, 0.0, 4.0]])
    physical = np.array([[1.0, -2.0, 0.5], [0.25, 1.5, -1.0]])
    white = rewhiten(physical, w)
    recovered = dewhiten(white, w)
    wrong = white @ np.linalg.inv(w)
    if not np.allclose(recovered, physical, rtol=0, atol=1e-12) or np.allclose(wrong, physical):
        raise RuntimeError("non-symmetric whitening fixture failed")
    geometry = np.array([[0.0, 0.0], [0.0, 40.0], [0.0, 80.0]])
    mapping = exact_coordinate_mapping(geometry, 40.0)
    impulse = np.array([[1.0, 2.0, 3.0]])
    moved = remap_time_channel(impulse, mapping)
    if not np.array_equal(mapping, [1, 2, -1]) or not np.array_equal(moved, [[2.0, 3.0, 0.0]]):
        raise RuntimeError("exact mapping impulse fixture failed")
    wave = np.zeros((NT, 3)); wave[NT // 2, 1] = 1.0
    placed = inject(wave)
    if placed.shape != (3, N_SAMPLES) or placed[1, CENTER] != 1.0 or placed.sum() != 1.0:
        raise RuntimeError("alignment fixture failed")
    payload = {"schema": SCHEMA, "phase": "fixture", "status": "pass",
               "nonsymmetric_whitening": True, "exact_mapping_impulse": True,
               "fixed_center_alignment": True, "wrong_orientation_rejected": True}
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, payload)
    return payload


def run(config: dict[str, Any], output: Path) -> dict[str, Any]:
    started = time.perf_counter()
    if output.exists() or output.with_name(output.name + ".partial").exists():
        raise FileExistsError(output)
    if sha256(Path(__file__)) != config["source_sha256"]:
        raise RuntimeError("frozen source differs")
    fixture_path, selection_path = Path(config["fixture_path"]), Path(config["selection_path"])
    if sha256(fixture_path) != config["fixture_sha256"] or sha256(selection_path) != config["selection_sha256"]:
        raise RuntimeError("fixture or selection differs")
    parent, _, sort_root = upstream(config)
    selection = json.loads(selection_path.read_text())
    partial = output.with_name(output.name + ".partial")
    partial.mkdir(parents=True)
    torch.set_num_threads(MAX_THREADS)
    torch.set_num_interop_threads(1)
    loaded = {}
    for arm in ("REF384", "B384"):
        sorter = sort_root / arm / "sorter_output"
        snapshot = sort_root / arm / "pre_extraction_snapshot"
        raw_ops = np.load(sorter / "ops.npy", allow_pickle=True).item()
        bank_np = np.asarray(np.load(snapshot / "Wall3.npy"), dtype=np.float32)
        captured_wpca = np.asarray(np.load(snapshot / "wPCA.npy"), dtype=np.float32)
        if not np.array_equal(np.asarray(raw_ops["wPCA"], dtype=np.float32), captured_wpca):
            raise RuntimeError(f"{arm}: captured wPCA differs from matcher ops wPCA")
        ops = tensor_ops(raw_ops)
        bank = torch.from_numpy(bank_np)
        ctc = prepare_matching(ops, bank)
        loaded[arm] = {"raw_ops": raw_ops, "ops": ops, "bank_np": bank_np, "bank": bank,
                       "ctc": ctc, "wpca": captured_wpca,
                       "whitening": np.asarray(raw_ops["Wrot"], dtype=np.float64),
                       "positions": np.asarray(np.load(sorter / "channel_positions.npy"), dtype=np.float64)}
    if not np.array_equal(loaded["REF384"]["positions"], loaded["B384"]["positions"]):
        raise RuntimeError("REF384/B384 geometry differs")
    mapping = exact_coordinate_mapping(loaded["REF384"]["positions"], 40.0)
    trial_summaries, detections = [], []
    zero = np.zeros((NT, 384), dtype=np.float64)
    for arm in ("REF384", "B384"):
        summary, rows = trial(f"{arm}_zero", None, None, zero, loaded[arm]["ops"],
                              loaded[arm]["bank"], loaded[arm]["ctc"])
        trial_summaries.append(summary); detections.extend(rows)
    for row in selection["rows"]:
        rank = int(row["selection_rank"])
        ref_id, b_id = int(row["reference_detection_template"]), int(row["b_detection_template"])
        ref_white = reconstruct_white(loaded["REF384"]["bank_np"], loaded["REF384"]["wpca"], ref_id)
        b_white = reconstruct_white(loaded["B384"]["bank_np"], loaded["B384"]["wpca"], b_id)
        physical = dewhiten(ref_white, loaded["REF384"]["whitening"])
        q0_b = rewhiten(physical, loaded["B384"]["whitening"])
        q40_b = rewhiten(remap_time_channel(physical, mapping), loaded["B384"]["whitening"])
        conditions = [
            ("REF_self", ref_id, ref_white, "REF384"),
            ("B_self", b_id, b_white, "B384"),
            ("REF_model_to_B_q0", None, q0_b, "B384"),
            ("REF_model_to_B_q40", None, q40_b, "B384"),
        ]
        for name, expected, wave, bank_arm in conditions:
            summary, rows = trial(name, rank, expected, wave, loaded[bank_arm]["ops"],
                                  loaded[bank_arm]["bank"], loaded[bank_arm]["ctc"])
            trial_summaries.append(summary); detections.extend(rows)
    summary_frame = pd.DataFrame(trial_summaries)
    detection_frame = pd.DataFrame(detections, columns=["condition", "selection_rank", "returned_order",
        "sample", "temporal_offset", "template_id", "amplitude", "threshold_amplitude"])
    zero_pass = bool(summary_frame[summary_frame.condition.str.endswith("_zero")].detections.eq(0).all())
    self_rows = summary_frame[summary_frame.condition.isin(["REF_self", "B_self"])]
    control_details = []
    self_behavior_pass = True
    all_self_templates_above_threshold = True
    for trial_row, selected_row in zip(self_rows.itertuples(index=False),
                                       [item for item in selection["rows"] for _ in range(2)]):
        arm = "REF384" if trial_row.condition == "REF_self" else "B384"
        tid = int(selected_row["reference_detection_template"] if arm == "REF384" else selected_row["b_detection_template"])
        norm = float(np.linalg.norm(loaded[arm]["bank_np"][tid]))
        expected = norm > float(loaded[arm]["ops"]["Th_learned"])
        all_self_templates_above_threshold &= expected
        passed = bool(expected and trial_row.detections == 1 and trial_row.centered_detections == 1
                      and trial_row.residual_energy_fraction is not None
                      and trial_row.residual_energy_fraction <= 1e-8)
        self_behavior_pass &= passed
        returned_ids = detection_frame.loc[
            (detection_frame.condition == trial_row.condition)
            & (detection_frame.selection_rank == trial_row.selection_rank), "template_id"
        ].astype(int).tolist()
        control_details.append({"condition": trial_row.condition, "selection_rank": int(trial_row.selection_rank),
                                "template_id": tid, "template_norm": norm,
                                "native_norm_exceeds_threshold": expected,
                                "returned_template_ids": returned_ids,
                                "identity_of_returned_template_required": False,
                                "residual_energy_tolerance": 1e-8, "passed": passed})
    self_pass = bool(all_self_templates_above_threshold and self_behavior_pass)
    controls_pass = bool(zero_pass and self_pass)
    summary_frame.to_csv(partial / "TRIAL_SUMMARY.csv", index=False)
    detection_frame.to_csv(partial / "DETECTIONS.csv", index=False)
    atomic_json(partial / "CONTROL_DETAILS.json", control_details)
    atomic_json(partial / "SELECTION_COPY.json", selection)
    result = {"schema": SCHEMA, "phase": "run", "status": "complete",
              "controls_pass": controls_pass, "zero_controls_pass": zero_pass,
              "self_controls_pass": bool(self_pass), "science_interpretation_allowed": controls_pass,
              "all_self_templates_above_threshold": bool(all_self_templates_above_threshold),
              "model_definition": "filtered/CAR-referenced REF bank model, correctly dewhitened, optionally exact-lattice-remapped, then B-whitened; no new CAR/HP/noise/quantization",
              "faithful_recording_replay": False, "threads": MAX_THREADS,
              "raw_voltage_bytes_read": 0, "sorts_launched": 0, "rf_accesses": 0,
              "wall_seconds": time.perf_counter() - started,
              "completed_at": datetime.now(timezone.utc).isoformat()}
    atomic_json(partial / "RESULT.json", result)
    products = {p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in sorted(partial.iterdir())}
    atomic_json(partial / "MANIFEST.json", {"schema": SCHEMA, "products": products})
    atomic_json(partial / "COMPLETE.json", {"schema": SCHEMA, "status": "complete",
                "manifest_sha256": sha256(partial / "MANIFEST.json")})
    os.replace(partial, output)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--phase", choices=("fixture", "selection", "run"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.phase == "fixture":
        result = fixture(args.output)
    else:
        if args.config is None:
            raise RuntimeError("--config is required outside fixture phase")
        config = json.loads(args.config.read_text())
        if config.get("schema") != SCHEMA:
            raise RuntimeError("unexpected config schema")
        result = select(config, args.output) if args.phase == "selection" else run(config, args.output)
    print(json.dumps(clean(result), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
