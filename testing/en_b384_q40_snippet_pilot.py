#!/usr/bin/env python3
"""Bounded reviewed-only real-data snippet pilot for frozen B384 q+40 pairs.

This source is prepared for review but must not be run until an execution config
sets ``voltage_stage_reviewed`` and binds the frozen preparation artifacts.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd
import torch

from kilosort.io import bfile_from_ops, load_ops
from kilosort.template_matching import prepare_matching, run_matching


SCHEMA = "en-b384-q40-snippet-pilot-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def build_waveform(bank: torch.Tensor, wpca: torch.Tensor, template_id: int) -> torch.Tensor:
    return torch.einsum("pc,pt->ct", bank[template_id], wpca)


def embed(waveform: torch.Tensor, center: int, lo: int, hi: int) -> torch.Tensor:
    result = torch.zeros((waveform.shape[0], hi - lo), dtype=waveform.dtype)
    start = center - waveform.shape[1] // 2 - lo
    result[:, start:start + waveform.shape[1]] = waveform
    return result.reshape(-1)


def fit_pair(x: torch.Tensor, bank: torch.Tensor, wpca: torch.Tensor,
             first_template: int, second_template: int, first_center: int,
             second_center: int, half: int) -> dict[str, Any]:
    lo, hi = first_center - half, second_center + half + 1
    observed = x[:, lo:hi].cpu().reshape(-1)
    first = embed(build_waveform(bank, wpca, first_template).cpu(), first_center, lo, hi)
    second = embed(build_waveform(bank, wpca, second_template).cpu(), second_center, lo, hi)
    design = torch.stack((first, second), dim=1)
    energy = float(torch.dot(observed, observed))
    single_sse = []
    single_coef = []
    for component in (first, second):
        coef = torch.dot(component, observed) / torch.dot(component, component)
        single_coef.append(float(coef))
        single_sse.append(float(torch.sum((observed - coef * component) ** 2)))
    coefficients = torch.linalg.lstsq(design, observed[:, None]).solution[:, 0]
    two_sse = float(torch.sum((observed - design @ coefficients) ** 2))
    cosine = float(torch.dot(first, second) / torch.sqrt(torch.dot(first, first) * torch.dot(second, second)))
    return {
        "observed_energy": energy,
        "best_single_residual_fraction": min(single_sse) / energy if energy else None,
        "two_component_residual_fraction": two_sse / energy if energy else None,
        "two_vs_best_single_sse_reduction_fraction": (min(single_sse) - two_sse) / min(single_sse) if min(single_sse) else None,
        "first_single_coefficient": single_coef[0], "second_single_coefficient": single_coef[1],
        "two_component_first_coefficient": float(coefficients[0]),
        "two_component_second_coefficient": float(coefficients[1]),
        "embedded_component_cosine": cosine,
    }


def execute(config: dict[str, Any]) -> None:
    if not config.get("voltage_stage_reviewed", False):
        raise RuntimeError("voltage stage lacks coordinator review receipt")
    if config.get("device") != "cpu" or int(config.get("threads_max", 0)) > 4:
        raise RuntimeError("execution limits differ")
    torch.set_num_threads(int(config["threads_max"]))
    for item in config["frozen_inputs"]:
        if sha256(Path(item["path"])) != item["sha256"]:
            raise RuntimeError(f"frozen input differs: {item['path']}")
    pairs = pd.read_csv(config["selected_pairs_path"])
    isolated = pd.read_csv(config["isolated_controls_path"])
    batches = sorted(pairs.batch_index.astype(int).unique().tolist())
    ops = load_ops(config["ops_path"], device=torch.device("cpu"))
    nt, nt0min, batch_size = int(ops["nt"]), int(ops["nt0min"]), int(ops["batch_size"])
    expected_bytes = len(batches) * (batch_size + 2 * nt) * int(ops["n_chan_bin"]) * np.dtype(ops["data_dtype"]).itemsize
    if len(batches) > 4 or expected_bytes > int(config["logical_read_bytes_max"]):
        raise RuntimeError("frozen read budget exceeded")
    bank = torch.from_numpy(np.load(config["bank_path"])).to(torch.float32)
    wpca = ops["wPCA"].to(torch.float32)
    ctc = prepare_matching(ops, bank)
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    pair_rows, control_rows, batch_rows = [], [], []
    bytes_read = 0
    with bfile_from_ops(ops=ops, filename=config["recording_path"], device=torch.device("cpu")) as bfile:
        zero = torch.zeros((int(ops["Nchan"]), 1024), dtype=torch.float32)
        zero_st, _, _, _ = run_matching(ops, zero, bank, ctc, device=torch.device("cpu"))
        if len(zero_st):
            raise RuntimeError("zero matcher control produced detections")
        for ibatch in batches:
            x, inds = bfile.padded_batch_to_torch(ibatch, ops=ops, return_inds=True)
            bytes_read += (batch_size + 2 * nt) * int(ops["n_chan_bin"]) * np.dtype(ops["data_dtype"]).itemsize
            replay_st, replay_amp, replay_threshold, _ = run_matching(ops, x, bank, ctc, device=torch.device("cpu"))
            replay_global = (
                (replay_st[:, 0].cpu().numpy() - nt) + ibatch * batch_size
                - nt // 2 + nt0min + int(bfile.imin)
            ).astype(np.int64)
            replay_templates = replay_st[:, 1].cpu().numpy().astype(np.int64)
            selected = pairs[pairs.batch_index.eq(ibatch)]
            batch_rows.append({"batch_index": ibatch, "returned_detections": len(replay_st),
                               "raw_logical_bytes": (batch_size + 2 * nt) * int(ops["n_chan_bin"]) * np.dtype(ops["data_dtype"]).itemsize})
            for row in selected.itertuples(index=False):
                first_center = int(row.first_time_global - int(bfile.imin) - ibatch * batch_size + nt + nt // 2 - nt0min)
                second_center = int(row.second_time_global - int(bfile.imin) - ibatch * batch_size + nt + nt // 2 - nt0min)
                metrics = fit_pair(x, bank, wpca, int(row.first_template), int(row.second_template),
                                   first_center, second_center, nt // 2)
                pair_rows.append({
                    **row._asdict(), **metrics,
                    "first_exactly_replayed": bool(np.any((replay_global == row.first_time_global) & (replay_templates == row.first_template))),
                    "second_exactly_replayed": bool(np.any((replay_global == row.second_time_global) & (replay_templates == row.second_template))),
                })
            for row in isolated[isolated.batch_index.eq(ibatch)].itertuples(index=False):
                center = int(row.time_global - int(bfile.imin) - ibatch * batch_size + nt + nt // 2 - nt0min)
                lo, hi = center - nt // 2, center + nt // 2 + 1
                observed = x[:, lo:hi].cpu().reshape(-1)
                waveform = build_waveform(bank, wpca, int(row.detection_template)).cpu().reshape(-1)
                coef = torch.dot(waveform, observed) / torch.dot(waveform, waveform)
                energy = float(torch.dot(observed, observed))
                residual = float(torch.sum((observed - coef * waveform) ** 2))
                control_rows.append({**row._asdict(), "single_coefficient": float(coef),
                    "single_residual_fraction": residual / energy if energy else None,
                    "exactly_replayed": bool(np.any((replay_global == row.time_global) & (replay_templates == row.detection_template)))})
    wall = time.perf_counter() - started
    if bytes_read != expected_bytes or wall > float(config["wall_seconds_max"]):
        raise RuntimeError("execution budget receipt failed")
    pd.DataFrame(pair_rows).to_csv(output / "PAIR_READOUTS.csv", index=False)
    pd.DataFrame(control_rows).to_csv(output / "ISOLATED_CONTROLS.csv", index=False)
    pd.DataFrame(batch_rows).to_csv(output / "BATCH_RECEIPTS.csv", index=False)
    atomic_json(output / "RESULT.json", {"schema": SCHEMA, "status": "complete",
        "completed_at": datetime.now(timezone.utc).isoformat(), "wall_seconds": wall,
        "logical_recording_bytes_read": bytes_read, "sorts_launched": 0, "rf_accesses": 0,
        "interpretation": "Descriptive model-fit evidence only; residual structure is not biological identity or truth."})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    execute(json.loads(args.config.read_text()))


if __name__ == "__main__":
    main()
