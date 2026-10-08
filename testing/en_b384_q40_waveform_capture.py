#!/usr/bin/env python3
"""Reviewed-only bounded measured-signal capture for frozen B384 q+40 events."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time
import traceback
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from kilosort.io import BinaryRWFile, bfile_from_ops, load_ops
from kilosort.template_matching import prepare_matching, run_matching
from testing.en_b384_q40_snippet_pilot_v2 import build_waveform, fit_pair


SCHEMA = "en-b384-q40-waveform-capture-v1"


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


def matcher_center(global_sample: int, ibatch: int, imin: int, batch_size: int, nt: int, nt0min: int) -> int:
    return int(global_sample - imin - ibatch * batch_size + nt + nt // 2 - nt0min)


def matcher_global(center: np.ndarray, ibatch: int, imin: int, batch_size: int, nt: int, nt0min: int) -> np.ndarray:
    return ((center - nt) + ibatch * batch_size - nt // 2 + nt0min + imin).astype(np.int64)


def verify(config: dict[str, Any], allow_run: bool) -> None:
    if not config.get("human_scope_approved", False):
        raise RuntimeError("exact human scope approval is absent")
    if allow_run and not config.get("hub_source_reviewed", False):
        raise RuntimeError("hub source review is pending")
    if config["device"] != "cpu" or int(config["threads_max"]) > 4:
        raise RuntimeError("execution limits differ")
    for item in config["frozen_inputs"]:
        if sha256(Path(item["path"])) != item["sha256"]:
            raise RuntimeError(f"frozen input differs: {item['path']}")
    manifest = json.loads(Path(config["recording_manifest_path"]).read_text())
    files = manifest.get("recording_binary_files", [])
    if len(files) != 1 or files[0].get("sha256") != config["accepted_recording_sha256"]:
        raise RuntimeError("recording receipt binding differs")
    recording = Path(config["recording_path"])
    if recording.stat().st_size != int(config["recording_bytes"]):
        raise RuntimeError("recording byte size differs")


def independent_fixtures() -> dict[str, Any]:
    nt, nt0min, batch_size, imin = 61, 20, 60000, 208498882
    known = ((0, imin + 120, 191), (69, imin + 69 * batch_size + 32123, 32194))
    for ibatch, sample, expected in known:
        if matcher_center(sample, ibatch, imin, batch_size, nt, nt0min) != expected:
            raise RuntimeError("event-to-matcher-center fixture failed")
    wpca = torch.zeros((1, nt), dtype=torch.float32)
    wpca[0, nt // 2] = 1
    bank = torch.zeros((2, 1, 4), dtype=torch.float32)
    bank[0, 0, 0] = 1
    bank[1, 0, 1] = 1
    x = torch.zeros((4, 128), dtype=torch.float32)
    x[0, 48], x[1, 59] = 0.8, 0.4
    fitted = fit_pair(x, bank, wpca, 0, 1, 48, 59, nt // 2)
    if abs(fitted["two_component_first_coefficient"] - 0.8) > 1e-6 or abs(fitted["two_component_second_coefficient"] - 0.4) > 1e-6 or fitted["two_component_residual_fraction"] > 1e-12:
        raise RuntimeError("independent two-component fixture failed")
    ops = {"nt": nt, "wPCA": wpca, "Th_learned": 9, "max_peels": 100}
    ctc = prepare_matching(ops, bank)
    st, _, _, _ = run_matching(ops, torch.zeros((4, 1024)), bank, ctc, device=torch.device("cpu"))
    if len(st):
        raise RuntimeError("zero matcher fixture failed")
    return {"event_center_known_answer": "pass", "two_component_known_answer": "pass", "zero_matcher": "pass"}


def build_windows(pairs: pd.DataFrame, singles: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    context = int(config["context_samples"])
    max_lag = int(config["max_pair_lag_samples"])
    width = 2 * context + max_lag + 1
    rows = []
    for row in pairs.itertuples(index=False):
        rows.append({"window_id": f"pair_{int(row.unit_id)}_{int(row.sample_rank)}",
            "kind": "pair", "unit_id": int(row.unit_id), "selection_role": row.selection_role,
            "batch_index": int(row.batch_index), "first_time_global": int(row.first_time_global),
            "second_time_global": int(row.second_time_global), "first_template": int(row.first_template),
            "second_template": int(row.second_template), "lag_samples": int(row.lag_samples),
            "width_samples": width})
    for row in singles.itertuples(index=False):
        rows.append({"window_id": f"single_{int(row.unit_id)}", "kind": "single",
            "unit_id": int(row.unit_id), "selection_role": row.selection_role,
            "batch_index": int(row.batch_index), "first_time_global": int(row.time_global),
            "second_time_global": None, "first_template": int(row.detection_template),
            "second_template": None, "lag_samples": None, "width_samples": width})
    frame = pd.DataFrame(rows)
    if len(frame) != 20 or frame.window_id.duplicated().any() or not np.all(frame.width_samples == width):
        raise RuntimeError("frozen window construction differs")
    return frame


def embedded_component(bank: torch.Tensor, wpca: torch.Tensor, template_id: int,
                       center: int, start: int, width: int) -> torch.Tensor:
    waveform = build_waveform(bank, wpca, template_id).cpu()
    result = torch.zeros((waveform.shape[0], width), dtype=waveform.dtype)
    offset = center - waveform.shape[1] // 2 - start
    result[:, offset:offset + waveform.shape[1]] = waveform
    return result


def plot_window(path: Path, raw: np.ndarray, prewhite: np.ndarray, white: np.ndarray,
                fitted: np.ndarray, residual: np.ndarray, global_samples: np.ndarray,
                row: pd.Series, template_energy: np.ndarray,
                saved_events: list[int], matcher_centers: list[int]) -> None:
    chosen = np.argsort(template_energy)[-4:][::-1]
    fig, axes = plt.subplots(5, 1, figsize=(11, 15), constrained_layout=True)
    panels = ((raw, "raw ADC counts (all 384 rows)"),
              (prewhite, "native CAR/high-pass pre-W units (all 384 rows)"),
              (white, "native whitened units (all 384 rows)"))
    for axis, (values, title) in zip(axes[:3], panels):
        vmax = float(np.percentile(np.abs(values), 99.5)) or 1.0
        axis.imshow(values, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                    extent=[global_samples[0], global_samples[-1], 0, values.shape[0]-1])
        axis.set_ylabel("selected channel index")
        axis.set_title(f"{row.window_id}: {title}")
    for channel in chosen:
        axes[3].plot(global_samples, white[channel], alpha=.7, label=f"data ch{channel}")
        axes[3].plot(global_samples, fitted[channel], linestyle="--", linewidth=1)
    axes[3].legend(ncol=2, fontsize=7)
    axes[3].set_ylabel("whitened units")
    axes[3].set_title("Observed traces and frozen saved-template fit")
    rmax = float(np.percentile(np.abs(residual), 99.5)) or 1.0
    axes[4].imshow(residual, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-rmax, vmax=rmax,
                   extent=[global_samples[0], global_samples[-1], 0, residual.shape[0]-1])
    axes[4].set_ylabel("selected channel index")
    axes[4].set_xlabel("global recording sample")
    axes[4].set_title("Model residual (descriptive; not biological truth)")
    for axis in axes:
        for sample in saved_events:
            axis.axvline(sample, color="black", linestyle=":", linewidth=1, label="saved event")
        for sample in matcher_centers:
            axis.axvline(sample, color="#009E73", linestyle="--", linewidth=1, label="matcher center")
    fig.savefig(path, dpi=130)
    plt.close(fig)


def execute(config: dict[str, Any]) -> None:
    verify(config, allow_run=True)
    torch.set_num_threads(int(config["threads_max"]))
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=False)
    plots = output / "plots"
    plots.mkdir()
    pair_rows: list[dict[str, Any]] = []
    single_rows: list[dict[str, Any]] = []
    batch_rows: list[dict[str, Any]] = []
    bytes_read = 0
    raw_snippets = prewhite_snippets = white_snippets = global_axes = None
    initialized = np.zeros(20, dtype=bool)
    windows = None
    channel_map = xc = yc = None
    started = time.perf_counter()
    try:
        pairs = pd.read_csv(config["selected_pairs_path"])
        singles = pd.read_csv(config["isolated_controls_path"])
        selected_batches = pd.read_csv(config["selected_batches_path"])
        windows = build_windows(pairs, singles, config)
        ops = load_ops(config["ops_path"], device=torch.device("cpu"))
        expected_ops = config["expected_ops"]
        observed_ops = {"n_chan_bin": int(ops["n_chan_bin"]), "Nchan": int(ops["Nchan"]),
            "batch_size": int(ops["batch_size"]), "nt": int(ops["nt"]), "nt0min": int(ops["nt0min"]),
            "data_dtype": str(ops["data_dtype"]), "imin": int(np.int64(ops["tmin"] * ops["fs"])),
            "imax": int(np.int64(ops["tmax"] * ops["fs"]))}
        if observed_ops != expected_ops or ops["dshift"] is not None or ops["shift"] is not None or ops["scale"] is not None:
            raise RuntimeError("effective ops/dshift/scale/shift differ")
        channel_map = np.asarray(ops["probe"]["chanMap"].cpu() if hasattr(ops["probe"]["chanMap"], "cpu") else ops["probe"]["chanMap"], dtype=np.int64)
        xc = np.asarray(ops["xc"].cpu() if hasattr(ops["xc"], "cpu") else ops["xc"], dtype=float)
        yc = np.asarray(ops["yc"].cpu() if hasattr(ops["yc"], "cpu") else ops["yc"], dtype=float)
        if not np.array_equal(channel_map, np.arange(384)) or len(xc) != 384 or len(yc) != 384:
            raise RuntimeError("full 384-row target geometry differs")
        bank = torch.from_numpy(np.load(config["bank_path"])).to(torch.float32)
        wpca = ops["wPCA"].to(torch.float32)
        captured_wpca = torch.from_numpy(np.load(config["wpca_path"])).to(torch.float32)
        if tuple(bank.shape) != tuple(config["expected_bank_shape"]) or not torch.equal(wpca.cpu(), captured_wpca):
            raise RuntimeError("bank/wPCA binding differs")
        ctc = prepare_matching(ops, bank)
        zero_st, _, _, _ = run_matching(ops, torch.zeros((384, 1024)), bank, ctc, device=torch.device("cpu"))
        if len(zero_st):
            raise RuntimeError("zero matcher control produced detections")
        width = int(windows.width_samples.iloc[0])
        raw_snippets = np.empty((20, 384, width), dtype=np.int16)
        prewhite_snippets = np.empty((20, 384, width), dtype=np.float32)
        white_snippets = np.empty((20, 384, width), dtype=np.float32)
        global_axes = np.empty((20, width), dtype=np.int64)
        window_index = {value: index for index, value in enumerate(windows.window_id)}
        with bfile_from_ops(ops=ops, filename=config["recording_path"], device=torch.device("cpu")) as bfile:
            if bfile.whiten_mat is None or bfile.dshift is not None:
                raise RuntimeError("native whitening/dshift state differs")
            for batch in selected_batches.itertuples(index=False):
                ibatch = int(batch.batch_index)
                raw_float, inds = BinaryRWFile.padded_batch_to_torch(bfile, ibatch, return_inds=True)
                bytes_read += int(batch.logical_recording_bytes)
                if [int(inds[0]), int(inds[1])] != [int(batch.padded_read_start_global), int(batch.padded_read_stop_global_exclusive)]:
                    raise RuntimeError("actual raw padded read bounds differ")
                if not torch.equal(raw_float, raw_float.round()) or raw_float.min() < -32768 or raw_float.max() > 32767:
                    raise RuntimeError("raw int16 roundtrip differs")
                whitening = bfile.whiten_mat
                bfile.whiten_mat = None
                prewhite = bfile.filter(raw_float.clone(), ops=ops, ibatch=ibatch)
                bfile.whiten_mat = whitening
                white = bfile.filter(raw_float.clone(), ops=ops, ibatch=ibatch)
                white_from_pre = whitening @ prewhite
                exact_prewhite_map = torch.equal(white, white_from_pre)
                if not exact_prewhite_map:
                    raise RuntimeError("captured pre-W signal does not map exactly to matcher input")
                replay_st, replay_amp, replay_threshold, _ = run_matching(ops, white, bank, ctc, device=torch.device("cpu"))
                replay_global = matcher_global(replay_st[:, 0].cpu().numpy(), ibatch, int(bfile.imin),
                                               int(ops["batch_size"]), int(ops["nt"]), int(ops["nt0min"]))
                replay_templates = replay_st[:, 1].cpu().numpy().astype(np.int64)
                batch_rows.append({"batch_index": ibatch, "raw_read_calls": 1,
                    "returned_read_start_global": int(inds[0]), "returned_read_stop_global_exclusive": int(inds[1]),
                    "logical_recording_bytes": int(batch.logical_recording_bytes),
                    "prewhite_to_matcher_input_exact": exact_prewhite_map,
                    "prewhite_to_matcher_input_max_abs_error": float(torch.max(torch.abs(white - white_from_pre))),
                    "returned_matcher_detections": len(replay_st)})
                for _, window in windows[windows.batch_index.eq(ibatch)].iterrows():
                    first_center = matcher_center(int(window.first_time_global), ibatch, int(bfile.imin),
                                                  int(ops["batch_size"]), int(ops["nt"]), int(ops["nt0min"]))
                    start_index = first_center - int(config["context_samples"])
                    stop_index = start_index + width
                    if start_index < 0 or stop_index > raw_float.shape[1]:
                        raise RuntimeError("snippet lies outside one padded batch")
                    destination = window_index[window.window_id]
                    raw_snippets[destination] = raw_float[:, start_index:stop_index].cpu().numpy().astype(np.int16)
                    prewhite_snippets[destination] = prewhite[:, start_index:stop_index].cpu().numpy()
                    white_snippets[destination] = white[:, start_index:stop_index].cpu().numpy()
                    global_axes[destination] = np.arange(int(inds[0]) + start_index, int(inds[0]) + stop_index, dtype=np.int64)
                    initialized[destination] = True
                    first_match = np.flatnonzero((replay_global == int(window.first_time_global)) & (replay_templates == int(window.first_template)))
                    if len(first_match) != 1:
                        raise RuntimeError(f"{window.window_id}: first endpoint exact replay count {len(first_match)}")
                    common = {"window_id": window.window_id, "kind": window.kind, "unit_id": int(window.unit_id),
                        "selection_role": window.selection_role, "batch_index": ibatch,
                        "window_start_global": int(global_axes[destination, 0]),
                        "window_stop_global_exclusive": int(global_axes[destination, -1] + 1),
                        "first_saved_event_global": int(window.first_time_global),
                        "first_matcher_center_global": int(inds[0]) + first_center,
                        "first_event_to_matcher_center_samples": int(inds[0]) + first_center - int(window.first_time_global),
                        "first_template": int(window.first_template),
                        "first_replay_amplitude": float(replay_amp[first_match[0]]),
                        "first_replay_threshold_amplitude": float(replay_threshold[first_match[0]])}
                    if window.kind == "pair":
                        second_center = matcher_center(int(window.second_time_global), ibatch, int(bfile.imin),
                                                       int(ops["batch_size"]), int(ops["nt"]), int(ops["nt0min"]))
                        second_match = np.flatnonzero((replay_global == int(window.second_time_global)) & (replay_templates == int(window.second_template)))
                        if len(second_match) != 1:
                            raise RuntimeError(f"{window.window_id}: second endpoint exact replay count {len(second_match)}")
                        metrics = fit_pair(white, bank, wpca, int(window.first_template), int(window.second_template),
                                           first_center, second_center, int(ops["nt"]) // 2)
                        first_component = embedded_component(bank, wpca, int(window.first_template), first_center, start_index, width)
                        second_component = embedded_component(bank, wpca, int(window.second_template), second_center, start_index, width)
                        fitted = (metrics["two_component_first_coefficient"] * first_component
                                  + metrics["two_component_second_coefficient"] * second_component).numpy()
                        template_energy = np.sum(first_component.numpy()**2 + second_component.numpy()**2, axis=1)
                        residual = white_snippets[destination] - fitted
                        pair_rows.append({**common, "second_saved_event_global": int(window.second_time_global),
                            "second_matcher_center_global": int(inds[0]) + second_center,
                            "second_event_to_matcher_center_samples": int(inds[0]) + second_center - int(window.second_time_global),
                            "second_template": int(window.second_template), "lag_samples": int(window.lag_samples),
                            "second_replay_amplitude": float(replay_amp[second_match[0]]),
                            "second_replay_threshold_amplitude": float(replay_threshold[second_match[0]]), **metrics})
                    else:
                        waveform = build_waveform(bank, wpca, int(window.first_template)).cpu()
                        observed = white[:, first_center-int(ops["nt"])//2:first_center+int(ops["nt"])//2+1].cpu().reshape(-1)
                        component = waveform.reshape(-1)
                        coefficient = torch.dot(component, observed) / torch.dot(component, component)
                        residual_fraction = float(torch.sum((observed - coefficient * component)**2) / torch.sum(observed**2))
                        embedded = embedded_component(bank, wpca, int(window.first_template), first_center, start_index, width)
                        fitted = (float(coefficient) * embedded).numpy()
                        template_energy = np.sum(embedded.numpy()**2, axis=1)
                        residual = white_snippets[destination] - fitted
                        single_rows.append({**common, "single_coefficient": float(coefficient),
                            "single_residual_fraction": residual_fraction})
                    saved_events = [int(window.first_time_global)]
                    matcher_centers = [int(inds[0]) + first_center]
                    if window.kind == "pair":
                        saved_events.append(int(window.second_time_global))
                        matcher_centers.append(int(inds[0]) + second_center)
                    plot_window(plots / f"{window.window_id}.png", raw_snippets[destination],
                                prewhite_snippets[destination], white_snippets[destination], fitted, residual,
                                global_axes[destination], window, template_energy, saved_events, matcher_centers)
        if len(pair_rows) != 16 or len(single_rows) != 4 or len(batch_rows) != 4:
            raise RuntimeError("captured row counts differ")
        numeric_nbytes = raw_snippets.nbytes + prewhite_snippets.nbytes + white_snippets.nbytes + global_axes.nbytes
        if numeric_nbytes > int(config["retained_numeric_bytes_max"]):
            raise RuntimeError("uncompressed retained numeric arrays exceed cap")
        np.savez_compressed(output / "SNIPPETS.npz", raw=raw_snippets, prewhite=prewhite_snippets,
                            whitened=white_snippets, global_samples=global_axes,
                            physical_channel_ids=channel_map, x_um=xc, y_um=yc)
        if (output / "SNIPPETS.npz").stat().st_size > int(config["retained_numeric_bytes_max"]):
            raise RuntimeError("retained numeric file exceeds cap")
        windows.to_csv(output / "WINDOWS.csv", index=False)
        pd.DataFrame(pair_rows).to_csv(output / "PAIR_METRICS.csv", index=False)
        pd.DataFrame(single_rows).to_csv(output / "SINGLE_METRICS.csv", index=False)
        pd.DataFrame(batch_rows).to_csv(output / "BATCH_RECEIPTS.csv", index=False)
        pd.DataFrame({"selected_channel_index": np.arange(384), "physical_channel_id": channel_map,
                      "x_um": xc, "y_um": yc}).to_csv(output / "CHANNELS.csv", index=False)
        wall = time.perf_counter() - started
        if bytes_read > int(config["logical_read_bytes_max"]) or wall > float(config["wall_seconds_max"]):
            raise RuntimeError("execution budget exceeded")
        result = {"schema": SCHEMA, "status": "complete", "completed_at": datetime.now(timezone.utc).isoformat(),
            "wall_seconds": wall, "raw_read_calls": len(batch_rows), "logical_recording_bytes_read": bytes_read,
            "retained_numeric_uncompressed_bytes": numeric_nbytes,
            "retained_numeric_file_bytes": (output / "SNIPPETS.npz").stat().st_size,
            "windows": 20, "pair_windows": 16, "single_windows": 4, "plots": 20,
            "sorts_launched": 0, "rf_accesses": 0,
            "capability": "Measured raw/pre-W/whitened capture, exact matcher reproduction, and saved-bank model fit only.",
            "interpretation": "No residual or overlay establishes biological identity, purity, spike count, or spike/non-spike truth."}
        atomic_json(output / "RESULT.json", result)
        products = ["SNIPPETS.npz", "WINDOWS.csv", "PAIR_METRICS.csv", "SINGLE_METRICS.csv",
                    "BATCH_RECEIPTS.csv", "CHANNELS.csv", "RESULT.json"]
        for plot in sorted(plots.glob("*.png")):
            products.append(str(plot.relative_to(output)))
        atomic_json(output / "MANIFEST.json", {"schema": SCHEMA, "products": {name: {
            "bytes": (output / name).stat().st_size, "sha256": sha256(output / name)} for name in products}})
        atomic_json(output / "COMPLETE.json", {"schema": SCHEMA, "status": "complete",
            "manifest_sha256": sha256(output / "MANIFEST.json"), "result_sha256": sha256(output / "RESULT.json")})
    except BaseException as exc:
        if raw_snippets is not None and np.any(initialized):
            keep = np.flatnonzero(initialized)
            partial_nbytes = (raw_snippets[keep].nbytes + prewhite_snippets[keep].nbytes
                              + white_snippets[keep].nbytes + global_axes[keep].nbytes)
            if partial_nbytes <= int(config["retained_numeric_bytes_max"]):
                partial_kwargs = {"raw": raw_snippets[keep], "prewhite": prewhite_snippets[keep],
                    "whitened": white_snippets[keep], "global_samples": global_axes[keep],
                    "window_ids": windows.iloc[keep].window_id.to_numpy(dtype="U64")}
                if channel_map is not None:
                    partial_kwargs.update(physical_channel_ids=channel_map, x_um=xc, y_um=yc)
                np.savez_compressed(output / "PARTIAL_SNIPPETS.npz", **partial_kwargs)
                windows.iloc[keep].to_csv(output / "PARTIAL_WINDOWS.csv", index=False)
                if (output / "PARTIAL_SNIPPETS.npz").stat().st_size > int(config["retained_numeric_bytes_max"]):
                    raise RuntimeError("partial snippet archive exceeds cap") from exc
        for name, rows in (("PAIR_DIAGNOSTICS.csv", pair_rows), ("SINGLE_DIAGNOSTICS.csv", single_rows),
                           ("BATCH_DIAGNOSTICS.csv", batch_rows)):
            pd.DataFrame(rows).to_csv(output / name, index=False)
        atomic_json(output / "FAILURE.json", {"schema": SCHEMA, "status": "failed_implementation",
            "failed_at": datetime.now(timezone.utc).isoformat(), "error_type": type(exc).__name__,
            "error": str(exc), "traceback": traceback.format_exc(), "logical_recording_bytes_read": bytes_read,
            "scientific_readout_allowed": False})
        raise


def preflight(config: dict[str, Any]) -> None:
    verify(config, allow_run=False)
    output = Path(config["preflight_output"])
    output.mkdir(parents=True, exist_ok=False)
    fixture = independent_fixtures()
    pairs = pd.read_csv(config["selected_pairs_path"])
    singles = pd.read_csv(config["isolated_controls_path"])
    windows = build_windows(pairs, singles, config)
    width = int(windows.width_samples.iloc[0])
    numeric_nbytes = 20 * 384 * width * (2 + 4 + 4) + 20 * width * 8
    if numeric_nbytes > int(config["retained_numeric_bytes_max"]):
        raise RuntimeError("planned retained arrays exceed cap")
    (output / "PROTOCOL.md").write_text(config["protocol"])
    atomic_json(output / "FIXTURE.json", fixture)
    atomic_json(output / "PREFLIGHT.json", {"schema": SCHEMA, "status": "ready_for_source_review",
        "source_sha256": sha256(Path(__file__)), "config_sha256": sha256(Path(config["config_path"])),
        "windows": 20, "width_samples": width, "planned_uncompressed_numeric_bytes": numeric_nbytes,
        "retained_numeric_bytes_max": int(config["retained_numeric_bytes_max"]),
        "logical_recording_read_bytes": int(config["expected_logical_read_bytes"]),
        "hub_source_reviewed": config["hub_source_reviewed"], "recording_bytes_read": 0})
    products = ["PROTOCOL.md", "FIXTURE.json", "PREFLIGHT.json"]
    atomic_json(output / "MANIFEST.json", {"schema": SCHEMA, "products": {name: {
        "bytes": (output / name).stat().st_size, "sha256": sha256(output / name)} for name in products}})
    atomic_json(output / "COMPLETE.json", {"schema": SCHEMA, "status": "ready_for_source_review",
        "manifest_sha256": sha256(output / "MANIFEST.json"), "preflight_sha256": sha256(output / "PREFLIGHT.json")})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--phase", choices=("preflight", "run"), required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    config["config_path"] = str(args.config.resolve())
    if args.phase == "preflight":
        preflight(config)
    else:
        execute(config)


if __name__ == "__main__":
    main()
