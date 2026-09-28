#!/usr/bin/env python
"""Track frozen compact-core identities in predeclared extra imec1 windows.

No motion estimate, sorter output, stimulus label, or absolute-depth gate is
used for detection or identity competition. Completed windows are hash-sealed
checkpoints and are safe to reuse after interruption.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from testing.luke_full_session_medicine import sha
from testing.luke_imec1_compact_alignment_pilot_v2 import score_global_explained
from testing.luke_imec1_dots_lighthouse_direct_check import atomic_json
from testing.luke_imec1_dots_raw_lighthouse_check import (
    MANIFEST, geometry_and_mapping, patches, preprocess, raw_sha_contract,
)
from testing.luke_imec1_dots_sorterfree_waveform_discovery import detect_waveforms
from testing.luke_imec1_dots_sorterfree_waveform_discovery_v3 import classify_scores


ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = ROOT / "testing/outputs/luke_imec1_compact_alignment_pilot_v2/frozen_training_templates.npz"
EXTERNAL = ROOT / "testing/outputs/luke_imec1_dots_sorterfree_waveform_discovery_v3"
PRIMARY_IDS = {"p08_f025", "p08_f033", "p06_f010", "p08_f029", "p06_f045", "p09_f020", "p09_f024", "p08_f044"}


def validate_checkpoint(path: Path, receipt_path: Path) -> bool:
    if not path.exists() or not receipt_path.exists():
        return False
    receipt = json.loads(receipt_path.read_text())
    return receipt.get("status") == "complete" and receipt.get("sha256") == sha(path)


def score_window(raw, geom, kept, bases, patch_channels, fs, start, stop, combined, bank, scales, external):
    signal = preprocess(raw, round(start * fs), round(stop * fs), kept, fs)
    found, waves = detect_waveforms(signal, fs, geom, bases, patch_channels)
    scored = score_global_explained(
        waves, found[:, 3].astype(int), bank, combined.phase_id.to_numpy(int), scales,
        np.load(external / "family_templates.npz")["relative_geometry"],
    )
    status = classify_scores(scored); winner = scored["winner_template"]; runner = scored["runner_template"]
    centroids = np.empty(len(waves), float)
    for event in range(len(waves)):
        block = int(found[event, 2]); energy = np.square(waves[event].astype(float)).sum(axis=0)
        centroids[event] = energy @ geom[patch_channels[block], 1] / max(energy.sum(), 1e-20)
    table = pd.DataFrame({
        "window_start_s": start, "window_stop_s": stop,
        "event_index_window": np.arange(len(waves)), "time_s": start + found[:, 0] / fs,
        "detector_phase": found[:, 3].astype(int), "peak_channel": found[:, 1].astype(int),
        "peak_snr": found[:, 4], "winner_bank": combined.bank.to_numpy()[winner],
        "winner_family": combined.family_id.to_numpy()[winner], "winner_kind": scored["winner_kind"],
        "score": scored["score"], "margin": scored["margin"], "gain": scored["gain"],
        "best_lag": scored["best_lag"], "status": status,
        "runner_bank": combined.bank.to_numpy()[runner], "runner_family": combined.family_id.to_numpy()[runner],
        "runner_kind": scored["runner_kind"], "runner_score": scored["runner_score"],
        "waveform_centroid_um": centroids,
    })
    return table


def run(config: dict) -> None:
    output = Path(config["output"]); output.mkdir(parents=True, exist_ok=True)
    settings_path = output / "settings.json"
    if settings_path.exists() and json.loads(settings_path.read_text()) != config:
        raise RuntimeError("output belongs to a different frozen configuration")
    if not settings_path.exists(): atomic_json(settings_path, config)
    manifest = json.loads(MANIFEST.read_text()); binary = Path(manifest["binary_path"]); fs = float(manifest["sampling_frequency_hz"])
    before = raw_sha_contract(binary)
    raw = np.memmap(binary, dtype="<i2", mode="r", shape=(manifest["num_samples"], manifest["saved_channels_in_binary"]))
    geom, kept = geometry_and_mapping(); bases, patch_channels = patches(geom)
    templates_path = Path(config.get("templates", TEMPLATES))
    external_path = Path(config.get("external_bank", EXTERNAL))
    with np.load(templates_path, allow_pickle=True) as saved:
        own = pd.DataFrame({"family_id": saved["family_id"].astype(str), "phase_id": saved["phase_id"], "bank": "s300"})
        own_templates = saved["compact"]; own_scales = saved["compact_scale"]
    external = pd.read_csv(external_path / "families_after_depth_reveal.csv")[["family_id", "phase_id"]]; external["bank"] = "s036"
    with np.load(external_path / "family_templates.npz") as saved:
        bank = np.concatenate([own_templates, saved["waveforms"]]); scales = np.r_[own_scales, saved["template_scale_uv"]]
    combined = pd.concat([own, external], ignore_index=True)
    elapsed = []
    for index, (start, stop) in enumerate(config["windows_s"]):
        path = output / f"window_{index:02d}_events.csv"; receipt_path = output / f"window_{index:02d}_receipt.json"
        if validate_checkpoint(path, receipt_path):
            continue
        if path.exists() or receipt_path.exists():
            raise RuntimeError(f"unsealed prior output must be preserved before restart: window {index}")
        begun = time.monotonic()
        table = score_window(raw, geom, kept, bases, patch_channels, fs, float(start), float(stop), combined, bank, scales, external_path)
        table.to_csv(path, index=False)
        duration = time.monotonic() - begun; elapsed.append(duration)
        atomic_json(receipt_path, {
            "status": "complete", "index": index, "window_s": [start, stop], "detections": len(table),
            "strict_primary": int((table.winner_bank.eq("s300") & table.winner_family.isin(PRIMARY_IDS) & table.winner_kind.eq(0) & table.status.eq("strict")).sum()),
            "sha256": sha(path), "elapsed_s": duration,
        })
    if raw_sha_contract(binary) != before: raise RuntimeError("raw source changed")
    paths = sorted(output.glob("window_*_events.csv"))
    tables = [pd.read_csv(path) for path in paths]
    combined_events = pd.concat(tables, ignore_index=True)
    primary = combined_events.loc[
        combined_events.winner_bank.eq("s300") & combined_events.winner_family.isin(PRIMARY_IDS)
        & combined_events.winner_kind.eq(0) & combined_events.status.eq("strict")
    ].copy()
    primary.to_csv(output / "strict_primary_events.csv", index=False)
    ambiguous = combined_events.loc[
        combined_events.winner_bank.eq("s300") & combined_events.winner_family.isin(PRIMARY_IDS)
        & combined_events.status.isin(["lower_score", "identity_ambiguous"])
    ]
    ambiguous.to_csv(output / "primary_uncertain_events.csv", index=False)
    summary = {
        "schema": "luke0804-imec1-lighthouse-extra-windows-v1", "status": "complete",
        "windows_s_ap_recording": config["windows_s"], "total_detections": len(combined_events),
        "strict_primary_events": len(primary), "strict_primary_families": int(primary.winner_family.nunique()),
        "uncertain_primary_events": len(ambiguous), "motion_inputs_read": False, "sorter_inputs_read": False,
        "stimulus_or_rf_outcomes_read": False, "window_receipts": {p.name: sha(p) for p in sorted(output.glob("window_*_receipt.json"))},
        "new_runtime_s": elapsed, "interpretation": "Fresh waveform-only validation windows; identity competition is independent of every evaluated motion field.",
    }
    atomic_json(output / "summary.json", summary)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(); run(json.loads(args.config.read_text()))


if __name__ == "__main__": main()
