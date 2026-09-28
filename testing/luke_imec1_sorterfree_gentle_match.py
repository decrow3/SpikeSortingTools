#!/usr/bin/env python
"""Apply the frozen sorter-free waveform bank to preselected gentle windows."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from testing.luke_imec1_dots_lighthouse_direct_check import atomic_json, sha256
from testing.luke_imec1_dots_raw_lighthouse_check import MANIFEST, geometry_and_mapping, patches, preprocess, raw_sha_contract
from testing.luke_imec1_dots_sorterfree_waveform_discovery import detect_waveforms, match_interval

SCHEMA = "luke0804-imec1-sorterfree-gentle-match-v1"


def run(config: dict) -> None:
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "settings.json", config)
    if config.get("dummy"):
        time.sleep(20)
        atomic_json(output / "summary.json", {"schema": SCHEMA, "status": "complete", "dummy": True})
        return
    discovery = Path(config["frozen_discovery"])
    source_summary = json.loads((discovery / "summary.json").read_text())
    if source_summary.get("sorter_inputs_read") is not False or source_summary.get("motion_estimate_read") is not False:
        raise RuntimeError("Frozen discovery does not satisfy sorter/motion-free contract")
    families = pd.read_csv(discovery / "families_after_depth_reveal.csv")
    templates = np.load(discovery / "family_templates.npz")["waveforms"]
    if len(families) != len(templates) or int(families.selected_depth_blind.sum()) != 50:
        raise RuntimeError("Expected the frozen 50-candidate waveform bank")
    manifest = json.loads(MANIFEST.read_text())
    binary = Path(manifest["binary_path"])
    fs = float(manifest["sampling_frequency_hz"])
    before = raw_sha_contract(binary)
    raw = np.memmap(binary, dtype="<i2", mode="r", shape=(manifest["num_samples"], manifest["saved_channels_in_binary"]))
    geom, kept_raw = geometry_and_mapping()
    bases, patch_channels = patches(geom)
    parts = []
    counts = []
    for index, (start, stop) in enumerate(config["gentle_windows_s"]):
        x = preprocess(raw, round(start * fs), round(stop * fs), kept_raw, fs)
        found, waves = detect_waveforms(x, fs, geom, bases, patch_channels)
        matched, detail = match_interval(x, found, waves, start, fs, geom, patch_channels, families, templates)
        path = output / f"gentle_{index:02d}_events.csv"
        matched.to_csv(path, index=False)
        atomic_json(output / f"gentle_{index:02d}.complete.json", {"sha256": sha256(path), **detail})
        parts.append(matched)
        counts.append({"start_s": start, "stop_s": stop, **detail})
        print(index, start, stop, detail, flush=True)
    events = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    events.to_csv(output / "gentle_events.csv", index=False)
    pd.DataFrame(counts).to_csv(output / "gentle_counts.csv", index=False)
    if raw_sha_contract(binary) != before:
        raise RuntimeError("Raw source changed")
    strict = events[events.status.eq("strict")] if len(events) else events
    atomic_json(output / "summary.json", {
        "schema": SCHEMA, "status": "complete", "sorter_inputs_read": False,
        "motion_estimate_used_for_identity_or_matching": False,
        "motion_estimate_used_only_to_preselect_gentle_windows": True,
        "gentle_windows_s": config["gentle_windows_s"],
        "strict_events": len(strict), "strict_families": int(strict.family_id.nunique()) if len(strict) else 0,
        "source_discovery": str(discovery.resolve()),
    })


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--dummy", action="store_true")
    args = ap.parse_args()
    config = json.loads(args.config.read_text())
    config["dummy"] = args.dummy
    run(config)


if __name__ == "__main__":
    main()
