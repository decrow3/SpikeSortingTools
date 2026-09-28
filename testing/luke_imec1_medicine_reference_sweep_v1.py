#!/usr/bin/env python
"""Reference-first, no-sort MEDiCINe sensitivity sweep for Luke0804 imec1.

The reference phase is deliberately separate from extraction/fitting.  It reads
the frozen rescue 12/9 motion-off sort and raw AP voltage, but no candidate
MEDiCINe field.  The sweep reuses sealed pilot peak populations where possible,
fits estimator fields only, and never modifies voltage or launches a sort.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import zipfile

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OUT = ROOT / "testing/outputs/luke_imec1_medicine_reference_sweep_v1"
PILOT = ROOT / "testing/outputs/cross_dataset_fast_motion_v2"
STATIC = Path(
    "/mnt/NPX/Luke/20250804/shared_analysis/luke_group1_handoff_v1/"
    "arms/rescue_12_9_motion_off"
)
DARTSORT = Path("/home/huklab/Documents/DARTsort")
PILOT_DARTSORT_SOURCE = Path(os.environ.get("LUKE_PILOT_DARTSORT_SOURCE", "/tmp/dartsort-pilot-source"))
DSPY = DARTSORT / ".venv/bin/python"
MEDPY = DARTSORT / "environments/medicine-estimator/.venv/bin/python"
TEMPLATES = ROOT / "testing/outputs/luke_imec1_compact_alignment_pilot_v2/frozen_training_templates.npz"
FAMILIES = ROOT / "testing/outputs/luke_imec1_dots_sorterfree_waveform_discovery_v3_s300/families_after_depth_reveal.csv"
EXTERNAL = ROOT / "testing/outputs/luke_imec1_dots_sorterfree_waveform_discovery_v3"
FS_STATIC = 29999.835983263598
BIN = 0.25
FIT_TIMEOUT_S = 900.0
EXTRACTION_TIMEOUT_S = 3600.0
GPU_BUDGET_S = 8.0 * 3600.0

# Frozen from the static-sort raster/raw-depth view before any new fit.
WINDOWS = {
    "p10": (987.3553929363488, 1107.3553896029887),
    "p90": (9366.198489760098, 9486.198486426738),
    "w8160": (8160.0, 8280.0),
    "gaborA": (2091.25, 2211.25),
    "gaborB": (2211.25, 2331.25),
    "p50": (5176.776924681423, 5296.776921348063),
}
EPISODES = {
    "p10": [(997.0, 1004.5), (1014.0, 1021.5), (1031.0, 1037.5),
             (1071.0, 1077.5), (1097.5, 1105.5)],
    "p90": [(9377.0, 9383.0), (9395.5, 9401.5), (9432.5, 9439.0),
             (9447.5, 9453.5), (9468.5, 9475.0)],
    "w8160": [(8178.0, 8186.0), (8212.5, 8219.5), (8220.0, 8227.0),
              (8241.5, 8248.5), (8262.0, 8268.0)],
}
REAL = tuple(EPISODES)
PAUSE = ("gaborA", "gaborB")
PRIMARY_IDS = {
    "p08_f025", "p08_f033", "p06_f010", "p08_f029",
    "p06_f045", "p09_f020", "p09_f024", "p08_f044",
}
BASE_MED = dict(
    motion_bound=500.0, time_bin_size=0.25, time_kernel_width=1.0,
    num_depth_bins=4, amplitude_threshold_quantile=0.0,
    training_steps=10000, batch_size=4096,
    activity_network_hidden_features=(256, 256), learning_rate=5e-4,
    initial_motion_noise=0.1, motion_noise_steps=2000, epsilon=0.001,
    plot_figures=False,
)
CONFIGS = {
    "base": {}, "seed1": {"seed": 1},
    "kern0p5": {"time_kernel_width": 0.5}, "kern2": {"time_kernel_width": 2.0},
    "kern5": {"time_kernel_width": 5.0}, "kern10": {"time_kernel_width": 10.0},
    "kern30": {"time_kernel_width": 30.0},
    "depth2": {"num_depth_bins": 2}, "depth8": {"num_depth_bins": 8},
    "steps30k": {"training_steps": 30000},
    "amp50": {"amplitude_threshold_quantile": 0.50},
    "amp25": {"amplitude_threshold_quantile": 0.75},
    "rateeq": {"rate_equalize": True},
    "win60": {"fit_duration_s": 60.0},
    "win240": {"fit_duration_s": 240.0},
    "crop": {"depth_crop_um": [2020.0, 3820.0]},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def interval_mask(values: np.ndarray, intervals: list[tuple[float, float]]) -> np.ndarray:
    result = np.zeros(values.shape, dtype=bool)
    for start, stop in intervals:
        result |= (values >= start) & (values < stop)
    return result


def erode_boolean(mask: np.ndarray, bins_each_side: int) -> np.ndarray:
    """Symmetric erosion with false padding; kept explicit for testability."""
    mask = np.asarray(mask, dtype=bool)
    result = mask.copy()
    for shift in range(1, bins_each_side + 1):
        left = np.r_[np.zeros(shift, bool), mask[:-shift]]
        right = np.r_[mask[shift:], np.zeros(shift, bool)]
        result &= left & right
    return result


def contiguous_intervals(edges: np.ndarray, mask: np.ndarray) -> list[tuple[float, float]]:
    padded = np.r_[False, np.asarray(mask, bool), False].astype(np.int8)
    changes = np.diff(padded)
    starts, stops = np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)
    return [(float(edges[a]), float(edges[b])) for a, b in zip(starts, stops, strict=True)]


def static_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cur = STATIC / "cur/cur_output"
    times = np.load(cur / "spike_times.npy", mmap_mode="r") / FS_STATIC
    clusters = np.load(cur / "spike_clusters.npy", mmap_mode="r")
    depths = np.load(cur / "spike_positions.npy", mmap_mode="r")[:, 1]
    amplitudes = np.load(cur / "full_st.npy", mmap_mode="r")[:, 2]
    if not (len(times) == len(clusters) == len(depths) == len(amplitudes)):
        raise RuntimeError("Frozen static-sort arrays have changed grain")
    return times, clusters, depths, amplitudes


def ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else np.nan


def summarize_static_reference(output: Path) -> dict:
    import pandas as pd
    times, clusters, depths, amplitudes = static_arrays()
    unit_rows: list[dict] = []
    top_rows: list[dict] = []
    pause_rows: list[dict] = []
    pause_bins: list[dict] = []
    for window, (start, stop) in WINDOWS.items():
        ix = np.flatnonzero((times >= start) & (times < stop))
        local_t, local_c = times[ix], clusters[ix]
        ids, counts = np.unique(local_c, return_counts=True)
        order = ids[np.argsort(counts)[::-1]]
        if window in REAL:
            episodes = EPISODES[window]
            core_all = interval_mask(local_t, episodes)
            core_duration = sum(b - a for a, b in episodes)
            rest_duration = (stop - start) - core_duration
            for rank, unit in enumerate(order[:10], 1):
                ui = ix[local_c == unit]
                core = interval_mask(times[ui], episodes)
                home_depth = float(np.median(depths[ui][~core])) if (~core).any() else np.nan
                top_rows.append({"window": window, "rate_rank": rank, "unit_id": int(unit),
                                 "spikes": int(len(ui)), "home_depth_um": home_depth})
            for episode_index, (a, b) in enumerate(episodes):
                for unit in ids:
                    ui = ix[local_c == unit]
                    core = (times[ui] >= a) & (times[ui] < b)
                    rest = ~core_all[local_c == unit]
                    if not core.any() or not rest.any():
                        continue
                    unit_rows.append({
                        "window": window, "episode_index": episode_index,
                        "episode_start_s": a, "episode_stop_s": b, "unit_id": int(unit),
                        "core_spikes": int(core.sum()), "rest_spikes": int(rest.sum()),
                        "raw_depth_shift_um": float(np.median(depths[ui][core]) - np.median(depths[ui][rest])),
                        "amplitude_ratio": ratio(float(np.median(amplitudes[ui][core])), float(np.median(amplitudes[ui][rest]))),
                        "rate_ratio": ratio(core.sum() / (b - a), rest.sum() / rest_duration),
                    })
        if window in PAUSE:
            edges = np.arange(start, stop + BIN + 1e-9, BIN)
            counts = np.histogram(local_t, edges)[0].astype(float)
            smooth = np.convolve(counts, np.ones(4) / 4.0, mode="same")
            low = smooth < 0.5 * np.median(smooth)
            core_mask = erode_boolean(low, 2)
            pauses = contiguous_intervals(edges, core_mask)
            for bin_index, keep in enumerate(core_mask):
                pause_bins.append({"window": window, "bin_index": bin_index,
                                   "start_s": edges[bin_index], "stop_s": edges[bin_index + 1],
                                   "count": counts[bin_index], "smoothed_count": smooth[bin_index],
                                   "pause_core": bool(keep)})
            is_pause = interval_mask(local_t, pauses)
            pause_duration = float(core_mask.sum() * BIN)
            rest_duration = (stop - start) - pause_duration
            for unit in ids:
                ui = ix[local_c == unit]
                core = interval_mask(times[ui], pauses)
                rest = ~core
                if not core.any() or not rest.any():
                    continue
                pause_rows.append({
                    "window": window, "unit_id": int(unit), "pause_spikes": int(core.sum()),
                    "rest_spikes": int(rest.sum()),
                    "raw_depth_shift_um": float(np.median(depths[ui][core]) - np.median(depths[ui][rest])),
                    "amplitude_ratio": ratio(float(np.median(amplitudes[ui][core])), float(np.median(amplitudes[ui][rest]))),
                    "rate_ratio": ratio(core.sum() / pause_duration, rest.sum() / rest_duration),
                })
    pd.DataFrame(unit_rows).to_csv(output / "references/static_episode_unit_metrics.csv", index=False)
    pd.DataFrame(top_rows).to_csv(output / "references/top10_units.csv", index=False)
    pd.DataFrame(pause_rows).to_csv(output / "references/pause_unit_metrics.csv", index=False)
    pd.DataFrame(pause_bins).to_csv(output / "references/pause_bins.csv", index=False)
    return {"episode_unit_rows": len(unit_rows), "pause_unit_rows": len(pause_rows),
            "top10_rows": len(top_rows), "pause_core_bins": int(sum(x["pause_core"] for x in pause_bins))}


def scan_lighthouse(output: Path) -> dict:
    """Whole-probe waveform discovery on episode cores and the quiet window."""
    import pandas as pd
    from testing.luke_imec1_compact_alignment_pilot_v2 import score_global_explained
    from testing.luke_imec1_dots_raw_lighthouse_check import (
        MANIFEST, geometry_and_mapping, patches, preprocess, raw_sha_contract,
    )
    from testing.luke_imec1_dots_sorterfree_waveform_discovery import detect_waveforms

    manifest = json.loads(MANIFEST.read_text())
    binary = Path(manifest["binary_path"])
    fs = float(manifest["sampling_frequency_hz"])
    before = raw_sha_contract(binary)
    raw = np.memmap(binary, dtype="<i2", mode="r",
                    shape=(manifest["num_samples"], manifest["saved_channels_in_binary"]))
    geom, kept = geometry_and_mapping()
    bases, patch_channels = patches(geom)
    with np.load(TEMPLATES, allow_pickle=True) as saved:
        own = pd.DataFrame({"family_id": saved["family_id"].astype(str),
                            "phase_id": saved["phase_id"], "bank": "s300"})
        own_templates, own_scales = saved["compact"], saved["compact_scale"]
    external_families = pd.read_csv(EXTERNAL / "families_after_depth_reveal.csv")[["family_id", "phase_id"]]
    external_families["bank"] = "s036"
    with np.load(EXTERNAL / "family_templates.npz") as saved:
        external_templates = saved["waveforms"]
        external_scales = saved["template_scale_uv"]
        relative_geometry = saved["relative_geometry"]
    combined = pd.concat([own, external_families], ignore_index=True)
    bank = np.concatenate([own_templates, external_templates])
    scales = np.r_[own_scales, external_scales]
    reference_depth = pd.read_csv(FAMILIES).set_index("family_id").seed_depth_median_um
    event_rows: list[dict] = []
    chunks = [(w, i, a, b) for w in REAL for i, (a, b) in enumerate(EPISODES[w])]
    chunks.append(("p50", 0, WINDOWS["p50"][0], WINDOWS["p50"][1]))
    runtimes = []
    for window, episode_index, start, stop in chunks:
        begun = time.monotonic()
        signal = preprocess(raw, round(start * fs), round(stop * fs), kept, fs)
        found, waves = detect_waveforms(signal, fs, geom, bases, patch_channels)
        scored = score_global_explained(waves, found[:, 3].astype(int), bank,
                                        combined.phase_id.to_numpy(int), scales,
                                        relative_geometry)
        winner = scored["winner_template"]
        centroids = np.empty(len(waves), float)
        for event in range(len(waves)):
            block = int(found[event, 2])
            energy = np.square(waves[event].astype(float)).sum(axis=0)
            centroids[event] = energy @ geom[patch_channels[block], 1] / max(energy.sum(), 1e-20)
        for event in range(len(waves)):
            family = str(combined.family_id.iloc[winner[event]])
            is_primary = (combined.bank.iloc[winner[event]] == "s300" and family in PRIMARY_IDS
                          and scored["winner_kind"][event] == 0)
            if not is_primary:
                continue
            displacement = centroids[event] - float(reference_depth.loc[family])
            event_rows.append({
                "window": window, "episode_index": episode_index,
                "episode_start_s": start, "episode_stop_s": stop,
                "time_s": start + found[event, 0] / fs, "family_id": family,
                "waveform_centroid_um": centroids[event], "reference_depth_um": float(reference_depth.loc[family]),
                "displacement_um": displacement, "cosine": float(scored["score"][event]),
                "margin": float(scored["margin"][event]), "gain": float(scored["gain"][event]),
                "direction": "target_negative" if -320.0 <= displacement <= 0.0 else
                             "opposite_positive" if 0.0 < displacement <= 320.0 else "outside",
            })
        runtimes.append({"window": window, "episode_index": episode_index,
                         "start_s": start, "stop_s": stop, "detections": len(found),
                         "runtime_s": time.monotonic() - begun})
    events = pd.DataFrame(event_rows)
    events.to_csv(output / "references/lighthouse_events.csv", index=False)
    tracks = []
    qualified = events.loc[events.cosine >= 0.9].copy()
    qualified["bin_start_s"] = np.floor(qualified.time_s / BIN) * BIN
    for (window, episode_index, family, direction, bin_start), group in qualified.groupby(
        ["window", "episode_index", "family_id", "direction", "bin_start_s"]
    ):
        if direction == "outside" or len(group) < 2:
            continue
        tracks.append({"window": window, "episode_index": int(episode_index),
                       "family_id": family, "direction": direction,
                       "bin_start_s": float(bin_start), "bin_center_s": float(bin_start + BIN / 2),
                       "matches": int(len(group)), "best_cosine": float(group.cosine.max()),
                       "median_cosine": float(group.cosine.median()),
                       "displacement_um": float(group.displacement_um.median())})
    track_table = pd.DataFrame(tracks)
    track_table.to_csv(output / "references/lighthouse_tracks_250ms.csv", index=False)
    real_tracks = track_table.loc[(track_table.window.isin(REAL)) & track_table.direction.eq("target_negative")]
    quiet = track_table.loc[track_table.window.eq("p50") & track_table.direction.ne("outside")].copy()
    quiet["bin5"] = np.floor((quiet.bin_center_s - WINDOWS["p50"][0]) / 5.0).astype(int)
    levels = quiet.groupby(["family_id", "bin5"]).displacement_um.median().reset_index()
    increments = []
    for family, table in levels.groupby("family_id"):
        table = table.sort_values("bin5")
        for left, right in zip(table.iloc[:-1].itertuples(), table.iloc[1:].itertuples()):
            if right.bin5 == left.bin5 + 1:
                increments.append({"family_id": family, "bin_from": int(left.bin5),
                                   "bin_to": int(right.bin5),
                                   "lighthouse_delta_um": float(right.displacement_um - left.displacement_um)})
    pd.DataFrame(increments).to_csv(output / "references/quiet_lighthouse_increments.csv", index=False)
    if raw_sha_contract(binary) != before:
        raise RuntimeError("Raw AP source changed during read-only lighthouse scan")
    write_csv(output / "references/lighthouse_scan_runtimes.csv", runtimes)
    episode_summary = real_tracks.groupby(["window", "episode_index"]).displacement_um.median().rename(
        "median_displacement_um").reset_index()
    episode_summary.to_csv(output / "references/real_episode_summary.csv", index=False)
    return {"events": len(events), "real_tracks": int(real_tracks.family_id.nunique()),
            "real_bins": len(real_tracks), "opposite_bins": int(((track_table.window.isin(REAL)) &
            track_table.direction.eq("opposite_positive")).sum()), "quiet_increments": len(increments),
            "scan_runtime_s": float(sum(x["runtime_s"] for x in runtimes))}


def plot_reference_rasters(output: Path) -> None:
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    times, _, depths, _ = static_arrays()
    pause = pd.read_csv(output / "references/pause_bins.csv")
    for window, (start, stop) in WINDOWS.items():
        selected = np.flatnonzero((times >= start) & (times < stop))
        stride = max(1, len(selected) // 180000)
        fig, ax = plt.subplots(figsize=(15, 5), layout="constrained")
        chosen = selected[::stride]
        ax.scatter(times[chosen], depths[chosen], s=0.15, c="#303030", alpha=0.2, rasterized=True)
        if window in REAL:
            for a, b in EPISODES[window]:
                ax.axvspan(a, b, color="#EE7733", alpha=0.13, linewidth=0)
        elif window in PAUSE:
            for row in pause.loc[(pause.window == window) & pause.pause_core].itertuples():
                ax.axvspan(row.start_s, row.stop_s, color="#0077BB", alpha=0.14, linewidth=0)
        ax.set(xlim=(start, stop), ylim=(0, 3850), xlabel="Session time (s)", ylabel="Raw spike depth (µm)",
               title=f"{window}: frozen rescue 12/9 static-sort raster")
        fig.savefig(output / f"references/{window}_reference_raster.png", dpi=170)
        plt.close(fig)


def prepare_references(output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"Refusing existing reference output: {output}")
    (output / "references").mkdir(parents=True)
    prereg = {
        "schema": "luke0804-imec1-medicine-reference-sweep-v1",
        "status": "reference_design_frozen_before_new_sweep",
        "windows": WINDOWS, "episodes": EPISODES,
        "pause_definition": "population count <0.5x window median after centered 1s boxcar smoothing; 0.5s symmetric erosion",
        "lighthouse_definition": "whole-probe waveform identity competition; cosine>=0.9; >=2 matches per family/0.25s bin; target 0 to -320um; positive direction retained as control",
        "quiet_definition": "zero motion plus 5s adjacent lighthouse-increment RMS wherever adjacent family bins exist",
        "selection_rule": "pause_abs<=10um and quiet_inc<=5um; then lowest pooled motion_rmse if 0.8<=motion_ratio<=1.2; otherwise no selection and report Pareto front",
        "configs": CONFIGS, "fit_timeout_s": FIT_TIMEOUT_S,
        "extraction_timeout_s": EXTRACTION_TIMEOUT_S, "gpu_budget_s": GPU_BUDGET_S,
        "static_sort": str(STATIC), "static_sort_identity": json.loads((STATIC / "sort_identity.json").read_text()),
        "tools_planned": ["cross_dataset_fast_motion.py", "luke_imec1_compact_alignment_pilot_v2.py",
                          "luke_imec1_cross_seed_replication_v1.py", "luke_imec1_lighthouse_motion_comparison_v1.py",
                          "motion_saved_array_audit_20260923/quiet_motion.csv", "pipeline/unit_quality.py",
                          "fit_matched_medicine.py", "qualify_local_motion_fields.py", "external_medicine_motion.py"],
        "no_sort": True, "voltage_modified": False,
    }
    atomic_json(output / "reference_preregistration.json", prereg)
    static_summary = summarize_static_reference(output)
    plot_reference_rasters(output)
    lighthouse_summary = scan_lighthouse(output)
    summary = {"status": "complete", "static": static_summary, "lighthouse": lighthouse_summary,
               "preregistration_sha256": sha256(output / "reference_preregistration.json")}
    atomic_json(output / "references/summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def prepare_extraction_config(output: Path) -> None:
    source = json.loads((PILOT / "config.json").read_text())
    if PILOT_DARTSORT_SOURCE != DARTSORT:
        if not (PILOT_DARTSORT_SOURCE / "src/dartsort").is_dir():
            raise FileNotFoundError(f"Missing sealed pilot source worktree: {PILOT_DARTSORT_SOURCE}")
        source["dartsort_source_sha256"] = {
            str(PILOT_DARTSORT_SOURCE / Path(path).relative_to(DARTSORT)): digest
            for path, digest in source["dartsort_source_sha256"].items()
        }
        source["dartsort_source_root"] = str(PILOT_DARTSORT_SOURCE)
    record = next(x for x in source["records"] if x["dataset"] == "Luke" and x["probe"] == "imec1")
    fs = float(record["sampling_frequency_hz"])
    requested = []
    for window, (start, stop) in WINDOWS.items():
        for duration in (120, 240):
            if duration == 120 and window in {"p10", "p90", "p50"}:
                continue
            center = (start + stop) / 2
            a, b = center - duration / 2, center + duration / 2
            requested.append({"id": f"{window}_{duration}", "dataset": "Luke", "probe": "imec1",
                              "fraction": None, "start_frame": round(a * fs), "stop_frame": round(b * fs),
                              "start_s": round(a * fs) / fs, "stop_s": round(b * fs) / fs})
    source["windows"] = requested
    source["purpose"] = "Peak cache for reference-first MEDiCINe sweep; no sorting or voltage modification"
    source["parent_config"] = str(PILOT / "config.json")
    source["parent_config_sha256"] = sha256(PILOT / "config.json")
    atomic_json(output / "extraction_config.json", source)


def extract_one(output: Path, window: str) -> None:
    import h5py

    from testing.cross_dataset_fast_motion import extract, seal
    config = json.loads((output / "extraction_config.json").read_text())
    spec = next(x for x in config["windows"] if x["id"] == window)
    stage = output / "peak_cache" / window / "extraction"
    extract(output / "peak_cache", config, spec)
    subtraction = stage / "detection/subtraction.h5"
    with h5py.File(subtraction, "r") as h:
        samples = h["times_samples"][:]
        locations = h["point_source_localizations"][:]
        amplitudes = np.abs(h["denoised_ptp_amplitudes"][:])
        fs = float(h["sampling_frequency"][()])
    valid = np.isfinite(locations[:, 0]) & np.isfinite(locations[:, 2])
    valid &= np.isfinite(amplitudes) & (amplitudes > 0)
    population = np.load(stage / "population.npz", allow_pickle=False)
    if not np.allclose(population["time_s"], samples[valid] / fs):
        raise RuntimeError("Serialized population does not match subtraction peaks")
    augmented = stage / "population.with_x.npz"
    np.savez_compressed(
        augmented,
        time_s=population["time_s"], depth_um=population["depth_um"],
        x_um=locations[valid, 0], amplitude=population["amplitude"],
    )
    os.replace(augmented, stage / "population.npz")
    # population.npz is the reusable peak cache. Retain the large subtraction
    # product's sealed digest and size, but not the redundant waveform HDF5.
    original_seal = json.loads((stage / "complete.json").read_text())
    audit_path = stage / "audit.json"
    audit = json.loads(audit_path.read_text())
    audit["discarded_intermediate"] = {
        "path": "detection/subtraction.h5",
        "bytes": subtraction.stat().st_size,
        "sha256": original_seal["detection/subtraction.h5"],
        "reason": "population.npz is the reusable fit input; retain space for all windows",
    }
    atomic_json(audit_path, audit)
    shutil.rmtree(stage / "detection")
    seal(stage)


def population_source(output: Path, window: str, duration: int) -> tuple[Path, float]:
    if duration == 120 and window in {"p10", "p90", "p50"}:
        mapping = {"p10": "luke_imec1_p10", "p90": "luke_imec1_p90", "p50": "luke_imec1_p50"}
        root = PILOT / mapping[window]
        receipt = root / "extraction/complete.json"
        if not receipt.exists():
            raise FileNotFoundError(receipt)
        start = json.loads((PILOT / "config.json").read_text())
        start_s = next(w["start_s"] for w in start["windows"] if w["id"] == mapping[window])
        return root / "extraction/population.npz", float(start_s)
    config = json.loads((output / "extraction_config.json").read_text())
    spec = next(w for w in config["windows"] if w["id"] == f"{window}_{duration}")
    return output / f"peak_cache/{window}_{duration}/extraction/population.npz", float(spec["start_s"])


def filtered_population(output: Path, config_id: str, window: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, dict]:
    change = dict(CONFIGS[config_id])
    duration = int(change.pop("fit_duration_s", 120))
    path, fit_start_s = population_source(output, window, 240 if duration == 240 else 120)
    with np.load(path) as data:
        times, depths, amps = (np.asarray(data[k]) for k in ("time_s", "depth_um", "amplitude"))
    if duration == 60:
        keep = (times >= 30.0) & (times < 90.0)
        times, depths, amps = times[keep] - 30.0, depths[keep], amps[keep]
        fit_start_s += 30.0
    if "depth_crop_um" in change:
        low, high = change.pop("depth_crop_um")
        keep = (depths >= low) & (depths <= high)
        times, depths, amps = times[keep], depths[keep], amps[keep]
    if change.pop("rate_equalize", False):
        rng = np.random.default_rng(0)
        bins = np.floor(times / BIN).astype(int)
        _, counts = np.unique(bins, return_counts=True)
        cap = int(np.median(counts))
        selected = []
        for value in np.unique(bins):
            rows = np.flatnonzero(bins == value)
            selected.extend(rng.choice(rows, size=min(cap, len(rows)), replace=False))
        selected = np.sort(np.asarray(selected, int))
        times, depths, amps = times[selected], depths[selected], amps[selected]
    return times, depths, amps, fit_start_s, change


def fit_one(output: Path, config_id: str, window: str) -> None:
    import torch
    import medicine
    target = output / f"fields/{config_id}/{window}"
    target.mkdir(parents=True, exist_ok=False)
    times, depths, amps, fit_start_s, change = filtered_population(output, config_id, window)
    seed = int(change.pop("seed", 0))
    settings = dict(BASE_MED)
    settings.update(change)
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    begun = time.monotonic()
    trainer = medicine.run_medicine(peak_times=times, peak_depths=depths,
                                    peak_amplitudes=amps, output_dir=target / "medicine",
                                    optimizer=torch.optim.Adam, **settings)
    runtime = time.monotonic() - begun
    medicine_dir = target / "medicine"
    field = {
        "time_s": np.load(medicine_dir / "time_bins.npy"),
        "depth_um": np.load(medicine_dir / "depth_bins.npy"),
        "displacement_um": np.load(medicine_dir / "motion.npy"),
    }
    np.savez_compressed(target / "field.npz", **field,
                        session_time_s=field["time_s"] + fit_start_s,
                        sign_contract=np.asarray("observed_minus_registered; corrected=observed-displacement"))
    np.save(target / "loss.npy", np.asarray(trainer.losses))
    atomic_json(target / "receipt.json", {
        "status": "complete", "config": config_id, "window": window,
        "runtime_s": runtime, "fit_start_s": fit_start_s, "peaks": len(times),
        "settings": settings, "seed": seed, "source_population": str(population_source(
            output, window, 240 if CONFIGS[config_id].get("fit_duration_s") == 240 else 120)[0]),
        "field_sha256": sha256(target / "field.npz"), "no_sort": True, "voltage_modified": False,
    })


def run_sweep(output: Path) -> None:
    if json.loads((output / "references/summary.json").read_text()).get("status") != "complete":
        raise RuntimeError("Reference phase is not complete")
    prepare_extraction_config(output)
    extraction_cfg = json.loads((output / "extraction_config.json").read_text())
    for window in extraction_cfg["windows"]:
        stage = output / f"peak_cache/{window['id']}/extraction"
        if (stage / "complete.json").exists():
            continue
        command = [str(DSPY), str(Path(__file__).resolve()), "extract-one", "--output", str(output),
                   "--window", window["id"]]
        result = subprocess.run(command, timeout=EXTRACTION_TIMEOUT_S)
        if result.returncode:
            raise RuntimeError(f"Extraction failed: {window['id']}")
    extraction_runtime = 0.0
    for window in extraction_cfg["windows"]:
        audit = output / f"peak_cache/{window['id']}/extraction/audit.json"
        extraction_runtime += float(json.loads(audit.read_text())["seconds"])
    runtime_total = extraction_runtime
    completed = []
    for config_id in CONFIGS:
        for window in WINDOWS:
            target = output / f"fields/{config_id}/{window}/receipt.json"
            if target.exists():
                receipt = json.loads(target.read_text())
                runtime_total += float(receipt["runtime_s"])
                completed.append(receipt)
                continue
            if runtime_total >= GPU_BUDGET_S:
                atomic_json(output / "stopped.json", {"reason": "gpu_budget", "runtime_s": runtime_total})
                raise RuntimeError("Stopped at the 8 GPU-hour budget")
            command = [str(MEDPY), str(Path(__file__).resolve()), "fit-one", "--output", str(output),
                       "--config-id", config_id, "--window", window]
            begun = time.monotonic()
            try:
                result = subprocess.run(command, timeout=FIT_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                atomic_json(output / "stopped.json", {"reason": "single_fit_timeout", "config": config_id,
                            "window": window, "limit_s": FIT_TIMEOUT_S, "elapsed_s": time.monotonic() - begun})
                raise RuntimeError(f"Stopped: {config_id}/{window} exceeded 15 minutes")
            if result.returncode:
                atomic_json(output / "stopped.json", {"reason": "fit_failure", "config": config_id,
                            "window": window, "returncode": result.returncode})
                raise RuntimeError(f"Fit failed: {config_id}/{window}")
            receipt = json.loads(target.read_text())
            runtime_total += float(receipt["runtime_s"])
            completed.append(receipt)
    atomic_json(output / "sweep_complete.json", {"status": "complete", "fits": len(completed),
                "fit_runtime_s": runtime_total - extraction_runtime,
                "extraction_walltime_counted_conservatively_s": extraction_runtime,
                "budget_accounted_runtime_s": runtime_total, "budget_s": GPU_BUDGET_S})


def sample_field(field: dict, times: np.ndarray, depths: np.ndarray) -> np.ndarray:
    """Bilinear interpolation, returning NaN outside support."""
    source_t = np.asarray(field["session_time_s"], float)
    source_z = np.asarray(field["depth_um"], float)
    values = np.asarray(field["displacement_um"], float)
    result = np.full(np.broadcast(times, depths).shape, np.nan)
    bt, bz = np.broadcast_arrays(times, depths)
    valid = ((bt >= source_t[0]) & (bt <= source_t[-1]) &
             (bz >= source_z[0]) & (bz <= source_z[-1]))
    for index in np.flatnonzero(valid.ravel()):
        t, z = bt.ravel()[index], bz.ravel()[index]
        by_depth = np.asarray([np.interp(t, source_t, values[:, j]) for j in range(len(source_z))])
        result.ravel()[index] = np.interp(z, source_z, by_depth)
    return result


def centered_samples(field: dict, sample_t: np.ndarray, sample_z: np.ndarray,
                     anchor_start: float) -> np.ndarray:
    prediction = sample_field(field, sample_t, sample_z)
    baseline_t = np.arange(anchor_start - 4.0, anchor_start, BIN) + BIN / 2
    baseline = np.asarray([sample_field(field, baseline_t, np.full(len(baseline_t), z)) for z in sample_z])
    offsets = np.nanmedian(baseline, axis=1)
    return prediction - offsets


def score_field(output: Path, config_id: str, window: str) -> dict:
    import pandas as pd
    with np.load(output / f"fields/{config_id}/{window}/field.npz", allow_pickle=False) as data:
        field = {key: np.asarray(data[key]) for key in ("time_s", "session_time_s", "depth_um", "displacement_um")}
    receipt = json.loads((output / f"fields/{config_id}/{window}/receipt.json").read_text())
    row = {"config": config_id, "window": window, "runtime_s": receipt["runtime_s"],
           "motion_rmse": np.nan, "motion_ratio": np.nan, "motion_reference_bins": 0,
           "pause_abs": np.nan, "pause_samples": 0, "quiet_inc": np.nan, "quiet_increments": 0,
           "seed_delta": np.nan}
    if window in REAL:
        tracks = pd.read_csv(output / "references/lighthouse_tracks_250ms.csv")
        tracks = tracks.loc[(tracks.window == window) & tracks.direction.eq("target_negative")]
        observed, predicted = [], []
        for episode_index, group in tracks.groupby("episode_index"):
            start = EPISODES[window][int(episode_index)][0]
            values = centered_samples(field, group.bin_center_s.to_numpy(float),
                                      group.reference_depth_um.to_numpy(float) if "reference_depth_um" in group else
                                      np.full(len(group), np.nan), start)
            # Older/compact track tables carry family but not depth; recover it here.
            if not np.isfinite(values).any():
                depth_map = pd.read_csv(FAMILIES).set_index("family_id").seed_depth_median_um
                z = group.family_id.map(depth_map).to_numpy(float)
                values = centered_samples(field, group.bin_center_s.to_numpy(float), z, start)
            good = np.isfinite(values) & np.isfinite(group.displacement_um.to_numpy(float))
            observed.extend(group.displacement_um.to_numpy(float)[good])
            predicted.extend(values[good])
        observed, predicted = np.asarray(observed), np.asarray(predicted)
        if len(observed):
            row["motion_rmse"] = float(np.sqrt(np.mean((predicted - observed) ** 2)))
            valid_ratio = np.abs(observed) >= 10.0
            row["motion_ratio"] = float(np.median(predicted[valid_ratio] / observed[valid_ratio])) if valid_ratio.any() else np.nan
            row["motion_reference_bins"] = int(len(observed))
    elif window in PAUSE:
        bins = pd.read_csv(output / "references/pause_bins.csv")
        bins = bins.loc[(bins.window == window) & bins.pause_core]
        depths = np.asarray([2400.0, 2800.0, 3200.0, 3600.0])
        values = []
        intervals = contiguous_intervals(np.r_[bins.start_s.to_numpy(float), bins.stop_s.iloc[-1]], np.ones(len(bins), bool)) if len(bins) else []
        # Centre each contiguous pause run on its own preceding four seconds.
        groups = (bins.start_s.diff().fillna(BIN).sub(BIN).abs() > 1e-6).cumsum()
        for _, group in bins.groupby(groups):
            t = (group.start_s.to_numpy(float) + group.stop_s.to_numpy(float)) / 2
            for z in depths:
                values.extend(centered_samples(field, t, np.full(len(t), z), float(group.start_s.iloc[0])))
        values = np.asarray(values, float)
        values = values[np.isfinite(values)]
        if len(values):
            row["pause_abs"] = float(np.median(np.abs(values)))
            row["pause_samples"] = int(len(values))
    else:
        depths = np.asarray([700.0, 1300.0, 2000.0, 2550.0, 3200.0])
        start, stop = WINDOWS[window]
        grid = np.arange(start, stop + 1e-6, 5.0)
        levels = np.asarray([sample_field(field, grid, np.full(len(grid), z)) for z in depths])
        increments = np.diff(levels, axis=1).ravel()
        increments = increments[np.isfinite(increments)]
        if len(increments):
            row["quiet_inc"] = float(np.sqrt(np.mean(increments ** 2)))
            row["quiet_increments"] = int(len(increments))
    return row


def pareto_front(table, columns: list[str]):
    keep = []
    values = table[columns].to_numpy(float)
    for i, row in enumerate(values):
        dominated = np.any(np.all(values <= row, axis=1) & np.any(values < row, axis=1))
        keep.append(not dominated)
    return table.loc[keep]


def write_dartsort_postfilter_audit(output: Path) -> dict:
    """Read-only replay of DREDge's internal post-filter on the saved p10 field."""
    from scipy import ndimage
    source = PILOT / "luke_imec1_p10/fit/field.npz"
    with np.load(source) as saved:
        times = np.asarray(saved["time_s"], float)
        displacement = np.asarray(saved["displacement_um"], float).T
    velocity = np.gradient(displacement, times, axis=1, edge_order=1)
    local_median = ndimage.median_filter(displacement, size=51, axes=(1,))
    speed_bad = np.abs(velocity) > 500.0
    band_bad = np.abs(displacement - local_median) > 250.0
    replaced = speed_bad | band_bad
    replaced[:, [0, -1]] = False
    # Match scipy interp1d's linear fill used by dredge.motion_util.speed_limit_filter.
    filtered = displacement.copy()
    for depth in range(len(displacement)):
        good = ~replaced[depth]
        filtered[depth] = np.interp(times, times[good], displacement[depth, good])
    first_four_s = times < times[0] + 4.0
    centered = displacement - np.median(displacement[:, first_four_s], axis=1, keepdims=True)
    audit = {
        "status": "complete_read_only",
        "source_field": str(source), "source_field_sha256": sha256(source),
        "settings_in_both_saved_runs": {"speed_limit_um_per_s": 500.0,
            "max_dist_from_median_um": 250.0, "median_neighborhood_bins": 51,
            "temporal_bin_length_s": 0.25},
        "pilot_p10": {"time_bins": int(displacement.shape[1]),
            "depth_time_bins": int(displacement.size),
            "replaced_unique_time_bins": int(replaced.any(axis=0).sum()),
            "replaced_depth_time_bins": int(replaced.sum()),
            "speed_only_depth_time_bins": int(speed_bad.sum()),
            "median_band_depth_time_bins": int(band_bad.sum()),
            "largest_raw_abs_excursion_removed_um": float(np.max(np.abs(displacement[replaced]))),
            "largest_4s_centered_abs_excursion_removed_um": float(np.max(np.abs(centered[replaced]))),
            "largest_interpolation_change_um": float(np.max(np.abs((displacement - filtered)[replaced])))},
        "external_field_behavior": {
            "postfilters_apply": False,
            "resampled_to_temporal_bin_length_s": False,
            "reason": "External dredge_motion_est is wrapped directly by MotionInfo.from_motion_est; dredge_estimate_motion and speed_limit_filter run only when DARTsort estimates motion internally.",
        },
        "code_locations": {
            "internal_filter_call": "/home/huklab/Documents/DARTsort/src/dartsort/util/registration_util.py:62",
            "external_branch": "/home/huklab/Documents/DARTsort/src/dartsort/main.py:220",
            "external_wrap": "/home/huklab/Documents/DARTsort/src/dartsort/main.py:227",
            "motion_wrapper_no_resample": "/home/huklab/Documents/DARTsort/src/dartsort/util/motion.py:164",
            "defaults": "/home/huklab/Documents/DARTsort/src/dartsort/config.py:232",
            "shared_recovery_adapter": "/media/huklab/Data/NPX/Ryansorting/Luke/DARTsort_motion_experiments/luke0804-imec1-full-medicine-shared-recovery-20260913-223213/source/pipeline.py:410",
            "merge_bias_adapter": "/home/huklab/DARTsort_runs/luke0804-imec1-full-medicine-merge-bias-nvme-v1-20260920/source/pipeline.py:474",
        },
    }
    atomic_json(output / "dartsort_postfilter_audit.json", audit)
    return audit


def report(output: Path) -> None:
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [score_field(output, config_id, window) for config_id in CONFIGS for window in WINDOWS]
    table = pd.DataFrame(rows)
    for window in WINDOWS:
        base_path = output / f"fields/base/{window}/field.npz"
        seed_path = output / f"fields/seed1/{window}/field.npz"
        with np.load(base_path) as a, np.load(seed_path) as b:
            times = np.asarray(a["session_time_s"])
            depths = np.asarray(a["depth_um"])
            other = {k: np.asarray(b[k]) for k in ("session_time_s", "depth_um", "displacement_um")}
            bt, bz = np.meshgrid(times, depths, indexing="ij")
            resampled = sample_field(other, bt, bz)
            delta = resampled - np.asarray(a["displacement_um"])
            value = float(np.sqrt(np.nanmean(delta ** 2)))
        table.loc[table.window == window, "seed_delta"] = value
    table.to_csv(output / "scores.csv", index=False)
    aggregates = []
    for config_id, group in table.groupby("config", sort=False):
        real = group[group.window.isin(REAL)]
        weights = real.motion_reference_bins.to_numpy(float)
        pooled_rmse = float(np.sqrt(np.average(real.motion_rmse.to_numpy(float) ** 2, weights=weights))) if weights.sum() else np.nan
        pooled_ratio = float(np.average(real.motion_ratio.to_numpy(float), weights=weights)) if weights.sum() else np.nan
        pause_abs = float(np.nanmedian(group.loc[group.window.isin(PAUSE), "pause_abs"]))
        quiet_inc = float(group.loc[group.window.eq("p50"), "quiet_inc"].iloc[0])
        aggregates.append({"config": config_id, "motion_rmse": pooled_rmse, "motion_ratio": pooled_ratio,
                           "pause_abs": pause_abs, "quiet_inc": quiet_inc,
                           "reference_bins": int(weights.sum()), "runtime_s": float(group.runtime_s.sum())})
    aggregate = pd.DataFrame(aggregates)
    eligible = aggregate.loc[(aggregate.pause_abs <= 10) & (aggregate.quiet_inc <= 5) &
                             aggregate.motion_ratio.between(0.8, 1.2)]
    if len(eligible):
        best = str(eligible.sort_values("motion_rmse").iloc[0].config)
        selection = {"outcome": "selected", "config": best}
        representative = best
    else:
        front = pareto_front(aggregate.dropna(subset=["motion_rmse", "pause_abs", "quiet_inc"]),
                             ["motion_rmse", "pause_abs", "quiet_inc"])
        selection = {"outcome": "no_selection", "pareto_front": front.config.tolist()}
        representative = str(front.sort_values("motion_rmse").iloc[0].config) if len(front) else "base"
    aggregate["selected"] = aggregate.config.eq(selection.get("config"))
    aggregate.to_csv(output / "config_summary.csv", index=False)
    atomic_json(output / "selection.json", selection)

    tracks = pd.read_csv(output / "references/lighthouse_tracks_250ms.csv")
    pause = pd.read_csv(output / "references/pause_bins.csv")
    for window, (start, stop) in WINDOWS.items():
        population, pop_start = population_source(output, window, 120)
        with np.load(population) as p:
            pt, pz = p["time_s"] + pop_start, p["depth_um"]
        stride = max(1, len(pt) // 160000)
        fig, axes = plt.subplots(3, 1, figsize=(15, 11), sharex=True, layout="constrained")
        axes[0].scatter(pt[::stride], pz[::stride], s=.12, c="#303030", alpha=.16, rasterized=True)
        if window in REAL:
            sub = tracks.loc[(tracks.window == window) & tracks.direction.eq("target_negative")]
            depth_map = pd.read_csv(FAMILIES).set_index("family_id").seed_depth_median_um
            for family, group in sub.groupby("family_id"):
                axes[0].plot(group.bin_center_s, depth_map.loc[family] + group.displacement_um,
                             color="#EE7733", linestyle="none", marker="o", ms=2, alpha=.8)
            for a, b in EPISODES[window]: axes[0].axvspan(a, b, color="#EE7733", alpha=.08)
        elif window in PAUSE:
            for row in pause.loc[(pause.window == window) & pause.pause_core].itertuples():
                axes[0].axvspan(row.start_s, row.stop_s, color="#0077BB", alpha=.10)
        axes[0].set(ylabel="Localized depth (µm)", title=f"{window}: cached native detections and independent references")
        for axis, config_id, color, style in [(axes[1], "base", "#0077BB", "-"),
                                               (axes[2], representative, "#AA3377", "--")]:
            with np.load(output / f"fields/{config_id}/{window}/field.npz") as f:
                for j, depth in enumerate(f["depth_um"]):
                    values = f["displacement_um"][:, j]
                    axis.plot(f["session_time_s"], values - np.median(values), color=color,
                              linestyle=style, lw=.8, alpha=.8)
            axis.axhline(0, color="0.25", lw=.6)
            axis.set(ylabel="Displacement (µm)", title=("Base" if config_id == "base" else
                     ("Selected" if selection["outcome"] == "selected" else "Pareto representative")) + f": {config_id}")
        axes[-1].set(xlabel="Session time (s)", xlim=(start, stop))
        fig.savefig(output / f"figure_{window}.png", dpi=170)
        plt.close(fig)

    postfilter = write_dartsort_postfilter_audit(output)
    reference_summary = json.loads((output / "references/summary.json").read_text())
    episodes = pd.read_csv(output / "references/real_episode_summary.csv")
    quiet_file = output / "references/quiet_lighthouse_increments.csv"
    quiet = pd.read_csv(quiet_file) if quiet_file.read_text().strip() else pd.DataFrame()
    quiet_rms = float(np.sqrt(np.mean(quiet.lighthouse_delta_um ** 2))) if len(quiet) else np.nan
    readme = [
        "# Luke0804 imec1 MEDiCINe reference-first sweep", "",
        "No spike sort or voltage modification was run. New work consists only of read-only reference measurements, native peak detection/localization caches, and MEDiCINe fits.", "",
        "## References", "",
        f"The frozen rescue 12/9 motion-off sort was used (identity `{json.loads((STATIC/'sort_identity.json').read_text()).get('identity_digest', 'see sort_identity.json')}`).",
        f"The real-motion reference contains **{reference_summary['lighthouse']['real_tracks']} tracks** and **{reference_summary['lighthouse']['real_bins']} qualified 0.25 s bins** (cosine ≥0.9, at least two matches).",
        "Median displacement per episode is in `references/real_episode_summary.csv`.",
        f"The quiet p50 lighthouse 5 s-increment RMS is **{quiet_rms:.3f} µm** across {len(quiet)} adjacent increments." if np.isfinite(quiet_rms) else "No adjacent p50 lighthouse increments were available; the quiet reference remains zero.",
        "Positive/opposite-direction controls were retained separately and were not folded into the target reference.", "",
        "## Reused tools", "",
        "- `testing/cross_dataset_fast_motion.py`: exact amended-coherence-gate DARTsort ibllikecmr frontend and sealed peak-cache format.",
        "- `testing/luke_imec1_compact_alignment_pilot_v2.py`: frozen compact templates and whole-probe identity scoring.",
        "- `testing/luke_imec1_cross_seed_replication_v1.py` and `testing/luke_imec1_lighthouse_motion_comparison_v1.py`: cross-seed identity and 5 s increment conventions.",
        "- `testing/outputs/motion_saved_array_audit_20260923/quiet_motion.csv`: quiet-motion comparison convention.",
        "- `pipeline/unit_quality.py`: standard unit-quality definitions were reviewed; no new sorting/QC was run.",
        "- DARTsort `fit_matched_medicine.py`, `qualify_local_motion_fields.py`, and `external_medicine_motion.py`: estimator isolation, interpolation, and sign contracts.", "",
        "## DARTsort deployment post-filters", "",
        "The shared-recovery and merge-bias runs passed the saved MEDiCINe field as an external `dredge_motion_est`. DARTsort wrapped that object directly, so the internal 500 µm/s speed filter and 250 µm/51-bin median-band filter did **not** clip either deployed MEDiCINe field. The field was also **not resampled** to the config's 0.25 s `temporal_bin_length_s`; its own 0.25 s time grid was retained. Both saved internal configs contain 500, 250, 51, and 0.25 s respectively.",
        f"As a counterfactual replay, those settings would replace {postfilter['pilot_p10']['replaced_unique_time_bins']}/480 p10 time bins ({postfilter['pilot_p10']['replaced_depth_time_bins']} depth×time cells). The largest removed raw absolute excursion is {postfilter['pilot_p10']['largest_raw_abs_excursion_removed_um']:.1f} µm; the largest four-second-centered excursion is {postfilter['pilot_p10']['largest_4s_centered_abs_excursion_removed_um']:.1f} µm and the largest interpolation change is {postfilter['pilot_p10']['largest_interpolation_change_um']:.1f} µm.",
        "Exact code locations and run-path evidence are in `dartsort_postfilter_audit.json`.", "",
        "## Selection", "", f"Outcome: `{selection['outcome']}`. " +
        (f"Selected `{selection['config']}` by the frozen rule." if selection["outcome"] == "selected" else
         f"No configuration qualified; Pareto front: {', '.join(selection['pareto_front'])}. No best configuration was picked."), "",
        "## Proposed stage-2 grid (not started)", "",
        "At most eight configs: the Pareto-front/selected setting, base, the closest useful neighbors among the 0.5/1/2/5/10/30 s kernels, one neighboring depth-bin count, seed 0/1 replication of the leader, native amplitude top-50%, and either rate equalization or a focused crop/full-probe check. Freeze the exact list after reviewing this stage-1 scorecard.", "",
        "`win60` is fit only on the central 60 s and is scored only where the fixed central-120 references lie inside its support; `win240` uses a separately cached 240 s detection window and is scored only on the central 120 s.",
    ]
    (output / "README.md").write_text("\n".join(readme) + "\n")
    archive = output / "sweep_fields.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zipped:
        for config_id in CONFIGS:
            for window in WINDOWS:
                path = output / f"fields/{config_id}/{window}/field.npz"
                zipped.write(path, arcname=f"{config_id}/{window}/field.npz")
    atomic_json(output / "validation.json", validate_outputs(output, archive))


def validate_outputs(output: Path, archive: Path) -> dict:
    import pandas as pd
    scores = pd.read_csv(output / "scores.csv")
    if len(scores) != len(CONFIGS) * len(WINDOWS):
        raise RuntimeError("Score table does not have one row per config x window")
    if scores[["config", "window"]].duplicated().any():
        raise RuntimeError("Duplicate score rows")
    with zipfile.ZipFile(archive) as zipped:
        names = zipped.namelist()
    if len(names) != len(CONFIGS) * len(WINDOWS):
        raise RuntimeError("Field archive is incomplete")
    for config_id in CONFIGS:
        for window in WINDOWS:
            with np.load(output / f"fields/{config_id}/{window}/field.npz", allow_pickle=False) as field:
                if field["displacement_um"].shape != (len(field["time_s"]), len(field["depth_um"])):
                    raise RuntimeError(f"Bad field shape: {config_id}/{window}")
                if not np.isfinite(field["displacement_um"]).all():
                    raise RuntimeError(f"Nonfinite field: {config_id}/{window}")
    figures = [output / f"figure_{window}.png" for window in WINDOWS]
    if not all(path.stat().st_size > 10000 for path in figures):
        raise RuntimeError("One or more figures failed visual-file validation")
    return {"status": "ready_within_reviewed_scope", "score_rows": len(scores),
            "archive_fields": len(names), "archive_bytes": archive.stat().st_size,
            "figures": [str(path) for path in figures], "no_sort": True, "voltage_modified": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["references", "extract-one", "fit-one", "sweep", "report", "all"])
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--window")
    parser.add_argument("--config-id")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.phase == "references": prepare_references(output)
    elif args.phase == "extract-one": extract_one(output, args.window)
    elif args.phase == "fit-one": fit_one(output, args.config_id, args.window)
    elif args.phase == "sweep": run_sweep(output)
    elif args.phase == "report": report(output)
    else:
        try:
            run_sweep(output)
            report(output)
            atomic_json(output / "controller_receipt.json", {"status": "complete", "finished_at": time.time()})
        except BaseException as error:
            atomic_json(output / "controller_receipt.json", {"status": "failed", "finished_at": time.time(),
                        "error": repr(error), "traceback": traceback.format_exc()})
            raise


if __name__ == "__main__":
    main()
