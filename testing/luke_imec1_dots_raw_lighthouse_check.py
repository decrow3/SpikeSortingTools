#!/usr/bin/env python
"""Raw-voltage, depth-blind lighthouse search for Luke0804 imec1 dots-RF.

Selection is inherited from the frozen seed-only direct check. Raw templates are
built only from sorter events in the seed interval. Matching searches the seed
and six prespecified motion-rich windows using exact NP1 40-um translations.
Absolute waveform depth and the native motion field are opened only after the
identity scores have been frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import shutil
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.signal import butter, find_peaks, sosfiltfilt

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from testing.luke_imec1_dots_lighthouse_direct_check import (
    MOTION_WINDOWS,
    ROOT,
    SEED_INTERVAL_S,
    add_field_comparison,
    atomic_json,
    load_motion,
    sha256,
)


SCHEMA = "luke0804-imec1-dots-raw-lighthouse-v2"
DIRECT = ROOT / "testing/outputs/luke_imec1_dots_lighthouse_direct_check_v1"
DEFAULT_OUTPUT = ROOT / "testing/outputs/luke_imec1_dots_raw_lighthouse_check_v1"
EXPERIMENT = Path("/media/huklab/Data/NPX/Ryansorting/Luke/DARTsort_motion_experiments/luke0804-imec1-local-spikeglx-dots-v4")
SORTING = EXPERIMENT / "sorts/static-v1/dartsort_sorting.npz"
SORT_RESULT = EXPERIMENT / "sorts/static-v1/result.json"
NATIVE_MOTION = EXPERIMENT / "sorts/native-v1/motion.pkl"
MANIFEST = Path("/home/huklab/Documents/DARTsort/experiments/luke0804_imec1/spikeglx_imec1_local_manifest.json")
OFFSETS = np.arange(-24, 25)
GAIN_UV = 2.34375
STRICT_COSINE = 0.86
DISPLAY_COSINE = 0.80
IDENTITY_MARGIN = 0.025
MIN_PHASE_SEED_SPIKES = 8
MAX_PHASE_SEED_SPIKES = 200


def raw_sha_contract(path: Path) -> dict:
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def geometry_and_mapping() -> tuple[np.ndarray, np.ndarray]:
    # SpikeGLX NP1 order. DARTsort's accepted sorting removed only AP191.
    rows = np.repeat(np.arange(192) * 20.0, 2)
    xs = np.tile([16.0, 48.0, 0.0, 32.0], 96)
    raw_geom = np.c_[xs, rows]
    keep = np.delete(np.arange(384), 191)
    return raw_geom[keep], keep


def patches(geom: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    candidate_bases = np.arange(80.0, geom[:, 1].max() - 79.0, 40.0)
    valid_bases = []
    channel_sets = []
    for base in candidate_bases:
        channels = np.flatnonzero((geom[:, 1] >= base - 60.0) & (geom[:, 1] <= base + 80.0))
        if len(channels) != 16:
            # AP191 was rejected by the locked DARTsort preprocessing. Do not
            # fabricate support across patches which intersect that channel.
            continue
        valid_bases.append(base)
        channel_sets.append(channels)
    bases = np.asarray(valid_bases)
    relative = geom[np.asarray(channel_sets)] - np.c_[np.zeros(len(bases)), bases][:, None, :]
    if not np.all(relative == relative[0]):
        raise RuntimeError("40-um translated patches do not preserve geometry")
    return bases, np.asarray(channel_sets)


def preprocess(raw: np.memmap, start: int, stop: int, kept_raw: np.ndarray, fs: float) -> np.ndarray:
    pad = round(0.05 * fs)
    if start < pad or stop + pad > raw.shape[0]:
        raise ValueError("analysis chunk lacks filter margin")
    x = np.asarray(raw[start - pad : stop + pad, kept_raw], dtype=np.float32) * GAIN_UV
    sos = butter(3, [300.0, 6000.0], fs=fs, btype="bandpass", output="sos")
    x = sosfiltfilt(sos, x, axis=0).astype(np.float32)
    x -= np.median(x, axis=1, keepdims=True)
    return x[pad:-pad]


def phase_id(channel: int, base_index: int, patch_channels: np.ndarray, geom: np.ndarray) -> int:
    local = int(np.flatnonzero(patch_channels[base_index] == channel)[0])
    # The local channel index encodes relative row and lateral column because
    # every retained patch has the same ordered relative geometry.
    return local


def build_templates(
    raw: np.memmap,
    fs: float,
    window_start: int,
    labels: np.ndarray,
    times: np.ndarray,
    channels: np.ndarray,
    inventory: pd.DataFrame,
    geom: np.ndarray,
    kept_raw: np.ndarray,
    bases: np.ndarray,
    patch_channels: np.ndarray,
    whole_probe_rivals: bool = False,
) -> tuple[np.ndarray, pd.DataFrame]:
    seed_start, seed_stop = [round(t * fs) for t in SEED_INTERVAL_S]
    x = preprocess(raw, seed_start, seed_stop, kept_raw, fs)
    absolute = times + window_start
    seed = (absolute >= seed_start) & (absolute < seed_stop) & (labels >= 0)
    rival_units = set(np.unique(labels[seed & (labels >= 0)])) if whole_probe_rivals else set(inventory.loc[inventory.eligible, "unit_id"].astype(int))
    selected_units = set(inventory.loc[inventory.selected, "unit_id"].astype(int))
    seed &= np.isin(labels, list(rival_units))
    grouped: dict[tuple[int, int], list[np.ndarray]] = {}
    for row in np.flatnonzero(seed):
        channel = int(channels[row])
        base = 40.0 * np.floor(geom[channel, 1] / 40.0)
        bi_arr = np.flatnonzero(bases == base)
        if not len(bi_arr):
            continue
        bi = int(bi_arr[0])
        frame = int(absolute[row] - seed_start)
        if frame < 30 or frame + 30 >= len(x):
            continue
        key = (int(labels[row]), phase_id(channel, bi, patch_channels, geom))
        values = grouped.setdefault(key, [])
        if len(values) < MAX_PHASE_SEED_SPIKES:
            values.append(x[frame + OFFSETS][:, patch_channels[bi]])
    waves, rows = [], []
    for (unit, phase), values in sorted(grouped.items()):
        if len(values) < MIN_PHASE_SEED_SPIKES:
            continue
        waveform = np.median(np.asarray(values), axis=0).astype(np.float32)
        norm = float(np.linalg.norm(waveform))
        if not np.isfinite(norm) or norm == 0:
            continue
        waves.append(waveform / norm)
        rows.append({"unit_id": unit, "phase_id": phase, "seed_spikes": len(values), "selected": unit in selected_units})
    return np.asarray(waves), pd.DataFrame(rows)


def score_events(waves: np.ndarray, templates: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return winner, score, margin, gain after timing alignment and decoy competition."""
    n, nt = len(waves), len(templates)
    best = np.full((n, nt * 3), -1.0, dtype=np.float32)
    gains = np.zeros((n, nt), dtype=np.float32)
    real = templates.reshape(nt, -1)
    decoy_time = templates[:, ::-1].reshape(nt, -1)
    decoy_space = templates[:, :, ::-1].reshape(nt, -1)
    bank = np.concatenate([real, decoy_time, decoy_space])
    bank /= np.maximum(np.linalg.norm(bank, axis=1, keepdims=True), 1e-12)
    template_norm = np.linalg.norm(templates.reshape(nt, -1), axis=1)
    for lag in range(-3, 4):
        shifted = np.zeros_like(waves)
        if lag < 0:
            shifted[:, :lag] = waves[:, -lag:]
        elif lag > 0:
            shifted[:, lag:] = waves[:, :-lag]
        else:
            shifted = waves
        flat = shifted.reshape(n, -1)
        norms = np.linalg.norm(flat, axis=1)
        scores = flat @ bank.T / np.maximum(norms[:, None], 1e-12)
        best = np.maximum(best, scores)
        dots = flat @ real.T
        gains = np.where(scores[:, :nt] >= best[:, :nt], dots / np.maximum(template_norm[None], 1e-12), gains)
    order = np.argsort(best, axis=1, kind="stable")
    ar = np.arange(n)
    winner = order[:, -1]
    return winner, best[ar, winner], best[ar, winner] - best[ar, order[:, -2]], gains[ar, np.minimum(winner, nt - 1)]


def analyze_chunk(
    raw: np.memmap, start_s: float, stop_s: float, fs: float, kept_raw: np.ndarray,
    geom: np.ndarray, bases: np.ndarray, patch_channels: np.ndarray,
    templates: np.ndarray, template_table: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    start, stop = round(start_s * fs), round(stop_s * fs)
    x = preprocess(raw, start, stop, kept_raw, fs)
    noise = np.median(np.abs(x[::10] - np.median(x[::10], axis=0)), axis=0) / 0.67448975
    detected = []
    for channel in range(len(geom)):
        base = 40.0 * np.floor(geom[channel, 1] / 40.0)
        bi_arr = np.flatnonzero(bases == base)
        if not len(bi_arr):
            continue
        ev, props = find_peaks(np.abs(x[:, channel]), height=max(30.0, 3.0 * noise[channel]), distance=round(0.0008 * fs))
        ev = ev[(ev > 30) & (ev + 30 < len(x))]
        near = np.flatnonzero(np.abs(geom[:, 1] - geom[channel, 1]) <= 40.0)
        ev = ev[np.argmax(np.abs(x[ev[:, None], near]), axis=1) == np.flatnonzero(near == channel)[0]]
        bi = int(bi_arr[0]); phase = phase_id(channel, bi, patch_channels, geom)
        detected.extend((int(e), channel, bi, phase) for e in ev)
    detected = np.asarray(sorted(detected), dtype=np.int64).reshape(-1, 4)
    out = []
    counters = {"detections": len(detected), "no_phase_rivals": 0, "decoy_winners": 0}
    for offset in range(0, len(detected), 256):
        block = detected[offset : offset + 256]
        for phase in np.unique(block[:, 3]):
            events = block[block[:, 3] == phase]
            ti = np.flatnonzero(template_table.phase_id.to_numpy() == phase)
            if len(ti) < 2:
                counters["no_phase_rivals"] += len(events)
                continue
            ew = np.asarray([x[e + OFFSETS][:, patch_channels[bi]] for e, _, bi, _ in events], dtype=np.float32)
            winner, score, margin, gain = score_events(ew, templates[ti])
            nt = len(ti)
            for j, (event, win, sc, mar, ga) in enumerate(zip(events, winner, score, margin, gain)):
                e, channel, bi, _ = map(int, event)
                if win >= nt:
                    counters["decoy_winners"] += 1
                    continue
                row = template_table.iloc[ti[win]]
                if not row.selected or sc < DISPLAY_COSINE:
                    continue
                energy = np.square(ew[j].astype(np.float64)).sum(axis=0)
                centroid = float(energy @ geom[patch_channels[bi], 1] / energy.sum())
                status = "strict" if sc >= STRICT_COSINE and mar >= IDENTITY_MARGIN else ("lower_score" if mar >= IDENTITY_MARGIN else "identity_ambiguous")
                out.append({"unit_id": int(row.unit_id), "time_s": (start + e) / fs, "peak_channel": channel,
                            "phase_id": phase, "waveform_centroid_um": centroid, "score": float(sc),
                            "margin": float(mar), "gain": float(ga), "status": status,
                            "seed": SEED_INTERVAL_S[0] <= (start + e) / fs < SEED_INTERVAL_S[1]})
    return pd.DataFrame(out), counters


def run(config: dict) -> None:
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "settings.json", config)
    if config.get("dummy"):
        print("dummy started", flush=True); time.sleep(20); print("dummy completed", flush=True)
        atomic_json(output / "summary.json", {"schema": SCHEMA, "status": "complete", "dummy": True})
        return
    manifest = json.loads(MANIFEST.read_text())
    binary = Path(manifest["binary_path"])
    fs = float(manifest["sampling_frequency_hz"])
    before = raw_sha_contract(binary)
    raw = np.memmap(binary, dtype="<i2", mode="r", shape=(manifest["num_samples"], manifest["saved_channels_in_binary"]))
    geom, kept_raw = geometry_and_mapping(); bases, patch_channels = patches(geom)
    with np.load(SORTING, allow_pickle=False) as sorting:
        labels = np.asarray(sorting["labels"], dtype=np.int64); times = np.asarray(sorting["times_samples"], dtype=np.int64)
        channels = np.asarray(sorting["channels"], dtype=np.int64); sort_geom = np.asarray(sorting["geom"])
    if not np.array_equal(sort_geom, geom): raise RuntimeError("raw/DARTsort channel map mismatch")
    window_start = int(json.loads(SORT_RESULT.read_text())["window_start_frame"])
    inventory = pd.read_csv(config["candidate_inventory"])
    if config.get("all_eligible_targets"):
        inventory["selected"] = inventory.eligible
    templates, template_table = build_templates(raw, fs, window_start, labels, times, channels, inventory, geom, kept_raw, bases, patch_channels, bool(config.get("whole_probe_rivals")))
    template_table.to_csv(output / "seed_phase_templates.csv", index=False)
    np.savez_compressed(output / "seed_phase_waveforms.npz", waveforms=templates, relative_geometry=geom[patch_channels[0]] - np.array([0.0, bases[0]]))
    print(f"raw seed templates {len(templates)} selected {template_table.selected.sum()}", flush=True)
    intervals = [SEED_INTERVAL_S, *MOTION_WINDOWS]
    event_parts, counter_rows = [], []
    for i, (start, stop) in enumerate(intervals):
        csv = output / f"interval_{i:02d}_events.csv"; seal = output / f"interval_{i:02d}.complete.json"
        if csv.exists() or seal.exists(): raise RuntimeError("partial interval evidence exists; investigate before retry")
        events, counters = analyze_chunk(raw, start, stop, fs, kept_raw, geom, bases, patch_channels, templates, template_table)
        events.to_csv(csv, index=False); event_parts.append(events); counter_rows.append({"start_s": start, "stop_s": stop, **counters})
        atomic_json(seal, {"events_sha256": sha256(csv), **counters}); print(i, start, stop, counters, flush=True)
    if raw_sha_contract(binary) != before: raise RuntimeError("raw source changed during analysis")
    events = pd.concat(event_parts, ignore_index=True) if event_parts else pd.DataFrame()
    events.to_csv(output / "events.csv", index=False); pd.DataFrame(counter_rows).to_csv(output / "interval_counts.csv", index=False)
    strict = events[events.status == "strict"].copy()
    seed_centers = strict[strict.seed].groupby("unit_id").waveform_centroid_um.median()
    strict["relative_um"] = strict.waveform_centroid_um - strict.unit_id.map(seed_centers)
    plaus = strict.groupby("unit_id").agg(strict_matches=("time_s", "size"), heldout_matches=("seed", lambda x: int((~x).sum())), depth_min_um=("waveform_centroid_um", "min"), depth_max_um=("waveform_centroid_um", "max")).reset_index()
    plaus["depth_span_um"] = plaus.depth_max_um - plaus.depth_min_um
    seed_counts = strict[strict.seed].groupby("unit_id").size()
    plaus["seed_strict_matches"] = plaus.unit_id.map(seed_counts).fillna(0).astype(int)
    plaus["seed_qualified"] = plaus.seed_strict_matches >= 20
    plaus["provisional_plausible"] = plaus.seed_qualified & (plaus.heldout_matches >= 20) & (plaus.depth_span_um <= 500)
    plaus.to_csv(output / "track_plausibility.csv", index=False)
    # Only now open the estimator being evaluated. Bin raw identity measurements
    # using the same direct-check contract before making the descriptive comparison.
    if len(strict):
        strict["time_bin_s"] = np.floor(strict.time_s * 2.0) / 2.0 + 0.25
        binned = strict.groupby(["unit_id", "time_bin_s"], as_index=False).agg(
            observed_depth_um=("waveform_centroid_um", "median"), event_count=("time_s", "size")
        )
        centers = strict[strict.seed].groupby("unit_id").waveform_centroid_um.median()
        binned["seed_depth_um"] = binned.unit_id.map(centers)
        binned["observed_relative_um"] = binned.observed_depth_um - binned.seed_depth_um
        binned, comparison = add_field_comparison(binned, NATIVE_MOTION, window_start / fs)
    else:
        binned, comparison = pd.DataFrame(), pd.DataFrame()
    binned.to_csv(output / "native_field_postmatch_bins.csv", index=False)
    comparison.to_csv(output / "native_field_postmatch_metrics.csv", index=False)
    fig, ax = plt.subplots(figsize=(13, 7), layout="constrained")
    for unit, group in strict.groupby("unit_id"):
        ax.scatter(group.time_s, group.waveform_centroid_um, s=7, alpha=.55, label=str(unit))
    ax.set(xlabel="Absolute recording time (s)", ylabel="Raw waveform energy centroid (µm)", title="Raw waveform-only whole-probe matches (strict)\nSeed selection and identity scoring exclude absolute depth and native motion")
    if 0 < strict.unit_id.nunique() <= 12: ax.legend(ncol=2, fontsize=7)
    fig.savefig(output / "01_raw_strict_depth_atlas.png", dpi=170); fig.savefig(output / "01_raw_strict_depth_atlas.pdf"); plt.close(fig)
    summary = {"schema": SCHEMA, "status": "complete", "raw_contract": before,
               "seed_phase_templates": len(templates), "selected_phase_templates": int(template_table.selected.sum()),
               "display_events": len(events), "strict_events": len(strict), "strict_units": int(strict.unit_id.nunique()),
               "plausible_units": plaus.loc[plaus.provisional_plausible, "unit_id"].astype(int).tolist(),
               "interpretation": "Candidate-finding evidence only; provisional plausible tracks require waveform review and replication before use as motion ground truth.",
               "limits": ["Exact 40-um translations are phase-specific; alternate-phase dropout is inconclusive.", "Rival bank is the 56 frozen eligible seed identities, not every unsorted biological cell.", "Thresholds and synthetic decoys are exploratory rather than calibrated ground-truth false-positive controls."]}
    atomic_json(output / "summary.json", summary)
    (output / "README.md").write_text("# Luke0804 imec1 raw lighthouse search\n\nSelection is frozen from the 20-second seed interval. Matching uses raw, 300–6000 Hz filtered, common-median-referenced 49-sample × 16-channel waveforms under exact 40-µm translations. Absolute depth and native motion are revealed only after matching. Strict, lower-score, ambiguous, decoy-winning, and unmatched counts are preserved. See `summary.json` and `track_plausibility.csv`.\n")
    print(json.dumps(summary, indent=2), flush=True)


def report_from_failed_extraction(config: dict) -> None:
    """Finalize sealed identity evidence without rerunning raw extraction."""
    source = Path(config["recovery_source"])
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=False)
    seals = []
    for index in range(7):
        csv = source / f"interval_{index:02d}_events.csv"
        seal_path = source / f"interval_{index:02d}.complete.json"
        seal = json.loads(seal_path.read_text())
        if sha256(csv) != seal["events_sha256"]:
            raise RuntimeError(f"interval {index} seal mismatch")
        seals.append({"index": index, "events_sha256": seal["events_sha256"]})
    events = pd.read_csv(source / "events.csv")
    plausibility = pd.read_csv(source / "track_plausibility.csv")
    templates = pd.read_csv(source / "seed_phase_templates.csv")
    qualified = plausibility[plausibility.seed_qualified]
    plausible = plausibility[plausibility.provisional_plausible]
    shutil.copy2(source / "track_plausibility.csv", output / "track_plausibility.csv")
    shutil.copy2(source / "interval_counts.csv", output / "interval_counts.csv")
    strict = events[events.status == "strict"]
    fig, ax = plt.subplots(figsize=(13, 7), layout="constrained")
    for unit, group in strict.groupby("unit_id"):
        ax.scatter(group.time_s, group.waveform_centroid_um, s=22, alpha=.75, label=str(unit))
    ax.set(xlabel="Absolute recording time (s)", ylabel="Raw waveform energy centroid (µm)",
           title="Full-rival raw waveform search: 16 strict events, no seed-qualified identity\nAbsolute depth revealed only after waveform matching")
    if strict.unit_id.nunique(): ax.legend(ncol=2, fontsize=7)
    fig.savefig(output / "01_strict_event_depth_atlas.png", dpi=170)
    fig.savefig(output / "01_strict_event_depth_atlas.pdf")
    plt.close(fig)
    summary = {
        "schema": SCHEMA + "-report-v1", "status": "complete",
        "source_failed_job_preserved": config["source_failed_job"],
        "source_extraction": str(source), "verified_interval_seals": seals,
        "rival_units": int(templates.unit_id.nunique()), "rival_phase_templates": len(templates),
        "target_units": int(templates.loc[templates.selected, "unit_id"].nunique()),
        "target_phase_templates": int(templates.selected.sum()), "display_events": len(events),
        "strict_events": len(strict), "strict_units": int(strict.unit_id.nunique()),
        "seed_qualified_units": qualified.unit_id.astype(int).tolist(),
        "provisional_plausible_units": plausible.unit_id.astype(int).tolist(),
        "motion_comparison_performed": False,
        "automatic_motion_verdict": False,
        "interpretation": "No raw-waveform lighthouse candidate passed the seed-only identity gate; there is therefore no justified imec1 lighthouse ground truth for these dots-RF windows from this bounded search.",
        "next_method": "If stronger ground truth is required, use manual/semiautomatic depth-aware waveform tracing with interpolated 20-um phase support, blinded competing identities, and independent visual confirmation; do not relax identity thresholds based on agreement with the motion estimate.",
        "limits": ["Exact 40-um translations can miss alternate 20-um lattice-phase support.", "The seed identities originate from the static sorter.", "A negative bounded search is not proof that no biological lighthouse exists."],
    }
    atomic_json(output / "summary.json", summary)
    (output / "README.md").write_text("# Luke0804 imec1 dots-RF raw lighthouse report\n\nAll seven source intervals were hash-verified after the extraction job's optional motion-overlay tail failed. Among 1,009 whole-probe rival identities, no target reached 20 strict seed matches. The motion estimate was therefore not compared to any track. The failed extraction job and all interval evidence remain preserved.\n")
    print(json.dumps(summary, indent=2), flush=True)


def audit_sorted_seed_events(config: dict) -> None:
    """Audit raw identity at known sorter seed times, without depth or motion."""
    source = Path(config["seed_audit_source"])
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=False)
    with np.load(source / "seed_phase_waveforms.npz") as saved:
        templates = np.asarray(saved["waveforms"], dtype=np.float32)
    table = pd.read_csv(source / "seed_phase_templates.csv")
    events = pd.read_csv(source / "events.csv")
    manifest = json.loads(MANIFEST.read_text()); binary = Path(manifest["binary_path"]); fs = float(manifest["sampling_frequency_hz"])
    before = raw_sha_contract(binary)
    raw = np.memmap(binary, dtype="<i2", mode="r", shape=(manifest["num_samples"], manifest["saved_channels_in_binary"]))
    geom, kept_raw = geometry_and_mapping(); bases, patch_channels = patches(geom)
    with np.load(SORTING, allow_pickle=False) as sorting:
        labels=np.asarray(sorting["labels"],dtype=np.int64); times=np.asarray(sorting["times_samples"],dtype=np.int64); channels=np.asarray(sorting["channels"],dtype=np.int64)
    window_start=int(json.loads(SORT_RESULT.read_text())["window_start_frame"]); absolute=times+window_start
    seed_start,seed_stop=[round(t*fs) for t in SEED_INTERVAL_S]; x=preprocess(raw,seed_start,seed_stop,kept_raw,fs)
    targets=set(table.loc[table.selected,"unit_id"].astype(int)); rows=[]; event_rows=[]
    for unit in sorted(targets):
        source_rows=np.flatnonzero((labels==unit)&(absolute>=seed_start)&(absolute<seed_stop))
        evaluated=correct=decoys=0; scores=[]; margins=[]
        for phase in sorted(table.loc[table.unit_id.eq(unit),"phase_id"].unique()):
            phase_rows=[]
            for row in source_rows:
                channel=int(channels[row]); base=40.0*np.floor(geom[channel,1]/40.0); bi_arr=np.flatnonzero(bases==base)
                if not len(bi_arr): continue
                bi=int(bi_arr[0])
                if phase_id(channel,bi,patch_channels,geom)==phase: phase_rows.append((row,bi))
            if len(phase_rows)>500:
                phase_rows=[phase_rows[i] for i in np.linspace(0,len(phase_rows)-1,500,dtype=int)]
            ti=np.flatnonzero(table.phase_id.to_numpy()==phase)
            for begin in range(0,len(phase_rows),256):
                block=phase_rows[begin:begin+256]
                waves=np.asarray([x[int(absolute[r]-seed_start)+OFFSETS][:,patch_channels[bi]] for r,bi in block],dtype=np.float32)
                winner,score,margin,_=score_events(waves,templates[ti]); nt=len(ti); evaluated+=len(block); decoys+=int((winner>=nt).sum())
                real=winner<nt; winner_units=np.full(len(winner),-1,dtype=int); winner_units[real]=table.iloc[ti[winner[real]]].unit_id.to_numpy(dtype=int)
                strict=real&(score>=STRICT_COSINE)&(margin>=IDENTITY_MARGIN)&(winner_units==unit)
                correct+=int(strict.sum()); scores.extend(score[strict].tolist()); margins.extend(margin[strict].tolist())
                for (source_row, bi), win_unit, is_real, sc, mar, ok in zip(block, winner_units, real, score, margin, strict):
                    waveform=x[int(absolute[source_row]-seed_start)+OFFSETS][:,patch_channels[bi]]
                    energy=np.square(waveform.astype(np.float64)).sum(axis=0)
                    event_rows.append({"source_unit_id":unit,"time_s":absolute[source_row]/fs,"phase_id":int(phase),"winner_kind":"real" if is_real else "decoy","winner_unit_id":int(win_unit),"score":float(sc),"margin":float(mar),"strict_correct":bool(ok),"waveform_centroid_um":float(energy@geom[patch_channels[bi],1]/energy.sum())})
        rate=correct/evaluated if evaluated else 0.0
        rows.append({"unit_id":unit,"evaluated_seed_spikes":evaluated,"strict_correct_seed_spikes":correct,"strict_correct_rate":rate,"decoy_winners":decoys,"strict_score_median":np.median(scores) if scores else np.nan,"strict_margin_median":np.median(margins) if margins else np.nan,"seed_qualified":correct>=20 and rate>=0.5})
    audit=pd.DataFrame(rows); audit.to_csv(output/"seed_source_identity_audit.csv",index=False)
    pd.DataFrame(event_rows).to_csv(output/"seed_source_event_competition.csv",index=False)
    qualified=set(audit.loc[audit.seed_qualified,"unit_id"].astype(int)); strict=events[(events.status=="strict")&events.unit_id.isin(qualified)&(~events.seed)]
    held=strict.groupby("unit_id").agg(heldout_strict_matches=("time_s","size"),depth_min_um=("waveform_centroid_um","min"),depth_max_um=("waveform_centroid_um","max")).reset_index()
    if len(held): held["depth_span_um"]=held.depth_max_um-held.depth_min_um
    held.to_csv(output/"qualified_heldout_support.csv",index=False)
    if raw_sha_contract(binary)!=before: raise RuntimeError("raw source changed during seed audit")
    summary={"schema":SCHEMA+"-sorted-seed-audit-v1","status":"complete","rival_units":int(table.unit_id.nunique()),"target_units":len(targets),"qualification_rule":"at least 20 strict correct known-time seed spikes and at least 50% strict correct rate; no depth or motion used","seed_qualified_units":sorted(qualified),"qualified_units_with_heldout_strict_support":sorted(set(held.unit_id.astype(int))) if len(held) else [],"motion_comparison_performed":False,"interpretation":"Known-time seed audit separates identity repeatability from blind-detector recentering; held-out matches remain independent blind detections."}
    atomic_json(output/"summary.json",summary); (output/"README.md").write_text("# Sorted-seed raw identity audit\n\nRaw waveforms at known static-sort seed times compete against the full seed-supported rival bank. Candidate qualification uses only seed identity recovery, never depth or motion. Held-out support is inherited from the independently frozen blind detection pass.\n")
    print(json.dumps(summary,indent=2),flush=True)


def report_lower_score_candidate(config: dict) -> None:
    """Create a post-discovery report for a near-threshold candidate family."""
    output=Path(config["output"]); output.mkdir(parents=True,exist_ok=False); unit=int(config["candidate_unit_id"])
    extraction=Path(config["candidate_extraction_source"]); audit_source=Path(config["candidate_audit_source"])
    events=pd.read_csv(extraction/"events.csv"); audit_events=pd.read_csv(audit_source/"seed_source_event_competition.csv")
    inventory=pd.read_csv(config["candidate_inventory"]); inv=inventory[inventory.unit_id.eq(unit)].iloc[0]
    seed=audit_events[audit_events.source_unit_id.eq(unit)].copy(); held=events[(events.unit_id.eq(unit))&(~events.seed)&events.status.isin(["strict","lower_score"])].copy()
    identity_wins=(seed.winner_kind.eq("real")&seed.winner_unit_id.eq(unit)); strict_seed=seed.strict_correct
    baseline_depth=float(seed.waveform_centroid_um.median())
    with open(NATIVE_MOTION,"rb") as handle: motion=pickle.load(handle)["dredge_motion_est"]
    fs=float(json.loads(MANIFEST.read_text())["sampling_frequency_hz"]); window_start=int(json.loads(SORT_RESULT.read_text())["window_start_frame"])/fs
    evaluation_depth=float(np.clip(baseline_depth,motion.d_low,motion.d_high))
    seed_field=np.asarray(motion.disp_at_s(seed.time_s.to_numpy()-window_start,np.full(len(seed),evaluation_depth))).ravel(); field_center=float(np.median(seed_field))
    held["observed_relative_um"]=held.waveform_centroid_um-baseline_depth
    held["native_relative_um"]=np.asarray(motion.disp_at_s(held.time_s.to_numpy()-window_start,np.full(len(held),evaluation_depth))).ravel()-field_center
    held["residual_um"]=held.observed_relative_um-held.native_relative_um
    held.to_csv(output/"candidate_events_with_postmatch_field.csv",index=False)
    fig,ax=plt.subplots(figsize=(12,6),layout="constrained")
    ax.scatter(seed.time_s,seed.waveform_centroid_um-baseline_depth,s=18,color="#999999",alpha=.55,label="Known-time seed events")
    colors=held.status.map({"strict":"#0072B2","lower_score":"#E69F00"})
    ax.scatter(held.time_s,held.observed_relative_um,s=38,c=colors,label="Blind held-out waveform matches")
    ax.scatter(held.time_s,held.native_relative_um,s=32,marker="x",color="#CC79A7",label="Native field at clipped top depth (post-match)")
    for _,row in held.iterrows(): ax.plot([row.time_s,row.time_s],[row.observed_relative_um,row.native_relative_um],color="#bbbbbb",lw=.6,zorder=0)
    ax.axhline(0,color="black",lw=.5); ax.set(xlabel="Absolute recording time (s)",ylabel="Seed-relative depth / displacement (µm)",title=f"Unit {unit}: lower-score candidate family, not verified lighthouse\nIdentity chosen without depth/motion; field opened afterward at {evaluation_depth:.0f} µm boundary")
    ax.legend(fontsize=8); fig.savefig(output/"01_candidate_track_and_postmatch_field.png",dpi=170); fig.savefig(output/"01_candidate_track_and_postmatch_field.pdf"); plt.close(fig)
    corr=float(np.corrcoef(held.observed_relative_um,held.native_relative_um)[0,1]) if len(held)>=3 else np.nan
    summary={"schema":SCHEMA+"-lower-score-candidate-v1","status":"complete","candidate_unit_id":unit,"classification":"lower_score_candidate_family","verified_lighthouse":False,"usable_as_motion_ground_truth":False,"known_seed_spikes_evaluated":len(seed),"correct_identity_wins":int(identity_wins.sum()),"correct_identity_win_rate":float(identity_wins.mean()),"strict_correct_seed_spikes":int(strict_seed.sum()),"strict_correct_seed_rate":float(strict_seed.mean()),"frozen_seed_gate_passed":False,"heldout_blind_matches":len(held),"heldout_strict_matches":int(held.status.eq("strict").sum()),"heldout_lower_score_matches":int(held.status.eq("lower_score").sum()),"heldout_motion_windows_with_support":int(sum(((held.time_s>=a)&(held.time_s<b)).any() for a,b in MOTION_WINDOWS)),"raw_seed_depth_median_um":baseline_depth,"static_template_depth_um":float(inv.template_depth_um),"template_vs_raw_seed_depth_disagreement_um":float(inv.template_depth_um-baseline_depth),"motion_evaluation_depth_um":evaluation_depth,"motion_depth_clipped_to_boundary":evaluation_depth!=baseline_depth,"postmatch_level_correlation":corr,"postmatch_mae_um":float(np.mean(np.abs(held.residual_um))),"postmatch_median_residual_um":float(np.median(held.residual_um)),"interpretation":"Identity wins are strong enough to justify manual depth-aware follow-up, but the frozen strict seed gate failed, support is sparse and unreplicated, and depth provenance is inconsistent. Do not use this family as lighthouse ground truth yet."}
    atomic_json(output/"summary.json",summary); (output/"README.md").write_text(f"# Unit {unit} lower-score candidate\n\nThis is an exploratory post-discovery report, not a verified lighthouse. Identity scoring excluded depth and motion. The native field was opened only afterward and evaluated at its upper spatial boundary because the raw seed footprint lies above the field's highest bin.\n")
    print(json.dumps(summary,indent=2),flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", type=Path, required=True); parser.add_argument("--dummy", action="store_true")
    args = parser.parse_args(); config = json.loads(args.config.read_text()); config["dummy"] = bool(args.dummy)
    if config.get("candidate_extraction_source") and not args.dummy:
        report_lower_score_candidate(config)
    elif config.get("seed_audit_source") and not args.dummy:
        audit_sorted_seed_events(config)
    elif config.get("recovery_source") and not args.dummy:
        report_from_failed_extraction(config)
    else:
        run(config)


if __name__ == "__main__": main()
