#!/usr/bin/env python
"""Correction-O scorer, held-out references, and no-sort MEDiCINe stage 2."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from testing import luke_imec1_medicine_reference_sweep_v1 as stage1
from testing import luke_imec1_medicine_m_reference_v1 as mref


PARENT = ROOT / "testing/outputs/luke_imec1_medicine_reference_sweep_v2"
DEFAULT_OUTPUT = PARENT / "stage2_o"
SESSION_DURATION_S = 10473.55
DEV_WINDOWS = dict(stage1.WINDOWS)
HELDOUT_CENTERS = {
    "h25": 0.25 * SESSION_DURATION_S,
    "h40": 0.40 * SESSION_DURATION_S,
    "h65": 0.65 * SESSION_DURATION_S,
    "h75": 0.75 * SESSION_DURATION_S,
}
HELDOUT_WINDOWS = {key: (center - 60.0, center + 60.0)
                   for key, center in HELDOUT_CENTERS.items()}
ALL_WINDOWS = {**DEV_WINDOWS, **HELDOUT_WINDOWS}
BLOCKS = [(0.0, 950.0), (950.0, 1900.0), (1900.0, 2850.0), (2850.0, 3840.0)]
CONFIGS = {
    "amp50": {"amplitude_threshold_quantile": 0.5},
    "amp25": {"amplitude_threshold_quantile": 0.75},
    "depth2": {"num_depth_bins": 2},
    "amp50_d2": {"amplitude_threshold_quantile": 0.5, "num_depth_bins": 2},
    "amp25_d2": {"amplitude_threshold_quantile": 0.75, "num_depth_bins": 2},
    "amp50_d1": {"amplitude_threshold_quantile": 0.5, "num_depth_bins": 1},
    "amp50_d2_s1": {"amplitude_threshold_quantile": 0.5, "num_depth_bins": 2, "seed": 1},
    "amp50_d2_k2": {"amplitude_threshold_quantile": 0.5, "num_depth_bins": 2,
                     "time_kernel_width": 2.0},
}
FIT_TIMEOUT_S = 900.0
EXTRACTION_TIMEOUT_S = 3600.0
GPU_BUDGET_S = 6 * 3600.0
MIN_QUIET_PAIRS = 3


def heldout_overlap_check() -> dict:
    result = {}
    for key, (start, stop) in HELDOUT_WINDOWS.items():
        overlaps = [name for name, (a, b) in DEV_WINDOWS.items()
                    if max(start, a) < min(stop, b)]
        result[key] = {"requested_center_s": HELDOUT_CENTERS[key],
                       "chosen_center_s": HELDOUT_CENTERS[key],
                       "adjustment_s": 0.0, "start_s": start, "stop_s": stop,
                       "overlapping_stage1_windows": overlaps}
        if overlaps:
            raise RuntimeError(f"Held-out window {key} overlaps {overlaps}")
    return result


def prepare(output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    (output / "runtime_cache/tmp").mkdir(parents=True)
    (output / "runtime_cache/numba").mkdir(parents=True)
    (output / "runtime_cache/mpl").mkdir(parents=True)
    source = json.loads((PARENT / "extraction_config.json").read_text())
    record = next(x for x in source["records"] if x["dataset"] == "Luke" and x["probe"] == "imec1")
    fs = float(record["sampling_frequency_hz"])
    windows = []
    for key, (start, stop) in HELDOUT_WINDOWS.items():
        windows.append({"id": key, "dataset": "Luke", "probe": "imec1", "fraction": None,
                        "start_frame": round(start * fs), "stop_frame": round(stop * fs),
                        "start_s": round(start * fs) / fs, "stop_s": round(stop * fs) / fs})
    source["windows"] = windows
    source["purpose"] = "Correction O held-out pilot-frontend caches; no sorting or voltage modification"
    stage1.atomic_json(output / "extraction_config.json", source)
    plan = {
        "schema": "luke0804-imec1-medicine-stage2-o-v1",
        "created_before_heldout_extraction_or_fit": True,
        "session_duration_s": SESSION_DURATION_S,
        "heldout_rule": "centres at 25/40/65/75% of 10473.55s; adjust <=60s only for stage1 overlap",
        "heldout_windows": heldout_overlap_check(),
        "candidate_union_heldout": {
            "field_p5": "empty: no field exists before held-out reference freeze",
            "invalid_eye": "unavailable for Luke0804",
            "low_rate": "raw pilot-frontend count <0.5x median; gaps <=1s joined",
        },
        "block_definition_um": BLOCKS,
        "block_peak_support": "nominal block expanded 150um at each edge",
        "block_acceptance": "global accepted episode; >=150 peaks in episode/rest; corr gain>=0.05",
        "corrected_quiet_inc": "5s endpoint differences only when both endpoints lie outside every candidate episode +/-2s",
        "minimum_quiet_pairs_per_window": MIN_QUIET_PAIRS,
        "configs": CONFIGS,
        "selection": "heldout only; quiet_abs<=10; quiet_inc<=8; ratio .8..1.2; block_err<=1.2x best; within 3um episode_err prefer fewer depth bins then lower amplitude quantile",
        "fit_timeout_s": FIT_TIMEOUT_S, "gpu_budget_s": GPU_BUDGET_S,
        "no_sort": True, "voltage_modified": False,
    }
    stage1.atomic_json(output / "preregistration.json", plan)


def heldout_population(output: Path, window: str) -> tuple[Path, float]:
    cfg = json.loads((output / "extraction_config.json").read_text())
    spec = next(x for x in cfg["windows"] if x["id"] == window)
    return output / f"peak_cache/{window}/extraction/population.npz", float(spec["start_s"])


def population(output: Path, window: str, *, reference: bool) -> tuple[Path, float]:
    if window in HELDOUT_WINDOWS:
        return heldout_population(output, window)
    return stage1.population_source(PARENT, window, 240 if reference else 120)


def peaks(output: Path, window: str) -> dict[str, np.ndarray]:
    path, start = population(output, window, reference=True)
    with np.load(path, allow_pickle=False) as data:
        if "x_um" not in data.files:
            raise RuntimeError(f"Reference cache lacks x_um: {path}")
        return {"time": np.asarray(data["time_s"], float) + start,
                "depth": np.asarray(data["depth_um"], float),
                "x": np.asarray(data["x_um"], float)}


def heldout_candidates(output: Path):
    import pandas as pd
    rows, bin_rows = [], []
    for window, (start, stop) in HELDOUT_WINDOWS.items():
        pk = peaks(output, window)
        centers = np.arange(start + stage1.BIN / 2, stop, stage1.BIN)
        counts = np.histogram(pk["time"], bins=np.r_[centers - stage1.BIN / 2, stop])[0]
        median = float(np.median(counts))
        mask = counts < 0.5 * median
        joined = mref.join_short_gaps(mask, round(mref.MAX_JOIN_GAP_S / stage1.BIN))
        for index in np.flatnonzero(mask):
            bin_rows.append({"window": window, "bin_center_s": centers[index],
                             "source": "rate_below_half_median", "value": counts[index],
                             "threshold": 0.5 * median})
        for episode_id, (i0, i1, a, b) in enumerate(mref.intervals(joined, centers)):
            rows.append({"window": window, "episode_id": episode_id,
                         "start_s": a, "stop_s": b, "duration_s": b - a,
                         "n_bins": i1 - i0, "field_p5_source": False,
                         "low_rate_source": bool(mask[i0:i1].any()),
                         "invalid_eye_source": False})
    return pd.DataFrame(rows), pd.DataFrame(bin_rows)


def measure_window(pk: dict[str, np.ndarray], candidates, bounds: tuple[float, float], seed: int):
    import pandas as pd
    start, stop = bounds
    centers = np.arange(start + stage1.BIN / 2, stop, stage1.BIN)
    active = np.zeros(len(centers), bool)
    for row in candidates.itertuples():
        active |= (centers >= row.start_s) & (centers < row.stop_s)
    peak_bins = np.floor((pk["time"] - start) / stage1.BIN).astype(int)
    in_window = (peak_bins >= 0) & (peak_bins < len(active))
    candidate_peak = np.zeros(len(pk["time"]), bool)
    candidate_peak[in_window] = active[peak_bins[in_window]]
    measured, nulls = [], []
    rng = np.random.default_rng(seed)
    for row in candidates.itertuples():
        episode = (pk["time"] >= row.start_s) & (pk["time"] < row.stop_s)
        rest = ((pk["time"] >= row.start_s - 4) & (pk["time"] < row.start_s - 1)
                & ~candidate_peak)
        result = mref.map_shift(pk, episode, rest)
        rest_in = row.start_s - 4 >= start
        accepted = (rest_in and result["episode_peaks"] >= mref.MIN_PEAKS
                    and result["rest_peaks"] >= mref.MIN_PEAKS
                    and np.isfinite(result["corr_gain"])
                    and result["corr_gain"] >= mref.MIN_CORR_GAIN)
        measured.append({"window": row.window, "episode_id": int(row.episode_id),
                         "start_s": row.start_s, "stop_s": row.stop_s,
                         "duration_s": row.duration_s,
                         **{k: v for k, v in result.items() if k != "correlations"},
                         "accepted": bool(accepted), "rest_fully_in_window": bool(rest_in)})
        duration_bins = int(row.n_bins)
        placements = []
        for i0 in range(round(4 / stage1.BIN), len(active) - duration_bins + 1):
            i1 = i0 + duration_bins
            r0, r1 = i0 - round(4 / stage1.BIN), i0 - round(1 / stage1.BIN)
            if active[i0:i1].any() or active[r0:r1].any():
                continue
            a, b = centers[i0] - stage1.BIN / 2, centers[i1 - 1] + stage1.BIN / 2
            ra, rb = a - 4, a - 1
            ne = int(((pk["time"] >= a) & (pk["time"] < b)).sum())
            nr = int(((pk["time"] >= ra) & (pk["time"] < rb)).sum())
            if ne >= result["episode_peaks"] and nr >= result["rest_peaks"]:
                placements.append((a, b, ra, rb))
        if not placements:
            nulls.append({"window": row.window, "matched_episode_id": int(row.episode_id),
                          "resolved": False, "reason": "no matched quiet placement"})
            continue
        a, b, ra, rb = placements[int(rng.integers(len(placements)))]
        ep = (pk["time"] >= a) & (pk["time"] < b)
        rest = (pk["time"] >= ra) & (pk["time"] < rb)
        null = mref.map_shift(pk, ep, rest, rng, result["episode_peaks"], result["rest_peaks"])
        nulls.append({"window": row.window, "matched_episode_id": int(row.episode_id),
                      "start_s": a, "stop_s": b, "rest_start_s": ra, "rest_stop_s": rb,
                      **{k: v for k, v in null.items() if k != "correlations"},
                      "resolved": bool(np.isfinite(null["best_shift_um"]))})
    return pd.DataFrame(measured), pd.DataFrame(nulls)


def validate_nulls(nulls, windows) -> dict:
    result = {"status": "pass", "criterion": "per-window mode=0um and median=0um", "windows": {}}
    for window in windows:
        values = nulls.loc[(nulls.window == window) & nulls.resolved, "best_shift_um"].to_numpy(float)
        if not len(values):
            result["windows"][window] = {"n": 0, "status": "fail_no_resolved_nulls"}
            result["status"] = "fail"
            continue
        unique, counts = np.unique(values, return_counts=True)
        mode, median = float(unique[np.argmax(counts)]), float(np.median(values))
        status = "pass" if mode == 0 and median == 0 else "fail"
        if status == "fail": result["status"] = "fail"
        result["windows"][window] = {"n": len(values), "mode_um": mode, "median_um": median,
            "exact_zero_fraction": float(np.mean(values == 0)), "status": status,
            "counts": {str(int(k)): int(v) for k, v in zip(unique, counts)}}
    return result


def block_references(output: Path, candidates, measured, windows) -> "object":
    import pandas as pd
    rows = []
    for window in windows:
        pk = peaks(output, window)
        group = candidates.loc[candidates.window == window]
        candidate_peak = np.zeros(len(pk["time"]), bool)
        for row in group.itertuples():
            candidate_peak |= (pk["time"] >= row.start_s) & (pk["time"] < row.stop_s)
        for episode in measured.loc[(measured.window == window) & measured.accepted].itertuples():
            for block_index, (low, high) in enumerate(BLOCKS):
                support = (pk["depth"] >= low - 150) & (pk["depth"] < high + 150)
                bp = {key: value[support] for key, value in pk.items()}
                ep = (bp["time"] >= episode.start_s) & (bp["time"] < episode.stop_s)
                original_indices = np.flatnonzero(support)
                rest_excluded = candidate_peak[original_indices]
                rest = ((bp["time"] >= episode.start_s - 4) &
                        (bp["time"] < episode.start_s - 1) & ~rest_excluded)
                value = mref.map_shift(bp, ep, rest)
                accepted = (value["episode_peaks"] >= mref.MIN_PEAKS and
                            value["rest_peaks"] >= mref.MIN_PEAKS and
                            np.isfinite(value["corr_gain"]) and
                            value["corr_gain"] >= mref.MIN_CORR_GAIN)
                rows.append({"window": window, "episode_id": int(episode.episode_id),
                             "start_s": episode.start_s, "stop_s": episode.stop_s,
                             "block_index": block_index, "block_low_um": low,
                             "block_high_um": high, "block_center_um": (low + high) / 2,
                             **{k: v for k, v in value.items() if k != "correlations"},
                             "accepted": bool(accepted)})
    return pd.DataFrame(rows)


def corrected_quiet_values(field: dict, candidates, bounds: tuple[float, float]) -> tuple[np.ndarray, int]:
    start, stop = bounds
    endpoints = np.arange(start, stop + 1e-6, 5.0)
    allowed = np.ones(len(endpoints), bool)
    for row in candidates.itertuples():
        allowed &= ~((endpoints >= row.start_s - 2) & (endpoints <= row.stop_s + 2))
    valid_pairs = allowed[:-1] & allowed[1:]
    depths = np.asarray(field["depth_um"], float)
    values = stage1.sample_field(field, *np.meshgrid(endpoints, depths, indexing="ij"))
    differences = np.diff(values, axis=0)[valid_pairs].ravel()
    return differences[np.isfinite(differences)], int(valid_pairs.sum())


def depth_prediction(field: dict, start: float, stop: float, depth: float) -> float:
    ep_t = np.arange(start + stage1.BIN / 2, stop, stage1.BIN)
    rest_t = np.arange(start - 4 + stage1.BIN / 2, start - 1, stage1.BIN)
    if not len(ep_t) or not len(rest_t): return np.nan
    # A one-bin field is rigid: its single trajectory applies at every probe
    # depth.  Sampling it at another block centre would otherwise be rejected
    # by the interpolation bounds and incorrectly yield a NaN block score.
    source_depths = np.asarray(field["depth_um"], float)
    sample_depth = float(source_depths[0]) if len(source_depths) == 1 else depth
    ep = stage1.sample_field(field, ep_t, np.full(len(ep_t), sample_depth))
    rest = stage1.sample_field(field, rest_t, np.full(len(rest_t), sample_depth))
    return float(np.nanmedian(ep) - np.nanmedian(rest))


def score_fields(field_root: Path, configs, windows, candidates, measured, nulls,
                 blocks, quiet_windows, prefix: Path):
    import pandas as pd
    score_rows, episode_rows, block_rows, quiet_pool, increment_pool = [], [], [], {}, {}
    for config in configs:
        quiet_pool[config], increment_pool[config] = [], []
        for window in windows:
            with np.load(field_root / f"{config}/{window}/field.npz", allow_pickle=False) as data:
                field = {key: np.asarray(data[key]) for key in data.files}
            ep_group = measured.loc[(measured.window == window) & measured.accepted]
            obs, pred = [], []
            for ep in ep_group.itertuples():
                value = mref.episode_prediction(field, ep.start_s, ep.stop_s)
                obs.append(ep.best_shift_um); pred.append(value)
                episode_rows.append({"config": config, "window": window,
                    "episode_id": int(ep.episode_id), "measured_shift_um": ep.best_shift_um,
                    "field_displacement_um": value, "abs_error_um": abs(value - ep.best_shift_um)})
            bg = blocks.loc[(blocks.window == window) & blocks.accepted]
            bo, bp = [], []
            for ref in bg.itertuples():
                value = depth_prediction(field, ref.start_s, ref.stop_s, ref.block_center_um)
                bo.append(ref.best_shift_um); bp.append(value)
                block_rows.append({"config": config, "window": window,
                    "episode_id": int(ref.episode_id), "block_index": int(ref.block_index),
                    "block_center_um": ref.block_center_um, "measured_shift_um": ref.best_shift_um,
                    "field_displacement_um": value, "abs_error_um": abs(value - ref.best_shift_um)})
            qvals = []
            for null in nulls.loc[(nulls.window == window) & nulls.resolved].itertuples():
                qvals.append(abs(mref.episode_prediction(field, null.start_s, null.stop_s)))
            quiet_pool[config].extend(x for x in qvals if np.isfinite(x))
            inc, pairs = corrected_quiet_values(field, candidates.loc[candidates.window == window],
                                                 ALL_WINDOWS[window])
            if window in quiet_windows and pairs >= MIN_QUIET_PAIRS:
                increment_pool[config].extend(inc.tolist())
            obs, pred, bo, bp = map(np.asarray, (obs, pred, bo, bp))
            good = np.isfinite(obs) & np.isfinite(pred)
            bgood = np.isfinite(bo) & np.isfinite(bp)
            score_rows.append({"config": config, "window": window,
                "episode_err": float(np.median(np.abs(pred[good] - obs[good]))) if good.any() else np.nan,
                "episode_ratio": float(np.median(pred[good] / obs[good])) if good.any() else np.nan,
                "accepted_episodes": int(good.sum()),
                "block_err": float(np.median(np.abs(bp[bgood] - bo[bgood]))) if bgood.any() else np.nan,
                "accepted_blocks": int(bgood.sum()),
                "quiet_abs": float(np.median(qvals)) if len(qvals) else np.nan,
                "null_episodes": len(qvals), "quiet_pairs": pairs,
                "quiet_inc": float(np.sqrt(np.mean(inc ** 2))) if pairs >= MIN_QUIET_PAIRS and len(inc) else np.nan})
    scores = pd.DataFrame(score_rows); episodes = pd.DataFrame(episode_rows); block_scores = pd.DataFrame(block_rows)
    aggregates = []
    for config in configs:
        e = episodes.loc[episodes.config == config].dropna()
        b = block_scores.loc[block_scores.config == config].dropna()
        ratios = e.field_displacement_um.to_numpy() / e.measured_shift_um.to_numpy()
        q, inc = np.asarray(quiet_pool[config]), np.asarray(increment_pool[config])
        aggregates.append({"config": config,
            "episode_err": float(np.median(e.abs_error_um)) if len(e) else np.nan,
            "episode_ratio": float(np.median(ratios)) if len(ratios) else np.nan,
            "accepted_episodes": len(e), "block_err": float(np.median(b.abs_error_um)) if len(b) else np.nan,
            "accepted_blocks": len(b), "quiet_abs": float(np.median(q)) if len(q) else np.nan,
            "quiet_inc": float(np.sqrt(np.mean(inc ** 2))) if len(inc) else np.nan,
            "quiet_increment_values": len(inc)})
    aggregate = pd.DataFrame(aggregates)
    scores.to_csv(prefix.with_name(prefix.name + "_by_window.csv"), index=False)
    aggregate.to_csv(prefix, index=False)
    episodes.to_csv(prefix.with_name(prefix.stem + "_episodes.csv"), index=False)
    block_scores.to_csv(prefix.with_name(prefix.stem + "_blocks.csv"), index=False)
    return scores, aggregate, episodes, block_scores


def reference_stage(output: Path) -> None:
    import pandas as pd
    if list((output / "fields").glob("*/*/receipt.json")) if (output / "fields").exists() else []:
        raise RuntimeError("Stage-2 fit exists before held-out reference freeze")
    for window in HELDOUT_WINDOWS:
        if not (output / f"peak_cache/{window}/extraction/complete.json").exists():
            raise RuntimeError(f"Missing held-out cache {window}")
    refdir = output / "references_heldout"; refdir.mkdir(exist_ok=False)
    candidates, bins = heldout_candidates(output)
    candidates.to_csv(refdir / "candidate_episodes.csv", index=False)
    bins.to_csv(refdir / "candidate_source_bins.csv", index=False)
    measured_parts, null_parts = [], []
    for index, window in enumerate(HELDOUT_WINDOWS):
        measured, nulls = measure_window(peaks(output, window),
            candidates.loc[candidates.window == window], HELDOUT_WINDOWS[window], 20260924 + index)
        measured_parts.append(measured); null_parts.append(nulls)
    measured = pd.concat(measured_parts, ignore_index=True)
    nulls = pd.concat(null_parts, ignore_index=True)
    measured.to_csv(refdir / "measured_episodes.csv", index=False)
    nulls.to_csv(refdir / "null_pseudo_episodes.csv", index=False)
    null_check = validate_nulls(nulls, HELDOUT_WINDOWS)
    stage1.atomic_json(refdir / "null_validation.json", null_check)
    if null_check["status"] != "pass":
        raise RuntimeError("Held-out null failed before any stage-2 fit")
    blocks = block_references(output, candidates, measured, HELDOUT_WINDOWS)
    blocks.to_csv(refdir / "block_measurements.csv", index=False)

    dev_candidates = pd.read_csv(PARENT / "references_m/candidate_episodes.csv")
    dev_measured = pd.read_csv(PARENT / "references_m/measured_episodes.csv")
    dev_nulls = pd.read_csv(PARENT / "references_m/null_pseudo_episodes.csv")
    dev_blocks = block_references(output, dev_candidates, dev_measured, DEV_WINDOWS)
    dev_blocks.to_csv(output / "stage1_block_measurements.csv", index=False)
    score_fields(PARENT / "fields", stage1.CONFIGS, DEV_WINDOWS, dev_candidates,
                 dev_measured, dev_nulls, dev_blocks, {"p50"}, output / "scores_stage1_rescored.csv")
    receipt = {"status": "complete_before_stage2_fit", "completed_at": time.time(),
        "heldout_candidates": len(candidates), "heldout_accepted": int(measured.accepted.sum()),
        "heldout_blocks_accepted": int(blocks.accepted.sum()), "null": null_check,
        "stage2_fit_receipts_at_freeze": 0,
        "artifact_sha256": {str(p.relative_to(output)): stage1.sha256(p) for p in
            [refdir / "candidate_episodes.csv", refdir / "measured_episodes.csv",
             refdir / "null_pseudo_episodes.csv", refdir / "block_measurements.csv",
             output / "scores_stage1_rescored.csv", output / "stage1_block_measurements.csv"]}}
    stage1.atomic_json(output / "references_complete_before_fit.json", receipt)


def fit_one(output: Path, config: str, window: str) -> None:
    import medicine
    import torch
    receipt = json.loads((output / "references_complete_before_fit.json").read_text())
    if receipt["status"] != "complete_before_stage2_fit":
        raise RuntimeError("References not frozen")
    target = output / f"fields/{config}/{window}"
    target.mkdir(parents=True, exist_ok=False)
    source, start = population(output, window, reference=False)
    with np.load(source, allow_pickle=False) as data:
        times, depths, amps = (np.asarray(data[k]) for k in ("time_s", "depth_um", "amplitude"))
    change = dict(CONFIGS[config]); seed = int(change.pop("seed", 0))
    settings = dict(stage1.BASE_MED); settings.update(change)
    torch.set_num_threads(4)
    if not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
    np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    begun = time.monotonic()
    trainer = medicine.run_medicine(peak_times=times, peak_depths=depths, peak_amplitudes=amps,
        output_dir=target / "medicine", optimizer=torch.optim.Adam, **settings)
    runtime = time.monotonic() - begun
    med = target / "medicine"
    ft, fz, fm = (np.load(med / name) for name in ("time_bins.npy", "depth_bins.npy", "motion.npy"))
    np.savez_compressed(target / "field.npz", time_s=ft, session_time_s=ft + start,
        depth_um=fz, displacement_um=fm,
        sign_contract=np.asarray("observed_minus_registered; corrected=observed-displacement"))
    np.save(target / "loss.npy", np.asarray(trainer.losses))
    stage1.atomic_json(target / "receipt.json", {"status": "complete", "config": config,
        "window": window, "runtime_s": runtime, "fit_start_s": start, "peaks": len(times),
        "settings": settings, "seed": seed, "source_population": str(source),
        "reference_receipt_sha256": stage1.sha256(output / "references_complete_before_fit.json"),
        "field_sha256": stage1.sha256(target / "field.npz"), "no_sort": True,
        "voltage_modified": False})


def sweep(output: Path) -> None:
    ref = output / "references_complete_before_fit.json"
    if not ref.exists(): raise RuntimeError("References must complete first")
    extraction_runtime = sum(json.loads((output / f"peak_cache/{w}/extraction/audit.json").read_text())["seconds"]
                             for w in HELDOUT_WINDOWS)
    runtime = float(extraction_runtime); completed = 0
    for config in CONFIGS:
        for window in ALL_WINDOWS:
            receipt = output / f"fields/{config}/{window}/receipt.json"
            if receipt.exists():
                runtime += json.loads(receipt.read_text())["runtime_s"]; completed += 1; continue
            if runtime >= GPU_BUDGET_S:
                raise RuntimeError("Stopped at O 6 GPU-hour budget")
            command = [str(stage1.MEDPY), str(Path(__file__).resolve()), "fit-one",
                       "--output", str(output), "--config", config, "--window", window]
            try:
                result = __import__("subprocess").run(command, timeout=FIT_TIMEOUT_S)
            except __import__("subprocess").TimeoutExpired:
                raise RuntimeError(f"Fit exceeded 15 minutes: {config}/{window}")
            if result.returncode: raise RuntimeError(f"Fit failed: {config}/{window}")
            runtime += json.loads(receipt.read_text())["runtime_s"]; completed += 1
    stage1.atomic_json(output / "sweep_complete.json", {"status": "complete", "fits": completed,
        "extraction_runtime_s": extraction_runtime, "budget_accounted_runtime_s": runtime,
        "budget_s": GPU_BUDGET_S})


def selection(aggregate) -> dict:
    pre = aggregate.loc[(aggregate.quiet_abs <= 10) & (aggregate.quiet_inc <= 8) &
                        aggregate.episode_ratio.between(.8, 1.2) & aggregate.block_err.notna()].copy()
    if not len(pre): return {"outcome": "no_selection", "reason": "no quiet/ratio candidates"}
    best_block = float(pre.block_err.min())
    eligible = pre.loc[pre.block_err <= 1.2 * best_block].copy()
    if not len(eligible): return {"outcome": "no_selection", "reason": "block gate"}
    best_episode = float(eligible.episode_err.min())
    tied = eligible.loc[eligible.episode_err <= best_episode + 3].copy()
    def simplicity(config):
        change = CONFIGS[config]
        return (change.get("num_depth_bins", 4), change.get("amplitude_threshold_quantile", 0),
                int(change.get("seed", 0) != 0), abs(change.get("time_kernel_width", 1) - 1), config)
    chosen = min(tied.config, key=simplicity)
    return {"outcome": "selected", "config": chosen, "best_block_err": best_block,
            "best_episode_err": best_episode, "within_3um": tied.config.tolist(),
            "eligible": eligible.config.tolist(),
            "tie_break": "fewer depth bins, then lower amplitude quantile, then seed0/kernel1"}


def seed_stability(output: Path) -> dict:
    rows, rms = [], []
    for window in ALL_WINDOWS:
        with np.load(output / f"fields/amp50_d2/{window}/field.npz") as a, np.load(
                output / f"fields/amp50_d2_s1/{window}/field.npz") as b:
            bt, bz = np.meshgrid(a["session_time_s"], a["depth_um"], indexing="ij")
            other = {k: np.asarray(b[k]) for k in ("session_time_s", "depth_um", "displacement_um")}
            delta = stage1.sample_field(other, bt, bz) - a["displacement_um"]
            value = float(np.sqrt(np.nanmean(delta ** 2))); rms.append(value)
            rows.append({"window": window, "field_rms_delta_um": value})
    return {"per_window": rows, "median_field_rms_delta_um": float(np.median(rms))}


def report(output: Path) -> None:
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ref = output / "references_heldout"
    hc = pd.read_csv(ref / "candidate_episodes.csv"); hm = pd.read_csv(ref / "measured_episodes.csv")
    hn = pd.read_csv(ref / "null_pseudo_episodes.csv"); hb = pd.read_csv(ref / "block_measurements.csv")
    dc = pd.read_csv(PARENT / "references_m/candidate_episodes.csv")
    dm = pd.read_csv(PARENT / "references_m/measured_episodes.csv")
    dn = pd.read_csv(PARENT / "references_m/null_pseudo_episodes.csv")
    db = pd.read_csv(output / "stage1_block_measurements.csv")
    _, devagg, _, _ = score_fields(output / "fields", CONFIGS, DEV_WINDOWS, dc, dm, dn, db,
                                    {"p50"}, output / "scores_stage2_development.csv")
    _, heldagg, _, heldblocks = score_fields(output / "fields", CONFIGS, HELDOUT_WINDOWS, hc, hm, hn, hb,
                                              set(HELDOUT_WINDOWS), output / "scores_stage2_heldout.csv")
    select = selection(heldagg); stage1.atomic_json(output / "selection.json", select)
    stability = seed_stability(output); stage1.atomic_json(output / "seed_stability.json", stability)

    palette = ["#0077BB", "#33BBEE", "#009988", "#EE7733", "#CC3311", "#AA3377", "#BBBBBB", "#000000"]
    for window in HELDOUT_WINDOWS:
        fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")
        refs = hb.loc[(hb.window == window) & hb.accepted]
        for ep, group in refs.groupby("episode_id"):
            ax.plot(group.block_center_um, group.best_shift_um, color="0.72", lw=.8,
                    marker="o", ms=3, alpha=.7)
        for color, config in zip(palette, CONFIGS):
            group = heldblocks.loc[(heldblocks.config == config) & (heldblocks.window == window)]
            med = group.groupby("block_center_um").field_displacement_um.median()
            ax.plot(med.index, med.values, marker="o", lw=1.4, color=color, label=config)
        ax.axhline(0, color="0.3", lw=.7); ax.set(xlabel="Depth-block centre (µm)",
            ylabel="Rest-centred displacement (µm)", title=f"{window}: held-out per-depth episode shifts")
        ax.legend(ncol=4, fontsize=8); fig.savefig(output / f"figure_{window}_per_depth.png", dpi=180); plt.close(fig)

    archive = output / "stage2_fields.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for config in CONFIGS:
            for window in ALL_WINDOWS:
                path = output / f"fields/{config}/{window}/field.npz"
                z.write(path, arcname=f"{config}/{window}/field.npz")
    null_check = json.loads((ref / "null_validation.json").read_text())
    counts = hm.groupby("window").accepted.agg(["sum", "count"])
    selected_text = (f"Selected `{select['config']}`." if select["outcome"] == "selected" else
                     f"No selection: {select['reason']}.")
    readme = ["# Luke0804 imec1 MEDiCINe correction O / stage 2", "",
        "No spike sort or voltage modification was run. Held-out references were hash-frozen before the first stage-2 fit.", "",
        "## O.1 corrected stage-1 scorer", "",
        "`quiet_inc` excludes every 5 s increment unless both endpoints lie outside all candidate episodes expanded by ±2 s. `block_err` uses four nominal blocks (0–950, 950–1900, 1900–2850, 2850–3840 µm), raw peaks from each block expanded by 150 µm, and gain ≥0.05.",
        "See `scores_stage1_rescored.csv` and `stage1_block_measurements.csv`.", "",
        "## Held-out references", "",
        "Centres were fixed at 25%, 40%, 65%, and 75% of 10,473.55 s; none overlapped stage 1, so all adjustments were 0 s. Because references had to precede all stage-2 fits, field-P5 candidates were unavailable by design; eye validity was also unavailable. Candidate runs therefore use the frozen raw-peak low-rate component only.",
        f"Null validation: **{null_check['status']}**.", ""]
    readme.extend([f"- {w}: {int(counts.loc[w, 'sum'])}/{int(counts.loc[w, 'count'])} accepted; null {null_check['windows'][w]}" for w in HELDOUT_WINDOWS])
    readme.extend(["", "## Held-out-only selection", "", selected_text,
        f"Seed stability median field RMS (amp50_d2 seed0 vs seed1): {stability['median_field_rms_delta_um']:.3f} µm.",
        "Development windows were not used for selection. See `scores_stage2_development.csv`, `scores_stage2_heldout.csv`, and `selection.json`.", "",
        "## Artifacts", "", "- `scores_stage1_rescored.csv`", "- `scores_stage2_development.csv`",
        "- `scores_stage2_heldout.csv`", "- `references_heldout/`", "- `figure_h*_per_depth.png`",
        "- `stage2_fields.zip`", ""])
    (output / "README.md").write_text("\n".join(readme))
    with zipfile.ZipFile(archive) as z: names = z.namelist()
    if len(names) != len(CONFIGS) * len(ALL_WINDOWS): raise RuntimeError("Stage2 archive incomplete")
    validation = {"status": "complete", "fits": len(names), "unique_archive_fields": len(set(names)),
        "heldout_reference_frozen_before_fit": True, "heldout_candidates": len(hc),
        "heldout_accepted": int(hm.accepted.sum()), "heldout_blocks": int(hb.accepted.sum()),
        "selection": select, "seed_stability": stability,
        "artifacts": {str(p.relative_to(output)): stage1.sha256(p) for p in [output / "README.md",
            output / "scores_stage1_rescored.csv", output / "scores_stage2_development.csv",
            output / "scores_stage2_heldout.csv", ref / "measured_episodes.csv",
            ref / "null_pseudo_episodes.csv", ref / "block_measurements.csv", archive]}}
    stage1.atomic_json(output / "validation.json", validation)


def reference_all(output: Path) -> None:
    cfg = json.loads((output / "extraction_config.json").read_text())
    import subprocess
    for spec in cfg["windows"]:
        stage = output / f"peak_cache/{spec['id']}/extraction"
        if (stage / "complete.json").exists(): continue
        command = [str(stage1.DSPY), str(Path(__file__).resolve()), "extract-one",
                   "--output", str(output), "--window", spec["id"]]
        result = subprocess.run(command, timeout=EXTRACTION_TIMEOUT_S)
        if result.returncode: raise RuntimeError(f"Extraction failed: {spec['id']}")
    reference_stage(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["prepare", "extract-one", "references", "reference-all",
                                          "fit-one", "sweep", "report"])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--window"); parser.add_argument("--config")
    args = parser.parse_args(); output = args.output.resolve()
    if args.phase == "prepare": prepare(output)
    elif args.phase == "extract-one": stage1.extract_one(output, args.window)
    elif args.phase == "references": reference_stage(output)
    elif args.phase == "reference-all": reference_all(output)
    elif args.phase == "fit-one": fit_one(output, args.config, args.window)
    elif args.phase == "sweep": sweep(output)
    else: report(output)


if __name__ == "__main__":
    main()
