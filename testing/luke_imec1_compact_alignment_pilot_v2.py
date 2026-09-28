#!/usr/bin/env python
"""Bounded compact-template alignment and identity-evidence pilot for imec1.

Training-only operations use 300--310 s. Qualification is frozen at 310--320 s.
The arms are broad, compact without applying member lags, and compact with
member lags applied before averaging. Sorter labels, motion, and absolute depth
are excluded from proposal construction and identity scoring.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from testing.luke_imec1_compact_core_pilot_v1 import make_template
from testing.luke_imec1_dots_lighthouse_direct_check import atomic_json
from testing.luke_imec1_dots_raw_lighthouse_check import (
    MANIFEST, geometry_and_mapping, patches, preprocess, raw_sha_contract,
)
from testing.luke_imec1_dots_sorterfree_waveform_discovery import (
    DISPLAY_COSINE, IDENTITY_MARGIN, STRICT_COSINE, detect_waveforms,
    normalize_waveforms,
)
from testing.luke_imec1_dots_sorterfree_waveform_discovery_v3 import (
    GAIN_RANGE, MIN_OVERLAP_ENERGY, classify_scores,
)
from testing.luke_imec1_sorterfree_phase_audit_v1 import overlap_indices

SCHEMA = "luke0804-imec1-compact-alignment-pilot-v2"
TRAIN = (300.0, 310.0)
QUALIFY = (310.0, 320.0)
CORE_COSINE = 0.86
MIN_HALF_CORE = 5
REPRODUCIBLE_TEMPLATE_COSINE = 0.90


def shift_waveforms(waves: np.ndarray, lags: np.ndarray) -> np.ndarray:
    """Apply score-compatible integer lags with zero padding."""
    shifted = np.zeros_like(waves)
    for lag in np.unique(lags).astype(int):
        rows = np.flatnonzero(lags == lag)
        if lag < 0:
            shifted[rows, :lag] = waves[rows, -lag:]
        elif lag > 0:
            shifted[rows, lag:] = waves[rows, :-lag]
        else:
            shifted[rows] = waves[rows]
    return shifted


def best_lag_scores(waves: np.ndarray, template: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    target = template.reshape(-1)
    target = target / max(np.linalg.norm(target), 1e-20)
    scores = np.full(len(waves), -np.inf, np.float32)
    lags = np.zeros(len(waves), np.int8)
    for lag in range(-3, 4):
        candidate = normalize_waveforms(shift_waveforms(waves, np.full(len(waves), lag))) @ target
        better = candidate > scores
        scores[better] = candidate[better]
        lags[better] = lag
    return scores, lags


def split_core(waves: np.ndarray, left: np.ndarray, right: np.ndarray) -> dict:
    """Select and independently construct two reciprocal training-only cores."""
    if min(len(left), len(right)) < MIN_HALF_CORE:
        return {"valid": False, "left_ids": np.array([], int), "right_ids": np.array([], int)}
    left_broad, _ = make_template(waves[left])
    right_broad, _ = make_template(waves[right])
    left_score, left_lag = best_lag_scores(waves[left], right_broad)
    right_score, right_lag = best_lag_scores(waves[right], left_broad)
    keep_left = left_score >= CORE_COSINE
    keep_right = right_score >= CORE_COSINE
    left_ids, right_ids = left[keep_left], right[keep_right]
    out = {
        "valid": min(len(left_ids), len(right_ids)) >= MIN_HALF_CORE,
        "left_ids": left_ids,
        "right_ids": right_ids,
        "left_lags": left_lag[keep_left],
        "right_lags": right_lag[keep_right],
    }
    if not out["valid"]:
        return out
    left_aligned = shift_waveforms(waves[left_ids], out["left_lags"])
    right_aligned = shift_waveforms(waves[right_ids], out["right_lags"])
    left_compact, _ = make_template(left_aligned)
    right_compact, _ = make_template(right_aligned)
    similarity, template_lag = best_lag_scores(right_compact[None], left_compact)
    reciprocal_left, _ = best_lag_scores(waves[left], right_compact)
    reciprocal_right, _ = best_lag_scores(waves[right], left_compact)
    out.update({
        "left_template": left_compact,
        "right_template": right_compact,
        "template_cosine": float(similarity[0]),
        "template_lag": int(template_lag[0]),
        "reciprocal_left": int((reciprocal_left >= CORE_COSINE).sum()),
        "reciprocal_right": int((reciprocal_right >= CORE_COSINE).sum()),
    })
    return out


def build_arms(found, waves, membership, families, fs):
    train_stop = round((TRAIN[1] - TRAIN[0]) * fs)
    rows, broad_bank, compact_bank, aligned_bank, broad_scale, compact_scale, aligned_scale = ([] for _ in range(7))
    for family in families.itertuples():
        ids = membership.loc[membership.family_id.eq(family.family_id), "event_index"].to_numpy(int)
        ids = ids[found[ids, 0] < train_stop]
        if len(ids) < 20:
            continue
        broad, bs = make_template(waves[ids])
        odd_even = split_core(waves, ids[::2], ids[1::2])
        midpoint = round((TRAIN[1] - TRAIN[0]) * fs / 2)
        time_block = split_core(waves, ids[found[ids, 0] < midpoint], ids[found[ids, 0] >= midpoint])
        if odd_even["valid"]:
            core_ids = np.r_[odd_even["left_ids"], odd_even["right_ids"]]
            compact, cs = make_template(waves[core_ids])
            # Put every selected member onto one training-only canonical reference.
            canonical = odd_even["left_template"]
            _, canonical_lags = best_lag_scores(waves[core_ids], canonical)
            aligned, als = make_template(shift_waveforms(waves[core_ids], canonical_lags))
        else:
            core_ids = np.array([], int)
            compact, cs, aligned, als = broad, bs, broad, bs
        rows.append({
            "family_id": family.family_id,
            "phase_id": int(family.phase_id),
            "proposal_train_events": len(ids),
            "odd_even_core_events": len(core_ids),
            "odd_even_core_fraction": len(core_ids) / len(ids),
            "odd_even_valid": bool(odd_even["valid"]),
            "odd_even_compact_template_cosine": odd_even.get("template_cosine", np.nan),
            "odd_even_compact_template_lag": odd_even.get("template_lag", np.nan),
            "odd_even_reciprocal_left": odd_even.get("reciprocal_left", 0),
            "odd_even_reciprocal_right": odd_even.get("reciprocal_right", 0),
            "time_block_valid": bool(time_block["valid"]),
            "time_block_left_core_events": len(time_block["left_ids"]),
            "time_block_right_core_events": len(time_block["right_ids"]),
            "time_block_compact_template_cosine": time_block.get("template_cosine", np.nan),
            "time_block_reciprocal_left": time_block.get("reciprocal_left", 0),
            "time_block_reciprocal_right": time_block.get("reciprocal_right", 0),
            "reproduces_odd_even": bool(odd_even["valid"] and odd_even.get("template_cosine", -np.inf) >= REPRODUCIBLE_TEMPLATE_COSINE),
            "reproduces_time_blocks": bool(time_block["valid"] and time_block.get("template_cosine", -np.inf) >= REPRODUCIBLE_TEMPLATE_COSINE),
        })
        broad_bank.append(broad); compact_bank.append(compact); aligned_bank.append(aligned)
        broad_scale.append(bs); compact_scale.append(cs); aligned_scale.append(als)
    return (
        pd.DataFrame(rows),
        {"broad": np.asarray(broad_bank), "compact": np.asarray(compact_bank), "compact_aligned": np.asarray(aligned_bank)},
        {"broad": np.asarray(broad_scale), "compact": np.asarray(compact_scale), "compact_aligned": np.asarray(aligned_scale)},
    )


def score_global_explained(waves, event_phases, templates, template_phases, scales, relative_geometry, block_size=256):
    """v3 scoring plus winning lag and runner-up identity for event explanations."""
    n_events, n_templates = len(waves), len(templates)
    keys = ("winner_template", "winner_kind", "runner_template", "runner_kind")
    result = {key: np.full(n_events, -1, int) for key in keys}
    result.update(score=np.full(n_events, -np.inf, np.float32), runner_score=np.full(n_events, -np.inf, np.float32),
                  margin=np.full(n_events, np.nan, np.float32), gain=np.full(n_events, np.nan, np.float32),
                  best_lag=np.zeros(n_events, np.int8))
    raw_templates = templates.astype(np.float32) * scales[:, None, None]
    banks = (raw_templates, raw_templates[:, ::-1], raw_templates[:, :, ::-1])
    for event_phase in np.unique(event_phases).astype(int):
        event_index = np.flatnonzero(event_phases == event_phase)
        for start in range(0, len(event_index), block_size):
            rows = event_index[start:start + block_size]
            scores = np.full((len(rows), 3, n_templates), -np.inf, np.float32)
            gains = np.full_like(scores, np.nan)
            lags = np.zeros_like(scores, np.int8)
            for template_phase in np.unique(template_phases).astype(int):
                columns = np.flatnonzero(template_phases == template_phase)
                event_channels, template_channels = overlap_indices(relative_geometry, event_phase, template_phase)
                if len(event_channels) < 8:
                    continue
                event_energy = np.square(waves[rows]).sum(axis=(1, 2))
                event_overlap = np.square(waves[rows][:, :, event_channels]).sum(axis=(1, 2))
                event_supported = event_overlap / np.maximum(event_energy, 1e-20) >= MIN_OVERLAP_ENERGY
                for kind, bank in enumerate(banks):
                    target = bank[columns][:, :, template_channels].reshape(len(columns), -1)
                    target_energy = np.square(bank[columns]).sum(axis=(1, 2))
                    target_overlap = np.square(target).sum(axis=1)
                    target_supported = target_overlap / np.maximum(target_energy, 1e-20) >= MIN_OVERLAP_ENERGY
                    target_norm = np.sqrt(target_overlap)
                    for lag in range(-3, 4):
                        observed = shift_waveforms(waves[rows], np.full(len(rows), lag))[:, :, event_channels].reshape(len(rows), -1)
                        dots = observed @ target.T
                        candidate_scores = dots / np.maximum(np.linalg.norm(observed, axis=1)[:, None] * target_norm[None], 1e-20)
                        candidate_gains = dots / np.maximum(target_overlap[None], 1e-20)
                        candidate_scores[~(event_supported[:, None] & target_supported[None])] = -np.inf
                        old = scores[:, kind, columns]
                        better = candidate_scores > old
                        scores[:, kind, columns] = np.where(better, candidate_scores, old)
                        gains[:, kind, columns] = np.where(better, candidate_gains, gains[:, kind, columns])
                        lags[:, kind, columns] = np.where(better, lag, lags[:, kind, columns])
            flat = scores.reshape(len(rows), -1)
            order = np.argsort(flat, axis=1, kind="stable")
            rr = np.arange(len(rows)); winner = order[:, -1]; runner = order[:, -2]
            wk, wt = winner // n_templates, winner % n_templates
            result["winner_kind"][rows] = wk; result["winner_template"][rows] = wt
            result["runner_kind"][rows] = runner // n_templates; result["runner_template"][rows] = runner % n_templates
            result["score"][rows] = flat[rr, winner]; result["runner_score"][rows] = flat[rr, runner]
            result["margin"][rows] = flat[rr, winner] - flat[rr, runner]
            result["gain"][rows] = gains[rr, wk, wt]; result["best_lag"][rows] = lags[rr, wk, wt]
    return result


def qualification_table(arm, scored, combined, found, waves, geom, patch_channels, fs):
    status = classify_scores(scored)
    winner = scored["winner_template"]
    runner = scored["runner_template"]
    centroids = np.empty(len(waves), float)
    for event in range(len(waves)):
        bi = int(found[event, 2]); energy = np.square(waves[event].astype(float)).sum(axis=0)
        centroids[event] = energy @ geom[patch_channels[bi], 1] / max(energy.sum(), 1e-20)
    return pd.DataFrame({
        "event_id": np.arange(len(waves)),
        "time_s": QUALIFY[0] + found[:, 0] / fs,
        "detector_phase": found[:, 3].astype(int),
        "peak_channel": found[:, 1].astype(int),
        "peak_snr": found[:, 4],
        "arm": arm,
        "winner_bank": combined.bank.to_numpy()[winner],
        "winner_family": combined.family_id.to_numpy()[winner],
        "winner_kind": scored["winner_kind"],
        "score": scored["score"],
        "margin": scored["margin"],
        "gain": scored["gain"],
        "best_lag": scored["best_lag"],
        "status": status,
        "runner_bank": combined.bank.to_numpy()[runner],
        "runner_family": combined.family_id.to_numpy()[runner],
        "runner_kind": scored["runner_kind"],
        "runner_score": scored["runner_score"],
        "waveform_centroid_um": centroids,
    })


def change_reason(old: pd.Series, new: pd.Series) -> str:
    if new.status == "strict" and old.status != "strict":
        direction = "gained"
        failing = old
    elif old.status == "strict" and new.status != "strict":
        direction = "lost"
        failing = new
    elif old.status == new.status == "strict" and old.winner_family != new.winner_family:
        return "strict_identity_reassigned"
    else:
        return "other_change"
    if failing.winner_kind > 0:
        cause = "from_decoy"
    elif old.winner_family != new.winner_family or old.winner_bank != new.winner_bank:
        cause = "identity_reassigned"
    elif not (GAIN_RANGE[0] <= failing.gain <= GAIN_RANGE[1]):
        cause = "gain_gate"
    elif failing.score < STRICT_COSINE:
        cause = "score_gate"
    elif failing.margin < IDENTITY_MARGIN:
        cause = "margin_gate"
    else:
        cause = "combined_gate_change"
    return direction + "_" + cause


def comparison_ledger(old: pd.DataFrame, new: pd.DataFrame, label: str) -> pd.DataFrame:
    a = old.set_index("event_id"); b = new.set_index("event_id")
    changed = (a.status != b.status) | (a.winner_family != b.winner_family) | (a.winner_bank != b.winner_bank)
    rows = []
    for event in a.index[changed]:
        x, y = a.loc[event], b.loc[event]
        rows.append({
            "comparison": label, "event_id": event, "time_s": y.time_s,
            "detector_phase": y.detector_phase, "waveform_centroid_um": y.waveform_centroid_um,
            "reason": change_reason(x, y),
            "old_bank": x.winner_bank, "old_family": x.winner_family, "old_kind": x.winner_kind,
            "old_status": x.status, "old_score": x.score, "old_margin": x.margin, "old_gain": x.gain,
            "old_lag": x.best_lag, "old_runner_family": x.runner_family,
            "new_bank": y.winner_bank, "new_family": y.winner_family, "new_kind": y.winner_kind,
            "new_status": y.status, "new_score": y.score, "new_margin": y.margin, "new_gain": y.gain,
            "new_lag": y.best_lag, "new_runner_family": y.runner_family,
        })
    return pd.DataFrame(rows)


def run(config):
    output = Path(config["output"]); output.mkdir(parents=True, exist_ok=False); atomic_json(output / "settings.json", config)
    manifest = json.loads(MANIFEST.read_text()); binary = Path(manifest["binary_path"]); fs = float(manifest["sampling_frequency_hz"])
    before = raw_sha_contract(binary)
    raw = np.memmap(binary, dtype="<i2", mode="r", shape=(manifest["num_samples"], manifest["saved_channels_in_binary"]))
    geom, kept = geometry_and_mapping(); bases, patch_channels = patches(geom)
    x = preprocess(raw, round(TRAIN[0] * fs), round(QUALIFY[1] * fs), kept, fs)
    found, waves = detect_waveforms(x, fs, geom, bases, patch_channels)
    source, external = Path(config["source_bank"]), Path(config["external_bank"])
    expected = json.loads((source / "summary.json").read_text())["seed_detections"]
    if len(found) != expected:
        raise RuntimeError(f"detection replay mismatch {len(found)} != {expected}")
    membership = pd.read_csv(source / "seed_cluster_events.csv")[["event_index", "family_id"]]
    families = pd.read_csv(source / "families_after_depth_reveal.csv")
    proposal, templates, scales = build_arms(found, waves, membership, families, fs)
    proposal.to_csv(output / "training_core_reproducibility.csv", index=False)
    np.savez_compressed(output / "frozen_training_templates.npz", family_id=proposal.family_id.to_numpy(),
                        phase_id=proposal.phase_id.to_numpy(), **templates, **{f"{k}_scale": v for k, v in scales.items()})

    external_families = pd.read_csv(external / "families_after_depth_reveal.csv")[["family_id", "phase_id"]]
    with np.load(external / "family_templates.npz") as saved:
        external_templates = saved["waveforms"]; external_scales = saved["template_scale_uv"]
        relative_geometry = saved["relative_geometry"]
    own = proposal[["family_id", "phase_id"]].copy(); own["bank"] = "s300"
    ext = external_families.copy(); ext["bank"] = "s036"
    combined = pd.concat([own, ext], ignore_index=True)
    qualify = found[:, 0] >= round((QUALIFY[0] - TRAIN[0]) * fs)
    qfound = found[qualify].copy(); qfound[:, 0] -= round((QUALIFY[0] - TRAIN[0]) * fs); qwaves = waves[qualify]
    tables = {}
    for arm in ("broad", "compact", "compact_aligned"):
        bank = np.concatenate([templates[arm], external_templates])
        bank_scale = np.r_[scales[arm], external_scales]
        scored = score_global_explained(qwaves, qfound[:, 3].astype(int), bank,
                                        combined.phase_id.to_numpy(int), bank_scale, relative_geometry)
        tables[arm] = qualification_table(arm, scored, combined, qfound, qwaves, geom, patch_channels, fs)
        tables[arm].to_csv(output / f"qualification_events_{arm}.csv", index=False)
    ledgers = pd.concat([
        comparison_ledger(tables["broad"], tables["compact_aligned"], "broad_to_compact_aligned"),
        comparison_ledger(tables["compact"], tables["compact_aligned"], "compact_to_compact_aligned"),
    ], ignore_index=True)
    ledgers.to_csv(output / "event_level_changes.csv", index=False)
    counters = []
    for arm, table in tables.items():
        counters.append({
            "arm": arm, "detections": len(table), "strict_all": int(table.status.eq("strict").sum()),
            "strict_own": int((table.status.eq("strict") & table.winner_bank.eq("s300")).sum()),
            "strict_external": int((table.status.eq("strict") & table.winner_bank.eq("s036")).sum()),
            "decoy_winners": int(table.status.eq("decoy_winner").sum()),
        })
    pd.DataFrame(counters).to_csv(output / "arm_counters.csv", index=False)
    reason_counts = ledgers.groupby(["comparison", "reason"]).size().rename("events").reset_index()
    reason_counts.to_csv(output / "event_change_reason_counts.csv", index=False)
    if raw_sha_contract(binary) != before:
        raise RuntimeError("raw source changed")
    summary = {
        "schema": SCHEMA, "status": "complete", "raw_voltage_read": True,
        "motion_estimate_read": False, "sorter_input_read": False,
        "train_s": TRAIN, "qualification_s": QUALIFY,
        "qualification_event_count": len(qwaves), "proposal_families": len(proposal),
        "odd_even_valid_cores": int(proposal.odd_even_valid.sum()),
        "odd_even_reproduced_templates": int(proposal.reproduces_odd_even.sum()),
        "time_block_valid_cores": int(proposal.time_block_valid.sum()),
        "time_block_reproduced_templates": int(proposal.reproduces_time_blocks.sum()),
        "arms": {row["arm"]: {k: int(v) for k, v in row.items() if k not in {"arm"}}
                 for row in counters},
        "rate_contract": "Training support rates and qualification match rates are reported separately; neither is called recovery probability.",
        "qualification_independence_limit": "Existing proposal assignments saw the full 300--320 s interval; only template construction and scoring are split.",
    }
    atomic_json(output / "summary.json", summary); print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(); run(json.loads(args.config.read_text()))


if __name__ == "__main__":
    main()
