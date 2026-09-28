#!/usr/bin/env python
"""Revised sorter-free imec1 lighthouse discovery using lessons from imec0.

Changes from v2:
- all real and decoy templates compete across detector phases;
- cross-phase comparisons use exact peak-centered relative geometry;
- hypotheses with <50% common waveform energy are unsupported;
- physical amplitude gain 0.35--3 is enforced;
- seed-event recovery is required before strict candidate qualification;
- complete evidence remains depth blind until the candidate panel is frozen.

This remains candidate discovery, not cell certification or motion estimation.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from testing.luke_imec1_dots_lighthouse_direct_check import MOTION_WINDOWS, SEED_INTERVAL_S, atomic_json, sha256
from testing.luke_imec1_dots_raw_lighthouse_check import MANIFEST, geometry_and_mapping, patches, preprocess, raw_sha_contract
from testing.luke_imec1_dots_sorterfree_waveform_discovery import (
    DISPLAY_COSINE,
    IDENTITY_MARGIN,
    MIN_FAMILY_EVENTS,
    MIN_HALF_EVENTS,
    MIN_MEDIAN_SNR,
    MIN_MEMBER_COSINE,
    MIN_REPEATABILITY,
    N_SELECTED,
    OFFSETS,
    STRICT_COSINE,
    add_seed_depth_after_freeze,
    build_families,
    detect_waveforms,
    plot_candidate_review,
)
from testing.luke_imec1_sorterfree_phase_audit_v1 import similarity_matrices, overlap_indices

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "luke0804-imec1-dots-sorterfree-waveform-discovery-v3"
DEFAULT_OUTPUT = ROOT / "testing/outputs/luke_imec1_dots_sorterfree_waveform_discovery_v3"
MIN_OVERLAP_ENERGY = 0.50
GAIN_RANGE = (0.35, 3.0)
MIN_SEED_RECOVERY_EVENTS = 5
MIN_SEED_RECOVERY_FRACTION = 0.20


def template_scales(membership: pd.DataFrame, waves: np.ndarray, family_ids: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(waves.reshape(len(waves), -1), axis=1)
    mapped = membership.assign(waveform_norm_uv=norms[membership.event_index.to_numpy(int)])
    medians = mapped.groupby("family_id").waveform_norm_uv.median()
    return np.asarray([medians[fid] for fid in family_ids], dtype=np.float32)


def global_rivals(families: pd.DataFrame, templates: np.ndarray, relative_geometry: np.ndarray) -> pd.DataFrame:
    cosine, coverage, common = similarity_matrices(templates, families.phase_id.to_numpy(int), relative_geometry)
    supported = coverage >= MIN_OVERLAP_ENERGY
    np.fill_diagonal(supported, False)
    eligible_scores = np.where(supported, cosine, -np.inf)
    rivals = np.argmax(eligible_scores, axis=1)
    out = families.copy()
    out["local_phase_rival_cosine_v2"] = out.nearest_rival_cosine
    out["global_rival_id"] = out.family_id.to_numpy()[rivals]
    out["global_rival_phase"] = out.phase_id.to_numpy()[rivals]
    out["global_rival_cosine"] = eligible_scores[np.arange(len(out)), rivals]
    out["global_rival_overlap_energy"] = coverage[np.arange(len(out)), rivals]
    out["global_rival_common_channels"] = common[np.arange(len(out)), rivals]
    out["unsupported_cross_phase_rivals"] = [
        int((~supported[i] & (out.phase_id.to_numpy() != out.phase_id.iloc[i])).sum()) for i in range(len(out))
    ]
    # The shared review renderer labels this legacy field "nearest rival".
    # In v3 it must display the global supported rival, not the v2 phase-local one.
    out["nearest_rival_cosine"] = out.global_rival_cosine
    return out


def score_global(
    waves: np.ndarray,
    event_phases: np.ndarray,
    templates: np.ndarray,
    template_phases: np.ndarray,
    scales: np.ndarray,
    relative_geometry: np.ndarray,
    block_size: int = 256,
) -> dict[str, np.ndarray]:
    """Score real, time-reversed, and channel-reversed hypotheses globally."""
    n_events, n_templates = len(waves), len(templates)
    result = {
        "winner_template": np.full(n_events, -1, int),
        "winner_kind": np.full(n_events, -1, int),
        "score": np.full(n_events, -np.inf, np.float32),
        "margin": np.full(n_events, np.nan, np.float32),
        "gain": np.full(n_events, np.nan, np.float32),
    }
    raw_templates = templates.astype(np.float32) * scales[:, None, None]
    banks = (raw_templates, raw_templates[:, ::-1], raw_templates[:, :, ::-1])
    for event_phase in np.unique(event_phases).astype(int):
        event_index = np.flatnonzero(event_phases == event_phase)
        for start in range(0, len(event_index), block_size):
            rows = event_index[start:start + block_size]
            scores = np.full((len(rows), 3, n_templates), -np.inf, np.float32)
            gains = np.full_like(scores, np.nan)
            for template_phase in np.unique(template_phases).astype(int):
                columns = np.flatnonzero(template_phases == template_phase)
                event_channels, template_channels = overlap_indices(
                    relative_geometry, event_phase, template_phase
                )
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
                        shifted = np.zeros_like(waves[rows])
                        if lag < 0:
                            shifted[:, :lag] = waves[rows][:, -lag:]
                        elif lag > 0:
                            shifted[:, lag:] = waves[rows][:, :-lag]
                        else:
                            shifted = waves[rows]
                        observed = shifted[:, :, event_channels].reshape(len(rows), -1)
                        dots = observed @ target.T
                        denominator = np.linalg.norm(observed, axis=1)[:, None] * target_norm[None]
                        candidate_scores = dots / np.maximum(denominator, 1e-20)
                        candidate_gains = dots / np.maximum(target_overlap[None], 1e-20)
                        valid = event_supported[:, None] & target_supported[None]
                        candidate_scores[~valid] = -np.inf
                        old = scores[:, kind, columns]
                        better = candidate_scores > old
                        scores[:, kind, columns] = np.where(better, candidate_scores, old)
                        gains[:, kind, columns] = np.where(better, candidate_gains, gains[:, kind, columns])
            flat = scores.reshape(len(rows), -1)
            order = np.argsort(flat, axis=1, kind="stable")
            rr = np.arange(len(rows))
            winner = order[:, -1]
            runner = order[:, -2]
            kind = winner // n_templates
            template = winner % n_templates
            result["winner_template"][rows] = template
            result["winner_kind"][rows] = kind
            result["score"][rows] = flat[rr, winner]
            result["margin"][rows] = flat[rr, winner] - flat[rr, runner]
            result["gain"][rows] = gains[rr, kind, template]
    return result


def classify_scores(scored: dict[str, np.ndarray]) -> np.ndarray:
    real = scored["winner_kind"] == 0
    gain = (scored["gain"] >= GAIN_RANGE[0]) & (scored["gain"] <= GAIN_RANGE[1])
    strict = real & gain & (scored["score"] >= STRICT_COSINE) & (scored["margin"] >= IDENTITY_MARGIN)
    lower = real & gain & ~strict & (scored["score"] >= DISPLAY_COSINE) & (scored["margin"] >= IDENTITY_MARGIN)
    ambiguous = real & gain & ~strict & ~lower & (scored["score"] >= DISPLAY_COSINE)
    status = np.full(len(real), "unmatched", dtype=object)
    status[ambiguous] = "identity_ambiguous"
    status[lower] = "lower_score"
    status[strict] = "strict"
    status[scored["winner_kind"] > 0] = "decoy_winner"
    return status


def seed_recovery(
    families: pd.DataFrame,
    membership: pd.DataFrame,
    scored: dict[str, np.ndarray],
    status: np.ndarray,
) -> pd.DataFrame:
    family_ids = families.family_id.to_numpy()
    owner = membership.set_index("event_index").family_id
    rows = []
    for family in family_ids:
        events = owner.index[owner == family].to_numpy(int)
        correct = (family_ids[scored["winner_template"][events]] == family) & (status[events] == "strict")
        rows.append({"family_id": family, "seed_member_events": len(events),
                     "strict_seed_recovered": int(correct.sum()),
                     "strict_seed_recovery_fraction": float(correct.mean()) if len(events) else 0.0})
    return pd.DataFrame(rows)


def match_interval(
    found: np.ndarray,
    waves: np.ndarray,
    absolute_start: float,
    fs: float,
    geom: np.ndarray,
    patch_channels: np.ndarray,
    families: pd.DataFrame,
    templates: np.ndarray,
    scales: np.ndarray,
    relative_geometry: np.ndarray,
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    scored = score_global(waves, found[:, 3].astype(int), templates, families.phase_id.to_numpy(int), scales, relative_geometry)
    status = classify_scores(scored)
    selected = families.selected_depth_blind.to_numpy(bool)
    rows = []
    for event in np.flatnonzero(np.isin(status, ["strict", "lower_score", "identity_ambiguous"])):
        winner = scored["winner_template"][event]
        if not selected[winner]:
            continue
        bi = int(found[event, 2])
        energy = np.square(waves[event].astype(np.float64)).sum(axis=0)
        centroid = float(energy @ geom[patch_channels[bi], 1] / energy.sum())
        rows.append({"family_id": families.family_id.iloc[winner], "time_s": absolute_start + found[event, 0] / fs,
                     "observed_phase": int(found[event, 3]), "template_phase": int(families.phase_id.iloc[winner]),
                     "peak_channel": int(found[event, 1]), "peak_snr": float(found[event, 4]),
                     "score": float(scored["score"][event]), "margin": float(scored["margin"][event]),
                     "gain": float(scored["gain"][event]), "status": status[event],
                     "waveform_centroid_um": centroid})
    decoy = pd.DataFrame({"family_id": families.family_id.to_numpy()[scored["winner_template"]],
                          "decoy_kind": scored["winner_kind"]})
    decoy = decoy[decoy.decoy_kind > 0].value_counts().rename("wins").reset_index()
    counters = {"detections": len(found), "strict_all_bank": int((status == "strict").sum()),
                "decoy_winners": int((status == "decoy_winner").sum()),
                "unsupported_or_below_display": int((status == "unmatched").sum())}
    return pd.DataFrame(rows), counters, decoy


def run(config: dict) -> None:
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "settings.json", config)
    if config.get("dummy"):
        time.sleep(20)
        atomic_json(output / "summary.json", {"schema": SCHEMA, "status": "complete", "dummy": True})
        return
    manifest = json.loads(MANIFEST.read_text())
    binary = Path(manifest["binary_path"])
    fs = float(manifest["sampling_frequency_hz"])
    before = raw_sha_contract(binary)
    raw = np.memmap(binary, dtype="<i2", mode="r", shape=(manifest["num_samples"], manifest["saved_channels_in_binary"]))
    geom, kept_raw = geometry_and_mapping()
    bases, patch_channels = patches(geom)
    relative_geometry = geom[patch_channels[0]] - np.array([0.0, bases[0]])

    seed_interval = tuple(config.get("seed_interval_s", SEED_INTERVAL_S))
    if len(seed_interval) != 2 or not seed_interval[0] < seed_interval[1]:
        raise ValueError("seed_interval_s must be an increasing pair")
    seed_start, seed_stop = seed_interval
    x = preprocess(raw, round(seed_start * fs), round(seed_stop * fs), kept_raw, fs)
    found, waves = detect_waveforms(x, fs, geom, bases, patch_channels)
    families, membership, templates = build_families(found, waves)
    scales = template_scales(membership, waves, families.family_id.to_numpy())
    families = global_rivals(families, templates, relative_geometry)
    seed_scores = score_global(waves, found[:, 3].astype(int), templates, families.phase_id.to_numpy(int), scales, relative_geometry)
    seed_status = classify_scores(seed_scores)
    recovery = seed_recovery(families, membership, seed_scores, seed_status)
    families = families.merge(recovery, on="family_id", validate="one_to_one")
    families["shape_eligible_v3"] = (
        (families.seed_events >= MIN_FAMILY_EVENTS)
        & (families.first_half_events >= MIN_HALF_EVENTS)
        & (families.second_half_events >= MIN_HALF_EVENTS)
        & (families.split_half_cosine >= MIN_REPEATABILITY)
        & (families.median_member_cosine >= MIN_MEMBER_COSINE)
        & (families.global_rival_cosine < 0.95)
        & (families.median_peak_snr >= MIN_MEDIAN_SNR)
    )
    families["seed_qualified_v3"] = families.shape_eligible_v3 & (families.strict_seed_recovered >= MIN_SEED_RECOVERY_EVENTS) & (families.strict_seed_recovery_fraction >= MIN_SEED_RECOVERY_FRACTION)
    families["rank_score_v3"] = ((1 - families.global_rival_cosine.clip(upper=1)) * families.split_half_cosine * families.median_member_cosine * np.sqrt(families.strict_seed_recovery_fraction.clip(lower=0)) * np.log1p(families.seed_events) * np.log1p(families.median_peak_snr))
    families = families.sort_values(["seed_qualified_v3", "rank_score_v3", "family_id"], ascending=[False, False, True]).reset_index(drop=True)
    order = {fid: i for i, fid in enumerate(families.family_id)}
    old_ids = list(recovery.family_id)
    old_pos = {fid: i for i, fid in enumerate(old_ids)}
    templates = np.asarray([templates[old_pos[fid]] for fid in families.family_id], dtype=np.float32)
    scales = np.asarray([scales[old_pos[fid]] for fid in families.family_id], dtype=np.float32)
    families["selected_depth_blind"] = False
    families.loc[: min(N_SELECTED, len(families)) - 1, "selected_depth_blind"] = True
    families["candidate_rank_depth_blind"] = np.nan
    families.loc[families.selected_depth_blind, "candidate_rank_depth_blind"] = np.arange(1, int(families.selected_depth_blind.sum()) + 1)
    families["candidate_tier"] = np.where(families.selected_depth_blind, np.where(families.seed_qualified_v3, "strict_seed_qualified", "lower_confidence"), "not_selected")
    frozen = families.copy()
    frozen.to_csv(output / "families_depth_blind_frozen.csv", index=False)
    np.savez_compressed(output / "family_templates.npz", waveforms=templates, template_scale_uv=scales, relative_geometry=relative_geometry)
    families, seed_events = add_seed_depth_after_freeze(families, membership, found, waves, geom, patch_channels)
    families["postmatch_candidate_plausible"] = families.seed_qualified_v3 & families.postmatch_seed_spatially_plausible
    # A span near 250 um was declared biologically plausible before the later
    # seed-window results were inspected. Keep it as a separate motion-scale
    # tier requiring independent replication, not as a localized seed identity.
    families["postmatch_motion_scale_candidate"] = families.seed_qualified_v3 & (families.seed_depth_p90_span_um > 120.0) & (families.seed_depth_p90_span_um <= 300.0)
    families.to_csv(output / "families_after_depth_reveal.csv", index=False)
    seed_events.to_csv(output / "seed_cluster_events.csv", index=False)
    del x, waves

    parts, counts, decoys = [], [], []
    for index, (start, stop) in enumerate(MOTION_WINDOWS):
        x = preprocess(raw, round(start * fs), round(stop * fs), kept_raw, fs)
        detected, wave = detect_waveforms(x, fs, geom, bases, patch_channels)
        matched, detail, decoy = match_interval(detected, wave, start, fs, geom, patch_channels, families, templates, scales, relative_geometry)
        matched.to_csv(output / f"heldout_{index:02d}_events.csv", index=False)
        atomic_json(output / f"heldout_{index:02d}.complete.json", {"sha256": sha256(output / f"heldout_{index:02d}_events.csv"), **detail})
        if len(decoy): decoy["window_index"] = index
        parts.append(matched); counts.append({"start_s": start, "stop_s": stop, **detail}); decoys.append(decoy)
        print(index, start, stop, detail, flush=True)
    events = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    events.to_csv(output / "heldout_events.csv", index=False)
    pd.DataFrame(counts).to_csv(output / "heldout_counts.csv", index=False)
    pd.concat(decoys, ignore_index=True).to_csv(output / "decoy_wins_by_family.csv", index=False)
    plot_candidate_review(output, families, templates, seed_events, events, fs, seed_interval_s=seed_interval)
    if raw_sha_contract(binary) != before:
        raise RuntimeError("Raw source changed")
    strict = events[events.status == "strict"] if len(events) else events
    summary = {
        "schema": SCHEMA, "status": "complete", "sorter_inputs_read": False, "motion_estimate_read": False,
        "seed_interval_s": list(seed_interval), "seed_detections": len(found), "waveform_proposals": len(families),
        "seed_qualified_depth_blind": int(families.seed_qualified_v3.sum()),
        "postmatch_seed_plausible_candidates": int(families.postmatch_candidate_plausible.sum()),
        "postmatch_motion_scale_candidates_requiring_replication": int(families.postmatch_motion_scale_candidate.sum()),
        "selected_review_candidates": int(families.selected_depth_blind.sum()),
        "strict_heldout_events": len(strict), "strict_heldout_families": int(strict.family_id.nunique()) if len(strict) else 0,
        "method_changes": ["global cross-phase real/decoy rivalry", "exact peak-centered overlap geometry", "minimum 50% overlap energy", "gain 0.35-3", "seed recovery qualification"],
        "interpretation": "Candidates only. Motion comparison remains blocked until at least two postmatch-plausible identities reproduce shared movement."
    }
    atomic_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dummy", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    config["dummy"] = args.dummy
    run(config)


if __name__ == "__main__":
    main()
