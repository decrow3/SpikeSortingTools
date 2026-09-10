"""Build and score MEDiCINe motion-state templates for the bounded premerge run.

Outputs are candidate evidence only.  This stage does not resolve the merge
graph because independent lighthouse/rival and CCG vetoes are separate gates.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

from pipeline.motion_aware_merge import (
    StatePair,
    aggregate_reciprocal_state_pairs,
    aligned_template_similarity,
    build_motion_state_templates,
    feature_channel_indices,
)
from pipeline.motion_coordinates import interpolate_motion_at_spikes, load_qualified_motion_field
from testing.luke_medicine_motion_aware_merge import DEFAULT_OUTPUT, DEFAULT_SORT


OUT = DEFAULT_OUTPUT / "state_template_replay"
MIN_STATE_SPIKES = 100
MIN_OVERLAP_CHANNELS = 8
SCORE_THRESHOLD = 0.90
INCOMPATIBILITY_THRESHOLD = 0.50
MIN_RECIPROCAL_PAIRS = 2
MIN_STATE_COVERAGE = 0.50
CORRECTED_DEPTH_EVALUATION_RADIUS_UM = 120.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--control", choices=("eligible", "opposite-sign", "time-shift-60s"),
        default="eligible",
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    output = args.out or OUT
    if args.control != "eligible" and args.out is None:
        output = DEFAULT_OUTPUT / "controls" / args.control
    partial = output.with_name(output.name + ".partial")
    if partial.exists():
        raise RuntimeError(f"Partial state replay requires inspection: {partial}")
    if output.exists():
        print((output / "receipt.json").read_text())
        return
    premerge = DEFAULT_OUTPUT / "bounded_premerge"
    qualification = DEFAULT_OUTPUT / "field_qualification"
    cluster_receipt = json.loads((premerge / "receipt.json").read_text())
    qualification_receipt = json.loads((qualification / "receipt.json").read_text())
    if not cluster_receipt.get("complete") or cluster_receipt.get("merging_function_called"):
        raise RuntimeError("Invalid bounded premerge checkpoint")
    if not qualification_receipt.get("complete"):
        raise RuntimeError("Field qualification is incomplete")
    settings = {
        "minimum_state_spikes": MIN_STATE_SPIKES,
        "minimum_overlap_channels": MIN_OVERLAP_CHANNELS,
        "score_threshold": SCORE_THRESHOLD,
        "incompatibility_threshold": INCOMPATIBILITY_THRESHOLD,
        "minimum_reciprocal_pairs": MIN_RECIPROCAL_PAIRS,
        "minimum_state_coverage": MIN_STATE_COVERAGE,
        "corrected_depth_evaluation_radius_um": CORRECTED_DEPTH_EVALUATION_RADIUS_UM,
        "state_step_um": 40.0,
        "primary_offset": "MEDiCINe-predicted quantized offset only",
        "control": args.control,
    }
    partial.mkdir(parents=True)
    (partial / "frozen_settings.json").write_text(json.dumps(settings, indent=2) + "\n")

    st = np.load(premerge / "st.npy", mmap_mode="r")
    clu = np.load(premerge / "clu.npy", mmap_mode="r")
    wall = np.load(premerge / "Wall.npy", mmap_mode="r")
    source_indices = np.load(premerge / "source_spike_indices.npy", mmap_mode="r")
    audit = DEFAULT_OUTPUT / "audit"
    raw_depth = np.load(audit / "observed_depth_um.npy", mmap_mode="r")
    raw_depth_good = np.load(audit / "observed_depth_eligible.npy", mmap_mode="r")
    if not (len(st) == len(clu) == len(source_indices) == len(raw_depth)):
        raise RuntimeError("bounded spike arrays disagree")
    ops = np.load(DEFAULT_SORT / "ops.npy", allow_pickle=True).item()
    fs = float(ops["fs"])
    times_s = st[:, 0] / fs - 930.0
    sample_times_s = (
        np.mod(times_s + 60.0, 300.0)
        if args.control == "time-shift-60s" else times_s
    )
    field = load_qualified_motion_field(
        qualification / "qualified_field.npz", recording_duration_s=300.0
    )
    sampled = interpolate_motion_at_spikes(
        field, sample_times_s, raw_depth, min_support=MIN_STATE_SPIKES * 0.2,
        min_confidence=1.0,
    )
    if args.control == "opposite-sign":
        sampled["displacement_um"] *= -1.0
    eligible = raw_depth_good & sampled["supported"]
    corrected_depth = raw_depth - sampled["displacement_um"]
    np.save(partial / "displacement_um.npy", sampled["displacement_um"])
    np.save(partial / "corrected_depth_um.npy", corrected_depth)
    np.save(partial / "eligible.npy", eligible)

    full_t_f = np.load(DEFAULT_SORT / "tF.npy", mmap_mode="r")
    t_f = np.asarray(full_t_f[source_indices[0]:source_indices[-1] + 1])
    detection_templates = np.asarray(st[:, 1], dtype=np.int64)
    channels = feature_channel_indices(detection_templates, ops["iCC"], ops["iU"])
    states, ineligible = build_motion_state_templates(
        t_f, clu, channels, sampled["displacement_um"], eligible,
        n_channels=len(ops["yc"]), state_step_um=40.0,
        min_spikes=MIN_STATE_SPIKES,
    )
    if not states:
        raise RuntimeError("No supported motion-state templates")
    state_templates = np.stack([state.template for state in states])
    np.save(partial / "state_templates.npy", state_templates)
    rounded_state = np.zeros(len(clu), dtype=np.int64)
    rounded_state[eligible] = np.rint(
        sampled["displacement_um"][eligible] / 40.0
    ).astype(np.int64)
    state_meta = []
    for i, state in enumerate(states):
        members = eligible & (clu == state.cluster) & (rounded_state == state.state)
        if int(np.sum(members)) != state.spike_count:
            raise RuntimeError("state metadata membership disagrees with template")
        disp = sampled["displacement_um"][members]
        corr = corrected_depth[members]
        amp = st[members, 2] if st.shape[1] > 2 else np.full(state.spike_count, np.nan)
        support = sampled["support"][members]
        channel_support = np.any(state.template != 0, axis=1)
        state_meta.append({
            "index": i, "cluster": state.cluster, "state": state.state,
            "displacement_center_um": state.displacement_center_um,
            "spike_count": state.spike_count,
            "time_start_s": float(np.min(times_s[members])),
            "time_stop_s": float(np.max(times_s[members])),
            "displacement_min_um": float(np.min(disp)),
            "displacement_median_um": float(np.median(disp)),
            "displacement_max_um": float(np.max(disp)),
            "corrected_depth_p05_um": float(np.quantile(corr, 0.05)),
            "corrected_depth_median_um": float(np.median(corr)),
            "corrected_depth_p95_um": float(np.quantile(corr, 0.95)),
            "amplitude_p05": float(np.nanquantile(amp, 0.05)),
            "amplitude_median": float(np.nanmedian(amp)),
            "amplitude_p95": float(np.nanquantile(amp, 0.95)),
            "effective_support_min": float(np.min(support)),
            "effective_support_median": float(np.median(support)),
            "physical_channel_support_count": int(np.sum(channel_support)),
            "physical_channel_first": int(np.flatnonzero(channel_support)[0]),
            "physical_channel_last": int(np.flatnonzero(channel_support)[-1]),
            "ineligible_cluster_spikes": ineligible[state.cluster],
        })
    with (partial / "state_templates.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(state_meta[0]))
        writer.writeheader(); writer.writerows(state_meta)

    positions = np.load(DEFAULT_SORT / "channel_positions.npy")
    shanks = np.load(DEFAULT_SORT / "channel_shanks.npy").reshape(-1)
    w_pca = np.asarray(ops["wPCA"], dtype=float)
    waveforms = np.einsum("kcp,pt->kct", state_templates, w_pca, optimize=True)
    by_cluster = defaultdict(list)
    for i, state in enumerate(states):
        by_cluster[state.cluster].append(i)
    eligible_count = {int(c): int(np.sum(eligible & (clu == c))) for c in np.unique(clu)}
    median_corrected = {int(c): float(np.nanmedian(corrected_depth[eligible & (clu == c)])) for c in np.unique(clu) if np.any(eligible & (clu == c))}
    lags = tuple(range(-6, 7))
    pair_rows, aggregate_rows = [], []
    clusters = sorted(by_cluster)
    for ia, cluster_a in enumerate(clusters):
        for cluster_b in clusters[ia + 1:]:
            if abs(median_corrected[cluster_a] - median_corrected[cluster_b]) > CORRECTED_DEPTH_EVALUATION_RADIUS_UM:
                continue
            evidence = []
            for sa in by_cluster[cluster_a]:
                for sb in by_cluster[cluster_b]:
                    predicted = states[sa].displacement_center_um - states[sb].displacement_center_um
                    result = aligned_template_similarity(
                        waveforms[sa], waveforms[sb], positions, predicted,
                        channel_shanks=shanks, temporal_lags=lags,
                        min_channels=MIN_OVERLAP_CHANNELS,
                    )
                    score = float(result["score"])
                    compatible = bool(np.isfinite(score) and score >= INCOMPATIBILITY_THRESHOLD)
                    evidence.append(StatePair(
                        states[sa].state, states[sb].state, score,
                        states[sa].spike_count, states[sb].spike_count,
                        compatible=compatible,
                    ))
                    pair_rows.append({
                        "cluster_a": cluster_a, "cluster_b": cluster_b,
                        "state_a": states[sa].state, "state_b": states[sb].state,
                        "spikes_a": states[sa].spike_count, "spikes_b": states[sb].spike_count,
                        "predicted_offset_um": predicted, "score": score,
                        "temporal_lag_samples": result["lag"],
                        "overlap_channels": result["overlap_channels"],
                        "compatible": compatible,
                    })
            aggregate = aggregate_reciprocal_state_pairs(
                evidence, eligible_spikes_a=eligible_count[cluster_a],
                eligible_spikes_b=eligible_count[cluster_b],
                min_pairs=MIN_RECIPROCAL_PAIRS, min_coverage=MIN_STATE_COVERAGE,
                score_threshold=SCORE_THRESHOLD,
                incompatibility_threshold=INCOMPATIBILITY_THRESHOLD,
            )
            aggregate_rows.append({
                "cluster_a": cluster_a, "cluster_b": cluster_b,
                "corrected_depth_a_um": median_corrected[cluster_a],
                "corrected_depth_b_um": median_corrected[cluster_b],
                **{k: v for k, v in aggregate.items() if k != "reciprocal_pairs"},
                "reciprocal_pairs": json.dumps(aggregate["reciprocal_pairs"]),
                "eligible_merge": False,
                "status": "candidate_requires_ccg_rival_and_independent_lighthouse_vetoes" if aggregate["accepted"] else "state_aggregation_failed",
            })
    for name, rows in (("state_pair_scores.csv", pair_rows), ("cluster_pair_aggregates.csv", aggregate_rows)):
        if not rows:
            raise RuntimeError(f"No rows for {name}")
        with (partial / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    provisional = sum(row["status"].startswith("candidate_requires") for row in aggregate_rows)
    receipt = {
        "schema": "medicine-aware-state-template-replay-v1",
        "complete": True,
        "scientific_status": (
            "candidate_evidence_only_no_merges_accepted" if args.control == "eligible"
            else "negative_control_no_merges_eligible"
        ),
        "control": args.control,
        "cluster_request_digest": cluster_receipt["request_digest"],
        "qualification_digest": qualification_receipt["qualification_digest"],
        "settings": settings,
        "spikes": len(st), "eligible_spikes": int(np.sum(eligible)),
        "eligible_spike_fraction": float(np.mean(eligible)),
        "premerge_clusters": len(np.unique(clu)),
        "state_templates": len(states),
        "state_pairs_scored": len(pair_rows),
        "cluster_pairs_evaluated": len(aggregate_rows),
        "provisional_state_passes": provisional,
        "accepted_merges": 0,
        "remaining_gates": ["CCG", "rival separation", "independent lighthouse identity", "graph conflict resolution"],
    }
    (partial / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    output.parent.mkdir(parents=True, exist_ok=True)
    os.replace(partial, output)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
