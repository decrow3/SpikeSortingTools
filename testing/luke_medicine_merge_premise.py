"""Score exact 0/40/80 um alignments in the bounded premerge checkpoint.

This is an exploratory representation check only.  It never accepts merges and
does not use MEDiCINe or absolute corrected depth to define identity.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np

from pipeline.motion_aware_merge import aligned_template_similarity
from testing.luke_medicine_motion_aware_merge import DEFAULT_OUTPUT, DEFAULT_SORT


OFFSETS = (-80.0, -40.0, 0.0, 40.0, 80.0)


def main() -> None:
    source = DEFAULT_OUTPUT / "bounded_premerge"
    output = DEFAULT_OUTPUT / "premise"
    partial = output.with_name(output.name + ".partial")
    if partial.exists():
        raise RuntimeError(f"Incomplete premise stage requires inspection: {partial}")
    if output.exists():
        receipt = json.loads((output / "receipt.json").read_text())
        if not receipt.get("complete"):
            raise RuntimeError("Existing premise output is incomplete")
        print(json.dumps(receipt, indent=2))
        return
    cluster_receipt = json.loads((source / "receipt.json").read_text())
    if not cluster_receipt.get("complete") or cluster_receipt.get("merging_function_called"):
        raise RuntimeError("A complete before-merge checkpoint is required")
    partial.mkdir(parents=True)
    wall = np.load(source / "Wall.npy", mmap_mode="r")
    clu = np.load(source / "clu.npy", mmap_mode="r")
    ops = np.load(DEFAULT_SORT / "ops.npy", allow_pickle=True).item()
    positions = np.load(DEFAULT_SORT / "channel_positions.npy")
    shanks = np.load(DEFAULT_SORT / "channel_shanks.npy").reshape(-1)
    w_pca = np.asarray(ops["wPCA"], dtype=np.float64)
    waveforms = np.einsum("kcp,pt->kct", wall, w_pca, optimize=True)
    counts = np.bincount(np.asarray(clu, dtype=np.int64), minlength=len(wall))
    energy = np.sum(waveforms * waveforms, axis=2)
    peak_channel = np.argmax(energy, axis=1)
    lags = tuple(range(-6, 7))
    rows = []
    eligible_clusters = np.flatnonzero(counts >= 100)
    for ii, a in enumerate(eligible_clusters):
        pa = positions[peak_channel[a]]
        for b in eligible_clusters[ii + 1:]:
            pb = positions[peak_channel[b]]
            if shanks[peak_channel[a]] != shanks[peak_channel[b]]:
                continue
            if abs(pa[0] - pb[0]) > 1e-6 or abs(pa[1] - pb[1]) > 80.001:
                continue
            scores = {}
            overlaps = {}
            for offset in OFFSETS:
                result = aligned_template_similarity(
                    waveforms[a], waveforms[b], positions, offset,
                    channel_shanks=shanks, temporal_lags=lags, min_channels=8,
                )
                scores[offset] = result["score"]
                overlaps[offset] = result["overlap_channels"]
            finite = [(score, offset) for offset, score in scores.items() if np.isfinite(score)]
            best_score, best_offset = max(finite)
            rows.append({
                "cluster_a": int(a), "cluster_b": int(b),
                "spikes_a": int(counts[a]), "spikes_b": int(counts[b]),
                "peak_x_a_um": float(pa[0]), "peak_y_a_um": float(pa[1]),
                "peak_x_b_um": float(pb[0]), "peak_y_b_um": float(pb[1]),
                "static_score": float(scores[0.0]),
                "best_score": float(best_score), "best_offset_um": float(best_offset),
                "score_gain": float(best_score - scores[0.0]),
                **{f"score_{int(o):+d}um": float(scores[o]) for o in OFFSETS},
                **{f"overlap_{int(o):+d}um": int(overlaps[o]) for o in OFFSETS},
            })
    rows.sort(key=lambda row: (row["best_score"], row["score_gain"]), reverse=True)
    if not rows:
        raise RuntimeError("No geometry-compatible cluster pairs passed the premise screen")
    fields = list(rows[0])
    with (partial / "pair_scores.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    top = rows[:20]
    receipt = {
        "schema": "medicine-aware-merge-representation-premise-v1",
        "complete": True,
        "scientific_status": "exploratory_representation_only_no_merges_accepted",
        "source_cluster_request_digest": cluster_receipt["request_digest"],
        "offsets_um": list(OFFSETS),
        "temporal_lags_samples": [min(lags), max(lags)],
        "minimum_cluster_spikes": 100,
        "pairs_scored": len(rows),
        "pairs_best_nonzero": int(sum(row["best_offset_um"] != 0 for row in rows)),
        "pairs_gain_over_0_1": int(sum(row["score_gain"] > 0.1 for row in rows)),
        "top_pairs": top,
        "limitation": "Top aligned pairs are hypotheses, not independent identities or safe merges.",
    }
    (partial / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    os.replace(partial, output)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
