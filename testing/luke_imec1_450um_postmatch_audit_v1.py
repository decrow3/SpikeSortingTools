#!/usr/bin/env python
"""Reclassify frozen compact-core tracks with a 450-um post-match allowance."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MAX_MOTION_SCALE_UM = 450.0
LOCALIZED_UM = 120.0
MIN_STRICT = 5


def p90_span(series: pd.Series) -> float:
    return float(series.quantile(.95) - series.quantile(.05)) if len(series) else np.nan


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True); args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    events = pd.read_csv(args.input / "qualification_events_compact.csv")
    training = pd.read_csv(args.input / "training_core_reproducibility.csv")
    families = training.loc[training.odd_even_valid].copy()
    rows = []
    for family in families.itertuples():
        q = events[(events.winner_bank == "s300") & (events.winner_family == family.family_id)]
        strict = q[q.status == "strict"]
        displayed = q[q.status.isin(["strict", "lower_score", "identity_ambiguous"])]
        strict_span = p90_span(strict.waveform_centroid_um)
        displayed_span = p90_span(displayed.waveform_centroid_um)
        if not np.isfinite(strict_span):
            category = "insufficient_strict_support"
        elif strict_span <= LOCALIZED_UM:
            category = "localized_le120um"
        elif strict_span <= MAX_MOTION_SCALE_UM:
            category = "motion_scale_120to450um"
        else:
            category = "broad_gt450um"
        rows.append({
            "family_id": family.family_id, "phase_id": int(family.phase_id),
            "strict_matches": len(strict), "strict_match_rate_hz": len(strict) / 10,
            "strict_depth_p90_span_um": strict_span,
            "strict_depth_minmax_span_um": float(strict.waveform_centroid_um.max() - strict.waveform_centroid_um.min()) if len(strict) else np.nan,
            "displayed_real_matches": len(displayed), "displayed_real_match_rate_hz": len(displayed) / 10,
            "displayed_depth_p90_span_um": displayed_span,
            "postmatch_depth_category_450um": category,
            "passes_strict_count": len(strict) >= MIN_STRICT,
            "within_450um_postmatch_allowance": bool(np.isfinite(strict_span) and strict_span <= MAX_MOTION_SCALE_UM),
            "candidate_under_450um_audit": bool(len(strict) >= MIN_STRICT and np.isfinite(strict_span) and strict_span <= MAX_MOTION_SCALE_UM),
            "reproduces_odd_even": bool(family.reproduces_odd_even),
            "time_block_valid": bool(family.time_block_valid),
            "reproduces_time_blocks": bool(family.reproduces_time_blocks),
        })
    audit = pd.DataFrame(rows).sort_values(["candidate_under_450um_audit", "strict_matches"], ascending=False)
    audit.to_csv(args.output / "candidate_tracks_450um.csv", index=False)
    candidates = audit[audit.candidate_under_450um_audit]
    summary = {
        "schema": "luke0804-imec1-450um-postmatch-audit-v1", "status": "complete",
        "identity_matching_rerun": False,
        "reason_no_raw_rerun": "Existing detection, proposal construction, and global waveform scoring are depth blind; only post-match categorization changes.",
        "qualification_window_s": [310.0, 320.0], "motion_scale_max_um": MAX_MOTION_SCALE_UM,
        "strict_minimum_events": MIN_STRICT, "core_families": len(audit),
        "localized_le120um": int(audit.postmatch_depth_category_450um.eq("localized_le120um").sum()),
        "motion_scale_120to450um": int(audit.postmatch_depth_category_450um.eq("motion_scale_120to450um").sum()),
        "broad_gt450um": int(audit.postmatch_depth_category_450um.eq("broad_gt450um").sum()),
        "insufficient_strict_support": int(audit.postmatch_depth_category_450um.eq("insufficient_strict_support").sum()),
        "candidates_ge5_strict_within450um": candidates.family_id.tolist(),
        "candidate_count": len(candidates),
        "temporal_reproduction_failures_among_candidates": candidates.loc[
            candidates.time_block_valid & ~candidates.reproduces_time_blocks, "family_id"
        ].tolist(),
        "interpretation": "Post-match retention audit only. Passing 450 um does not certify identity, cell status, or motion amplitude.",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    ordered = audit[np.isfinite(audit.strict_depth_p90_span_um)].sort_values("strict_depth_p90_span_um")
    fig, ax = plt.subplots(figsize=(9, 6.5))
    colors = np.where(ordered.candidate_under_450um_audit, "#4C78A8", "#B8B8B8")
    ax.barh(ordered.family_id, ordered.strict_depth_p90_span_um, color=colors)
    ax.axvline(120, color="#555555", linestyle="--", linewidth=1, label="Localized boundary: 120 µm")
    ax.axvline(450, color="#D97706", linestyle="--", linewidth=1.4, label="Motion-scale allowance: 450 µm")
    ax.set_xlabel("Strict-event P90 waveform-depth span (µm)")
    ax.set_title("imec1 compact-core tracks under 450 µm post-match allowance\nBlue requires ≥5 strict matches; depth did not enter identity matching")
    ax.grid(axis="x", alpha=.2); ax.legend(); fig.tight_layout()
    fig.savefig(args.output / "01_strict_depth_spans_450um.png", dpi=180)
    fig.savefig(args.output / "01_strict_depth_spans_450um.pdf")


if __name__ == "__main__":
    main()
