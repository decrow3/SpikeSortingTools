#!/usr/bin/env python
"""Assess all 20 frozen compact-core proposals under the 450-um policy."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

STRICT_SCORE = 0.86
STRICT_MARGIN = 0.025
NEAR_SCORE = 0.84
NEAR_MARGIN = 0.015


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    events = pd.read_csv(args.events / "qualification_events_compact.csv")
    candidates = pd.read_csv(args.audit / "candidate_tracks_450um.csv")
    if len(candidates) != 20 or candidates.family_id.duplicated().any():
        raise RuntimeError("expected exactly 20 unique compact-core proposals")
    if len(events) != 47013 or events.event_id.duplicated().any():
        raise RuntimeError("qualification event grain changed")
    rows = []
    for candidate in candidates.itertuples():
        q = events[(events.winner_bank == "s300") & (events.winner_family == candidate.family_id)]
        real = q[(q.winner_kind == 0) & q.gain.between(.35, 3)]
        near = real[(real.score >= NEAR_SCORE) & (real.margin >= NEAR_MARGIN) & (real.status != "strict")]
        ambiguous = real[real.status.isin(["identity_ambiguous", "lower_score"])]
        rival_counts = ambiguous.groupby(["runner_bank", "runner_family"]).size().sort_values(ascending=False)
        if len(rival_counts):
            rival_bank, rival_family = rival_counts.index[0]; rival_events = int(rival_counts.iloc[0])
        else:
            rival_bank = rival_family = ""; rival_events = 0
        if candidate.candidate_under_450um_audit:
            tier = "retained_450um"
            reason = ">=5 strict matches and strict P90 span <=450 um"
        elif candidate.strict_matches >= 2 and candidate.within_450um_postmatch_allowance:
            tier = "near_count"
            reason = f"strict count {int(candidate.strict_matches)}/5; depth allowance passes"
        elif candidate.strict_matches >= 4 and np.isfinite(candidate.strict_depth_p90_span_um) and candidate.strict_depth_p90_span_um > 450:
            tier = "split_required"
            reason = f"strict depth P90 span {candidate.strict_depth_p90_span_um:.1f} um exceeds 450 um"
        else:
            tier = "weak_ambiguous"
            reason = f"{int(candidate.strict_matches)} strict; {len(near)} near-threshold events"
        rows.append({
            "family_id": candidate.family_id, "phase_id": int(candidate.phase_id), "assessment_tier": tier,
            "tier_reason": reason, "strict_matches": int(candidate.strict_matches),
            "strict_match_rate_hz": candidate.strict_match_rate_hz,
            "near_threshold_events_diagnostic": len(near),
            "lower_score_events": int((q.status == "lower_score").sum()),
            "identity_ambiguous_events": int((q.status == "identity_ambiguous").sum()),
            "displayed_real_match_rate_hz": candidate.displayed_real_match_rate_hz,
            "strict_depth_p90_span_um": candidate.strict_depth_p90_span_um,
            "displayed_depth_p90_span_um": candidate.displayed_depth_p90_span_um,
            "reproduces_odd_even": bool(candidate.reproduces_odd_even),
            "time_block_valid": bool(candidate.time_block_valid),
            "reproduces_time_blocks": bool(candidate.reproduces_time_blocks),
            "dominant_rival_bank": rival_bank, "dominant_rival_family": rival_family,
            "dominant_rival_events": rival_events,
        })
    out = pd.DataFrame(rows)
    tier_order = {"retained_450um": 0, "near_count": 1, "split_required": 2, "weak_ambiguous": 3}
    out["tier_order"] = out.assessment_tier.map(tier_order)
    out = out.sort_values(["tier_order", "strict_matches", "near_threshold_events_diagnostic"], ascending=[True, False, False])
    out.insert(0, "assessment_rank", np.arange(1, len(out) + 1)); out = out.drop(columns="tier_order")
    out.to_csv(args.output / "twenty_candidate_assessment.csv", index=False)
    summary = {
        "schema": "luke0804-imec1-twenty-candidate-assessment-v1", "status": "complete",
        "candidate_grain": "20 unique reciprocal compact-core waveform proposals",
        "qualification_window_s": [310.0, 320.0], "qualification_events": len(events),
        "tier_counts": out.assessment_tier.value_counts().to_dict(),
        "retained_450um": out.loc[out.assessment_tier == "retained_450um", "family_id"].tolist(),
        "near_count": out.loc[out.assessment_tier == "near_count", "family_id"].tolist(),
        "split_required": out.loc[out.assessment_tier == "split_required", "family_id"].tolist(),
        "weak_ambiguous": out.loc[out.assessment_tier == "weak_ambiguous", "family_id"].tolist(),
        "near_threshold_definition": "Diagnostic only: real winner, gain 0.35-3, score >=0.84, margin >=0.015, not strict.",
        "data_quality": {"duplicate_candidate_ids": 0, "duplicate_event_ids": 0,
                         "candidate_rows": len(out), "qualification_event_rows": len(events)},
        "interpretation": "Assessment pool, not 20 units. Only retained_450um passes the current bounded count/depth audit; all identities remain provisional.",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
