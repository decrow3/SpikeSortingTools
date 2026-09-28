#!/usr/bin/env python
"""Cached audit and figures for the imec1 compact alignment pilot."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from testing.luke_imec1_compact_alignment_pilot_v2 import comparison_ledger

ARMS = ("broad", "compact", "compact_aligned")


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--input", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    training = pd.read_csv(args.input / "training_core_reproducibility.csv").set_index("family_id")
    tables = {arm: pd.read_csv(args.input / f"qualification_events_{arm}.csv") for arm in ARMS}
    comparisons = {
        "broad_to_compact": (tables["broad"], tables["compact"]),
        "compact_to_compact_aligned": (tables["compact"], tables["compact_aligned"]),
        "broad_to_compact_aligned": (tables["broad"], tables["compact_aligned"]),
    }
    ledger = pd.concat([comparison_ledger(old, new, label) for label, (old, new) in comparisons.items()], ignore_index=True)
    ledger["score_delta"] = ledger.new_score - ledger.old_score
    ledger["margin_delta"] = ledger.new_margin - ledger.old_margin
    ledger["gain_delta"] = ledger.new_gain - ledger.old_gain
    ledger["winner_changed"] = (ledger.old_bank != ledger.new_bank) | (ledger.old_family != ledger.new_family) | (ledger.old_kind != ledger.new_kind)
    ledger["runner_changed"] = ledger.old_runner_family != ledger.new_runner_family
    ledger.to_csv(args.output / "event_level_change_ledger_corrected.csv", index=False)
    reasons = ledger.groupby(["comparison", "reason"]).size().rename("events").reset_index()
    reasons.to_csv(args.output / "event_change_reason_counts.csv", index=False)
    diagnostics = ledger.groupby(["comparison", "reason", "detector_phase"]).agg(
        events=("event_id", "size"), median_score_delta=("score_delta", "median"),
        median_margin_delta=("margin_delta", "median"), median_gain_delta=("gain_delta", "median"),
        winner_changed_rate=("winner_changed", "mean"), runner_changed_rate=("runner_changed", "mean"),
    ).reset_index()
    diagnostics.to_csv(args.output / "event_change_diagnostics_by_phase.csv", index=False)

    core_ids = training.index[training.odd_even_valid]
    count_rows = []
    for arm, table in tables.items():
        strict = table[table.status.eq("strict") & table.winner_bank.eq("s300")]
        count = strict.groupby("winner_family").size()
        count_rows.extend({"family_id": family, "arm": arm, "strict_matches": int(count.get(family, 0))} for family in core_ids)
    counts = pd.DataFrame(count_rows).pivot(index="family_id", columns="arm", values="strict_matches").fillna(0).astype(int)
    evidence = training.loc[core_ids, ["phase_id", "proposal_train_events", "odd_even_core_events",
                                       "odd_even_compact_template_cosine", "reproduces_odd_even",
                                       "time_block_valid", "time_block_compact_template_cosine", "reproduces_time_blocks"]].join(counts)
    evidence["proposal_training_rate_hz"] = evidence.proposal_train_events / 10
    evidence["core_training_rate_hz"] = evidence.odd_even_core_events / 10
    for arm in ARMS:
        evidence[f"{arm}_qualification_match_rate_hz"] = evidence[arm] / 10
    evidence["compact_minus_broad"] = evidence.compact - evidence.broad
    evidence["alignment_minus_compact"] = evidence.compact_aligned - evidence.compact
    evidence.sort_values(["compact_minus_broad", "alignment_minus_compact"], ascending=False).to_csv(
        args.output / "core_family_rate_and_match_evidence.csv"
    )

    summary = json.loads((args.input / "summary.json").read_text())
    base = tables["broad"].set_index("event_id")
    required = ["event_id", "time_s", "detector_phase", "peak_channel", "peak_snr", "winner_bank",
                "winner_family", "winner_kind", "best_lag", "status", "runner_family"]
    quality = {
        "rows_per_arm": {arm: int(len(table)) for arm, table in tables.items()},
        "duplicate_event_ids": {arm: int(table.event_id.duplicated().sum()) for arm, table in tables.items()},
        "missing_required_values": {arm: int(table[required].isna().sum().sum()) for arm, table in tables.items()},
        "missing_scoring_values_by_arm": {arm: {field: int(table[field].isna().sum()) for field in ("score", "margin", "gain")}
                                          for arm, table in tables.items()},
        "cross_arm_event_key_mismatches": {},
    }
    for arm in ("compact", "compact_aligned"):
        other = tables[arm].set_index("event_id")
        quality["cross_arm_event_key_mismatches"][arm] = int(
            (~np.isclose(base.time_s, other.time_s)).sum()
            + (base.detector_phase != other.detector_phase).sum()
            + (base.peak_channel != other.peak_channel).sum()
        )
    transitions = reasons.pivot(index="reason", columns="comparison", values="events").fillna(0).astype(int)
    summary.update({
        "audit_schema": "luke0804-imec1-compact-alignment-audit-v1",
        "odd_even_template_cosine_median": float(training.loc[training.odd_even_valid, "odd_even_compact_template_cosine"].median()),
        "odd_even_template_cosine_min": float(training.loc[training.odd_even_valid, "odd_even_compact_template_cosine"].min()),
        "time_block_template_cosine_median": float(training.loc[training.time_block_valid, "time_block_compact_template_cosine"].median()),
        "time_block_nonreproduced": training.index[training.time_block_valid & ~training.reproduces_time_blocks].tolist(),
        "compact_core_families_improved_vs_broad": int((evidence.compact_minus_broad > 0).sum()),
        "compact_core_families_tied_vs_broad": int((evidence.compact_minus_broad == 0).sum()),
        "compact_core_families_worse_vs_broad": int((evidence.compact_minus_broad < 0).sum()),
        "aligned_core_families_improved_vs_unaligned": int((evidence.alignment_minus_compact > 0).sum()),
        "aligned_core_families_tied_vs_unaligned": int((evidence.alignment_minus_compact == 0).sum()),
        "aligned_core_families_worse_vs_unaligned": int((evidence.alignment_minus_compact < 0).sum()),
        "event_transition_counts": {column: {index: int(value) for index, value in transitions[column].items()}
                                    for column in transitions.columns},
        "data_quality": quality,
        "decision": "Retain compact core selection; do not add pre-average integer alignment to the next bank by default.",
    })
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    plot = evidence.sort_values("compact_minus_broad")
    y = np.arange(len(plot)); fig, ax = plt.subplots(figsize=(9.5, 7.5))
    ax.scatter(plot.broad, y, label="Broad", color="#4C78A8", s=34)
    ax.scatter(plot.compact, y, label="Compact", color="#F58518", s=34)
    ax.scatter(plot.compact_aligned, y, label="Compact + member alignment", color="#54A24B", s=34, marker="x")
    ax.set_yticks(y, plot.index); ax.set_xlabel("Strict matches in frozen 310–320 s window")
    ax.set_title("imec1 proposal-template arms on identical qualification detections\nCounts are match rates over 10 s, not recovery probabilities")
    ax.grid(axis="x", alpha=.25); ax.legend(); fig.tight_layout()
    fig.savefig(args.output / "01_three_arm_core_family_matches.png", dpi=180); fig.savefig(args.output / "01_three_arm_core_family_matches.pdf")


if __name__ == "__main__":
    main()
