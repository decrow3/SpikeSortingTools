#!/usr/bin/env python
"""Reframe the compact-core pilot with paired evidence and honest denominators."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    q = pd.read_csv(args.input / "qualification_comparison.csv")
    build = pd.read_csv(args.input / "proposal_core_build.csv").set_index("family_id")
    broad = q[q.arm.eq("broad")].set_index("family_id")
    compact = q[q.arm.eq("compact")].set_index("family_id")
    ids = broad.index[broad.has_compact_core]
    paired = broad.loc[ids, ["phase_id", "train_members", "qualification_strict"]].rename(
        columns={"qualification_strict": "broad_strict"}
    )
    paired["compact_strict"] = compact.loc[ids, "qualification_strict"]
    paired["strict_delta"] = paired.compact_strict - paired.broad_strict
    paired = paired.join(build[["core_events", "core_fraction", "compact_vs_broad_cosine"]])
    paired["proposal_support_rate_hz"] = paired.train_members / 10.0
    paired["core_support_rate_hz"] = paired.core_events / 10.0
    paired["broad_qualification_rate_hz"] = paired.broad_strict / 10.0
    paired["compact_qualification_rate_hz"] = paired.compact_strict / 10.0
    paired["compact_postmatch_depth_p90_span_um"] = compact.loc[
        ids, "qualification_depth_p90_span_um"
    ]
    paired["postmatch_motion_scale_span"] = paired.compact_postmatch_depth_p90_span_um.le(300)
    paired.sort_values(["strict_delta", "compact_strict"], ascending=False).to_csv(
        args.output / "paired_core_evidence.csv"
    )

    counters = pd.read_csv(args.input / "arm_counters.csv").set_index("arm")
    summary = {
        "schema": "luke0804-imec1-compact-core-pilot-audit-v1",
        "input": str(args.input.resolve()),
        "core_families": int(len(paired)),
        "core_family_strict_broad": int(paired.broad_strict.sum()),
        "core_family_strict_compact": int(paired.compact_strict.sum()),
        "core_family_strict_delta": int(paired.strict_delta.sum()),
        "families_improved": int(paired.strict_delta.gt(0).sum()),
        "families_tied": int(paired.strict_delta.eq(0).sum()),
        "families_worsened": int(paired.strict_delta.lt(0).sum()),
        "all_own_strict_broad": int(broad.qualification_strict.sum()),
        "all_own_strict_compact": int(compact.qualification_strict.sum()),
        "all_combined_strict_broad": int(counters.loc["broad", "strict_all_combined"]),
        "all_combined_strict_compact": int(counters.loc["compact", "strict_all_combined"]),
        "decoy_winners_broad": int(counters.loc["broad", "decoy_winners"]),
        "decoy_winners_compact": int(counters.loc["compact", "decoy_winners"]),
        "external_winners_broad": int(counters.loc["broad", "external_winners"]),
        "external_winners_compact": int(counters.loc["compact", "external_winners"]),
        "legacy_full_proposal_denominator_warning": (
            "The pilot's qualifies_depth_blind flag divides both arms by broad train_members. "
            "It is retained for provenance but is not a fair compact-core comparison."
        ),
        "interpretation": (
            "Compact timing-aligned cores improve direct frozen match yield in this bounded pilot. "
            "A fresh split must independently form proposals before any lighthouse identity is promoted."
        ),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    plot = paired.sort_values("strict_delta")
    y = np.arange(len(plot))
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.hlines(y, plot.broad_strict, plot.compact_strict, color="0.78", lw=2)
    ax.scatter(plot.broad_strict, y, label="Broad proposal template", s=36, color="#4C78A8")
    ax.scatter(plot.compact_strict, y, label="Timing-aligned compact core", s=36, color="#F58518")
    ax.set_yticks(y, plot.index)
    ax.set_xlabel("Strict matches in frozen 310–320 s qualification window")
    ax.set_title("imec1 compact-core pilot: paired held-out match yield\n20 families with reciprocal split-half cores; no sorter or motion input")
    ax.grid(axis="x", alpha=.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(args.output / "01_paired_strict_matches.png", dpi=180)
    fig.savefig(args.output / "01_paired_strict_matches.pdf")


if __name__ == "__main__":
    main()
