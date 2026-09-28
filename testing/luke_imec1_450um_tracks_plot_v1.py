#!/usr/bin/env python
"""Plot the eight compact-core proposals retained by the 450-um audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from testing.luke_imec1_compact_tracks_plot_v1 import make_grid
from testing.luke_imec1_dots_raw_lighthouse_check import geometry_and_mapping


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    events = pd.read_csv(args.input / "qualification_events_compact.csv")
    audit = pd.read_csv(args.audit / "candidate_tracks_450um.csv")
    kept = audit[audit.candidate_under_450um_audit].sort_values("strict_matches", ascending=False)
    families = kept.family_id.tolist()
    notes = {}
    for row in kept.itertuples():
        category = "local" if row.postmatch_depth_category_450um == "localized_le120um" else "motion-scale"
        if row.time_block_valid:
            temporal = "TB pass" if row.reproduces_time_blocks else "TB FAIL"
        else:
            temporal = "TB insufficient"
        notes[row.family_id] = f"P90 {row.strict_depth_p90_span_um:.1f} µm · {category} · {temporal}"
    geom, _ = geometry_and_mapping()
    background = events[["event_id", "time_s", "peak_channel"]].copy()
    background["peak_depth_um"] = geom[background.peak_channel.to_numpy(int), 1]
    make_grid(args.output / "01_candidates_450um_local_depth_tracks", families, events, background,
              "imec1 compact-core proposals retained under 450 µm allowance", ncols=4, local=True, notes=notes)
    make_grid(args.output / "02_candidates_450um_whole_probe_context", families, events, background,
              "imec1 retained 450 µm proposals in whole-probe context", ncols=4, local=False, notes=notes)
    plotted = events[events.winner_bank.eq("s300") & events.winner_family.isin(families)].copy()
    plotted.to_csv(args.output / "plotted_candidate_hypotheses.csv", index=False)
    receipt = {
        "schema": "luke0804-imec1-450um-tracks-plot-v1", "arm": "compact",
        "time_window_s": [310.0, 320.0], "candidate_count": len(families), "candidate_families": families,
        "selection": ">=5 strict events and strict-event P90 waveform-depth span <=450 um",
        "candidate_hypotheses": len(plotted), "identity_status_counts": plotted.status.value_counts().to_dict(),
        "depth_contract": "Absolute waveform centroid was revealed after depth-blind identity scoring; pale points are all detector peak depths.",
    }
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
