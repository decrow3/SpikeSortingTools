#!/usr/bin/env python
"""Plot compact-core proposal evidence as absolute depth versus time."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from testing.luke_imec1_dots_raw_lighthouse_check import geometry_and_mapping

STATUS_STYLE = {
    "rejected": dict(marker="x", color="#9C9AA5", s=8, alpha=.25, linewidths=.45, label="Rejected winner"),
    "identity_ambiguous": dict(marker="o", facecolors="none", edgecolors="#E69F00", s=18, alpha=.75,
                               linewidths=.8, label="Identity ambiguous"),
    "lower_score": dict(marker="^", color="#4C78A8", s=17, alpha=.85, linewidths=.4, label="Lower score"),
    "strict": dict(marker="o", color="#111111", s=19, alpha=.9, linewidths=.3, label="Strict"),
}


def draw_panel(ax, family, events, background, local=True, note=""):
    candidate = events[events.winner_family.eq(family)].copy()
    candidate["plot_status"] = candidate.status.where(
        candidate.status.isin(["strict", "lower_score", "identity_ambiguous"]), "rejected"
    )
    accepted = candidate[candidate.plot_status.ne("rejected")]
    source = accepted.waveform_centroid_um if len(accepted) else candidate.waveform_centroid_um
    if local and len(source):
        lo, hi = np.quantile(source, [.01, .99])
        pad = max(35.0, .12 * max(hi - lo, 1.0)); ylim = (max(0, lo - pad), min(3820, hi + pad))
    else:
        ylim = (0, 3820)
    bg = background[(background.peak_depth_um >= ylim[0]) & (background.peak_depth_um <= ylim[1])]
    ax.scatter(bg.time_s, bg.peak_depth_um, s=1.2, color="#D0D0D0", alpha=.20, linewidths=0,
               rasterized=True, label="All detector peaks")
    for status in ("rejected", "identity_ambiguous", "lower_score", "strict"):
        rows = candidate[candidate.plot_status.eq(status)]
        if len(rows):
            ax.scatter(rows.time_s, rows.waveform_centroid_um, rasterized=True, **STATUS_STYLE[status])
    counts = candidate.plot_status.value_counts()
    supported = sum(int(counts.get(status, 0)) for status in ("strict", "lower_score", "identity_ambiguous"))
    strict = int(counts.get("strict", 0))
    title = f"{family}  strict {strict/10:.1f} Hz | displayed {supported/10:.1f} Hz"
    if note:
        title += "\n" + note
    ax.set_title(title, fontsize=9)
    ax.set_xlim(310, 320); ax.set_ylim(*ylim); ax.grid(alpha=.16, linewidth=.5)
    ax.tick_params(labelsize=8)


def make_grid(path, families, events, background, title, ncols, local=True, notes=None):
    nrows = int(np.ceil(len(families) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.15 * ncols, 3.05 * nrows), sharex=True, squeeze=False)
    for ax, family in zip(axes.flat, families):
        draw_panel(ax, family, events, background, local=local, note=(notes or {}).get(family, ""))
    for ax in axes.flat[len(families):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        if ax.get_visible(): ax.set_xlabel("Recording-relative time (s)")
    for row in axes:
        if row[0].get_visible(): row[0].set_ylabel("Absolute waveform depth (µm)")
    handles = [plt.Line2D([], [], linestyle="", marker=".", color="#B8B8B8", label="All detector peak depths"),
               plt.Line2D([], [], linestyle="", marker="x", color="#9C9AA5", label="Rejected winner"),
               plt.Line2D([], [], linestyle="", marker="o", markerfacecolor="none", markeredgecolor="#E69F00", label="Identity ambiguous"),
               plt.Line2D([], [], linestyle="", marker="^", color="#4C78A8", label="Lower score"),
               plt.Line2D([], [], linestyle="", marker="o", color="#111111", label="Strict")]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=9)
    subtitle = "Preferred compact-unshifted arm; 310–320 s frozen qualification; independent depth scales" if local else \
               "Preferred compact-unshifted arm; 310–320 s frozen qualification; common whole-probe depth scale"
    fig.suptitle(title + "\n" + subtitle, fontsize=14)
    fig.tight_layout(rect=(0, .045, 1, .94))
    fig.savefig(path.with_suffix(".png"), dpi=180)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    events = pd.read_csv(args.input / "qualification_events_compact.csv")
    cores = pd.read_csv(args.input / "training_core_reproducibility.csv")
    evidence = pd.read_csv(args.audit / "core_family_rate_and_match_evidence.csv")
    families = evidence.sort_values(["compact", "compact_minus_broad"], ascending=False).family_id.tolist()
    active = evidence[evidence.compact >= 5].sort_values("compact", ascending=False).family_id.tolist()
    geom, _ = geometry_and_mapping()
    background = events[["event_id", "time_s", "peak_channel"]].copy()
    background["peak_depth_um"] = geom[background.peak_channel.to_numpy(int), 1]
    core_set = set(cores.loc[cores.odd_even_valid, "family_id"])
    if set(families) != core_set or len(events) != events.event_id.nunique():
        raise RuntimeError("plot inputs do not preserve the 20-core/event grain")
    make_grid(args.output / "01_all20_local_depth_tracks", families, events, background,
              "imec1 compact-core proposal evidence over time", ncols=4, local=True)
    make_grid(args.output / "02_active_candidates_local_depth_tracks", active, events, background,
              "imec1 compact-core proposals with ≥5 strict matches", ncols=4, local=True)
    make_grid(args.output / "03_all20_whole_probe_context", families, events, background,
              "imec1 compact-core proposals in whole-probe context", ncols=4, local=False)
    plotted = events[events.winner_bank.eq("s300") & events.winner_family.isin(families)].copy()
    plotted.to_csv(args.output / "plotted_candidate_hypotheses.csv", index=False)
    receipt = {
        "schema": "luke0804-imec1-compact-tracks-plot-v1", "arm": "compact",
        "time_window_s": [310.0, 320.0], "candidate_families": families,
        "active_families_ge5_strict": active, "qualification_detections": len(events),
        "candidate_winner_hypotheses": len(plotted),
        "identity_status_counts": plotted.status.value_counts().to_dict(),
        "depth_contract": "Waveform centroids are post-match overlays; detector peak depths are background context.",
        "rate_contract": "Strict and displayed event counts divided by the fixed 10 s duration are rates, not recovery probabilities.",
    }
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
