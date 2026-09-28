#!/usr/bin/env python
"""Plot frozen waveform-only lighthouse events over dense motion-field traces."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from testing.luke_imec1_lighthouse_motion_comparison_v1 import (
    CROP_START_FRAME,
    DEFAULT_LFP,
    FS,
    load_fields,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVENTS = ROOT / "testing/outputs/luke_imec1_lighthouse_motion_comparison_v6/event_predictions.csv"
DEFAULT_OUTPUT = ROOT / "testing/outputs/luke_imec1_lighthouse_motion_comparison_v6/lighthouses_alongside_motion_traces.png"
DEFAULT_CONTEXT = ROOT / "testing/outputs/luke_imec1_dots_sorterfree_waveform_discovery_v3_s300/heldout_events.csv"
WINDOWS = ((120, 150), (310, 320), (320, 350), (484, 514), (696, 726), (941, 971), (1120, 1150))
COLORS = {
    "native_ap": "#0072B2",
    "medicine": "#CC79A7",
    "lfp80": "#D55E00",
    "lfp260": "#009E73",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--lfp-root", type=Path, default=DEFAULT_LFP)
    parser.add_argument("--context-events", type=Path, default=DEFAULT_CONTEXT)
    parser.add_argument("--trace-step-s", type=float, default=.25)
    parser.add_argument("--fields", default="native_ap,medicine,lfp80,lfp260")
    args = parser.parse_args()

    events = pd.read_csv(args.events)
    context = pd.read_csv(args.context_events)
    context = context.loc[context.status.eq("strict")].copy()
    fields, _ = load_fields(args.lfp_root)
    requested_fields = [name.strip() for name in args.fields.split(",") if name.strip()]
    shown_fields = [name for name in requested_fields if name in fields]
    time = np.arange(0.0, max(1160.0, float(events.time_s.max()) + 1), args.trace_step_s)

    fig, axes = plt.subplots(4, 2, figsize=(16, 13), sharex=True, layout="constrained")
    for ax, (family_uid, group) in zip(axes.flat, events.groupby("family_uid")):
        family_id = family_uid.split(":")[2]
        reference_depth = float(group.reference_depth_um.iloc[0])
        qualification = group.loc[group.evidence_role.eq("qualification")]
        lighthouse_anchor = float(qualification.observed_relative_um.median())
        for start, stop in WINDOWS:
            ax.axvspan(start, stop, color="#E5E5E5", alpha=.35, linewidth=0)
        for name in shown_fields:
            depth = np.full_like(time, reference_depth)
            crop_time = time - CROP_START_FRAME / FS
            displacement, supported = fields[name].sample(crop_time, depth)
            anchor_mask = supported & (time >= 310.0) & (time < 320.0)
            if not anchor_mask.any():
                continue
            plotted = displacement - np.median(displacement[anchor_mask])
            plotted[~supported] = np.nan
            ax.plot(time, plotted, color=COLORS[name], lw=1.15, alpha=.9, label=name)
        family_context = context.loc[context.family_id.eq(family_id)]
        if not family_context.empty:
            ax.scatter(
                family_context.time_s,
                family_context.waveform_centroid_um - reference_depth - lighthouse_anchor,
                s=18,
                facecolor="none",
                edgecolor="#666666",
                linewidth=.8,
                alpha=.8,
                zorder=4,
                label="earlier same-bank broad strict",
            )
        ax.scatter(
            group.time_s,
            group.observed_relative_um - lighthouse_anchor,
            s=19,
            facecolor="black",
            edgecolor="white",
            linewidth=.25,
            alpha=.78,
            zorder=5,
            label="waveform depth",
        )
        ax.axhline(0.0, color="black", lw=.6, alpha=.3)
        ax.set_title(f"{family_id}  (reference {reference_depth:.0f} µm)")
        ax.set_ylabel("relative depth / displacement (µm)")
        ax.grid(axis="y", alpha=.15)

    axes[0, 0].legend(loc="upper right", fontsize=8, ncol=3)
    axes[-1, 0].set_xlabel("AP recording-relative time (s)")
    axes[-1, 1].set_xlabel("AP recording-relative time (s)")
    fig.suptitle(
        "Luke0804 imec1 waveform-only lighthouse candidates alongside motion traces\n"
        "All series zeroed to their 310–320 s interval median; no lag or scale fit"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
