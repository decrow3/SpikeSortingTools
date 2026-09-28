#!/usr/bin/env python
"""Candidate-atlas figures with one independently sampled motion field per set."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from testing.luke_imec1_lighthouse_motion_comparison_v1 import (
    CROP_START_FRAME,
    DEFAULT_LFP,
    FAMILIES,
    FS,
    load_fields,
)


ROOT = Path(__file__).resolve().parents[1]
ASSESSMENT = ROOT / "testing/outputs/luke_imec1_twenty_candidate_assessment_v1/twenty_candidate_assessment.csv"
QUALIFICATION = ROOT / "testing/outputs/luke_imec1_compact_alignment_pilot_v2/qualification_events_compact.csv"
EXTRA = ROOT / "testing/outputs/luke_imec1_lighthouse_extra_windows_v3"
HELDOUT = ROOT / "testing/outputs/luke_imec1_dots_sorterfree_waveform_discovery_v3_s300/heldout_events.csv"
DEFAULT_OUTPUT = ROOT / "testing/outputs/luke_imec1_lighthouse_motion_comparison_v7/candidate_motion_atlases"
VALIDATION_WINDOWS = ((120, 150), (320, 350), (484, 514), (696, 726), (941, 971), (1120, 1150))
COMPACT_SEGMENTS = (
    (120, 150), (310, 350), (484, 514), (696, 726), (935, 971),
    (974, 986), (998, 1002), (1014, 1025), (1071, 1075),
    (1097, 1112), (1120, 1150),
)
FIELD_LABELS = {
    "native_ap": "Native DARTsort AP DREDGE",
    "medicine": "Fast MEDiCINe (nonrigid)",
    "medicine_rigid": "Fast MEDiCINe rigid projection",
    "lfp80": "DREDGE-LFP unsharpened, 80 µm search",
    "lfp260": "DREDGE-LFP unsharpened, 260 µm search",
}
STATUS_STYLE = {
    "strict": dict(marker="o", facecolors="black", edgecolors="black", s=15, linewidths=.3),
    "identity_ambiguous": dict(marker="o", facecolors="none", edgecolors="#E69F00", s=18, linewidths=.8),
    "lower_score": dict(marker="^", facecolors="#4477AA", edgecolors="#4477AA", s=19, linewidths=.3),
    "unmatched": dict(marker="x", c="#888888", s=13, linewidths=.6),
}


def load_evidence() -> tuple[pd.DataFrame, pd.DataFrame]:
    qualification = pd.read_csv(QUALIFICATION)
    qualification["source"] = "compact_qualification"
    qualification = qualification.rename(columns={"winner_family": "family_id"})
    qualification = qualification.loc[
        qualification.winner_bank.eq("s300") & qualification.winner_kind.eq(0)
    ]

    extra_tables = []
    background_tables = [qualification[["time_s", "waveform_centroid_um"]]]
    for path in sorted(EXTRA.glob("window_[0-9][0-9]_events.csv")):
        table = pd.read_csv(path)
        background_tables.append(table[["time_s", "waveform_centroid_um"]])
        table = table.loc[table.winner_bank.eq("s300") & table.winner_kind.eq(0)].copy()
        table["source"] = "compact_validation"
        table = table.rename(columns={"winner_family": "family_id"})
        extra_tables.append(table)

    heldout = pd.read_csv(HELDOUT)
    heldout["source"] = "same_bank_broad_heldout"
    evidence = pd.concat([qualification, *extra_tables, heldout], ignore_index=True, sort=False)
    background = pd.concat(background_tables, ignore_index=True)
    if len(background) > 100_000:
        background = background.iloc[np.linspace(0, len(background) - 1, 100_000, dtype=int)]
    return evidence, background


def compact_coordinates(time_s: np.ndarray) -> np.ndarray:
    result = np.full(np.asarray(time_s).shape, np.nan, dtype=float)
    offset = 0.0
    for start, stop in COMPACT_SEGMENTS:
        selected = (time_s >= start) & (time_s < stop)
        result[selected] = offset + time_s[selected] - start
        offset += stop - start
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--lfp-root", type=Path, default=DEFAULT_LFP)
    parser.add_argument("--trace-step-s", type=float, default=.25)
    parser.add_argument("--compact-time", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    assessment = pd.read_csv(ASSESSMENT).sort_values("assessment_rank")
    families = pd.read_csv(FAMILIES).set_index("family_id")
    evidence, background = load_evidence()
    fields, receipt = load_fields(args.lfp_root)
    time_ap = np.arange(0.0, 1160.0, args.trace_step_s)
    time_crop = time_ap - CROP_START_FRAME / FS
    background_x = compact_coordinates(background.time_s.to_numpy(float)) if args.compact_time else background.time_s.to_numpy(float)
    background_keep = np.isfinite(background_x)
    segment_starts = np.cumsum([0, *[stop - start for start, stop in COMPACT_SEGMENTS[:-1]]])
    segment_lengths = np.asarray([stop - start for start, stop in COMPACT_SEGMENTS])

    for field_name in FIELD_LABELS:
        if field_name not in fields:
            continue
        fig, axes = plt.subplots(5, 4, figsize=(18, 17), sharex=True, layout="constrained")
        field = fields[field_name]
        for ax, row in zip(axes.flat, assessment.itertuples()):
            family_id = row.family_id
            reference_depth = float(families.loc[family_id, "seed_depth_median_um"])
            candidate = evidence.loc[evidence.family_id.eq(family_id)]

            ax.scatter(
                background_x[background_keep],
                background.waveform_centroid_um.to_numpy(float)[background_keep],
                marker=".", s=.18, color="#B8B8B8", alpha=.13, rasterized=True,
            )
            if args.compact_time:
                for index, (display_start, length) in enumerate(zip(segment_starts, segment_lengths)):
                    if index % 2:
                        ax.axvspan(display_start, display_start + length, color="#F3F3F3", alpha=.5, linewidth=0)
                    if index:
                        ax.axvline(display_start, color="#B0B0B0", lw=.45, alpha=.7)
            else:
                for start, stop in VALIDATION_WINDOWS:
                    ax.axvspan(start, stop, color="#ECECEC", alpha=.55, linewidth=0)
                ax.axvspan(310, 320, color="#DCEAF7", alpha=.45, linewidth=0)

            displacement, supported = field.sample(time_crop, np.full_like(time_crop, reference_depth))
            anchor = supported & (time_ap >= 310) & (time_ap < 320)
            trace_depth = np.array([])
            if anchor.any():
                field_anchor = np.median(displacement[anchor])
                trace_depth = reference_depth + displacement - field_anchor
                trace_depth[~supported] = np.nan
                if args.compact_time:
                    for display_start, (start, stop) in zip(segment_starts, COMPACT_SEGMENTS):
                        selected = (time_ap >= start) & (time_ap < stop)
                        ax.plot(
                            display_start + time_ap[selected] - start,
                            trace_depth[selected], color="#7B3294", lw=.9, alpha=.9, zorder=2,
                        )
                else:
                    ax.plot(time_ap, trace_depth, color="#7B3294", lw=.9, alpha=.9, zorder=2)

            for status, style in STATUS_STYLE.items():
                selected = candidate.loc[candidate.status.eq(status)]
                if not selected.empty:
                    selected_x = compact_coordinates(selected.time_s.to_numpy(float)) if args.compact_time else selected.time_s.to_numpy(float)
                    keep = np.isfinite(selected_x)
                    ax.scatter(selected_x[keep], selected.waveform_centroid_um.to_numpy(float)[keep], zorder=4, **style)

            if anchor.any():
                trace_keep = np.isfinite(trace_depth)
                if args.compact_time:
                    trace_keep &= np.isfinite(compact_coordinates(time_ap))
                finite_trace = trace_depth[trace_keep]
            else:
                finite_trace = np.array([])
            candidate_x_for_limits = compact_coordinates(candidate.time_s.to_numpy(float)) if args.compact_time else candidate.time_s.to_numpy(float)
            values = candidate.loc[np.isfinite(candidate_x_for_limits), "waveform_centroid_um"].dropna().to_numpy(float)
            if finite_trace.size:
                values = np.r_[values, np.quantile(finite_trace, [.01, .99])]
            if values.size:
                low, high = np.min(values), np.max(values)
                pad = max(25.0, .06 * max(high - low, 1.0))
                ax.set_ylim(max(-50, low - pad), min(3900, high + pad))
            ax.set_title(f"{family_id}  {row.assessment_tier}", fontsize=9)
            ax.grid(alpha=.12, linewidth=.4)

        for ax in axes[:, 0]:
            ax.set_ylabel("absolute waveform depth (µm)")
        for ax in axes[-1]:
            if args.compact_time:
                ticks = segment_starts + segment_lengths / 2
                ax.set_xticks(ticks, [f"{start:g}–{stop:g}" for start, stop in COMPACT_SEGMENTS], rotation=55, ha="right", fontsize=6)
                ax.set_xlabel("concatenated evaluated windows (AP recording time, s)")
            else:
                ax.set_xlabel("AP recording-relative time (s)")
        fig.suptitle(
            f"Luke0804 imec1 candidate atlas — {FIELD_LABELS[field_name]}\n"
            + ("Evaluated intervals concatenated without bridging gaps; " if args.compact_time else "")
            + "purple trace zeroed over 310–320 s; independent depth scales"
        )
        legend = [
            Line2D([], [], color="#7B3294", lw=1.2, label="motion trace"),
            Line2D([], [], marker="o", color="black", lw=0, markersize=5, label="strict"),
            Line2D([], [], marker="o", markerfacecolor="none", markeredgecolor="#E69F00", lw=0, markersize=5, label="identity ambiguous"),
            Line2D([], [], marker="^", color="#4477AA", lw=0, markersize=5, label="lower score"),
            Line2D([], [], marker="x", color="#888888", lw=0, markersize=5, label="unmatched"),
            Line2D([], [], marker=".", color="#B8B8B8", lw=0, markersize=4, label="detector-depth context"),
        ]
        fig.legend(handles=legend, loc="outside lower center", ncol=6, fontsize=9)
        suffix = "_compact" if args.compact_time else ""
        fig.savefig(args.output / f"candidate_atlas_{field_name}{suffix}.png", dpi=170)
        plt.close(fig)

    summary = {
        "schema": "luke0804-imec1-motion-candidate-atlas-v1",
        "candidate_count": int(len(assessment)),
        "candidate_order": assessment.family_id.tolist(),
        "evidence_rows": int(len(evidence)),
        "background_rows_displayed": int(len(background)),
        "fields": [name for name in FIELD_LABELS if name in fields],
        "trace_step_s": args.trace_step_s,
        "compact_time": args.compact_time,
        "compact_segments_s": COMPACT_SEGMENTS if args.compact_time else None,
        "clock": "x is AP-stream time; field sampled at x - crop_start_frame/fs",
        "vertical_alignment": "seed depth + displacement minus each field's 310-320 s median; no lag or scale fit",
        "source_receipt": receipt,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
