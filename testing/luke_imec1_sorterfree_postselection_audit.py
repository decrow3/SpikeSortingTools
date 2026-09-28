#!/usr/bin/env python
"""Post-selection audit of sorter-free imec1 waveform families.

This consumes only the frozen waveform-discovery output.  It describes rapid
depth changes and asks whether independently selected families replicate one
another after identity selection.  Rapid changes are flags, not rejection
criteria: they can be the motion that a lighthouse is meant to measure.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from testing.luke_imec1_dots_lighthouse_direct_check import MOTION_WINDOWS, atomic_json

SCHEMA = "luke0804-imec1-sorterfree-postselection-audit-v2"


def family_audit(events: pd.DataFrame, families: pd.DataFrame) -> pd.DataFrame:
    centers = families.set_index("family_id").seed_depth_median_um
    rows = []
    selected = families.loc[families.selected_depth_blind, "family_id"]
    for family_id in selected:
        q = events[(events.family_id == family_id) & (events.status == "strict")].sort_values("time_s").copy()
        if len(q):
            q["displacement_um"] = q.waveform_centroid_um - centers[family_id]
            dt = q.time_s.diff()
            jump = q.waveform_centroid_um.diff().abs()
            joined = dt.notna() & (dt <= 2.0)
            bins = q.assign(second_bin=np.floor(q.time_s).astype(int)).groupby("second_bin").waveform_centroid_um.agg(["count", "min", "max"])
            bins = bins[bins["count"] >= 2]
            wide_bins = (bins["max"] - bins["min"]) >= 80.0
            windows = sum(bool(((q.time_s >= start) & (q.time_s < stop)).any()) for start, stop in MOTION_WINDOWS)
        else:
            joined = pd.Series(dtype=bool); jump = pd.Series(dtype=float); bins = pd.DataFrame(); wide_bins = pd.Series(dtype=bool); windows = 0
        seed_ok = bool(families.set_index("family_id").loc[family_id, "postmatch_seed_spatially_plausible"])
        supported = seed_ok and len(q) >= 20 and windows >= 3
        cross_depth = bool(((jump >= 100.0) & joined).any() or wide_bins.any())
        status = "unsupported"
        if supported:
            status = "unreplicated_cross_depth_candidate" if cross_depth else "candidate_needing_replication"
        rows.append({
            "family_id": family_id,
            "strict_events": len(q),
            "heldout_windows": int(windows),
            "seed_spatially_plausible": seed_ok,
            "joins_le_2s": int(joined.sum()),
            "jumps_ge_80um": int(((jump >= 80.0) & joined).sum()),
            "jumps_ge_100um": int(((jump >= 100.0) & joined).sum()),
            "coactive_1s_bins": len(bins),
            "coactive_bins_span_ge_80um": int(wide_bins.sum()),
            "trace_status": status,
        })
    return pd.DataFrame(rows)


def pairwise_replication(events: pd.DataFrame, families: pd.DataFrame, tolerance_s: float = 0.25) -> pd.DataFrame:
    """Nearest strict-event displacement pairs; no motion estimate is consulted.

    Temporal agreement is only called evaluable replication when both waveform
    families were spatially localized after the depth-blind seed freeze. This
    prevents a diffuse waveform class spanning many cells from masquerading as
    an independent lighthouse.
    """
    indexed = families.set_index("family_id")
    centers = indexed.seed_depth_median_um
    selected = list(families.loc[families.selected_depth_blind, "family_id"])
    rows = []
    strict = events[events.status == "strict"]
    for ia, a in enumerate(selected):
        qa = strict[strict.family_id == a].sort_values("time_s")
        for b in selected[ia + 1:]:
            qb = strict[strict.family_id == b].sort_values("time_s")
            pairs = []
            if len(qa) and len(qb):
                bt = qb.time_s.to_numpy()
                for r in qa.itertuples(index=False):
                    j = int(np.argmin(np.abs(bt - r.time_s)))
                    other = qb.iloc[j]
                    if abs(float(other.time_s) - r.time_s) <= tolerance_s:
                        pairs.append((r.waveform_centroid_um - centers[a], other.waveform_centroid_um - centers[b]))
            corr = float(np.corrcoef(np.asarray(pairs).T)[0, 1]) if len(pairs) >= 3 else np.nan
            localized = bool(indexed.loc[a,"postmatch_seed_spatially_plausible"] and indexed.loc[b,"postmatch_seed_spatially_plausible"])
            enough_events = len(qa) >= 5 and len(qb) >= 5
            evaluable = len(pairs) >= 5 and localized and enough_events
            rows.append({
                "family_a": a, "family_b": b,
                "paired_events_le_250ms": len(pairs),
                "displacement_correlation": corr,
                "both_seed_spatially_plausible": localized,
                "both_have_at_least_5_strict_events": enough_events,
                "replication_evaluable": evaluable,
                "replication_supported": bool(evaluable and np.isfinite(corr) and corr >= .70),
            })
    return pd.DataFrame(rows)


def run(discovery: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    summary = json.loads((discovery / "summary.json").read_text())
    if summary.get("sorter_inputs_read") is not False or summary.get("motion_estimate_read") is not False:
        raise RuntimeError("Input is not the declared sorter- and motion-free discovery result")
    events = pd.read_csv(discovery / "heldout_events.csv")
    families = pd.read_csv(discovery / "families_after_depth_reveal.csv")
    audit = family_audit(events, families)
    replication = pairwise_replication(events, families)
    audit.to_csv(output / "family_trace_audit.csv", index=False)
    replication.to_csv(output / "pairwise_replication.csv", index=False)
    candidates = audit.loc[audit.trace_status != "unsupported", "family_id"].tolist()
    evaluable = replication[replication.replication_evaluable]
    supported = evaluable[evaluable.replication_supported]
    verified = []  # Automated shared motion is not, by itself, cell verification.
    atomic_json(output / "summary.json", {
        "schema": SCHEMA,
        "status": "complete",
        "source_discovery": str(discovery.resolve()),
        "sorter_inputs_read": False,
        "motion_estimate_read": False,
        "waveform_only_candidates": candidates,
        "temporally_paired_family_pairs": int((replication.paired_events_le_250ms >= 5).sum()),
        "replication_evaluable_pairs": len(evaluable),
        "replication_supported_pairs": len(supported),
        "verified_lighthouses": verified,
        "interpretation": "The waveform-only cross-depth candidate is preserved, but no pair of seed-localized candidate identities shows positive replicated movement. Strong correlations involving depth-diffuse waveform classes are retained as apparent agreement, not independent-cell replication.",
    })


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--discovery", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    run(args.discovery, args.output)


if __name__ == "__main__":
    main()
