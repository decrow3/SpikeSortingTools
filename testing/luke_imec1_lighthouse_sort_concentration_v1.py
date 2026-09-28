#!/usr/bin/env python
"""One-to-one matching of frozen imec1 lighthouse events to existing sorts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVENTS = ROOT / "testing/outputs/luke_imec1_lighthouse_motion_comparison_v1/event_predictions.parquet"
DEFAULT_OUT = ROOT / "testing/outputs/luke_imec1_lighthouse_sort_concentration_v1"
FS = 29999.759166666667
CROP_START_FRAME = 193737
SORTS = {
    "native_ap": Path("/media/huklab/Data/NPX/Ryansorting/Luke/DARTsort_motion_experiments/luke0804-imec1-local-spikeglx-dots-v4/sorts/native-v1"),
    "medicine": Path("/media/huklab/Data/NPX/Ryansorting/Luke/DARTsort_motion_experiments/jobs/full-probe-fine-medicine-v1/sort"),
}


def one_to_one_time_assignment(reference_samples: np.ndarray, spike_samples: np.ndarray, tolerance_samples: int) -> list[tuple[int, int]]:
    """Greedy minimum-distance assignment; each reference and spike is used once."""
    candidates: list[tuple[int, int, int]] = []
    for reference_index, sample in enumerate(reference_samples):
        left = np.searchsorted(spike_samples, sample - tolerance_samples, side="left")
        right = np.searchsorted(spike_samples, sample + tolerance_samples, side="right")
        candidates.extend((abs(int(spike_samples[j] - sample)), reference_index, j) for j in range(left, right))
    assigned_reference: set[int] = set()
    assigned_spike: set[int] = set()
    pairs = []
    for _, reference_index, spike_index in sorted(candidates):
        if reference_index in assigned_reference or spike_index in assigned_spike:
            continue
        assigned_reference.add(reference_index); assigned_spike.add(spike_index)
        pairs.append((reference_index, spike_index))
    return sorted(pairs)


def match_arm(events: pd.DataFrame, sort_dir: Path, arm: str, tolerance_ms: float, spatial_um: float) -> pd.DataFrame:
    with np.load(sort_dir / "dartsort_sorting.npz") as sorting:
        times = sorting["times_samples"].astype(np.int64)
        labels = sorting["labels"].astype(np.int32)
        channels = sorting["channels"].astype(np.int32)
        geom = sorting["geom"].astype(float)
    retained = labels >= 0
    original_index = np.flatnonzero(retained)
    times, labels, channels = times[retained], labels[retained], channels[retained]
    order = np.argsort(times, kind="stable")
    times, labels, channels = times[order], labels[order], channels[order]
    tolerance = int(round(tolerance_ms * FS / 1000))
    reference = (events.raw_ap_frame.to_numpy(np.int64) - CROP_START_FRAME)
    pairs = one_to_one_time_assignment(reference, times, tolerance)
    rows = []
    for event_index, spike_index in pairs:
        event = events.iloc[event_index]
        channel = int(channels[spike_index])
        depth = float(geom[channel, 1]) if 0 <= channel < len(geom) else np.nan
        if not np.isfinite(depth) or abs(depth - event.waveform_centroid_um) > spatial_um:
            continue
        rows.append({
            "arm": arm, "tolerance_ms": tolerance_ms, "event_key": event.event_key, "family_uid": event.family_uid,
            "reference_raw_frame": int(event.raw_ap_frame), "sort_spike_index": int(original_index[order[spike_index]]),
            "time_error_samples": int(times[spike_index] - reference[event_index]),
            "reference_depth_um": float(event.waveform_centroid_um), "sort_channel_depth_um": depth,
            "depth_error_um": depth - float(event.waveform_centroid_um), "label": int(labels[spike_index]),
        })
    return pd.DataFrame(rows)


def family_metrics(events: pd.DataFrame, matches: pd.DataFrame, arm: str, tolerance_ms: float) -> pd.DataFrame:
    rows = []
    arm_matches = matches.loc[matches.arm.eq(arm) & matches.tolerance_ms.eq(tolerance_ms)]
    for family_uid, reference in events.groupby("family_uid"):
        matched = arm_matches.loc[arm_matches.family_uid.eq(family_uid)]
        counts = matched.loc[matched.label.ge(0), "label"].value_counts()
        dominant = int(counts.iloc[0]) if len(counts) else 0
        n90 = int(np.searchsorted(counts.cumsum().to_numpy(), .9 * counts.sum()) + 1) if counts.sum() else 0
        rows.append({
            "arm": arm, "tolerance_ms": tolerance_ms, "family_uid": family_uid, "reference_events": len(reference),
            "matched_events": len(matched), "recovery": len(matched) / len(reference),
            "dominant_cluster_events": dominant, "overall_concentration": dominant / len(reference),
            "conditional_concentration": dominant / len(matched) if len(matched) else np.nan,
            "nonnegative_clusters": int(len(counts)), "clusters_to_90pct": n90,
            "eligible_20_events": bool(len(reference) >= 20),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--tolerance-ms", type=float, default=.5)
    parser.add_argument("--spatial-um", type=float, default=100.0)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    events_path = args.events if args.events.exists() else args.events.with_suffix(".csv")
    events = pd.read_parquet(events_path) if events_path.suffix == ".parquet" else pd.read_csv(events_path)
    tolerances = [args.tolerance_ms, 0.2] if args.tolerance_ms != 0.2 else [0.2]
    matches = pd.concat([match_arm(events, path, arm, tolerance, args.spatial_um)
                         for tolerance in tolerances for arm, path in SORTS.items()], ignore_index=True)
    try: matches.to_parquet(args.output / "sort_event_matches.parquet", index=False)
    except ImportError: matches.to_csv(args.output / "sort_event_matches.csv", index=False)
    metrics = pd.concat([family_metrics(events, matches, arm, tolerance)
                         for tolerance in tolerances for arm in SORTS], ignore_index=True)
    metrics.to_csv(args.output / "sort_family_metrics.csv", index=False)
    contingency = matches.loc[matches.label.ge(0)].groupby(["arm", "tolerance_ms", "label", "family_uid"]).size().rename("events").reset_index()
    contingency.to_csv(args.output / "family_cluster_contingency.csv", index=False)
    mixed = contingency.groupby(["arm", "tolerance_ms", "label"]).family_uid.nunique().rename("families").reset_index()
    arm_summary = {}
    for (arm, tolerance), group in metrics.groupby(["arm", "tolerance_ms"]):
        match_group = matches.loc[matches.arm.eq(arm) & matches.tolerance_ms.eq(tolerance)]
        mix_group = mixed.loc[mixed.arm.eq(arm) & mixed.tolerance_ms.eq(tolerance)]
        arm_summary[f"{arm}:{tolerance:g}ms"] = {
            "matched_events": int(len(match_group)),
            "mean_family_recovery": float(group.recovery.mean()),
            "mean_overall_concentration": float(group.overall_concentration.mean()),
            "mean_conditional_concentration": float(group.conditional_concentration.mean()),
            "mean_nonnegative_clusters": float(group.nonnegative_clusters.mean()),
            "labels_receiving_multiple_families": int((mix_group.families > 1).sum()),
        }
    summary = {
        "schema": "luke0804-imec1-lighthouse-sort-concentration-v1",
        "status": "validation_windows_complete" if "evidence_role" in events and events.evidence_role.eq("extra_validation").any() else "qualification_smoke_complete",
        "reference_events": len(events), "arms": list(SORTS), "tolerance_ms": tolerances,
        "spatial_gate_um_raw_acquisition_coordinates": args.spatial_um,
        "families_eligible_20_events": int(metrics.groupby("family_uid").eligible_20_events.first().sum()),
        "labels_receiving_multiple_families": int((mixed.families > 1).sum()),
        "arm_summary": arm_summary,
        "interpretation": "Existing-sort concentration is accompanied by recovery and cross-family mixing and is not a motion-quality verdict.",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
