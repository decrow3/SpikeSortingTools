"""Summarize per-unit amplitude completeness for the completed Group 2 arms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from testing.sort_comparison import load_comparison_inputs


ARMS = (
    "rescue_10_9_motion_off",
    "rescue_9_9_motion_off",
    "rescue_9_8_motion_off",
)


def summarize(arms_root: Path, recording_dir: Path, minimum_spikes: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    recording = json.loads((recording_dir / "rescue_recording_manifest.json").read_text())
    fs = float(recording["sampling_frequency_hz"])
    rows: list[dict] = []
    for arm in ARMS:
        manifest = json.loads((arms_root / arm / "candidate_manifest.json").read_text())
        sort, qc = load_comparison_inputs(
            arm,
            manifest["curated_output"],
            manifest["qc_directory"],
            sampling_frequency_hz=fs,
        )
        cluster_ids, spike_counts = np.unique(sort["cl"], return_counts=True)
        windows = qc["amplitude_windows"]
        for cluster_id, spike_count in zip(cluster_ids, spike_counts):
            # The requested threshold is strict: retain units with >2000 spikes.
            if int(spike_count) <= minimum_spikes:
                continue
            unit_windows = windows[windows.cluster_id == int(cluster_id)]
            finite = unit_windows[
                (unit_windows.status == "finite_interior")
                & np.isfinite(unit_windows.missing_pct)
            ]
            boundary_count = int((unit_windows.status == "boundary_pinned").sum())
            finite_count = int(len(finite))
            measured = finite_count >= 2
            median_missing = float(finite.missing_pct.median()) if measured else np.nan
            rows.append(
                {
                    "arm": arm,
                    "cluster_id": int(cluster_id),
                    "spike_count": int(spike_count),
                    "finite_window_count": finite_count,
                    "boundary_pinned_window_count": boundary_count,
                    "other_unusable_window_count": int(len(unit_windows) - finite_count - boundary_count),
                    "measurement_status": "measured" if measured else "fewer_than_two_finite_windows",
                    "median_missing_pct": median_missing,
                    "amplitude_completeness_pct": 100.0 - median_missing if measured else np.nan,
                }
            )
    units = pd.DataFrame(rows)
    ecdf_rows: list[dict] = []
    for arm, frame in units[units.measurement_status == "measured"].groupby("arm", sort=False):
        ordered = frame.sort_values(["amplitude_completeness_pct", "cluster_id"])
        count = len(ordered)
        for rank, row in enumerate(ordered.itertuples(index=False), start=1):
            ecdf_rows.append(
                {
                    "arm": arm,
                    "cluster_id": row.cluster_id,
                    "spike_count": row.spike_count,
                    "amplitude_completeness_pct": row.amplitude_completeness_pct,
                    "median_missing_pct": row.median_missing_pct,
                    "finite_window_count": row.finite_window_count,
                    "ecdf_fraction": rank / count,
                    "measured_unit_count": count,
                }
            )
    return units, pd.DataFrame(ecdf_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms-root", type=Path, required=True)
    parser.add_argument("--recording-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-spikes", type=int, default=2000)
    args = parser.parse_args()
    if args.minimum_spikes < 0:
        parser.error("--minimum-spikes must be nonnegative")
    units, ecdf = summarize(args.arms_root, args.recording_dir, args.minimum_spikes)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    units.to_csv(args.output_dir / "units_gt2000.csv", index=False)
    ecdf.to_csv(args.output_dir / "amplitude_completeness_ecdf.csv", index=False)
    summary = (
        units.groupby("arm", sort=False)
        .agg(
            eligible_units=("cluster_id", "size"),
            measured_units=("amplitude_completeness_pct", "count"),
            median_completeness_pct=("amplitude_completeness_pct", "median"),
            q25_completeness_pct=("amplitude_completeness_pct", lambda x: x.quantile(0.25)),
            q75_completeness_pct=("amplitude_completeness_pct", lambda x: x.quantile(0.75)),
        )
        .reset_index()
    )
    summary["measured_fraction"] = summary.measured_units / summary.eligible_units
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    print(summary.to_json(orient="records", indent=2))


if __name__ == "__main__":
    main()
