"""Trace reviewed Luke raw events through uncurated and curated sort outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .luke_raw_high_amplitude_recovery import PROBES
except ImportError:
    from luke_raw_high_amplitude_recovery import PROBES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", choices=sorted(PROBES), default="imec1")
    parser.add_argument(
        "--review-dir",
        type=Path,
        default=Path("testing/outputs/luke_multichannel_event_validation/imec1"),
    )
    parser.add_argument("--time-tolerance-ms", type=float, default=0.5)
    parser.add_argument("--depth-tolerance-um", type=float, default=100.0)
    return parser.parse_args()


def local_match_details(
    event_samples: np.ndarray,
    event_depths: np.ndarray,
    spike_times: np.ndarray,
    spike_depths: np.ndarray,
    tolerance_samples: int,
    depth_tolerance_um: float,
    assume_sorted: bool = True,
) -> pd.DataFrame:
    """Find local spike matches without materializing an event-by-spike matrix."""
    if assume_sorted:
        times = spike_times
        depths = spike_depths
    else:
        order = np.argsort(spike_times, kind="stable")
        times = np.asarray(spike_times[order], dtype=np.int64)
        depths = np.asarray(spike_depths[order], dtype=float)
    rows: list[dict] = []
    for sample, depth in zip(event_samples.astype(np.int64), event_depths.astype(float)):
        left = int(np.searchsorted(times, sample - tolerance_samples, side="left"))
        right = int(np.searchsorted(times, sample + tolerance_samples, side="right"))
        if left == right:
            rows.append(
                {
                    "present": False,
                    "n_temporal_candidates": 0,
                    "nearest_time_samples": np.nan,
                    "nearest_depth_um": np.nan,
                    "nearest_depth_error_um": np.nan,
                }
            )
            continue
        candidate_times = times[left:right]
        candidate_depths = depths[left:right]
        depth_error = np.abs(candidate_depths - depth)
        local = depth_error <= depth_tolerance_um
        if np.any(local):
            local_indices = np.flatnonzero(local)
            best_local = local_indices[
                np.argmin(np.abs(candidate_times[local_indices] - sample))
            ]
            best = int(best_local)
            present = True
        else:
            # Report the spatially nearest temporally coincident spike even when
            # it falls outside the local matching radius.
            best = int(np.argmin(depth_error))
            present = False
        rows.append(
            {
                "present": present,
                "n_temporal_candidates": int(right - left),
                "nearest_time_samples": int(candidate_times[best] - sample),
                "nearest_depth_um": float(candidate_depths[best]),
                "nearest_depth_error_um": float(depth_error[best]),
            }
        )
    return pd.DataFrame(rows)


def load_stage(path: Path) -> tuple[np.ndarray, np.ndarray]:
    times = np.load(path / "spike_times.npy", mmap_mode="r").reshape(-1)
    positions = np.load(path / "spike_positions.npy", mmap_mode="r")
    if len(times) != len(positions):
        raise ValueError(f"Array length mismatch in {path}")
    return times, positions[:, 1]


def main() -> None:
    args = parse_args()
    config = PROBES[args.probe]
    labels = pd.read_csv(args.review_dir / "review_labels.csv")
    key = pd.read_csv(args.review_dir / "review_key.csv")
    labels["review_label"] = labels["review_label"].fillna("").str.strip().str.lower()
    if (labels["review_label"] == "").any():
        raise ValueError("Complete the blinded review before tracing events")
    events = key.merge(
        labels[["review_id", "review_label", "review_confidence"]],
        on="review_id",
        validate="one_to_one",
    )
    tolerance_samples = int(
        round(args.time_tolerance_ms * 1e-3 * config.sample_rate_hz)
    )
    stage_paths = {
        "uncurated": config.sorting_path.parents[1] / "kilosort4" / "sorter_output",
        "curated": config.sorting_path,
    }
    for stage, path in stage_paths.items():
        times, depths = load_stage(path)
        details = local_match_details(
            events["sample_index"].to_numpy(),
            events["peak_depth_um"].to_numpy(),
            times,
            depths,
            tolerance_samples,
            args.depth_tolerance_um,
        ).add_prefix(f"{stage}_")
        events = pd.concat([events.reset_index(drop=True), details], axis=1)

    events["sort_stage_outcome"] = np.select(
        [
            events["curated_present"],
            events["uncurated_present"] & ~events["curated_present"],
        ],
        ["present_curated", "lost_during_curation"],
        default="absent_from_uncurated_kilosort",
    )
    events.to_csv(args.review_dir / "event_stage_trace.csv", index=False)
    summary = (
        events.groupby(["review_label", "status", "sort_stage_outcome"], observed=True)
        .size()
        .rename("n_events")
        .reset_index()
    )
    denominators = (
        events.groupby(["review_label", "status"], observed=True)
        .size()
        .rename("denominator")
        .reset_index()
    )
    summary = summary.merge(denominators, on=["review_label", "status"])
    summary["fraction"] = summary["n_events"] / summary["denominator"]
    summary.to_csv(args.review_dir / "event_stage_summary.csv", index=False)

    target = events[
        (events["review_label"] == "neural") & (events["status"] == "unmatched")
    ]
    counts = target["sort_stage_outcome"].value_counts().to_dict()
    result = {
        "population": "blinded-review neural events originally unmatched to curated spikes",
        "n_events": int(len(target)),
        "time_tolerance_ms": args.time_tolerance_ms,
        "depth_tolerance_um": args.depth_tolerance_um,
        "outcomes": {str(key): int(value) for key, value in counts.items()},
        "interpretation": {
            "lost_during_curation": "detected by raw Kilosort output but absent after curation",
            "absent_from_uncurated_kilosort": "failure occurred by Kilosort detection/template assignment or earlier",
        },
        "paths": {stage: str(path) for stage, path in stage_paths.items()},
    }
    (args.review_dir / "event_stage_result.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(summary.to_string(index=False))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
