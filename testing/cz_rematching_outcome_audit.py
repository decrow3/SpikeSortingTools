#!/usr/bin/env python3
"""Independent saved-output accounting for CY fixed-bank rematching arms."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


FS = 29_999.759166666667
ORIGIN = 26_999_783
WINDOW_START_S = 900.0
WINDOW_END_S = 1240.0
WINDOW_SAMPLES = 10_200_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def merge(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out: list[list[float]] = []
    for left, right in sorted(intervals):
        if out and left <= out[-1][1]:
            out[-1][1] = max(out[-1][1], right)
        else:
            out.append([left, right])
    return [(left, right) for left, right in out]


def accepted_segments(catalogue: Path) -> list[tuple[int, int]]:
    table = pd.read_csv(catalogue)
    table = table[
        (table.status == "accepted")
        & (table.end_s > WINDOW_START_S)
        & (table.start_s < WINDOW_END_S)
    ]
    intervals = merge(
        [
            (max(WINDOW_START_S, float(row.start_s)), min(WINDOW_END_S, float(row.end_s)))
            for row in table.itertuples()
        ]
    )
    return [(int(np.ceil(left * FS)), int(np.ceil(right * FS))) for left, right in intervals]


def summarize(path: Path, segments: list[tuple[int, int]]) -> dict:
    with np.load(path, allow_pickle=False) as z:
        times = np.asarray(z["times_samples"])
        labels = np.asarray(z["labels"])
        channels = np.asarray(z["channels"])
        sampling_frequency = float(z["sampling_frequency"])
        geom = np.asarray(z["geom"])
    assigned = labels >= 0
    source_frames = ORIGIN + times
    accepted = np.zeros(len(times), dtype=bool)
    segment_id = np.full(len(times), -1, dtype=np.int32)
    for sid, (left, right) in enumerate(segments):
        take = (source_frames >= left) & (source_frames < right)
        accepted[take] = True
        segment_id[take] = sid

    endpoints = {"8": 0, "9_29": 0, "30": 0}
    accepted_endpoints = {"8": 0, "9_29": 0, "30": 0}
    for unit in np.unique(labels[assigned]):
        rows = np.flatnonzero(labels == unit)
        order = rows[np.argsort(times[rows], kind="stable")]
        delta = np.diff(times[order])
        pair_accepted = (
            accepted[order[:-1]]
            & accepted[order[1:]]
            & (segment_id[order[:-1]] == segment_id[order[1:]])
        )
        for name, mask in (
            ("8", delta == 8),
            ("9_29", (delta >= 9) & (delta <= 29)),
            ("30", delta == 30),
        ):
            endpoints[name] += int(np.count_nonzero(mask))
            accepted_endpoints[name] += int(np.count_nonzero(mask & pair_accepted))

    return {
        "path": str(path),
        "sha256": sha256(path),
        "rows": int(len(times)),
        "assigned_rows": int(np.count_nonzero(assigned)),
        "noise_rows": int(np.count_nonzero(~assigned)),
        "assigned_plus_noise_closes": int(np.count_nonzero(assigned) + np.count_nonzero(~assigned)) == len(times),
        "assigned_units": int(np.unique(labels[assigned]).size),
        "label_min": int(labels.min()),
        "label_max": int(labels.max()),
        "sampling_frequency": sampling_frequency,
        "sampling_frequency_exact": sampling_frequency == FS,
        "geometry_shape": list(geom.shape),
        "channel_min": int(channels.min()),
        "channel_max": int(channels.max()),
        "local_time_min": int(times.min()),
        "local_time_max": int(times.max()),
        "local_times_in_window": bool(np.all((times >= 0) & (times < WINDOW_SAMPLES))),
        "source_frame_min": int(source_frames.min()),
        "source_frame_max": int(source_frames.max()),
        "source_clock_rule": "source_frame = 26999783 + local times_samples",
        "endpoints_all_states": endpoints,
        "endpoints_accepted_only": accepted_endpoints,
        "accepted_rows": int(np.count_nonzero(accepted)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalogue", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--rematch0", type=Path, required=True)
    parser.add_argument("--cd1", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    segments = accepted_segments(args.catalogue)
    arms = {
        "accepted_baseline": summarize(args.baseline, segments),
        "REMATCH0": summarize(args.rematch0, segments),
    }
    if args.cd1 is not None and args.cd1.exists():
        arms["CD1_FULL"] = summarize(args.cd1, segments)
    base_rows = arms["accepted_baseline"]["rows"]
    for value in arms.values():
        value["row_delta_vs_baseline"] = value["rows"] - base_rows
        value["row_fraction_vs_baseline"] = value["rows"] / base_rows
    result = {
        "status": "pass" if all(
            arm["assigned_plus_noise_closes"]
            and arm["sampling_frequency_exact"]
            and arm["local_times_in_window"]
            and arm["geometry_shape"] == [182, 2]
            and arm["channel_min"] >= 0
            and arm["channel_max"] < 182
            for arm in arms.values()
        ) else "fail",
        "accepted_segment_count": len(segments),
        "accepted_segments_source_frames": segments,
        "arms": arms,
        "interpretation_limits": [
            "new matching rows have no valid index join to baseline rows",
            "time candidates within +/-7 samples are not identity",
            "endpoint counts are descriptive and cannot establish biological identity",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
