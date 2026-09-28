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


def accepted_segments(
    catalogue: Path, start_frame: int, end_frame: int, sampling_frequency: float
) -> list[tuple[int, int]]:
    table = pd.read_csv(catalogue)
    starts = np.ceil(table.start_s.to_numpy(float) * sampling_frequency).astype(np.int64)
    ends = np.ceil(table.end_s.to_numpy(float) * sampling_frequency).astype(np.int64)
    table = table.assign(start_frame=starts, end_frame=ends)
    table = table[(table.status == "accepted") & (table.end_frame > start_frame) & (table.start_frame < end_frame)]
    intervals = merge(
        [
            (max(start_frame, int(row.start_frame)), min(end_frame, int(row.end_frame)))
            for row in table.itertuples()
        ]
    )
    return [(int(left), int(right)) for left, right in intervals]


def summarize(
    path: Path,
    segments: list[tuple[int, int]],
    start_frame: int,
    end_frame: int,
    metadata_sampling_frequency: float,
) -> dict:
    with np.load(path, allow_pickle=False) as z:
        times = np.asarray(z["times_samples"])
        labels = np.asarray(z["labels"])
        channels = np.asarray(z["channels"])
        sampling_frequency = float(z["sampling_frequency"])
        geom = np.asarray(z["geom"])
    assigned = labels >= 0
    source_frames = start_frame + times
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
        "sampling_frequency_exact": sampling_frequency == metadata_sampling_frequency == FS,
        "geometry_shape": list(geom.shape),
        "channel_min": int(channels.min()),
        "channel_max": int(channels.max()),
        "local_time_min": int(times.min()),
        "local_time_max": int(times.max()),
        "metadata_start_frame": start_frame,
        "metadata_end_frame": end_frame,
        "metadata_window_samples": end_frame - start_frame,
        "local_times_in_window": bool(np.all((times >= 0) & (times < end_frame - start_frame))),
        "source_frame_min": int(source_frames.min()),
        "source_frame_max": int(source_frames.max()),
        "source_clock_rule": f"source_frame = {start_frame} + local times_samples",
        "endpoints_all_states": endpoints,
        "endpoints_accepted_only": accepted_endpoints,
        "accepted_rows_total": int(np.count_nonzero(accepted)),
        "accepted_rows_assigned": int(np.count_nonzero(accepted & assigned)),
        "accepted_rows_noise": int(np.count_nonzero(accepted & ~assigned)),
        "accepted_assigned_plus_noise_closes": int(np.count_nonzero(accepted & assigned) + np.count_nonzero(accepted & ~assigned)) == int(np.count_nonzero(accepted)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalogue", type=Path, required=True)
    parser.add_argument("--window-manifest", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--rematch0", type=Path, required=True)
    parser.add_argument("--cd1", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metadata = json.loads(args.window_manifest.read_text())["window"]
    start_frame = int(metadata["start_frame"])
    end_frame = int(metadata["end_frame"])
    metadata_fs = float(metadata["sampling_frequency"])
    segments = accepted_segments(args.catalogue, start_frame, end_frame, metadata_fs)
    arms = {
        "accepted_baseline": summarize(args.baseline, segments, start_frame, end_frame, metadata_fs),
        "REMATCH0": summarize(args.rematch0, segments, start_frame, end_frame, metadata_fs),
    }
    if args.cd1 is not None and args.cd1.exists():
        arms["CD1_FULL"] = summarize(args.cd1, segments, start_frame, end_frame, metadata_fs)
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
        "window_metadata": {
            "start_frame": start_frame,
            "end_frame": end_frame,
            "window_samples": end_frame - start_frame,
            "sampling_frequency": metadata_fs,
            "local_time_origin_seconds": metadata.get("local_time_origin_seconds"),
            "split_channel_preprocessing": metadata.get("split_channel_preprocessing"),
            "source_channel_count": len(metadata.get("source_channel_ids", [])),
            "sort_channel_count": len(metadata.get("channel_ids", [])),
            "first_sort_channel": metadata.get("channel_ids", [None])[0],
            "last_sort_channel": metadata.get("channel_ids", [None])[-1],
            "probe": "imec1",
        },
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
