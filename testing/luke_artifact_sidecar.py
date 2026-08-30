"""Build a sparse raw-threshold artifact sidecar for the Luke core strip.

The selected baseline clips phase-corrected samples outside +/-500 uV before
bad-channel interpolation.  This sidecar preserves the locations and original
stored-count values of those samples without changing the materialized voltage.
It also stores unique affected sample indices excluding physical channel 191,
which is synthetic in the sorter input and excluded from claim metrics.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from testing import luke_two_axis_pilot as pilot


CHANNELS = np.arange(176, 272, dtype=int)
BAD_CHANNEL = 191
THRESHOLD_UV = 500.0
GAIN_UV_PER_COUNT = 2.34375
THRESHOLD_COUNTS = THRESHOLD_UV / GAIN_UV_PER_COUNT
OUTPUT = Path(
    "/media/huklab/Data/NPX/Ryansorting/Luke/"
    "Luke0804_two_axis_pilot_imec1/artifact_sidecars/"
    "core_depth_strip_500uv.h5"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--chunk-duration-s", type=float, default=10.0)
    parser.add_argument("--start-s", type=float, default=0.0)
    parser.add_argument("--duration-s", type=float)
    return parser.parse_args()


def threshold_points(
    traces: np.ndarray,
    start_frame: int,
    channel_ids: np.ndarray = CHANNELS,
    threshold_counts: float = THRESHOLD_COUNTS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return sparse threshold points and claim-active samples for one chunk."""
    values = np.asarray(traces)
    ids = np.asarray(channel_ids, dtype=int)
    if values.ndim != 2 or values.shape[1] != len(ids):
        raise ValueError("traces and channel_ids have incompatible shapes")
    rows, columns = np.nonzero(np.abs(values.astype(np.float32)) > threshold_counts)
    samples = rows.astype(np.int64) + int(start_frame)
    channels = ids[columns].astype(np.int16)
    point_values = values[rows, columns].astype(np.int16, copy=False)
    claim_samples = np.unique(samples[channels != BAD_CHANNEL])
    return samples, channels, point_values, claim_samples


def append_dataset(dataset: h5py.Dataset, values: np.ndarray) -> None:
    if values.size == 0:
        return
    old_size = dataset.shape[0]
    dataset.resize((old_size + values.size,))
    dataset[old_size:] = values


def create_dataset(handle: h5py.File, name: str, dtype: str) -> h5py.Dataset:
    return handle.create_dataset(
        name,
        shape=(0,),
        maxshape=(None,),
        dtype=dtype,
        chunks=(262_144,),
        compression="gzip",
        compression_opts=4,
        shuffle=True,
    )


def build_sidecar(
    output: Path,
    chunk_duration_s: float,
    start_s: float = 0.0,
    duration_s: float | None = None,
) -> dict:
    if chunk_duration_s <= 0 or start_s < 0 or (duration_s is not None and duration_s <= 0):
        raise ValueError("chunk duration and requested frame range must be positive")
    recording = pilot.load_source_recording(CHANNELS, "conditioning_v2")
    fs = float(recording.get_sampling_frequency())
    recording_frames = int(recording.get_num_samples())
    start_frame = int(round(start_s * fs))
    stop_frame = (
        recording_frames
        if duration_s is None
        else min(recording_frames, start_frame + int(round(duration_s * fs)))
    )
    if start_frame >= stop_frame:
        raise ValueError("requested frame range is empty")
    chunk_frames = int(round(chunk_duration_s * fs))
    partial = output.with_suffix(output.suffix + ".partial")
    if output.exists() or partial.exists():
        raise RuntimeError(f"Sidecar target already exists: {output} or {partial}")
    output.parent.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    started = time.perf_counter()
    with h5py.File(partial, "w") as handle:
        sample_ds = create_dataset(handle, "sample_index", "<i8")
        channel_ds = create_dataset(handle, "channel_id", "<i2")
        value_ds = create_dataset(handle, "value_counts", "<i2")
        claim_ds = create_dataset(handle, "claim_active_sample_index", "<i8")
        handle.attrs.update(
            {
                "complete": False,
                "source_stage": "phase_shift_before_500uv_clipping",
                "threshold_uv": THRESHOLD_UV,
                "threshold_counts": THRESHOLD_COUNTS,
                "comparison": "abs(value_counts) > threshold_counts",
                "gain_uv_per_count": GAIN_UV_PER_COUNT,
                "physical_bad_channel": BAD_CHANNEL,
                "claim_active_samples_exclude_bad_channel": True,
                "sampling_frequency_hz": fs,
                "start_frame": start_frame,
                "stop_frame": stop_frame,
                "channel_first": int(CHANNELS[0]),
                "channel_last": int(CHANNELS[-1]),
            }
        )
        for chunk_index, chunk_start in enumerate(
            range(start_frame, stop_frame, chunk_frames)
        ):
            chunk_stop = min(stop_frame, chunk_start + chunk_frames)
            traces = recording.get_traces(
                start_frame=chunk_start,
                end_frame=chunk_stop,
                return_scaled=False,
            )
            samples, channels, values, claim_samples = threshold_points(
                traces, chunk_start
            )
            append_dataset(sample_ds, samples)
            append_dataset(channel_ds, channels)
            append_dataset(value_ds, values)
            append_dataset(claim_ds, claim_samples)
            summary_rows.append(
                {
                    "chunk_index": chunk_index,
                    "start_frame": chunk_start,
                    "stop_frame": chunk_stop,
                    "point_count": int(samples.size),
                    "claim_active_sample_count": int(claim_samples.size),
                    "positive_point_count": int(np.sum(values > 0)),
                    "negative_point_count": int(np.sum(values < 0)),
                    "active_channel_count": int(np.unique(channels).size),
                }
            )
            if chunk_index % 25 == 0:
                handle.flush()
                print(
                    f"chunk {chunk_index + 1}: frames {chunk_start}:{chunk_stop}; "
                    f"points={sample_ds.shape[0]}",
                    flush=True,
                )
        handle.attrs["complete"] = True
        handle.attrs["point_count"] = sample_ds.shape[0]
        handle.attrs["claim_active_sample_count"] = claim_ds.shape[0]
        handle.flush()
    os.replace(partial, output)
    summaries = pd.DataFrame(summary_rows)
    summary_csv = output.with_suffix(".chunks.csv")
    summaries.to_csv(summary_csv, index=False)
    result = {
        "output": str(output),
        "chunk_summary": str(summary_csv),
        "complete": True,
        "start_frame": start_frame,
        "stop_frame": stop_frame,
        "duration_s": (stop_frame - start_frame) / fs,
        "channel_ids": CHANNELS.tolist(),
        "threshold_uv": THRESHOLD_UV,
        "threshold_counts": THRESHOLD_COUNTS,
        "gain_uv_per_count": GAIN_UV_PER_COUNT,
        "point_count": int(summaries.point_count.sum()),
        "claim_active_sample_count": int(summaries.claim_active_sample_count.sum()),
        "positive_point_count": int(summaries.positive_point_count.sum()),
        "negative_point_count": int(summaries.negative_point_count.sum()),
        "elapsed_s": time.perf_counter() - started,
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    return result


def main() -> None:
    args = parse_args()
    result = build_sidecar(
        args.output, args.chunk_duration_s, args.start_s, args.duration_s
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
