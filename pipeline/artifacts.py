"""Sparse raw-threshold sidecar used to exclude blanker-proximal claims."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def threshold_points(
    traces: np.ndarray,
    start_frame: int,
    channel_ids: np.ndarray,
    threshold_counts: float,
    excluded_channel_ids: Iterable[Any] = (),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return threshold points plus unique claim-active sample indices."""
    values = np.asarray(traces)
    ids = np.asarray(channel_ids)
    if values.ndim != 2 or values.shape[1] != ids.size:
        raise ValueError("traces and channel_ids have incompatible shapes")
    rows, columns = np.nonzero(np.abs(values.astype(np.float32)) > threshold_counts)
    samples = rows.astype(np.int64) + int(start_frame)
    channels = ids[columns]
    point_values = values[rows, columns].astype(np.int16, copy=False)
    excluded = np.isin(channels, np.asarray(list(excluded_channel_ids)))
    claim_samples = np.unique(samples[~excluded])
    return samples, channels, point_values, claim_samples


def _append(dataset, values: np.ndarray) -> None:
    if values.size:
        old_size = dataset.shape[0]
        dataset.resize((old_size + values.size,))
        dataset[old_size:] = values


def write_artifact_sidecar(
    phase_corrected_recording,
    output: Path,
    *,
    threshold_uv: float,
    excluded_channel_ids: Iterable[Any],
    chunk_duration_s: float = 10.0,
) -> dict[str, Any]:
    """Scan a phase-corrected recording without changing sorter voltage."""
    import h5py

    output = Path(output)
    partial = output.with_suffix(output.suffix + ".partial")
    if output.exists() or partial.exists():
        raise RuntimeError(f"Sidecar target already exists: {output} or {partial}")
    gains = np.unique(phase_corrected_recording.get_property("gain_to_uV"))
    if gains.size != 1:
        raise ValueError(f"Expected one gain_to_uV value, got {gains}")
    gain = float(gains[0])
    threshold_counts = threshold_uv / gain
    fs = float(phase_corrected_recording.get_sampling_frequency())
    chunk_frames = int(round(chunk_duration_s * fs))
    if chunk_frames < 1:
        raise ValueError("chunk_duration_s is too small")
    channel_ids = np.asarray(phase_corrected_recording.get_channel_ids())
    id_to_index = {str(value): index for index, value in enumerate(channel_ids)}
    excluded = list(excluded_channel_ids)
    output.parent.mkdir(parents=True, exist_ok=True)
    totals = {"point_count": 0, "claim_active_sample_count": 0}
    with h5py.File(partial, "w") as handle:
        datasets = {
            "sample_index": handle.create_dataset(
                "sample_index", shape=(0,), maxshape=(None,), dtype="<i8", chunks=True
            ),
            "channel_index": handle.create_dataset(
                "channel_index", shape=(0,), maxshape=(None,), dtype="<i4", chunks=True
            ),
            "value_counts": handle.create_dataset(
                "value_counts", shape=(0,), maxshape=(None,), dtype="<i2", chunks=True
            ),
            "claim_active_sample_index": handle.create_dataset(
                "claim_active_sample_index",
                shape=(0,),
                maxshape=(None,),
                dtype="<i8",
                chunks=True,
            ),
        }
        handle.attrs.update(
            {
                "complete": False,
                "source_stage": "phase_corrected_raw_before_500uv_blanking",
                "threshold_uv": threshold_uv,
                "threshold_counts": threshold_counts,
                "gain_uv_per_count": gain,
                "sampling_frequency_hz": fs,
                "channel_ids_json": json.dumps([str(value) for value in channel_ids]),
                "excluded_channel_ids_json": json.dumps(
                    [str(value) for value in excluded]
                ),
            }
        )
        stop = int(phase_corrected_recording.get_num_samples())
        for start in range(0, stop, chunk_frames):
            end = min(stop, start + chunk_frames)
            traces = phase_corrected_recording.get_traces(
                start_frame=start, end_frame=end, return_scaled=False
            )
            samples, channels, values, claim_samples = threshold_points(
                traces, start, channel_ids, threshold_counts, excluded
            )
            channel_indices = np.asarray(
                [id_to_index[str(value)] for value in channels], dtype=np.int32
            )
            _append(datasets["sample_index"], samples)
            _append(datasets["channel_index"], channel_indices)
            _append(datasets["value_counts"], values)
            _append(datasets["claim_active_sample_index"], claim_samples)
            totals["point_count"] += int(samples.size)
            totals["claim_active_sample_count"] += int(claim_samples.size)
        handle.attrs["complete"] = True
        for key, value in totals.items():
            handle.attrs[key] = value
    os.replace(partial, output)
    result = {
        "output": str(output),
        "complete": True,
        "threshold_uv": threshold_uv,
        "threshold_counts": threshold_counts,
        "excluded_channel_ids": [str(value) for value in excluded],
        **totals,
    }
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    return result
