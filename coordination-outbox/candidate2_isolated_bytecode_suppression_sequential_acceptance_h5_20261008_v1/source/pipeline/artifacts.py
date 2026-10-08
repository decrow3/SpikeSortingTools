"""Sparse raw-threshold sidecar used to exclude blanker-proximal claims."""

from __future__ import annotations

import json
import os
from pathlib import Path
from datetime import datetime, timezone
import hashlib
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


def _request_digest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _sidecar_result(output: Path, expected_digest: str) -> dict[str, Any]:
    import h5py

    with h5py.File(output, "r") as handle:
        observed_digest = str(handle.attrs.get("request_digest", ""))
        if not bool(handle.attrs.get("complete", False)):
            raise RuntimeError(f"Artifact sidecar is incomplete: {output}")
        if observed_digest != expected_digest:
            raise RuntimeError(
                "Existing artifact sidecar belongs to another recording/configuration"
            )
        return {
            "output": str(output),
            "complete": True,
            "reused": True,
            "request_digest": observed_digest,
            "threshold_uv": float(handle.attrs["threshold_uv"]),
            "threshold_counts": float(handle.attrs["threshold_counts"]),
            "n_jobs": int(handle.attrs["n_jobs"]),
            "chunk_duration": str(handle.attrs["chunk_duration"]),
            "excluded_channel_ids": json.loads(
                handle.attrs["excluded_channel_ids_json"]
            ),
            "point_count": int(handle.attrs["point_count"]),
            "claim_active_sample_count": int(
                handle.attrs["claim_active_sample_count"]
            ),
        }


def _init_threshold_worker(
    recording,
    channel_ids: np.ndarray,
    threshold_counts: float,
    excluded_channel_ids: list[Any],
) -> dict[str, Any]:
    return {
        "recording": recording,
        "channel_ids": channel_ids,
        "threshold_counts": threshold_counts,
        "excluded_channel_ids": excluded_channel_ids,
        "id_to_index": {
            str(value): index for index, value in enumerate(channel_ids)
        },
    }


def _threshold_chunk(
    segment_index: int,
    start_frame: int,
    end_frame: int,
    worker_ctx: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    recording = worker_ctx["recording"]
    traces = recording.get_traces(
        segment_index=segment_index,
        start_frame=start_frame,
        end_frame=end_frame,
        return_scaled=False,
    )
    samples, channels, values, claim_samples = threshold_points(
        traces,
        start_frame,
        worker_ctx["channel_ids"],
        worker_ctx["threshold_counts"],
        worker_ctx["excluded_channel_ids"],
    )
    channel_indices = np.asarray(
        [worker_ctx["id_to_index"][str(value)] for value in channels],
        dtype=np.int32,
    )
    return samples, channel_indices, values, claim_samples


def write_artifact_sidecar(
    phase_corrected_recording,
    output: Path,
    *,
    threshold_uv: float,
    excluded_channel_ids: Iterable[Any],
    chunk_duration_s: float | str = 10.0,
    n_jobs: int = 1,
    progress_bar: bool = True,
    source_identity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Scan phase-corrected voltage without changing sorter input.

    A complete exact sidecar is reused.  A complete ``.partial`` is recovered;
    an incomplete partial is archived before a clean restart so interrupted
    multi-hour scans never need manual deletion.
    """
    import h5py
    from spikeinterface.core.job_tools import ChunkRecordingExecutor

    output = Path(output)
    partial = output.with_suffix(output.suffix + ".partial")
    gains = np.unique(phase_corrected_recording.get_property("gain_to_uV"))
    if gains.size != 1:
        raise ValueError(f"Expected one gain_to_uV value, got {gains}")
    gain = float(gains[0])
    threshold_counts = threshold_uv / gain
    fs = float(phase_corrected_recording.get_sampling_frequency())
    channel_ids = np.asarray(phase_corrected_recording.get_channel_ids())
    excluded = list(excluded_channel_ids)
    request = {
        "source_stage": "phase_corrected_raw_before_500uv_blanking",
        "threshold_uv": float(threshold_uv),
        "gain_uv_per_count": gain,
        "sampling_frequency_hz": fs,
        "num_samples": int(phase_corrected_recording.get_num_samples()),
        "channel_ids": [str(value) for value in channel_ids],
        "excluded_channel_ids": [str(value) for value in excluded],
        "source_identity": source_identity,
    }
    request_digest = _request_digest(request)
    if output.exists():
        return _sidecar_result(output, request_digest)
    if partial.exists():
        import h5py

        try:
            with h5py.File(partial, "r") as handle:
                partial_complete = bool(handle.attrs.get("complete", False))
                partial_digest = str(handle.attrs.get("request_digest", ""))
        except OSError:
            partial_complete = False
            partial_digest = ""
        if partial_complete and partial_digest == request_digest:
            os.replace(partial, output)
            return _sidecar_result(output, request_digest)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        archived = partial.with_name(partial.name + f".interrupted-{timestamp}")
        os.replace(partial, archived)
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
                "n_jobs": int(n_jobs),
                "chunk_duration": str(chunk_duration_s),
                "request_digest": request_digest,
                "request_json": json.dumps(request, sort_keys=True),
                "num_samples": int(phase_corrected_recording.get_num_samples()),
                "channel_ids_json": json.dumps([str(value) for value in channel_ids]),
                "excluded_channel_ids_json": json.dumps(
                    [str(value) for value in excluded]
                ),
            }
        )
        def gather(result) -> None:
            samples, channel_indices, values, claim_samples = result
            _append(datasets["sample_index"], samples)
            _append(datasets["channel_index"], channel_indices)
            _append(datasets["value_counts"], values)
            _append(datasets["claim_active_sample_index"], claim_samples)
            totals["point_count"] += int(samples.size)
            totals["claim_active_sample_count"] += int(claim_samples.size)

        executor = ChunkRecordingExecutor(
            phase_corrected_recording,
            _threshold_chunk,
            _init_threshold_worker,
            (phase_corrected_recording, channel_ids, threshold_counts, excluded),
            gather_func=gather,
            pool_engine="process",
            n_jobs=n_jobs,
            chunk_duration=chunk_duration_s,
            progress_bar=progress_bar,
            verbose=True,
            job_name="write_artifact_sidecar",
        )
        executor.run()
        handle.attrs["complete"] = True
        for key, value in totals.items():
            handle.attrs[key] = value
    os.replace(partial, output)
    result = {
        "output": str(output),
        "complete": True,
        "reused": False,
        "request_digest": request_digest,
        "threshold_uv": threshold_uv,
        "threshold_counts": threshold_counts,
        "n_jobs": int(n_jobs),
        "chunk_duration": str(chunk_duration_s),
        "excluded_channel_ids": [str(value) for value in excluded],
        **totals,
    }
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    return result
