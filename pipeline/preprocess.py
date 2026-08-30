"""The tested rescue conditioning graph and guarded materialization.

Sorter input stops after phase correction, 500-uV bilateral sample blanking,
and bad-channel interpolation.  There is deliberately no external AP filter,
common reference, whitening, or voltage motion correction here.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.signal import medfilt, welch

from .config import PIPELINE_VERSION, RescueConfig, fingerprint


MANIFEST_NAME = "rescue_recording_manifest.json"


def _single_gain_uv_per_count(recording) -> float:
    gains = np.unique(np.asarray(recording.get_property("gain_to_uV"), dtype=float))
    if gains.size != 1 or not np.isfinite(gains[0]) or gains[0] <= 0:
        raise ValueError(f"Expected one positive gain_to_uV value, got {gains}")
    return float(gains[0])


def phase_correct(recording):
    """Apply Neuropixels phase correction only when shifts are present."""
    shifts = recording.get_property("inter_sample_shift")
    if shifts is None or not np.any(shifts):
        return recording
    from spikeinterface.preprocessing import phase_shift

    return phase_shift(recording)


def _channel_metrics(
    recording,
    *,
    gain_uv_per_count: float,
    n_batches: int,
    batch_duration_s: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce the channel metrics used to select the tested baseline."""
    from spikeinterface.preprocessing import highpass_filter

    highpassed = highpass_filter(recording, freq_min=300, direction="forward-backward")
    fs = float(highpassed.get_sampling_frequency())
    batch_size = int(round(batch_duration_s * fs))
    starts = np.arange(0, highpassed.get_num_samples() - batch_size + 1, batch_size)
    if starts.size == 0:
        raise ValueError("Recording is too short for one channel-metric batch")
    count = min(int(n_batches), int(starts.size))
    selected = np.random.RandomState(seed).choice(starts, count, replace=False)
    similarity = np.zeros(highpassed.get_num_channels(), dtype=float)
    noise = np.zeros_like(similarity)
    f_threshold = 0.8 * fs / 2
    for start in selected:
        traces = highpassed.get_traces(
            start_frame=int(start), end_frame=int(start + batch_size)
        ).astype(np.float64)
        traces *= gain_uv_per_count
        median = np.median(traces, axis=1)
        median_energy = float(np.sum(median**2))
        if median_energy == 0:
            raise ValueError("Channel-metric batch has zero median energy")
        correlation = np.sum(traces * median[:, None], axis=0) / median_energy
        similarity += correlation - medfilt(correlation, 11)
        frequencies, psd = welch(traces, fs=fs, nperseg=2048, axis=0)
        noise += np.mean(psd[frequencies > f_threshold], axis=0)
    return similarity / count, noise / count


def select_bad_channel_ids(
    channel_ids: Iterable[Any],
    similarity: np.ndarray,
    noise: np.ndarray,
    *,
    similarity_threshold: float,
    noise_threshold: float,
) -> list[Any]:
    """Select bad channel IDs from the frozen legacy metrics."""
    ids = np.asarray(list(channel_ids))
    similarity = np.asarray(similarity)
    noise = np.asarray(noise)
    if similarity.shape != ids.shape or noise.shape != ids.shape:
        raise ValueError("Channel IDs, similarity, and noise must have equal shape")
    bad = (similarity < similarity_threshold) | (noise > noise_threshold)
    return ids[bad].tolist()


def _metric_request(recording, config: RescueConfig) -> dict[str, Any]:
    return {
        "pipeline_version": PIPELINE_VERSION,
        "config_digest": config.digest,
        "num_samples": int(recording.get_num_samples()),
        "num_channels": int(recording.get_num_channels()),
        "sampling_frequency_hz": float(recording.get_sampling_frequency()),
        "channel_ids": [str(value) for value in recording.get_channel_ids()],
    }


def _load_or_compute_metrics(
    recording,
    cache_dir: Path,
    config: RescueConfig,
    *,
    recompute: bool,
) -> tuple[np.ndarray, np.ndarray]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    values_path = cache_dir / "channel_metrics.npz"
    manifest_path = cache_dir / "channel_metrics_manifest.json"
    request = _metric_request(recording, config)
    request_digest = fingerprint(request)
    if values_path.exists() or manifest_path.exists():
        if not values_path.exists() or not manifest_path.exists():
            raise RuntimeError(f"Incomplete channel-metric cache in {cache_dir}")
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("request_digest") != request_digest:
            if not recompute:
                raise RuntimeError(
                    "Channel-metric cache does not match this source/configuration; "
                    "use --recompute-channel-metrics or a new output directory"
                )
        elif not recompute:
            values = np.load(values_path)
            return values["similarity"], values["noise"]
    gain = _single_gain_uv_per_count(recording)
    similarity, noise = _channel_metrics(
        recording,
        gain_uv_per_count=gain,
        n_batches=config.channel_metric_batches,
        batch_duration_s=config.channel_metric_batch_duration_s,
        seed=config.channel_metric_seed,
    )
    np.savez(values_path, similarity=similarity, noise=noise)
    manifest_path.write_text(
        json.dumps({**request, "request_digest": request_digest}, indent=2) + "\n"
    )
    return similarity, noise


def build_rescue_recording(
    raw_recording,
    *,
    cache_dir: Path,
    config: RescueConfig,
    bad_channel_ids: Iterable[Any] | None = None,
    recompute_channel_metrics: bool = False,
):
    """Return the lazy rescue recording and a preprocessing receipt."""
    from spikeinterface.preprocessing import blank_staturation, interpolate_bad_channels

    gain = _single_gain_uv_per_count(raw_recording)
    shifted = phase_correct(raw_recording)
    threshold_counts = config.saturation_threshold_uv / gain
    blanked = blank_staturation(shifted, threshold_counts, direction="both")
    if bad_channel_ids is None:
        similarity, noise = _load_or_compute_metrics(
            blanked,
            Path(cache_dir) / "channel_metrics",
            config,
            recompute=recompute_channel_metrics,
        )
        bad_ids = select_bad_channel_ids(
            raw_recording.get_channel_ids(),
            similarity,
            noise,
            similarity_threshold=config.similarity_threshold,
            noise_threshold=config.noise_threshold,
        )
        bad_source = "frozen_similarity_and_noise_metrics"
    else:
        available = set(raw_recording.get_channel_ids().tolist())
        bad_ids = list(bad_channel_ids)
        missing = [value for value in bad_ids if value not in available]
        if missing:
            raise ValueError(f"Explicit bad channels are absent: {missing}")
        bad_source = "explicit"
    conditioned = interpolate_bad_channels(blanked, bad_channel_ids=bad_ids)
    if np.dtype(conditioned.dtype) != np.dtype("int16"):
        raise TypeError(
            f"Expected int16 production quantization after interpolation, got {conditioned.dtype}"
        )
    receipt = {
        "pipeline_version": PIPELINE_VERSION,
        "config": config.as_dict(),
        "graph": [
            "neuropixels_phase_correction_if_present",
            "samplewise_bilateral_blanking_500uv",
            "bad_channel_interpolation",
        ],
        "external_filter": None,
        "external_reference": None,
        "external_voltage_motion_correction": False,
        "gain_uv_per_count": gain,
        "threshold_counts": threshold_counts,
        "bad_channel_ids": [str(value) for value in bad_ids],
        "bad_channel_source": bad_source,
        "dtype": str(conditioned.dtype),
        "num_samples": int(conditioned.get_num_samples()),
        "num_channels": int(conditioned.get_num_channels()),
        "sampling_frequency_hz": float(conditioned.get_sampling_frequency()),
    }
    return conditioned, receipt


def _materialization_request(
    raw_recording,
    source_folder: Path,
    stream_id: str,
    config: RescueConfig,
    explicit_bad_channel_ids: Iterable[Any] | None,
    start_frame: int,
    end_frame: int,
) -> dict[str, Any]:
    return {
        "pipeline_version": PIPELINE_VERSION,
        "source_folder": str(Path(source_folder).resolve()),
        "stream_id": stream_id,
        "num_samples": int(raw_recording.get_num_samples()),
        "num_channels": int(raw_recording.get_num_channels()),
        "sampling_frequency_hz": float(raw_recording.get_sampling_frequency()),
        "dtype": str(raw_recording.dtype),
        "selected_start_frame": start_frame,
        "selected_end_frame": end_frame,
        "config": config.as_dict(),
        "explicit_bad_channel_ids": (
            None
            if explicit_bad_channel_ids is None
            else [str(value) for value in explicit_bad_channel_ids]
        ),
    }


def _validate_materialized_recording(folder: Path, expected_recording) -> dict[str, Any]:
    """Run cheap structural and population checks before accepting a cache."""
    from spikeinterface.core import load_extractor

    loaded = load_extractor(folder)
    expected_shape = (
        int(expected_recording.get_num_samples()),
        int(expected_recording.get_num_channels()),
    )
    actual_shape = (int(loaded.get_num_samples()), int(loaded.get_num_channels()))
    if actual_shape != expected_shape:
        raise RuntimeError(f"Materialized shape {actual_shape} != expected {expected_shape}")
    if np.dtype(loaded.dtype) != np.dtype(expected_recording.dtype):
        raise RuntimeError(f"Materialized dtype {loaded.dtype} != {expected_recording.dtype}")
    expected_bytes = int(np.prod(expected_shape) * np.dtype(loaded.dtype).itemsize)
    binaries = list(folder.glob("*.raw")) + list(folder.glob("*.bin"))
    actual_bytes = sum(path.stat().st_size for path in binaries)
    if actual_bytes != expected_bytes:
        raise RuntimeError(f"Materialized bytes {actual_bytes} != expected {expected_bytes}")
    sample_frames = min(int(round(loaded.get_sampling_frequency())), actual_shape[0])
    starts = sorted(
        {
            min(actual_shape[0] - sample_frames, int(fraction * actual_shape[0]))
            for fraction in (0.0, 0.25, 0.5, 0.75, 0.99)
        }
    )
    sampled = []
    channel_min_std = np.inf
    for start in starts:
        traces = loaded.get_traces(start_frame=start, end_frame=start + sample_frames)
        standard_deviation = np.std(traces.astype(np.float32), axis=0)
        channel_min_std = min(channel_min_std, float(np.min(standard_deviation)))
        sampled.append(
            {
                "start_frame": start,
                "nonzero_fraction": float(np.mean(traces != 0)),
                "minimum_counts": int(np.min(traces)),
                "maximum_counts": int(np.max(traces)),
            }
        )
    if not sampled or any(row["nonzero_fraction"] == 0 for row in sampled):
        raise RuntimeError("Materialized recording contains an empty validation chunk")
    if channel_min_std == 0:
        raise RuntimeError("Materialized recording contains a stuck channel")
    return {
        "actual_binary_bytes": actual_bytes,
        "structural_integrity_passed": True,
        "minimum_sampled_channel_std_counts": channel_min_std,
        "sampled_chunks": sampled,
    }


def materialize_rescue_recording(
    raw_recording,
    output_dir: Path,
    *,
    source_folder: Path,
    stream_id: str,
    config: RescueConfig,
    bad_channel_ids: Iterable[Any] | None = None,
    recompute_channel_metrics: bool = False,
    start_frame: int = 0,
    end_frame: int | None = None,
):
    """Materialize once, reusing only a cache with an exact request fingerprint."""
    from spikeinterface.core import load_extractor

    output_dir = Path(output_dir)
    partial = output_dir.with_name(output_dir.name + ".partial")
    manifest_path = output_dir / MANIFEST_NAME
    source_frames = int(raw_recording.get_num_samples())
    if end_frame is None:
        end_frame = source_frames
    if start_frame < 0 or end_frame > source_frames or start_frame >= end_frame:
        raise ValueError("Invalid materialization frame range")
    request = _materialization_request(
        raw_recording,
        source_folder,
        stream_id,
        config,
        bad_channel_ids,
        int(start_frame),
        int(end_frame),
    )
    request_digest = fingerprint(request)
    if partial.exists():
        raise RuntimeError(f"Incomplete materialization requires inspection: {partial}")
    if output_dir.exists():
        if not manifest_path.exists():
            raise RuntimeError(f"Existing recording lacks {MANIFEST_NAME}: {output_dir}")
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("request_digest") != request_digest:
            raise RuntimeError("Existing recording cache belongs to another request")
        return load_extractor(output_dir), manifest
    conditioned, receipt = build_rescue_recording(
        raw_recording,
        cache_dir=output_dir.parent,
        config=config,
        bad_channel_ids=bad_channel_ids,
        recompute_channel_metrics=recompute_channel_metrics,
    )
    # Slice after phase correction so bounded smoke tests retain genuine source
    # voltage in the phase-shift margin instead of padding the requested edge.
    if start_frame != 0 or end_frame != source_frames:
        conditioned = conditioned.frame_slice(
            start_frame=int(start_frame), end_frame=int(end_frame)
        )
        receipt["selected_start_frame"] = int(start_frame)
        receipt["selected_end_frame"] = int(end_frame)
        receipt["num_samples"] = int(conditioned.get_num_samples())
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    conditioned.save(
        folder=partial,
        n_jobs=config.materialize_n_jobs,
        chunk_duration=config.materialize_chunk_duration,
        progress_bar=True,
    )
    integrity = _validate_materialized_recording(partial, conditioned)
    manifest = {
        **request,
        **receipt,
        "request_digest": request_digest,
        "expected_binary_bytes": int(
            conditioned.get_num_samples()
            * conditioned.get_num_channels()
            * np.dtype(conditioned.dtype).itemsize
        ),
        "integrity": integrity,
        "complete": True,
    }
    (partial / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n")
    os.replace(partial, output_dir)
    return load_extractor(output_dir), manifest
