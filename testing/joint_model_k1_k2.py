#!/usr/bin/env python3
"""K1 CAR side-channel and K2 pulse-to-sorter lag primitives.

The K1 path is deliberately bound to the exact C4 prospective reader snapshot.
It reproduces that reader's private batch transform while retaining the median
that CAR removes.  K2 consumes only integer session-clock arrays and channel
IDs; cluster or candidate-unit labels are not part of its interface.
"""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping

import numpy as np
import torch


READER_SHA256 = "e36d3066c794a3caec99f0d647104aa30b8aa91d902fc16ea8700778fbe5115c"
PULSE_THRESHOLD_UV = -120.0
PULSE_REFRACTORY_MS = 3.0
LAG_SEARCH_HALF_WIDTH_MS = 1.5
K2_CHANNEL_IDS = np.asarray([251, 252], dtype=np.int64)


def sha256_path(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_exact_reader(path: str | Path) -> ModuleType:
    """Load only the card-bound reader bytes, failing closed on drift."""
    source = Path(path).resolve()
    actual = sha256_path(source)
    if actual != READER_SHA256:
        raise ValueError(f"reader SHA-256 mismatch: {actual}")
    spec = importlib.util.spec_from_file_location("joint_model_exact_c4_reader", source)
    if spec is None or spec.loader is None:
        raise ImportError("could not construct exact reader module spec")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _round_samples(milliseconds: float, sampling_frequency: float) -> int:
    if not np.isfinite(sampling_frequency) or sampling_frequency <= 0:
        raise ValueError("sampling_frequency must be finite and positive")
    return int(np.floor(milliseconds * sampling_frequency / 1000.0 + 0.5))


class CARSignalSideChannel:
    """Expose exact reader traces plus high-passed removed CAR median."""

    def __init__(self, recording: Any, reader_module: ModuleType) -> None:
        segments = getattr(recording, "_recording_segments", None)
        if not isinstance(segments, list) or len(segments) != 1:
            raise ValueError("expected one exact-reader recording segment")
        segment = segments[0]
        if not isinstance(segment, reader_module._ProspectiveAPrewhiteningSegment):
            raise TypeError("recording segment is not from the exact reader module")
        if not segment.do_car:
            raise ValueError("CAR side-channel requires do_car=true")
        self.recording = recording
        self.segment = segment
        self.reader_module = reader_module

    def _filter_batch_with_median(self, batch_index: int) -> tuple[np.ndarray, np.ndarray]:
        segment = self.segment
        values = torch.from_numpy(segment._padded_raw_batch(batch_index))
        values = values[segment.channel_map]
        if segment.invert_sign:
            values = values * -1
        if segment.temporal_centering:
            values = values - values.mean(1).unsqueeze(1)
        removed_median = torch.median(values, 0)[0]
        values = values - removed_median
        if segment.highpass_filter is not None:
            frequency_filter = self.reader_module._fft_highpass(
                segment.highpass_filter, values.shape[1]
            )
            values = torch.real(
                torch.fft.ifft(torch.fft.fft(values) * torch.conj(frequency_filter))
            )
            values = torch.fft.fftshift(values, dim=-1)
            removed_median = torch.real(
                torch.fft.ifft(
                    torch.fft.fft(removed_median) * torch.conj(frequency_filter)
                )
            )
            removed_median = torch.fft.fftshift(removed_median, dim=-1)
        artifact = segment.artifact_threshold < np.inf and torch.any(
            torch.abs(values) >= segment.artifact_threshold
        )
        if bool(artifact):
            values = torch.zeros_like(values)
            removed_median = torch.zeros_like(removed_median)
        core_length = min(
            segment.batch_size,
            segment.num_samples - batch_index * segment.batch_size,
        )
        core_slice = slice(segment.padding, segment.padding + core_length)
        traces = values[:, core_slice].numpy(force=True).T
        median = removed_median[core_slice].numpy(force=True)
        return traces.astype(np.float32, copy=False), median.astype(np.float32, copy=False)

    def get_traces_and_hp_median(
        self,
        start_frame: int | None = None,
        end_frame: int | None = None,
        channel_indices: Any = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        segment = self.segment
        start = 0 if start_frame is None else int(start_frame)
        end = segment.num_samples if end_frame is None else int(end_frame)
        if start < 0 or end < start or end > segment.num_samples:
            raise IndexError("sample request is outside the recording")
        selector = self.reader_module._normalize_channel_indices(
            channel_indices, len(segment.channel_map)
        )
        if start == end:
            width = np.empty((0, len(segment.channel_map)), dtype=np.float32)[:, selector].shape[1]
            return np.empty((0, width), dtype=np.float32), np.empty(0, dtype=np.float32)
        first_batch = start // segment.batch_size
        last_batch = (end - 1) // segment.batch_size
        trace_pieces: list[np.ndarray] = []
        median_pieces: list[np.ndarray] = []
        for batch_index in range(first_batch, last_batch + 1):
            batch_global_start = batch_index * segment.batch_size
            local_start = max(start, batch_global_start) - batch_global_start
            local_end = min(end, batch_global_start + segment.batch_size) - batch_global_start
            traces, median = self._filter_batch_with_median(batch_index)
            trace_pieces.append(traces[local_start:local_end, selector])
            median_pieces.append(median[local_start:local_end])
        result_traces = (
            trace_pieces[0]
            if len(trace_pieces) == 1
            else np.concatenate(trace_pieces, axis=0)
        )
        result_median = (
            median_pieces[0]
            if len(median_pieces) == 1
            else np.concatenate(median_pieces)
        )
        if result_traces.shape[0] != end - start or result_median.shape != (end - start,):
            raise RuntimeError("batch join omitted or duplicated samples")
        return (
            result_traces.astype(np.float32, copy=False),
            result_median.astype(np.float32, copy=False),
        )


def detect_frozen_pulses(
    hp_median_uv: Any,
    *,
    sampling_frequency: float,
    sample_origin: int = 0,
) -> np.ndarray:
    """Detect frozen HP(median) minima with deterministic refractory NMS."""
    signal = np.asarray(hp_median_uv)
    if signal.ndim != 1 or not np.issubdtype(signal.dtype, np.number):
        raise ValueError("hp_median_uv must be a numeric one-dimensional array")
    signal = signal.astype(np.float64, copy=False)
    if not np.isfinite(signal).all():
        raise ValueError("hp_median_uv must be finite")
    if not isinstance(sample_origin, (int, np.integer)):
        raise ValueError("sample_origin must be an integer acquisition sample")
    if signal.size < 3:
        return np.empty(0, dtype=np.int64)
    candidates = np.flatnonzero(
        (signal[1:-1] < PULSE_THRESHOLD_UV)
        & (signal[1:-1] < signal[:-2])
        & (signal[1:-1] <= signal[2:])
    ) + 1
    refractory = _round_samples(PULSE_REFRACTORY_MS, sampling_frequency)
    if refractory < 1:
        raise ValueError("sampling_frequency makes the refractory interval empty")
    accepted: list[int] = []
    for candidate in sorted(
        map(int, candidates), key=lambda index: (float(signal[index]), index)
    ):
        if all(abs(candidate - prior) >= refractory for prior in accepted):
            accepted.append(candidate)
    return np.asarray(sorted(index + int(sample_origin) for index in accepted), dtype=np.int64)


def _integer_1d(name: str, values: Any, length: int | None = None) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or (length is not None and array.size != length):
        raise ValueError(f"{name} must be a one-dimensional companion array")
    if not np.issubdtype(array.dtype, np.integer):
        if not np.isfinite(array).all() or not np.equal(array, np.rint(array)).all():
            raise ValueError(f"{name} must contain exact integer values")
    return array.astype(np.int64, copy=False)


def freeze_pulse_sorter_lag(
    pulse_samples: Any,
    detection_samples: Any,
    detection_channel_ids: Any,
    *,
    sampling_frequency: float,
    clock_identity: str,
) -> Mapping[str, Any]:
    """Return the frozen K2 per-sample CCG peak and contiguous half-max width."""
    pulses = _integer_1d("pulse_samples", pulse_samples)
    detections = _integer_1d("detection_samples", detection_samples)
    channels = _integer_1d("detection_channel_ids", detection_channel_ids, detections.size)
    if not isinstance(clock_identity, str) or not clock_identity:
        raise ValueError("a nonempty session clock_identity is required")
    if pulses.size == 0:
        raise ValueError("at least one pulse is required")
    selected = np.isin(channels, K2_CHANNEL_IDS)
    focus = np.sort(detections[selected], kind="stable")
    if focus.size == 0:
        raise ValueError("no detections on channel IDs 251/252")
    half_width = _round_samples(LAG_SEARCH_HALF_WIDTH_MS, sampling_frequency)
    lags = np.arange(-half_width, half_width + 1, dtype=np.int64)
    counts = np.zeros(lags.size, dtype=np.int64)
    for pulse in pulses:
        left = np.searchsorted(focus, pulse - half_width, side="left")
        right = np.searchsorted(focus, pulse + half_width, side="right")
        pair_lags = focus[left:right] - pulse
        if pair_lags.size:
            np.add.at(counts, pair_lags + half_width, 1)
    peak_count = int(counts.max(initial=0))
    if peak_count == 0:
        raise ValueError("no pulse/detection pairs within frozen lag support")
    peak_candidates = np.flatnonzero(counts == peak_count)
    peak_index = min(
        map(int, peak_candidates),
        key=lambda index: (abs(int(lags[index])), int(lags[index])),
    )
    half_count = (peak_count + 1) // 2
    left_index = peak_index
    while left_index > 0 and counts[left_index - 1] >= half_count:
        left_index -= 1
    right_index = peak_index
    while right_index + 1 < counts.size and counts[right_index + 1] >= half_count:
        right_index += 1
    return {
        "schema": "joint-model-k2-pulse-sorter-lag-v1",
        "clock_identity": clock_identity,
        "sampling_frequency_hz": float(sampling_frequency),
        "lag_convention": "detection_sample_minus_pulse_sample",
        "focus_channel_ids": [251, 252],
        "search_half_width_samples_inclusive": half_width,
        "search_half_width_ms_frozen": LAG_SEARCH_HALF_WIDTH_MS,
        "n_pulses": int(pulses.size),
        "n_focus_detections": int(focus.size),
        "n_pairs_in_support": int(counts.sum()),
        "peak_lag_samples": int(lags[peak_index]),
        "peak_lag_ms": float(1000.0 * lags[peak_index] / sampling_frequency),
        "peak_count": peak_count,
        "half_max_min_count": half_count,
        "half_max_left_lag_samples": int(lags[left_index]),
        "half_max_right_lag_samples": int(lags[right_index]),
        "half_max_width_sample_bins": int(right_index - left_index + 1),
        "lags_samples": lags.tolist(),
        "counts": counts.tolist(),
    }
