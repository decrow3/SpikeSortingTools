"""Prospective lazy SpikeInterface view of Kilosort's immediate-prewhitening batches.

This is new prospective source.  It does not claim identity with the missing
historical ``kilosort_prewhitening_si.py`` wrapper.  The transform is applied
to Kilosort's fixed, padded batch tensor and only the central recording-time
core is exposed.  Requests crossing batch joins are assembled in time order;
channel selection occurs only after full-probe centering and CAR.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from spikeinterface.core import BaseRecording, BaseRecordingSegment


def _fft_highpass(hp_filter: torch.Tensor, length: int) -> torch.Tensor:
    """Exact Kilosort 4.0.27 ``fft_highpass`` construction."""

    filter_length = int(hp_filter.shape[0])
    if filter_length < length:
        left = (length - filter_length) // 2
        right = length - left - filter_length
        padded = torch.cat(
            (
                torch.zeros(left, device=hp_filter.device),
                hp_filter,
                torch.zeros(right, device=hp_filter.device),
            )
        )
    elif filter_length > length:
        crop = (filter_length - length) // 2
        padded = hp_filter[crop : crop + length]
    else:
        padded = hp_filter
    return torch.fft.fft(padded)


def _normalize_channel_indices(channel_indices: Any, channel_count: int) -> Any:
    if channel_indices is None:
        return slice(None)
    if isinstance(channel_indices, slice):
        return channel_indices
    indices = np.asarray(channel_indices)
    if indices.dtype == bool:
        if indices.shape != (channel_count,):
            raise IndexError("boolean channel selector has wrong length")
        return indices
    indices = indices.astype(np.int64, copy=False)
    if indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= channel_count):
        raise IndexError("channel indices are not a valid one-dimensional selector")
    return indices


class _ProspectiveAPrewhiteningSegment(BaseRecordingSegment):
    def __init__(
        self,
        *,
        binary_path: str,
        num_samples: int,
        num_channels_raw: int,
        dtype: str,
        sampling_frequency: float,
        batch_size: int,
        padding: int,
        channel_map: np.ndarray,
        highpass_filter: np.ndarray | None,
        temporal_centering: bool,
        do_car: bool,
        invert_sign: bool,
        artifact_threshold: float,
        scale: float | None,
        shift: float | None,
    ) -> None:
        super().__init__(sampling_frequency=float(sampling_frequency), t_start=0.0)
        self.binary_path = str(binary_path)
        self.num_samples = int(num_samples)
        self.num_channels_raw = int(num_channels_raw)
        self.dtype = np.dtype(dtype)
        self.batch_size = int(batch_size)
        self.padding = int(padding)
        self.channel_map = np.asarray(channel_map, dtype=np.int64)
        self.highpass_filter = (
            None
            if highpass_filter is None
            else torch.as_tensor(np.asarray(highpass_filter, dtype=np.float32), device="cpu")
        )
        self.temporal_centering = bool(temporal_centering)
        self.do_car = bool(do_car)
        self.invert_sign = bool(invert_sign)
        self.artifact_threshold = float(artifact_threshold)
        self.scale = None if scale is None else float(scale)
        self.shift = None if shift is None else float(shift)
        self.num_batches = int(np.ceil(self.num_samples / self.batch_size))
        self.logical_batch_reads = 0
        self.logical_source_frames_read = 0
        self.logical_source_bytes_read = 0
        final_core = self.num_samples - (self.num_batches - 1) * self.batch_size
        if final_core < self.padding:
            raise ValueError("Kilosort would drop a terminal core shorter than padding")

    def get_num_samples(self) -> int:
        return self.num_samples

    def _read_frames(self, start: int, end: int) -> np.ndarray:
        expected_bytes = self.num_samples * self.num_channels_raw * self.dtype.itemsize
        if Path(self.binary_path).stat().st_size != expected_bytes:
            raise RuntimeError("binary size changed or does not match declared shape")
        frames = int(end - start)
        byte_count = frames * self.num_channels_raw * self.dtype.itemsize
        byte_offset = int(start) * self.num_channels_raw * self.dtype.itemsize
        buffer = bytearray(byte_count)
        view = memoryview(buffer)
        completed = 0
        with open(self.binary_path, "rb", buffering=0) as stream:
            while completed < byte_count:
                count = os.preadv(
                    stream.fileno(), [view[completed:]], byte_offset + completed
                )
                if count == 0:
                    raise EOFError("bounded Candidate3 read ended early")
                completed += count
            if hasattr(os, "posix_fadvise"):
                os.posix_fadvise(
                    stream.fileno(), byte_offset, byte_count, os.POSIX_FADV_DONTNEED
                )
        return np.frombuffer(buffer, dtype=self.dtype).reshape(
            frames, self.num_channels_raw
        )

    def _padded_raw_batch(self, batch_index: int) -> np.ndarray:
        if not 0 <= batch_index < self.num_batches:
            raise IndexError("batch index out of range")
        width = self.batch_size + 2 * self.padding
        padded = np.empty((self.num_channels_raw, width), dtype=np.float32)
        if batch_index == 0:
            raw_start = 0
            raw_end = min(self.num_samples, self.batch_size + self.padding)
            values = self._read_frames(raw_start, raw_end).T
            if self.scale is not None:
                values = values * self.scale
            if self.shift is not None:
                values = values + self.shift
            values = values.astype(np.float32, copy=False)
            padded[:, self.padding : self.padding + values.shape[1]] = values
            padded[:, : self.padding] = values[:, :1]
            filled = self.padding + values.shape[1]
            if filled < width:
                padded[:, filled:] = values[:, -1:]
        else:
            raw_start = batch_index * self.batch_size - self.padding
            raw_end = min(
                self.num_samples,
                raw_start + self.batch_size + 2 * self.padding,
            )
            values = self._read_frames(raw_start, raw_end).T
            if self.scale is not None:
                values = values * self.scale
            if self.shift is not None:
                values = values + self.shift
            values = values.astype(np.float32, copy=False)
            padded[:, : values.shape[1]] = values
            if values.shape[1] < width:
                padded[:, values.shape[1] :] = values[:, -1:]
        frames_read = int(raw_end - raw_start)
        self.logical_batch_reads += 1
        self.logical_source_frames_read += frames_read
        self.logical_source_bytes_read += (
            frames_read * self.num_channels_raw * self.dtype.itemsize
        )
        return padded

    def _filter_batch(self, batch_index: int) -> np.ndarray:
        values = torch.from_numpy(self._padded_raw_batch(batch_index))
        values = values[self.channel_map]
        if self.invert_sign:
            values = values * -1
        if self.temporal_centering:
            values = values - values.mean(1).unsqueeze(1)
        if self.do_car:
            values = values - torch.median(values, 0)[0]
        if self.highpass_filter is not None:
            frequency_filter = _fft_highpass(self.highpass_filter, values.shape[1])
            values = torch.real(
                torch.fft.ifft(torch.fft.fft(values) * torch.conj(frequency_filter))
            )
            values = torch.fft.fftshift(values, dim=-1)
        if self.artifact_threshold < np.inf and torch.any(
            torch.abs(values) >= self.artifact_threshold
        ):
            values = torch.zeros_like(values)
        core_length = min(
            self.batch_size,
            self.num_samples - batch_index * self.batch_size,
        )
        core = values[:, self.padding : self.padding + core_length]
        return core.numpy(force=True).T

    def get_traces(self, start_frame, end_frame, channel_indices):
        start = 0 if start_frame is None else int(start_frame)
        end = self.num_samples if end_frame is None else int(end_frame)
        if start < 0 or end < start or end > self.num_samples:
            raise IndexError("sample request is outside the recording")
        selector = _normalize_channel_indices(channel_indices, len(self.channel_map))
        if start == end:
            width = np.empty((0, len(self.channel_map)), dtype=np.float32)[:, selector].shape[1]
            return np.empty((0, width), dtype=np.float32)
        first_batch = start // self.batch_size
        last_batch = (end - 1) // self.batch_size
        pieces: list[np.ndarray] = []
        for batch_index in range(first_batch, last_batch + 1):
            batch_global_start = batch_index * self.batch_size
            local_start = max(start, batch_global_start) - batch_global_start
            local_end = min(end, batch_global_start + self.batch_size) - batch_global_start
            filtered = self._filter_batch(batch_index)
            pieces.append(filtered[local_start:local_end, selector])
        result = pieces[0] if len(pieces) == 1 else np.concatenate(pieces, axis=0)
        if result.shape[0] != end - start:
            raise RuntimeError("batch join omitted or duplicated samples")
        return result.astype(np.float32, copy=False)


class ProspectiveAPrewhiteningRecording(BaseRecording):
    """Lazy original-A batch representation immediately before whitening."""

    def __init__(
        self,
        *,
        binary_path: str | Path,
        num_samples: int,
        num_channels_raw: int,
        dtype: str,
        sampling_frequency: float,
        batch_size: int,
        padding: int,
        channel_map: Sequence[int],
        channel_ids: Sequence[Any],
        channel_locations: np.ndarray,
        highpass_filter: np.ndarray | None,
        temporal_centering: bool = True,
        do_car: bool = True,
        invert_sign: bool = False,
        artifact_threshold: float = np.inf,
        scale: float | None = None,
        shift: float | None = None,
    ) -> None:
        path = Path(binary_path).resolve()
        mapping = np.asarray(channel_map, dtype=np.int64)
        ids = np.asarray(channel_ids)
        locations = np.asarray(channel_locations, dtype=np.float64)
        if mapping.ndim != 1 or len(mapping) != len(ids):
            raise ValueError("channel map and output channel IDs must have the same length")
        if np.any(mapping < 0) or np.any(mapping >= int(num_channels_raw)):
            raise ValueError("channel map contains an invalid raw channel")
        if len(np.unique(mapping)) != len(mapping) or len(np.unique(ids)) != len(ids):
            raise ValueError("channel map and IDs must be unique")
        if locations.shape[0] != len(ids) or locations.ndim != 2:
            raise ValueError("channel locations do not match output channels")
        if int(batch_size) <= 0 or int(padding) < 0:
            raise ValueError("batch size must be positive and padding nonnegative")
        if int(num_samples) <= 0 or int(num_channels_raw) <= 0:
            raise ValueError("recording shape must be positive")
        super().__init__(
            sampling_frequency=float(sampling_frequency),
            channel_ids=ids,
            dtype=np.dtype("float32"),
        )
        self.set_channel_locations(locations)
        self.add_recording_segment(
            _ProspectiveAPrewhiteningSegment(
                binary_path=str(path),
                num_samples=int(num_samples),
                num_channels_raw=int(num_channels_raw),
                dtype=str(np.dtype(dtype)),
                sampling_frequency=float(sampling_frequency),
                batch_size=int(batch_size),
                padding=int(padding),
                channel_map=mapping,
                highpass_filter=highpass_filter,
                temporal_centering=bool(temporal_centering),
                do_car=bool(do_car),
                invert_sign=bool(invert_sign),
                artifact_threshold=float(artifact_threshold),
                scale=scale,
                shift=shift,
            )
        )
        self.annotate(
            is_filtered=True,
            candidate3_boundary="prospective_A_immediate_prewhitening_fixed_batches",
            candidate3_whitening_applied=False,
        )
        self._kwargs = {
            "binary_path": str(path),
            "num_samples": int(num_samples),
            "num_channels_raw": int(num_channels_raw),
            "dtype": str(np.dtype(dtype)),
            "sampling_frequency": float(sampling_frequency),
            "batch_size": int(batch_size),
            "padding": int(padding),
            "channel_map": mapping.tolist(),
            "channel_ids": ids.tolist(),
            "channel_locations": locations.tolist(),
            "highpass_filter": (
                None
                if highpass_filter is None
                else np.asarray(highpass_filter, dtype=np.float32).tolist()
            ),
            "temporal_centering": bool(temporal_centering),
            "do_car": bool(do_car),
            "invert_sign": bool(invert_sign),
            "artifact_threshold": float(artifact_threshold),
            "scale": scale,
            "shift": shift,
        }

    def get_candidate3_read_audit(self) -> dict[str, int]:
        if len(self._recording_segments) != 1:
            raise RuntimeError("Candidate3 adapter expects exactly one segment")
        segment = self._recording_segments[0]
        return {
            "logical_batch_reads": int(segment.logical_batch_reads),
            "logical_source_frames_read": int(segment.logical_source_frames_read),
            "logical_source_bytes_read": int(segment.logical_source_bytes_read),
        }
