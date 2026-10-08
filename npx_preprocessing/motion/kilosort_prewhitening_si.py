"""Lazy SpikeInterface adapter for the Kilosort immediate pre-whitening boundary.

Each SpikeInterface trace request is filtered as one Kilosort request.  The
adapter deliberately adds no hidden padding or caching: callers that need a
padded core must request the padded interval and crop the returned values.
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import torch
    from kilosort.io import BinaryFiltered
    from kilosort.preprocessing import get_highpass_filter
    from spikeinterface.preprocessing.basepreprocessor import (
        BasePreprocessor,
        BasePreprocessorSegment,
    )
except ImportError:  # pragma: no cover - import guard for geometry-only installs
    torch = None  # type: ignore[assignment]
    BinaryFiltered = None  # type: ignore[assignment]
    get_highpass_filter = None  # type: ignore[assignment]
    BasePreprocessor = None  # type: ignore[assignment]
    BasePreprocessorSegment = object  # type: ignore[assignment,misc]


def _finite_parameter(value: Any, name: str) -> float | np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float32)
    if array.ndim > 1 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite scalar or one-dimensional array")
    if array.ndim == 0:
        return float(array)
    return array


def _serialize_parameter(value: float | np.ndarray | None) -> float | list[float] | None:
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


if BasePreprocessor is not None:

    class _KilosortPrewhiteningSegment(BasePreprocessorSegment):
        def __init__(
            self,
            parent_recording_segment: Any,
            *,
            parent_dtype: np.dtype,
            parent_channel_indices: np.ndarray,
            hp_filter: Any,
            do_car: bool,
            invert_sign: bool,
            scale: float | np.ndarray | None,
            shift: float | np.ndarray | None,
            device: Any,
        ) -> None:
            super().__init__(parent_recording_segment)
            self.parent_dtype = np.dtype(parent_dtype)
            self.parent_channel_indices = np.asarray(parent_channel_indices, dtype=np.int64)
            self.scale = scale
            self.shift = shift
            self.device = device
            producer = object.__new__(BinaryFiltered)
            producer.chan_map = torch.as_tensor(
                self.parent_channel_indices, dtype=torch.int64, device=device
            )
            producer.invert_sign = bool(invert_sign)
            producer.do_CAR = bool(do_car)
            producer.hp_filter = hp_filter
            producer.artifact_threshold = np.inf
            producer.whiten_mat = None
            producer.dshift = None
            producer.device = device
            self.producer = producer

        def get_traces(self, start_frame, end_frame, channel_indices):
            start = 0 if start_frame is None else int(start_frame)
            end = (
                int(self.parent_recording_segment.get_num_samples())
                if end_frame is None
                else int(end_frame)
            )
            if end <= start:
                raise ValueError("Kilosort prewhitening requests must contain samples")
            raw = self.parent_recording_segment.get_traces(start, end, slice(None))
            values = np.asarray(raw)
            if self.parent_dtype == np.dtype("uint16"):
                values = values.astype(np.float32) - np.float32(2**15)
            else:
                values = values.astype(np.float32, copy=False)
            if self.scale is not None:
                values = values * self.scale
            if self.shift is not None:
                values = values + self.shift
            tensor = torch.from_numpy(np.ascontiguousarray(values.T)).to(self.device).float()
            filtered = BinaryFiltered.filter(self.producer, tensor)
            output = filtered.detach().cpu().numpy().T
            if output.dtype != np.float32:
                output = output.astype(np.float32, copy=False)
            if channel_indices is None:
                return output
            return output[:, channel_indices]


    class KilosortPrewhiteningRecording(BasePreprocessor):
        """Expose the installed Kilosort pre-whitening filter as a lazy recording.

        ``channel_ids`` defines the Kilosort channel map and its order.  Scaling
        and shifting occur before channel selection, exactly as in
        ``BinaryRWFile.__getitem__``.  Whitening, artifacts, drift correction,
        implicit padding and whole-recording materialization are intentionally
        absent.
        """

        def __init__(
            self,
            recording,
            *,
            channel_ids=None,
            highpass_cutoff_hz: float = 300.0,
            do_car: bool = True,
            invert_sign: bool = False,
            scale=None,
            shift=None,
            device: str = "cpu",
        ) -> None:
            if BinaryFiltered is None or get_highpass_filter is None or torch is None:
                raise ImportError("Kilosort, Torch and SpikeInterface are required")
            cutoff = float(highpass_cutoff_hz)
            if not np.isfinite(cutoff) or cutoff <= 0:
                raise ValueError("highpass_cutoff_hz must be finite and positive")
            if device != "cpu":
                raise ValueError("the reviewed lazy recording currently supports device='cpu' only")
            torch_device = torch.device(device)
            selected_ids = recording.channel_ids if channel_ids is None else np.asarray(channel_ids)
            if np.unique(selected_ids).size != len(selected_ids):
                raise ValueError("channel_ids must be unique")
            try:
                parent_indices = np.asarray(
                    recording.ids_to_indices(selected_ids, prefer_slice=False), dtype=np.int64
                )
            except Exception as exc:
                raise ValueError("channel_ids must all exist in the parent recording") from exc
            scale_value = _finite_parameter(scale, "scale")
            shift_value = _finite_parameter(shift, "shift")
            for name, value in (("scale", scale_value), ("shift", shift_value)):
                if isinstance(value, np.ndarray) and value.shape != (recording.get_num_channels(),):
                    raise ValueError(
                        f"{name} arrays must have one value per parent channel before channel mapping"
                    )
            hp_filter = get_highpass_filter(
                fs=float(recording.get_sampling_frequency()),
                cutoff=cutoff,
                device=torch_device,
            )
            super().__init__(recording, channel_ids=selected_ids, dtype=np.dtype("float32"))
            for parent_segment in recording._recording_segments:
                self.add_recording_segment(
                    _KilosortPrewhiteningSegment(
                        parent_segment,
                        parent_dtype=recording.get_dtype(),
                        parent_channel_indices=parent_indices,
                        hp_filter=hp_filter,
                        do_car=do_car,
                        invert_sign=invert_sign,
                        scale=scale_value,
                        shift=shift_value,
                        device=torch_device,
                    )
                )
            self._kwargs = {
                "recording": recording,
                "channel_ids": selected_ids.tolist(),
                "highpass_cutoff_hz": cutoff,
                "do_car": bool(do_car),
                "invert_sign": bool(invert_sign),
                "scale": _serialize_parameter(scale_value),
                "shift": _serialize_parameter(shift_value),
                "device": device,
            }

else:

    class KilosortPrewhiteningRecording:  # pragma: no cover - import guard
        def __init__(self, *args, **kwargs):
            raise ImportError("Kilosort, Torch and SpikeInterface are required")


__all__ = ["KilosortPrewhiteningRecording"]

