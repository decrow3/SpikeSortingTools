"""One bounded waveform-injection function for the existing C4 reader."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def inject_waveforms_before_centering_car(
    values: torch.Tensor,
    *,
    padded_start_sample: int,
    recording_num_samples: int,
    event_samples: Any,
    event_waveforms: Any,
    trough_offset_samples: int,
) -> torch.Tensor:
    """Add preplaced event waveforms to one channel-by-time padded reader batch.

    ``event_samples`` are session-clock trough samples. ``event_waveforms`` has
    shape event x selected-channel x waveform-sample and already encodes each
    event's frozen donor amplitude and physical placement/state. The function
    clips only at batch and recording bounds and preserves deterministic input
    row order for overlapping additions.
    """
    if not isinstance(values, torch.Tensor) or values.ndim != 2:
        raise ValueError("values must be a channel-by-time torch tensor")
    if not values.dtype.is_floating_point:
        raise ValueError("values must have floating dtype")
    if isinstance(padded_start_sample, (bool, np.bool_)):
        raise ValueError("padded_start_sample must be an integer")
    if isinstance(recording_num_samples, (bool, np.bool_)):
        raise ValueError("recording_num_samples must be an integer")
    if isinstance(trough_offset_samples, (bool, np.bool_)):
        raise ValueError("trough_offset_samples must be an integer")
    padded_start = int(padded_start_sample)
    recording_samples = int(recording_num_samples)
    trough_offset = int(trough_offset_samples)
    if recording_samples <= 0 or trough_offset < 0:
        raise ValueError("recording length must be positive and trough offset nonnegative")

    samples = np.asarray(event_samples)
    if samples.ndim != 1 or np.issubdtype(samples.dtype, np.bool_) or not np.issubdtype(samples.dtype, np.integer):
        raise ValueError("event_samples must be a one-dimensional integer array")
    if samples.size and (
        np.any(samples < 0)
        or np.any(samples >= recording_samples)
        or np.any(samples > np.iinfo(np.int64).max)
    ):
        raise ValueError("event_samples fall outside the recording")
    samples = samples.astype(np.int64, copy=False)

    waveforms = torch.as_tensor(event_waveforms, device=values.device)
    if waveforms.ndim != 3:
        raise ValueError("event_waveforms must have event x channel x sample shape")
    if waveforms.shape[0] != samples.size or waveforms.shape[1] != values.shape[0]:
        raise ValueError("event waveforms do not align with samples and selected channels")
    if trough_offset >= waveforms.shape[2]:
        raise ValueError("trough offset falls outside the waveform")
    if not waveforms.dtype.is_floating_point:
        raise ValueError("event_waveforms must have floating dtype")
    waveforms = waveforms.to(dtype=values.dtype)
    if not torch.isfinite(waveforms).all():
        raise ValueError("event_waveforms contain nonfinite values")

    output = values.clone()
    padded_stop = padded_start + int(values.shape[1])
    for row in range(samples.size):
        waveform_start = int(samples[row]) - trough_offset
        waveform_stop = waveform_start + int(waveforms.shape[2])
        overlap_start = max(waveform_start, padded_start, 0)
        overlap_stop = min(waveform_stop, padded_stop, recording_samples)
        if overlap_start >= overlap_stop:
            continue
        output_start = overlap_start - padded_start
        waveform_local_start = overlap_start - waveform_start
        width = overlap_stop - overlap_start
        output[:, output_start : output_start + width] += waveforms[
            row, :, waveform_local_start : waveform_local_start + width
        ]
    return output
