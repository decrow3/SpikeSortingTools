"""Frozen CPU float32 waveform preprocessing matching Kilosort 4.0.27.

This narrow adapter copies the exercised semantics of
``BinaryFiltered.filter`` and ``preprocessing.fft_highpass`` without importing
the rest of Kilosort. It intentionally performs common-*median* referencing.
"""

from __future__ import annotations

import torch
from torch.fft import fft, fftshift, ifft


def center_and_common_median(x: torch.Tensor) -> torch.Tensor:
    """Subtract per-channel time mean, then across-channel median per sample."""
    if x.device.type != "cpu" or x.dtype != torch.float32 or x.ndim != 2:
        raise ValueError("input must be a two-dimensional CPU float32 tensor")
    centered = x - x.mean(1).unsqueeze(1)
    return centered - torch.median(centered, 0)[0]


def fft_highpass(hp_filter: torch.Tensor, nt: int) -> torch.Tensor:
    """Exact crop/pad and FFT rule from Kilosort preprocessing.py."""
    if hp_filter.device.type != "cpu" or hp_filter.dtype != torch.float32 or hp_filter.ndim != 1:
        raise ValueError("high-pass impulse must be a one-dimensional CPU float32 tensor")
    ft = hp_filter.shape[0]
    if ft < nt:
        pad = (nt - ft) // 2
        values = torch.cat((torch.zeros(pad), hp_filter, torch.zeros(pad + (nt - pad * 2 - ft))))
    elif ft > nt:
        crop = (ft - nt) // 2
        values = hp_filter[crop:crop + nt]
    else:
        values = hp_filter
    return fft(values)


def preprocess_padded_record(x: torch.Tensor, hp_filter: torch.Tensor) -> torch.Tensor:
    """Apply the frozen reference/high-pass path to one channels-by-samples record."""
    if x.shape != (384, 1145):
        raise ValueError("waveform record must have shape (384, 1145)")
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    values = center_and_common_median(x)
    fwav = fft_highpass(hp_filter, values.shape[1])
    values = torch.real(ifft(fft(values) * torch.conj(fwav)))
    return fftshift(values, dim=-1)
