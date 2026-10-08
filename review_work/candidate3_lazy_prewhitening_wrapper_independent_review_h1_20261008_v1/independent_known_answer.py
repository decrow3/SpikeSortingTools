"""Independent synthetic known-answer check for the candidate-3 lazy wrapper.

This deliberately does not use BinaryFiltered.filter to construct the expected
array.  It spells out the reviewed operation sequence with lower-level Torch and
Kilosort FFT primitives, while the production wrapper remains the only caller
of BinaryFiltered.filter.
"""

from __future__ import annotations

import json

import numpy as np
import torch
from kilosort import io as kilosort_io
from kilosort.preprocessing import fft_highpass, get_highpass_filter
from spikeinterface.core import NumpyRecording

from npx_preprocessing.motion import KilosortPrewhiteningRecording


FS = 29999.835983263598


def main() -> None:
    rows = np.arange(79, dtype=np.uint32)[:, None]
    cols = np.arange(5, dtype=np.uint32)[None, :]
    values = ((rows * 1297 + cols * 7919 + 32761) % 65536).astype(np.uint16)
    channel_ids = np.array(["u", "v", "w", "x", "y"])
    geometry = np.c_[np.arange(5, dtype=float) * 13.0, np.arange(5, dtype=float) * 21.0]
    recording = NumpyRecording(
        values,
        sampling_frequency=FS,
        t_starts=[23.125],
        channel_ids=channel_ids,
    )
    recording.set_channel_locations(geometry)
    selected_ids = np.array(["y", "v", "x"])
    selected_indices = np.array([4, 1, 3])
    scale = np.array([0.25, 1.5, 0.75, 2.0, 0.5], dtype=np.float32)
    shift = np.array([-3.0, 7.25, 0.5, -11.0, 2.0], dtype=np.float32)
    wrapper = KilosortPrewhiteningRecording(
        recording,
        channel_ids=selected_ids,
        scale=scale,
        shift=shift,
        invert_sign=True,
        do_car=True,
    )

    observed_nt: list[int] = []
    original_fft_highpass = kilosort_io.fft_highpass

    def recording_fft_highpass(hp_filter, NT):
        observed_nt.append(int(NT))
        return original_fft_highpass(hp_filter, NT=NT)

    kilosort_io.fft_highpass = recording_fft_highpass
    try:
        actual = wrapper.get_traces(start_frame=9, end_frame=62)
    finally:
        kilosort_io.fft_highpass = original_fft_highpass

    # Independent spelling of the source-bound operation order.
    expected_input = values[9:62].astype(np.float32) - np.float32(2**15)
    expected_input = expected_input * scale
    expected_input = expected_input + shift
    expected_tensor = torch.from_numpy(np.ascontiguousarray(expected_input.T)).float()
    expected_tensor = expected_tensor[torch.as_tensor(selected_indices, dtype=torch.int64)]
    expected_tensor = -expected_tensor
    expected_tensor = expected_tensor - expected_tensor.mean(1).unsqueeze(1)
    expected_tensor = expected_tensor - torch.median(expected_tensor, 0)[0]
    hp_filter = get_highpass_filter(fs=FS, cutoff=300.0, device=torch.device("cpu"))
    fwav = fft_highpass(hp_filter, NT=expected_tensor.shape[1])
    expected_tensor = torch.real(
        torch.fft.ifft(torch.fft.fft(expected_tensor) * torch.conj(fwav))
    )
    expected_tensor = torch.fft.fftshift(expected_tensor, dim=-1)
    expected = expected_tensor.numpy(force=True).T

    result = {
        "array_equal_manual_known_answer": bool(np.array_equal(actual, expected)),
        "fft_nt_exact_request_length": observed_nt == [53],
        "dtype_float32": actual.dtype == np.float32,
        "selected_ids_equal": bool(np.array_equal(wrapper.channel_ids, selected_ids)),
        "selected_geometry_equal": bool(
            np.array_equal(wrapper.get_channel_locations(), geometry[selected_indices])
        ),
        "nonzero_clock_equal": wrapper.sample_index_to_time(0) == 23.125,
        "shape_equal": actual.shape == (53, 3),
    }
    print(json.dumps(result, sort_keys=True))
    if not all(result.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
