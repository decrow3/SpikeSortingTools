"""Independent adversarial checks for the frozen K1/K2 fixture packet.

These tests intentionally load the immutable snapshot rather than importing the
author's working source.  They use only synthetic binaries and integer arrays.
"""

from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path

import numpy as np
import pytest


HERE = Path(__file__).resolve().parent
AUTHOR = HERE.parent / "joint_model_k1_k2_fixture_h1_20261012_v1"


def load_snapshot(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


impl = load_snapshot("reviewed_joint_model_k1_k2", AUTHOR / "IMPLEMENTATION.py")
reader = impl.load_exact_reader(AUTHOR / "EXACT_READER.py")


def make_recording(
    tmp_path: Path,
    *,
    highpass: bool,
    invert: bool,
    artifact_threshold: float = np.inf,
):
    n_samples, n_raw = 29, 6
    raw = (
        (np.arange(n_samples * n_raw).reshape(n_samples, n_raw) * 7 + 3) % 61
    ).astype(np.int16) - 30
    raw[:, 0] += np.asarray(([0, -9, 4, 12, -5, 2, 7] * 5)[:n_samples], dtype=np.int16)
    path = tmp_path / f"synthetic_{highpass}_{invert}_{artifact_threshold}.bin"
    path.write_bytes(raw.tobytes())
    kwargs = dict(
        binary_path=path,
        num_samples=n_samples,
        num_channels_raw=n_raw,
        dtype="int16",
        sampling_frequency=2_000.0,
        batch_size=8,
        padding=2,
        channel_map=np.asarray([5, 2, 0, 4]),
        channel_ids=np.asarray([251, 252, 7, 8]),
        channel_locations=np.column_stack((np.zeros(4), np.arange(4) * 20.0)),
        highpass_filter=(
            np.asarray([0.125, -0.25, 0.125], dtype=np.float32)
            if highpass
            else None
        ),
        temporal_centering=True,
        do_car=True,
        invert_sign=invert,
        artifact_threshold=artifact_threshold,
        scale=0.25,
        shift=-1.5,
    )
    return raw, kwargs


def independent_batch_before_car(raw: np.ndarray, kwargs: dict, batch_index: int):
    """Construct one padded/mapped/centered batch without reader helpers."""
    batch = kwargs["batch_size"]
    pad = kwargs["padding"]
    width = batch + 2 * pad
    if batch_index == 0:
        start = 0
        end = min(raw.shape[0], batch + pad)
        values = raw[start:end].T.astype(np.float32) * kwargs["scale"] + kwargs["shift"]
        padded = np.empty((raw.shape[1], width), dtype=np.float32)
        padded[:, pad : pad + values.shape[1]] = values
        padded[:, :pad] = values[:, :1]
        filled = pad + values.shape[1]
        if filled < width:
            padded[:, filled:] = values[:, -1:]
    else:
        start = batch_index * batch - pad
        end = min(raw.shape[0], start + width)
        values = raw[start:end].T.astype(np.float32) * kwargs["scale"] + kwargs["shift"]
        padded = np.empty((raw.shape[1], width), dtype=np.float32)
        padded[:, : values.shape[1]] = values
        if values.shape[1] < width:
            padded[:, values.shape[1] :] = values[:, -1:]
    mapped = padded[kwargs["channel_map"]]
    if kwargs["invert_sign"]:
        mapped = -mapped
    return mapped - mapped.mean(axis=1, keepdims=True, dtype=np.float32)


def lower_median(values: np.ndarray) -> np.ndarray:
    return np.partition(values, (values.shape[0] - 1) // 2, axis=0)[
        (values.shape[0] - 1) // 2
    ]


def numpy_highpass(values: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    length = values.shape[-1]
    left = (length - kernel.size) // 2
    padded = np.pad(kernel, (left, length - left - kernel.size))
    transformed = np.fft.ifft(
        np.fft.fft(values, axis=-1) * np.conj(np.fft.fft(padded)), axis=-1
    ).real
    return np.fft.fftshift(transformed, axes=-1)


@pytest.mark.parametrize("highpass,invert", [(False, False), (False, True), (True, False), (True, True)])
def test_k1_full_join_subset_equality_and_independent_median(
    tmp_path, highpass, invert
):
    raw, kwargs = make_recording(tmp_path, highpass=highpass, invert=invert)
    baseline = reader.ProspectiveAPrewhiteningRecording(**kwargs)
    candidate_recording = reader.ProspectiveAPrewhiteningRecording(**kwargs)
    candidate = impl.CARSignalSideChannel(candidate_recording, reader)
    selector = np.asarray([True, False, True, True])

    expected_traces = baseline._recording_segments[0].get_traces(0, 29, selector)
    actual_traces, actual_median = candidate.get_traces_and_hp_median(0, 29, selector)
    assert np.array_equal(actual_traces, expected_traces)

    pieces = []
    for batch_index in range(4):
        centered = independent_batch_before_car(raw, kwargs, batch_index)
        median = lower_median(centered)
        if highpass:
            median = numpy_highpass(median, kwargs["highpass_filter"])
        core = min(8, 29 - 8 * batch_index)
        pieces.append(median[2 : 2 + core])
    independent = np.concatenate(pieces).astype(np.float32)
    assert np.allclose(actual_median, independent, rtol=2e-5, atol=2e-5)


def test_k1_artifact_zeroes_first_interior_and_final_batches(tmp_path):
    _, kwargs = make_recording(
        tmp_path, highpass=True, invert=False, artifact_threshold=1e-6
    )
    recording = reader.ProspectiveAPrewhiteningRecording(**kwargs)
    candidate = impl.CARSignalSideChannel(recording, reader)
    traces, median = candidate.get_traces_and_hp_median(0, 29, [0, 2])
    assert np.array_equal(traces, np.zeros((29, 2), dtype=np.float32))
    assert np.array_equal(median, np.zeros(29, dtype=np.float32))


def test_k1_detector_plateau_threshold_refractory_boundary_and_origin():
    signal = np.zeros(30, dtype=np.float32)
    signal[2:4] = -121.0  # strict-left/non-strict-right aligns to 2
    signal[8] = -121.0  # exactly 6 samples away: 3 ms at 2 kHz
    signal[11] = -120.0  # strict threshold excludes it
    signal[14] = -130.0
    signal[18] = -180.0  # suppresses 14 because it is deeper and < refractory
    assert impl.detect_frozen_pulses(
        signal, sampling_frequency=2_000.0, sample_origin=10_000
    ).tolist() == [10_002, 10_008, 10_018]


def brute_ccg(pulses, detections, channels, half_width):
    counts = []
    for lag in range(-half_width, half_width + 1):
        counts.append(
            sum(
                int(detection - pulse == lag)
                for pulse in pulses
                for detection, channel in zip(detections, channels)
                if channel in (251, 252)
            )
        )
    return counts


def test_k2_matches_independent_brute_force_oracle_and_tie_rule():
    pulses = np.asarray([10, 20, 20, 40], dtype=np.int64)
    detections = np.asarray([7, 9, 11, 17, 19, 21, 23, 39, 41, 43, 10_000])
    channels = np.asarray([251, 252, 251, 252, 251, 252, 251, 252, 251, 252, 99])
    result = impl.freeze_pulse_sorter_lag(
        pulses,
        detections,
        channels,
        sampling_frequency=2_000.0,
        clock_identity="independent-session-clock",
    )
    expected = brute_ccg(pulses, detections, channels, 3)
    assert result["lags_samples"] == list(range(-3, 4))
    assert result["counts"] == expected
    maximum = max(expected)
    candidates = [lag for lag, count in zip(range(-3, 4), expected) if count == maximum]
    assert result["peak_lag_samples"] == min(candidates, key=lambda lag: (abs(lag), lag))
    assert result["n_focus_detections"] == 10
    assert "cluster" not in inspect.signature(impl.freeze_pulse_sorter_lag).parameters
    with pytest.raises(TypeError):
        impl.freeze_pulse_sorter_lag(
            pulses,
            detections,
            channels,
            sampling_frequency=2_000.0,
            clock_identity="independent-session-clock",
            cluster_labels=np.zeros_like(detections),
        )


def test_k2_rejects_complex_clock_values_instead_of_silently_casting():
    with pytest.raises(ValueError, match="integer"):
        impl.freeze_pulse_sorter_lag(
            np.asarray([100 + 0j]),
            np.asarray([101]),
            np.asarray([251]),
            sampling_frequency=30_000.0,
            clock_identity="independent-session-clock",
        )


@pytest.mark.parametrize(
    "pulses,detections,channels,clock",
    [
        (np.asarray([1.25]), np.asarray([2]), np.asarray([251]), "clock"),
        (np.asarray([1]), np.asarray([2.25]), np.asarray([251]), "clock"),
        (np.asarray([1]), np.asarray([2]), np.asarray([251.25]), "clock"),
        (np.asarray([1]), np.asarray([2]), np.asarray([251]), ""),
    ],
)
def test_k2_fractional_and_clock_guards(pulses, detections, channels, clock):
    with pytest.raises(ValueError):
        impl.freeze_pulse_sorter_lag(
            pulses,
            detections,
            channels,
            sampling_frequency=30_000.0,
            clock_identity=clock,
        )
