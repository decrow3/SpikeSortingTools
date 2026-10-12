"""Focused independent rereview fixtures for the repaired K1/K2 v2 packet."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import numpy as np
import pytest


HERE = Path(__file__).resolve().parent
REVIEW_WORK = HERE.parent
V1_PACKET = REVIEW_WORK / "joint_model_k1_k2_fixture_h1_20261012_v1"
V2_PACKET = REVIEW_WORK / "joint_model_k1_k2_fixture_h1_20261012_v2"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


v1 = load_module("joint_model_k1_k2_review_v1", V1_PACKET / "IMPLEMENTATION.py")
v2 = load_module("joint_model_k1_k2_review_v2", V2_PACKET / "IMPLEMENTATION.py")


def module_without_integer_validator(path: Path) -> str:
    tree = ast.parse(path.read_text())
    for index, node in enumerate(tree.body):
        if isinstance(node, ast.FunctionDef) and node.name == "_integer_1d":
            tree.body[index] = ast.FunctionDef(
                name="_integer_1d",
                args=node.args,
                body=[ast.Pass()],
                decorator_list=[],
                returns=node.returns,
                type_comment=None,
            )
            break
    return ast.dump(tree, include_attributes=False)


def test_v2_changes_only_the_integer_validator():
    assert module_without_integer_validator(V1_PACKET / "IMPLEMENTATION.py") == (
        module_without_integer_validator(V2_PACKET / "IMPLEMENTATION.py")
    )


def test_v1_failure_is_preserved_and_v2_rejects_it():
    bad = np.asarray([100 + 0j])
    with pytest.warns(np.exceptions.ComplexWarning):
        accepted = v1._integer_1d("pulse_samples", bad)
    assert accepted.tolist() == [100]
    with pytest.raises(ValueError, match="integer"):
        v2._integer_1d("pulse_samples", bad)


@pytest.mark.parametrize(
    "bad",
    [
        np.asarray([100 + 0j], dtype=np.complex64),
        np.asarray([100 + 2j], dtype=np.complex128),
        np.asarray([True], dtype=np.bool_),
        np.asarray([100], dtype=object),
        np.asarray(["100"]),
        np.asarray([b"100"]),
        np.asarray([2**63], dtype=np.uint64),
        np.asarray([np.iinfo(np.uint64).max], dtype=np.uint64),
        np.asarray([2.0**63], dtype=np.float64),
        np.asarray([np.nextafter(-(2.0**63), -np.inf)], dtype=np.float64),
        np.asarray([100.25], dtype=np.float64),
        np.asarray([np.nan], dtype=np.float64),
        np.asarray([np.inf], dtype=np.float64),
    ],
)
def test_v2_rejects_noninteger_or_out_of_int64_domain_arrays(bad):
    with pytest.raises(ValueError):
        v2._integer_1d("review_samples", bad)


@pytest.mark.parametrize(
    "valid,expected",
    [
        (np.asarray([np.iinfo(np.int64).min], dtype=np.int64), [-(2**63)]),
        (np.asarray([np.iinfo(np.int64).max], dtype=np.int64), [2**63 - 1]),
        (np.asarray([2**63 - 1], dtype=np.uint64), [2**63 - 1]),
        (np.asarray([-(2.0**63)], dtype=np.float64), [-(2**63)]),
        (np.asarray([-17.0, 0.0, 42.0], dtype=np.float32), [-17, 0, 42]),
        (np.asarray([], dtype=np.float64), []),
    ],
)
def test_v2_accepts_representable_integral_arrays(valid, expected):
    result = v2._integer_1d("review_samples", valid)
    assert result.dtype == np.int64
    assert result.tolist() == expected


@pytest.mark.parametrize("position", ["pulse", "detection", "channel"])
def test_v2_public_k2_interface_rejects_every_bad_companion_position(position):
    pulses = np.asarray([100])
    detections = np.asarray([101])
    channels = np.asarray([251])
    if position == "pulse":
        pulses = np.asarray([100 + 0j])
    elif position == "detection":
        detections = np.asarray([101], dtype=object)
    else:
        channels = np.asarray([True])
    with pytest.raises(ValueError):
        v2.freeze_pulse_sorter_lag(
            pulses,
            detections,
            channels,
            sampling_frequency=30_000.0,
            clock_identity="rereview-session-clock",
        )


def test_k1_detector_and_k2_statistics_are_unchanged_on_valid_inputs():
    signal = np.zeros(260, dtype=np.float32)
    signal[10] = -130.0
    signal[50] = -200.0
    signal[140:142] = -150.0
    signal[230] = -120.0
    assert np.array_equal(
        v1.detect_frozen_pulses(signal, sampling_frequency=30_000.0, sample_origin=1000),
        v2.detect_frozen_pulses(signal, sampling_frequency=30_000.0, sample_origin=1000),
    )

    pulses = np.asarray([100, 300, 500], dtype=np.int64)
    detections = np.asarray([105, 106, 107, 305, 306, 307, 506, 900, 106, 306])
    channels = np.asarray([251, 252, 251, 251, 252, 252, 251, 251, 99, 99])
    kwargs = dict(
        sampling_frequency=30_000.0,
        clock_identity="rereview-session-clock",
    )
    assert v1.freeze_pulse_sorter_lag(pulses, detections, channels, **kwargs) == (
        v2.freeze_pulse_sorter_lag(pulses, detections, channels, **kwargs)
    )
