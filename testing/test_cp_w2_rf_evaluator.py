from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat

from testing.cp_w2_rf_evaluator import explicit_imec1_clock, parse_arm


def test_parse_arm_requires_explicit_name_and_path():
    assert parse_arm("static=/tmp/s.npz") == ("static", Path("/tmp/s.npz"))
    with pytest.raises(Exception):
        parse_arm("/tmp/s.npz")


def test_clock_requires_explicit_imec1_not_imec0(tmp_path):
    path = tmp_path / "clock.mat"
    samples = np.arange(20, dtype=float) * 30_000
    savemat(path, {"clock": {"imec0Time": samples, "NidaqTime": samples}})
    with pytest.raises(ValueError, match="explicit imec1"):
        explicit_imec1_clock(path)


def test_clock_maps_explicit_imec1(tmp_path):
    path = tmp_path / "clock.mat"
    samples = np.arange(20, dtype=float) * 30_000
    savemat(
        path,
        {"clock": {"imec1Time": samples, "NidaqTime": 1.0001 * samples + 7}},
    )
    transform, info = explicit_imec1_clock(path)
    assert info["imec_field"] == "imec1Time"
    assert np.allclose(transform([0, 30_000]), [7, 30_010], atol=1e-6)
