"""Independent boundary controls for the K9 v2 guard-only repair."""

from __future__ import annotations

import numpy as np
import pytest

from PRIOR_ADVERSARIAL_REPLAY_V2 import prior, v2


@pytest.mark.parametrize(
    "bad",
    [
        np.ones((12, 2), dtype=bool),
        np.ones((12, 2), dtype=object),
        np.ones((12, 2), dtype=np.complex64) * (1 + 1j),
        np.ones((12, 2), dtype="U2"),
    ],
)
def test_geometry_guard_rejects_ambiguous_dtype_families(bad):
    with pytest.raises(ValueError, match="real numeric"):
        prior.inject([prior.event("bad")], geom=bad)


@pytest.mark.parametrize("dtype", [np.int16, np.float32, np.float64])
def test_geometry_guard_preserves_supported_real_numeric_families(dtype):
    out, rows = prior.inject([prior.event("ok")], geom=prior.geometry().astype(dtype))
    assert out.dtype == np.float32
    assert rows[0]["session_sample"] == prior.ORIGIN + 100


def test_derived_clock_accepts_exact_int64_upper_boundary():
    origin = int(np.iinfo(np.int64).max) - 100
    _, rows = prior.inject([prior.event("edge", sample=100)], origin=origin)
    assert rows[0]["session_sample"] == np.iinfo(np.int64).max
    assert isinstance(rows[0]["session_sample"], int)


def test_derived_clock_rejects_positive_overflow_and_preserves_low_boundary():
    with pytest.raises(ValueError, match="session_sample.*int64"):
        prior.inject([prior.event("positive", sample=100)], origin=np.iinfo(np.int64).max)
    _, rows = prior.inject([prior.event("low", sample=4)], origin=np.iinfo(np.int64).min)
    assert rows[0]["session_sample"] == int(np.iinfo(np.int64).min) + 4
