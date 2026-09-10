from types import SimpleNamespace

import json
import numpy as np
import pytest

from testing.luke_medicine_motion_aware_merge import (
    begin_atomic_stage,
    finish_atomic_stage,
    geometry_audit,
    window_indices,
)


def test_geometry_audit_requires_40um_same_column_recurrence():
    pos = np.array([[16, 0], [48, 0], [0, 20], [32, 20], [16, 40], [48, 40]])
    result = geometry_audit(pos, np.zeros(len(pos), dtype=int))
    assert result["coordinate_matched_channel_counts"]["20"] == 0
    assert result["coordinate_matched_channel_counts"]["40"] == 2


def test_window_indices_are_half_open():
    st = np.array([[0, 0], [10, 0], [20, 0], [30, 0]], dtype=float)
    assert window_indices(st, 10.0, 1.0, 3.0) == (1, 3)


def test_atomic_stage_refuses_digest_mismatch_and_partial(tmp_path):
    out = tmp_path / "stage"
    stage, digest = begin_atomic_stage(out, {"x": 1})
    finish_atomic_stage(stage, out, {"request_digest": digest, "complete": True})
    with pytest.raises(RuntimeError, match="another"):
        begin_atomic_stage(out, {"x": 2})
    partial = tmp_path / "other.partial"
    partial.mkdir()
    with pytest.raises(RuntimeError, match="requires inspection"):
        begin_atomic_stage(tmp_path / "other", {"x": 1})
