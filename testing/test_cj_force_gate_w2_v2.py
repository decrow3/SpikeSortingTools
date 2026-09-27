import json

import numpy as np

from testing.cj_force_gate_w2_v2 import (
    EXPECTED_MASK_SHA256,
    lag_counts,
    load_exact_rest_blocks,
    session_time_to_local_frame,
)


def test_excluded_inner_lags_and_exact_outer_boundaries():
    values = np.array([
        -90, -89, -45, -30, -29, -9, -8, 0, 8, 9, 29, 30, 45, 89, 90,
    ])
    central, shoulder = lag_counts(np.array([0]), values)
    assert central == 4
    assert shoulder == 4


def test_off_grid_mask_boundary_builds_exact_complement(tmp_path):
    mask = tmp_path / "mask.csv"
    mask.write_text("start_s,end_s,source\n5.100,6.100,E\n")
    import hashlib
    digest = hashlib.sha256(mask.read_bytes()).hexdigest()
    metadata = tmp_path / "metadata.json"
    metadata.write_text(json.dumps({
        "sampling_frequency_hz": 10.3,
        "exact_source_frame_interval_half_open": [0, 126],
        "local_frame_count": 126,
        "session_time_offset_s": 0.0,
        "exact_realized_session_end_s": 12.2,
        "mask_sha256": digest,
    }))
    # The fixture exercises off-grid conversion, not the production hash pin.
    import testing.cj_force_gate_w2_v2 as module
    original = module.EXPECTED_MASK_SHA256
    module.EXPECTED_MASK_SHA256 = digest
    try:
        blocks, intervals, _ = load_exact_rest_blocks(mask, metadata)
    finally:
        module.EXPECTED_MASK_SHA256 = original
    assert intervals == [(5.1, 6.1, "E")]
    assert [(b["start_sample"], b["end_sample"]) for b in blocks] == [
        (0, 52), (63, 115)]
    assert session_time_to_local_frame(5.1, 10.3, 0) == 53
    assert blocks[0]["end_sample"] <= 53
    assert blocks[1]["start_sample"] >= 63
    assert len(EXPECTED_MASK_SHA256) == 64
