#!/usr/bin/env python3
"""Translation-invariance known-answer fixture for nonzero-origin crop metrics."""

from __future__ import annotations

import json
import numpy as np

from testing.en_common_support_crop_panel import crop_field_cells_from_global_midpoints, localize_global_samples


def main() -> None:
    centers = np.arange(0.5, 8.0, 1.0)
    state = np.array([0, 0, -40, -40, 0, 40, 40, 0], dtype=float)
    crop_start, crop_stop = 2.25, 6.75
    cells = crop_field_cells_from_global_midpoints(centers, state, 8.0, crop_start, crop_stop, 2.0)
    global_seconds = np.array([2.25, 2.999, 3.0, 4.999, 5.0, 6.749])
    local_seconds = global_seconds - crop_start
    observed_state, observed_segment = cells.assign(local_seconds, 1.0)
    global_edges = np.r_[0.0, (centers[:-1] + centers[1:]) / 2, 8.0]
    expected_cell = np.searchsorted(global_edges, global_seconds, side="right") - 1
    expected_state_values = state[expected_cell]
    observed_state_values = cells.states_um[observed_state]
    global_segment = np.r_[0, np.cumsum(np.diff(state) != 0)]
    if not np.array_equal(observed_state_values, expected_state_values):
        raise RuntimeError("state assignment changed under crop translation")
    if not np.array_equal(observed_segment, global_segment[expected_cell]):
        raise RuntimeError("original state-segment identity was reset at crop")
    frames = np.array([224, 225, 300, 674, 675], dtype=np.int64)
    localized = localize_global_samples(frames, 225, 675)
    if not np.array_equal(localized, [0, 75, 449]):
        raise RuntimeError("global sample origin was not subtracted exactly once")
    fs = 10.1
    requested_tmin = 2.25
    effective_frame = int(np.int64(requested_tmin * fs))
    effective_origin_s = effective_frame / fs
    fractional_floor_origin_checked = effective_frame == 22 and not np.isclose(effective_origin_s, requested_tmin)
    if not fractional_floor_origin_checked:
        raise RuntimeError("fractional-sample floor-origin fixture failed")
    floor_centers = np.array([0.5, 1.5, 2.9, 4.0])
    floor_states = np.array([0.0, -40.0, 40.0, 0.0])
    effective_cells = crop_field_cells_from_global_midpoints(
        floor_centers, floor_states, 5.0, effective_origin_s, 3.0, 1.0
    )
    requested_cells = crop_field_cells_from_global_midpoints(
        floor_centers, floor_states, 5.0, requested_tmin, 3.0, 1.0
    )
    minus = int(np.flatnonzero(effective_cells.states_um == -40.0)[0])
    fractional_floor_boundary_integration_pass = (
        effective_cells.state_exposure_s[minus] > 0 and requested_cells.state_exposure_s[minus] == 0
    )
    if not fractional_floor_boundary_integration_pass:
        raise RuntimeError("floored field-boundary integration fixture failed")
    result = {
        "schema": "en-common-support-crop-panel-fixture-v1",
        "status": "pass",
        "nonzero_crop_origin_s": crop_start,
        "boundary_knots_checked": global_seconds.tolist(),
        "translation_invariant_states": True,
        "original_segment_ids_preserved": True,
        "half_open_sample_filter_and_single_subtraction": True,
        "fractional_floor_origin_checked": fractional_floor_origin_checked,
        "fractional_floor_boundary_integration_pass": bool(fractional_floor_boundary_integration_pass),
        "exposure_sum_s": float(cells.state_exposure_s.sum()),
        "block_exposure_closes": bool(np.allclose(cells.block_exposure_s.sum(1), np.diff(cells.block_edges_s))),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
