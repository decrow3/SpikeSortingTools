"""Crop adapters for applying the frozen EN panel without origin/exposure drift."""

from __future__ import annotations

import numpy as np

from testing.en_tier1_panel import FieldCells


def localize_global_samples(times: np.ndarray, start: int, stop: int) -> np.ndarray:
    """Filter a global half-open crop and subtract its origin exactly once."""
    times = np.asarray(times, dtype=np.int64)
    selected = times[(times >= start) & (times < stop)]
    return selected - np.int64(start)


def crop_field_cells_from_global_midpoints(
    temporal_bins_s: np.ndarray,
    rounded_states_um: np.ndarray,
    global_duration_s: float,
    crop_start_s: float,
    crop_end_s: float,
    block_seconds: float,
) -> FieldCells:
    """Clip full-session midpoint cells, retain original segments, then localize."""
    temporal_bins_s = np.asarray(temporal_bins_s, dtype=np.float64)
    rounded = np.asarray(rounded_states_um, dtype=np.float64)
    if len(temporal_bins_s) != len(rounded) or len(rounded) == 0:
        raise ValueError("field times/states differ or are empty")
    if not (0 <= crop_start_s < crop_end_s <= global_duration_s):
        raise ValueError("invalid crop")
    global_edges = np.empty(len(temporal_bins_s) + 1, dtype=np.float64)
    global_edges[1:-1] = (temporal_bins_s[:-1] + temporal_bins_s[1:]) / 2.0
    global_edges[0], global_edges[-1] = 0.0, float(global_duration_s)
    global_edges = np.clip(global_edges, 0.0, float(global_duration_s))
    states = np.unique(rounded)
    state_index = np.searchsorted(states, rounded).astype(np.int16)
    segment = np.r_[0, np.cumsum(np.diff(state_index) != 0)].astype(np.int32)
    overlap = (global_edges[:-1] < crop_end_s) & (global_edges[1:] > crop_start_s)
    indices = np.flatnonzero(overlap)
    if not len(indices) or np.any(np.diff(indices) != 1):
        raise RuntimeError("crop cells are empty or noncontiguous")
    left = np.maximum(global_edges[indices], crop_start_s)
    right = np.minimum(global_edges[indices + 1], crop_end_s)
    local_edges = np.r_[left[0], right] - crop_start_s
    duration = crop_end_s - crop_start_s
    if not np.isclose(local_edges[0], 0.0) or not np.isclose(local_edges[-1], duration):
        raise RuntimeError("localized field cells do not close crop")
    local_state = state_index[indices]
    local_segment = segment[indices]
    widths = np.diff(local_edges)
    exposure = np.bincount(local_state, weights=widths, minlength=len(states)).astype(float)
    block_edges = np.unique(np.r_[np.arange(0.0, duration, block_seconds), duration])
    block_exposure = np.zeros((len(block_edges) - 1, len(states)), dtype=np.float64)
    for cell, state in enumerate(local_state):
        cell_left, cell_right = local_edges[cell : cell + 2]
        first = max(0, np.searchsorted(block_edges, cell_left, side="right") - 1)
        last = min(len(block_edges) - 2, np.searchsorted(block_edges, cell_right, side="left"))
        for block in range(first, last + 1):
            block_exposure[block, state] += max(
                0.0, min(cell_right, block_edges[block + 1]) - max(cell_left, block_edges[block])
            )
    if not np.isclose(exposure.sum(), duration) or not np.allclose(block_exposure.sum(1), np.diff(block_edges)):
        raise RuntimeError("crop field exposure does not close")
    return FieldCells(states, local_edges, local_state, local_segment, exposure, block_exposure, block_edges)
