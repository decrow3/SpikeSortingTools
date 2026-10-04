"""Opt-in unsupported-site mask between native filtering and whitening.

This candidate intentionally calls the installed ``BinaryFiltered.filter``
with whitening temporarily disabled. Channel selection, sign inversion,
per-channel centering, CAR, high-pass filtering, and artifact handling therefore
remain native and in their original order. Only then are geometrically
unsupported target rows zeroed before the unchanged whitening matrix is applied.

The pointwise mask does not undo high-pass transients already created on rows
that are supported at a state transition. It isolates the known contribution
from rows that are unsupported at each sample; it is not a complete transition
repair.

The bounded B384 launcher separately requires the original 384-row geometry and
identity channel map. On the verified EN 40-um lattice states, exact-site
support from ``ExactLatticeRemapRecording`` and SI's inclusive bounding-box
support are identical. This helper's bounding-box implementation is not a
license to apply the candidate to cropped or unverified geometries.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class SupportMaskedBatch:
    prewhite: torch.Tensor
    support: torch.Tensor
    baseline_whitened: torch.Tensor
    candidate_prewhite: torch.Tensor
    candidate_whitened: torch.Tensor


def si_stepwise_states(
    temporal_centers_s: np.ndarray,
    shifts_um: np.ndarray,
    sample_frames: np.ndarray,
    sampling_frequency_hz: float,
    *,
    sample_frame_origin: int = 0,
) -> np.ndarray:
    """Evaluate the frozen EN field using SI's explicit half-open time cells.

    ``temporal_centers_s`` are already on the frozen field's global clock.
    ``sample_frame_origin`` converts local batch/crop frame indices to that
    global clock.  It deliberately applies only to sample frames; shifting both
    clocks together would cancel and could conceal a crop-origin error.
    """
    centers = np.asarray(temporal_centers_s, dtype=np.float64)
    shifts = np.asarray(shifts_um, dtype=np.float64)
    frames = np.asarray(sample_frames, dtype=np.int64)
    if centers.ndim != 1 or shifts.shape != centers.shape or centers.size < 2:
        raise ValueError("centers and shifts must be equal-length 1-D arrays")
    if not np.isfinite(centers).all() or not np.isfinite(shifts).all():
        raise ValueError("field values must be finite")
    steps = np.diff(centers)
    dt = float(np.median(steps))
    if not np.allclose(steps, dt, rtol=0, atol=1e-6):
        raise ValueError("frozen EN time centers must be uniform")
    if not np.isfinite(sampling_frequency_hz) or sampling_frequency_hz <= 0:
        raise ValueError("sampling frequency must be positive")
    global_frames = frames + int(sample_frame_origin)
    times = global_frames.astype(np.float64) / float(sampling_frequency_hz)
    # build_adapter supplies centers +/- dt/2 as explicit edges. SI uses
    # half-open cells, so a sample exactly on an interior edge selects the cell
    # to its right. Endpoint behavior is clamped to the supplied end cells.
    right_edges = centers + dt / 2
    positions = np.searchsorted(right_edges, times, side="right")
    positions = np.clip(positions, 0, centers.size - 1)
    return shifts[positions]


def support_for_shift(
    source_channel_locations_um: np.ndarray,
    shift_um: float,
    *,
    target_channel_locations_um: np.ndarray | None = None,
    direction: str = "y",
) -> np.ndarray:
    """Return targets inside SI's inclusive source bounding box.

    SpikeInterface nearest interpolation first assigns the nearest source site
    to every moved target, including non-lattice coordinates. With
    ``force_extrapolate=False`` (the implementation of ``force_zeros``), only
    targets outside the inclusive source bounding box in any geometry dimension
    are zeroed. Support is therefore a bounds property, not exact-site
    existence.
    """
    source = np.asarray(source_channel_locations_um, dtype=np.float64)
    target = source if target_channel_locations_um is None else np.asarray(target_channel_locations_um, dtype=np.float64)
    if source.ndim != 2 or source.shape[1] < 2 or target.ndim != 2 or target.shape[1] != source.shape[1]:
        raise ValueError("source/target locations must be 2-D arrays with matching dimensions")
    axis = {"x": 0, "y": 1}.get(direction)
    if axis is None:
        raise ValueError("direction must be 'x' or 'y'")
    desired = target.copy()
    desired[:, axis] += float(shift_um)
    lower = np.min(source, axis=0)
    upper = np.max(source, axis=0)
    return np.all((desired >= lower) & (desired <= upper), axis=1)


def support_mask_from_geometry_and_field(
    channel_locations_um: np.ndarray,
    temporal_centers_s: np.ndarray,
    shifts_um: np.ndarray,
    sample_frames: np.ndarray,
    sampling_frequency_hz: float,
    *,
    sample_frame_origin: int = 0,
    direction: str = "y",
) -> np.ndarray:
    """Build a channel-by-sample mask without inspecting voltage amplitudes."""
    states = si_stepwise_states(
        temporal_centers_s, shifts_um, sample_frames, sampling_frequency_hz,
        sample_frame_origin=sample_frame_origin,
    )
    geometry = np.asarray(channel_locations_um, dtype=np.float64)
    support = np.empty((geometry.shape[0], states.size), dtype=bool)
    for state in np.unique(states):
        columns = states == state
        support[:, columns] = support_for_shift(
            geometry, float(state), direction=direction
        )[:, None]
    return support


def filter_then_mask_before_whitening(
    bfile,
    raw_batch: torch.Tensor,
    *,
    ops: dict,
    ibatch: int,
    global_start_frame: int,
    channel_locations_um: np.ndarray,
    temporal_centers_s: np.ndarray,
    shifts_um: np.ndarray,
    direction: str = "y",
) -> SupportMaskedBatch:
    """Run native pre-W filtering, mask unsupported entries, then apply Wrot.

    This function is opt-in and mutates no persistent sorter configuration. It
    restores ``bfile.whiten_mat`` even if native filtering raises.
    """
    if getattr(bfile, "dshift", None) is not None:
        raise ValueError("candidate requires do_correction=False / dshift=None")
    whitening = getattr(bfile, "whiten_mat", None)
    if whitening is None:
        raise ValueError("native whitening matrix is required")
    if raw_batch.ndim != 2:
        raise ValueError("raw_batch must be channels by samples")
    bfile.whiten_mat = None
    try:
        prewhite = bfile.filter(raw_batch.clone(), ops=ops, ibatch=int(ibatch))
    finally:
        bfile.whiten_mat = whitening
    if prewhite.shape[0] != len(channel_locations_um):
        raise ValueError("post-chanMap rows differ from supplied geometry")
    frames = np.arange(
        int(global_start_frame), int(global_start_frame) + prewhite.shape[1], dtype=np.int64
    )
    support_np = support_mask_from_geometry_and_field(
        channel_locations_um,
        temporal_centers_s,
        shifts_um,
        frames,
        float(ops["fs"]),
        direction=direction,
    )
    support = torch.from_numpy(support_np).to(device=prewhite.device)
    candidate_prewhite = prewhite.masked_fill(~support, 0)
    baseline_whitened = whitening @ prewhite
    candidate_whitened = whitening @ candidate_prewhite
    return SupportMaskedBatch(
        prewhite=prewhite,
        support=support,
        baseline_whitened=baseline_whitened,
        candidate_prewhite=candidate_prewhite,
        candidate_whitened=candidate_whitened,
    )
