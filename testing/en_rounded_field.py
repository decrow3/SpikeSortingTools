"""Frozen helpers for EN full-probe rounded rigid voltage remapping."""

from __future__ import annotations

import numpy as np


def round_half_away(values: np.ndarray) -> np.ndarray:
    """Round halves away from zero, unlike NumPy's ties-to-even rule."""
    values = np.asarray(values, dtype=np.float64)
    return np.copysign(np.floor(np.abs(values) + 0.5), values)


def rounded_rigid_field(displacement_um: np.ndarray, period_um: float = 40.0) -> tuple[np.ndarray, float]:
    """Median-reference and lattice-round one frozen full-session field."""
    displacement_um = np.asarray(displacement_um, dtype=np.float64)
    if displacement_um.ndim != 1 or displacement_um.size < 2 or not np.all(np.isfinite(displacement_um)):
        raise ValueError("rigid displacement must be a finite vector with at least two samples")
    reference_um = float(np.median(displacement_um))
    rounded = period_um * round_half_away((displacement_um - reference_um) / period_um)
    return rounded.astype(np.float64), reference_um


def build_adapter(recording, temporal_bins_s: np.ndarray, displacement_um: np.ndarray):
    """Build the frozen SI nearest/force-zeros adapter and checked int16 cast.

    SI assigns trace samples to half-open cells around ``temporal_bins_s`` and
    evaluates the Motion exactly at the selected bin center. Thus rounded knot
    values remain piecewise constant rather than being linearly mixed in time.
    """
    from spikeinterface.core.motion import Motion
    from spikeinterface.preprocessing import astype
    from spikeinterface.sortingcomponents.motion import InterpolateMotionRecording
    from testing.checked_int16_recording import CheckedInt16Recording

    temporal_bins_s = np.asarray(temporal_bins_s, dtype=np.float64)
    displacement_um = np.asarray(displacement_um, dtype=np.float64)
    if temporal_bins_s.ndim != 1 or displacement_um.shape != temporal_bins_s.shape:
        raise ValueError("time and rigid displacement must be equal-length vectors")
    if temporal_bins_s.size < 3:
        raise ValueError("SI 0.102.1 requires at least three centers for initialized bin edges")
    if not np.all(np.diff(temporal_bins_s) > 0):
        raise ValueError("time bins must be strictly increasing")
    steps = np.diff(temporal_bins_s)
    if not np.allclose(steps, steps[0], rtol=0, atol=1e-6):
        raise ValueError("frozen EN adapter requires a uniform time grid")
    origin_s = float(recording.sample_index_to_time(0))
    # SI 0.102.1's recording segment forwards only bin edges, then rebuilds
    # centers from them. Motion's default endpoint-clamped edges would move the
    # first and last evaluation by dt/4. Explicit extrapolated edges make every
    # rebuilt center exactly equal to the frozen field center.
    dt = float(steps[0])
    edges = np.r_[temporal_bins_s - dt / 2, temporal_bins_s[-1] + dt / 2] + origin_s
    motion = Motion(
        displacement=[displacement_um[:, None]],
        temporal_bins_s=[temporal_bins_s + origin_s],
        spatial_bins_um=np.asarray([np.mean(recording.get_channel_locations()[:, 1])]),
        direction="y",
        interpolation_method="linear",
    )
    corrected = InterpolateMotionRecording(
        astype(recording, "float32"),
        motion,
        border_mode="force_zeros",
        spatial_interpolation_method="nearest",
        interpolation_time_bin_edges_s=[edges],
        dtype="float32",
    )
    return CheckedInt16Recording(corrected)


def zero_filled_channels(channel_locations: np.ndarray, displacement_um: float) -> int:
    """Count full-probe target sites outside source support at one rigid state."""
    locations = np.asarray(channel_locations, dtype=np.float64)
    moved_y = locations[:, 1] + float(displacement_um)
    return int(np.count_nonzero((moved_y < locations[:, 1].min()) | (moved_y > locations[:, 1].max())))


def zero_fill_channel_seconds(
    temporal_bins_s: np.ndarray,
    displacement_um: np.ndarray,
    duration_s: float,
    channel_locations: np.ndarray,
) -> tuple[float, float]:
    """Integrate zero-filled channel count over SI's clipped half-open cells."""
    temporal_bins_s = np.asarray(temporal_bins_s, dtype=np.float64)
    displacement_um = np.asarray(displacement_um, dtype=np.float64)
    edges = np.empty(temporal_bins_s.size + 1, dtype=np.float64)
    edges[1:-1] = (temporal_bins_s[:-1] + temporal_bins_s[1:]) / 2
    edges[0], edges[-1] = 0.0, float(duration_s)
    edges = np.clip(edges, 0.0, duration_s)
    widths = np.maximum(0.0, np.diff(edges))
    counts = np.asarray([zero_filled_channels(channel_locations, q) for q in displacement_um])
    channel_seconds = float(np.dot(widths, counts))
    denominator = float(duration_s * len(channel_locations))
    return channel_seconds, channel_seconds / denominator
