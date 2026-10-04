"""Deterministic retained-pair correlation with frozen failure precedence."""

from __future__ import annotations

from typing import Any

import numpy as np


REASON_PRIORITY = (
    "MISSING_ENDPOINT",
    "ENDPOINT_NOT_SELECTED",
    "ENDPOINT_NOT_TECHNICALLY_EVALUABLE",
    "ALIGNMENT_FAILURE",
    "INSUFFICIENT_COMMON_CHANNELS",
    "INSUFFICIENT_COMMON_FRAMES",
    "NONFINITE_VALUES",
    "ZERO_VARIANCE",
)


def _unmeasured(reason: str, *, common_channel_n: int = 0, common_frame_n: int = 0) -> dict[str, Any]:
    if reason not in REASON_PRIORITY:
        raise ValueError(f"unknown correlation reason {reason}")
    return {
        "status": "UNMEASURED",
        "reason": reason,
        "common_channel_n": int(common_channel_n),
        "common_frame_n": int(common_frame_n),
        "correlation": None,
    }


def retained_pair_correlation(
    reference_values: Any,
    candidate_values: Any,
    reference_channels: Any,
    candidate_channels: Any,
    reference_global_frame: Any,
    candidate_global_frame: Any,
    *,
    reference_selected: bool = True,
    candidate_selected: bool = True,
    reference_technically_evaluable: bool = True,
    candidate_technically_evaluable: bool = True,
) -> dict[str, Any]:
    """Correlate two baseline-referenced channel-by-41-sample signal windows.

    Validation follows ``REASON_PRIORITY`` exactly. No invalid sample is dropped,
    no endpoint is substituted, and channel order in the input is immaterial.
    """
    required = (
        reference_values,
        candidate_values,
        reference_channels,
        candidate_channels,
        reference_global_frame,
        candidate_global_frame,
    )
    if any(value is None for value in required):
        return _unmeasured("MISSING_ENDPOINT")
    if not reference_selected or not candidate_selected:
        return _unmeasured("ENDPOINT_NOT_SELECTED")
    if not reference_technically_evaluable or not candidate_technically_evaluable:
        return _unmeasured("ENDPOINT_NOT_TECHNICALLY_EVALUABLE")

    try:
        rv = np.asarray(reference_values)
        cv = np.asarray(candidate_values)
        rc = np.asarray(reference_channels)
        cc = np.asarray(candidate_channels)
        frames = np.asarray([reference_global_frame, candidate_global_frame])
        valid_layout = (
            rv.ndim == 2
            and cv.ndim == 2
            and rv.shape[1] == 41
            and cv.shape[1] == 41
            and rc.ndim == 1
            and cc.ndim == 1
            and len(rc) == rv.shape[0]
            and len(cc) == cv.shape[0]
            and np.issubdtype(rc.dtype, np.integer)
            and np.issubdtype(cc.dtype, np.integer)
            and np.issubdtype(frames.dtype, np.integer)
            and len(np.unique(rc)) == len(rc)
            and len(np.unique(cc)) == len(cc)
            and np.all((rc >= 0) & (rc < 384))
            and np.all((cc >= 0) & (cc < 384))
        )
    except (TypeError, ValueError):
        valid_layout = False
    if not valid_layout:
        return _unmeasured("ALIGNMENT_FAILURE")

    common_channels = np.intersect1d(rc.astype(np.int64), cc.astype(np.int64), assume_unique=True)
    common_channel_n = int(len(common_channels))
    if common_channel_n < 2:
        return _unmeasured("INSUFFICIENT_COMMON_CHANNELS", common_channel_n=common_channel_n)

    ref_frame = int(reference_global_frame)
    candidate_frame = int(candidate_global_frame)
    ref_absolute = np.arange(ref_frame - 20, ref_frame + 21, dtype=np.int64)
    candidate_absolute = np.arange(candidate_frame - 20, candidate_frame + 21, dtype=np.int64)
    common_frames = np.intersect1d(ref_absolute, candidate_absolute, assume_unique=True)
    common_frame_n = int(len(common_frames))
    if common_frame_n < 26:
        return _unmeasured(
            "INSUFFICIENT_COMMON_FRAMES",
            common_channel_n=common_channel_n,
            common_frame_n=common_frame_n,
        )

    ref_channel_index = {int(channel): index for index, channel in enumerate(rc)}
    candidate_channel_index = {int(channel): index for index, channel in enumerate(cc)}
    ref_time_index = common_frames - (ref_frame - 20)
    candidate_time_index = common_frames - (candidate_frame - 20)
    try:
        ref_vector = np.concatenate([
            np.asarray(rv[ref_channel_index[int(channel)], ref_time_index], dtype=np.float64)
            for channel in common_channels
        ])
        candidate_vector = np.concatenate([
            np.asarray(cv[candidate_channel_index[int(channel)], candidate_time_index], dtype=np.float64)
            for channel in common_channels
        ])
    except (IndexError, KeyError, TypeError, ValueError):
        return _unmeasured(
            "ALIGNMENT_FAILURE",
            common_channel_n=common_channel_n,
            common_frame_n=common_frame_n,
        )
    if not np.isfinite(ref_vector).all() or not np.isfinite(candidate_vector).all():
        return _unmeasured(
            "NONFINITE_VALUES",
            common_channel_n=common_channel_n,
            common_frame_n=common_frame_n,
        )

    ref_centered = ref_vector - ref_vector.mean()
    candidate_centered = candidate_vector - candidate_vector.mean()
    ref_norm = float(np.linalg.norm(ref_centered))
    candidate_norm = float(np.linalg.norm(candidate_centered))
    if ref_norm == 0.0 or candidate_norm == 0.0:
        return _unmeasured(
            "ZERO_VARIANCE",
            common_channel_n=common_channel_n,
            common_frame_n=common_frame_n,
        )
    correlation = float(np.dot(ref_centered, candidate_centered) / (ref_norm * candidate_norm))
    correlation = float(np.clip(correlation, -1.0, 1.0))
    return {
        "status": "MEASURED",
        "reason": None,
        "common_channel_n": common_channel_n,
        "common_frame_n": common_frame_n,
        "correlation": correlation,
    }
