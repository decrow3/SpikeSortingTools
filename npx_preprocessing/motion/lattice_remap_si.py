"""Exact-coordinate, whole-lattice remapping for SpikeInterface recordings.

SpikeInterface's ``nearest`` spatial interpolation and DD's lattice remap are
different operators when an expected source coordinate is absent.  Nearest
selects another recorded site; DD emits zero.  This module implements the DD
operator as a lazy SpikeInterface preprocessor while keeping the geometry and
sample-clock primitives usable without SpikeInterface installed.

The displacement passed here is the coordinate lookup shift: output channel
``(x, y)`` reads source channel ``(x, y + shift)``.  Callers must resolve the
scientific motion sign before constructing this operator.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

try:
    from spikeinterface.preprocessing.basepreprocessor import (
        BasePreprocessor,
        BasePreprocessorSegment,
    )
except ImportError:  # Pure geometry/time helpers remain importable without SI.
    BasePreprocessor = None  # type: ignore[assignment]
    BasePreprocessorSegment = object  # type: ignore[assignment,misc]


def _geometry_2d(channel_locations: np.ndarray) -> np.ndarray:
    geometry = np.asarray(channel_locations, dtype=np.float64)
    if geometry.ndim != 2 or geometry.shape[1] < 2:
        raise ValueError("channel_locations must have shape (channels, >=2)")
    if not np.isfinite(geometry[:, :2]).all():
        raise ValueError("channel_locations contain nonfinite x/y coordinates")
    return geometry[:, :2]


def exact_coordinate_mapping(
    source_locations: np.ndarray,
    shift_um: float,
    *,
    target_locations: np.ndarray | None = None,
    direction: str = "y",
    atol_um: float = 1e-6,
) -> np.ndarray:
    """Map each output site to the exactly translated source, or ``-1``.

    Matching uses an absolute coordinate tolerance only to absorb serialized
    floating-point geometry noise.  Ambiguous matches are rejected rather than
    resolved by channel order.
    """

    if direction not in ("x", "y"):
        raise ValueError("direction must be 'x' or 'y'")
    if not np.isfinite(shift_um):
        raise ValueError("shift_um must be finite")
    if not np.isfinite(atol_um) or atol_um < 0:
        raise ValueError("atol_um must be finite and nonnegative")

    source = _geometry_2d(source_locations)
    target_geometry = source if target_locations is None else _geometry_2d(target_locations)
    moved = target_geometry.copy()
    moved[:, 0 if direction == "x" else 1] += float(shift_um)
    mapping = np.full(target_geometry.shape[0], -1, dtype=np.int64)
    for target, desired in enumerate(moved):
        matches = np.flatnonzero(np.all(np.abs(source - desired) <= atol_um, axis=1))
        if matches.size > 1:
            raise ValueError(
                f"ambiguous exact source for target {target} at {desired.tolist()}"
            )
        if matches.size == 1:
            mapping[target] = int(matches[0])
    valid = mapping >= 0
    if np.unique(mapping[valid]).size != int(valid.sum()):
        raise ValueError("translation produced a non-injective source mapping")
    return mapping


def nearest_coordinate_mapping(
    source_locations: np.ndarray,
    shift_um: float,
    *,
    target_locations: np.ndarray | None = None,
    direction: str = "y",
    tie_atol_um: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return unconstrained 2-D nearest sources, distances, and tie counts.

    This describes the spatial argmin itself, not SpikeInterface border-mode
    masking.  The installed-version audit must test border behavior separately.
    """

    if direction not in ("x", "y"):
        raise ValueError("direction must be 'x' or 'y'")
    source = _geometry_2d(source_locations)
    target_geometry = source if target_locations is None else _geometry_2d(target_locations)
    moved = target_geometry.copy()
    moved[:, 0 if direction == "x" else 1] += float(shift_um)
    distance = np.linalg.norm(moved[:, None, :] - source[None, :, :], axis=2)
    mapping = np.argmin(distance, axis=1).astype(np.int64)
    minimum = distance[np.arange(distance.shape[0]), mapping]
    ties = np.sum(np.isclose(distance, minimum[:, None], rtol=0, atol=tie_atol_um), axis=1)
    return mapping, minimum, ties.astype(np.int64)


def audit_lattice_mappings(
    source_locations: np.ndarray,
    shifts_um: Sequence[float] | np.ndarray,
    *,
    target_locations: np.ndarray | None = None,
    direction: str = "y",
    atol_um: float = 1e-6,
) -> list[dict[str, Any]]:
    """Compare exact-coordinate and unconstrained-nearest maps row by row."""

    source = _geometry_2d(source_locations)
    target_geometry = source if target_locations is None else _geometry_2d(target_locations)
    axis = 0 if direction == "x" else 1
    lo = float(np.min(source[:, axis]))
    hi = float(np.max(source[:, axis]))
    rows: list[dict[str, Any]] = []
    for shift in np.unique(np.asarray(shifts_um, dtype=np.float64)):
        exact = exact_coordinate_mapping(
            source,
            float(shift),
            target_locations=target_geometry,
            direction=direction,
            atol_um=atol_um,
        )
        nearest, distances, ties = nearest_coordinate_mapping(
            source, float(shift), target_locations=target_geometry, direction=direction
        )
        desired_axis = target_geometry[:, axis] + float(shift)
        for target in range(target_geometry.shape[0]):
            if exact[target] >= 0:
                support_class = "exact"
            elif desired_axis[target] < lo - atol_um or desired_axis[target] > hi + atol_um:
                support_class = "outer_missing"
            else:
                support_class = "interior_missing"
            if ties[target] > 1:
                support_class = f"{support_class}_tie"
            rows.append(
                {
                    "shift_um": float(shift),
                    "target_index": target,
                    "target_x_um": float(target_geometry[target, 0]),
                    "target_y_um": float(target_geometry[target, 1]),
                    "exact_source_index": int(exact[target]),
                    "nearest_source_index": int(nearest[target]),
                    "nearest_distance_um": float(distances[target]),
                    "nearest_tie_count": int(ties[target]),
                    "support_class": support_class,
                    "nearest_matches_exact": bool(
                        exact[target] >= 0 and nearest[target] == exact[target]
                    ),
                }
            )
    return rows


def audit_spikeinterface_nearest_kernel(
    source_locations: np.ndarray,
    shifts_um: Sequence[float] | np.ndarray,
    *,
    target_locations: np.ndarray | None = None,
    direction: str = "y",
    atol_um: float = 1e-6,
) -> list[dict[str, Any]]:
    """Audit the installed SI nearest kernel, including its border mask."""

    try:
        from spikeinterface.sortingcomponents.motion.motion_interpolation import (
            get_spatial_interpolation_kernel,
        )
    except ImportError as exc:  # pragma: no cover - environment guard
        raise ImportError("installed SpikeInterface motion interpolation is required") from exc

    source = _geometry_2d(source_locations)
    target_geometry = source if target_locations is None else _geometry_2d(target_locations)
    axis = 0 if direction == "x" else 1
    rows: list[dict[str, Any]] = []
    for shift in np.unique(np.asarray(shifts_um, dtype=np.float64)):
        exact = exact_coordinate_mapping(
            source,
            float(shift),
            target_locations=target_geometry,
            direction=direction,
            atol_um=atol_um,
        )
        moved = target_geometry.copy()
        moved[:, axis] += float(shift)
        kernel = get_spatial_interpolation_kernel(
            source,
            moved,
            method="nearest",
            dtype="float32",
            force_extrapolate=False,
        )
        if kernel.shape != (source.shape[0], target_geometry.shape[0]):
            raise RuntimeError(f"unexpected nearest-kernel shape {kernel.shape}")
        nonzero = np.count_nonzero(kernel, axis=0)
        for target in range(target_geometry.shape[0]):
            stock_source = -1 if nonzero[target] == 0 else int(np.argmax(kernel[:, target]))
            rows.append(
                {
                    "shift_um": float(shift),
                    "target_index": target,
                    "exact_source_index": int(exact[target]),
                    "stock_source_index": stock_source,
                    "stock_nonzero_weights": int(nonzero[target]),
                    "stock_weight_sum": float(np.sum(kernel[:, target])),
                    "stock_matches_exact": bool(stock_source == exact[target]),
                }
            )
    return rows


def write_mapping_audit(
    output_directory: str | Path,
    source_locations: np.ndarray,
    shifts_um: Sequence[float] | np.ndarray,
    *,
    target_locations: np.ndarray | None = None,
    direction: str = "y",
    atol_um: float = 1e-6,
    include_spikeinterface: bool = False,
) -> dict[str, Any]:
    """Write the compact geometry-only EM audit as CSV plus summary JSON."""

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=False)
    source = _geometry_2d(source_locations)
    target_geometry = source if target_locations is None else _geometry_2d(target_locations)
    shifts = np.unique(np.asarray(shifts_um, dtype=np.float64))
    if shifts.size == 0:
        raise ValueError("at least one shift is required")
    rows = audit_lattice_mappings(
        source,
        shifts,
        target_locations=target_geometry,
        direction=direction,
        atol_um=atol_um,
    )
    fieldnames = list(rows[0]) if rows else []
    with (output / "MAPPING_AUDIT.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    support_counts: dict[str, int] = {}
    for row in rows:
        key = str(row["support_class"])
        support_counts[key] = support_counts.get(key, 0) + 1
    exact_rows = [row for row in rows if str(row["support_class"]).startswith("exact")]
    unsupported_rows = [row for row in rows if row["exact_source_index"] == -1]
    summary: dict[str, Any] = {
        "schema": "em-lattice-mapping-audit-v1",
        "direction": direction,
        "atol_um": float(atol_um),
        "num_source_channels": int(source.shape[0]),
        "num_target_channels": int(target_geometry.shape[0]),
        "shifts_um": shifts.tolist(),
        "source_geometry_sha256": hashlib.sha256(source.tobytes()).hexdigest(),
        "target_geometry_sha256": hashlib.sha256(target_geometry.tobytes()).hexdigest(),
        "row_count": len(rows),
        "support_counts": support_counts,
        "exact_rows_all_match_nearest": all(
            bool(row["nearest_matches_exact"]) for row in exact_rows
        ),
        "unsupported_target_count": len(unsupported_rows),
        "unsupported_targets_nearest_would_substitute": sum(
            int(row["nearest_source_index"] >= 0) for row in unsupported_rows
        ),
        "interpretation": (
            "Geometry-only comparison of DD exact-coordinate lookup with the "
            "unconstrained nearest argmin. Installed SpikeInterface border behavior "
            "must be audited separately."
        ),
    }
    if include_spikeinterface:
        import spikeinterface

        stock_rows = audit_spikeinterface_nearest_kernel(
            source,
            shifts,
            target_locations=target_geometry,
            direction=direction,
            atol_um=atol_um,
        )
        with (output / "STOCK_NEAREST_AUDIT.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(stock_rows[0]))
            writer.writeheader()
            writer.writerows(stock_rows)
        mismatches = [row for row in stock_rows if not row["stock_matches_exact"]]
        summary["spikeinterface_version"] = spikeinterface.__version__
        summary["stock_nearest_full_equivalence"] = not mismatches
        summary["stock_nearest_mismatch_count"] = len(mismatches)
        summary["stock_nearest_substitution_count"] = sum(
            int(row["exact_source_index"] == -1 and row["stock_source_index"] >= 0)
            for row in mismatches
        )
        summary["stock_nearest_false_zero_count"] = sum(
            int(row["exact_source_index"] >= 0 and row["stock_source_index"] == -1)
            for row in mismatches
        )
    (output / "SUMMARY.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary


def sample_stepwise_shifts(
    temporal_bin_centers_s: np.ndarray,
    shifts_um: np.ndarray,
    sample_times_s: np.ndarray,
    *,
    cell_width_s: float,
) -> np.ndarray:
    """Sample DD-style half-open motion cells at exact sample times.

    Cells are ``[center - width/2, center + width/2)`` and an exact shared
    boundary belongs to the later cell.  The centers must be contiguous and
    uniformly spaced.  Times before the first right edge select the first cell,
    matching DD's executed ``searchsorted`` rule; times at or beyond the final
    right edge are rejected as unsupported.
    """

    centers = np.asarray(temporal_bin_centers_s, dtype=np.float64)
    shifts = np.asarray(shifts_um, dtype=np.float64)
    times = np.asarray(sample_times_s, dtype=np.float64)
    if centers.ndim != 1 or shifts.ndim != 1 or centers.size != shifts.size:
        raise ValueError("temporal centers and shifts must be equal-length 1-D arrays")
    if centers.size == 0:
        raise ValueError("at least one temporal bin is required")
    if not np.isfinite(centers).all() or not np.isfinite(shifts).all():
        raise ValueError("temporal centers and shifts must be finite")
    if not np.isfinite(times).all():
        raise ValueError("sample times must be finite")
    if not np.isfinite(cell_width_s) or cell_width_s <= 0:
        raise ValueError("cell_width_s must be finite and positive")
    if np.any(np.diff(centers) <= 0):
        raise ValueError("temporal centers must be strictly increasing")
    tolerance = max(1e-12, abs(float(cell_width_s)) * 1e-9)
    if centers.size > 1 and not np.allclose(
        np.diff(centers), cell_width_s, rtol=0, atol=tolerance
    ):
        raise ValueError("temporal centers must be contiguous at cell_width_s")
    right_edges = centers + float(cell_width_s) / 2.0
    positions = np.searchsorted(right_edges, times, side="right")
    if np.any(positions >= centers.size):
        first = int(np.flatnonzero(positions >= centers.size)[0])
        raise ValueError(f"motion cells do not cover sample time {times[first]:.17g}")
    return shifts[positions]


def _per_segment_arrays(value: Any, n_segments: int, name: str) -> list[np.ndarray]:
    if n_segments == 1:
        single = np.asarray(value)
        if single.ndim == 1:
            return [single]
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must provide one array per recording segment")
    arrays = [np.asarray(item) for item in value]
    if len(arrays) != n_segments:
        raise ValueError(f"{name} must provide one array per recording segment")
    return arrays


if BasePreprocessor is not None:

    class _ExactLatticeRemapSegment(BasePreprocessorSegment):
        def __init__(
            self,
            parent_recording_segment: Any,
            *,
            sampling_frequency: float,
            time_origin_s: float,
            temporal_bin_centers_s: np.ndarray,
            shifts_um: np.ndarray,
            cell_width_s: float,
            mappings: dict[float, np.ndarray],
            num_channels: int,
        ) -> None:
            super().__init__(parent_recording_segment)
            self.sampling_frequency = float(sampling_frequency)
            self.time_origin_s = float(time_origin_s)
            self.temporal_bin_centers_s = temporal_bin_centers_s
            self.shifts_um = shifts_um
            self.cell_width_s = float(cell_width_s)
            self.mappings = mappings
            self.num_channels = int(num_channels)

        def get_traces(self, start_frame, end_frame, channel_indices):
            start = 0 if start_frame is None else int(start_frame)
            if end_frame is None:
                end = int(self.parent_recording_segment.get_num_samples())
            else:
                end = int(end_frame)
            sample_times = self.time_origin_s + np.arange(start, end) / self.sampling_frequency
            sampled = sample_stepwise_shifts(
                self.temporal_bin_centers_s,
                self.shifts_um,
                sample_times,
                cell_width_s=self.cell_width_s,
            )
            source = self.parent_recording_segment.get_traces(start, end, slice(None))
            output = np.zeros((end - start, self.num_channels), dtype=source.dtype)
            for shift in np.unique(sampled):
                mapping = self.mappings[float(shift)]
                rows = np.flatnonzero(sampled == shift)
                valid_targets = np.flatnonzero(mapping >= 0)
                if rows.size and valid_targets.size:
                    output[np.ix_(rows, valid_targets)] = source[
                        np.ix_(rows, mapping[valid_targets])
                    ]
            if channel_indices is None:
                return output
            return output[:, channel_indices]


    class ExactLatticeRemapRecording(BasePreprocessor):
        """Lazy exact-coordinate remap with zero fill for unsupported sites."""

        def __init__(
            self,
            recording,
            temporal_bin_centers_s,
            shifts_um,
            *,
            cell_width_s: float,
            direction: str = "y",
            atol_um: float = 1e-6,
            time_origins_s: Sequence[float] | None = None,
        ) -> None:
            n_segments = recording.get_num_segments()
            centers_by_segment = _per_segment_arrays(
                temporal_bin_centers_s, n_segments, "temporal_bin_centers_s"
            )
            shifts_by_segment = _per_segment_arrays(shifts_um, n_segments, "shifts_um")
            if time_origins_s is None:
                origins = [
                    float(recording.sample_index_to_time(0, segment_index=i))
                    for i in range(n_segments)
                ]
            else:
                origins = [float(value) for value in time_origins_s]
                if len(origins) != n_segments:
                    raise ValueError("time_origins_s must have one value per segment")

            geometry = _geometry_2d(recording.get_channel_locations())
            unique_shifts = np.unique(np.concatenate(shifts_by_segment).astype(np.float64))
            mappings = {
                float(shift): exact_coordinate_mapping(
                    geometry, float(shift), direction=direction, atol_um=atol_um
                )
                for shift in unique_shifts
            }

            super().__init__(recording, dtype=recording.get_dtype())
            for segment_index, parent_segment in enumerate(recording._recording_segments):
                centers = np.asarray(centers_by_segment[segment_index], dtype=np.float64)
                segment_shifts = np.asarray(shifts_by_segment[segment_index], dtype=np.float64)
                # Validate the time table at construction, including contiguity.
                sample_stepwise_shifts(
                    centers,
                    segment_shifts,
                    centers,
                    cell_width_s=cell_width_s,
                )
                self.add_recording_segment(
                    _ExactLatticeRemapSegment(
                        parent_segment,
                        sampling_frequency=recording.get_sampling_frequency(),
                        time_origin_s=origins[segment_index],
                        temporal_bin_centers_s=centers,
                        shifts_um=segment_shifts,
                        cell_width_s=cell_width_s,
                        mappings=mappings,
                        num_channels=recording.get_num_channels(),
                    )
                )
            self._kwargs = {
                "recording": recording,
                "temporal_bin_centers_s": [value.tolist() for value in centers_by_segment],
                "shifts_um": [value.tolist() for value in shifts_by_segment],
                "cell_width_s": float(cell_width_s),
                "direction": direction,
                "atol_um": float(atol_um),
                "time_origins_s": origins,
            }

else:

    class ExactLatticeRemapRecording:  # pragma: no cover - environment guard
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("ExactLatticeRemapRecording requires spikeinterface")
