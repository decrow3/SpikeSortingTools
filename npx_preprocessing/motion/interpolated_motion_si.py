"""Lazy production-order bad-channel interpolation plus SI motion remapping."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from .lattice_remap_si import _per_segment_arrays, sample_stepwise_shifts

try:
    from spikeinterface.core import BaseRecording, BaseRecordingSegment
except ImportError:  # pragma: no cover - import guard for lightweight environments
    BaseRecording = BaseRecordingSegment = None


def bad_channel_insertion(
    parent_channel_ids: Sequence[Any],
    full_channel_ids: Sequence[Any],
    parent_locations: np.ndarray,
    full_locations: np.ndarray,
    *,
    bad_channel_id: Any,
    sigma_um: float,
    p: float = 1.3,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a parent-to-full insertion matrix and the bad-channel weights."""
    from spikeinterface.preprocessing.preprocessing_tools import get_kriging_channel_weights

    parent_ids = list(parent_channel_ids)
    full_ids = list(full_channel_ids)
    if len(parent_ids) != len(set(parent_ids)) or len(full_ids) != len(set(full_ids)):
        raise ValueError("channel IDs must be unique")
    if bad_channel_id in parent_ids or bad_channel_id not in full_ids:
        raise ValueError("bad channel must be absent from parent and present in full lattice")
    expected = set(full_ids) - {bad_channel_id}
    if set(parent_ids) != expected:
        raise ValueError("parent channels must equal the full lattice minus the bad channel")
    parent_locations = np.asarray(parent_locations, dtype=np.float64)
    full_locations = np.asarray(full_locations, dtype=np.float64)
    if parent_locations.shape != (len(parent_ids), 2) or full_locations.shape != (len(full_ids), 2):
        raise ValueError("locations must be two-dimensional and match channel IDs")
    insertion = np.zeros((len(parent_ids), len(full_ids)), dtype=np.float32)
    parent_lookup = {channel_id: index for index, channel_id in enumerate(parent_ids)}
    for full_index, channel_id in enumerate(full_ids):
        if channel_id != bad_channel_id:
            insertion[parent_lookup[channel_id], full_index] = 1.0
    bad_index = full_ids.index(bad_channel_id)
    weights = get_kriging_channel_weights(
        parent_locations,
        full_locations[[bad_index]],
        sigma_um=float(sigma_um),
        p=float(p),
    )[:, 0].astype(np.float32)
    if not np.isclose(weights.sum(), 1.0, rtol=0, atol=1e-6):
        raise RuntimeError("bad-channel interpolation weights do not sum to one")
    insertion[:, bad_index] = weights
    return insertion, weights


if BaseRecording is not None:

    class _InterpolatedMotionSegment(BaseRecordingSegment):
        def __init__(
            self,
            parent_segment,
            *,
            sampling_frequency: float,
            time_origin_s: float,
            temporal_bin_centers_s: np.ndarray,
            displacements_um: np.ndarray,
            cell_width_s: float,
            insertion: np.ndarray,
            full_locations: np.ndarray,
            target_locations: np.ndarray,
            method: str,
            kernel_kwargs: dict[str, object],
        ) -> None:
            super().__init__(sampling_frequency=sampling_frequency, t_start=0.0)
            self.parent_segment = parent_segment
            self.time_origin_s = float(time_origin_s)
            self.temporal_bin_centers_s = temporal_bin_centers_s
            self.displacements_um = displacements_um
            self.cell_width_s = float(cell_width_s)
            self.insertion = insertion
            self.full_locations = full_locations
            self.target_locations = target_locations
            self.method = method
            self.kernel_kwargs = kernel_kwargs
            self._kernel_cache: dict[float, np.ndarray] = {}

        def get_num_samples(self) -> int:
            return int(self.parent_segment.get_num_samples())

        def _kernel(self, displacement_um: float) -> np.ndarray:
            key = float(displacement_um)
            cached = self._kernel_cache.get(key)
            if cached is not None:
                return cached
            from spikeinterface.preprocessing import get_spatial_interpolation_kernel

            moved = self.target_locations.copy()
            moved[:, 1] += key
            full_kernel = get_spatial_interpolation_kernel(
                self.full_locations,
                moved,
                method=self.method,
                **self.kernel_kwargs,
            )
            effective = (self.insertion @ full_kernel).astype(np.float32, copy=False)
            # Duplicate calculation during a first concurrent read is benign;
            # avoiding a lock keeps the extractor multiprocessing-picklable.
            return self._kernel_cache.setdefault(key, effective)

        def get_traces(self, start_frame, end_frame, channel_indices):
            start = 0 if start_frame is None else int(start_frame)
            end = self.get_num_samples() if end_frame is None else int(end_frame)
            times = self.time_origin_s + np.arange(start, end) / self.sampling_frequency
            states = sample_stepwise_shifts(
                self.temporal_bin_centers_s,
                self.displacements_um,
                times,
                cell_width_s=self.cell_width_s,
            )
            source = self.parent_segment.get_traces(start, end, slice(None)).astype(
                np.float32, copy=False
            )
            output = np.empty((end - start, self.target_locations.shape[0]), dtype=np.float32)
            for state in np.unique(states):
                rows = states == state
                output[rows] = source[rows] @ self._kernel(float(state))
            if channel_indices is None:
                return output
            return output[:, channel_indices]


    class InterpolatedMotionRecording(BaseRecording):
        """Interpolate one missing channel, remap motion, then expose target sites.

        Motion is stepwise on half-open cells. At each state the spatial kernel
        comes from SpikeInterface, and ``force_extrapolate=False`` gives the
        production ``force_zeros`` border behavior used by EM.1/EM.2a.
        """

        def __init__(
            self,
            recording,
            temporal_bin_centers_s,
            displacements_um,
            *,
            full_channel_ids: Sequence[Any],
            full_channel_locations: np.ndarray,
            target_channel_ids: Sequence[Any],
            bad_channel_id: Any,
            cell_width_s: float = 0.25,
            method: str = "kriging",
            sigma_um: float = 20.0,
            p: float = 1.0,
            num_closest: int = 4,
            bad_channel_sigma_um: float = 20.0,
            bad_channel_p: float = 1.3,
            time_origins_s: Sequence[float] | None = None,
        ) -> None:
            if recording.get_num_segments() < 1:
                raise ValueError("recording needs at least one segment")
            parent_ids = recording.get_channel_ids().tolist()
            full_ids = list(full_channel_ids)
            target_ids = list(target_channel_ids)
            if not set(target_ids).issubset(set(full_ids)):
                raise ValueError("target channels must belong to the full lattice")
            parent_locations = np.asarray(recording.get_channel_locations(), dtype=np.float64)[:, :2]
            full_locations = np.asarray(full_channel_locations, dtype=np.float64)[:, :2]
            insertion, weights = bad_channel_insertion(
                parent_ids,
                full_ids,
                parent_locations,
                full_locations,
                bad_channel_id=bad_channel_id,
                sigma_um=bad_channel_sigma_um,
                p=bad_channel_p,
            )
            full_lookup = {channel_id: index for index, channel_id in enumerate(full_ids)}
            target_locations = full_locations[[full_lookup[channel_id] for channel_id in target_ids]]
            segments = recording.get_num_segments()
            centers_by_segment = _per_segment_arrays(
                temporal_bin_centers_s, segments, "temporal_bin_centers_s"
            )
            displacement_by_segment = _per_segment_arrays(
                displacements_um, segments, "displacements_um"
            )
            origins = (
                [float(recording.sample_index_to_time(0, segment_index=index)) for index in range(segments)]
                if time_origins_s is None
                else [float(value) for value in time_origins_s]
            )
            if len(origins) != segments:
                raise ValueError("time_origins_s must have one value per segment")
            if method not in {"nearest", "idw", "kriging"}:
                raise ValueError("method must be nearest, idw, or kriging")
            kernel_kwargs: dict[str, object] = {
                "dtype": "float32",
                "force_extrapolate": False,
            }
            if method == "idw":
                kernel_kwargs["num_closest"] = int(num_closest)
            elif method == "kriging":
                kernel_kwargs.update(sigma_um=float(sigma_um), p=float(p), sparse_thresh=None)

            super().__init__(
                sampling_frequency=float(recording.get_sampling_frequency()),
                channel_ids=target_ids,
                dtype=np.dtype("float32"),
            )
            # Production targets are supported parent channels. Keep the class
            # usable for fixtures that also expose the reconstructed bad site:
            # copy a property only when every target has a parent value.
            if set(target_ids).issubset(set(parent_ids)):
                indices = recording.ids_to_indices(target_ids)
                for key in recording.get_property_keys():
                    if key != "location":
                        self.set_property(key, recording.get_property(key)[indices])
            self.set_channel_locations(target_locations)
            for index, parent_segment in enumerate(recording._recording_segments):
                centers = np.asarray(centers_by_segment[index], dtype=np.float64)
                values = np.asarray(displacement_by_segment[index], dtype=np.float64)
                sample_stepwise_shifts(centers, values, centers, cell_width_s=cell_width_s)
                self.add_recording_segment(
                    _InterpolatedMotionSegment(
                        parent_segment,
                        sampling_frequency=recording.get_sampling_frequency(),
                        time_origin_s=origins[index],
                        temporal_bin_centers_s=centers,
                        displacements_um=values,
                        cell_width_s=cell_width_s,
                        insertion=insertion,
                        full_locations=full_locations,
                        target_locations=target_locations,
                        method=method,
                        kernel_kwargs=kernel_kwargs,
                    )
                )
            self.annotate(
                is_filtered=recording.is_filtered(),
                em2_bad_channel_id=str(bad_channel_id),
                em2_bad_channel_weight_nonzero=int(np.count_nonzero(weights)),
                em2_motion_method=method,
            )
            self._kwargs = {
                "recording": recording,
                "temporal_bin_centers_s": [value.tolist() for value in centers_by_segment],
                "displacements_um": [value.tolist() for value in displacement_by_segment],
                "full_channel_ids": full_ids,
                "full_channel_locations": full_locations.tolist(),
                "target_channel_ids": target_ids,
                "bad_channel_id": bad_channel_id,
                "cell_width_s": float(cell_width_s),
                "method": method,
                "sigma_um": float(sigma_um),
                "p": float(p),
                "num_closest": int(num_closest),
                "bad_channel_sigma_um": float(bad_channel_sigma_um),
                "bad_channel_p": float(bad_channel_p),
                "time_origins_s": origins,
            }

else:

    class InterpolatedMotionRecording:  # pragma: no cover - environment guard
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("InterpolatedMotionRecording requires spikeinterface")
