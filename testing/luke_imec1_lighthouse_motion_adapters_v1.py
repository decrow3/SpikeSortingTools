"""Portable motion-field adapters for the Luke0804 imec1 lighthouse audit.

Identity tracking is deliberately absent from this module.  It only converts
already-frozen event coordinates to field predictions and preserves native
support/gap semantics.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import pickle

import numpy as np


SIGN_POLICY = "corrected_depth = observed_depth - displacement"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class MotionField:
    name: str
    time_s: np.ndarray
    displacement_um: np.ndarray
    depth_um: np.ndarray | None = None
    supported: np.ndarray | None = None
    invalid: np.ndarray | None = None
    sampling: str = "linear"
    time_tolerance_s: float | None = None
    coordinate_operator: str = "forward_reference_depth"

    def __post_init__(self) -> None:
        time = np.asarray(self.time_s, dtype=float)
        disp = np.asarray(self.displacement_um, dtype=float)
        if time.ndim != 1 or len(time) == 0 or not np.all(np.diff(time) > 0):
            raise ValueError(f"{self.name}: time axis must be finite and increasing")
        if disp.ndim == 1 and disp.shape != time.shape:
            raise ValueError(f"{self.name}: rigid field shape mismatch")
        if disp.ndim == 2:
            if self.depth_um is None or disp.shape != (len(time), len(self.depth_um)):
                raise ValueError(f"{self.name}: nonrigid field shape mismatch")
            depth = np.asarray(self.depth_um, dtype=float)
            if not np.all(np.diff(depth) > 0):
                raise ValueError(f"{self.name}: depth axis must be increasing")
        if disp.ndim not in (1, 2):
            raise ValueError(f"{self.name}: displacement must be time or time-by-depth")

    def sample(self, query_time_s: np.ndarray, query_depth_um: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return displacement and native support without endpoint clamping."""
        qt = np.asarray(query_time_s, dtype=float)
        qd = np.asarray(query_depth_um, dtype=float)
        if qt.shape != qd.shape:
            raise ValueError("query time and depth must have identical shape")
        if self.name == "zero":
            return np.zeros(qt.shape, dtype=float), np.isfinite(qt) & np.isfinite(qd)
        if self.sampling == "nearest":
            return self._sample_nearest(qt)
        return self._sample_linear(qt, qd)

    def _native_support(self) -> np.ndarray:
        support = np.ones(len(self.time_s), dtype=bool)
        if self.supported is not None:
            support &= np.asarray(self.supported, dtype=bool)
        if self.invalid is not None:
            support &= ~np.asarray(self.invalid, dtype=bool)
        if self.displacement_um.ndim == 1:
            support &= np.isfinite(self.displacement_um)
        else:
            support &= np.isfinite(self.displacement_um).all(axis=1)
        return support

    def _sample_nearest(self, qt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.displacement_um.ndim != 1:
            raise ValueError("nearest sampling is reserved for rigid LFP fields")
        right = np.clip(np.searchsorted(self.time_s, qt), 0, len(self.time_s) - 1)
        left = np.clip(right - 1, 0, len(self.time_s) - 1)
        choose_right = np.abs(self.time_s[right] - qt) < np.abs(self.time_s[left] - qt)
        index = np.where(choose_right, right, left)
        error = np.abs(self.time_s[index] - qt)
        tolerance = 0.01 if self.time_tolerance_s is None else self.time_tolerance_s
        good = self._native_support()[index] & (error <= tolerance)
        values = np.asarray(self.displacement_um, dtype=float)[index]
        return np.where(good, values, np.nan), good

    def _sample_linear(self, qt: np.ndarray, qd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        time = np.asarray(self.time_s, dtype=float)
        in_time = np.isfinite(qt) & (qt >= time[0]) & (qt <= time[-1])
        right_t = np.clip(np.searchsorted(time, qt, side="right"), 1, len(time) - 1)
        left_t = right_t - 1
        denom_t = time[right_t] - time[left_t]
        alpha_t = np.divide(qt - time[left_t], denom_t, out=np.zeros_like(qt), where=denom_t != 0)
        native = self._native_support()
        good = in_time & native[left_t] & native[right_t]
        displacement = np.asarray(self.displacement_um, dtype=float)
        if displacement.ndim == 1:
            values = displacement[left_t] * (1 - alpha_t) + displacement[right_t] * alpha_t
            return np.where(good, values, np.nan), good

        depth = np.asarray(self.depth_um, dtype=float)
        in_depth = np.isfinite(qd) & (qd >= depth[0]) & (qd <= depth[-1])
        right_d = np.clip(np.searchsorted(depth, qd, side="right"), 1, len(depth) - 1)
        left_d = right_d - 1
        denom_d = depth[right_d] - depth[left_d]
        alpha_d = np.divide(qd - depth[left_d], denom_d, out=np.zeros_like(qd), where=denom_d != 0)
        rows = np.arange(len(qt))
        lower = displacement[left_t, left_d] * (1 - alpha_t) + displacement[right_t, left_d] * alpha_t
        upper = displacement[left_t, right_d] * (1 - alpha_t) + displacement[right_t, right_d] * alpha_t
        values = lower * (1 - alpha_d) + upper * alpha_d
        good &= in_depth & np.isfinite(values)
        return np.where(good, values, np.nan), good


def zero_field() -> MotionField:
    return MotionField("zero", np.array([0.0]), np.array([0.0]))


def load_medicine(directory: Path, *, name: str = "medicine") -> MotionField:
    time = np.load(directory / "time_bins.npy").astype(float)
    depth = np.load(directory / "depth_bins.npy").astype(float)
    motion = np.load(directory / "motion.npy").astype(float)
    if motion.shape == (len(depth), len(time)):
        motion = motion.T
    return MotionField(name, time, motion, depth_um=depth)


def load_lfp(path: Path, *, name: str | None = None, crop_start_source_s: float = 0.0) -> MotionField:
    with np.load(path) as data:
        time = np.asarray(data["time_s"], dtype=float)
        # The transferred full-crop files retain the source/AP clock; normalize
        # only when their spec provides a nonzero source origin.
        time = time - crop_start_source_s
        return MotionField(
            name or path.stem,
            time,
            np.asarray(data["displacement_um"], dtype=float).reshape(-1),
            supported=np.asarray(data["supported"], dtype=bool) if "supported" in data else None,
            invalid=np.asarray(data["invalid"], dtype=bool) if "invalid" in data else None,
            sampling="nearest",
            time_tolerance_s=0.01,
        )


def load_dartsort_pickle(path: Path, *, name: str) -> MotionField:
    """Load only in the producing DARTsort environment (pickle is not portable)."""
    with path.open("rb") as stream:
        payload = pickle.load(stream)
    motion = payload.get("dredge_motion_est") if isinstance(payload, dict) else payload
    if motion is None:
        raise RuntimeError(f"{path} lacks dredge_motion_est")
    time = np.asarray(motion.time_bin_centers_s, dtype=float)
    depth = np.asarray(motion.spatial_bin_centers_um, dtype=float)
    displacement = np.asarray(motion.displacement, dtype=float).T
    return MotionField(name, time, displacement, depth_um=depth)


def rigid_projection(field: MotionField, *, name: str) -> MotionField:
    if field.displacement_um.ndim != 2:
        raise ValueError("rigid projection requires a nonrigid field")
    return MotionField(name, field.time_s, np.mean(field.displacement_um, axis=1))

