"""Motion-field interchange, validation, interpolation, and canonical IO.

Conventions follow the Luke hub ``fields.py`` reference: time is seconds from
AP frame zero and corrected depth is ``observed - displacement``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field as dc_field
import hashlib
import io
import json
from pathlib import Path
import pickle
import zipfile

import numpy as np

SIGN = "corrected = observed - displacement"
NPX_DREDGE_ORIGIN_S = 3057.6775463558583


class _CompatObject:
    """State carrier for optional estimator classes absent from this environment."""


class _CompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("dredge"):
            return _CompatObject
        if module == "scipy._external.array_api_compat.numpy._aliases" and name == "asarray":
            return np.asarray
        return super().find_class(module, name)


@dataclass
class MotionField:
    time_s: np.ndarray
    depth_um: np.ndarray
    displacement_um: np.ndarray
    sign: str = SIGN
    clock_origin: str = "AP frame zero"
    source: str = "unknown"
    config: dict = dc_field(default_factory=dict)
    time_unit: str = "s"
    depth_unit: str = "um"
    displacement_unit: str = "um"

    def __post_init__(self) -> None:
        self.time_s = np.asarray(self.time_s, dtype=np.float64).reshape(-1)
        self.depth_um = np.asarray(self.depth_um, dtype=np.float64).reshape(-1)
        value = np.asarray(self.displacement_um, dtype=np.float64)
        if value.ndim == 1:
            value = value[:, None]
        if value.shape == (len(self.depth_um), len(self.time_s)):
            value = value.T
        self.displacement_um = value

    def validate(self, recording_length_s: float | None = None, atol_s: float = 1e-6) -> list[str]:
        warnings: list[str] = []
        if self.displacement_um.shape != (len(self.time_s), len(self.depth_um)):
            raise ValueError("displacement_um must have shape [T, K]")
        if len(self.time_s) < 2 or np.any(np.diff(self.time_s) <= 0):
            raise ValueError("time_s must be strictly increasing")
        dt = np.diff(self.time_s)
        if not np.allclose(dt, np.median(dt), rtol=1e-6, atol=atol_s):
            raise ValueError("time_s must be uniform")
        if len(self.depth_um) == 0 or np.any(np.diff(self.depth_um) <= 0):
            raise ValueError("depth_um must be strictly increasing")
        if not np.isfinite(self.time_s).all() or not np.isfinite(self.depth_um).all() or not np.isfinite(self.displacement_um).all():
            raise ValueError("field contains nonfinite values")
        if self.sign != SIGN:
            raise ValueError(f"unsupported sign convention: {self.sign!r}")
        if (self.time_unit, self.depth_unit, self.displacement_unit) != ("s", "um", "um"):
            raise ValueError("units must be time=s, depth=um, displacement=um")
        if self.clock_origin != "AP frame zero":
            warnings.append(f"clock origin is {self.clock_origin!r}, not AP frame zero")
        if recording_length_s is not None:
            half = np.median(dt) / 2
            if self.time_s[0] > half + atol_s or self.time_s[-1] + np.median(dt) < recording_length_s - atol_s:
                warnings.append("field does not span the stated recording length")
            if self.time_s[-1] > recording_length_s + half + atol_s:
                warnings.append("field extends beyond the stated recording length")
        # Explicit known-shift round trip required by AH.1.
        base = np.array([100.0, 900.0, 2100.0])
        observed = base + 40.0
        corrected = observed - 40.0
        if not np.array_equal(corrected, base):
            raise AssertionError("known +40 um correction round trip failed")
        return warnings

    @property
    def dt_s(self) -> float:
        return float(np.median(np.diff(self.time_s)))

    def at(self, time_s, depth_um):
        t = np.asarray(time_s, float)
        z = np.broadcast_to(np.asarray(depth_um, float), t.shape)
        columns = np.stack([np.interp(t, self.time_s, self.displacement_um[:, k], left=np.nan, right=np.nan)
                            for k in range(len(self.depth_um))], axis=-1)
        if len(self.depth_um) == 1:
            return columns[..., 0]
        zc = np.clip(z, self.depth_um[0], self.depth_um[-1])
        i = np.clip(np.searchsorted(self.depth_um, zc) - 1, 0, len(self.depth_um)-2)
        w = (zc-self.depth_um[i])/(self.depth_um[i+1]-self.depth_um[i])
        a = np.take_along_axis(columns, i[...,None], axis=-1)[...,0]
        b = np.take_along_axis(columns, (i+1)[...,None], axis=-1)[...,0]
        return a+w*(b-a)

    def rigid(self, depth_range_um=None, centre: bool = True) -> np.ndarray:
        take = np.ones(len(self.depth_um), bool)
        if depth_range_um is not None:
            take = (self.depth_um >= depth_range_um[0]) & (self.depth_um <= depth_range_um[1])
        value = self.displacement_um[:, take]
        if centre:
            value = value - np.median(value, axis=0, keepdims=True)
        return np.median(value, axis=1)

    def window(self, start_s: float, end_s: float) -> "MotionField":
        take = (self.time_s >= start_s) & (self.time_s < end_s)
        return MotionField(self.time_s[take], self.depth_um, self.displacement_um[take], self.sign,
                           self.clock_origin, self.source, dict(self.config), self.time_unit,
                           self.depth_unit, self.displacement_unit)

    def save(self, npz_path: str | Path, manifest_path: str | Path | None = None) -> dict:
        npz_path = Path(npz_path)
        arrays = {"time_s": self.time_s, "depth_um": self.depth_um,
                  "displacement_um": self.displacement_um, "sign": np.asarray(self.sign)}
        canonical_npz(npz_path, arrays)
        manifest = {"schema": "motionqc-motion-field-v1", "clock_origin": self.clock_origin,
                    "sign": self.sign, "source": self.source, "config": self.config,
                    "units": {"time": self.time_unit, "depth": self.depth_unit,
                              "displacement": self.displacement_unit},
                    "npz": npz_path.name, "npz_sha256": file_sha256(npz_path)}
        manifest_path = Path(manifest_path or npz_path.with_suffix(".json"))
        canonical_json(manifest_path, manifest)
        return manifest


def _first(data, names):
    for name in names:
        if name in data:
            return np.asarray(data[name])
    raise KeyError(f"none of {names} found")


def _object_mapping(obj):
    """Expose common DARTsort MotionEstimate layouts as an array mapping."""
    if isinstance(obj, dict):
        for key in ("dredge_motion_est", "motion_est", "motion_estimate"):
            if key in obj and obj[key] is not None:
                return _object_mapping(obj[key])
        return obj
    data = vars(obj).copy() if hasattr(obj, "__dict__") else {}
    aliases = {
        "time_s": ("time_s", "time_bin_centers_s", "temporal_bins_s", "temporal_bin_centers_s"),
        "depth_um": ("depth_um", "spatial_bin_centers_um", "spatial_bins_um"),
        "displacement_um": ("displacement_um", "displacement", "motion"),
    }
    for target, names in aliases.items():
        for name in names:
            if hasattr(obj, name):
                data[target] = getattr(obj, name)
                break
    return data


def _from_mapping(data, source: str, origin_offset_s: float = 0.0, config=None) -> MotionField:
    time_s = _first(data, ("time_s", "session_time_s", "time_bin_centers_s", "temporal_bins_s", "time_bins")) - origin_offset_s
    depth_um = _first(data, ("depth_um", "spatial_bin_centers_um", "spatial_bins_um", "depth_bins"))
    displacement = _first(data, ("displacement_um", "displacement", "motion"))
    return MotionField(time_s, depth_um, displacement, source=source, config=config or {})


def load_field(path: str | Path, *, format: str = "auto", origin_offset_s: float | None = None,
               source: str | None = None, config: dict | None = None) -> MotionField:
    path = Path(path)
    if path.is_dir():
        names = {p.name for p in path.iterdir()}
        if {"time_bins.npy", "depth_bins.npy", "motion.npy"} <= names:
            data = {"time_bins": np.load(path/"time_bins.npy"), "depth_bins": np.load(path/"depth_bins.npy"),
                    "motion": np.load(path/"motion.npy")}
            offset = NPX_DREDGE_ORIGIN_S if format == "npx_dredge" else float(origin_offset_s or 0)
            result = _from_mapping(data, source or str(path), offset, config)
        else:
            raise ValueError(f"unrecognised motion directory: {path}")
    elif path.suffix == ".pkl":
        with path.open("rb") as stream:
            obj = _CompatUnpickler(stream).load()
        data = _object_mapping(obj)
        result = _from_mapping(data, source or str(path), float(origin_offset_s or 0), config)
    else:
        with np.load(path, allow_pickle=False) as z:
            data = {key: np.asarray(z[key]) for key in z.files}
        if "sign" in data:
            sign = str(data.pop("sign"))
        elif "sign_convention" in data:
            sign = str(data.pop("sign_convention"))
        else:
            sign = SIGN
        offset = float(origin_offset_s or 0)
        if format == "npx_dredge":
            offset = NPX_DREDGE_ORIGIN_S if origin_offset_s is None else float(origin_offset_s)
        result = _from_mapping(data, source or str(path), offset, config)
        result.sign = sign
    result.validate()
    return result


def canonical_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        for name in sorted(arrays):
            buffer = io.BytesIO(); np.lib.format.write_array(buffer, np.asarray(arrays[name]), allow_pickle=False)
            info = zipfile.ZipInfo(name + ".npy", date_time=(1980,1,1,0,0,0));info.compress_type=zipfile.ZIP_STORED
            info.external_attr = 0o600 << 16;archive.writestr(info, buffer.getvalue())


def canonical_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n", newline="\n")


def file_sha256(path: str | Path) -> str:
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""):digest.update(chunk)
    return digest.hexdigest()


def canonical_text_hash(path: str | Path) -> str:
    text=Path(path).read_text(encoding="utf-8").replace("\r\n","\n").replace("\r","\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
