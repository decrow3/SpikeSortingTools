"""Provenance-safe rigid DREDGE estimation without voltage resampling.

This module deliberately keeps estimation beside the accepted sorter recording.
It never constructs a voltage motion operator and returns the exact recording
object supplied through ``recording_for_sorting``.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal, Mapping

import numpy as np

from .config import PIPELINE_VERSION, fingerprint
from .preprocess import RECORDING_MANIFEST_SCHEMA, recording_geometry_receipt


MOTION_SIDECAR_CONFIG_SCHEMA = "dredge-sidecar-config-v1"
MOTION_ESTIMATE_SCHEMA = "dredge-rigid-estimate-v1"
MOTION_FAILURE_SCHEMA = "dredge-estimation-failure-v1"
MOTION_QC_SCHEMA = "motion-estimate-qc-v1"
MOTION_SIDECAR_MANIFEST = "manifest.json"
MOTION_METHOD_DIR = "dredge-rigid-sidecar"
LEGACY_AUTO_APPLICATION_PATH = Path("dredge-motion") / "motion.npy"


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(
        json.dumps(_jsonable(dict(payload)), indent=2, allow_nan=False) + "\n"
    )
    os.replace(temporary, path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def software_versions() -> dict[str, str]:
    """Return versions that can affect detection, localization, or DREDGE."""
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": _package_version("scipy"),
        "spikeinterface": _package_version("spikeinterface"),
        "dredge_backend": "bundled-in-spikeinterface",
        "torch": _package_version("torch"),
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class MotionEstimatorInputConfig:
    """Frozen estimator-only branch inherited from the historical pipeline."""

    version: str = "luke-motion-input-300-3000-local-median-v1"
    filter_band_hz: tuple[float, float] = (300.0, 3000.0)
    filter_order: int = 12
    filter_type: str = "butter"
    filter_direction: str = "forward-backward"
    reference: str = "local"
    reference_operator: str = "median"
    local_radius_um: tuple[float, float] = (40.0, 140.0)

    def __post_init__(self) -> None:
        low, high = self.filter_band_hz
        if low <= 0 or high <= low:
            raise ValueError("Motion estimator filter band must be positive and ordered")
        if self.filter_order < 1:
            raise ValueError("Motion estimator filter order must be positive")
        inner, outer = self.local_radius_um
        if inner < 0 or outer <= inner:
            raise ValueError("Local reference radii must be nonnegative and ordered")


@dataclass(frozen=True)
class PeakDetectionConfig:
    method: str = "locally_exclusive"
    peak_sign: str = "neg"
    detect_threshold: float = 5.0
    exclude_sweep_ms: float = 0.1
    radius_um: float = 50.0

    def __post_init__(self) -> None:
        if self.method != "locally_exclusive":
            raise ValueError("Production peak detection must be locally_exclusive")
        if self.peak_sign not in {"neg", "pos", "both"}:
            raise ValueError("Invalid peak sign")
        if self.detect_threshold <= 0 or self.radius_um <= 0:
            raise ValueError("Peak threshold and radius must be positive")


@dataclass(frozen=True)
class PeakLocalizationConfig:
    method: str = "monopolar_triangulation"
    ms_before: float = 0.5
    ms_after: float = 0.5
    radius_um: float = 75.0
    max_distance_um: float = 150.0
    optimizer: str = "minimize_with_log_penality"
    enforce_decrease: bool = True
    feature: str = "ptp"

    def __post_init__(self) -> None:
        if self.method != "monopolar_triangulation":
            raise ValueError("Production localization must be monopolar_triangulation")
        if min(self.ms_before, self.ms_after, self.radius_um, self.max_distance_um) <= 0:
            raise ValueError("Localization windows and radii must be positive")


@dataclass(frozen=True)
class DredgeRigidConfig:
    """Explicit SpikeInterface 0.102.1 DREDGE-AP effective settings."""

    method: str = "dredge_ap"
    direction: str = "y"
    rigid: bool = True
    win_shape: str = "rect"
    win_step_um: float = 400.0
    win_scale_um: float = 450.0
    win_margin_um: float | None = None
    bin_um: float = 1.0
    bin_s: float = 1.0
    max_disp_um: float | None = None
    time_horizon_s: float = 1000.0
    mincorr: float = 0.1
    do_window_weights: bool = True
    weights_threshold_low: float = 0.2
    weights_threshold_high: float = 0.2
    mincorr_percentile: float | None = None
    mincorr_percentile_nneighbs: int | None = None
    histogram_depth_smooth_um: float = 1.0
    histogram_time_smooth_s: float = 1.0
    avg_in_bin: bool = False
    device: str | None = None
    progress_bar: bool = True
    verbose: bool = True

    def __post_init__(self) -> None:
        if self.method != "dredge_ap" or not self.rigid:
            raise ValueError("Production DREDGE must use direct rigid dredge_ap")
        if self.direction != "y" or self.win_shape != "rect":
            raise ValueError("Production DREDGE must use y-directed rigid windows")
        if min(self.bin_um, self.bin_s, self.time_horizon_s) <= 0:
            raise ValueError("DREDGE bins and time horizon must be positive")
        if not 0 <= self.mincorr <= 1:
            raise ValueError("DREDGE mincorr must lie in [0, 1]")


@dataclass(frozen=True)
class MotionQCConfig:
    policy_version: str | None = None
    thresholds_validated: bool = False
    min_peak_count_per_time: int | None = None
    min_occupied_depth_bins: int | None = None
    max_step_um: float | None = None
    max_speed_um_s: float | None = None

    def __post_init__(self) -> None:
        thresholds = (
            self.min_peak_count_per_time,
            self.min_occupied_depth_bins,
            self.max_step_um,
            self.max_speed_um_s,
        )
        if self.thresholds_validated:
            if not self.policy_version or any(value is None for value in thresholds):
                raise ValueError("Validated QC requires a version and every threshold")
            if any(float(value) <= 0 for value in thresholds if value is not None):
                raise ValueError("Validated QC thresholds must be positive")
        elif self.policy_version is not None or any(value is not None for value in thresholds):
            raise ValueError("Unvalidated QC cannot carry authoritative thresholds")


@dataclass(frozen=True)
class MotionSidecarConfig:
    schema_version: str = MOTION_SIDECAR_CONFIG_SCHEMA
    estimate: bool = True
    estimator: str = "dredge"
    estimator_mode: Literal["rigid"] = "rigid"
    estimator_input: MotionEstimatorInputConfig = field(
        default_factory=MotionEstimatorInputConfig
    )
    detection: PeakDetectionConfig = field(default_factory=PeakDetectionConfig)
    localization: PeakLocalizationConfig = field(default_factory=PeakLocalizationConfig)
    dredge: DredgeRigidConfig = field(default_factory=DredgeRigidConfig)
    qc: MotionQCConfig = field(default_factory=MotionQCConfig)
    support_depth_bin_um: float = 100.0
    reference_method: str = "median_all_finite"
    split_half: bool = False
    save_nonrigid_for_diagnostics: bool = False
    voltage_correction_enabled: bool = False
    legacy_analysis_export: bool = False
    legacy_correction_cache_export: bool = False
    fallback: Literal["identity"] = "identity"

    def __post_init__(self) -> None:
        if self.schema_version != MOTION_SIDECAR_CONFIG_SCHEMA:
            raise ValueError("Unsupported motion sidecar configuration schema")
        if self.estimator != "dredge" or self.estimator_mode != "rigid":
            raise ValueError("Production motion estimation is rigid DREDGE only")
        if self.save_nonrigid_for_diagnostics:
            raise ValueError("Production configuration cannot generate a nonrigid field")
        if self.voltage_correction_enabled:
            raise ValueError("Voltage motion correction is not authorized")
        if self.legacy_correction_cache_export:
            raise ValueError("Legacy correction-ready cache export is forbidden")
        if self.legacy_analysis_export:
            raise ValueError("Legacy analysis export is not implemented")
        if self.fallback != "identity":
            raise ValueError("The only production fallback is identity")
        if self.support_depth_bin_um <= 0:
            raise ValueError("Support depth bin must be positive")
        if self.reference_method != "median_all_finite":
            raise ValueError("Unsupported rigid displacement reference method")

    def as_dict(self) -> dict[str, Any]:
        return {
            "pipeline_version": PIPELINE_VERSION,
            **_jsonable(asdict(self)),
        }

    @property
    def digest(self) -> str:
        return fingerprint(self.as_dict())


@dataclass(frozen=True)
class JobConfig:
    n_jobs: int = 1
    chunk_duration: str = "2s"
    progress_bar: bool = True

    def __post_init__(self) -> None:
        if self.n_jobs < 1:
            raise ValueError("n_jobs must be positive")
        if not self.chunk_duration:
            raise ValueError("chunk_duration must be nonempty")

    def as_kwargs(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MotionBackend:
    detect_peaks: Callable[..., np.ndarray]
    localize_peaks: Callable[..., np.ndarray]
    estimate_motion: Callable[..., Any]
    versions: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class RigidMotionEstimate:
    displacement_native_um: np.ndarray
    displacement_reference_centered_um: np.ndarray
    time_s: np.ndarray
    depth_um: np.ndarray
    peak_count_by_time: np.ndarray
    peak_count_by_time_depth: np.ndarray
    depth_bin_centers_um: np.ndarray
    support_by_time: np.ndarray
    reference_method: str
    reference_value_um: float
    provenance: Mapping[str, Any]
    cache_lineage: Mapping[str, Any]


@dataclass(frozen=True)
class MotionQC:
    status: Literal["VALID", "PARTIALLY_VALID", "INVALID", "NOT_EVALUATED"]
    valid_by_time: np.ndarray
    uncertainty_by_time_um: np.ndarray
    reason_codes_by_time: np.ndarray
    metrics: Mapping[str, Any]
    policy_version: str | None


@dataclass(frozen=True)
class MotionSidecarRun:
    recording_for_sorting: Any
    estimate: RigidMotionEstimate | None
    qc: MotionQC
    status: str
    artifact_dir: Path
    request_digest: str
    cache_lineage: Mapping[str, Any]


def build_motion_estimator_input(accepted_recording, config: MotionEstimatorInputConfig):
    """Construct the historical narrowband/local-reference estimator view."""
    from spikeinterface.preprocessing import common_reference, filter

    filtered = filter(
        accepted_recording,
        band=list(config.filter_band_hz),
        btype="bandpass",
        filter_order=config.filter_order,
        ftype=config.filter_type,
        direction=config.filter_direction,
    )
    return common_reference(
        filtered,
        reference=config.reference,
        operator=config.reference_operator,
        local_radius=config.local_radius_um,
    )


def _recording_identity(recording) -> dict[str, Any]:
    try:
        geometry = recording_geometry_receipt(recording)
    except Exception as exc:
        raise ValueError("Recording must expose valid channel locations") from exc
    return {
        **geometry,
        "num_channels": len(geometry["physical_channel_ids"]),
    }


def _validate_recording_lineage(estimator_recording, recording_for_sorting) -> dict[str, Any]:
    estimator = _recording_identity(estimator_recording)
    sorter = _recording_identity(recording_for_sorting)
    if estimator["physical_channel_ids"] != sorter["physical_channel_ids"]:
        raise ValueError("Estimator and sorter physical channel IDs differ")
    if estimator["probe_geometry_hash"] != sorter["probe_geometry_hash"]:
        raise ValueError("Estimator and sorter probe geometry differ")
    return {"estimator": estimator, "sorter": sorter, "match": True}


def _recording_request(recording, lineage: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "num_samples": int(recording.get_num_samples()),
        "num_channels": int(recording.get_num_channels()),
        "sampling_frequency_hz": float(recording.get_sampling_frequency()),
        "dtype": str(recording.dtype),
        "lineage": lineage,
    }


def _default_backend() -> MotionBackend:
    from spikeinterface.sortingcomponents.motion import estimate_motion
    from spikeinterface.sortingcomponents.peak_detection import detect_peaks
    from spikeinterface.sortingcomponents.peak_localization import localize_peaks

    return MotionBackend(
        detect_peaks=detect_peaks,
        localize_peaks=localize_peaks,
        estimate_motion=estimate_motion,
        versions=software_versions(),
    )


def _detection_kwargs(config: PeakDetectionConfig) -> dict[str, Any]:
    return asdict(config)


def _localization_kwargs(config: PeakLocalizationConfig) -> dict[str, Any]:
    return asdict(config)


def _dredge_kwargs(config: DredgeRigidConfig) -> dict[str, Any]:
    values = asdict(config)
    values.update(
        {
            "extra_outputs": True,
            "post_transform": np.log1p,
            "amp_scale_fn": None,
            "thomas_kw": None,
            "xcorr_kw": None,
            "precomputed_D_C_maxdisp": None,
        }
    )
    return values


def _dredge_effective_receipt(config: DredgeRigidConfig) -> dict[str, Any]:
    return {
        **_jsonable(asdict(config)),
        "extra_outputs": True,
        "post_transform": "numpy.log1p",
        "amp_scale_fn": None,
        "thomas_kw": None,
        "xcorr_kw": None,
        "precomputed_D_C_maxdisp": None,
    }


def _motion_arrays(motion) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    displacement = np.asarray(motion.displacement[0], dtype=np.float64)
    times = np.asarray(motion.temporal_bins_s[0], dtype=np.float64).reshape(-1)
    depths = np.asarray(motion.spatial_bins_um, dtype=np.float64).reshape(-1)
    if displacement.ndim == 1:
        displacement = displacement[:, None]
    expected = (times.size, depths.size)
    if displacement.shape != expected:
        raise ValueError(f"DREDGE displacement shape {displacement.shape} != {expected}")
    if depths.size != 1:
        raise ValueError("Production rigid DREDGE must return exactly one spatial bin")
    if times.size < 2 or not np.all(np.isfinite(times)) or not np.all(np.diff(times) > 0):
        raise ValueError("DREDGE time bins must be finite and strictly increasing")
    if np.any(~np.isfinite(displacement)):
        raise ValueError("DREDGE rigid displacement contains nonfinite values")
    return displacement, times, depths


def _centers_to_edges(centers: np.ndarray, lower: float, upper: float) -> np.ndarray:
    centers = np.asarray(centers, dtype=np.float64).reshape(-1)
    if centers.size < 2:
        raise ValueError("At least two bin centers are required")
    middle = (centers[:-1] + centers[1:]) / 2
    edges = np.r_[centers[0] - (middle[0] - centers[0]), middle, centers[-1] + (centers[-1] - middle[-1])]
    edges[0] = min(edges[0], lower)
    edges[-1] = max(edges[-1], upper)
    if not np.all(np.diff(edges) > 0):
        raise ValueError("Derived bin edges are not strictly increasing")
    return edges


def _support_arrays(
    recording,
    peaks: np.ndarray,
    peak_locations: np.ndarray,
    time_centers: np.ndarray,
    depth_bin_um: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if peaks.dtype.names is None or "sample_index" not in peaks.dtype.names:
        raise ValueError("Detected peaks must contain sample_index")
    if peak_locations.dtype.names is None or "y" not in peak_locations.dtype.names:
        raise ValueError("Peak locations must contain y")
    if peaks.shape[0] != peak_locations.shape[0]:
        raise ValueError("Peaks and peak locations must have equal length")
    fs = float(recording.get_sampling_frequency())
    duration = int(recording.get_num_samples()) / fs
    peak_time_s = np.asarray(peaks["sample_index"], dtype=np.float64) / fs
    peak_depth_um = np.asarray(peak_locations["y"], dtype=np.float64)
    time_edges = _centers_to_edges(time_centers, 0.0, duration)
    geometry_y = np.asarray(recording.get_channel_locations(), dtype=float)[:, 1]
    # Freeze support bins to physical geometry so full and split-half maps share
    # an identical axis. Localizations outside the probe span are not counted as
    # supported physical depth evidence.
    depth_min = float(np.min(geometry_y))
    depth_max = float(np.max(geometry_y))
    first = np.floor(depth_min / depth_bin_um) * depth_bin_um
    last = np.ceil(depth_max / depth_bin_um) * depth_bin_um
    if last <= first:
        last = first + depth_bin_um
    depth_edges = np.arange(first, last + depth_bin_um * 1.01, depth_bin_um)
    finite = np.isfinite(peak_time_s) & np.isfinite(peak_depth_um)
    counts, _, _ = np.histogram2d(
        peak_time_s[finite], peak_depth_um[finite], bins=(time_edges, depth_edges)
    )
    counts = counts.astype(np.int64)
    count_by_time = np.sum(counts, axis=1)
    occupied_by_time = np.sum(counts > 0, axis=1).astype(np.int64)
    depth_centers = (depth_edges[:-1] + depth_edges[1:]) / 2
    return count_by_time, counts, depth_centers, occupied_by_time


def _safe_distribution(values: np.ndarray) -> dict[str, float | None]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not values.size:
        return {"median": None, "p95": None, "p99": None, "maximum": None}
    return {
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "maximum": float(np.max(values)),
    }


def evaluate_motion_qc(estimate: RigidMotionEstimate, config: MotionQCConfig) -> MotionQC:
    """Compute metrics and apply only explicitly validated threshold policies."""
    displacement = estimate.displacement_reference_centered_um[:, 0]
    times = estimate.time_s
    steps = np.abs(np.diff(displacement))
    speeds = steps / np.diff(times)
    metrics = {
        "rigid_range_um": float(np.ptp(displacement)),
        "absolute_displacement_um": _safe_distribution(np.abs(displacement)),
        "absolute_step_um": _safe_distribution(steps),
        "speed_um_s": _safe_distribution(speeds),
        "peak_count_by_time": _safe_distribution(estimate.peak_count_by_time),
        "occupied_depth_bins_by_time": _safe_distribution(estimate.support_by_time),
        "empty_time_fraction": float(np.mean(estimate.peak_count_by_time == 0)),
        "reference_method": estimate.reference_method,
        "reference_value_um": estimate.reference_value_um,
    }
    n_time = times.size
    uncertainty = np.full(n_time, np.nan, dtype=np.float64)
    if not config.thresholds_validated:
        return MotionQC(
            status="NOT_EVALUATED",
            valid_by_time=np.zeros(n_time, dtype=bool),
            uncertainty_by_time_um=uncertainty,
            reason_codes_by_time=np.full(n_time, "QC_POLICY_NOT_VALIDATED", dtype="U64"),
            metrics=metrics,
            policy_version=None,
        )

    valid = (
        (estimate.peak_count_by_time >= int(config.min_peak_count_per_time))
        & (estimate.support_by_time >= int(config.min_occupied_depth_bins))
    )
    reasons = np.full(n_time, "ADEQUATE_SUPPORT", dtype="U128")
    low_count = estimate.peak_count_by_time < int(config.min_peak_count_per_time)
    low_depth = estimate.support_by_time < int(config.min_occupied_depth_bins)
    reasons[low_count] = "LOW_PEAK_COUNT"
    reasons[low_depth] = "LOW_DEPTH_SUPPORT"
    reasons[low_count & low_depth] = "LOW_PEAK_COUNT;LOW_DEPTH_SUPPORT"
    bad_step = np.r_[False, steps > float(config.max_step_um)]
    bad_speed = np.r_[False, speeds > float(config.max_speed_um_s)]
    valid &= ~bad_step & ~bad_speed
    for mask, code in ((bad_step, "IMPLAUSIBLE_STEP"), (bad_speed, "EXCESSIVE_SPEED")):
        for index in np.flatnonzero(mask):
            reasons[index] = (
                code if reasons[index] == "ADEQUATE_SUPPORT" else f"{reasons[index]};{code}"
            )
    status: Literal["VALID", "PARTIALLY_VALID", "INVALID", "NOT_EVALUATED"]
    if np.all(valid):
        status = "VALID"
    elif np.any(valid):
        status = "PARTIALLY_VALID"
    else:
        status = "INVALID"
    return MotionQC(
        status=status,
        valid_by_time=valid,
        uncertainty_by_time_um=uncertainty,
        reason_codes_by_time=reasons,
        metrics=metrics,
        policy_version=config.policy_version,
    )


def _extra_summary(extra: Mapping[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for name, value in extra.items():
        if isinstance(value, np.ndarray):
            finite = np.isfinite(value) if value.dtype.kind in "fc" else None
            summary[name] = {
                "type": "ndarray",
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "finite_fraction": None if finite is None else float(np.mean(finite)),
            }
        else:
            summary[name] = {"type": type(value).__name__}
    return summary


def _deterministic_coverage_split(
    recording,
    peaks: np.ndarray,
    peak_locations: np.ndarray,
    *,
    time_bin_s: float,
    depth_bin_um: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Alternate peaks within time-depth cells to preserve evidence coverage."""
    fs = float(recording.get_sampling_frequency())
    times = np.asarray(peaks["sample_index"], dtype=np.float64) / fs
    depths = np.asarray(peak_locations["y"], dtype=np.float64)
    time_cells = np.floor(times / time_bin_s).astype(np.int64)
    depth_cells = np.floor(depths / depth_bin_um).astype(np.int64)
    order = np.lexsort((np.asarray(peaks["sample_index"]), depth_cells, time_cells))
    choose_a = np.zeros(peaks.size, dtype=bool)
    previous: tuple[int, int] | None = None
    parity = 0
    for index in order:
        cell = (int(time_cells[index]), int(depth_cells[index]))
        if cell != previous:
            previous = cell
            # Alternate the starting half across neighboring cells so singleton
            # cells do not all land in the same half.
            parity = (cell[0] + cell[1]) & 1
        choose_a[index] = parity % 2 == 0
        parity += 1
    choose_b = ~choose_a
    if not np.any(choose_a) or not np.any(choose_b):
        raise RuntimeError("Deterministic split-half produced an empty half")
    return choose_a, choose_b


def _run_split_half_audit(
    audit_dir: Path,
    *,
    recording,
    peaks: np.ndarray,
    peak_locations: np.ndarray,
    full_estimate_times: np.ndarray,
    backend: MotionBackend,
    config: MotionSidecarConfig,
) -> dict[str, Any]:
    """Run deterministic half estimates and preserve each half's support map."""
    audit_dir.mkdir(parents=True)
    try:
        half_a, half_b = _deterministic_coverage_split(
            recording,
            peaks,
            peak_locations,
            time_bin_s=config.dredge.bin_s,
            depth_bin_um=config.support_depth_bin_um,
        )
        traces = []
        supports = []
        for label, mask in (("a", half_a), ("b", half_b)):
            result = backend.estimate_motion(
                recording=recording,
                peaks=peaks[mask],
                peak_locations=peak_locations[mask],
                **_dredge_kwargs(config.dredge),
            )
            if not isinstance(result, tuple) or len(result) != 2:
                raise RuntimeError("Split-half DREDGE extra_outputs contract was not honored")
            half_motion, _ = result
            displacement, times, depths = _motion_arrays(half_motion)
            if not np.array_equal(times, full_estimate_times):
                raise RuntimeError("Split-half DREDGE time bins differ from the full estimate")
            reference = float(np.median(displacement))
            centered = displacement - reference
            count_time, count_map, depth_centers, occupied = _support_arrays(
                recording,
                peaks[mask],
                peak_locations[mask],
                times,
                config.support_depth_bin_um,
            )
            np.save(audit_dir / f"half_{label}_motion_native.npy", displacement)
            np.save(audit_dir / f"half_{label}_motion_reference_centered.npy", centered)
            np.save(audit_dir / f"half_{label}_peak_count_by_time.npy", count_time)
            np.save(audit_dir / f"half_{label}_peak_count_by_time_depth.npy", count_map)
            np.save(audit_dir / f"half_{label}_depth_bin_centers_um.npy", depth_centers)
            np.save(audit_dir / f"half_{label}_support_by_time.npy", occupied)
            np.save(audit_dir / f"half_{label}_depth_bins.npy", depths)
            traces.append(centered[:, 0])
            supports.append(occupied)
        difference = np.abs(traces[0] - traces[1])
        ranges = [float(np.ptp(trace)) for trace in traces]
        if min(ranges) <= np.finfo(float).eps:
            correlation = None
        else:
            correlation = float(np.corrcoef(traces[0], traces[1])[0, 1])
        excursion = np.maximum(np.abs(traces[0]), np.abs(traces[1]))
        large = excursion >= np.percentile(excursion, 75)
        sign_agreement = (
            None
            if not np.any(large)
            else float(np.mean(np.sign(traces[0][large]) == np.sign(traces[1][large])))
        )
        metrics = {
            "schema_version": "dredge-rigid-split-half-audit-v1",
            "split_method": "alternate_within_time_depth_cells",
            "half_a_peak_count": int(np.sum(half_a)),
            "half_b_peak_count": int(np.sum(half_b)),
            "half_a_dynamic_range_um": ranges[0],
            "half_b_dynamic_range_um": ranges[1],
            "correlation": correlation,
            "median_absolute_difference_um": float(np.median(difference)),
            "p95_absolute_difference_um": float(np.percentile(difference, 95)),
            "large_excursion_sign_agreement": sign_agreement,
            "half_a_empty_support_fraction": float(np.mean(supports[0] == 0)),
            "half_b_empty_support_fraction": float(np.mean(supports[1] == 0)),
            "thresholds_validated": False,
            "authorizes_voltage_correction": False,
            "complete": True,
        }
        _atomic_json(audit_dir / "split_half_metrics.json", metrics)
        return metrics
    except Exception as exc:
        failure = {
            "schema_version": "dredge-rigid-split-half-failure-v1",
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "thresholds_validated": False,
            "authorizes_voltage_correction": False,
            "complete": False,
        }
        _atomic_json(audit_dir / "split_half_failure.json", failure)
        return failure


def _write_qc(cache_dir: Path, qc: MotionQC) -> None:
    _atomic_json(
        cache_dir / "motion_qc.json",
        {
            "schema_version": MOTION_QC_SCHEMA,
            "status": qc.status,
            "valid_by_time": qc.valid_by_time,
            "uncertainty_by_time_um": qc.uncertainty_by_time_um,
            "reason_codes_by_time": qc.reason_codes_by_time,
            "metrics": qc.metrics,
            "policy_version": qc.policy_version,
            "correction_policy_validated": False,
            "correction_eligible_epochs": "NOT_EVALUATED",
            "voltage_correction_applied": False,
        },
    )


def _summary_markdown(
    status: str,
    estimate: RigidMotionEstimate | None,
    qc: MotionQC,
    cache_lineage: Mapping[str, Any],
    split_half_status: str = "not run",
) -> str:
    metrics = qc.metrics
    lines = [
        "# Motion sidecar summary",
        "",
        "- Motion estimator: DREDGE rigid",
        f"- Estimate status: {status}",
        f"- Cache lineage: {cache_lineage.get('status', 'none')}",
        f"- QC status: {qc.status}",
        f"- Split-half audit: {split_half_status}",
    ]
    if estimate is not None:
        lines.extend(
            [
                f"- Rigid range: {metrics['rigid_range_um']:.6g} um",
                f"- Empty time-bin fraction: {metrics['empty_time_fraction']:.6g}",
                f"- Reference method: {estimate.reference_method}",
                f"- Reference value: {estimate.reference_value_um:.6g} um",
            ]
        )
    lines.extend(
        [
            "- Correction policy validated: NO",
            "- Correction-eligible epochs: NOT EVALUATED",
            "- Voltage motion correction applied: NO",
            "- KS4 internal motion correction: OFF (enforced by sorter stage)",
            "",
        ]
    )
    return "\n".join(lines)


def _write_plots(cache_dir: Path, estimate: RigidMotionEstimate) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/spikeglx-rescue-matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = cache_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    times = estimate.time_s
    native = estimate.displacement_native_um[:, 0]
    centered = estimate.displacement_reference_centered_um[:, 0]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(times, native, label="estimator-native", alpha=0.7)
    ax.plot(times, centered, label="reference-centered")
    ax.set(xlabel="Selected-recording time (s)", ylabel="Displacement (um)")
    ax.set_title("Rigid DREDGE sidecar — voltage unchanged")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "rigid_trace.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.step(times, estimate.peak_count_by_time, where="mid", label="peaks")
    ax.step(times, estimate.support_by_time, where="mid", label="occupied depth bins")
    ax.set(xlabel="Selected-recording time (s)", ylabel="Count")
    ax.set_title("Estimator support — voltage unchanged")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "support_vs_time.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    speed = np.abs(np.diff(centered)) / np.diff(times)
    ax.step(times[1:], speed, where="mid")
    ax.set(xlabel="Selected-recording time (s)", ylabel="Absolute speed (um/s)")
    ax.set_title("Rigid motion speed — voltage unchanged")
    fig.tight_layout()
    fig.savefig(figures / "motion_speed.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    image = ax.imshow(
        estimate.peak_count_by_time_depth.T,
        aspect="auto",
        origin="lower",
        extent=(
            times[0],
            times[-1],
            estimate.depth_bin_centers_um[0],
            estimate.depth_bin_centers_um[-1],
        ),
    )
    ax.set(xlabel="Selected-recording time (s)", ylabel="Probe depth (um)")
    ax.set_title("Peak time-depth support — voltage unchanged")
    fig.colorbar(image, ax=ax, label="Peak count")
    fig.tight_layout()
    fig.savefig(figures / "peak_time_depth_support.png", dpi=150)
    fig.savefig(figures / "depth_raster.png", dpi=150)
    plt.close(fig)


def _write_peak_diagnostic_plot(
    cache_dir: Path, peaks: np.ndarray, peak_locations: np.ndarray
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/spikeglx-rescue-matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = cache_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    stride = max(1, int(np.ceil(peaks.size / 200_000)))
    amplitude = np.abs(np.asarray(peaks["amplitude"], dtype=float)[::stride])
    depth = np.asarray(peak_locations["y"], dtype=float)[::stride]
    finite = np.isfinite(amplitude) & np.isfinite(depth)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(amplitude[finite], depth[finite], s=1, alpha=0.15, rasterized=True)
    ax.set(xlabel="Absolute detected amplitude", ylabel="Localized depth (um)")
    ax.set_title("Peak amplitude-depth diagnostic — voltage unchanged")
    fig.tight_layout()
    fig.savefig(figures / "amplitude_depth_comparison.png", dpi=150)
    plt.close(fig)


def _load_estimate(method_dir: Path, lineage: Mapping[str, Any]) -> RigidMotionEstimate:
    manifest = json.loads((method_dir / MOTION_SIDECAR_MANIFEST).read_text())
    if manifest.get("schema_version") != MOTION_ESTIMATE_SCHEMA or not manifest.get("complete"):
        raise RuntimeError("Motion sidecar manifest is incomplete or unsupported")
    if manifest.get("voltage_modified") is not False:
        raise RuntimeError("Motion sidecar manifest does not preserve unchanged voltage")
    core_required = {
        "motion_native.npy",
        "motion_reference_centered.npy",
        "time_bins.npy",
        "depth_bins.npy",
        "peak_count_by_time.npy",
        "peak_count_by_time_depth.npy",
        "depth_bin_centers_um.npy",
        "support_by_time.npy",
    }
    required = manifest.get("required_files")
    if not isinstance(required, list) or not required:
        raise RuntimeError("Motion sidecar manifest lacks a required-file contract")
    if not core_required.issubset(required):
        raise RuntimeError("Motion sidecar required-file contract omits core arrays")
    hashes = manifest.get("file_sha256")
    if not isinstance(hashes, dict) or set(hashes) != set(required):
        raise RuntimeError("Motion sidecar file-digest contract is incomplete")
    missing = [name for name in required if not (method_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"Incomplete motion sidecar artifact: {sorted(missing)}")
    for name, expected in hashes.items():
        if _sha256_file(method_dir / name) != expected:
            raise RuntimeError(f"Motion sidecar artifact digest mismatch for {name}")
    native = np.load(method_dir / "motion_native.npy", allow_pickle=False)
    centered = np.load(
        method_dir / "motion_reference_centered.npy", allow_pickle=False
    )
    times = np.load(method_dir / "time_bins.npy", allow_pickle=False)
    depths = np.load(method_dir / "depth_bins.npy", allow_pickle=False)
    count_time = np.load(method_dir / "peak_count_by_time.npy", allow_pickle=False)
    count_map = np.load(
        method_dir / "peak_count_by_time_depth.npy", allow_pickle=False
    )
    depth_centers = np.load(
        method_dir / "depth_bin_centers_um.npy", allow_pickle=False
    )
    support = np.load(method_dir / "support_by_time.npy", allow_pickle=False)
    expected_motion = (times.size, depths.size)
    if native.shape != expected_motion or centered.shape != expected_motion:
        raise RuntimeError("Motion sidecar displacement arrays have incompatible shapes")
    if depths.size != 1 or times.size < 2 or not np.all(np.diff(times) > 0):
        raise RuntimeError("Motion sidecar is not a valid rigid time series")
    if count_time.shape != times.shape or support.shape != times.shape:
        raise RuntimeError("Motion sidecar time support has an incompatible shape")
    if count_map.shape != (times.size, depth_centers.size):
        raise RuntimeError("Motion sidecar time-depth support has an incompatible shape")
    if np.any(~np.isfinite(native)) or np.any(~np.isfinite(centered)):
        raise RuntimeError("Motion sidecar displacement contains nonfinite values")
    return RigidMotionEstimate(
        displacement_native_um=native,
        displacement_reference_centered_um=centered,
        time_s=times,
        depth_um=depths,
        peak_count_by_time=count_time,
        peak_count_by_time_depth=count_map,
        depth_bin_centers_um=depth_centers,
        support_by_time=support,
        reference_method=manifest["reference_method"],
        reference_value_um=float(manifest["reference_value_um"]),
        provenance=manifest["provenance"],
        cache_lineage=lineage,
    )


def _save_estimate(
    partial: Path,
    *,
    displacement_native: np.ndarray,
    displacement_centered: np.ndarray,
    times: np.ndarray,
    depths: np.ndarray,
    count_by_time: np.ndarray,
    count_by_time_depth: np.ndarray,
    depth_centers: np.ndarray,
    support_by_time: np.ndarray,
    peaks: np.ndarray,
    peak_locations: np.ndarray,
    reference_method: str,
    reference_value: float,
    request: Mapping[str, Any],
    request_digest: str,
    provenance: Mapping[str, Any],
    extra: Mapping[str, Any],
) -> dict[str, Any]:
    partial.mkdir(parents=True)
    arrays = {
        "motion_native.npy": displacement_native,
        "motion_reference_centered.npy": displacement_centered,
        "time_bins.npy": times,
        "depth_bins.npy": depths,
        "peak_count_by_time.npy": count_by_time,
        "peak_count_by_time_depth.npy": count_by_time_depth,
        "depth_bin_centers_um.npy": depth_centers,
        "support_by_time.npy": support_by_time,
        "peaks.npy": peaks,
        "peak_locations.npy": peak_locations,
    }
    for name, values in arrays.items():
        np.save(partial / name, values, allow_pickle=False)
    np.savez_compressed(
        partial / "estimate.npz",
        schema_version=MOTION_ESTIMATE_SCHEMA,
        displacement_native_um=displacement_native,
        displacement_reference_centered_um=displacement_centered,
        time_s=times,
        depth_um=depths,
        peak_count_by_time=count_by_time,
        peak_count_by_time_depth=count_by_time_depth,
        depth_bin_centers_um=depth_centers,
        support_by_time=support_by_time,
        reference_method=reference_method,
        reference_value_um=reference_value,
        time_reference="selected_recording_start",
        depth_reference="probe_y_um",
        displacement_convention="observed_depth_offset_um",
    )
    _atomic_json(partial / "extra_summary.json", _extra_summary(extra))
    manifest = {
        "schema_version": MOTION_ESTIMATE_SCHEMA,
        "pipeline_version": PIPELINE_VERSION,
        "request": request,
        "request_digest": request_digest,
        "artifact_digest": None,
        "file_sha256": {},
        "required_files": [],
        "reference_method": reference_method,
        "reference_value_um": reference_value,
        "time_reference": "selected_recording_start",
        "depth_reference": "probe_y_um",
        "displacement_convention": "observed_depth_offset_um",
        "estimator_mode": "rigid",
        "voltage_modified": False,
        "authorized_for_voltage_application": False,
        "provenance": provenance,
        "cache_lineage": {
            "status": "pending",
            "source_artifact_digest": None,
            "accepted_artifact_digest": None,
        },
        "complete": False,
    }
    _atomic_json(partial / MOTION_SIDECAR_MANIFEST, manifest)
    return manifest


def _finalize_estimate_artifact(
    partial: Path, manifest: Mapping[str, Any], *, split_half_requested: bool
) -> dict[str, Any]:
    """Seal every required output into the accepted artifact digest."""
    expected = {
        "motion_native.npy",
        "motion_reference_centered.npy",
        "time_bins.npy",
        "depth_bins.npy",
        "peak_count_by_time.npy",
        "peak_count_by_time_depth.npy",
        "depth_bin_centers_um.npy",
        "support_by_time.npy",
        "peaks.npy",
        "peak_locations.npy",
        "estimate.npz",
        "extra_summary.json",
        "figures/rigid_trace.png",
        "figures/support_vs_time.png",
        "figures/motion_speed.png",
        "figures/peak_time_depth_support.png",
        "figures/depth_raster.png",
        "figures/amplitude_depth_comparison.png",
    }
    if split_half_requested:
        audit = partial / "audits" / "split_half"
        completed = audit / "split_half_metrics.json"
        failed = audit / "split_half_failure.json"
        if completed.exists():
            expected.update(
                {
                    "audits/split_half/split_half_metrics.json",
                    "audits/split_half/half_a_motion_native.npy",
                    "audits/split_half/half_a_motion_reference_centered.npy",
                    "audits/split_half/half_a_peak_count_by_time.npy",
                    "audits/split_half/half_a_peak_count_by_time_depth.npy",
                    "audits/split_half/half_a_depth_bin_centers_um.npy",
                    "audits/split_half/half_a_support_by_time.npy",
                    "audits/split_half/half_a_depth_bins.npy",
                    "audits/split_half/half_b_motion_native.npy",
                    "audits/split_half/half_b_motion_reference_centered.npy",
                    "audits/split_half/half_b_peak_count_by_time.npy",
                    "audits/split_half/half_b_peak_count_by_time_depth.npy",
                    "audits/split_half/half_b_depth_bin_centers_um.npy",
                    "audits/split_half/half_b_support_by_time.npy",
                    "audits/split_half/half_b_depth_bins.npy",
                }
            )
        elif failed.exists():
            expected.add("audits/split_half/split_half_failure.json")
        else:
            raise RuntimeError("Requested split-half audit produced no terminal artifact")
    missing = sorted(name for name in expected if not (partial / name).is_file())
    if missing:
        raise RuntimeError(f"Motion sidecar is missing required outputs: {missing}")
    required_files = sorted(expected)
    file_sha256 = {
        name: _sha256_file(partial / name) for name in required_files
    }
    artifact_digest = fingerprint(
        {
            "request_digest": manifest["request_digest"],
            "file_sha256": file_sha256,
        }
    )
    finalized = {
        **dict(manifest),
        "artifact_digest": artifact_digest,
        "file_sha256": file_sha256,
        "required_files": required_files,
        "cache_lineage": {
            "status": "computed_new",
            "source_artifact_digest": None,
            "accepted_artifact_digest": artifact_digest,
        },
        "computed_at_utc": _utc_now(),
        "complete": True,
    }
    _atomic_json(partial / MOTION_SIDECAR_MANIFEST, finalized)
    return finalized


def _mirror_inspection_artifacts(cache_dir: Path, method_dir: Path) -> None:
    """Write familiar root-level arrays without creating a correction cache."""
    mapping = {
        "peaks.npy": "peaks.npy",
        "peak_locations.npy": "peak_locations.npy",
        "peak_count_by_time.npy": "peak_count_by_time.npy",
        "peak_count_by_time_depth.npy": "peak_count_by_time_depth.npy",
        "depth_bin_centers_um.npy": "depth_bin_centers_um.npy",
        "time_bins.npy": "time_bins.npy",
        "depth_bins.npy": "depth_bins.npy",
    }
    for source_name, target_name in mapping.items():
        source = method_dir / source_name
        target = cache_dir / target_name
        temporary = target.with_name(target.name + ".partial")
        shutil.copyfile(source, temporary)
        os.replace(temporary, target)


def _mirror_figures(cache_dir: Path, method_dir: Path) -> None:
    source_dir = method_dir / "figures"
    if not source_dir.exists():
        return
    target_dir = cache_dir / "figures"
    target_dir.mkdir(parents=True, exist_ok=True)
    for source in source_dir.glob("*.png"):
        target = target_dir / source.name
        temporary = target.with_name(target.name + ".partial")
        shutil.copyfile(source, temporary)
        os.replace(temporary, target)


def _split_half_status(method_dir: Path) -> str:
    audit = method_dir / "audits" / "split_half"
    if (audit / "split_half_metrics.json").exists():
        return "completed (diagnostic only)"
    if (audit / "split_half_failure.json").exists():
        return "failed (does not authorize correction)"
    return "not run"


def _archive_failure_receipt(cache_dir: Path) -> None:
    failure = cache_dir / "estimation_failure.json"
    if not failure.exists():
        return
    try:
        payload = json.loads(failure.read_text())
        suffix = str(payload.get("request_digest", "unknown"))[:12]
    except Exception:
        suffix = "unknown"
    archived = cache_dir / f"estimation_failure.superseded-{suffix}.json"
    counter = 2
    while archived.exists():
        archived = cache_dir / f"estimation_failure.superseded-{suffix}-{counter}.json"
        counter += 1
    os.replace(failure, archived)


def _failure_qc() -> MotionQC:
    return MotionQC(
        status="INVALID",
        valid_by_time=np.zeros(0, dtype=bool),
        uncertainty_by_time_um=np.zeros(0, dtype=float),
        reason_codes_by_time=np.zeros(0, dtype="U1"),
        metrics={"failure": True},
        policy_version=None,
    )


def run_motion_sidecar(
    estimator_recording,
    *,
    recording_for_sorting,
    cache_dir: Path,
    config: MotionSidecarConfig | None = None,
    job_config: JobConfig | None = None,
    recompute: bool = False,
    strict: bool = False,
    backend: MotionBackend | None = None,
    accepted_recording_manifest: Mapping[str, Any] | None = None,
) -> MotionSidecarRun:
    """Estimate rigid motion and always return the exact supplied sorter input."""
    config = MotionSidecarConfig() if config is None else config
    job_config = JobConfig() if job_config is None else job_config
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    method_dir = cache_dir / MOTION_METHOD_DIR
    partial = cache_dir / f"{MOTION_METHOD_DIR}.partial"
    legacy_path = cache_dir / LEGACY_AUTO_APPLICATION_PATH
    legacy_cache_detected = legacy_path.exists()
    if partial.exists():
        raise RuntimeError(f"Incomplete motion sidecar requires inspection: {partial}")

    if accepted_recording_manifest is None:
        raise ValueError("Motion sidecar requires an accepted recording manifest")
    if accepted_recording_manifest.get("schema_version") != RECORDING_MANIFEST_SCHEMA:
        raise ValueError("Accepted recording manifest schema is unsupported")
    if not accepted_recording_manifest.get("complete"):
        raise ValueError("Accepted recording manifest is not complete")
    if not accepted_recording_manifest.get("request_digest"):
        raise ValueError("Accepted recording manifest lacks a request digest")
    if not accepted_recording_manifest.get("recording_content_sha256"):
        raise ValueError("Accepted recording manifest lacks a content digest")
    lineage = _validate_recording_lineage(estimator_recording, recording_for_sorting)
    accepted_manifest_identity = {
        "request_digest": accepted_recording_manifest["request_digest"],
        "schema_version": accepted_recording_manifest["schema_version"],
        "recording_content_sha256": accepted_recording_manifest[
            "recording_content_sha256"
        ],
        "selected_start_frame": accepted_recording_manifest.get("selected_start_frame"),
        "selected_end_frame": accepted_recording_manifest.get("selected_end_frame"),
    }
    manifest_channels = accepted_recording_manifest.get("physical_channel_ids")
    manifest_geometry = accepted_recording_manifest.get("probe_geometry_hash")
    if manifest_channels is not None and list(manifest_channels) != lineage["sorter"]["physical_channel_ids"]:
        raise ValueError("Accepted manifest physical channel IDs differ from sorter input")
    if manifest_geometry is not None and manifest_geometry != lineage["sorter"]["probe_geometry_hash"]:
        raise ValueError("Accepted manifest probe geometry differs from sorter input")
    versions = dict(software_versions())
    if backend is not None:
        versions.update({str(key): str(value) for key, value in backend.versions.items()})
    request = {
        "schema_version": MOTION_SIDECAR_CONFIG_SCHEMA,
        "pipeline_version": PIPELINE_VERSION,
        "config": config.as_dict(),
        "config_digest": config.digest,
        "recording": _recording_request(estimator_recording, lineage),
        "accepted_recording_manifest": accepted_manifest_identity,
        "job_config": _jsonable(asdict(job_config)),
        "effective_dredge_settings": _dredge_effective_receipt(config.dredge),
        "software_versions": versions,
        "legacy_correction_cache_detected_and_ignored": legacy_cache_detected,
    }
    request_digest = fingerprint(request)
    _atomic_json(cache_dir / "request.json", {**request, "request_digest": request_digest})
    _atomic_json(
        cache_dir / "estimator_input_receipt.json",
        {
            "schema_version": "motion-estimator-input-receipt-v1",
            "request_digest": request_digest,
            "estimator_input": _jsonable(asdict(config.estimator_input)),
            "recording_lineage": lineage,
            "accepted_recording_manifest": accepted_manifest_identity,
            "legacy_correction_cache_detected_and_ignored": legacy_cache_detected,
        },
    )

    if not config.estimate:
        qc = MotionQC(
            status="NOT_EVALUATED",
            valid_by_time=np.zeros(0, dtype=bool),
            uncertainty_by_time_um=np.zeros(0, dtype=float),
            reason_codes_by_time=np.zeros(0, dtype="U1"),
            metrics={"estimation_disabled": True},
            policy_version=None,
        )
        cache_lineage = {"status": "none", "source_artifact_digest": None}
        _write_qc(cache_dir, qc)
        (cache_dir / "motion_summary.md").write_text(
            _summary_markdown("ESTIMATE_DISABLED_IDENTITY_SORT_CONTINUED", None, qc, cache_lineage)
        )
        return MotionSidecarRun(
            recording_for_sorting, None, qc,
            "ESTIMATE_DISABLED_IDENTITY_SORT_CONTINUED", cache_dir,
            request_digest, cache_lineage,
        )

    if method_dir.exists() and not recompute:
        manifest_path = method_dir / MOTION_SIDECAR_MANIFEST
        if not manifest_path.exists():
            raise RuntimeError(f"Existing motion sidecar lacks {MOTION_SIDECAR_MANIFEST}")
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("request_digest") != request_digest:
            raise RuntimeError("Existing motion sidecar belongs to another request")
        cache_lineage = {
            "status": "reused_exact_match",
            "source_artifact_digest": manifest["artifact_digest"],
            "accepted_artifact_digest": manifest["artifact_digest"],
        }
        estimate = _load_estimate(method_dir, cache_lineage)
        qc = evaluate_motion_qc(estimate, config.qc)
        _archive_failure_receipt(cache_dir)
        _mirror_inspection_artifacts(cache_dir, method_dir)
        _write_qc(cache_dir, qc)
        (cache_dir / "motion_summary.md").write_text(
            _summary_markdown(
                "ESTIMATE_COMPLETED",
                estimate,
                qc,
                cache_lineage,
                _split_half_status(method_dir),
            )
        )
        return MotionSidecarRun(
            recording_for_sorting, estimate, qc, "ESTIMATE_COMPLETED",
            cache_dir, request_digest, cache_lineage,
        )

    failure_stage = "backend_import"
    try:
        selected_backend = _default_backend() if backend is None else backend
        failure_stage = "peak_detection"
        peaks = selected_backend.detect_peaks(
            estimator_recording,
            **_detection_kwargs(config.detection),
            **job_config.as_kwargs(),
        )
        if not np.asarray(peaks).size:
            raise RuntimeError("Peak detection returned no peaks")
        failure_stage = "peak_localization"
        peak_locations = selected_backend.localize_peaks(
            estimator_recording,
            peaks,
            **_localization_kwargs(config.localization),
            **job_config.as_kwargs(),
        )
        failure_stage = "rigid_dredge_estimation"
        motion_result = selected_backend.estimate_motion(
            recording=estimator_recording,
            peaks=peaks,
            peak_locations=peak_locations,
            **_dredge_kwargs(config.dredge),
        )
        if not isinstance(motion_result, tuple) or len(motion_result) != 2:
            raise RuntimeError("DREDGE extra_outputs contract was not honored")
        motion, extra = motion_result
        displacement, times, depths = _motion_arrays(motion)
        reference_value = float(np.median(displacement[np.isfinite(displacement)]))
        centered = displacement - reference_value
        count_by_time, count_by_time_depth, depth_centers, occupied = _support_arrays(
            estimator_recording,
            np.asarray(peaks),
            np.asarray(peak_locations),
            times,
            config.support_depth_bin_um,
        )
        provenance = {
            "software_versions": versions,
            "estimated_at_utc": _utc_now(),
            "estimator_input": _jsonable(asdict(config.estimator_input)),
            "detection": _jsonable(asdict(config.detection)),
            "localization": _jsonable(asdict(config.localization)),
            "dredge": _jsonable(asdict(config.dredge)),
            "effective_dredge": _dredge_effective_receipt(config.dredge),
            "physical_channel_ids": lineage["estimator"]["physical_channel_ids"],
            "probe_geometry_hash": lineage["estimator"]["probe_geometry_hash"],
        }
        failure_stage = "artifact_materialization"
        manifest = _save_estimate(
            partial,
            displacement_native=displacement,
            displacement_centered=centered,
            times=times,
            depths=depths,
            count_by_time=count_by_time,
            count_by_time_depth=count_by_time_depth,
            depth_centers=depth_centers,
            support_by_time=occupied,
            peaks=np.asarray(peaks),
            peak_locations=np.asarray(peak_locations),
            reference_method=config.reference_method,
            reference_value=reference_value,
            request=request,
            request_digest=request_digest,
            provenance=provenance,
            extra=extra,
        )
        if config.split_half:
            failure_stage = "split_half_audit"
            _run_split_half_audit(
                partial / "audits" / "split_half",
                recording=estimator_recording,
                peaks=np.asarray(peaks),
                peak_locations=np.asarray(peak_locations),
                full_estimate_times=times,
                backend=selected_backend,
                config=config,
            )
        failure_stage = "figure_generation"
        pending_lineage = dict(manifest["cache_lineage"])
        pending_estimate = RigidMotionEstimate(
            displacement_native_um=displacement,
            displacement_reference_centered_um=centered,
            time_s=times,
            depth_um=depths,
            peak_count_by_time=count_by_time,
            peak_count_by_time_depth=count_by_time_depth,
            depth_bin_centers_um=depth_centers,
            support_by_time=occupied,
            reference_method=config.reference_method,
            reference_value_um=reference_value,
            provenance=provenance,
            cache_lineage=pending_lineage,
        )
        _write_plots(partial, pending_estimate)
        _write_peak_diagnostic_plot(partial, np.asarray(peaks), np.asarray(peak_locations))
        failure_stage = "artifact_finalization"
        manifest = _finalize_estimate_artifact(
            partial, manifest, split_half_requested=config.split_half
        )
        failure_stage = "atomic_acceptance"
        archived = None
        if method_dir.exists():
            old_manifest = json.loads(
                (method_dir / MOTION_SIDECAR_MANIFEST).read_text()
            )
            suffix = str(old_manifest.get("artifact_digest", "unknown"))[:12]
            archived = cache_dir / f"{MOTION_METHOD_DIR}.superseded-{suffix}"
            counter = 2
            while archived.exists():
                archived = cache_dir / f"{MOTION_METHOD_DIR}.superseded-{suffix}-{counter}"
                counter += 1
            os.replace(method_dir, archived)
        try:
            os.replace(partial, method_dir)
        except Exception:
            if archived is not None and archived.exists() and not method_dir.exists():
                os.replace(archived, method_dir)
            raise
        cache_lineage = dict(manifest["cache_lineage"])
        estimate = _load_estimate(method_dir, cache_lineage)
        qc = evaluate_motion_qc(estimate, config.qc)
        _archive_failure_receipt(cache_dir)
        _mirror_inspection_artifacts(cache_dir, method_dir)
        _mirror_figures(cache_dir, method_dir)
        _write_qc(cache_dir, qc)
        _atomic_json(
            cache_dir / "support_metrics.json",
            {
                "peak_count_by_time": estimate.peak_count_by_time,
                "occupied_depth_bins_by_time": estimate.support_by_time,
                "depth_bin_centers_um": estimate.depth_bin_centers_um,
                "peak_count_by_time_depth_shape": list(
                    estimate.peak_count_by_time_depth.shape
                ),
            },
        )
        (cache_dir / "motion_summary.md").write_text(
            _summary_markdown(
                "ESTIMATE_COMPLETED",
                estimate,
                qc,
                cache_lineage,
                _split_half_status(method_dir),
            )
        )
        _atomic_json(
            cache_dir / "estimate_manifest.json",
            {
                "schema_version": MOTION_ESTIMATE_SCHEMA,
                "request_digest": request_digest,
                "artifact_digest": manifest["artifact_digest"],
                "method_dir": str(method_dir),
                "voltage_modified": False,
            },
        )
        return MotionSidecarRun(
            recording_for_sorting, estimate, qc, "ESTIMATE_COMPLETED",
            cache_dir, request_digest, cache_lineage,
        )
    except Exception as exc:
        if partial.exists():
            # Keep the partial directory for inspection rather than deleting it.
            partial_state = str(partial)
        else:
            partial_state = None
        failure = {
            "schema_version": MOTION_FAILURE_SCHEMA,
            "pipeline_version": PIPELINE_VERSION,
            "request_digest": request_digest,
            "attempted_effective_configuration": config.as_dict(),
            "failure_stage": failure_stage,
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
            "software_versions": versions,
            "failed_at_utc": _utc_now(),
            "partial_artifact_dir": partial_state,
            "safe_fallback": "identity",
            "voltage_modified": False,
        }
        _atomic_json(cache_dir / "estimation_failure.json", failure)
        qc = _failure_qc()
        cache_lineage = {"status": "none", "source_artifact_digest": None}
        _write_qc(cache_dir, qc)
        (cache_dir / "motion_summary.md").write_text(
            _summary_markdown(
                "ESTIMATE_FAILED_IDENTITY_SORT_CONTINUED", None, qc, cache_lineage
            )
        )
        if strict:
            raise
        return MotionSidecarRun(
            recording_for_sorting, None, qc,
            "ESTIMATE_FAILED_IDENTITY_SORT_CONTINUED", cache_dir,
            request_digest, cache_lineage,
        )


def run_motion_sidecar_safely(
    estimator_recording,
    *,
    recording_for_sorting,
    cache_dir: Path,
    config: MotionSidecarConfig | None = None,
    job_config: JobConfig | None = None,
    recompute: bool = False,
    strict: bool = False,
    backend: MotionBackend | None = None,
    accepted_recording_manifest: Mapping[str, Any] | None = None,
) -> MotionSidecarRun:
    """Convert sidecar-only preflight/cache errors into an identity fallback."""
    try:
        return run_motion_sidecar(
            estimator_recording,
            recording_for_sorting=recording_for_sorting,
            cache_dir=cache_dir,
            config=config,
            job_config=job_config,
            recompute=recompute,
            strict=strict,
            backend=backend,
            accepted_recording_manifest=accepted_recording_manifest,
        )
    except Exception as exc:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        selected_config = MotionSidecarConfig() if config is None else config
        request_digest = fingerprint(
            {
                "schema_version": MOTION_FAILURE_SCHEMA,
                "config": selected_config.as_dict(),
                "accepted_recording_request_digest": (
                    None
                    if accepted_recording_manifest is None
                    else accepted_recording_manifest.get("request_digest")
                ),
                "accepted_recording_content_sha256": (
                    None
                    if accepted_recording_manifest is None
                    else accepted_recording_manifest.get("recording_content_sha256")
                ),
            }
        )
        failure = {
            "schema_version": MOTION_FAILURE_SCHEMA,
            "pipeline_version": PIPELINE_VERSION,
            "request_digest": request_digest,
            "attempted_effective_configuration": selected_config.as_dict(),
            "failure_stage": "sidecar_preflight_or_cache_validation",
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
            "software_versions": software_versions(),
            "failed_at_utc": _utc_now(),
            "safe_fallback": "identity",
            "voltage_modified": False,
        }
        _atomic_json(cache_dir / "estimation_failure.json", failure)
        qc = _failure_qc()
        cache_lineage = {"status": "none", "source_artifact_digest": None}
        _write_qc(cache_dir, qc)
        (cache_dir / "motion_summary.md").write_text(
            _summary_markdown(
                "ESTIMATE_FAILED_IDENTITY_SORT_CONTINUED", None, qc, cache_lineage
            )
        )
        if strict:
            raise
        return MotionSidecarRun(
            recording_for_sorting=recording_for_sorting,
            estimate=None,
            qc=qc,
            status="ESTIMATE_FAILED_IDENTITY_SORT_CONTINUED",
            artifact_dir=cache_dir,
            request_digest=request_digest,
            cache_lineage=cache_lineage,
        )


def run_motion_sidecar_for_accepted_recording(
    recording_dir: Path,
    *,
    cache_dir: Path,
    config: MotionSidecarConfig | None = None,
    job_config: JobConfig | None = None,
    recompute: bool = False,
    strict: bool = False,
) -> MotionSidecarRun:
    """Run the sidecar from a verified accepted-recording directory."""
    from spikeinterface.core import load

    from .preprocess import MANIFEST_NAME, validate_accepted_recording

    recording_dir = Path(recording_dir)
    accepted_recording = load(recording_dir)
    accepted_manifest = json.loads((recording_dir / MANIFEST_NAME).read_text())
    validate_accepted_recording(recording_dir, accepted_manifest)
    selected_config = MotionSidecarConfig() if config is None else config
    try:
        estimator_recording = build_motion_estimator_input(
            accepted_recording,
            selected_config.estimator_input,
        )
        backend = None
    except Exception as estimator_input_error:
        # Sidecar construction may fail without changing the sorter input.
        estimator_recording = accepted_recording

        def fail_estimator_input(*args, _error=estimator_input_error, **kwargs):
            raise RuntimeError(
                f"Motion estimator input construction failed: {_error}"
            ) from _error

        backend = MotionBackend(
            fail_estimator_input,
            fail_estimator_input,
            fail_estimator_input,
            {"estimator_input": "construction-failed"},
        )
    return run_motion_sidecar_safely(
        estimator_recording,
        recording_for_sorting=accepted_recording,
        cache_dir=cache_dir,
        config=selected_config,
        job_config=job_config,
        recompute=recompute,
        strict=strict,
        backend=backend,
        accepted_recording_manifest=accepted_manifest,
    )


def plot_motion_sidecar(result: MotionSidecarRun, cache_dir: Path | None = None) -> None:
    """Regenerate sidecar plots without implying voltage correction."""
    if result.estimate is None:
        raise ValueError("Cannot plot a motion sidecar without an estimate")
    target = result.artifact_dir if cache_dir is None else Path(cache_dir)
    _write_plots(target, result.estimate)
    method_dir = result.artifact_dir / MOTION_METHOD_DIR
    peaks_path = method_dir / "peaks.npy"
    locations_path = method_dir / "peak_locations.npy"
    if peaks_path.exists() and locations_path.exists():
        _write_peak_diagnostic_plot(
            target,
            np.load(peaks_path, allow_pickle=False),
            np.load(locations_path, allow_pickle=False),
        )
